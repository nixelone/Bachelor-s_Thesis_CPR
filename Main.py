"""
CPR compression-rate estimator.
Pipeline: YOLO11x-pose → camera-motion correction (optical flow) →
CWT frequency analysis → pose-heuristic candidate filter → BPM output.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import pywt
import json
import os

# --- Tunable constants ---
CPR_BPM_MIN            = 40       # CWT search band lower bound (bpm)
CPR_BPM_MAX            = 200      # CWT search band upper bound (bpm)
CPR_TARGET_BPM         = 110      # Ideal BPM for candidate ranking
MIN_PERIODICITY        = 0.25     # Periodicity score threshold
MIN_BPM                = 70       # Minimum plausible CPR rate
MAX_WRIST_RATIO        = 1.5      # Max wrist-spread / shoulder-width
MIN_ELBOW_ANGLE        = 130.0    # Min elbow angle (degrees); <130 → arms bent
MIN_FACE_WRIST         = 1.0      # Min face–wrist dist / shoulder-width
KPT_CONF_THRESH        = 0.4      # YOLO keypoint confidence cutoff
REQUIRED_STABLE_FRAMES = 15       # Frames before we trust an output
MAX_GRACE_FRAMES       = 7        # Frames to hold last BPM after signal loss
SWITCH_COOLDOWN        = 30       # Frames between performer switches
WAVELET                = "cmor1.5-1.0"

FEATURE_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
LK_PARAMS      = dict(winSize=(15, 15), maxLevel=2,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def get_angle(a, b, c):
    """Angle in degrees at vertex b."""
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def aggregate_second(buffer):
    """Mean of non-None values, rounded to int, or None."""
    valid = [v for v in buffer if v is not None]
    return int(round(sum(valid) / len(valid))) if valid else None


class PersonBuffer:
    """
    Rolling history for one tracked person.
    buffer_seconds: try 2.0 (responsive/noisier) vs 2.5 (default) vs 3.0 (smooth/slower).
    """
    def __init__(self, fps=30.0, buffer_seconds=2.5):
        self.fps  = fps
        maxlen    = int(round(fps * buffer_seconds))

        # Camera-corrected vertical positions for shoulder/elbow/wrist midpoints.
        self.y_sh, self.y_el, self.y_wr             = (deque(maxlen=maxlen) for _ in range(3))
        # Parallel validity flags (False = value carried forward, not a real detection).
        self.valid_sh, self.valid_el, self.valid_wr = (deque(maxlen=maxlen) for _ in range(3))

        self.smoothed_bpm             = None  # None until first CWT estimate
        self.periodicity_score        = 0.0
        self.current_variance         = 0.0
        self.smoothed_wrist_ratio     = None  # None until first valid shoulder reading
        self.smoothed_elbow_angle     = None
        self.smoothed_face_wrist_dist = None

    def update(self, kpts, conf, camera_offset_y):
        """Ingest one frame of keypoints and update all smoothed features."""
        # --- Pose-heuristic features (normalised by shoulder width) ---
        if conf[5] > KPT_CONF_THRESH and conf[6] > KPT_CONF_THRESH:
            sh_w = np.linalg.norm(kpts[5] - kpts[6]) + 1e-6

            if conf[9] > KPT_CONF_THRESH and conf[10] > KPT_CONF_THRESH:
                ratio = np.linalg.norm(kpts[9] - kpts[10]) / sh_w
                self.smoothed_wrist_ratio = ratio if self.smoothed_wrist_ratio is None \
                    else 0.8 * self.smoothed_wrist_ratio + 0.2 * ratio

            face_pts  = [kpts[i] for i in range(5)  if conf[i] > KPT_CONF_THRESH]
            wrist_pts = [kpts[i] for i in [9, 10]   if conf[i] > KPT_CONF_THRESH]
            if face_pts and wrist_pts:
                dist = np.linalg.norm(np.mean(face_pts, axis=0) - np.mean(wrist_pts, axis=0)) / sh_w
                self.smoothed_face_wrist_dist = dist if self.smoothed_face_wrist_dist is None \
                    else 0.8 * self.smoothed_face_wrist_dist + 0.2 * dist

        angles = []
        if conf[5] > KPT_CONF_THRESH and conf[7] > KPT_CONF_THRESH and conf[9] > KPT_CONF_THRESH:
            angles.append(get_angle(kpts[5], kpts[7], kpts[9]))
        if conf[6] > KPT_CONF_THRESH and conf[8] > KPT_CONF_THRESH and conf[10] > KPT_CONF_THRESH:
            angles.append(get_angle(kpts[6], kpts[8], kpts[10]))
        if angles:
            avg = float(np.mean(angles))
            self.smoothed_elbow_angle = avg if self.smoothed_elbow_angle is None \
                else 0.8 * self.smoothed_elbow_angle + 0.2 * avg

        # --- Vertical position signals (camera-motion corrected) ---
        def _append(y_dq, v_dq, ia, ib):
            if conf[ia] > KPT_CONF_THRESH and conf[ib] > KPT_CONF_THRESH:
                y_dq.append((kpts[ia][1] + kpts[ib][1]) / 2.0 - camera_offset_y)
                v_dq.append(True)
            else:
                y_dq.append(y_dq[-1] if y_dq else 0.0)  # carry forward last value
                v_dq.append(False)

        _append(self.y_sh, self.valid_sh, 5, 6)
        _append(self.y_el, self.valid_el, 7, 8)
        _append(self.y_wr, self.valid_wr, 9, 10)

    def calculate_variance(self):
        signal = self._best_signal(window=30)
        self.current_variance = float(np.var(signal)) if signal is not None else 0.0
        return self.current_variance

    def extract_frequency(self):
        """CWT-based BPM estimate; returns int BPM or None if not periodic enough."""
        signal = self._best_signal(window=len(self.y_sh))

        if signal is None or np.var(signal) < 2.0:
            self.periodicity_score = 0   # fast decay when signal is flat
            self.smoothed_bpm = None
            return None

        signal -= np.mean(signal)

        # CWT over the CPR frequency band.
        target_freqs         = np.linspace(CPR_BPM_MIN / 60.0, CPR_BPM_MAX / 60.0, 100)
        scales               = (pywt.central_frequency(WAVELET) * self.fps) / target_freqs
        coeffs, actual_freqs = pywt.cwt(signal, scales, WAVELET, sampling_period=1.0 / self.fps)
        mean_power           = np.mean(np.abs(coeffs) ** 2, axis=1)

        # Periodicity score = fraction of total power concentrated in the dominant band.
        best_idx    = int(np.argmax(mean_power))
        lo, hi      = max(0, best_idx - 5), min(len(mean_power), best_idx + 6)
        periodicity = float(np.sum(mean_power[lo:hi])) / (float(np.sum(mean_power)) + 1e-6)
        self.periodicity_score = 0.7 * self.periodicity_score + 0.3 * periodicity

        if self.periodicity_score < MIN_PERIODICITY:
            self.smoothed_bpm = None
            return None

        new_bpm           = float(actual_freqs[best_idx]) * 60.0
        self.smoothed_bpm = new_bpm if self.smoothed_bpm is None \
            else 0.2 * self.smoothed_bpm + 0.8 * new_bpm

        return int(round(self.smoothed_bpm))

    def _best_signal(self, window):
        """Shoulders → wrists → elbows; return first with ≥50 % valid frames."""
        if len(self.y_sh) < window:
            return None
        thresh = window * 0.75
        for y_dq, v_dq in [(self.y_sh, self.valid_sh),
                            (self.y_wr, self.valid_wr),
                            (self.y_el, self.valid_el)]:
            if sum(list(v_dq)[-window:]) >= thresh:
                return np.array(list(y_dq)[-window:])
        return None


def is_cpr_candidate(db):
    """True if pose features are consistent with performing CPR."""
    if db.periodicity_score < MIN_PERIODICITY:                                                     return False
    if db.smoothed_bpm is None or db.smoothed_bpm < MIN_BPM:                                      return False
    if db.smoothed_wrist_ratio     is not None and db.smoothed_wrist_ratio     >= MAX_WRIST_RATIO: return False
    if db.smoothed_elbow_angle     is not None and db.smoothed_elbow_angle     <  MIN_ELBOW_ANGLE: return False
    if db.smoothed_face_wrist_dist is not None and db.smoothed_face_wrist_dist <  MIN_FACE_WRIST:  return False
    return True


def draw_overlays(frame, results, tracker_db, top_3, active_id, final_output):
    """Bounding boxes, debug strings, active-performer skeleton, BPM panel."""
    for r in results:
        if r.keypoints is None or r.boxes.id is None or r.boxes.xyxy is None:
            continue
        ids   = r.boxes.id.int().cpu().tolist()
        k_all = r.keypoints.xy.cpu().numpy()
        c_all = r.keypoints.conf.cpu().numpy()
        b_all = r.boxes.xyxy.cpu().numpy()

        for i, p_id in enumerate(ids):
            k, c, b = k_all[i], c_all[i], b_all[i]

            if p_id in top_3:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                db = tracker_db.get(p_id)
                if db:
                    # Use "?" for any feature not yet computed (still None).
                    W = f"{db.smoothed_wrist_ratio:.1f}"     if db.smoothed_wrist_ratio     is not None else "?"
                    E = f"{db.smoothed_elbow_angle:.0f}"     if db.smoothed_elbow_angle     is not None else "?"
                    F = f"{db.smoothed_face_wrist_dist:.1f}" if db.smoothed_face_wrist_dist is not None else "?"
                    cv2.putText(frame, f"S:{db.periodicity_score:.2f} W:{W} E:{E} F:{F}",
                                (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if p_id == active_id:
                for s, e in [(5, 6), (5, 7), (6, 8)]:
                    if c[s] > KPT_CONF_THRESH and c[e] > KPT_CONF_THRESH:
                        cv2.line(frame, tuple(k[s].astype(int)), tuple(k[e].astype(int)), (0, 255, 255), 2)
                for w in [9, 10]:
                    if c[w] > KPT_CONF_THRESH:
                        cv2.circle(frame, tuple(k[w].astype(int)), 5, (0, 255, 255), -1)

    cv2.rectangle(frame, (10, 10), (150, 60), (0, 0, 0), -1)
    cv2.putText(frame, str(final_output), (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if final_output is not None else (0, 0, 255), 3)


def main(source="videos/v2.mp4", save_output=True,
         output_json_path="cpr_results_6.json", buffer_seconds=2.5):

    model     = YOLO("yolo_models/yolo11x-pose.pt")
    cap       = cv2.VideoCapture(source)
    W, H      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps       = cap.get(cv2.CAP_PROP_FPS)  # keep as float to avoid duration drift

    out = cv2.VideoWriter("cpr_analysis_output_final_6.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if save_output else None

    tracker_db        = {}
    active_id         = None
    switch_cooldown   = 0

    # Optical-flow camera-motion state.
    old_gray          = None
    p0                = None
    cumulative_dy     = 0.0

    # Stability / grace-period state.
    consecutive_valid = 0
    grace_remaining   = 0
    last_valid_bpm    = None

    # JSON aggregation state.
    json_data         = []
    sec_buffer        = []
    frame_count       = 0
    current_sec       = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results      = model.track(frame, persist=True, device="0", verbose=False)

        # Build a mask that excludes detected people so optical flow only
        # tracks background pixels for a clean camera-motion estimate.
        motion_mask = np.ones_like(gray, dtype=np.uint8) * 255
        for r in results:
            if r.boxes is not None and r.boxes.xyxy is not None:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    motion_mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 0

        if old_gray is not None and p0 is not None and len(p0) > 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **LK_PARAMS)
            good_new, good_old = p1[st == 1], p0[st == 1]
            if len(good_new) > 0:
                cumulative_dy += float(np.median(good_new[:, 1] - good_old[:, 1]))
                p0             = good_new.reshape(-1, 1, 2)
            else:
                p0 = cv2.goodFeaturesToTrack(gray, mask=motion_mask, **FEATURE_PARAMS)
        else:
            p0 = cv2.goodFeaturesToTrack(gray, mask=motion_mask, **FEATURE_PARAMS)
        old_gray = gray.copy()

        # --- Update per-person buffers ---
        current_ids = []
        for r in results:
            if r.keypoints is None or r.boxes.id is None:
                continue
            for i, p_id in enumerate(r.boxes.id.int().cpu().tolist()):
                current_ids.append(p_id)
                if p_id not in tracker_db:
                    tracker_db[p_id] = PersonBuffer(fps=fps, buffer_seconds=buffer_seconds)
                tracker_db[p_id].update(r.keypoints.xy.cpu().numpy()[i],
                                        r.keypoints.conf.cpu().numpy()[i],
                                        cumulative_dy)
                tracker_db[p_id].calculate_variance()

        # Prune people no longer in the scene to prevent memory growth.
        for gone in set(tracker_db) - set(current_ids):
            del tracker_db[gone]

        # Run CWT only on the top-3 highest-motion people.
        top_3 = sorted(current_ids, key=lambda p: tracker_db[p].current_variance, reverse=True)[:3]
        for p_id in top_3:
            tracker_db[p_id].extract_frequency()

        # --- Candidate selection and performer tracking ---
        candidates = [p for p in top_3 if is_cpr_candidate(tracker_db[p])]
        best_id    = min(candidates, key=lambda p: abs(tracker_db[p].smoothed_bpm - CPR_TARGET_BPM)) \
                     if candidates else None

        if active_id is None:
            active_id = best_id
        elif active_id not in candidates:
            active_id = best_id
        elif best_id is not None and best_id != active_id and switch_cooldown <= 0:
            # Only switch if the challenger is meaningfully closer to the target BPM.
            if abs(tracker_db[best_id].smoothed_bpm - CPR_TARGET_BPM) < \
               (abs(tracker_db[active_id].smoothed_bpm - CPR_TARGET_BPM) - 10):
                active_id, switch_cooldown = best_id, SWITCH_COOLDOWN
        if switch_cooldown > 0:
            switch_cooldown -= 1

        # --- Stability filter / grace period ---
        raw = None
        if active_id is not None:
            bpm = tracker_db[active_id].smoothed_bpm
            if bpm is not None and bpm >= MIN_BPM:
                raw = int(round(bpm))

        if raw is not None:
            consecutive_valid += 1
            last_valid_bpm     = raw
            grace_remaining    = MAX_GRACE_FRAMES
            final_output       = raw
        elif consecutive_valid >= REQUIRED_STABLE_FRAMES and grace_remaining > 0:
            final_output    = last_valid_bpm
            grace_remaining -= 1
        else:
            consecutive_valid = 0
            final_output      = None

        print(final_output)

        # --- Per-second JSON aggregation ---
        sec_buffer.append(final_output)
        if frame_count >= (current_sec + 1) * fps:
            json_data.append({"second": current_sec, "rate": aggregate_second(sec_buffer)})
            current_sec += 1
            sec_buffer   = []

        draw_overlays(frame, results, tracker_db, top_3, active_id, final_output)

        if save_output:
            out.write(frame)
        cv2.imshow("CPR Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

    # Flush leftover frames from the final partial second.
    if sec_buffer:
        json_data.append({"second": current_sec, "rate": aggregate_second(sec_buffer)})

    with open(output_json_path, "w") as f:
        json.dump({"description": "CPR compression rate per second",
                   "videoFile": os.path.basename(source),
                   "unit": "compressions/min",
                   "data": json_data}, f, indent=2)
    print(f"\nResults saved to {output_json_path}")


if __name__ == "__main__":
    main(buffer_seconds=2.5)  # try 2.0 or 3.0 to experiment