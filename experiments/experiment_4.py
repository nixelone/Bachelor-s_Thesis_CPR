import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import pywt
import time
import os
import csv

# --- Tunable constants ---
CPR_BPM_MIN, CPR_BPM_MAX = 40, 200
WAVELET = "cmor1.5-1.0"
FEATURE_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class DummyPersonBuffer:
    """A stripped-down buffer purely to simulate the math load of the real pipeline."""
    def __init__(self, fps=30.0):
        self.fps = fps
        self.y_sh = deque(maxlen=int(fps * 2.5))

    def simulate_cwt_load(self):
        # Fill buffer with dummy data to force CWT computation
        if len(self.y_sh) < 10:
            self.y_sh.extend(np.random.rand(75).tolist())
        
        signal = np.array(self.y_sh)
        signal -= np.mean(signal)
        target_freqs = np.linspace(CPR_BPM_MIN / 60.0, CPR_BPM_MAX / 60.0, 100)
        scales = (pywt.central_frequency(WAVELET) * self.fps) / target_freqs
        pywt.cwt(signal, scales, WAVELET, sampling_period=1.0 / self.fps)

def profile_video(source, model):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    yolo_times = []
    logic_times = []
    
    tracker_db = {}
    old_gray = None
    p0 = None

    start_compute = time.perf_counter()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. PROFILE YOLO INFERENCE
        t0 = time.perf_counter()
        results = model.track(frame, persist=True, device="0", verbose=False)
        t_yolo = time.perf_counter() - t0
        yolo_times.append(t_yolo)

        # 2. PROFILE ALGORITHMIC LOGIC (Opt Flow + CWT)
        t1 = time.perf_counter()
        
        # Optical Flow simulation
        motion_mask = np.ones_like(gray, dtype=np.uint8) * 255
        if old_gray is not None and p0 is not None and len(p0) > 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **LK_PARAMS)
            if len(p1[st == 1]) > 0:
                p0 = p1[st == 1].reshape(-1, 1, 2)
            else:
                p0 = cv2.goodFeaturesToTrack(gray, mask=motion_mask, **FEATURE_PARAMS)
        else:
            p0 = cv2.goodFeaturesToTrack(gray, mask=motion_mask, **FEATURE_PARAMS)
        old_gray = gray.copy()

        # Simulate CWT scaling based on number of people detected
        current_ids = []
        for r in results:
            if r.boxes is not None and r.boxes.id is not None:
                current_ids = r.boxes.id.int().cpu().tolist()
                
        # Only process up to 3 people, as dictated by your pipeline rules
        top_3 = current_ids[:3] 
        for p_id in top_3:
            if p_id not in tracker_db:
                tracker_db[p_id] = DummyPersonBuffer(fps)
            tracker_db[p_id].simulate_cwt_load()

        t_logic = time.perf_counter() - t1
        logic_times.append(t_logic)

    end_compute = time.perf_counter()
    cap.release()

    total_compute_time = end_compute - start_compute
    overall_fps = total_frames / total_compute_time if total_compute_time > 0 else 0
    rtf = total_compute_time / video_duration if video_duration > 0 else 0

    avg_yolo_ms = (sum(yolo_times) / len(yolo_times)) * 1000 if yolo_times else 0
    avg_logic_ms = (sum(logic_times) / len(logic_times)) * 1000 if logic_times else 0

    return {
        "Total_Compute_Time_sec": round(total_compute_time, 2),
        "Video_Duration_sec": round(video_duration, 2),
        "Overall_FPS": round(overall_fps, 2),
        "Real_Time_Factor_(RTF)": round(rtf, 3),
        "Avg_YOLO_Latency_ms": round(avg_yolo_ms, 2),
        "Avg_Logic_Latency_ms": round(avg_logic_ms, 2)
    }

def main():
    # ADJUST THESE PATHS
    INPUT_FOLDER = r"N:\Data\School\Git\Bachelor-s_Thesis_CPR\experiment_videos"
    CSV_EXPORT_PATH = r"N:\Data\School\Git\Bachelor-s_Thesis_CPR\experiment_4_performance.csv"
    
    print("Loading YOLO model...")
    model = YOLO("yolo_models/yolo11x-pose.pt")

    results_data = []

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".MOV"):
            continue
            
        print(f"Profiling {filename}...")
        metrics = profile_video(os.path.join(INPUT_FOLDER, filename), model)
        
        # Classify the load based on the filename
        # Assuming files starting with 'output_3_' or 'v3_' are from Experiment 3 (2 people)
        # and everything else is just the rescuer (1 person).
        parts = filename.replace('.MOV', '').split('_')
        
        if len(parts) >= 2 and ('3' in parts[0] or parts[1] == '3'):
            load_type = "k=2 (Rescuer + Bystander)"
        else:
            load_type = "k=1 (Rescuer Only)"
        
        row = {"File_Name": filename, "Subject_Load": load_type}
        row.update(metrics)
        results_data.append(row)

    if results_data:
        with open(CSV_EXPORT_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_data[0].keys())
            writer.writeheader()
            writer.writerows(results_data)
        print(f"\nSuccess! Performance benchmark saved to {CSV_EXPORT_PATH}")
    else:
        print("No videos processed.")

if __name__ == "__main__":
    main()