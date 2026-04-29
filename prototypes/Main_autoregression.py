import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# NEW: Import Burg's method from the spectrum library
try:
    from spectrum import pburg
except ImportError:
    print("Please install the spectrum library: pip install spectrum")
    exit()

class PersonBuffer:
    def __init__(self, maxlen=150):
        self.history = deque(maxlen=maxlen)
        self.fps = 24
        self.smoothed_bpm = 0.0

    def update(self, kpts, conf):
        self.history.append({'kpts': kpts, 'conf': conf})

    def extract_frequency(self):
        # We only need a tight 2-second window (60 frames) for AR modeling
        y_coords = [(f['kpts'][5][1] + f['kpts'][6][1]) / 2 for f in self.history if f['conf'][5] > 0.4 and f['conf'][6] > 0.4][-60:]
        
        # Need at least ~1.5 seconds of data to fit a reliable AR model
        if len(y_coords) < 10: 
            return self.smoothed_bpm

        # 1. Detrend the signal (vital for AR modeling so it doesn't fit the DC offset)
        signal = np.array(y_coords)
        signal -= np.mean(signal)

        # 2. Fit the Autoregressive Model using Burg's Method
        # AR Order: How complex the formula is. For rhythmic CPR, an order of 5 to 8 is perfect. 
        # NFFT: How smoothly we want to draw the continuous curve (1024 gives massive sub-BPM precision).
        ar_order = 6 
        p = pburg(signal, ar_order, sampling=self.fps, NFFT=1024)
        
        # Extract the continuous frequency x-axis and the power y-axis
        frequencies = np.array(p.frequencies())
        psd = np.array(p.psd)

        # 3. Filter for our physical CPR bounds (60 BPM to 180 BPM)
        # 1.0 Hz = 60 BPM, 3.0 Hz = 180 BPM
        valid_mask = (frequencies >= 1.0) & (frequencies <= 3.0)
        valid_freqs = frequencies[valid_mask]
        valid_psd = psd[valid_mask]

        # 4. Find the exact sub-Hz peak
        if len(valid_freqs) > 0:
            best_freq = valid_freqs[np.argmax(valid_psd)]
            new_bpm = best_freq * 60.0
            
            # Snappy EMA: 30% old, 70% new to track slowing rhythms almost instantly
            self.smoothed_bpm = (0.3 * self.smoothed_bpm) + (0.7 * new_bpm) if self.smoothed_bpm > 0 else new_bpm
            
        return self.smoothed_bpm

def main(source="videos/v2.mp4", save_output=True):
# def main(source="metronome_videos/tiny_110.mp4", save_output=True):
    model = YOLO("yolo11x-pose.pt")
    cap = cv2.VideoCapture(source)
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter('cpr_ar_burg_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)) if save_output else None

    tracker_db = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model.track(frame, persist=True, device='0', verbose=False)

        if results and results[0].boxes.id is not None:
            r = results[0]
            k_all = r.keypoints.xy.cpu().numpy()
            c_all = r.keypoints.conf.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()

            for i, p_id in enumerate(ids):
                if p_id not in tracker_db: 
                    tracker_db[p_id] = PersonBuffer()
                
                # Update history
                tracker_db[p_id].update(k_all[i], c_all[i])
                
                # Extract frequency using AR Burg
                bpm = tracker_db[p_id].extract_frequency()

                k, c = k_all[i], c_all[i]
                
                # Draw minimal skeleton
                color = (200, 100, 255) # Purple for the AR test
                for s, e in [(5,6), (5,7), (7,9), (6,8), (8,10)]:
                    if c[s] > 0.4 and c[e] > 0.4:
                        cv2.line(frame, tuple(k[s].astype(int)), tuple(k[e].astype(int)), color, 2)

                # Draw the BPM number above the person
                if bpm > 0:
                    text_color = (0, 255, 0) if 100 <= bpm <= 120 else (0, 0, 255)
                    cv2.putText(frame, f"{bpm:.1f}", (int(k[5][0]), int(k[5][1]-20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        if save_output: out.write(frame)
        cv2.imshow("CPR AR Burg Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    if save_output: out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(save_output=True)