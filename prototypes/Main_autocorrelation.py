import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

class PersonBuffer:
    def __init__(self, maxlen=150):
        self.history = deque(maxlen=maxlen)
        self.fps = 30
        self.smoothed_bpm = 0.0

    def update(self, kpts, conf):
        self.history.append({'kpts': kpts, 'conf': conf})

    def extract_frequency(self):
        # We use a 3-second rolling window (90 frames) for Autocorrelation
        y_coords = [(f['kpts'][5][1] + f['kpts'][6][1]) / 2 for f in self.history if f['conf'][5] > 0.4 and f['conf'][6] > 0.4][-90:]
        
        # Need at least 2 seconds of confident points to run the math
        if len(y_coords) < 5: 
            return self.smoothed_bpm

        # 1. Detrend the signal (center it around 0)
        signal = np.array(y_coords)
        signal -= np.mean(signal)
        
        # 2. Compute Autocorrelation
        # np.correlate(mode='full') returns lags from -N to +N. 
        # We slice [len//2:] to only look at positive time shifts (lag >= 0)
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[len(corr)//2:] 

        # 3. Define the physical bounds of CPR
        # Max period: 60 BPM (1 Hz) = 1 compression every 30 frames
        # Min period: 180 BPM (3 Hz) = 1 compression every 10 frames
        min_lag = int(self.fps / 3.0) # 10 frames
        max_lag = int(self.fps / 1.0) # 30 frames
        
        if len(corr) <= max_lag:
            return self.smoothed_bpm

        # 4. Find the peak overlap within our logical CPR bounds
        search_window = corr[min_lag:max_lag+1]
        best_lag_idx = np.argmax(search_window)
        actual_lag = min_lag + best_lag_idx # This is the period in frames
        
        # 5. Convert lag (period) to BPM
        if actual_lag > 0:
            new_bpm = (self.fps / actual_lag) * 60
            # Snappy EMA: 20% old, 80% new to track slowing rhythms instantly
            self.smoothed_bpm = (0.2 * self.smoothed_bpm) + (0.8 * new_bpm) if self.smoothed_bpm > 0 else new_bpm
            
        return self.smoothed_bpm

def main(source="videos/v2.mp4", save_output=True):
# def main(source="metronome_videos/tiny_120.mp4", save_output=True):
    model = YOLO("yolo11x-pose.pt")
    cap = cv2.VideoCapture(source)
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter('cpr_autocorrelation_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)) if save_output else None

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
                
                # Extract frequency
                bpm = tracker_db[p_id].extract_frequency()

                k, c = k_all[i], c_all[i]
                
                # Draw minimal skeleton (shoulders and arms)
                color = (255, 150, 0)
                for s, e in [(5,6), (5,7), (7,9), (6,8), (8,10)]:
                    if c[s] > 0.4 and c[e] > 0.4:
                        cv2.line(frame, tuple(k[s].astype(int)), tuple(k[e].astype(int)), color, 2)

                # Draw just the BPM number above the person
                if bpm > 0:
                    text_color = (0, 255, 0) if 100 <= bpm <= 120 else (0, 0, 255)
                    cv2.putText(frame, f"{bpm:.1f}", (int(k[5][0]), int(k[5][1]-20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        if save_output: out.write(frame)
        cv2.imshow("CPR Autocorrelation Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    if save_output: out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(save_output=True)