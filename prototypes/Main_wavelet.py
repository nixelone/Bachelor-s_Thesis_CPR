import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# NEW: Import PyWavelets for the Continuous Wavelet Transform
try:
    import pywt
except ImportError:
    print("Please install the PyWavelets library: pip install PyWavelets")
    exit()

class PersonBuffer:
    def __init__(self, maxlen=150):
        self.history = deque(maxlen=maxlen)
        self.fps = 24
        self.smoothed_bpm = 0.0

    def update(self, kpts, conf):
        self.history.append({'kpts': kpts, 'conf': conf})

    def extract_frequency(self):
        # 3-second window (90 frames) is ideal for CWT to overcome edge artifacts
        y_coords = [(f['kpts'][5][1] + f['kpts'][6][1]) / 2 for f in self.history if f['conf'][5] > 0.4 and f['conf'][6] > 0.4][-90:]
        
        if len(y_coords) < 5: 
            return self.smoothed_bpm

        # 1. Detrend the signal
        signal = np.array(y_coords)
        signal -= np.mean(signal)

        # 2. Define the CPR frequency range (1.0 Hz to 3.0 Hz / 60 to 180 BPM)
        # We test 100 different specific frequencies within this range
        target_freqs = np.linspace(0.5, 5.0, 100)
        
        # 3. Setup the Wavelet
        # 'cmor1.5-1.0' is a Complex Morlet wavelet. It is the gold standard for human biomechanical rhythms.
        wavelet = 'cmor1.5-1.0'
        
        # Convert our target frequencies into wavelet "scales"
        center_freq = pywt.central_frequency(wavelet)
        scales = (center_freq * self.fps) / target_freqs

        # 4. Perform the Continuous Wavelet Transform
        # This returns a 2D matrix of coefficients (Scale x Time)
        coefficients, actual_freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1.0/self.fps)

        # 5. Calculate Power
        # Since it's a complex wavelet, we take the absolute magnitude squared
        power = np.abs(coefficients)**2

        # 6. Find the dominant frequency
        # We average the power across the time dimension to find what frequency dominated this 3-second window
        mean_power = np.mean(power, axis=1)
        
        best_idx = np.argmax(mean_power)
        best_freq = actual_freqs[best_idx]

        # Convert Hz to BPM
        new_bpm = best_freq * 60.0
            
        # Snappy EMA: 30% old, 70% new
        self.smoothed_bpm = (0.3 * self.smoothed_bpm) + (0.7 * new_bpm) if self.smoothed_bpm > 0 else new_bpm
            
        return self.smoothed_bpm

def main(source="videos/v2.mp4", save_output=True):
# def main(source="metronome_videos/tiny_110.mp4", save_output=True):
    model = YOLO("yolo11x-pose.pt")
    cap = cv2.VideoCapture(source)
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter('cpr_cwt_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)) if save_output else None

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
                
                # Extract frequency using Continuous Wavelet Transform
                bpm = tracker_db[p_id].extract_frequency()

                k, c = k_all[i], c_all[i]
                
                # Draw minimal skeleton
                color = (0, 255, 255) # Yellow for the CWT test
                for s, e in [(5,6), (5,7), (7,9), (6,8), (8,10)]:
                    if c[s] > 0.4 and c[e] > 0.4:
                        cv2.line(frame, tuple(k[s].astype(int)), tuple(k[e].astype(int)), color, 2)

                # Draw the BPM number above the person
                if bpm > 0:
                    text_color = (0, 255, 0) if 100 <= bpm <= 120 else (0, 0, 255)
                    cv2.putText(frame, f"{bpm:.1f}", (int(k[5][0]), int(k[5][1]-20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        if save_output: out.write(frame)
        cv2.imshow("CPR CWT Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    if save_output: out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(save_output=True)