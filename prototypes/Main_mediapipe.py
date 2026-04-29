import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseModule:
    def __init__(self, model_path="mediapipe_models/pose_landmarker_heavy.task"):
        # Configure the detector
        base_options = python.BaseOptions(model_asset_path=model_path)
        # We use VIDEO mode so it can track movement across frames
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=4,  # Change this to handle the paramedics + victim
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame, timestamp_ms):
        # Convert OpenCV BGR to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Perform detection
        return self.detector.detect_for_video(mp_image, timestamp_ms)

def main(source=0):
    cap = cv2.VideoCapture(source)
    detector = PoseModule()

    # Get the actual FPS of the video file
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30  # Fallback for webcams
    
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # CALCULATE TIMESTAMP MANUALLY
        # timestamp = (current_frame / frames_per_second) * 1000ms
        timestamp_ms = int((frame_count / fps) * 1000)
        
        # Process the frame
        results = detector.process_frame(frame, timestamp_ms)
        
        # Increment counter
        frame_count += 1

        # --- Your drawing/display logic here ---
        if results.pose_landmarks:
            for pose in results.pose_landmarks:
                for landmark in pose:
                    h, w, _ = frame.shape
                    cv2.circle(frame, (int(landmark.x*w), int(landmark.y*h)), 5, (0,255,0), -1)

        cv2.imshow("CPR Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(source="videos/v2.mp4")
    # main()