import cv2
from .ActionRecognitionEngine import ActionRecognitionEngine
from ultralytics import YOLO

class VideoAnalyticsEngine:
    def __init__(self, model_path="yolo11n-pose.pt"):
        self.model_path = model_path
        self.model = None
        self.action_recognizer = ActionRecognitionEngine()
        self.keypoints_buffer = []

    def load_model(self):
        try:
            print(f"Loading YOLO Pose model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            self.action_recognizer.load_model()
            print("Video Analytics models loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def analyze_video(self, video_path, callback=None):
        if not self.model:
            self.load_model()
            
        if not self.model:
            print("No model loaded. Skipping analysis.")
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model.track(frame, persist=True, verbose=False)
            
            annotated_frame = results[0].plot()
            
            if results[0].keypoints is not None:
                kpts = results[0].keypoints.xyn.cpu().numpy()
                if len(kpts) > 0:
                    current_kpts = kpts[0]
                    self.keypoints_buffer.append(current_kpts)
                    if len(self.keypoints_buffer) > 30:
                        self.keypoints_buffer.pop(0)
                        
                    if len(self.keypoints_buffer) >= 30:
                        action = self.action_recognizer.predict_action(self.keypoints_buffer)
                        cv2.putText(annotated_frame, f"Action: {action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_count += 1
            if callback:
                progress = int((frame_count / total_frames) * 100)
                callback(progress)
                
            yield annotated_frame, results[0]
            
        cap.release()
