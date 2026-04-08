import cv2
from .ActionRecognitionEngine import ActionRecognitionEngine
from ultralytics import YOLO

class VideoAnalyticsEngine:
    def __init__(self, model_path="yolo11n-pose.pt", lstm_model=None, sport="Football"):
        self.model_path = model_path
        self.model = None

        if lstm_model:
            self.action_recognizer = ActionRecognitionEngine(model_path=lstm_model, sport=sport)
        else:
            self.action_recognizer = ActionRecognitionEngine(sport=sport)

        self.player_buffers = {}
        self.player_actions = {}

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

            if (results[0].boxes is not None and results[0].boxes.id is not None
                    and results[0].keypoints is not None):

                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints_array = results[0].keypoints.xyn.cpu().numpy()
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()

                active_ids = set()

                for idx, track_id in enumerate(track_ids):
                    active_ids.add(track_id)
                    kpts = keypoints_array[idx]

                    if track_id not in self.player_buffers:
                        self.player_buffers[track_id] = []

                    self.player_buffers[track_id].append(kpts)

                    seq_len = self.action_recognizer.sequence_length
                    if len(self.player_buffers[track_id]) > seq_len:
                        self.player_buffers[track_id].pop(0)

                    if len(self.player_buffers[track_id]) >= seq_len:
                        action = self.action_recognizer.predict_action(
                            self.player_buffers[track_id]
                        )
                        self.player_actions[track_id] = action

                    if track_id in self.player_actions:
                        action_text = self.player_actions[track_id]
                        x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)

                        label = f"Player #{track_id}: {action_text}"
                        color = self._action_color(action_text)

                        (tw, th), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        label_y = max(y1 - 10, th + 5)
                        cv2.rectangle(annotated_frame,
                                      (x1, label_y - th - 5),
                                      (x1 + tw + 6, label_y + 5),
                                      color, -1)
                        cv2.putText(annotated_frame, label,
                                    (x1 + 3, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 2)

                stale_ids = [tid for tid in self.player_buffers if tid not in active_ids]
                for tid in stale_ids:
                    del self.player_buffers[tid]
                    if tid in self.player_actions:
                        del self.player_actions[tid]

            frame_count += 1
            if callback:
                progress = int((frame_count / total_frames) * 100)
                callback(progress)

            yield annotated_frame, results[0]

        cap.release()

    @staticmethod
    def _action_color(action):
        colors = {
            "Idle":       (128, 128, 128),
            "Sprinting":  (0, 200, 0),
            "Kicking":    (0, 100, 255),
            "Dribbling":  (255, 100, 0),
            "Shooting":   (0, 0, 255),
            "Buffering...": (200, 200, 200),
        }
        return colors.get(action, (255, 255, 255))
