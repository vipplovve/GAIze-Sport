import os
import sys
import json
import cv2
import numpy as np
from SoccerNet.Downloader import SoccerNetDownloader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.VideoAnalyticsEngine import VideoAnalyticsEngine

class SoccerNetLoader:
    def __init__(self, local_dir="data/SoccerNet"):
        self.local_dir = local_dir
        self.downloader = SoccerNetDownloader(LocalDirectory=local_dir)
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    def download_task(self, task="action-spotting", split=["train", "valid", "test"]):
        print(f"Downloading SoccerNet task: {task} for splits: {split}...")
        self.downloader.downloadDataTask(task=task, split=split)
        print("Download complete.")

    def extract_frames_for_esrgan(self, match_path, output_dir, num_frames=100):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_files = [f for f in os.listdir(match_path) if f.endswith(".mkv") or f.endswith(".mp4")]
        
        frame_idx = 0
        for video_file in video_files:
            video_path = str(Path(match_path) / video_file)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            step = max(1, total_frames // (num_frames // len(video_files)))
            
            count = 0
            while cap.isOpened() and count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % step == 0:
                    out_path = str(Path(output_dir) / f"sn_frame_{frame_idx:06d}.png")
                    cv2.imwrite(out_path, frame)
                    frame_idx += 1
                
                count += 1
            cap.release()
            
        print(f"Extracted {frame_idx} frames to {output_dir}")

    def prepare_lstm_data(self, match_path, annotation_file, output_data_path):
        analyzer = VideoAnalyticsEngine()
        analyzer.load_model()
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        all_sequences = []
        all_labels = []

        label_map = {"goal": 0, "card": 1, "substitution": 2}

        for action in annotations.get("predictions", []):
            label_name = action.get("label")
            if label_name not in label_map:
                continue
                
            label = label_map[label_name]
            timestamp = action.get("gameTime")
            half, time_str = timestamp.split(" - ")
            minutes, seconds = map(int, time_str.split(":"))
            total_seconds = (int(half)-1)*45*60 + minutes*60 + seconds
            
            video_file = f"{half}_720p.mkv"
            video_path = str(Path(match_path) / video_file)
            
            if not Path(video_path).exists():
                continue

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            target_frame = int(total_seconds * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 15))
            
            sequence = []
            for _ in range(30):
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = analyzer.model(frame, verbose=False)
                if results[0].keypoints is not None:
                    kpts = results[0].keypoints.xyn.cpu().numpy()
                    if len(kpts) > 0:
                        sequence.append(kpts[0])
                    else:
                        sequence.append(np.zeros((17, 2)))
                else:
                    sequence.append(np.zeros((17, 2)))
            
            cap.release()
            
            if len(sequence) == 30:
                all_sequences.append(sequence)
                all_labels.append(label)

        np.save(output_data_path + "_data.npy", np.array(all_sequences))
        np.save(output_data_path + "_labels.npy", np.array(all_labels))
        print(f"Processed {len(all_sequences)} sequences.")

if __name__ == "__main__":
    loader = SoccerNetLoader()
    print("SoccerNet Loader ready.")
