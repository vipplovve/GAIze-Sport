import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def extract_lstm_data_from_videos(source_dir, output_dir, seq_len=30, log_callback=None, stop_event=None):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    if not source_dir.exists():
        msg = f"ERROR: Source directory not found: {source_dir}"
        if log_callback: log_callback(msg)
        else: print(msg)
        return False
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yolo_path = Path(__file__).resolve().parent.parent / "yolo" / "yolo11n-pose.pt"
    if not yolo_path.exists():
        yolo_path = Path(__file__).resolve().parent.parent / "yolo11n-pose.pt"
        if not yolo_path.exists():
            yolo_path = "yolo11n-pose.pt"
            
    msg = f"[INFO] Loading YOLO Pose model from {yolo_path}..."
    if log_callback: log_callback(msg)
    else: print(msg)
    
    model = YOLO(str(yolo_path))
    
    all_sequences = []
    all_labels = []
    
    class_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    if not class_folders:
        msg = "ERROR: No class sub-folders found. Expected format: source_dir/0_class_name/vid.mp4"
        if log_callback: log_callback(msg)
        else: print(msg)
        return False
        
    msg = f"[INFO] Found {len(class_folders)} class folders."
    if log_callback: log_callback(msg)
    else: print(msg)

    for class_folder in sorted(class_folders):
        try:
            class_id = int(class_folder.name.split('_')[0])
        except ValueError:
            msg = f"[WARNING] Skipping {class_folder.name}. Folder name must start with integer ID (e.g. '0_idle')"
            if log_callback: log_callback(msg)
            else: print(msg)
            continue
            
        video_files = list(class_folder.glob("*.mp4")) + list(class_folder.glob("*.avi")) + list(class_folder.glob("*.mov"))
        if not video_files:
            continue
            
        msg = f"--- Processing Class {class_id} ({class_folder.name}) | {len(video_files)} videos found ---"
        if log_callback: log_callback(msg)
        else: print(msg)
        
        for vid_path in video_files:
            if stop_event and stop_event.is_set():
                msg = "[WARN] Extraction stopped by user."
                if log_callback: log_callback(msg)
                else: print(msg)
                return False
                
            msg = f"  -> Extracting from: {vid_path.name}"
            if log_callback: log_callback(msg)
            else: print(msg)
            
            cap = cv2.VideoCapture(str(vid_path))
            
            track_history = {}
            seqs_extracted = 0
            
            while cap.isOpened():
                if stop_event and stop_event.is_set():
                    cap.release()
                    return False
                    
                ret, frame = cap.read()
                if not ret:
                    break
                    
                results = model.track(frame, persist=True, verbose=False)
                
                if results[0].boxes is None or results[0].boxes.id is None or results[0].keypoints is None:
                    continue
                    
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints_array = results[0].keypoints.xyn.cpu().numpy()
                
                for idx, track_id in enumerate(track_ids):
                    if track_id not in track_history:
                        track_history[track_id] = []
                        
                    track_history[track_id].append(keypoints_array[idx])
                    
                    if len(track_history[track_id]) > seq_len:
                        track_history[track_id].pop(0)
                        
                    if len(track_history[track_id]) == seq_len:
                        seq_data = np.array(track_history[track_id], dtype=np.float32)
                        
                        all_sequences.append(seq_data)
                        all_labels.append(class_id)
                        seqs_extracted += 1
                        
                        track_history[track_id] = track_history[track_id][15:] 
                        
            cap.release()
            msg = f"     Extracted {seqs_extracted} sequences from {vid_path.name}"
            if log_callback: log_callback(msg)
            else: print(msg)

    if not all_sequences:
        msg = "[ERROR] Zero sequences extracted across all videos. Ensure videos contain trackable people."
        if log_callback: log_callback(msg)
        else: print(msg)
        return False
        
    data_matrix = np.array(all_sequences, dtype=np.float32)
    labels_matrix = np.array(all_labels, dtype=np.int64)
    
    msg = "[INFO] Shuffling dataset..."
    if log_callback: log_callback(msg)
    else: print(msg)
    
    shuffle_idx = np.random.permutation(len(data_matrix))
    data_matrix = data_matrix[shuffle_idx]
    labels_matrix = labels_matrix[shuffle_idx]

    data_path = output_dir / "lstm_train_data.npy"
    labels_path = output_dir / "lstm_train_labels.npy"
    
    np.save(str(data_path), data_matrix)
    np.save(str(labels_path), labels_matrix)
    
    msg = f"\n[SUCCESS] Extraction Complete!"
    if log_callback: log_callback(msg)
    else: print(msg)
    
    msg = f"  Final Data Shape:   {data_matrix.shape} -> {data_path.name}"
    if log_callback: log_callback(msg)
    else: print(msg)
    
    msg = f"  Final Labels Shape: {labels_matrix.shape} -> {labels_path.name}"
    if log_callback: log_callback(msg)
    else: print(msg)

    unique, counts = np.unique(labels_matrix, return_counts=True)
    msg = "  Class Sample Distribution:"
    if log_callback: log_callback(msg)
    else: print(msg)
    for cls, count in zip(unique, counts):
        msg = f"    Class {cls}: {count} sequences"
        if log_callback: log_callback(msg)
        else: print(msg)

    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract LSTM Data from Real Videos using YOLOv11 Pose")
    parser.add_argument("--source", type=str, required=True, help="Directory containing class-labeled video subfolders")
    parser.add_argument("--output", type=str, default="data", help="Output directory for .npy arrays")
    parser.add_argument("--seq_len", type=int, default=30, help="Frames per sequence")
    args = parser.parse_args()
    
    extract_lstm_data_from_videos(args.source, args.output, args.seq_len)
