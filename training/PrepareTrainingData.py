import os
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ESRGAN_HR_DIR = BASE_DIR / "esrgan" / "data" / "hr_frames"
LSTM_DATA_DIR = BASE_DIR / "lstm" / "data"
VIDEO_PATH = BASE_DIR / "data" / "videos" / "soccer_clip.mp4"

def extract_frames(num_frames=50):
    ESRGAN_HR_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_PATH.exists():
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        print("Please place your soccer video at:")
        print(f"  {VIDEO_PATH}")
        return False

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // num_frames)

    saved = 0
    frame_idx = 0
    while cap.isOpened() and saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            out_path = str(ESRGAN_HR_DIR / f"frame_{saved:04d}.png")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} HR frames to {ESRGAN_HR_DIR}")
    return True

BASE_POSE = np.array([
    [0.50, 0.15], [0.48, 0.12], [0.52, 0.12], [0.45, 0.14], [0.55, 0.14],
    [0.42, 0.28], [0.58, 0.28], [0.38, 0.42], [0.62, 0.42], [0.36, 0.55],
    [0.64, 0.55], [0.44, 0.55], [0.56, 0.55], [0.43, 0.72], [0.57, 0.72],
    [0.42, 0.90], [0.58, 0.90],
], dtype=np.float32)

def generate_idle(num_frames=30):
    sequence = []
    for t in range(num_frames):
        noise = np.random.normal(0, 0.005, BASE_POSE.shape).astype(np.float32)
        breathing = np.zeros_like(BASE_POSE)
        breathing[5:7, 0] = np.sin(t * 0.3) * 0.003
        pose = BASE_POSE + noise + breathing
        sequence.append(np.clip(pose, 0, 1))
    return np.array(sequence)

def generate_sprinting(num_frames=30):
    sequence = []
    speed = np.random.uniform(0.005, 0.015)
    for t in range(num_frames):
        pose = BASE_POSE.copy()
        pose[:, 0] += speed * t
        pose[0, 1] += 0.03
        pose[5:7, 1] += 0.02
        phase = np.sin(t * 0.8)
        pose[13, 1] += phase * 0.08
        pose[15, 1] += phase * 0.12
        pose[15, 0] += phase * 0.04
        pose[14, 1] -= phase * 0.08
        pose[16, 1] -= phase * 0.12
        pose[16, 0] -= phase * 0.04
        pose[7, 1] -= phase * 0.05
        pose[9, 1] -= phase * 0.08
        pose[8, 1] += phase * 0.05
        pose[10, 1] += phase * 0.08
        noise = np.random.normal(0, 0.008, pose.shape).astype(np.float32)
        pose += noise
        sequence.append(np.clip(pose, 0, 1))
    return np.array(sequence)

def generate_kicking(num_frames=30):
    sequence = []
    kick_frame = num_frames // 2
    kicking_leg = np.random.choice(['left', 'right'])
    for t in range(num_frames):
        pose = BASE_POSE.copy()
        progress = 1.0 - abs(t - kick_frame) / kick_frame
        progress = max(0, progress)
        intensity = progress ** 2
        if kicking_leg == 'right':
            knee_idx, ankle_idx = 14, 16
            plant_knee = 13
        else:
            knee_idx, ankle_idx = 13, 15
            plant_knee = 14
        pose[knee_idx, 0] += intensity * 0.15
        pose[knee_idx, 1] -= intensity * 0.10
        pose[ankle_idx, 0] += intensity * 0.25
        pose[ankle_idx, 1] -= intensity * 0.20
        pose[plant_knee, 1] += intensity * 0.03
        pose[0, 1] -= intensity * 0.04
        pose[5:7, 1] -= intensity * 0.02
        pose[11:13, 0] -= intensity * 0.02
        pose[7, 1] -= intensity * 0.06
        pose[9, 1] -= intensity * 0.10
        pose[8, 0] += intensity * 0.05
        noise = np.random.normal(0, 0.006, pose.shape).astype(np.float32)
        pose += noise
        sequence.append(np.clip(pose, 0, 1))
    return np.array(sequence)

def generate_lstm_data(samples_per_class=100):
    LSTM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = []
    labels = []
    generators = {0: generate_idle, 1: generate_sprinting, 2: generate_kicking}

    for class_id, gen_func in generators.items():
        for _ in range(samples_per_class):
            sequence = gen_func(30)
            data.append(sequence)
            labels.append(class_id)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    shuffle_idx = np.random.permutation(len(data))
    data = data[shuffle_idx]
    labels = labels[shuffle_idx]

    data_path = str(LSTM_DATA_DIR / "lstm_train_data.npy")
    labels_path = str(LSTM_DATA_DIR / "lstm_train_labels.npy")
    np.save(data_path, data)
    np.save(labels_path, labels)

    print(f"Generated LSTM data: {data.shape} -> {data_path}")
    print(f"Generated LSTM labels: {labels.shape} -> {labels_path}")

if __name__ == "__main__":
    print("=" * 60)
    print(" Preparing Training Data")
    print("=" * 60)

    print("\n[1/2] Extracting HR frames from soccer clip...")
    success = extract_frames(num_frames=50)
    if not success:
        print("Aborting. Please provide the video first.")
        exit(1)

    print(f"\n[2/2] Generating LSTM keypoint data...")
    generate_lstm_data(samples_per_class=100)

    print("\n" + "=" * 60)
    print(" Data preparation complete!")
    print(f"  ESRGAN frames: {ESRGAN_HR_DIR}")
    print(f"  LSTM data:     {LSTM_DATA_DIR}")
    print("=" * 60)
