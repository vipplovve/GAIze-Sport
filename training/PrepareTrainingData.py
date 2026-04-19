import os
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
ESRGAN_HR_DIR = BASE_DIR / "esrgan" / "data" / "hr_frames"
LSTM_DATA_DIR = BASE_DIR / "lstm" / "data"

DEFAULT_TRAINING_CLIPS = [
    PROJECT_DIR / "data" / "football training clip #1.mp4",
    PROJECT_DIR / "data" / "football training clip #2.mp4",
    PROJECT_DIR / "data" / "football training clip #3.mp4",
]

def _find_next_index(output_dir, prefix):
    import re
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.\w+$")
    max_idx = -1
    output_dir = Path(output_dir)
    if output_dir.exists():
        for f in output_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def extract_frames(video_paths=None, output_dir=None, num_frames_per_video=100,
                   frame_prefix="frame", log=print):
    if video_paths is None:
        video_paths = DEFAULT_TRAINING_CLIPS
    if output_dir is None:
        output_dir = ESRGAN_HR_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_idx = _find_next_index(output_dir, frame_prefix)
    if start_idx > 0:
        log(f"Found existing frames up to index {start_idx - 1}. "
            f"New frames will start at {frame_prefix}_{start_idx:06d}.")

    total_saved = 0
    current_idx = start_idx

    for vid_idx, video_path in enumerate(video_paths):
        video_path = Path(video_path)
        if not video_path.exists():
            log(f"WARNING: Video not found: {video_path}")
            continue

        log(f"[{vid_idx + 1}/{len(video_paths)}] Extracting from: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // num_frames_per_video)

        saved = 0
        frame_idx = 0
        while cap.isOpened() and saved < num_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                out_path = str(output_dir / f"{frame_prefix}_{current_idx:06d}.png")
                cv2.imwrite(out_path, frame)
                saved += 1
                total_saved += 1
                current_idx += 1

            frame_idx += 1

        cap.release()
        log(f"  Extracted {saved} frames from {video_path.name}")

    log(f"\nTotal: Extracted {total_saved} HR frames to {output_dir}")
    return total_saved > 0


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

def generate_dribbling(num_frames=30):
    sequence = []
    drift_x = np.random.uniform(0.002, 0.006)
    for t in range(num_frames):
        pose = BASE_POSE.copy()
        pose[:, 0] += drift_x * t
        bounce = np.abs(np.sin(t * 1.2)) * 0.03
        pose[0, 1] -= bounce
        pose[5:7, 1] -= bounce * 0.5
        foot_phase = np.sin(t * 1.5)
        pose[15, 1] += foot_phase * 0.06
        pose[15, 0] += foot_phase * 0.03
        pose[16, 1] -= foot_phase * 0.06
        pose[16, 0] -= foot_phase * 0.03
        pose[13, 1] += foot_phase * 0.03
        pose[14, 1] -= foot_phase * 0.03
        pose[7, 0] -= 0.04
        pose[8, 0] += 0.04
        pose[9, 0] -= 0.06
        pose[10, 0] += 0.06
        noise = np.random.normal(0, 0.008, pose.shape).astype(np.float32)
        pose += noise
        sequence.append(np.clip(pose, 0, 1))
    return np.array(sequence)

def generate_shooting(num_frames=30):
    sequence = []
    shoot_frame = int(num_frames * 0.4)
    for t in range(num_frames):
        pose = BASE_POSE.copy()
        if t < shoot_frame:
            squat = t / shoot_frame
            pose[13, 1] += squat * 0.1
            pose[14, 1] += squat * 0.1
            pose[9, 1] += squat * 0.05
            pose[10, 1] += squat * 0.05
        else:
            extension = min(1.0, (t - shoot_frame) / (num_frames * 0.2))
            pose[13, 1] += 0.1 - (extension * 0.1)
            pose[14, 1] += 0.1 - (extension * 0.1)
            pose[9, 1] -= extension * 0.3
            pose[10, 1] -= extension * 0.3
            pose[7, 1] -= extension * 0.15
            pose[8, 1] -= extension * 0.15
        noise = np.random.normal(0, 0.006, pose.shape).astype(np.float32)
        pose += noise
        sequence.append(np.clip(pose, 0, 1))
    return np.array(sequence)

def generate_guarding(num_frames=30):
    sequence = []
    for t in range(num_frames):
        pose = BASE_POSE.copy()
        pose[13, 1] += 0.05
        pose[14, 1] += 0.05
        pose[15, 0] -= 0.04
        pose[16, 0] += 0.04
        pose[9, 0] -= 0.1
        pose[9, 1] += 0.05
        pose[10, 0] += 0.1
        pose[10, 1] += 0.05
        shift = np.sin(t * 0.5) * 0.02
        pose[:, 0] += shift
        noise = np.random.normal(0, 0.005, pose.shape).astype(np.float32)
        pose += noise
        sequence.append(np.clip(pose, 0, 1))
    return np.array(sequence)



def generate_lstm_data(sport="Football", samples_per_class=100):
    LSTM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = []
    labels = []

    if sport == "Basketball":
        generators = {0: generate_idle, 1: generate_dribbling, 2: generate_shooting, 3: generate_guarding}
        prefix = "basketball"
    else:
        generators = {0: generate_idle, 1: generate_sprinting, 2: generate_kicking, 3: generate_dribbling}
        prefix = "football"

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

    data_path = str(LSTM_DATA_DIR / f"lstm_train_data_{prefix}.npy")
    labels_path = str(LSTM_DATA_DIR / f"lstm_train_labels_{prefix}.npy")
    np.save(data_path, data)
    np.save(labels_path, labels)

    print(f"Generated LSTM data ({sport}): {data.shape} -> {data_path}")
    print(f"Generated LSTM labels ({sport}): {labels.shape} -> {labels_path}")

if __name__ == "__main__":
    print("=" * 60)
    print(" Preparing Training Data")
    print("=" * 60)

    print("\n[1/2] Extracting HR frames from training clips...")
    success = extract_frames(num_frames_per_video=100)
    if not success:
        print("WARNING: No frames extracted. Check video paths.")

    print(f"\n[2/2] Generating LSTM keypoint data...")
    generate_lstm_data(sport="Football", samples_per_class=100)
    generate_lstm_data(sport="Basketball", samples_per_class=100)

    print("\n" + "=" * 60)
    print(" Data preparation complete!")
    print(f"  ESRGAN frames: {ESRGAN_HR_DIR}")
    print(f"  LSTM data:     {LSTM_DATA_DIR}")
    print("=" * 60)
