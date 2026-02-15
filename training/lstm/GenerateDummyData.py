import os
import argparse
import numpy as np

BASE_POSE = np.array([
    [0.50, 0.15],
    [0.48, 0.12],
    [0.52, 0.12],
    [0.45, 0.14],
    [0.55, 0.14],
    [0.42, 0.28],
    [0.58, 0.28],
    [0.38, 0.42],
    [0.62, 0.42],
    [0.36, 0.55],
    [0.64, 0.55],
    [0.44, 0.55],
    [0.56, 0.55],
    [0.43, 0.72],
    [0.57, 0.72],
    [0.42, 0.90],
    [0.58, 0.90],
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
            plant_knee, plant_ankle = 13, 15
        else:
            knee_idx, ankle_idx = 13, 15
            plant_knee, plant_ankle = 14, 16
        
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

def generate_dataset(num_samples_per_class=500, num_frames=30):
    data = []
    labels = []
    
    generators = {
        0: ("Idle", generate_idle),
        1: ("Sprinting", generate_sprinting),
        2: ("Kicking", generate_kicking),
    }
    
    for class_id, (name, gen_func) in generators.items():
        print(f"Generating {num_samples_per_class} samples for class '{name}'...")
        for i in range(num_samples_per_class):
            sequence = gen_func(num_frames)
            data.append(sequence)
            labels.append(class_id)
    
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    shuffle_idx = np.random.permutation(len(data))
    data = data[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return data, labels

def main():
    parser = argparse.ArgumentParser(description="Generate dummy LSTM training data")
    parser.add_argument('--samples', type=int, default=500, help='Samples per class')
    parser.add_argument('--frames', type=int, default=30, help='Frames per sequence')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data, labels = generate_dataset(args.samples, args.frames)
    
    data_path = os.path.join(args.output_dir, "lstm_train_data.npy")
    labels_path = os.path.join(args.output_dir, "lstm_train_labels.npy")
    
    np.save(data_path, data)
    np.save(labels_path, labels)
    
    print(f"\nDataset generated:")
    print(f"  Data shape: {data.shape}  -> {data_path}")
    print(f"  Labels shape: {labels.shape} -> {labels_path}")
    print(f"  Classes: 0=Idle, 1=Sprinting, 2=Kicking")
    print(f"  Samples per class: {args.samples}")
    print(f"  Total samples: {len(data)}")

if __name__ == "__main__":
    main()
