import os
import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BASE_DIR / "esrgan"))

import torch
import cv2

def verify_esrgan():
    print("\n[1/3] ESRGAN Generator")
    print("-" * 40)
    
    from ESRGANModel import RRDBNet
    
    model_path = BASE_DIR / "esrgan" / "checkpoints" / "esrgan_generator.pth"
    if not model_path.exists():
        print(f"  SKIP: No model at {model_path}")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RRDBNet(num_blocks=8).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    
    test_frame_dir = BASE_DIR / "esrgan" / "data" / "hr_frames"
    if not test_frame_dir.exists():
        print("  SKIP: No test frame dir found")
        return False
    test_frames = [f for f in os.listdir(test_frame_dir) if f.endswith(".png")]
    
    if not test_frames:
        print("  SKIP: No test frames found")
        return False
    
    frame = cv2.imread(str(test_frame_dir / test_frames[0]))
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 4, h // 4))
    
    img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sr = model(tensor)
    
    sr_shape = sr.shape
    print(f"  Input:  {tensor.shape}")
    print(f"  Output: {sr_shape}")
    print(f"  Device: {device}")
    print(f"  PASS: ESRGAN produced {sr_shape[2]}x{sr_shape[3]} from {tensor.shape[2]}x{tensor.shape[3]}")
    return True

def verify_lstm():
    print("\n[2/3] LSTM Action Recognition")
    print("-" * 40)
    
    from core.ActionRecognitionEngine import ActionLSTM
    
    model_path = BASE_DIR / "lstm" / "checkpoints" / "action_lstm_best.pth"
    if not model_path.exists():
        print(f"  SKIP: No model at {model_path}")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionLSTM(input_size=34, hidden_size=64, num_layers=2, num_classes=3).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    
    classes = ["Idle", "Sprinting", "Kicking"]
    dummy_seq = torch.randn(1, 30, 34).to(device)
    
    with torch.no_grad():
        output = model(dummy_seq)
        probs = torch.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)
    
    print(f"  Input:  {dummy_seq.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Predicted: {classes[predicted.item()]} (confidence: {probs[0][predicted.item()]:.2%})")
    print(f"  PASS: LSTM predicted action '{classes[predicted.item()]}'")
    return True

def verify_yolo():
    print("\n[3/3] YOLOv11 Pose Detection")
    print("-" * 40)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  SKIP: ultralytics not installed")
        return False
    
    print("  Loading yolo11n-pose.pt...")
    model = YOLO("yolo11n-pose.pt")
    
    test_frame_dir = BASE_DIR / "esrgan" / "data" / "hr_frames"
    if not test_frame_dir.exists():
         return False
    test_frames = [f for f in os.listdir(test_frame_dir) if f.endswith(".png")]
    
    if not test_frames:
        print("  SKIP: No test frames found")
        return False
    
    frame = cv2.imread(str(test_frame_dir / test_frames[0]))
    results = model(frame, verbose=False)
    
    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
    has_keypoints = results[0].keypoints is not None
    
    print(f"  Frame:      {frame.shape}")
    print(f"  Detections: {num_detections}")
    print(f"  Keypoints:  {'Yes' if has_keypoints else 'No'}")
    print(f"  PASS: YOLOv11 detected {num_detections} objects")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print(" Pipeline Verification")
    print("=" * 60)
    
    results = {}
    results["ESRGAN"] = verify_esrgan()
    results["LSTM"] = verify_lstm()
    results["YOLOv11"] = verify_yolo()
    
    print("\n" + "=" * 60)
    print(" Results Summary")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "SKIP/FAIL"
        icon = "✓" if passed else "✗"
        print(f"  {icon} {name}: {status}")
        if not passed:
            all_pass = False
    
    print("=" * 60)
    if all_pass:
        print(" ALL COMPONENTS VERIFIED SUCCESSFULLY!")
    else:
        print(" Some components skipped or failed. See above.")
    print("=" * 60)
