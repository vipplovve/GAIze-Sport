import os
import sys
import shutil
import random
import argparse
from pathlib import Path
import cv2
import numpy as np

def setup_yolo_dataset(source_dir, output_dir, model_path=None, val_split=0.2, 
                       log=print, stop_event=None):
    from ultralytics import YOLO

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        log(f"ERROR: Source directory not found: {source_dir}")
        return None

    if model_path is None:
        candidates = [
            Path(__file__).resolve().parent.parent / "yolo11n-pose.pt",
            Path(__file__).resolve().parent.parent / "models" / "yolo11n-pose.pt",
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break
        if model_path is None:
            model_path = "yolo11n-pose.pt"

    log(f"[INFO] Loading YOLO model from {model_path}...")
    model = YOLO(str(model_path))

    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    all_images = sorted([
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_ext
    ])

    if not all_images:
        log(f"ERROR: No images found in {source_dir}")
        return None

    log(f"[INFO] Found {len(all_images)} images in {source_dir}")

    random.shuffle(all_images)
    val_count = max(1, int(len(all_images) * val_split))
    val_images = all_images[:val_count]
    train_images = all_images[val_count:]

    log(f"[INFO] Split: {len(train_images)} train / {len(val_images)} val")

    dirs = {
        'train_images': output_dir / "images" / "train",
        'val_images': output_dir / "images" / "val",
        'train_labels': output_dir / "labels" / "train",
        'val_labels': output_dir / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    total_annotations = 0
    for split_name, images, img_dir, lbl_dir in [
        ("train", train_images, dirs['train_images'], dirs['train_labels']),
        ("val", val_images, dirs['val_images'], dirs['val_labels']),
    ]:
        log(f"\n--- Processing {split_name} set ({len(images)} images) ---")

        for i, img_path in enumerate(images):
            if stop_event and stop_event.is_set():
                log("[WARN] Dataset setup stopped by user.")
                return None

            dest_img = img_dir / img_path.name
            shutil.copy2(str(img_path), str(dest_img))

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            results = model(frame, verbose=False)
            h, w = frame.shape[:2]

            label_path = lbl_dir / (img_path.stem + ".txt")
            detections = []

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xywhn.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                has_kpts = False
                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    kpts = results[0].keypoints.xyn.cpu().numpy()
                    has_kpts = len(kpts) > 0

                for j, (box, conf) in enumerate(zip(boxes, confs)):
                    if conf < 0.3:
                        continue

                    x_center, y_center, bw, bh = box
                    label_line = f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"

                    if has_kpts:
                        kp_line = ""
                        for kp in kpts[j]:
                            kx, ky = kp
                            visibility = 2 if (kx > 0 or ky > 0) else 0
                            kp_line += f" {kx:.6f} {ky:.6f} {visibility}"
                        label_line += kp_line
                        
                    detections.append(label_line)
                    total_annotations += 1

            with open(str(label_path), 'w') as f:
                f.write('\n'.join(detections))

            if (i + 1) % 50 == 0 or (i + 1) == len(images):
                log(f"  [{split_name}] Processed {i + 1}/{len(images)} images")

    log(f"\n[SUCCESS] Dataset created!")
    log(f"  Directory:    {output_dir}")
    log(f"  Train images: {len(train_images)}")
    log(f"  Val images:   {len(val_images)}")
    log(f"  Total annotations: {total_annotations}")

    yaml_path = _generate_yaml(output_dir, is_pose=(model.task == 'pose'))
    log(f"  Dataset YAML: {yaml_path}")

    return str(yaml_path)


def _generate_yaml(dataset_dir, is_pose=False):
    dataset_dir = Path(dataset_dir).resolve()
    yaml_path = dataset_dir / "football_dataset.yaml"

    yaml_content = f"""path: {dataset_dir}
train: images/train
val: images/val

names:
  0: person
"""
    if is_pose:
        yaml_content += "\nkpt_shape: [17, 3]\n"

    with open(str(yaml_path), 'w') as f:
        f.write(yaml_content)

    return yaml_path


def setup_from_gui(config_dict, log_callback=print, stop_event=None):
    source_dir = config_dict.get('source_dir', '')
    output_dir = config_dict.get('output_dir', str(Path(__file__).resolve().parent / "dataset"))
    model_path = config_dict.get('model_path', None)
    val_split = config_dict.get('val_split', 0.2)

    return setup_yolo_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        model_path=model_path,
        val_split=val_split,
        log=log_callback,
        stop_event=stop_event
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, 
                        default=str(Path(__file__).resolve().parent.parent / "esrgan" / "data" / "hr_frames"))
    parser.add_argument("--output_dir", type=str, 
                        default=str(Path(__file__).resolve().parent / "dataset"))
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    setup_yolo_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        val_split=args.val_split
    )
