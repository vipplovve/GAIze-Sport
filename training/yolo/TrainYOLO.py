"""
YOLOv11 Fine-tuning wrapper for GUI integration.
Uses the Ultralytics API to fine-tune a YOLO pose model on custom data.
"""
import os
import time
import threading

def train_from_gui(config_dict, log_callback=print, stop_event=None):
    """
    Fine-tune a YOLOv11 model from the GUI.
    
    config_dict keys:
        base_model  : str  - Path to base YOLO model (e.g. yolo11n-pose.pt)
        data_yaml   : str  - Path to dataset YAML file
        epochs      : int  - Number of training epochs
        imgsz       : int  - Image size for training
        batch       : int  - Batch size
        device      : str  - Device to train on ('cpu', '0', 'cuda')
        project     : str  - Output project directory
        name        : str  - Run name
    """
    from ultralytics import YOLO

    base_model = config_dict.get('base_model', 'yolo11n-pose.pt')
    data_yaml = config_dict.get('data_yaml', '')
    epochs = config_dict.get('epochs', 50)
    imgsz = config_dict.get('imgsz', 640)
    batch = config_dict.get('batch', 8)
    device = config_dict.get('device', 'cpu')
    project = config_dict.get('project', 'runs/train')
    name = config_dict.get('name', 'yolo_finetune')

    log_callback("=" * 60)
    log_callback(" YOLOv11 Fine-tuning")
    log_callback("=" * 60)
    log_callback(f"Base Model : {base_model}")
    log_callback(f"Dataset    : {data_yaml}")
    log_callback(f"Epochs     : {epochs}")
    log_callback(f"Image Size : {imgsz}")
    log_callback(f"Batch Size : {batch}")
    log_callback(f"Device     : {device}")
    log_callback(f"Output     : {project}/{name}")
    log_callback("-" * 60)

    if not data_yaml or not os.path.exists(data_yaml):
        log_callback(f"ERROR: Dataset YAML not found: {data_yaml}")
        log_callback("Please provide a valid COCO-format dataset YAML file.")
        log_callback("")
        log_callback("Expected YAML format:")
        log_callback("  path: /path/to/dataset")
        log_callback("  train: images/train")
        log_callback("  val: images/val")
        log_callback("  names:")
        log_callback("    0: player")
        log_callback("    1: ball")
        log_callback("    2: referee")
        return None

    try:
        log_callback("Loading base model...")
        model = YOLO(base_model)

        log_callback("Starting training...")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            verbose=True,
            exist_ok=True,
        )

        log_callback("=" * 60)
        log_callback(" Training Complete!")
        log_callback(f" Results saved to: {project}/{name}")
        log_callback("=" * 60)

        best_path = os.path.join(project, name, "weights", "best.pt")
        if os.path.exists(best_path):
            log_callback(f"Best model: {best_path}")
            return best_path
        else:
            last_path = os.path.join(project, name, "weights", "last.pt")
            log_callback(f"Last model: {last_path}")
            return last_path

    except Exception as e:
        log_callback(f"ERROR: Training failed: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print(" YOLOv11 Fine-tuning Configuration")
    print("=" * 60)

    base_model = input("Base model path [yolo11n-pose.pt]: ").strip() or "yolo11n-pose.pt"
    data_yaml = input("Dataset YAML path: ").strip()
    epochs = int(input("Epochs [50]: ").strip() or "50")
    imgsz = int(input("Image size [640]: ").strip() or "640")
    batch = int(input("Batch size [8]: ").strip() or "8")
    device = input("Device (cpu/0) [cpu]: ").strip() or "cpu"

    config = {
        'base_model': base_model,
        'data_yaml': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': 'runs/train',
        'name': 'yolo_finetune',
    }

    confirm = input("\nProceed? (y/n) [y]: ").strip().lower() or "y"
    if confirm == "y":
        train_from_gui(config)
    else:
        print("Aborted.")
