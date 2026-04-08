import os
import time
import threading
from pathlib import Path

def train_from_gui(config_dict, log_callback=print, stop_event=None, progress_callback=None):
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

    frames_dir = config_dict.get('frames_dir', '')

    # Check if we need to setup/re-setup the dataset
    needs_setup = not data_yaml or not Path(data_yaml).exists()
    
    if not needs_setup and "pose" in str(base_model).lower():
        # Check if kpt_shape is in the existing YAML
        try:
            with open(data_yaml, 'r') as f:
                content = f.read()
                if "kpt_shape" not in content:
                    log_callback("[WARN] Selected model is Pose, but dataset YAML is missing 'kpt_shape'. Triggering re-setup...")
                    needs_setup = True
        except Exception:
            needs_setup = True

    if needs_setup:
        if frames_dir and Path(frames_dir).exists():
            log_callback(f"Setting up dataset from frames dir: {frames_dir}")
            try:
                from training.yolo.setup_yolo_dataset import setup_yolo_dataset
                output_dir = str(Path(data_yaml).parent) if data_yaml else str(Path(__file__).resolve().parent / "dataset")
                data_yaml_generated = setup_yolo_dataset(
                    source_dir=frames_dir,
                    output_dir=output_dir,
                    model_path=base_model,
                    log=log_callback,
                    stop_event=stop_event
                )
                if not data_yaml_generated:
                    log_callback("ERROR: Failed to setup YOLO dataset.")
                    return None
                data_yaml = data_yaml_generated
            except Exception as e:
                log_callback(f"ERROR: Dataset setup failed: {e}")
                return None
        else:
            if not data_yaml or not Path(data_yaml).exists():
                log_callback(f"ERROR: Dataset YAML not found: {data_yaml} and no valid frames dir provided.")
                return None

    try:
        log_callback("Loading base model...")
        model = YOLO(base_model)

        if progress_callback:
            def _on_epoch_end(trainer):
                current = trainer.epoch + 1
                total = trainer.epochs
                progress_callback(current, total)
                log_callback(f"Epoch [{current}/{total}] complete")
            model.add_callback("on_train_epoch_end", _on_epoch_end)

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

        best_path = Path(project) / name / "weights" / "best.pt"
        if best_path.exists():
            log_callback(f"Best model: {best_path}")
            
            sport = config_dict.get('sport')
            if sport:
                import shutil
                dest_dir = Path(__file__).resolve().parent.parent.parent / "models" / sport.lower()
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / "yolo_finetuned.pt"
                shutil.copy2(str(best_path), str(dest_path))
                log_callback(f"Copied best model to: {dest_path}")
                
            return str(best_path)
        else:
            last_path = Path(project) / name / "weights" / "last.pt"
            log_callback(f"Last model: {last_path}")
            return str(last_path)

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
