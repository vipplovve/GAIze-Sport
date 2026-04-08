import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.ActionRecognitionEngine import ActionLSTM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Visualization import plot_loss_curves, plot_accuracy_curves, plot_confusion_matrix

class KeypointDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)

        N, seq_len, num_kpts, coords = self.data.shape
        self.data = self.data.reshape(N, seq_len, num_kpts * coords)

        print(f"Loaded dataset: {self.data.shape[0]} samples, seq_len={seq_len}, features={num_kpts * coords}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

def get_config():
    print("=" * 60)
    print(" LSTM Training Configuration")
    print("=" * 60)

    data_dir = input("Data directory [data]: ").strip() or "data"
    sport = input("Sport (Football/Basketball) [Football]: ").strip() or "Football"
    epochs = int(input("Number of epochs [30]: ").strip() or "30")
    batch_size = int(input("Batch size [32]: ").strip() or "32")
    lr = float(input("Learning rate [0.001]: ").strip() or "0.001")
    input_size = int(input("Input feature size [34]: ").strip() or "34")
    hidden_size = int(input("LSTM hidden size [64]: ").strip() or "64")
    num_layers = int(input("Number of LSTM layers [2]: ").strip() or "2")
    num_classes = int(input("Number of action classes [4]: ").strip() or "4")
    checkpoint_dir = input("Checkpoint directory [checkpoints]: ").strip() or "checkpoints"

    config = argparse.Namespace(
        sport=sport, data_dir=data_dir, epochs=epochs, batch_size=batch_size,
        lr=lr, input_size=input_size, hidden_size=hidden_size,
        num_layers=num_layers, num_classes=num_classes,
        checkpoint_dir=checkpoint_dir
    )

    print("\n" + "-" * 40)
    print(f"  Sport:        {config.sport}")
    print(f"  Data Dir:     {config.data_dir}")
    print(f"  Epochs:       {config.epochs}")
    print(f"  Batch Size:   {config.batch_size}")
    print(f"  LR:           {config.lr}")
    print(f"  Input Size:   {config.input_size}")
    print(f"  Hidden Size:  {config.hidden_size}")
    print(f"  LSTM Layers:  {config.num_layers}")
    print(f"  Classes:      {config.num_classes}")
    print(f"  Checkpoints:  {config.checkpoint_dir}")
    print("-" * 40)

    confirm = input("\nProceed? (y/n) [y]: ").strip().lower() or "y"
    if confirm != "y":
        print("Aborted.")
        exit(0)

    return config

def train(args, log=print, stop_event=None, progress_callback=None):
    log("=" * 60)
    log(" LSTM Action Recognition Training")
    log("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    data_dir = Path(args.data_dir)
    sport_key = getattr(args, "sport", "Football").lower()
    data_path = str(data_dir / f"lstm_train_data_{sport_key}.npy")
    labels_path = str(data_dir / f"lstm_train_labels_{sport_key}.npy")

    dataset = KeypointDataset(data_path, labels_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    model = ActionLSTM(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=args.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_path = None

    history_train_loss = []
    history_val_loss = []
    history_train_acc = []
    history_val_acc = []

    sport_name = getattr(args, "sport", "Football")
    if sport_name == "Basketball":
        class_names = ["Idle", "Dribbling", "Shooting"]
    else:
        class_names = ["Idle", "Sprinting", "Kicking", "Dribbling"]

    for epoch in range(1, args.epochs + 1):
        if stop_event and stop_event.is_set():
            log("Training stopped by user.")
            break

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start_time = time.time()

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())

        scheduler.step()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)

        history_train_loss.append(t_loss)
        history_val_loss.append(v_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)

        elapsed = time.time() - start_time

        log(f"Epoch [{epoch:3d}/{args.epochs}] | "
            f"Train Loss: {t_loss:.4f} | Train Acc: {train_acc:.1f}% | "
            f"Val Loss: {v_loss:.4f} | Val Acc: {val_acc:.1f}% | "
            f"Time: {elapsed:.1f}s")

        if progress_callback:
            progress_callback(epoch, args.epochs)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            sport_dir = ckpt_dir / sport_name.lower()
            sport_dir.mkdir(parents=True, exist_ok=True)
            best_path = str(sport_dir / f"LSTM {sport_name} Action Recognition Model.pth")
            torch.save(model.state_dict(), best_path)
            log(f"  -> New best model! Val Acc: {val_acc:.1f}% -> {best_path}")

        if epoch % 5 == 0 or epoch == args.epochs:
            sport_dir = ckpt_dir / sport_name.lower()
            sport_dir.mkdir(parents=True, exist_ok=True)
            try:
                plot_loss_curves(history_train_loss, history_val_loss, title="LSTM Loss",
                                 filename="lstm_loss.png", save_dir=str(sport_dir))
                plot_accuracy_curves(history_train_acc, history_val_acc, title="LSTM Accuracy",
                                     filename="lstm_accuracy.png", save_dir=str(sport_dir))
                plot_confusion_matrix(all_val_labels, all_val_preds, class_names, title=f"LSTM Confusion Matrix (Epoch {epoch})",
                                      filename="lstm_confusion_matrix.png", save_dir=str(sport_dir))
            except Exception as e:
                log(f"  -> Could not generate plots: {e}")

    log(f"Training complete!")
    log(f"Best Val Accuracy: {best_val_acc:.1f}%")
    if best_path:
        log(f"Best model generated at: {best_path}")
    return best_path

def train_from_gui(config_dict, log_callback=print, stop_event=None, progress_callback=None):
    args = argparse.Namespace(**config_dict)
    return train(args, log=log_callback, stop_event=stop_event, progress_callback=progress_callback)

if __name__ == "__main__":
    config = get_config()
    train(config)
