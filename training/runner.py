import sys
import os
from pathlib import Path
import argparse

BASE_DIR = Path("/home/vipplovve/Codex/Antigravity/Major Project - II/GAIze-Sport/training")
sys.path.insert(0, str(BASE_DIR))

import PrepareTrainingData
print("\n--- Starting Frame Extraction ---")
PrepareTrainingData.extract_frames(num_frames=200)

print("\n--- Starting Real LSTM Data Extraction ---")
sys.path.insert(0, str(BASE_DIR / "lstm"))
import ExtractRealLSTMData
ExtractRealLSTMData.extract_lstm_data_from_videos(
    source_dir=str(BASE_DIR / "data" / "lstm_source"),
    output_dir=str(BASE_DIR / "lstm" / "data"),
    seq_len=30
)

print("\n--- Starting ESRGAN Phase 1 ---")
sys.path.insert(0, str(BASE_DIR / "esrgan"))
import esrgan.TrainESRGAN as TrainESRGAN
config_esrgan = argparse.Namespace(
    phase=1, 
    hr_dir=str(BASE_DIR / "esrgan" / "data" / "hr_frames"), 
    epochs=5,
    batch_size=4,
    lr=0.0002, 
    hr_crop=64,
    num_blocks=4,
    pretrained_gen=None, 
    checkpoint_dir=str(BASE_DIR / "esrgan" / "checkpoints"),
    save_every=5, 
    num_workers=0
)
TrainESRGAN.train_phase1(config_esrgan)

print("\n--- Starting LSTM Training ---")
import lstm.TrainLSTM as TrainLSTM
config_lstm = argparse.Namespace(
    data_dir=str(BASE_DIR / "lstm" / "data"), 
    epochs=10, 
    batch_size=8,
    lr=0.001, 
    input_size=34, 
    hidden_size=32,
    num_layers=1, 
    num_classes=3,
    checkpoint_dir=str(BASE_DIR / "lstm" / "checkpoints")
)
TrainLSTM.train(config_lstm)

print("\n--- Pipeline Execution Complete ---")
