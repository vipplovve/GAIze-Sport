# Training Directory

## ESRGAN (Super-Resolution)

**Two-phase training for video frame enhancement.**

### Quick Start
```bash
cd training/esrgan

# Phase 1: PSNR Pre-training (L1 loss only)
python train_esrgan.py --phase 1 --hr_dir ./data/hr_frames --epochs 50 --batch_size 4

# Phase 2: GAN Fine-tuning (Perceptual + Adversarial)
python train_esrgan.py --phase 2 --hr_dir ./data/hr_frames --epochs 100 --batch_size 4 --pretrained_gen checkpoints/phase1_gen.pth
```

### Preparing Data
Place HR frames in a folder (e.g., `data/hr_frames/`). You can extract frames from a video:
```python
from dataset import VideoFrameDataset
VideoFrameDataset.extract_frames("match.mp4", "data/hr_frames/", every_n_frames=5)
```

### Architecture
- **Generator**: RRDBNet with 8 RRDB blocks (GTX 1650 optimized)
- **Discriminator**: VGG-style PatchGAN
- **Losses**: L1 + VGG19 Perceptual + Relativistic Adversarial

---

## LSTM (Action Recognition)

**Classifies player actions from pose keypoint sequences.**

### Quick Start
```bash
cd training/lstm

# 1. Generate dummy training data
python generate_dummy_data.py --samples 500 --output_dir data

# 2. Train
python train_lstm.py --data_dir data --epochs 30 --batch_size 32
```

### Action Classes
| ID | Action | Motion Pattern |
|----|--------|---------------|
| 0 | Idle | Small jitter around standing pose |
| 1 | Sprinting | Horizontal movement + leg/arm swings |
| 2 | Kicking | Leg sweep forward, body leans back |

### Output
Trained models are saved to `checkpoints/`:
- `action_lstm_best.pth` — Best validation accuracy
- `action_lstm.pth` — Final epoch
