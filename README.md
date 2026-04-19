<div align="center">

# 🛡️ GAIze-Sport

### AI-Powered Sports Video Analysis Platform

**Real-time detection &nbsp;•&nbsp; Super-resolution &nbsp;•&nbsp; Action recognition &nbsp;•&nbsp; Tactical minimaps &nbsp;•&nbsp; PDF reports**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv11-green)
![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## 📖 Overview

**GAIze-Sport** is a desktop application for AI-driven sports video analysis. It combines three deep-learning models in a single, polished dark-themed GUI built with CustomTkinter:

| Model | Role |
|---|---|
| **YOLOv11-Pose** | Per-frame player detection + 17-keypoint pose estimation |
| **LSTM** | Temporal action recognition — Idle / Sprinting / Kicking |
| **ESRGAN** | Optional 4× super-resolution upscaling before inference |

The platform also includes a full **in-app model training suite** for all three models.

---

## ✨ Features

- 🔐 **Secure Login** — credential-protected entry screen
- 📁 **Video Upload** — local `.mp4 / .avi / .mov` files
- 🎯 **YOLOv11 Pose Detection** — player tracking with 17-keypoint skeleton overlay
- 🏃 **LSTM Action Recognition** — classifies actions from a rolling 30-frame keypoint buffer
- ✨ **ESRGAN Super-Resolution** — optional 4× frame upscaling pre-inference
- 🗺️ **2D Tactical Minimap** — live bird's-eye view of player positions
- 📊 **Analytics Dashboard** — pie chart, bar chart, action timeline, and per-frame log
- 🧠 **AI Assessment** — auto-generated match summary from aggregated stats
- 📄 **PDF Report Generator** — one-click match reports via ReportLab
- 🏋️ **Model Training Panel** — in-app training for ESRGAN, LSTM, and YOLOv11 fine-tuning

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- CUDA GPU *(optional — CPU fallback is available)*

### Installation

```bash
git clone https://github.com/<your-username>/GAIze-Sport.git
cd GAIze-Sport
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python GAIze-Sport.py
```

Login with:

| Field | Value |
|---|---|
| Username | `admin` |
| Password | `password` |

---

## 🎮 Usage

### Analysing a Video

1. Click **📁 Upload Video** in the sidebar.
2. Select a video file — the **Analysis Config Dialog** will appear.
3. Choose your YOLO / LSTM / ESRGAN model paths (or use built-in defaults) and confirm.
4. Analysis runs on a background thread. You'll see annotated frames in the player, live action labels, detection console output, and a live minimap.

### Viewing Results

Click **📊 Analysis** in the sidebar to see:
- Stat cards (frames processed, avg persons, dominant action, activity rate)
- Charts (action pie, detection density bar, frame-by-frame timeline)
- AI-generated assessment paragraph
- Per-frame detection log

### Generating a PDF Report

Click **📄 Generate Report** in the Analytics Panel — saved to `reports/match_report.pdf`.

### Model Training

Open **🧠 Model Training** in the sidebar:

| Tab | What it does |
|---|---|
| **ESRGAN** | Phase 1 (PSNR) + Phase 2 (GAN) super-resolution training |
| **LSTM** | Extract keypoints from clips, generate synthetic data, or train directly |
| **YOLO Fine-tune** | Run Ultralytics fine-tuning on a custom dataset YAML |

All three tabs stream live training logs and show a progress bar inside the app.

---

## 🧠 Model Details

### YOLOv11-Pose
Nano variant with persistent multi-object tracking (`.track()`). Detects 17 COCO keypoints per person per frame.

### LSTM Action Recogniser
```
Input:  (1, 30, 34)  — 30-frame window × 17 keypoints × (x, y)
Model:  2-layer LSTM (hidden=64) → FC → 3 classes
Output: Idle | Sprinting | Kicking
```
Class-weighted softmax balances imbalanced action frequencies.  
Weights: `models/LSTM Action Recognition Model.pth`

### ESRGAN (RRDB-Net)
- Upscales frames 4× before YOLO inference for better low-quality video accuracy.
- Two-phase training: pixel loss → perceptual + adversarial loss.  
- Weights: `models/ESRGAN Phase #2 Generator.pth`

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `CustomTkinter` | Desktop GUI |
| `OpenCV` | Video capture & frame processing |
| `Ultralytics` | YOLOv11 pose detection |
| `PyTorch` | LSTM & ESRGAN inference + training |
| `NumPy / Pandas / Matplotlib` | Data + charts |
| `ReportLab` | PDF report generation |
| `cryptography` | Fernet file/data encryption |

---

## 🗺️ Roadmap / Future Goals

- [ ] Multi-player per-entity LSTM tracking
- [ ] Live YouTube / RTMP stream analysis
- [ ] AI Chat widget with sports-specific fine-tuned model
- [ ] Team possession heatmaps
- [ ] Export annotated video with overlay
- [ ] Web-based dashboard mode
- [ ] Custom action class support (heading, passing, etc.)

---

## 👤 Author

**Viplove** — Major Project II (2026)

> *Powered by PyTorch • Ultralytics • OpenCV • CustomTkinter • NumPy • Matplotlib • ReportLab • Pillow*
