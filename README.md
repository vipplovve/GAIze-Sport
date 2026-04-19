<div align="center">

# 🛡️ GAIze-Sport

### AI-Driven Sports Video Analysis, Frame Enhancement & Action Recognition

**Real-time Detection &nbsp;•&nbsp; Frame Enhancement &nbsp;•&nbsp; Behavioral Logic &nbsp;•&nbsp; Data Sovereignty**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv11-green)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-orange)](LICENSE)
[![Environment](https://img.shields.io/badge/OS-Ubuntu_Linux-E9433F?logo=ubuntu&logoColor=white)](https://ubuntu.com/)

</div>

---

## 📖 Project Overview

**GAIze-Sport** is a localized AI platform designed to bridge the gap between amateur match footage and professional-grade sports science. Unlike cloud-heavy alternatives, this platform operates on a **Local-First** mandate, ensuring that tactical secrets and athlete biometrics never leave the coach's encrypted workstation.

By cascading super-resolution, pose estimation, and temporal sequence modeling, GAIze-Sport transforms raw video into a "mathematical skeleton" of the game, providing automated insights without the need for expensive wearable sensors or high-bandwidth cloud uploads.

---

## 🏗️ Project Architecture

The system operates as a sequential pipeline where each stage adds a layer of intelligence to the raw pixel data:

1.  **Enhancement (ESRGAN):** Low-resolution or "noisy" frames are upscaled by 4x. This stage ensures the system can "see" fine details like foot placement and ball proximity before any detection occurs.
2.  **Digitization (YOLO11-Pose):** The system extracts 17 anatomical keypoints per athlete, converting human movement into a precise mathematical skeleton.
3.  **Persistence (BoT-SORT):** Multi-object tracking with Camera Motion Compensation ensures player identities remain stable even during physical contact or camera pans.
4.  **Intent Recognition (Action LSTM):** A 3-layer LSTM studies the "rhythm" of the skeletons over a 30-frame rolling buffer to distinguish between actions (e.g., *Sprinting* vs. *Kicking*).
5.  **Synthesis (ReportLab):** Aggregated data is programmatically converted into a PDF Match Report, complete with tactical minimaps and performance timelines.



---

## 🎯 Use Cases

- **Budget-Conscious Scouting:** Provides a useful resource for local academies and collegiate teams who cannot afford GPS vests or high-end sensor arrays.
- **Injury Prevention:** By monitoring skeletal angles in real-time, the system can flag improper landing techniques or fatigue-based posture changes.
- **Tactical Narrative:** Coaches can review the analysis of the game to analyze team spacing and defensive structures immediately after the final whistle.
- **Secure Analysis:** Ideal for organizations with strict data privacy requirements that require air-gapped, local-only processing.

---

## 💻 Languages & Core Tools

<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=pytorch,opencv,python,linux,bash,vscode,git" />
  </a>
</p>

<p align="center">
  <em>Developed on <strong>Ubuntu 24.04</strong> using <strong>Antigravity</strong> as the primary agentic IDE. </em>
</p>

---

## 🚀 Installation & Setup

### Prerequisites
- NVIDIA GPU (Recommended: 8GB+ VRAM)
- CUDA Toolkit installed
- Python 3.10+

### Quick Start
```bash
git clone https://github.com/vipplovve/GAIze-Sport.git
cd GAIze-Sport
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python GAIze-Sport.py
```

> [!IMPORTANT]
> GAIze-Sport is developed strictly for academic, research, and educational purposes. It is a proof-of-concept for localized AI in sports science. This software is not intended for commercial broadcasting, sports betting, or other such applications. Any use of this tool to facilitate unauthorized data scraping or betting-related analytics is strictly prohibited. The software is provided "as-is" without any warranty. The author assumes no responsibility for any tactical, physical, or legal consequences resulting from the application of these AI-generated insights.

## 📜 Attribution & License
Copyright © 2026 Viplove Tyagi. This project is licensed under AGPL-3.0. Models were trained and fine-tuned using CC-BY footage from lozzzproductionz (Football) and SportsAlgo (Basketball). Full information available in Datasources.md.

Author: Viplove Tyagi
