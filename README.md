# 🎭 Multimodal Emotion Recognition (Audio, Image, Video)

## 📌 Project Overview
This project develops a comprehensive Multimodal Emotion Recognition system capable of identifying emotional states from three different modalities: **Speech (Audio)**, **Facial Expressions (Image)**, and **Video (Combined Audio-Visual)**.

We classify inputs into 6 core emotions: **Angry, Disgust, Fear, Happy, Neutral,** and **Sad**.

---

## 📂 Project Structure
- `notebooks/`: Jupyter Notebooks for research and training.
  - `audio/`: Audio Emotion Recognition experiments (WavLM, Whisper).
  - `image/`: Facial Expression Recognition (ResNet18, ResNet50).
  - `multimodal/`: Combined models using Cross-Attention fusion.
- `src/`: Core application source code.
  - `app.py`: Main Gradio interface.
  - `models/`: Directory for trained model weights (`.pth` files).
- `docs/`: Technical documentation and project reports.

---

## 🚀 Key Pipelines

### 1. 🎤 Audio Emotion Recognition (SER)
- **Model**: WavLM + SL-WDEE (Spec-Label Weighted Dual-Encoder).
- **Dataset**: CREMA-D (Actor-based split to avoid data leakage).
- **Results**: Achieved ~76% Unweighted Accuracy (UA).

### 2. 📷 Image (Facial) Emotion Recognition
- **Model**: ResNet18 & ResNet50 fine-tuned on facial datasets.
- **Approach**: Captures spatial features representing micro-expressions.

### 3. 🎬 Video (Multimodal) Recognition
- **Model**: Multimodal Fusion (ResNet50 + WavLM).
- **Fusion Technique**: Cross-Attention mechanism to learn interactions between audio and visual cues.
- **Processing**: Extracts frames and synchronized audio for holistic prediction.

---

## 🛠 Installation & Usage

### 1. Requirements
- Python 3.9+
- FFmpeg (for video/audio processing)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/longtran22/NMKHDL_nhom2_IT4930.git
cd NMKHDL_nhom2_IT4930

# Install dependencies
pip install -r src/requirements.txt
```

### 3. Running the App
The application provides a user-friendly Gradio interface with three tabs for each modality.
```bash
python src/app.py
```
Access the UI at: `http://127.0.0.1:7860`

---

## 🐳 Docker Deployment
You can also run the application using Docker:
```bash
cd src
docker-compose up --build
```
