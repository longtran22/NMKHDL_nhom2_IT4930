# Emotion Recognition App

This app allows you to perform speech emotion recognition using a WavLM + SL-WDEE model.

## Setup

1.  **Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

2.  **Models**:
    **IMPORTANT**: You must place your trained model files (`.pth`) in the `models/` directory.
    - Example: Copy `best_model.pth` to `Emotion_audio/models/best_model.pth`

## Usage

Run the Graphical User Interface (GUI):

```powershell
python app.py
```

- The app will open in your browser.
- Select your model from the dropdown.
- Upload an audio file or use the Microphone tab.
- Click **Predict Emotion**.

## Command Line

You can also use the CLI for batch processing:

```powershell
python inference.py --audio_path "path/to/audio.wav" --model_path "models/best_model.pth"
```
