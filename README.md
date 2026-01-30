# NMKHDL_nhom2_IT4930

## Project Structure
- `notebooks/`: Contains Jupyter Notebooks for experiments.
  - `audio/`: Notebooks related to Audio Emotion Recognition (WavLM, Whisper).
  - `image/`: Notebooks related to Image Emotion Recognition (ResNet).
  - `multimodal/`: Notebooks for combined models.
- `src/`: Main source code for the Application (Gradio App).
- `docs/`: Project documentation.

## Summary: Speech Emotion Recognition (SER) Project

### Problem Statement
The goal is to build a Speech Emotion Recognition (SER) system to classify short audio utterances into 6 emotions: Angry, Disgust, Fear, Happy, Neutral, and Sad.

### Data
We use the **CREMA-D** dataset (7,442 WAV files from 91 actors).
- **Challenge**: Potential data leakage due to actor overlap.
- **Solution**: Actor-based split (Train/Test sets do not share actors).

### Pipelines
1. **WavLM + SL-WDEE**: 
   - Uses pre-trained WavLM for feature extraction.
   - SL-WDEE encoder for capturing emotional context.
   - MLP Classifier.
   - **Performance**: ~76% Unweighted Accuracy (UA) on Test.
   
2. **Whisper Fine-tuning**:
   - Fine-tunes OpenAI's Whisper base model.
   - **Performance**: ~70% Accuracy.

### Results
The WavLM + SL-WDEE approach outperforms the baseline and Whisper fine-tuning, achieving robust results on the CREMA-D dataset.

### Usage
To run the application:
```bash
python src/app.py
```
