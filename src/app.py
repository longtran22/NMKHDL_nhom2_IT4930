import gradio as gr
import torch
import torchaudio
import os
import glob
from model import SER_WavLM_Final, IDX2EMO
# Import Image Model Dependencies
from model_image import ResNet18_Emotion, ResNet50_Emotion, preprocess_image
from model_combined import MultimodalEmotionRecognizer
import numpy as np
import cv2
import tempfile
from torchvision import transforms


# =========================
# Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_MODELS_DIR = os.path.join(MODELS_DIR, "images")

DEFAULT_WAVLM = "microsoft/wavlm-base-plus"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on device: {DEVICE}")

# =========================
# Shared Resources
# =========================
# Cache the model to avoid reloading on every prediction if not changed
current_model = None
current_model_path = None
# Cache for Image Model
current_image_model = None
current_image_model_path = None

# =========================
# AUDIO Loading & Inference
# =========================
def load_audio_model(model_path):
    global current_model, current_model_path
    
    if current_model is not None and current_model_path == model_path:
        return current_model

    print(f"Loading Audio model from {model_path}...")
    try:
        model = SER_WavLM_Final(wavlm_name=DEFAULT_WAVLM, wdee_hidden=512, wdee_out=256, dropout=0.5)
        model.to(DEVICE)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            
        # Helper to remove 'module.' prefix if wrapped in DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"WARNING: Missing keys: {missing}")
        if unexpected:
            print(f"WARNING: Unexpected keys: {unexpected}")
            
        model.eval()
        
        current_model = model
        current_model_path = model_path
        print("Audio Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load Audio model: {e}")
        return None

def preprocess_audio(audio_path, target_sr=16000):
    if audio_path is None:
        return None
    try:
        # torchaudio 2.x with soundfile installed should handle this automatically
        wav, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error reading audio: {e}")
        return None

    # Mix to mono
    wav = wav.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # Remove channel dim -> [T]
    wav = wav.squeeze(0)
    # Normalize
    wav = wav / (wav.abs().max() + 1e-9)
    return wav

def predict_audio_emotion(model_name, audio_source, microphone_source):
    audio_path = microphone_source if microphone_source else audio_source
    
    if not audio_path:
        return "Please provide an audio input (upload file or use microphone).", None

    if not model_name:
        return "Please select an Audio model to load.", None

    full_model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(full_model_path):
        return f"Model file not found: {full_model_path}", None
        
    model = load_audio_model(full_model_path)
    if model is None:
        return "Failed to load model. Check console for errors.", None

    wav_tensor = preprocess_audio(audio_path)
    if wav_tensor is None:
        return "Error processing audio file.", None
        
    wav_tensor = wav_tensor.to(DEVICE)
    # Add batch dim [1, T]
    wav_batch = wav_tensor.unsqueeze(0)
    lengths = torch.tensor([wav_tensor.shape[0]], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(wav_batch, lengths)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0] # [6]

    labels_conf = {IDX2EMO[i]: float(probs[i]) for i in range(len(probs))}
    top_label = IDX2EMO[probs.argmax()]
    return f"Predicted: **{top_label}**", labels_conf

# =========================
# IMAGE Loading & Inference
# =========================
def load_image_model(model_path):
    global current_image_model, current_image_model_path
    
    if current_image_model is not None and current_image_model_path == model_path:
        return current_image_model

    print(f"Loading Image model from {model_path}...")
    try:
        # Initialize Emotion model based on filename
        if "resnet50" in model_path.lower():
            print("Detected ResNet50 model.")
            model = ResNet50_Emotion(num_classes=6, dropout=0.3)
        else:
            print("Detected ResNet18 model (default).")
            model = ResNet18_Emotion(num_classes=6, dropout=0.3)
            
        model.to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        # Handle different saving variants
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint: # Sometimes saved as 'model'
            state_dict = checkpoint["model"]
            
        # Clean state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"WARNING (Image): Missing keys: {missing}")
        if unexpected:
            print(f"WARNING (Image): Unexpected keys: {unexpected}")

        model.eval()
        current_image_model = model
        current_image_model_path = model_path
        print("Image Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load Image model: {e}")
        return None

def predict_image_emotion(model_name, image_input):
    if image_input is None:
        return "Please provide an image.", None
        
    if not model_name:
        return "Please select an Image model.", None
        
    full_model_path = os.path.join(IMAGES_MODELS_DIR, model_name)
    if not os.path.exists(full_model_path):
        return f"Model file not found: {full_model_path}", None
    
    model = load_image_model(full_model_path)
    if model is None:
        return "Failed to load image model.", None
        
    # Preprocess
    img_tensor = preprocess_image(image_input) # Returns [C, H, W]
    if img_tensor is None:
        return "Error preprocessing image.", None
        
    img_batch = img_tensor.unsqueeze(0).to(DEVICE) # [1, C, H, W]
    
    with torch.no_grad():
        logits = model(img_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    labels_conf = {IDX2EMO[i]: float(probs[i]) for i in range(len(probs))}
    top_label = IDX2EMO[probs.argmax()]
    return f"Predicted: **{top_label}**", labels_conf

# =========================
# Helper Utils
# =========================
def refresh_audio_models():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    # Only pick files directly in models/, not subdirs
    files = [f for f in glob.glob(os.path.join(MODELS_DIR, "*.pth")) if os.path.isfile(f)]
    return [os.path.basename(f) for f in files]

def refresh_image_models():
    if not os.path.exists(IMAGES_MODELS_DIR):
        os.makedirs(IMAGES_MODELS_DIR)
    files = glob.glob(os.path.join(IMAGES_MODELS_DIR, "*.pth"))
    return [os.path.basename(f) for f in files]

# =========================
# VIDEO / COMBINED MODEL Inference
# =========================
def load_combined_model(model_path):
    print(f"Loading Combined Model from {model_path}...")
    try:
        # Default config from notebook: resnet50 + wavlm
        model = MultimodalEmotionRecognizer(
            num_classes=6, 
            fusion="crossattn", 
            image_backbone="resnet50", 
            audio_backbone="wavlm"
        )
        model.to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            
        # Clean state_dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        print("Combined Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load Combined Model: {e}")
        return None

def process_video_frames(video_path, num_frames=3):
    """
    Extracts `num_frames` from video evenly spaced.
    Returns: torch.Tensor [T, C, H, W]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # Fallback if read fails
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
    cap.release()
    
    # Preprocess frames
    # Matches dataset.py transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    frame_tensors = []
    for f in frames:
        # To PIL or directly to Tensor? transform accepts PIL or Tensor. 
        # transforms.ToTensor() expects HWC numpy array -> CHW tensor 0-1
        frame_tensors.append(transform(f))
        
    # Stack [T, C, H, W]
    return torch.stack(frame_tensors)

def process_video_audio(video_path, target_sr=16000, max_len=64000):
    """
    Extracts audio from video using moviepy (more robust for MP4 on Windows).
    """
    temp_wav_path = None
    try:
        try:
            # Try new moviepy 2.0+ import first
            from moviepy import VideoFileClip
        except ImportError:
            # Fallback to old moviepy.editor import
            from moviepy.editor import VideoFileClip
        except ImportError:
            # Fallback or error if moviepy not installed
            print("MoviePy not found. Installing it is recommended for video audio extraction.")
            # Try torchaudio directly as fallback (which likely failed before)
            wav, sr = torchaudio.load(video_path)
            # If we reached here, torchaudio worked
        else:
            # Use moviepy
            video_clip = VideoFileClip(video_path)
            if video_clip.audio is None:
                # Silent video
                video_clip.close()
                return torch.zeros(1, max_len)
            
            # Create temp wav
            fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            # Write audio
            video_clip.audio.write_audiofile(temp_wav_path, fps=target_sr, logger=None)
            video_clip.close()
            
            # Load with torchaudio
            wav, sr = torchaudio.load(temp_wav_path)
            
        # Mix to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            
        # Pad/Trim to max_len (64000 ~ 4s)
        wav = wav.squeeze(0) # [L]
        L = wav.shape[0]
        T = max_len
        
        if L > T:
            start = (L - T) // 2
            wav = wav[start : start+T]
        else:
            wav = torch.nn.functional.pad(wav, (0, T-L))
            
        return wav.unsqueeze(0) # [1, T]

    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return torch.zeros(1, max_len)
    finally:
        # Cleanup temp file
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except:
                pass

def predict_video_emotion(model_path, video_input, num_frames=10):
    if not video_input:
        return "Please upload a video.", None
        
    if not model_path:
        return "Please select a Combined model.", None
        
    # Load model (caching logic could be added similar to others)
    model = load_combined_model(model_path)
    if model is None:
        return "Failed to load model.", None
        
    # Process inputs
    # 1. Frames
    frames = process_video_frames(video_input, num_frames=int(num_frames))
    if frames is None:
        return "Error reading video frames.", None
    
    # 2. Audio
    audio = process_video_audio(video_input)
    
    # Prepare batch [1, ...]
    frames = frames.unsqueeze(0).to(DEVICE) # [1, T, C, H, W]
    audio = audio.to(DEVICE) # [1, T_audio] (T_audio=64000)
    
    with torch.no_grad():
        logits = model(frames, audio)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    labels_conf = {IDX2EMO[i]: float(probs[i]) for i in range(len(probs))}
    top_label = IDX2EMO[probs.argmax()]
    return f"Predicted: **{top_label}**", labels_conf

# =========================
# UI Construction
# =========================
def create_ui():
    audio_models = refresh_audio_models()
    image_models = refresh_image_models()
    
    init_audio = audio_models[0] if audio_models else None
    init_image = image_models[0] if image_models else None

    # Helper to find combined models
    def refresh_combined_models():
        combined_dir = os.path.join(MODELS_DIR, "combine")
        if not os.path.exists(combined_dir):
            os.makedirs(combined_dir)
        files = glob.glob(os.path.join(combined_dir, "*.pth"))
        return files # Return full paths for simplicity or just basenames if we handle paths later?
                     # Let's return full paths but display basenames if possible? 
                     # For consistency with others, let's assume predict function handles full path if we pass absolute, 
                     # but others take basename and join with dir.
                     # Let's return paths, and update predict to handle it.
        return [f for f in files]

    combined_models = refresh_combined_models()
    init_combined = combined_models[0] if combined_models else None

    with gr.Blocks(title="Multimodal Emotion Recognition") as demo:
        gr.Markdown("# 🎭 Multimodal Emotion Recognition (Audio & Image & Video)")
        gr.Markdown(f"Running on: `{DEVICE.upper()}`")
        
        with gr.Tabs():
            # === TAB 1: AUDIO ===
            with gr.TabItem("🎤 Audio Recognition"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_model_dd = gr.Dropdown(label="Select Audio Model", choices=audio_models, value=init_audio)
                        refresh_audio_btn = gr.Button("🔄 Refresh Audio Models")
                        
                        with gr.Tab("Upload File"):
                            audio_upload = gr.Audio(sources=["upload"], type="filepath", label="Upload WAV")
                        with gr.Tab("Microphone"):
                            audio_mic = gr.Audio(sources=["microphone"], type="filepath", label="Record Mic")
                            
                        pred_audio_btn = gr.Button("🚀 Predict Audio Emotion", variant="primary")
                    
                    with gr.Column(scale=1):
                        res_audio_text = gr.Markdown()
                        res_audio_plot = gr.Label(num_top_classes=6, label="Confidence (Audio)")

                refresh_audio_btn.click(lambda: gr.update(choices=refresh_audio_models()), None, audio_model_dd)
                pred_audio_btn.click(
                    fn=predict_audio_emotion,
                    inputs=[audio_model_dd, audio_upload, audio_mic],
                    outputs=[res_audio_text, res_audio_plot]
                )

            # === TAB 2: IMAGE ===
            with gr.TabItem("📷 Image Recognition"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_model_dd = gr.Dropdown(label="Select Image Model", choices=image_models, value=init_image)
                        refresh_image_btn = gr.Button("🔄 Refresh Image Models")
                        
                        # Image Input (Webcam or Upload)
                        image_input = gr.Image(type="pil", label="Input Image")
                        
                        pred_image_btn = gr.Button("🚀 Predict Image Emotion", variant="primary")
                        
                    with gr.Column(scale=1):
                        res_image_text = gr.Markdown()
                        res_image_plot = gr.Label(num_top_classes=6, label="Confidence (Image)")
                        
                refresh_image_btn.click(lambda: gr.update(choices=refresh_image_models()), None, image_model_dd)
                pred_image_btn.click(
                    fn=predict_image_emotion,
                    inputs=[image_model_dd, image_input],
                    outputs=[res_image_text, res_image_plot]
                )

            # === TAB 3: VIDEO (COMBINED) ===
            with gr.TabItem("🎬 Video Recognition"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Display basenames for dropdown, map back to full paths? 
                        # Or just listing full paths is fine for now usually.
                        combined_model_dd = gr.Dropdown(label="Select Combined Model", choices=combined_models, value=init_combined)
                        refresh_combined_btn = gr.Button("🔄 Refresh Combined Models")
                        
                        video_input = gr.Video(label="Upload Video (MP4/AVI)")
                        
                        # Add slider for number of frames
                        num_frames_slider = gr.Slider(minimum=3, maximum=30, value=10, step=1, label="Number of Frames to Extract")
                        
                        pred_video_btn = gr.Button("🚀 Predict Video Emotion", variant="primary")
                        
                    with gr.Column(scale=1):
                        res_video_text = gr.Markdown()
                        res_video_plot = gr.Label(num_top_classes=6, label="Confidence (video)")

                refresh_combined_btn.click(lambda: gr.update(choices=refresh_combined_models()), None, combined_model_dd)
                pred_video_btn.click(
                    fn=predict_video_emotion,
                    inputs=[combined_model_dd, video_input, num_frames_slider],
                    outputs=[res_video_text, res_video_plot]
                )

    return demo

if __name__ == "__main__":
    # Create dirs
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    if not os.path.exists(IMAGES_MODELS_DIR): os.makedirs(IMAGES_MODELS_DIR)
    
    print(f"Audio Models Path: {MODELS_DIR}")
    print(f"Image Models Path: {IMAGES_MODELS_DIR}")

    demo = create_ui()
    demo.launch(inbrowser=True)
