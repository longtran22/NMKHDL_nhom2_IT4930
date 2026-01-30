import torch
import torchaudio
import argparse
import os
import numpy as np
from model import SER_WavLM_WDEE, IDX2EMO

# =========================
# Preprocessing
# =========================
def load_and_preprocess_wav(path, sr=16000):
    try:
        wav, origin_sr = torchaudio.load(path)
    except Exception:
         # Fallback logic if needed, but standard load should work with soundfile
         import soundfile as sf
         wav, origin_sr = sf.read(path)
         wav = torch.tensor(wav).float()
         if len(wav.shape) == 1:
             wav = wav.unsqueeze(0) # [1, T]
         else:
             wav = wav.transpose(0, 1) # [C, T]
         origin_sr = 16000 # Assume or read from sf info if possible, but basic fallback

    wav = wav.mean(dim=0, keepdim=True)  # Chuyển sang mono
    if origin_sr != sr:
        wav = torchaudio.functional.resample(wav, origin_sr, sr)
    wav = wav.squeeze(0)  # Bỏ chiều kênh nếu là mono [T]
    # Chuẩn hóa năng lượng nhẹ về [-1, 1]
    wav = wav / (wav.abs().max() + 1e-9)
    return wav

# =========================
# Main Inference
# =========================
def main():
    parser = argparse.ArgumentParser(description="Inference Emotion from Audio")
    parser.add_argument("--audio_path", type=str, required=True, help="Đường dẫn file audio .wav")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Đường dẫn file model .pth")
    parser.add_argument("--wavlm_name", type=str, default="microsoft/wavlm-base-plus", help="Tên model WavLM trên HF")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị chạy (cuda/cpu)")

    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # 1. Khởi tạo model
    print("Initializing model...")
    model = SER_WavLM_WDEE(wavlm_name=args.wavlm_name, wdee_hidden=512, wdee_out=256, freeze_wavlm=True)
    model.to(args.device)
    model.eval()

    # 2. Load weights
    print(f"Loading weights from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Not found {args.model_path}")
        return

    try:
        # Load checkpoint
        checkpoint = torch.load(args.model_path, map_location=args.device)
        
        # Xử lý các trường hợp lưu khác nhau (nếu lưu full dict hoặc chỉ state_dict)
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        # Helper to remove 'module.' and REMAP keys
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k
            if key.startswith("module."):
                key = key[7:]
                
            # REMAPPING LOGIC based on user feedback to match model.py structure
            if key.startswith("norm."):
                key = "encoder." + key
            elif key.startswith("fc1."):
                key = "encoder." + key
            elif key.startswith("fc2."):
                key = "encoder." + key
            elif key.startswith("classifier."):
                 # Checkpoint has classifier.0.weight -> cls.net.0.weight
                 key = key.replace("classifier.", "cls.net.")
            
            new_state_dict[key] = v
        state_dict = new_state_dict
            
        # Load state dict vào model
        missing, unexpected = model.load_state_dict(state_dict, strict=False) # strict=False để tránh lỗi nếu có sai lệch nhỏ tên layer
        if missing:
            print(f"WARNING: Missing keys: {missing}")
        if unexpected:
            print(f"WARNING: Unexpected keys: {unexpected}")

        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # 3. Preprocess Audio
    print(f"Processing audio: {args.audio_path}")
    try:
        wav = load_and_preprocess_wav(args.audio_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # 4. Inference
    wav = wav.to(args.device)
    # Tạo batch dimension [1, T] và lengths [1]
    wav_batch = wav.unsqueeze(0)
    lengths = torch.tensor([wav.shape[0]], dtype=torch.long).to(args.device)

    with torch.no_grad():
        logits, _ = model(wav_batch, lengths)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    pred_label = IDX2EMO[pred_idx]
    confidence = probs[0, pred_idx].item()

    print("\n" + "="*30)
    print(f"RESULT: {pred_label} ({confidence*100:.2f}%)")
    print("="*30)
    print("Details:")
    for idx, prob in enumerate(probs[0]):
        label = IDX2EMO[idx]
        print(f"{label}: {prob.item()*100:.2f}%")

if __name__ == "__main__":
    main()
