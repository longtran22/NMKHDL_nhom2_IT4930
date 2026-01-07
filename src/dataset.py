import os
import torch
import cv2
import numpy as np
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms
from src import config

def add_noise(waveform, noise_level=0.005):
    '''Thêm nhiễu trắng vào audio'''
    if torch.rand(1) < 0.5:
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    return waveform

def load_wav(path, target_sr):
    '''Load và resample audio file'''
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 1:
            wav = torch.from_numpy(audio).unsqueeze(0)
        else:
            wav = torch.from_numpy(audio).T.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return torch.zeros(1, 64000)

class EmotionDataset(Dataset):
    def __init__(self, data_path, df, is_train=True):
        self.df = df
        self.is_train = is_train
        
        # Image transformations
        if is_train:
            self.tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load images
        images = []
        if os.path.exists(str(row["video_frame_dir"])):
            files = sorted(os.listdir(row["video_frame_dir"]))
            
            if len(files) > 3:
                if self.is_train:
                    indices = sorted(np.random.choice(len(files), 3, replace=False))
                else:
                    indices = np.linspace(0, len(files)-1, 3, dtype=int)
                files = [files[i] for i in indices]
            
            for p in files:
                img_path = os.path.join(row["video_frame_dir"], p)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(self.tfm(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        
        # Padding if needed
        while len(images) < 3:
            images.append(images[-1] if images else torch.zeros(3, 224, 224))
        images = torch.stack(images)

        # Load audio
        wav = load_wav(row["audio_path"], 16000)
        
        # Audio augmentation
        if self.is_train:
            wav = add_noise(wav)
            
        # Trim/Pad audio
        L, T = wav.size(-1), 64000
        if L > T:
            s = np.random.randint(0, L-T+1) if self.is_train else (L-T)//2
            wav = wav[:, s:s+T]
        else:
            wav = torch.nn.functional.pad(wav, (0, T-L))
            
        return images, wav.squeeze(0), torch.tensor(config.EMOTION_MAP[row["emotion"]])