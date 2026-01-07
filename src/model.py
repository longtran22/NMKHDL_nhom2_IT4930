import torch
import torch.nn as nn
import torchvision.models as tv
import timm  # ← THÊM THƯ VIỆN timm
from transformers import (
    AutoFeatureExtractor,
    ASTModel,
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    WhisperModel,
    WhisperFeatureExtractor
)

# Auto-mapping cho audio models
AUDIO_MODEL_CONFIGS = {
    "ast": {
        "model_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "feature_dim": 768,
        "model_class": ASTModel,
        "processor_class": AutoFeatureExtractor
    },
    "wavlm": {
        "model_id": "microsoft/wavlm-base",
        "feature_dim": 768,
        "model_class": Wav2Vec2Model,
        "processor_class": Wav2Vec2FeatureExtractor
    },
    "whisper": {
        "model_id": "openai/whisper-base",
        "feature_dim": 512,
        "model_class": WhisperModel,
        "processor_class": WhisperFeatureExtractor
    }
}

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_a, d_v):
        super().__init__()
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.a_proj = nn.Linear(d_a, 256)
        self.v_proj = nn.Linear(d_v, 256)
        self.ln_q = nn.LayerNorm(256)
        self.ln_kv = nn.LayerNorm(256)
        self.out = nn.Linear(256, 256)
        
    def forward(self, a, v):
        q = self.ln_q(self.a_proj(a)).unsqueeze(1)
        kv = self.ln_kv(self.v_proj(v)).unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        return self.out((q + out).squeeze(1))

class MultimodalEmotionRecognizer(nn.Module):
    def __init__(self, num_classes, fusion, image_backbone, audio_backbone, T):
        super().__init__()
        
        print(f"🏗️  Building: {image_backbone.upper()} + {audio_backbone.upper()}")
        
        # ==========================================
        # IMAGE BACKBONE
        # ==========================================
        if image_backbone == "resnet18":
            print("   📸 Loading ResNet18...")
            rn = tv.resnet18(weights="IMAGENET1K_V1")
            rn.fc = nn.Identity()
            self.visual_net = rn
            visual_feat_dim = 512
            
        elif image_backbone == "resnet50":
            print("   📸 Loading ResNet50...")
            rn = tv.resnet50(weights="IMAGENET1K_V2")
            rn.fc = nn.Identity()
            self.visual_net = rn
            visual_feat_dim = 2048
            
        # ==========================================
        # NEW: DARKNET-53 (via timm)
        # ==========================================
        elif image_backbone == "darknet53":
            print("   📸 Loading Darknet-53...")
            # Load từ timm với pretrained ImageNet
            darknet = timm.create_model('darknet53', pretrained=True, num_classes=0)
            # num_classes=0 → Remove classification head, return features
            self.visual_net = darknet
            visual_feat_dim = 1024  # Darknet-53 output dimension
            print("   ✅ Darknet-53 loaded from timm (ImageNet pretrained)")
            
        elif image_backbone == "efficientnet_b3":
            print("   📸 Loading EfficientNet-B3...")
            eff = tv.efficientnet_b3(weights="IMAGENET1K_V1")
            eff.classifier = nn.Identity()
            self.visual_net = eff
            visual_feat_dim = 1536
            
        elif image_backbone == "convnext_tiny":
            print("   📸 Loading ConvNeXt-Tiny...")
            cnx = tv.convnext_tiny(weights="IMAGENET1K_V1")
            cnx.classifier[2] = nn.Identity()
            self.visual_net = cnx
            visual_feat_dim = 768
            
        elif image_backbone == "vit_b_16":
            print("   📸 Loading ViT-B/16...")
            vit = tv.vit_b_16(weights="IMAGENET1K_V1")
            vit.heads = nn.Identity()
            self.visual_net = vit
            visual_feat_dim = 768
            
        elif image_backbone == "regnet_y_800mf":
            print("   📸 Loading RegNetY-800MF...")
            reg = tv.regnet_y_800mf(weights="IMAGENET1K_V2")
            reg.fc = nn.Identity()
            self.visual_net = reg
            visual_feat_dim = 784
            
        else:
            raise ValueError(f"Unsupported image backbone: {image_backbone}")
        
        self.img_proj = nn.Linear(visual_feat_dim, 128)
        self.img_norm = nn.LayerNorm(128)
        self.image_backbone_type = image_backbone  # ← Store type for utils
        
        # ==========================================
        # AUDIO BACKBONE
        # ==========================================
        if audio_backbone not in AUDIO_MODEL_CONFIGS:
            raise ValueError(f"Unsupported audio backbone: {audio_backbone}")
        
        audio_config = AUDIO_MODEL_CONFIGS[audio_backbone]
        model_id = audio_config["model_id"]
        audio_feat_dim = audio_config["feature_dim"]
        
        print(f"   🎵 Loading {audio_backbone.upper()} from {model_id}...")
        
        self.processor = audio_config["processor_class"].from_pretrained(model_id)
        self.audio_net = audio_config["model_class"].from_pretrained(model_id)
        self.audio_backbone_type = audio_backbone
        
        print(f"   ✅ Audio ready (output dim: {audio_feat_dim})")
        
        # ==========================================
        # FUSION & CLASSIFIER
        # ==========================================
        self.fusion = CrossAttentionFusion(audio_feat_dim, 256)
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        print(f"   🎉 Model built successfully!")

    def forward(self, img, wave):
        # Visual processing
        B, T, C, H, W = img.shape
        v = self.img_norm(self.img_proj(
            self.visual_net(img.view(B*T, C, H, W))
        )).view(B, T, -1)
        v_pool = torch.cat([v.mean(1), v.max(1).values], 1)
        
        # Audio processing
        if wave.dim() == 3:
            wave = wave.squeeze(1)
        
        if self.audio_backbone_type == "ast":
            inputs = self.processor(wave.cpu().numpy(), sampling_rate=16000, 
                                   return_tensors="pt", padding=True)
            inputs = {k: v.to(wave.device) for k, v in inputs.items()}
            audio_out = self.audio_net(**inputs).last_hidden_state
            a = audio_out.mean(dim=1)
            
        elif self.audio_backbone_type == "wavlm":
            inputs = self.processor(wave.cpu().numpy(), sampling_rate=16000, 
                                   return_tensors="pt", padding=True)
            inputs = {k: v.to(wave.device) for k, v in inputs.items()}
            audio_out = self.audio_net(**inputs).last_hidden_state
            a = audio_out.mean(dim=1)
            
        elif self.audio_backbone_type == "whisper":
            inputs = self.processor(wave.cpu().numpy(), sampling_rate=16000, 
                                   return_tensors="pt")
            input_features = inputs.input_features.to(wave.device)
            audio_out = self.audio_net.encoder(input_features).last_hidden_state
            a = audio_out.mean(dim=1)
        
        return self.classifier(self.fusion(a, v_pool))