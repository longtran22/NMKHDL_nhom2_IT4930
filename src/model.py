import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# =========================
# Utils
# =========================
EMO_MAP = {"ANG":0, "DIS":1, "FEA":2, "HAP":3, "NEU":4, "SAD":5}
IDX2EMO = {v:k for k,v in EMO_MAP.items()}

# =========================
# Audio Model Definitions (SER_WavLM_Final)
# =========================
class ContextualTransform(nn.Module):
    def __init__(self, left=3, right=3):
        super().__init__()
        self.l, self.r = left, right
    def forward(self, x):
        # x: [B, T, D]
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (self.l, self.r), mode="replicate").transpose(1, 2)
        ctx_list = []
        T = x.shape[1]
        for i in range(self.l + self.r + 1):
            ctx_list.append(x_pad[:, i : i+T, :])
        return torch.cat(ctx_list, dim=-1)

class SER_WavLM_Final(nn.Module):
    def __init__(self, wavlm_name="microsoft/wavlm-base-plus", wdee_hidden=512, wdee_out=256, dropout=0.5):
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(wavlm_name)
        self.wavlm.config.output_hidden_states = True
        
        # --- SMART FREEZING (Optional in inference, but structure must match) ---
        # We don't need to freeze here for inference, but creating parameters is crucial
        
        hidden_dim = self.wavlm.config.hidden_size # 768
        num_layers = self.wavlm.config.num_hidden_layers + 1 # 13
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        self.norm = nn.LayerNorm(hidden_dim)
        self.ctx = ContextualTransform(left=3, right=3) 
        self.fc1 = nn.Linear(hidden_dim * 7, wdee_hidden) # 768 * 7
        self.fc2 = nn.Linear(wdee_hidden, wdee_out)
        
        self.classifier = nn.Sequential(
            nn.Linear(wdee_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(128, 6)
        )

    def forward(self, wav, lengths):
        B, T_raw = wav.shape
        device = wav.device
        mask_raw = (torch.arange(T_raw, device=device)[None, :] < lengths[:, None]).long()
        
        outputs = self.wavlm(input_values=wav, attention_mask=mask_raw)
        
        # Weighted sum of hidden layers
        hidden_states = torch.stack(outputs.hidden_states, dim=0) # [Layer, B, T, D]
        weights = F.softmax(self.layer_weights, dim=0)
        weighted_feat = (hidden_states * weights[:, None, None, None]).sum(dim=0) # [B, T, D]
        
        x = self.norm(weighted_feat)
        x = self.ctx(x)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        # Resize mask to match feature length (if roughly same, use nearest)
        T_feat = x.shape[1]
        mask_feat = F.interpolate(mask_raw.unsqueeze(1).float(), size=T_feat, mode='nearest').squeeze(1)
        
        mask_expanded = mask_feat.unsqueeze(-1)
        # Global Average Pooling with Mask
        x_utt = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        
        logits = self.classifier(x_utt)
        return logits, x_utt

# Alias for compatibility if needed, or just use SER_WavLM_Final
SER_WavLM_WDEE = SER_WavLM_Final
