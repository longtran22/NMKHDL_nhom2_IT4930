NUM_CLASSES = 6
EMOTION_MAP = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "SAD": 4, "NEU": 5}
DATA_PATH = "" 
CSV_PATH = "./train_dataset.csv"
IMAGE_SIZE = 224
SEQ_LEN = 3
SAMPLE_RATE = 16000
WAVE_TARGET_LEN = 64000

# Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 25
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY_HEAD = 5e-3
WEIGHT_DECAY_BACKBONE = 1e-3
SEED = 42

# ==========================================
# MODEL SELECTION - CHỈ SỬA 2 DÒNG!
# ==========================================
IMAGE_BACKBONE = "resnet50"  # Options: resnet18, resnet50, darknet53, efficientnet_b3, convnext_tiny, vit_b_16
AUDIO_BACKBONE = "wavlm"      # Options: ast, wavlm, whisper

# Fine-tuning
FREEZE_BACKBONES = True
IMG_UNFREEZE_LAST_BLOCKS = 1
AUDIO_UNFREEZE_LAST_BLOCKS = 1

# Regularization
USE_MIXUP = True
MIXUP_PROB = 0.3
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 5

# Auto-generated paths
FUSION_TYPE = "crossattn"
MODEL_SAVE_PATH = f"./checkpoints/best_{IMAGE_BACKBONE}_{AUDIO_BACKBONE}.pth"