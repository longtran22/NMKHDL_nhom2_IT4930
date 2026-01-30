
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# Image Model Definition
# =========================
def ResNet18_Emotion(num_classes=6, dropout=0.3):
    """
    Returns a ResNet18 model modified for emotion recognition.
    Matches the structure: conv1, layer1..., fc.1...
    """
    # Load pre-trained ResNet18
    model = models.resnet18(weights="IMAGENET1K_V1")
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )
    
    return model

def ResNet50_Emotion(num_classes=6, dropout=0.3):
    """
    Returns a ResNet50 model modified for emotion recognition.
    """
    # Load pre-trained ResNet50
    model = models.resnet50(weights="IMAGENET1K_V2")
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes)
    )
    
    return model

# =========================
# Preprocessing
# =========================
def preprocess_image(image_path_or_obj, target_size=(224, 224)):
    """
    Load image, resize, normalize for ResNet.
    image_path_or_obj: str path or PIL Image
    """
    if isinstance(image_path_or_obj, str):
        try:
            image = Image.open(image_path_or_obj).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    else:
        # Assuming it's already a PIL Image (e.g. from Gradio)
        image = image_path_or_obj.convert('RGB')

    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image)
    return img_tensor
