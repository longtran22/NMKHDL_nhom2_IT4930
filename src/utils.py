import torch
import torch.nn as nn

def setup_finetune(model, img_unfreeze_last_blocks=1, audio_unfreeze_last_blocks=1):
    print("🔧 Setting up fine-tuning...")
    
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    
    # Unfreeze head
    for m in [model.img_proj, model.fusion, model.classifier]:
        for p in m.parameters():
            p.requires_grad = True
    
    # ==========================================
    # UNFREEZE IMAGE BACKBONE
    # ==========================================
    visual_net = model.visual_net
    backbone_type = model.image_backbone_type
    
    # ResNet18/50
    if backbone_type in ["resnet18", "resnet50"]:
        if hasattr(visual_net, 'layer1'):
            layers = [visual_net.layer1, visual_net.layer2, 
                      visual_net.layer3, visual_net.layer4]
            if img_unfreeze_last_blocks > 0:
                for blk in layers[-img_unfreeze_last_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                print(f"   📸 Unfroze last {img_unfreeze_last_blocks} block(s) of {backbone_type.upper()}")
    
    # ==========================================
    # DARKNET-53
    # ==========================================
    elif backbone_type == "darknet53":
        # Darknet-53 structure in timm:
        # stages: [stage0, stage1, stage2, stage3, stage4]
        # stage4 is the deepest
        if hasattr(visual_net, 'stages'):
            stages = visual_net.stages
            if img_unfreeze_last_blocks > 0:
                # Unfreeze last N stages
                for stage in list(stages)[-img_unfreeze_last_blocks:]:
                    for p in stage.parameters():
                        p.requires_grad = True
                print(f"   📸 Unfroze last {img_unfreeze_last_blocks} stage(s) of Darknet-53")
        else:
            # Fallback: try to access blocks
            # timm models có thể có cấu trúc khác
            print("   ⚠️ Darknet-53 structure detection, unfreezing all layers")
            for p in visual_net.parameters():
                p.requires_grad = True
    
    # EfficientNet
    elif backbone_type == "efficientnet_b3":
        if hasattr(visual_net, 'features'):
            blocks = list(visual_net.features.children())
            if img_unfreeze_last_blocks > 0:
                for blk in blocks[-img_unfreeze_last_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                print(f"   📸 Unfroze last {img_unfreeze_last_blocks} block(s) of EfficientNet")
    
    # ConvNeXt
    elif backbone_type == "convnext_tiny":
        if hasattr(visual_net, 'features'):
            stages = list(visual_net.features.children())
            if img_unfreeze_last_blocks > 0:
                for stage in stages[-img_unfreeze_last_blocks:]:
                    for p in stage.parameters():
                        p.requires_grad = True
                print(f"   📸 Unfroze last {img_unfreeze_last_blocks} stage(s) of ConvNeXt")
    
    # ViT
    elif backbone_type == "vit_b_16":
        if hasattr(visual_net, 'encoder'):
            layers = visual_net.encoder.layers
            if img_unfreeze_last_blocks > 0:
                for layer in layers[-img_unfreeze_last_blocks:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                print(f"   📸 Unfroze last {img_unfreeze_last_blocks} layer(s) of ViT")
    
    # RegNet
    elif backbone_type == "regnet_y_800mf":
        if hasattr(visual_net, 'trunk_output'):
            stages = [visual_net.stem] + list(visual_net.trunk_output.children())
            if img_unfreeze_last_blocks > 0:
                for stage in stages[-img_unfreeze_last_blocks:]:
                    for p in stage.parameters():
                        p.requires_grad = True
                print(f"   📸 Unfroze last {img_unfreeze_last_blocks} stage(s) of RegNet")

    # ==========================================
    # UNFREEZE AUDIO BACKBONE
    # ==========================================
    audio_type = model.audio_backbone_type
    
    if audio_type == "ast":
        if hasattr(model.audio_net, "encoder") and hasattr(model.audio_net.encoder, "layer"):
            layers = model.audio_net.encoder.layer
            if audio_unfreeze_last_blocks > 0:
                for blk in layers[-audio_unfreeze_last_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                print(f"   🎵 Unfroze last {audio_unfreeze_last_blocks} layers of AST")
    
    elif audio_type == "wavlm":
        if hasattr(model.audio_net, "encoder") and hasattr(model.audio_net.encoder, "layers"):
            layers = model.audio_net.encoder.layers
            if audio_unfreeze_last_blocks > 0:
                for blk in layers[-audio_unfreeze_last_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                print(f"   🎵 Unfroze last {audio_unfreeze_last_blocks} layers of WavLM")
    
    elif audio_type == "whisper":
        if hasattr(model.audio_net, "encoder") and hasattr(model.audio_net.encoder, "layers"):
            layers = model.audio_net.encoder.layers
            if audio_unfreeze_last_blocks > 0:
                for blk in layers[-audio_unfreeze_last_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                print(f"   🎵 Unfroze last {audio_unfreeze_last_blocks} layers of Whisper")
    
    # Unfreeze normalization layers
    norm_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            if "audio_net" in name or "visual_net" in name:
                for p in module.parameters():
                    p.requires_grad = True
                norm_count += 1
    
    if norm_count > 0:
        print(f"   🔓 Unfroze {norm_count} normalization layers")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

def build_optimizer(model, lr_head, lr_backbone, wd_head, wd_backbone):
    head_params = (
        list(model.img_proj.parameters()) +
        list(model.fusion.parameters()) +
        list(model.classifier.parameters())
    )
    
    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and not any(p is pp for pp in head_params)
    ]
    
    return torch.optim.AdamW([
        {"params": head_params, "lr": lr_head, "weight_decay": wd_head},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd_backbone}
    ])