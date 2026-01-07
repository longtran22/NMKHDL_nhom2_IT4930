import torch
import json
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR  # ← NO RESTART!
from sklearn.metrics import precision_recall_fscore_support

def mixup_data(x_img, x_aud, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x_img.size(0)
    index = torch.randperm(batch_size).to(x_img.device)
    mixed_img = lam * x_img + (1 - lam) * x_img[index]
    mixed_aud = lam * x_aud + (1 - lam) * x_aud[index]
    y_a, y_b = y, y[index]
    return mixed_img, mixed_aud, y_a, y_b, lam

def train(model, train_loader, val_loader, optimizer, criterion, device, 
          epochs, save_path, use_mixup=True, mixup_prob=0.3, mixup_alpha=0.2):
    
    scaler = torch.cuda.amp.GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)  # ← NO RESTART
    
    best_acc = 0
    patience = 5
    patience_counter = 0
    
    history = {
        "train_acc": [], "train_loss": [],
        "val_acc": [], "val_loss": [],
        "val_precision": [], "val_recall": [], "val_f1": [],
        "learning_rate": []
    }
    
    for ep in range(epochs):
        # TRAINING
        model.train()
        losses, corr, tot = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Ep {ep+1}/{epochs}")
        
        for img, aud, lbl in pbar:
            img, aud, lbl = img.to(device), aud.to(device), lbl.to(device)
            original_lbl = lbl.clone()
            apply_mixup = use_mixup and (np.random.random() > (1 - mixup_prob))
            
            if apply_mixup:
                img, aud, lbl_a, lbl_b, lam = mixup_data(img, aud, lbl, alpha=mixup_alpha)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    out = model(img, aud)
                    loss = lam * criterion(out, lbl_a) + (1 - lam) * criterion(out, lbl_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                _, pred = out.max(1)
                corr += (pred == lbl_a).sum().item()
            else:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    out = model(img, aud)
                    loss = criterion(out, original_lbl)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                _, pred = out.max(1)
                corr += (pred == original_lbl).sum().item()
            
            losses += loss.item()
            tot += original_lbl.size(0)
            pbar.set_postfix({'loss': f'{losses/len(train_loader):.4f}', 'acc': f'{corr/tot:.4f}'})
        
        train_acc = corr / tot
        train_loss = losses / len(train_loader)
        
        # VALIDATION
        model.eval()
        v_losses = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for img, aud, lbl in val_loader:
                img, aud, lbl = img.to(device), aud.to(device), lbl.to(device)
                out = model(img, aud)
                loss = criterion(out, lbl)
                v_losses += loss.item()
                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())
        
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        val_loss = v_losses / len(val_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["val_precision"].append(float(precision))
        history["val_recall"].append(float(recall))
        history["val_f1"].append(float(f1))
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        
        logger.info(
            f"Ep {ep+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f}"
        )
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': ep + 1,
                'best_acc': best_acc,
                'history': history
            }, save_path)
            logger.info(f"✅ Saved Best Model (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"⚠️ Early stopping at epoch {ep+1}")
                break
        
        scheduler.step()
        logger.info(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("✅ Saved training history")
    
    return history