import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import OffroadDataset
from model import get_model
from tqdm import tqdm
import os

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LR = 3e-5
EPOCHS = 80
VAL_SPLIT = 0.2
MODEL_PATH = "models/unet_best.pth"
NUM_CLASSES = 4   # change according to your dataset
# ==========================================

os.makedirs("models", exist_ok=True)

# ================= DATA ===================
dataset = OffroadDataset("data")

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ================= MODEL ==================
model = get_model().to(DEVICE)

# Class weights (IMPORTANT → adjust after checking pixel frequency)
weights = torch.tensor([]).to(DEVICE)
ce_loss = torch.nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)


# ================= LOSSES =================
def dice_loss(pred, target, smooth=1):
    pred = torch.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=NUM_CLASSES).permute(0,3,1,2).float()

    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))

    dice = (2*intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, mask):
    ce = ce_loss(pred, mask)
    dl = dice_loss(pred, mask)
    return ce + dl

# ================= METRIC =================
def iou_score(pred, target, num_classes=NUM_CLASSES):
    pred = torch.argmax(pred, dim=1)

    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(torch.tensor(1.0).to(DEVICE))
        else:
            ious.append(intersection / union)

    return torch.mean(torch.stack(ious))

# ================= TRAIN ==================
best_iou = 0
print("🚀 TRAINING STARTED")

for epoch in range(EPOCHS):

    # -------- TRAIN --------
    model.train()
    train_loss = 0
    train_iou = 0

    for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]"):
        img = img.to(DEVICE)
        mask = mask.to(DEVICE).long()

        pred = model(img)
        loss = combined_loss(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iou += iou_score(pred, mask).item()

    train_loss /= len(train_loader)
    train_iou /= len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0
    val_iou = 0

    with torch.no_grad():
        for img, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]"):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE).long()

            pred = model(img)
            loss = combined_loss(pred, mask)

            val_loss += loss.item()
            val_iou += iou_score(pred, mask).item()

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)

    scheduler.step(val_loss)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   IoU: {val_iou:.4f}")

    # -------- SAVE BEST MODEL --------
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), MODEL_PATH)
        print("💾 Best model saved!")

print(f"\n🏆 Training finished. Best IoU: {best_iou:.4f}")
