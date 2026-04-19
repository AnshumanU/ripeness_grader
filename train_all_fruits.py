"""
Trains one MobileNetV3 ripeness model per fruit and exports each as ONNX.
Output: models/banana_model.onnx, models/apple_model.onnx, etc.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ── CONFIG ───────────────────────────────────────────────────────────────────
FRUITS360 = Path("../fruits360/fruits-360_100x100/fruits-360")
TRAIN_DIR = FRUITS360 / "Training"
TEST_DIR = FRUITS360 / "Test"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 6
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Auto-detect dataset path if default doesn't exist
if not TRAIN_DIR.exists():
    for p in Path("../fruits360").rglob("Training"):
        TRAIN_DIR = p
        TEST_DIR = p.parent / "Test"
        print(f"Found dataset at: {p.parent}")
        break

# ── RIPENESS LABELS ──────────────────────────────────────────────────────────
RIPENESS_LABELS = ["unripe", "ripe", "overripe"]

RIPENESS_RULES = [
    (["black", "maroon", "dark", "dried"], "overripe"),
    (["green", "unripe", "husk", "seed", "pod", "peeled"], "unripe"),
]

def infer_ripeness(folder_name: str) -> int:
    name = re.sub(r"\s+\d+$", "", folder_name.lower()).strip()

    for keywords, label in RIPENESS_RULES:
        if any(kw in name for kw in keywords):
            return RIPENESS_LABELS.index(label)

    return RIPENESS_LABELS.index("ripe")

# ── FRUIT NAME EXTRACTOR ─────────────────────────────────────────────────────
def get_base_fruit(folder_name: str) -> str:
    name = re.sub(r"\s+\d+$", "", folder_name).strip()
    return name.split()[0].lower()

# ── BUILD PER-FRUIT DATA ─────────────────────────────────────────────────────
def collect_fruit_data(split_dir: Path):
    data = defaultdict(list)

    for cls_folder in sorted(split_dir.iterdir()):
        if not cls_folder.is_dir():
            continue

        fruit = get_base_fruit(cls_folder.name)
        ripe = infer_ripeness(cls_folder.name)

        for img_path in cls_folder.glob("*.jpg"):
            data[fruit].append((img_path, ripe))

    return data

# ── DATASET ──────────────────────────────────────────────────────────────────
class RipenessDataset(Dataset):
    def __init__(self, entries, transform=None):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        path, label = self.entries[i]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── MODEL ────────────────────────────────────────────────────────────────────
def build_model(num_classes=3):
    base = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    base.classifier[3] = nn.Linear(
        base.classifier[3].in_features,
        num_classes
    )
    return base

# ── TRAIN ONE FRUIT ──────────────────────────────────────────────────────────
def train_fruit(fruit, train_entries, test_entries):
    print(f"\n{'=' * 50}")
    print(f"Training: {fruit.upper()} ({len(train_entries)} train, {len(test_entries)} test)")

    classes_present = set(e[1] for e in train_entries)

    if len(classes_present) < 2:
        print(f"Skipping {fruit} — only 1 ripeness class in data")
        return

    train_ds = RipenessDataset(train_entries, train_tf)
    test_ds = RipenessDataset(test_entries, test_tf)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = 0
        total = 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            out = model(imgs)
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                out = model(imgs)

                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        improved = "✓" if val_acc > best_acc else ""

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train {train_acc:.3f} | "
            f"val {val_acc:.3f} {improved}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                k: v.clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    onnx_path = OUTPUT_DIR / f"{fruit}_model.onnx"

    torch.onnx.export(
    model,
    dummy,
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    },
    opset_version=12,
)

    print(f"Saved → {onnx_path} (best val acc: {best_acc:.3f})")

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("Scanning dataset...")

    train_data = collect_fruit_data(TRAIN_DIR)
    test_data = collect_fruit_data(TEST_DIR)

    all_fruits = sorted(train_data.keys())
    print(f"Found {len(all_fruits)} fruits")

    meta = {
        "ripeness_labels": RIPENESS_LABELS,
        "fruits": all_fruits
    }

    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    for fruit in all_fruits:
        onnx_path = OUTPUT_DIR / f"{fruit}_model.onnx"

        if onnx_path.exists():
            print(f"Skipping {fruit} — already trained")
            continue

        train_fruit(
            fruit,
            train_data[fruit],
            test_data.get(fruit, [])
        )

    print(f"\nAll done! Models saved to ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()