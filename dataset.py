"""
src/dataset.py
GastroVision dataset + split creation utilities.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T

from configs.config import (
    IMAGE_ROOT_DIR, DATA_DIR, SPLITS_DIR,
    IMG_SIZE, NUM_CLASSES, RANDOM_SEED, CLASS_MAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Split creation
# ─────────────────────────────────────────────────────────────────────────────

def create_gastrovision_splits(raw_dir=IMAGE_ROOT_DIR, splits_dir=SPLITS_DIR):
    """
    Builds stratified train/val/test splits from the folder structure.
    Handles tiny classes (n=1, n=2, n<10) gracefully.

    Returns: train_df, val_df, test_df, unreliable_classes
    """
    raw_dir    = Path(raw_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for class_folder in sorted(raw_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        if class_name not in CLASS_MAP:
            print(f"  WARNING: '{class_name}' not in CLASS_MAP — skipping")
            continue
        class_id = CLASS_MAP[class_name]
        images   = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg"))
        for img_path in images:
            rows.append({
                "image_path": str(img_path.relative_to(raw_dir)),
                "label":      class_id,
                "class_name": class_name,
            })
        print(f"  [{class_id:2d}] {class_name:<50} {len(images):>4} images")

    df = pd.DataFrame(rows)
    print(f"\nTotal images: {len(df)} | Total classes: {df['label'].nunique()}")

    train_rows, val_rows, test_rows = [], [], []

    for class_id, class_df in df.groupby("label"):
        class_df = class_df.sample(frac=1, random_state=RANDOM_SEED)
        n = len(class_df)

        if n == 1:
            train_rows.append(class_df)
        elif n == 2:
            train_rows.append(class_df.iloc[[0]])
            val_rows.append(class_df.iloc[[1]])
        elif n < 10:
            n_train = max(1, int(0.6 * n))
            n_val   = max(1, int(0.2 * n))
            train_rows.append(class_df.iloc[:n_train])
            val_part  = class_df.iloc[n_train:n_train + n_val]
            test_part = class_df.iloc[n_train + n_val:]
            if len(val_part)  > 0: val_rows.append(val_part)
            if len(test_part) > 0: test_rows.append(test_part)
        else:
            train_part, temp_part = train_test_split(
                class_df, test_size=0.2, random_state=RANDOM_SEED
            )
            val_part, test_part = train_test_split(
                temp_part, test_size=0.5, random_state=RANDOM_SEED
            )
            train_rows.append(train_part)
            val_rows.append(val_part)
            test_rows.append(test_part)

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df   = pd.concat(val_rows,   ignore_index=True)
    test_df  = pd.concat(test_rows,  ignore_index=True)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir   / "val.csv",   index=False)
    test_df.to_csv(splits_dir  / "test.csv",  index=False)

    unreliable = [
        cls for cls in df["label"].unique()
        if len(df[df["label"] == cls]) < 30
    ]

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Rare/unreliable classes (< 30 samples): {sorted(unreliable)}")
    return train_df, val_df, test_df, sorted(unreliable)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class GastroVisionDataset(Dataset):
    """
    Unified dataset for classifier and diffusion training.

    Parameters
    ----------
    csv_path  : path to train/val/test CSV
    split     : "train" | "val" | "test"
    mode      : "classifier" (ImageNet norm) | "diffusion" ([-1,1] norm)
    """

    def __init__(self, csv_path, split="train", mode="classifier"):
        self.split = split
        self.mode  = mode
        df = pd.read_csv(csv_path)

        required = {"image_path", "label"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        self.imagepaths  = df["image_path"].tolist()
        self.labels      = df["label"].astype(int).tolist()
        self.class_names = df["class_name"].tolist() if "class_name" in df.columns else None

        if mode == "diffusion":
            normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        elif mode == "classifier":
            normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        if split == "train" and mode == "classifier":
            self.transform = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                T.ToTensor(),
                normalize,
            ])
        elif split == "train" and mode == "diffusion":
            self.transform = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ToTensor(),
                normalize,
            ])
        else:
            self.transform = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, idx):
        rel_path = self.imagepaths[idx]
        label    = int(self.labels[idx])

        # BUG FIX: original code only checked "synthetic/" prefix but SYNTH_DIR
        # images are stored under DATA_DIR/synthetic/<cls>/synth_XXXXX.png.
        # The rel_path in SYNTH_CSV is relative to DATA_DIR, so resolve from there.
        if str(rel_path).startswith("synthetic/"):
            img_path = DATA_DIR / rel_path
        else:
            img_path = IMAGE_ROOT_DIR / rel_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion-specific dataset (used in SD domain adaptation)
# ─────────────────────────────────────────────────────────────────────────────

class GastroVisionSDDataset(Dataset):
    """Dataset for SD LoRA domain adaptation — returns pixel values + token ids."""

    def __init__(self, csv_path, tokenizer, size=512):
        from configs.config import CLASS_PROMPTS
        self.df        = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.size      = size
        self.prompts   = CLASS_PROMPTS
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        self.label_to_name = (
            dict(zip(self.df["label"].astype(int), self.df["class_name"]))
            if "class_name" in self.df.columns else {}
        )
        print(f"  SD dataset: {len(self.df)} images across {self.df['label'].nunique()} classes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])
        img   = Image.open(IMAGE_ROOT_DIR / row["image_path"]).convert("RGB")
        pixel = self.transform(img)

        prompt = self.prompts.get(
            label,
            "endoscopy photograph of gastrointestinal tissue, "
            "round endoscopic field with dark vignette border, specular highlights"
        )
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {"pixel_values": pixel, "input_ids": tokens, "label": label}


# ─────────────────────────────────────────────────────────────────────────────
# Sampler utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_weighted_sampler(csv_path, num_classes=NUM_CLASSES):
    """WeightedRandomSampler for class-balanced mini-batches."""
    df           = pd.read_csv(csv_path)
    labels       = df["label"].astype(int).tolist()
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights      = 1.0 / class_counts
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def get_class_weights(csv_path, device, num_classes=NUM_CLASSES):
    """Inverse-frequency class weights for use in CrossEntropyLoss / FocalLoss."""
    import torch
    df           = pd.read_csv(csv_path)
    class_counts = np.bincount(df["label"].astype(int), minlength=num_classes).astype(float)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights      = 1.0 / class_counts
    weights      = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float).to(device)
