#!/usr/bin/env python3
"""
gastrovision_pipeline.py
========================
Fully self-contained GastroVision rare-class augmentation pipeline.
Designed as a single-file Kubernetes Job entrypoint (Option B).

No imports from src/ or configs/ — everything is in this one file.

Usage
-----
  python gastrovision_pipeline.py \
      --data_dir   /data/gastrovision/data \
      --output_dir /data/gastrovision

Smoke test (completes in ~30 min):
  python gastrovision_pipeline.py \
      --data_dir            /data/gastrovision/data \
      --output_dir          /data/gastrovision \
      --domain_adapt_steps  500  \
      --samples_per_class   10   \
      --freeze_epochs       2    \
      --fine_tune_epochs    3    \
      --skip_domain_adapt        \
      --model_name          swin

Skip flags for resuming partial runs:
  --skip_domain_adapt   skip SD fine-tuning (adapter already saved)
  --skip_generation     skip image generation (synth CSV already exists)
  --skip_training       skip all classifier training
  --evaluate_only       only run evaluation on existing checkpoints
"""

import os
import sys
import gc
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support, roc_curve, auc,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import inception_v3
import timm
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import seaborn as sns

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed — hyperparameter tuning disabled.")

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False
    print("Grad-CAM not available — XAI disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    print("SHAP not available — SHAP analysis disabled.")

from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler as get_diffusers_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

warnings.filterwarnings("ignore")


# ==============================================================================
# SECTION 1 — Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GastroVision DDPM Augmentation Pipeline")

    # Paths
    p.add_argument("--data_dir",       default="/data",
                   help="Root data directory (PVC mount point)")
    p.add_argument("--output_dir",     default="/output",
                   help="Output directory for checkpoints, results, logs")
    p.add_argument("--image_root",     default="gastrovision_raw/Gastrovision",
                   help="Subfolder under data_dir containing class folders")
    p.add_argument("--train_csv",      default="splits/train.csv")
    p.add_argument("--val_csv",        default="splits/val.csv")
    p.add_argument("--test_csv",       default="splits/test.csv")
    p.add_argument("--aug_train_csv",  default="splits/train_aug.csv")
    p.add_argument("--synth_csv",      default="synthetic/synthetic_train.csv")
    p.add_argument("--synth_dir",      default="synthetic")

    # Classifier hyperparameters
    p.add_argument("--img_size",         type=int,   default=224)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--freeze_epochs",    type=int,   default=16,
                   help="Phase 1: frozen backbone, head-only training")
    p.add_argument("--fine_tune_epochs", type=int,   default=24,
                   help="Phase 2: full model fine-tuning with cosine LR")
    p.add_argument("--gamma",            type=float, default=2.0,
                   help="Focal loss focusing parameter")
    p.add_argument("--freeze_lr_mult",   type=float, default=10.0,
                   help="LR multiplier for frozen-phase head training")

    # Diffusion / SD config
    p.add_argument("--sd_model_id",         default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora_rank",           type=int,   default=32,
                   help="LoRA rank. 32 recommended for endoscopy domain.")
    p.add_argument("--lora_alpha",          type=int,   default=64,
                   help="LoRA alpha. Keep at 2x lora_rank.")
    p.add_argument("--lora_dropout",        type=float, default=0.1)
    p.add_argument("--domain_adapt_steps",  type=int,   default=15000,
                   help="SD LoRA fine-tuning steps. Target loss < 0.05.")
    p.add_argument("--sd_batch_size",       type=int,   default=4)
    p.add_argument("--sd_grad_accum",       type=int,   default=4,
                   help="Gradient accumulation. Effective batch = sd_batch_size x sd_grad_accum.")
    p.add_argument("--sd_lr",              type=float, default=1e-4)
    p.add_argument("--ema_decay",           type=float, default=0.9999)
    p.add_argument("--ema_warmup_steps",    type=int,   default=100)
    p.add_argument("--samples_per_class",   type=int,   default=500,
                   help="Synthetic images to generate per rare class.")
    p.add_argument("--gen_steps",           type=int,   default=40,
                   help="Denoising steps per generated image.")
    p.add_argument("--guidance_scale",      type=float, default=7.0)
    p.add_argument("--gen_batch_size",      type=int,   default=4,
                   help="Images per pipeline call during generation.")

    # Evaluation
    p.add_argument("--kfold_splits",         type=int, default=5)
    p.add_argument("--min_reliable_samples", type=int, default=10)
    p.add_argument("--seed",                 type=int, default=42)

    # Execution modes
    p.add_argument("--skip_domain_adapt", action="store_true")
    p.add_argument("--skip_generation",   action="store_true")
    p.add_argument("--skip_training",     action="store_true")
    p.add_argument("--evaluate_only",     action="store_true")
    p.add_argument("--models", nargs="+",
                   default=["efficientnetv2_rw_s", "swin", "mobile",
                            "hybrid_cnn_transformer", "hybrid_cnn_transformer_v2"],
                   help="Which classifier models to train and evaluate.")

    return p.parse_args()


# ==============================================================================
# SECTION 2 — Global config derived from args
# ==============================================================================

args = parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

DATA_DIR       = Path(args.data_dir)
OUTPUT_DIR     = Path(args.output_dir)
IMAGE_ROOT_DIR = DATA_DIR / args.image_root
SPLITS_DIR     = OUTPUT_DIR / "splits"
SYNTH_DIR      = OUTPUT_DIR / args.synth_dir
CKPT_DIR       = OUTPUT_DIR / "checkpoints"
RESULTS_DIR    = OUTPUT_DIR / "results"
LOGS_DIR       = OUTPUT_DIR / "logs"

for d in [SPLITS_DIR, SYNTH_DIR, CKPT_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Set after split creation
NUM_CLASSES  = None
RARE_CLASSES = []

# Per-model hyperparameters — updated by Optuna if tuning is run
HPARAMS = {
    "efficientnetv2_rw_s": {
        "lr": args.lr, "freeze_epochs": args.freeze_epochs,
        "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
        "gamma": args.gamma, "freeze_lr_mult": args.freeze_lr_mult, "weight_decay": args.weight_decay,
    },
    "swin": {
        "lr": args.lr, "freeze_epochs": args.freeze_epochs,
        "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
        "gamma": args.gamma, "freeze_lr_mult": args.freeze_lr_mult, "weight_decay": args.weight_decay,
    },
    "mobile": {
        "lr": args.lr * 1.5, "freeze_epochs": args.freeze_epochs,
        "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
        "gamma": args.gamma, "freeze_lr_mult": args.freeze_lr_mult, "weight_decay": args.weight_decay,
    },
    "hybrid_cnn_transformer": {
        "lr": args.lr * 0.67, "freeze_epochs": max(1, args.freeze_epochs - 6),
        "fine_tune_epochs": args.fine_tune_epochs + 6,
        # V1 dual-branch needs more VRAM — cap at 8 on small GPUs
        "batch_size": min(args.batch_size, 8),
        "gamma": args.gamma, "freeze_lr_mult": 5.0, "weight_decay": args.weight_decay,
    },
    "hybrid_cnn_transformer_v2": {
        "lr": args.lr * 0.67, "freeze_epochs": max(1, args.freeze_epochs - 8),
        "fine_tune_epochs": args.fine_tune_epochs,
        # V2 sequential is lighter — cap at 16 on small GPUs
        "batch_size": min(args.batch_size, 16),
        "gamma": args.gamma, "freeze_lr_mult": 5.0, "weight_decay": args.weight_decay,
    },
}

# On 11GB GPUs (RTX 2080 Ti) further cap all batch sizes
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1e9 < 20:
    for k in HPARAMS:
        HPARAMS[k]["batch_size"] = min(HPARAMS[k]["batch_size"], 16)
    # Swin base is large — 16 may OOM on 11GB, use 8
    HPARAMS["swin"]["batch_size"] = 8
    HPARAMS["hybrid_cnn_transformer"]["batch_size"] = 4    # V1 too large for 11GB at any useful batch
    HPARAMS["hybrid_cnn_transformer_v2"]["batch_size"] = 8
    print("  ⚠ RTX 2080 Ti detected — batch sizes capped for 11GB VRAM")


# ==============================================================================
# SECTION 3 — Prompt constants
# ==============================================================================

# Prepended to every class-specific prompt. Front-loading the domain context
# gives it higher weight in CLIP's causal attention before class tokens.
DOMAIN_PREFIX = (
    "real gastroscopy photograph, circular endoscopic view with dark vignette "
    "border, specular highlights from endoscope light source, pink mucosal "
    "background, high resolution clinical endoscopy image: "
)

NEGATIVE_PROMPT = (
    "diagram, illustration, drawing, cartoon, animation, painting, sketch, "
    "render, 3d model, cgi, "
    "blue background, white background, black background, grey background, "
    "anatomical model, textbook figure, medical poster, "
    "microscopy, histology, pathology slide, x-ray, ct scan, mri, ultrasound, "
    "blurry, out of focus, low quality, low resolution, noisy, grainy, "
    "overexposed, underexposed, dark, dim, "
    "watermark, text, caption, label, logo, border, frame, "
    "photograph of person, face, hands, natural scene, landscape, food, "
    "animal, plant, outdoor, indoor room"
)

CLASS_MAP = {
    "Accessory tools": 0, "Angiectasia": 1, "Barretts esophagus": 2,
    "Blood in lumen": 3, "Cecum": 4, "Colon diverticula": 5,
    "Colon polyps": 6, "Colorectal cancer": 7, "Duodenal bulb": 8,
    "Dyed-lifted-polyps": 9, "Dyed-resection-margins": 10, "Erythema": 11,
    "Esophageal varices": 12, "Esophagitis": 13, "Gastric polyps": 14,
    "Gastroesophageal_junction_normal z-line": 15, "Ileocecal valve": 16,
    "Mucosal inflammation large bowel": 17, "Normal esophagus": 18,
    "Normal mucosa and vascular pattern in the large bowel": 19,
    "Normal stomach": 20, "Pylorus": 21, "Resected polyps": 22,
    "Resection margins": 23, "Retroflex rectum": 24,
    "Small bowel_terminal ileum": 25, "Ulcer": 26,
}

CLASS_PROMPTS = {
    0:  "gastroscopy photograph showing accessory tools, metal endoscopic instruments in circular endoscopic field, pink mucosa background, dark vignette border, specular highlights",
    1:  "capsule endoscopy photograph of angiectasia, tortuous dilated red vessels on pink salmon mucosal surface, round endoscopic field with dark vignette, specular highlights, fish-eye lens distortion",
    2:  "upper GI endoscopy photograph of Barrett's esophagus, salmon-colored irregular mucosal patches in lower esophagus, pink esophageal wall, round endoscopic field with dark vignette border, specular highlights",
    3:  "gastroscopy photograph showing blood in lumen, dark red blood pooling in gastric lumen, pink mucosal walls, round endoscopic field with dark vignette border, specular highlights",
    4:  "colonoscopy photograph of cecum, pale pink cecal mucosa with appendiceal orifice visible, haustral folds, round endoscopic field with dark vignette, specular highlights from colonoscope light",
    5:  "colonoscopy photograph of colon diverticula, multiple dark circular openings in pink colonic mucosa, small pouches in bowel wall, round endoscopic field with dark vignette, specular highlights",
    6:  "colonoscopy photograph of colon polyp, pedunculated or sessile polyp protruding from pink colonic mucosa, round endoscopic field with dark vignette border, specular highlights",
    7:  "colonoscopy photograph of colorectal cancer, irregular friable mass with ulceration in colon, pink surrounding mucosa, round endoscopic field with dark vignette border, specular highlights",
    8:  "upper GI endoscopy photograph of duodenal bulb, pale pink smooth duodenal mucosa with circular folds, round endoscopic field with dark vignette, specular highlights from endoscope light",
    9:  "colonoscopy photograph of dyed lifted polyp, blue-dyed raised polyp on colonic mucosa after submucosal injection, round endoscopic field with dark vignette border, specular highlights",
    10: "colonoscopy photograph of dyed resection margins, blue-dyed mucosal edges after polypectomy, round endoscopic field with dark vignette, specular highlights from colonoscope light",
    11: "gastroscopy photograph of gastric erythema, diffuse reddish pink discoloration of stomach mucosa, hyperemic patches, round endoscopic field with dark vignette border, specular highlights",
    12: "upper GI endoscopy photograph of esophageal varices, prominent bluish purple longitudinal bulging veins along esophageal wall, pink esophageal mucosa, round endoscopic field with dark vignette, specular highlights",
    13: "upper GI endoscopy photograph of esophagitis, erythematous inflamed esophageal mucosa with linear erosions, round endoscopic field with dark vignette border, specular highlights",
    14: "gastroscopy photograph of gastric polyp, smooth rounded polyp protruding from pink gastric mucosa, round endoscopic field with dark vignette border, specular highlights",
    15: "upper GI endoscopy photograph of gastroesophageal junction normal z-line, clear squamocolumnar junction between esophagus and stomach, round endoscopic field with dark vignette, specular highlights",
    16: "colonoscopy photograph of ileocecal valve, lips of ileocecal valve in pink cecal mucosa, round endoscopic field with dark vignette border, specular highlights from colonoscope",
    17: "colonoscopy photograph of mucosal inflammation large bowel, granular friable reddish colonic mucosa with loss of vascular pattern, round endoscopic field with dark vignette, specular highlights",
    18: "upper GI endoscopy photograph of normal esophagus, smooth pale pink esophageal mucosa with visible longitudinal folds, round endoscopic field with dark vignette border, specular highlights",
    19: "colonoscopy photograph of normal colonic mucosa, smooth pink mucosa with clear vascular pattern, haustral folds, round endoscopic field with dark vignette border, specular highlights",
    20: "gastroscopy photograph of normal stomach, smooth pink gastric mucosa with rugal folds, pool of clear gastric fluid, round endoscopic field with dark vignette border, specular highlights",
    21: "gastroscopy photograph of pylorus, circular pyloric orifice in pink gastric mucosa, antral folds, round endoscopic field with dark vignette border, specular highlights",
    22: "colonoscopy photograph of resected polyp, post-polypectomy site with cauterized flat mucosal defect, pink surrounding mucosa, round endoscopic field with dark vignette, specular highlights",
    23: "endoscopy photograph of resection margins, post-surgical cauterized tissue edges with whitish fibrinous border, pink surrounding mucosa, round endoscopic field with dark vignette",
    24: "colonoscopy photograph of retroflex rectum, retroflexed view of rectal mucosa and anorectal junction, pink smooth mucosa, round endoscopic field with dark vignette border, specular highlights",
    25: "colonoscopy photograph of small bowel terminal ileum, pale pink villous mucosa with fine surface texture, round endoscopic field with dark vignette border, specular highlights",
    26: "gastroscopy photograph of gastric ulcer, well-defined mucosal crater with white fibrinous base surrounded by erythematous gastric mucosa, round endoscopic field with dark vignette, specular highlights",
}

# FID normalization — applied identically to both real and synthetic images.
# SD pipeline outputs PIL uint8 [0,255] PNGs; there is no [-1,1] leakage.
# T.ToTensor() maps to [0,1]; T.Normalize() applies ImageNet stats → ~[-2.5, 2.5].
FID_TRANSFORM = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==============================================================================
# SECTION 4 — Dataset
# ==============================================================================

def create_splits():
    """Stratified train/val/test splits from the raw folder structure."""
    raw_dir    = IMAGE_ROOT_DIR
    splits_dir = SPLITS_DIR
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
    print(f"\nTotal: {len(df)} images, {df['label'].nunique()} classes")

    train_rows, val_rows, test_rows = [], [], []
    for class_id, class_df in df.groupby("label"):
        class_df = class_df.sample(frac=1, random_state=args.seed)
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
            v = class_df.iloc[n_train:n_train + n_val]
            t = class_df.iloc[n_train + n_val:]
            if len(v) > 0: val_rows.append(v)
            if len(t) > 0: test_rows.append(t)
        else:
            tr, tmp = train_test_split(class_df, test_size=0.2, random_state=args.seed)
            v, t    = train_test_split(tmp,      test_size=0.5, random_state=args.seed)
            train_rows.append(tr); val_rows.append(v); test_rows.append(t)

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df   = pd.concat(val_rows,   ignore_index=True)
    test_df  = pd.concat(test_rows,  ignore_index=True)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir   / "val.csv",   index=False)
    test_df.to_csv(splits_dir  / "test.csv",  index=False)

    unreliable = sorted([
        cls for cls in df["label"].unique()
        if len(df[df["label"] == cls]) < 30
    ])
    print(f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")
    print(f"Rare classes (< 30 samples): {unreliable}")
    return train_df, val_df, test_df, unreliable


class GastroVisionDataset(Dataset):
    def __init__(self, csv_path, split="train", mode="classifier"):
        self.split = split
        df = pd.read_csv(csv_path)
        if {"image_path", "label"} - set(df.columns):
            raise ValueError("CSV missing image_path or label columns")
        self.imagepaths  = df["image_path"].tolist()
        self.labels      = df["label"].astype(int).tolist()
        self.class_names = df["class_name"].tolist() if "class_name" in df.columns else None

        if mode == "diffusion":
            norm = T.Normalize([0.5]*3, [0.5]*3)
        else:
            norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if split == "train" and mode == "classifier":
            self.transform = T.Compose([
                T.Resize((args.img_size, args.img_size)),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                T.ToTensor(), norm,
            ])
        elif split == "train" and mode == "diffusion":
            self.transform = T.Compose([
                T.Resize((args.img_size, args.img_size)),
                T.RandomHorizontalFlip(), T.RandomRotation(10),
                T.ToTensor(), norm,
            ])
        else:
            self.transform = T.Compose([
                T.Resize((args.img_size, args.img_size)),
                T.ToTensor(), norm,
            ])

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, idx):
        rel  = self.imagepaths[idx]
        path = (DATA_DIR / rel) if str(rel).startswith("synthetic/") else (IMAGE_ROOT_DIR / rel)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return self.transform(Image.open(path).convert("RGB")), int(self.labels[idx])


class GastroVisionSDDataset(Dataset):
    """Returns (pixel_values, input_ids, label) for SD LoRA training."""
    def __init__(self, csv_path, tokenizer, size=512):
        self.df        = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.transform = T.Compose([
            T.Resize((size, size)), T.RandomHorizontalFlip(),
            T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3),
        ])
        self.label_to_name = (
            dict(zip(self.df["label"].astype(int), self.df["class_name"]))
            if "class_name" in self.df.columns else {}
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        label  = int(row["label"])
        pixel  = self.transform(Image.open(IMAGE_ROOT_DIR / row["image_path"]).convert("RGB"))
        prompt = CLASS_PROMPTS.get(label,
            "endoscopy photograph of gastrointestinal tissue, "
            "round endoscopic field with dark vignette border, specular highlights")
        tokens = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.squeeze(0)
        return {"pixel_values": pixel, "input_ids": tokens, "label": label}


def get_weighted_sampler(csv_path):
    df     = pd.read_csv(csv_path)
    labels = df["label"].astype(int).tolist()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts = np.where(counts == 0, 1, counts)
    w      = 1.0 / counts
    sw     = [w[l] for l in labels]
    return WeightedRandomSampler(sw, len(sw), replacement=True)


# ==============================================================================
# SECTION 5 — Loss
# ==============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.reduction = reduction

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss


# ==============================================================================
# SECTION 6 — Model definitions
# ==============================================================================

def get_effnetv2_s(num_classes):
    return timm.create_model("efficientnetv2_rw_s", pretrained=True, num_classes=num_classes)

def get_swin_transformer(num_classes):
    return timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes)

def get_mobilenetv3(num_classes):
    return timm.create_model("tf_mobilenetv3_large_minimal_100", pretrained=True, num_classes=num_classes)


class CrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim, tfm_dim, num_heads=8, dropout=0.1):
        super().__init__()
        d = min(cnn_dim, tfm_dim)
        self.cp = nn.Linear(cnn_dim, d); self.tp = nn.Linear(tfm_dim, d)
        self.ca1 = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.ca2 = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.n1  = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.out_dim = d * 2

    def forward(self, cf, tf):
        cq = self.cp(cf).unsqueeze(1); tq = self.tp(tf).unsqueeze(1)
        a1, _ = self.ca1(cq, tq, tq); a1 = self.n1(a1.squeeze(1) + cq.squeeze(1))
        a2, _ = self.ca2(tq, cq, cq); a2 = self.n2(a2.squeeze(1) + tq.squeeze(1))
        return torch.cat([a1, a2], dim=-1)


class HybridCNNTransformer(nn.Module):
    """Dual-branch: ConvNeXt-Small + Swin-Tiny with cross-attention fusion."""
    def __init__(self, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.cnn = timm.create_model("convnext_small",              pretrained=pretrained, num_classes=0)
        self.tfm = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0)
        self.fusion = CrossAttentionFusion(self.cnn.num_features, self.tfm.num_features)
        fd = self.fusion.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fd), nn.Dropout(dropout),
            nn.Linear(fd, 512), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.fusion(self.cnn(x), self.tfm(x)))

    def freeze_backbones(self):
        for p in self.cnn.parameters(): p.requires_grad = False
        for p in self.tfm.parameters(): p.requires_grad = False
        for p in self.fusion.parameters(): p.requires_grad = True
        for p in self.head.parameters(): p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True


class HybridCNNTransformerV2(nn.Module):
    """
    Sequential: EfficientNetV2-S feature map → Transformer encoder.
    Lighter than V1; good for ablation. Fixes from original snippet:
      - n_tokens derived from dummy pass (not img_size // patch_size)
      - separate cls_pos embedding
      - norm_first=True (Pre-LN)
    """
    def __init__(self, num_classes, cnn_name="efficientnetv2_rw_s",
                 transformer_dim=512, depth=4, heads=8, mlp_dim=1024,
                 dropout=0.1, img_size=224):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=True, features_only=True)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            last  = self.cnn(dummy)[-1]
            cout  = last.shape[1]
            self.n_tokens = last.shape[2] * last.shape[3]
        self.cnn_proj  = nn.Conv2d(cout, transformer_dim, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        self.cls_pos   = nn.Parameter(torch.randn(1, 1, transformer_dim))
        self.patch_pos = nn.Parameter(torch.randn(1, self.n_tokens, transformer_dim))
        enc = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(transformer_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(transformer_dim, num_classes))

    def forward(self, x):
        B       = x.shape[0]
        proj    = self.cnn_proj(self.cnn(x)[-1])
        patches = proj.flatten(2).transpose(1, 2) + self.patch_pos
        cls     = self.cls_token.expand(B, -1, -1) + self.cls_pos
        tokens  = self.transformer(torch.cat([cls, patches], dim=1))
        return self.head(self.norm(tokens[:, 0]))

    def freeze_backbones(self):
        for p in self.cnn.parameters():      p.requires_grad = False
        for p in self.cnn_proj.parameters(): p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True


MODEL_REGISTRY = {
    "efficientnetv2_rw_s":       get_effnetv2_s,
    "swin":                      get_swin_transformer,
    "mobile":                    get_mobilenetv3,
    "hybrid_cnn_transformer":    lambda n: HybridCNNTransformer(n),
    "hybrid_cnn_transformer_v2": lambda n: HybridCNNTransformerV2(n),
}

def get_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](NUM_CLASSES).to(DEVICE)

def load_checkpoint(model_name, augmented=False):
    suffix = "_aug" if augmented else ""
    path   = CKPT_DIR / f"sota_{model_name}{suffix}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = get_model(model_name)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"Loaded {model_name} from {path}")
    return model


# ==============================================================================
# SECTION 7 — Training engine
# ==============================================================================

def _freeze(model, model_name):
    if "hybrid" in model_name:
        model.freeze_backbones()
    else:
        for p in model.parameters(): p.requires_grad = False
        head = getattr(model, "head", None) or getattr(model, "classifier", None)
        if head is None: raise AttributeError(f"No head on {model_name}")
        for p in head.parameters(): p.requires_grad = True

def _unfreeze(model, model_name):
    if "hybrid" in model_name: model.unfreeze_all()
    else:
        for p in model.parameters(): p.requires_grad = True

def _eval_acc(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            with autocast():
                preds = model(xb.to(DEVICE)).argmax(1)
            yp.append(preds.cpu().numpy()); yt.append(yb.numpy())
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return float((yt == yp).mean()), yt, yp


def train_classifier(model_name, train_csv, val_csv, augmented=False):
    """
    Two-phase training:
      Phase 1 — frozen backbone, head only  (freeze_epochs)
      Phase 2 — full fine-tune, cosine LR   (fine_tune_epochs)
    Saves checkpoint to CKPT_DIR/sota_{model_name}[_aug].pt.
    """
    cfg      = HPARAMS[model_name]
    crit     = FocalLoss(gamma=cfg["gamma"])
    scaler   = GradScaler()
    model    = get_model(model_name)
    history  = {"train_loss": [], "val_acc": [], "phase": []}
    ckpt     = CKPT_DIR / f"sota_{model_name}{'_aug' if augmented else ''}.pt"

    train_ds = GastroVisionDataset(train_csv, "train", "classifier")
    val_ds   = GastroVisionDataset(val_csv,   "val",   "classifier")
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    vl = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    _freeze(model, model_name)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"] * cfg.get("freeze_lr_mult", 10.0)
    )
    print(f"\n{'='*60}\n[{model_name}] Phase 1: frozen ({cfg['freeze_epochs']} epochs)\n{'='*60}")

    for ep in range(cfg["freeze_epochs"]):
        model.train(); rl = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast(): loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(opt); scaler.update()
            rl += loss.item()
        acc, _, _ = _eval_acc(model, vl)
        history["train_loss"].append(rl / len(tl))
        history["val_acc"].append(acc); history["phase"].append("freeze")
        print(f"  Ep {ep+1:2d}/{cfg['freeze_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    _unfreeze(model, model_name)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["fine_tune_epochs"])
    best_acc = 0.0
    print(f"\n{'='*60}\n[{model_name}] Phase 2: fine-tune ({cfg['fine_tune_epochs']} epochs)\n{'='*60}")

    for ep in range(cfg["fine_tune_epochs"]):
        model.train(); rl = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast(): loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            rl += loss.item()
        sch.step()
        acc, _, _ = _eval_acc(model, vl)
        history["train_loss"].append(rl / len(tl))
        history["val_acc"].append(acc); history["phase"].append("finetune")
        print(f"  Ep {ep+1:2d}/{cfg['fine_tune_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt)
            print(f"  ✅ Saved (val_acc={best_acc:.4f})")

    print(f"\n  ★ {model_name} best val_acc: {best_acc:.4f}")
    return history


# ==============================================================================
# SECTION 8 — EMA + SNR helpers
# ==============================================================================

class EMAModel:
    """Exponential Moving Average over trainable UNet params, stored on CPU."""
    def __init__(self, model, decay=0.9999, update_after_step=100):
        self.decay = decay; self.update_after_step = update_after_step; self.step_count = 0
        self.shadow = {n: p.detach().cpu().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def step(self, model):
        self.step_count += 1
        decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        if self.step_count < self.update_after_step:
            for n, p in model.named_parameters():
                if n in self.shadow and p.requires_grad:
                    self.shadow[n] = p.detach().cpu().clone()
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow and p.requires_grad:
                    s = self.shadow[n].to(p.device)
                    s.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
                    self.shadow[n] = s.cpu()

    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow and p.requires_grad:
                p.data.copy_(self.shadow[n].to(p.device))

    def restore(self, model, orig):
        for n, p in model.named_parameters():
            if n in orig and p.requires_grad:
                p.data.copy_(orig[n].to(p.device))

    def state_dict(self):
        return {"shadow": self.shadow, "step_count": self.step_count, "decay": self.decay}

    def load_state_dict(self, s):
        self.shadow = s["shadow"]; self.step_count = s["step_count"]; self.decay = s.get("decay", self.decay)

    def save_adapter(self, model, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        orig = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        try:
            self.copy_to(model); model.save_pretrained(path)
            print(f"  EMA adapter saved → {path}")
        finally:
            self.restore(model, orig)


def _snr_weights(scheduler, t, device, gamma=5.0):
    ac  = scheduler.alphas_cumprod.to(device)
    snr = (ac[t] ** 0.5 / ((1 - ac[t]) ** 0.5 + 1e-8)) ** 2
    return (torch.clamp(snr, max=gamma) / (snr + 1e-8)).detach()


# ==============================================================================
# SECTION 9 — Domain adaptation
# ==============================================================================

def domain_adapt_sd():
    """Fine-tune SD LoRA on all GastroVision images with EMA + Min-SNR weighting."""
    train_csv = SPLITS_DIR / args.train_csv

    print("Loading SD components...")
    vram_gb      = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    offload_cpu  = vram_gb < 20    # covers 2080 Ti (11GB) and anything below A100
    print(f"  GPU VRAM: {vram_gb:.0f}GB  —  CPU offload: {offload_cpu}")

    tokenizer    = CLIPTokenizer.from_pretrained(args.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model_id, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(args.sd_model_id, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(args.sd_model_id, subfolder="unet")
    noise_sched  = DDPMScheduler.from_pretrained(args.sd_model_id, subfolder="scheduler")

    # On 40GB: keep text encoder and VAE on CPU, move to GPU only for encoding,
    # then immediately move back. UNet stays on GPU throughout.
    # On 80GB: everything stays on GPU.
    if offload_cpu:
        unet         = unet.to(DEVICE)
        text_encoder = text_encoder.cpu()
        vae          = vae.cpu()
    else:
        text_encoder = text_encoder.to(DEVICE)
        vae          = vae.to(DEVICE)
        unet         = unet.to(DEVICE)

    vae.requires_grad_(False); text_encoder.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    ema     = EMAModel(unet, decay=args.ema_decay, update_after_step=args.ema_warmup_steps)
    dataset = GastroVisionSDDataset(train_csv, tokenizer)
    loader  = DataLoader(dataset, batch_size=args.sd_batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)

    opt    = torch.optim.AdamW(unet.parameters(), lr=args.sd_lr, weight_decay=1e-4)
    lrsched = get_diffusers_scheduler("cosine", optimizer=opt,
                                       num_warmup_steps=500,
                                       num_training_steps=args.domain_adapt_steps)
    scaler  = GradScaler()

    resume_path = CKPT_DIR / "resume_sd_lora.pt"
    step = 0; losses = []
    if resume_path.exists():
        ck = torch.load(resume_path, map_location=DEVICE)
        unet.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"])
        lrsched.load_state_dict(ck["scheduler"]); step = ck["global_step"]
        losses = ck.get("losses", [])
        if "ema" in ck: ema.load_state_dict(ck["ema"])
        print(f"Resumed at step {step}/{args.domain_adapt_steps}")

    print(f"\nDomain adaptation: {len(dataset)} images, {step}→{args.domain_adapt_steps} steps")
    unet.train(); opt.zero_grad(); it = iter(loader); rl = 0.0

    while step < args.domain_adapt_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)

        pv = batch["pixel_values"].to(DEVICE)
        ii = batch["input_ids"].to(DEVICE)

        with torch.no_grad():
            # Move to GPU for encoding, then back to CPU to free VRAM
            vae_device = next(vae.parameters()).device
            if vae_device != DEVICE:
                vae.to(DEVICE)
            lat = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor
            if offload_cpu:
                vae.cpu(); torch.cuda.empty_cache()

        noise = torch.randn_like(lat)
        t     = torch.randint(0, noise_sched.config.num_train_timesteps, (lat.shape[0],), device=DEVICE).long()
        w     = _snr_weights(noise_sched, t, DEVICE)
        nl    = noise_sched.add_noise(lat, noise, t)

        with torch.no_grad():
            te_device = next(text_encoder.parameters()).device
            if te_device != DEVICE:
                text_encoder.to(DEVICE)
            hs = text_encoder(ii)[0]
            if offload_cpu:
                text_encoder.cpu(); torch.cuda.empty_cache()

        with autocast():
            pred = unet(nl, t, hs).sample
            lps  = F.mse_loss(pred, noise, reduction="none").mean(dim=[1, 2, 3])
            loss = (lps * w).mean() / args.sd_grad_accum

        scaler.scale(loss).backward()
        rl += loss.item() * args.sd_grad_accum

        if (step + 1) % args.sd_grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt); scaler.update(); lrsched.step(); opt.zero_grad()
            ema.step(unet)

        step += 1
        if step % 100 == 0:
            avg = rl / 100; losses.append(avg); rl = 0.0
            print(f"  Step {step:5d}/{args.domain_adapt_steps}  loss={avg:.4f}  lr={opt.param_groups[0]['lr']:.2e}")
        if step % 500 == 0:
            torch.save({"state_dict": unet.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": lrsched.state_dict(), "global_step": step,
                        "losses": losses, "ema": ema.state_dict()}, resume_path)
            print(f"  Checkpoint at step {step}")

    torch.save(unet.state_dict(), CKPT_DIR / "sd_gastrovision_lora.pt")
    unet.save_pretrained(CKPT_DIR / "sd_gastrovision_lora_adapter")
    ema.save_adapter(unet, CKPT_DIR / "sd_gastrovision_lora_ema_adapter")

    final = losses[-1] if losses else float("nan")
    print(f"\n✅ Domain adaptation done — final loss: {final:.4f}")
    if final > 0.08:
        print(f"   ⚠ Loss > 0.08 — consider more steps (--domain_adapt_steps {args.domain_adapt_steps + 5000})")

    # Loss plot
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        steps_x = [i * 100 for i in range(1, len(losses) + 1)]
        ax.plot(steps_x, losses, color="#4878cf", linewidth=1.5)
        if len(losses) > 10:
            w = max(5, len(losses) // 20)
            sm = np.convolve(losses, np.ones(w)/w, mode="valid")
            ax.plot(steps_x[w-1:], sm, color="#d65f5f", linewidth=2.0, alpha=0.8)
        ax.axhline(0.05, color="#6acc65", linestyle="--", alpha=0.7, label="Target 0.05")
        ax.axhline(0.08, color="#f0a500", linestyle="--", alpha=0.7, label="Acceptable 0.08")
        ax.set_xlabel("Step"); ax.set_ylabel("SNR-weighted MSE Loss")
        ax.set_title("SD LoRA Domain Adaptation"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "sd_loss.png", dpi=150, bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"  Warning: could not save loss plot: {e}")

    del unet, vae, text_encoder, tokenizer; torch.cuda.empty_cache(); gc.collect()


# ==============================================================================
# SECTION 10 — Synthetic image generation
# ==============================================================================

def _postprocess(img, sharpen=1.4, contrast=1.15):
    """Mild sharpening + contrast lift to close the SD-to-endoscopy domain gap."""
    return ImageEnhance.Contrast(ImageEnhance.Sharpness(img).enhance(sharpen)).enhance(contrast)


def generate_synthetic():
    """Generate SAMPLES_PER_CLASS images per rare class from the EMA adapter."""
    ema_path = CKPT_DIR / "sd_gastrovision_lora_ema_adapter"
    raw_path = CKPT_DIR / "sd_gastrovision_lora_adapter"
    adapter  = ema_path if ema_path.exists() else raw_path
    if not adapter.exists():
        raise FileNotFoundError("No LoRA adapter found. Run domain adaptation first.")
    print(f"Loading pipeline from {adapter}...")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id, torch_dtype=torch.float16, safety_checker=None
    ).to(DEVICE)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, adapter)
    pipe.unet.eval()
    pipe.enable_attention_slicing()

    # On 40GB, enable sequential CPU offload so VAE decode and text encoder
    # are moved to CPU when not in use, freeing VRAM for the UNet forward pass.
    # This adds ~10% latency per image but prevents OOM during generation.
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb < 20:
        try:
            pipe.enable_sequential_cpu_offload()
            print(f"  CPU offload enabled (GPU VRAM: {vram_gb:.0f}GB)")
        except Exception as e:
            print(f"  CPU offload unavailable: {e}")
    else:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print(f"  xformers enabled (GPU VRAM: {vram_gb:.0f}GB)")
        except Exception:
            pass

    real_df = pd.read_csv(SPLITS_DIR / args.train_csv)
    l2n = (dict(zip(real_df["label"].astype(int), real_df["class_name"]))
           if "class_name" in real_df.columns else {})

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for cls in RARE_CLASSES:
        cls_name = l2n.get(cls, f"class_{cls}")
        cls_dir  = SYNTH_DIR / str(cls); cls_dir.mkdir(parents=True, exist_ok=True)
        prompt   = DOMAIN_PREFIX + CLASS_PROMPTS.get(cls,
            f"gastroscopy photograph of {cls_name}, round endoscopic field with dark vignette")

        existing = sorted(cls_dir.glob("synth_*.png"))
        for p in existing:
            rows.append({"image_path": str(p.relative_to(DATA_DIR)),
                         "label": int(cls), "class_name": cls_name, "source": "sd_ema"})
        if len(existing) >= args.samples_per_class:
            print(f"Class {cls}: {len(existing)} images already done — skipping"); continue

        start = len(existing)
        print(f"\nClass {cls} ({cls_name}): generating {args.samples_per_class - start} images")
        print(f"  Prompt: {prompt[:90]}...")
        idx = start

        while idx < args.samples_per_class:
            n = min(args.gen_batch_size, args.samples_per_class - idx)
            torch.cuda.empty_cache(); gc.collect()
            gens = [torch.Generator(device=DEVICE).manual_seed(args.seed + cls * 100000 + idx + i)
                    for i in range(n)]
            with torch.no_grad():
                imgs = pipe(
                    prompt=[prompt]*n, negative_prompt=[NEGATIVE_PROMPT]*n,
                    num_inference_steps=args.gen_steps, guidance_scale=args.guidance_scale,
                    height=512, width=512, generator=gens,
                ).images
            for img in imgs:
                img  = _postprocess(img.resize((args.img_size, args.img_size), Image.LANCZOS))
                path = cls_dir / f"synth_{idx:05d}.png"; img.save(path)
                rows.append({"image_path": str(path.relative_to(DATA_DIR)),
                             "label": int(cls), "class_name": cls_name, "source": "sd_ema"})
                idx += 1
            if idx % 100 == 0 or idx >= args.samples_per_class:
                print(f"  {idx}/{args.samples_per_class}")
        print(f"✅ Class {cls} done")

    del pipe; torch.cuda.empty_cache(); gc.collect()

    synth_df = pd.DataFrame(rows)
    synth_df.to_csv(SYNTH_DIR / "synthetic_train.csv", index=False)
    print(f"\n✅ {len(synth_df)} synthetic images saved → {SYNTH_DIR / 'synthetic_train.csv'}")
    return synth_df


# ==============================================================================
# SECTION 11 — Evaluation
# ==============================================================================

def _fid_features(df, root_dir, model, hook_list):
    feats = []
    for _, row in df.iterrows():
        try:
            img    = Image.open(root_dir / row["image_path"]).convert("RGB")
            assert img.mode == "RGB"
            tensor = FID_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            assert tensor.min() > -5.0 and tensor.max() < 5.0
            hook_list.clear()
            with torch.no_grad(): _ = model(tensor)
            if hook_list: feats.append(hook_list[0].flatten())
        except Exception:
            continue
    return np.array(feats) if feats else None


def _frechet(r, s):
    mr, ms   = r.mean(0), s.mean(0)
    sr, ss   = np.cov(r, rowvar=False) + 1e-6*np.eye(r.shape[1]), \
               np.cov(s, rowvar=False) + 1e-6*np.eye(s.shape[1])
    d        = mr - ms
    cov      = sqrtm(sr @ ss)
    if np.iscomplexobj(cov): cov = cov.real
    return float(d@d + np.trace(sr) + np.trace(ss) - 2*np.trace(cov))


def _kid(r, s):
    from sklearn.metrics.pairwise import polynomial_kernel
    n   = min(len(r), len(s), 500)
    rng = np.random.default_rng(args.seed)
    r   = r[rng.choice(len(r), n, replace=False)]
    s   = s[rng.choice(len(s), n, replace=False)]
    g   = 1.0 / r.shape[1]
    krr = polynomial_kernel(r, r, degree=3, gamma=g, coef0=1)
    kss = polynomial_kernel(s, s, degree=3, gamma=g, coef0=1)
    krs = polynomial_kernel(r, s, degree=3, gamma=g, coef0=1)
    np.fill_diagonal(krr, 0); np.fill_diagonal(kss, 0)
    return float((krr.sum()/(n*(n-1)) + kss.sum()/(n*(n-1)) - 2*krs.mean()) * 1000)


def compute_fid(real_df, synth_df):
    """Compute pooled FID + KID using ImageNet InceptionV3."""
    print("\nComputing FID / KID...")
    inc = inception_v3(pretrained=True, aux_logits=True, transform_input=False).to(DEVICE)
    inc.fc = nn.Identity(); inc.AuxLogits = None; inc.eval()

    hook_list = []
    def hook(m, i, o): hook_list.append(o.detach().flatten(1).cpu().numpy())
    h = inc.avgpool.register_forward_hook(hook)

    real_pooled  = real_df[real_df["label"].isin(RARE_CLASSES)]
    synth_pooled = synth_df[synth_df["label"].isin(RARE_CLASSES)]
    fr = _fid_features(real_pooled,  IMAGE_ROOT_DIR, inc, hook_list)
    fs = _fid_features(synth_pooled, DATA_DIR,       inc, hook_list)

    h.remove(); del inc; torch.cuda.empty_cache()

    if fr is None or fs is None:
        print("  FID: insufficient features"); return None, None

    fid = _frechet(fr, fs); kid = _kid(fr, fs)
    print(f"  Pooled FID     = {fid:.2f}  (n_real={len(fr)}, n_synth={len(fs)})")
    print(f"  Pooled KID×1000= {kid:.3f}")
    return fid, kid


def evaluate_all(augmented=False):
    """Full evaluation: classifier metrics per model + ensemble + FID."""
    print("\n" + "="*65)
    print(f"Evaluation ({'augmented' if augmented else 'baseline'})")
    print("="*65)

    val_ds  = GastroVisionDataset(SPLITS_DIR / args.val_csv, "val", "classifier")
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    results = {}

    for name in args.models:
        try:
            model = load_checkpoint(name, augmented)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}"); continue

        acc, yt, yp = _eval_acc(model, val_ldr)
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
        )
        print(f"\n{name}: acc={acc:.4f}  mean_f1={f1.mean():.4f}")
        print(classification_report(yt, yp, digits=4, zero_division=0))
        results[name] = {"acc": acc, "f1": f1.tolist(), "f1_mean": float(f1.mean()),
                         "f1_rare": float(f1[[c for c in RARE_CLASSES if c < NUM_CLASSES]].mean())}
        del model; torch.cuda.empty_cache()

    # Confusion matrix for best model
    if results:
        best = max(results, key=lambda k: results[k]["acc"])
        try:
            model = load_checkpoint(best, augmented)
            _, yt, yp = _eval_acc(model, val_ldr)
            cm = confusion_matrix(yt, yp)
            fig, ax = plt.subplots(figsize=(16, 14))
            sns.heatmap(cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8),
                        annot=True, fmt=".2f", cmap="Blues", ax=ax)
            ax.set_title(f"{best} — normalised confusion matrix ({'aug' if augmented else 'baseline'})")
            plt.tight_layout()
            tag = "_aug" if augmented else ""
            plt.savefig(RESULTS_DIR / f"confusion_matrix{tag}.png", dpi=150, bbox_inches="tight")
            plt.close()
            del model; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Warning: could not save confusion matrix: {e}")

    # Per-class F1 bar chart
    if results:
        fig, ax = plt.subplots(figsize=(max(12, NUM_CLASSES * 2), 5))
        x = np.arange(NUM_CLASSES); w = 0.8 / max(len(results), 1)
        cols = ["#4878cf", "#6acc65", "#d65f5f", "#f0a500", "#b47cc7"]
        for i, (nm, res) in enumerate(results.items()):
            ax.bar(x + i*w, res["f1"], w, label=nm, color=cols[i % len(cols)], alpha=0.85)
        for cls in RARE_CLASSES:
            if cls < NUM_CLASSES: ax.axvspan(cls - 0.4, cls + 0.4, alpha=0.07, color="yellow")
        ax.set_xlabel("Class"); ax.set_ylabel("F1")
        ax.set_title(f"Per-class F1 — {'augmented' if augmented else 'baseline'} (yellow=rare)")
        ax.legend(); ax.set_ylim(0, 1.15)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"f1_per_class{'_aug' if augmented else ''}.png",
                    dpi=150, bbox_inches="tight"); plt.close()

    # FID on augmented run
    if augmented:
        synth_csv = SYNTH_DIR / "synthetic_train.csv"
        if synth_csv.exists():
            real_df  = pd.read_csv(SPLITS_DIR / args.train_csv)
            synth_df = pd.read_csv(synth_csv)
            fid, kid = compute_fid(real_df, synth_df)
            results["_fid_pooled"] = fid
            results["_kid_pooled"] = kid

    # Save JSON
    tag = "_aug" if augmented else ""
    with open(RESULTS_DIR / f"eval_results{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_DIR / f'eval_results{tag}.json'}")

    print(f"\n{'Model':<35} {'Acc':>8}  {'Mean F1':>8}  {'Rare F1':>8}")
    print("-"*60)
    for nm, res in results.items():
        if nm.startswith("_"): continue
        print(f"  {nm:<33} {res['acc']:>8.4f}  {res['f1_mean']:>8.4f}  {res['f1_rare']:>8.4f}")

    return results


# ==============================================================================
# SECTION 12 — Main
# ==============================================================================

def main():
    global NUM_CLASSES, RARE_CLASSES

    print("=" * 65)
    print("GastroVision DDPM Augmentation Pipeline")
    print("=" * 65)
    print(f"  data_dir:            {DATA_DIR}")
    print(f"  output_dir:          {OUTPUT_DIR}")
    print(f"  models:              {args.models}")
    print(f"  freeze_epochs:       {args.freeze_epochs}")
    print(f"  fine_tune_epochs:    {args.fine_tune_epochs}")
    print(f"  lora_rank:           {args.lora_rank}")
    print(f"  domain_adapt_steps:  {args.domain_adapt_steps}")
    print(f"  samples_per_class:   {args.samples_per_class}")
    print(f"  gen_steps:           {args.gen_steps}")
    print(f"  guidance_scale:      {args.guidance_scale}")
    print()

    # ── Step 1: Splits ────────────────────────────────────────────────────────
    train_csv = SPLITS_DIR / args.train_csv
    val_csv   = SPLITS_DIR / args.val_csv
    test_csv  = SPLITS_DIR / args.test_csv

    if not train_csv.exists():
        print("Creating splits...")
        train_df, val_df, test_df, rare = create_splits()
        RARE_CLASSES = rare
    else:
        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        test_df  = pd.read_csv(test_csv)
        counts   = train_df["label"].value_counts()
        RARE_CLASSES = sorted(counts[counts < 30].index.tolist())
        print(f"Loaded splits. RARE_CLASSES={RARE_CLASSES}")

    NUM_CLASSES = int(train_df["label"].max()) + 1
    # Update HPARAMS now that NUM_CLASSES is known (used in model constructors)
    print(f"NUM_CLASSES={NUM_CLASSES}")

    if args.evaluate_only:
        evaluate_all(augmented=False)
        evaluate_all(augmented=True)
        return

    # ── Step 2: Train baselines on real data ──────────────────────────────────
    if not args.skip_training:
        print("\n" + "="*65)
        print("Step 2: Training classifiers on real data")
        print("="*65)
        for name in args.models:
            print(f"\n{'#'*65}\n  {name}\n{'#'*65}")
            train_classifier(name, train_csv, val_csv, augmented=False)

    # ── Step 3: Domain adaptation ─────────────────────────────────────────────
    if not args.skip_domain_adapt:
        print("\n" + "="*65)
        print("Step 3: SD LoRA domain adaptation")
        print("="*65)
        domain_adapt_sd()

    # ── Step 4: Generate synthetic images ─────────────────────────────────────
    if not args.skip_generation:
        print("\n" + "="*65)
        print("Step 4: Generating synthetic images for rare classes")
        print("="*65)
        synth_df = generate_synthetic()
    else:
        synth_csv_path = SYNTH_DIR / "synthetic_train.csv"
        synth_df = pd.read_csv(synth_csv_path) if synth_csv_path.exists() else None

    # ── Step 5: Build augmented CSV + retrain ─────────────────────────────────
    if not args.skip_training and synth_df is not None:
        print("\n" + "="*65)
        print("Step 5: Building augmented dataset + retraining")
        print("="*65)

        # Leakage check
        val_paths  = set(pd.read_csv(val_csv)["image_path"])
        test_paths = set(pd.read_csv(test_csv)["image_path"])
        assert not (set(synth_df["image_path"]) & val_paths),  "Leakage: synth overlaps val!"
        assert not (set(synth_df["image_path"]) & test_paths), "Leakage: synth overlaps test!"
        print("✅ No leakage detected")

        aug_csv = SPLITS_DIR / args.aug_train_csv
        aug_df  = pd.concat([train_df, synth_df], ignore_index=True)
        aug_df.to_csv(aug_csv, index=False)
        print(f"Augmented dataset: {len(train_df)} real + {len(synth_df)} synthetic = {len(aug_df)} total")

        for name in args.models:
            print(f"\n{'#'*65}\n  {name} (augmented)\n{'#'*65}")
            train_classifier(name, aug_csv, val_csv, augmented=True)

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("Step 6: Evaluation")
    print("="*65)
    evaluate_all(augmented=False)
    if not args.skip_training and synth_df is not None:
        evaluate_all(augmented=True)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
