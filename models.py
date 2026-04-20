"""
src/models.py
Model definitions:
  - EfficientNetV2-S
  - Swin Transformer Base
  - MobileNetV3-Large
  - HybridCNNTransformer  ← NEW: ConvNeXt stem + Swin blocks + cross-attention fusion
  - Confidence-based ensemble
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path

from configs.config import NUM_CLASSES, IMG_SIZE, CKPT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Baseline models
# ─────────────────────────────────────────────────────────────────────────────

def get_effnetv2_s(num_classes=NUM_CLASSES):
    return timm.create_model("efficientnetv2_rw_s", pretrained=True, num_classes=num_classes)

def get_swin_transformer(num_classes=NUM_CLASSES):
    return timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes)

def get_mobilenetv3(num_classes=NUM_CLASSES):
    return timm.create_model("tf_mobilenetv3_large_minimal_100", pretrained=True, num_classes=num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid CNN-Transformer (NEW)
# ─────────────────────────────────────────────────────────────────────────────
# Design rationale for medical imaging / rare class problem:
#
# CNN branch  → ConvNeXt-Small backbone: strong local texture + color features.
#               Critical for endoscopy where lesion texture (e.g., vessel
#               tortuosity in angiectasia, erythema color patterns) is diagnostic.
#
# Transformer branch → Swin-Tiny: captures long-range spatial context.
#               Critical for positional features (e.g., cecum anatomy,
#               pylorus position in the frame, retroflex rectum morphology).
#
# Cross-attention fusion → lets each modality attend to the other's features
#               before final classification, rather than naive concatenation.
#               This is especially useful for rare classes where each branch
#               may capture complementary discriminative signals.
#
# Why this helps FID / rare class quality:
#   The richer feature extractor produces a better representation space,
#   reducing the effective domain gap between real and synthetic images.
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between CNN and Transformer feature vectors.
    Query from CNN attends over Transformer keys/values and vice versa;
    both are concatenated for the final classifier head.
    """

    def __init__(self, cnn_dim: int, tfm_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Project both to a common dimension
        fused_dim = min(cnn_dim, tfm_dim)
        self.cnn_proj = nn.Linear(cnn_dim, fused_dim)
        self.tfm_proj = nn.Linear(tfm_dim, fused_dim)

        self.cross_attn_cnn2tfm = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_tfm2cnn = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(fused_dim)
        self.norm2 = nn.LayerNorm(fused_dim)
        self.out_dim = fused_dim * 2

    def forward(self, cnn_feat: torch.Tensor, tfm_feat: torch.Tensor):
        # cnn_feat, tfm_feat: (B, D)
        cnn_q = self.cnn_proj(cnn_feat).unsqueeze(1)   # (B, 1, D')
        tfm_q = self.tfm_proj(tfm_feat).unsqueeze(1)   # (B, 1, D')

        # CNN queries Transformer
        attn_cnn, _ = self.cross_attn_cnn2tfm(query=cnn_q, key=tfm_q, value=tfm_q)
        attn_cnn = self.norm1(attn_cnn.squeeze(1) + cnn_q.squeeze(1))

        # Transformer queries CNN
        attn_tfm, _ = self.cross_attn_tfm2cnn(query=tfm_q, key=cnn_q, value=cnn_q)
        attn_tfm = self.norm2(attn_tfm.squeeze(1) + tfm_q.squeeze(1))

        return torch.cat([attn_cnn, attn_tfm], dim=-1)   # (B, 2*D')


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer for GastroVision classification.

    Architecture:
        ConvNeXt-Small (CNN branch)  ─┐
                                       ├─ CrossAttentionFusion ─ MLP Head ─ logits
        Swin-Tiny (Transformer branch)─┘

    Parameters
    ----------
    num_classes  : number of output classes
    pretrained   : load ImageNet pretrained weights for both branches
    dropout      : dropout before classifier head
    freeze_cnn   : freeze CNN branch (useful for phase-1 head-only training)
    freeze_tfm   : freeze Transformer branch (same)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained:  bool = True,
        dropout:     float = 0.3,
        freeze_cnn:  bool = False,
        freeze_tfm:  bool = False,
    ):
        super().__init__()

        # CNN branch — ConvNeXt-Small, remove classifier head
        self.cnn_backbone = timm.create_model(
            "convnext_small", pretrained=pretrained, num_classes=0
        )
        cnn_dim = self.cnn_backbone.num_features   # 768 for convnext_small

        # Transformer branch — Swin-Tiny, remove classifier head
        self.tfm_backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0
        )
        tfm_dim = self.tfm_backbone.num_features   # 768 for swin_tiny

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(cnn_dim, tfm_dim, num_heads=8, dropout=0.1)
        fused_dim   = self.fusion.out_dim

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Optional freeze for phase-1 training
        if freeze_cnn:
            for p in self.cnn_backbone.parameters():
                p.requires_grad = False
        if freeze_tfm:
            for p in self.tfm_backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_backbone(x)   # (B, 768)
        tfm_feat = self.tfm_backbone(x)   # (B, 768)
        fused    = self.fusion(cnn_feat, tfm_feat)   # (B, 1536)
        return self.head(fused)           # (B, num_classes)

    def freeze_backbones(self):
        for p in self.cnn_backbone.parameters():
            p.requires_grad = False
        for p in self.tfm_backbone.parameters():
            p.requires_grad = False
        for p in self.fusion.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


def get_hybrid_cnn_transformer(num_classes=NUM_CLASSES):
    return HybridCNNTransformer(num_classes=num_classes, pretrained=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "efficientnetv2_rw_s":    get_effnetv2_s,
    "swin":                   get_swin_transformer,
    "mobile":                 get_mobilenetv3,
    "hybrid_cnn_transformer": get_hybrid_cnn_transformer,
}

def get_baseline_model(name: str, num_classes: int = NUM_CLASSES):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes=num_classes)


def load_trained_baseline(model_name: str, augmented: bool = False, ckpt_dir: Path = CKPT_DIR):
    """Loads a saved checkpoint. augmented=True → loads *_aug.pt variant."""
    suffix    = "_aug" if augmented else ""
    ckpt_file = f"sota_{model_name}{suffix}.pt"
    ckpt_path = Path(ckpt_dir) / ckpt_file

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_baseline_model(model_name).to(device)
    state  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {model_name} from {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Confidence-based ensemble
# ─────────────────────────────────────────────────────────────────────────────

class SOTAEnsemble:
    """
    Confidence-weighted ensemble over any subset of trained models.
    Each model's vote is weighted by its own per-sample confidence.
    Now includes the Hybrid CNN-Transformer.
    """

    def __init__(self, model_names=None, augmented=True, device=None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.augmented = augmented

        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())

        self.models = {}
        for name in model_names:
            suffix    = "_aug" if augmented else ""
            ckpt_path = CKPT_DIR / f"sota_{name}{suffix}.pt"
            if ckpt_path.exists():
                model = get_baseline_model(name).to(self.device)
                state = torch.load(ckpt_path, map_location=self.device)
                model.load_state_dict(state)
                model.eval()
                self.models[name] = model
                print(f"  Loaded {name}")
            else:
                print(f"  Skipping {name} — checkpoint not found: {ckpt_path}")

        if not self.models:
            raise RuntimeError("No models loaded.")
        print(f"\n  Ensemble: {len(self.models)} models ({'aug' if augmented else 'baseline'})")

    def predict_with_confidence(self, x):
        x           = x.to(self.device)
        probs_list  = []
        model_names = list(self.models.keys())

        with torch.no_grad():
            for model in self.models.values():
                probs = F.softmax(model(x), dim=1)
                probs_list.append(probs)

        stacked     = torch.stack(probs_list, dim=0)                    # (M, B, C)
        confidences = stacked.max(dim=2).values.permute(1, 0)           # (B, M)
        weights     = confidences / confidences.sum(dim=1, keepdim=True) # (B, M) normalized

        ensemble_probs = (stacked * weights.permute(1, 0).unsqueeze(-1)).sum(dim=0)  # (B, C)
        preds          = ensemble_probs.argmax(dim=1)

        weight_breakdown = {name: weights[:, i] for i, name in enumerate(model_names)}
        return preds, ensemble_probs, weight_breakdown

    def predict(self, x):
        preds, _, _ = self.predict_with_confidence(x)
        return preds

    def predict_proba(self, x):
        _, probs, _ = self.predict_with_confidence(x)
        return probs
