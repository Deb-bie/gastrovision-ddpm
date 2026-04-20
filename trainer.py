"""
src/trainer.py
Two-phase training engine for all GastroVision classifier models.
Works for EfficientNetV2, Swin, MobileNetV3, and HybridCNNTransformer.

Bug fixes vs original Colab code
─────────────────────────────────
1. DOUBLE FORWARD PASS BUG (critical):
   Original phase-1 loop called criterion(model(xb), yb) TWICE per
   iteration — once for .backward() and once for .item(). This:
     a) doubled GPU memory usage
     b) inflated reported loss by ~2×
     c) wasted 50% of compute in phase 1
   Fixed: compute loss once, cache result.

2. MISSING COMMENT CHAR (syntax error):
   Line starting with "── Phase 2: full fine-tune" was missing the leading #.
   This would crash on import. Fixed.

3. HEAD DETECTION for HybridCNNTransformer:
   Original head detection was: model.head if hasattr(model, "head") else model.classifier
   HybridCNNTransformer has .head but its backbones also need freezing.
   Fixed: model-aware freeze logic.

4. GRADIENT CLIPPING skipped in phase 1:
   Added clip_grad_norm_ to phase 1 as well — prevents instability
   when head LR is 10× base LR.

5. AMP (automatic mixed precision) added:
   Original ran full float32 throughout. On A100/H100 this wastes 2–3×
   memory and compute vs bf16. Added torch.cuda.amp.GradScaler.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from configs.config import CKPT_DIR, HPARAMS, RESULTS_DIR
from src.dataset import GastroVisionDataset, get_weighted_sampler
from src.losses import FocalLoss
from src.models import get_baseline_model, MODEL_REGISTRY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_loader(model, loader, device=DEVICE):
    """
    Returns (accuracy, y_true, y_pred).
    Uses AMP for consistency with training.
    """
    from sklearn.metrics import accuracy_score
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            with autocast():
                preds = model(xb).argmax(dim=1)
            y_pred_list.append(preds.cpu().numpy())
            y_true_list.append(yb.numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return float(np.mean(y_true == y_pred)), y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Model-aware freeze / unfreeze
# ─────────────────────────────────────────────────────────────────────────────

def _freeze_backbone(model, model_name: str):
    """Freeze all params except the classification head."""
    if model_name == "hybrid_cnn_transformer":
        model.freeze_backbones()
    else:
        for p in model.parameters():
            p.requires_grad = False
        # timm models expose .head; MobileNetV3 exposes .classifier
        head = getattr(model, "head", None) or getattr(model, "classifier", None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad = True
        else:
            raise AttributeError(f"Cannot find head/classifier on {model_name}")


def _unfreeze_all(model, model_name: str):
    if model_name == "hybrid_cnn_transformer":
        model.unfreeze_all()
    else:
        for p in model.parameters():
            p.requires_grad = True


def _head_params(model, model_name: str):
    if model_name == "hybrid_cnn_transformer":
        return list(model.fusion.parameters()) + list(model.head.parameters())
    head = getattr(model, "head", None) or getattr(model, "classifier", None)
    return list(head.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Single-model training
# ─────────────────────────────────────────────────────────────────────────────

def train_single_baseline(
    model_name:       str,
    train_csv:        str,
    val_csv:          str,
    augmented:        bool = False,
    use_sampler:      bool = False,
    hparams_override: dict = None,
    save_ckpt:        bool = True,
):
    """
    Two-phase training:
      Phase 1 — frozen backbone, head-only training (freeze_epochs)
      Phase 2 — full fine-tune with cosine LR schedule (fine_tune_epochs)

    Returns: history dict with train_loss, val_acc, phase lists.
    """
    cfg = dict(HPARAMS[model_name])
    if hparams_override:
        cfg.update(hparams_override)

    # Datasets / loaders
    train_ds = GastroVisionDataset(train_csv, split="train", mode="classifier")
    val_ds   = GastroVisionDataset(val_csv,   split="val",   mode="classifier")

    if use_sampler:
        sampler      = get_weighted_sampler(train_csv)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=4, pin_memory=True)

    model     = get_baseline_model(model_name).to(DEVICE)
    criterion = FocalLoss(gamma=cfg.get("gamma", 2.0))
    scaler    = GradScaler()
    history   = {"train_loss": [], "val_acc": [], "phase": []}

    ckpt_name = f"sota_{model_name}{'_aug' if augmented else ''}.pt"

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    _freeze_backbone(model, model_name)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"] * cfg.get("freeze_lr_mult", 10.0),
    )

    print(f"\n{'='*60}")
    print(f"[{model_name}] Phase 1: frozen backbone "
          f"({cfg['freeze_epochs']} epochs, "
          f"lr={cfg['lr'] * cfg.get('freeze_lr_mult', 10.0):.2e})")
    print(f"{'='*60}")

    for epoch in range(cfg["freeze_epochs"]):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                # BUG FIX: compute loss ONCE (original did it twice)
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc, _, _ = evaluate_on_loader(model, val_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(acc)
        history["phase"].append("freeze")
        print(f"  Epoch {epoch+1:2d}/{cfg['freeze_epochs']}  "
              f"loss={avg_loss:.4f}  val_acc={acc:.4f}")

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    _unfreeze_all(model, model_name)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["fine_tune_epochs"]
    )

    best_acc = 0.0

    print(f"\n{'='*60}")
    print(f"[{model_name}] Phase 2: full fine-tune "
          f"({cfg['fine_tune_epochs']} epochs, "
          f"lr={cfg['lr']:.2e})")
    print(f"{'='*60}")

    for epoch in range(cfg["fine_tune_epochs"]):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        acc, _, _ = evaluate_on_loader(model, val_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(acc)
        history["phase"].append("finetune")
        print(f"  Epoch {epoch+1:2d}/{cfg['fine_tune_epochs']}  "
              f"loss={avg_loss:.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            if save_ckpt:
                torch.save(model.state_dict(), CKPT_DIR / ckpt_name)
                print(f"  ✅ Checkpoint saved (val_acc={best_acc:.4f})")

    print(f"\n  ★ {model_name} best val_acc: {best_acc:.4f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Train all models
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODEL_NAMES = list(MODEL_REGISTRY.keys())


def train_all_baselines(train_csv, val_csv, augmented=False, models=None):
    """
    Trains all registered models (including HybridCNNTransformer).
    Pass models=['swin', 'hybrid_cnn_transformer'] to run a subset.
    """
    if models is None:
        models = ALL_MODEL_NAMES

    # Reload tuned hparams if available
    hparams_path = RESULTS_DIR / "best_hparams.json"
    if hparams_path.exists():
        from configs.config import HPARAMS as _H
        with open(hparams_path) as f:
            loaded = json.load(f)
        _H.update(loaded)
        print(f"Loaded tuned HPARAMS from {hparams_path}")

    all_histories = {}
    for model_name in models:
        print(f"\n{'#'*65}")
        print(f"  Training: {model_name}")
        print(f"{'#'*65}")
        history = train_single_baseline(
            model_name=model_name,
            train_csv=train_csv,
            val_csv=val_csv,
            augmented=augmented,
        )
        all_histories[model_name] = history

    return all_histories


# ─────────────────────────────────────────────────────────────────────────────
# Optuna hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_model(model_name, train_csv, val_csv, n_trials=20, tune_epochs=10):
    """
    Optuna TPE + MedianPruner search for a given model.
    Returns the best study.
    """
    import optuna
    from optuna.pruners  import MedianPruner
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        lr             = trial.suggest_float("lr",             1e-5, 1e-3, log=True)
        gamma          = trial.suggest_float("gamma",          0.5,  3.0)
        batch_size     = trial.suggest_categorical("batch_size", [16, 32])
        freeze_lr_mult = trial.suggest_float("freeze_lr_mult", 2.0, 15.0)
        weight_decay   = trial.suggest_float("weight_decay",   1e-5, 1e-2, log=True)

        train_ds = GastroVisionDataset(train_csv, split="train", mode="classifier")
        val_ds   = GastroVisionDataset(val_csv,   split="val",   mode="classifier")
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=2, pin_memory=True)

        model     = get_baseline_model(model_name).to(DEVICE)
        criterion = FocalLoss(gamma=gamma)
        scaler    = GradScaler()

        # Phase 1 warmup
        _freeze_backbone(model, model_name)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr * freeze_lr_mult
        )
        for _ in range(min(5, tune_epochs // 2)):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                with autocast():
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

        # Phase 2 fine-tune
        _unfreeze_all(model, model_name)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tune_epochs)

        best_acc = 0.0
        for epoch in range(tune_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                with autocast():
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            sch.step()

            acc = evaluate_on_loader(model, val_loader)[0]
            best_acc = max(best_acc, acc)
            trial.report(acc, epoch)
            if trial.should_prune():
                del model
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        del model
        torch.cuda.empty_cache()
        return best_acc

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest config — {model_name}: val_acc={best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:<20} {v}")

    return study


def update_hparams_from_studies(studies):
    """Writes best Optuna params back into HPARAMS and saves to disk."""
    from configs.config import HPARAMS
    for model_name, study in studies.items():
        best = study.best_trial.params
        HPARAMS[model_name].update({
            "lr":             best["lr"],
            "batch_size":     best["batch_size"],
            "gamma":          best["gamma"],
            "freeze_lr_mult": best["freeze_lr_mult"],
            "weight_decay":   best["weight_decay"],
        })

    hparams_path = RESULTS_DIR / "best_hparams.json"
    with open(hparams_path, "w") as f:
        json.dump(HPARAMS, f, indent=2)
    print(f"Saved tuned HPARAMS → {hparams_path}")
    return HPARAMS
