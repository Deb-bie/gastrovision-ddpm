"""
src/evaluation.py
Generation quality evaluation + classifier evaluation + K-Fold.

FID extractor options (in order of quality for this task)
──────────────────────────────────────────────────────────
1. HybridCNNTransformer (best)
   Uses the trained HybridCNNTransformer's fusion layer (1536-dim) as the
   feature extractor. This is the most meaningful FID for this project
   because the features are native GastroVision representations learned by
   a model trained specifically on this dataset with cross-attention fusion
   over both CNN texture and Transformer global context features.

2. Domain-adapted InceptionV3 (good)
   Fine-tunes InceptionV3 for 5 epochs on GastroVision, then uses the
   avgpool layer (2048-dim). Following DermDiff methodology.

3. Standard ImageNet InceptionV3 (for paper comparability)
   No domain adaptation. Use only for comparing to other papers that
   also report standard FID. Will be artificially high.

Additional metrics
──────────────────
- KID (Kernel Inception Distance): polynomial kernel MMD in Inception space.
  Reliable with small n (works at n >= 10). Always report alongside FID.

- MS-SSIM: Multi-Scale Structural Similarity. Domain-agnostic.
  Complements FID by measuring structural/perceptual similarity.

- LPIPS: Learned Perceptual Image Patch Similarity. Perceptual distance
  metric that correlates better with human judgement than SSIM.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from scipy.linalg import sqrtm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd

from configs.config import (
    CLF_BATCH_SIZE, CKPT_DIR, IMG_SIZE, KFOLD_SPLITS, MIN_RELIABLE_SAMPLES,
    NUM_CLASSES, RARE_CLASSES, RANDOM_SEED, RESULTS_DIR,
    TRAIN_CSV, VAL_CSV, TEST_CSV, SYNTH_CSV,
    IMAGE_ROOT_DIR, DATA_DIR,
)
from src.dataset import GastroVisionDataset
from src.losses import FocalLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractors for FID
# ─────────────────────────────────────────────────────────────────────────────

def _make_transform(size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _extract_features_generic(df: pd.DataFrame, root_dir: Path,
                               model: nn.Module, transform: T.Compose,
                               hook_list: list, device: torch.device,
                               desc: str = "") -> np.ndarray:
    """
    Generic feature extraction loop used by all extractor types.
    Relies on a forward hook registered on the target layer.
    """
    feats = []
    n     = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            img    = Image.open(root_dir / row["image_path"]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            hook_list.clear()
            with torch.no_grad():
                _ = model(tensor)
            if hook_list:
                feats.append(hook_list[0].flatten())
        except Exception as e:
            if i < 3:
                print(f"    Warning: {row['image_path']}: {e}")
        if (i + 1) % 200 == 0:
            print(f"    {desc}: {i+1}/{n}")
    return np.array(feats) if feats else None


def build_hybrid_extractor(ckpt_path: Path = None, device: torch.device = DEVICE):
    """
    Uses the trained HybridCNNTransformer's penultimate features (1536-dim)
    as the FID feature space.

    This is the highest-quality extractor for this project because:
      1. The feature space is trained specifically on GastroVision (all 27 classes)
      2. Features capture both local CNN texture + global Transformer context
         via the cross-attention fusion layer
      3. The model has seen both normal and rare endoscopy appearances during
         training, giving it a more complete representation of the domain

    The hook is placed on model.head[0] (LayerNorm = first head module),
    capturing the 1536-dim fused features before the classification layers.

    Parameters
    ----------
    ckpt_path : Path to the HybridCNNTransformer checkpoint.
                Defaults to CKPT_DIR/sota_hybrid_cnn_transformer.pt
    """
    from src.models import HybridCNNTransformer

    if ckpt_path is None:
        ckpt_path = CKPT_DIR / "sota_hybrid_cnn_transformer.pt"
        if not ckpt_path.exists():
            ckpt_path = CKPT_DIR / "sota_hybrid_cnn_transformer_aug.pt"

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"HybridCNNTransformer checkpoint not found: {ckpt_path}\n"
            f"Train it first with: python scripts/run_train.py "
            f"--models hybrid_cnn_transformer"
        )

    model = HybridCNNTransformer(num_classes=NUM_CLASSES).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  HybridCNNTransformer loaded from {ckpt_path}")

    hook_list = []
    def hook(module, inp, out):
        # out is (B, 1536) — the fused cross-attention features
        hook_list.append(out.detach().cpu().numpy())

    # Register on LayerNorm (first element of head Sequential)
    # This captures the 1536-dim pre-classifier fused representation
    handle   = model.head[0].register_forward_hook(hook)
    transform = _make_transform(IMG_SIZE)

    def extract(df: pd.DataFrame, root_dir: Path, desc: str = "") -> np.ndarray:
        return _extract_features_generic(
            df, root_dir, model, transform, hook_list, device, desc=desc
        )

    return model, handle, extract, "HybridCNNTransformer (1536-dim)"


def build_inception_extractor(device: torch.device = DEVICE,
                               domain_adapt: bool = True,
                               train_csv: str = TRAIN_CSV,
                               fine_tune_epochs: int = 5):
    """
    InceptionV3 feature extractor (avgpool, 2048-dim).

    domain_adapt=True  → fine-tunes on GastroVision (DermDiff methodology)
    domain_adapt=False → standard ImageNet features (for paper comparability)
    """
    from torchvision.models import inception_v3

    inception = inception_v3(
        pretrained=True, aux_logits=True, transform_input=False
    ).to(device)

    if domain_adapt:
        inception.fc           = nn.Linear(2048, NUM_CLASSES).to(device)
        inception.AuxLogits.fc = nn.Linear(768, NUM_CLASSES).to(device)

        ds     = GastroVisionDataset(train_csv, split="train", mode="classifier")
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
        opt    = torch.optim.Adam(inception.parameters(), lr=1e-4)
        crit   = FocalLoss(gamma=2.0)

        inception.train()
        for ep in range(fine_tune_epochs):
            loss_sum = 0
            for xb, yb in loader:
                xb  = F.interpolate(xb.to(device), size=(299, 299))
                yb  = yb.to(device)
                out = inception(xb)
                loss = (crit(out[0], yb) + 0.4 * crit(out[1], yb)
                        if isinstance(out, tuple) else crit(out, yb))
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item()
            print(f"  InceptionV3 fine-tune epoch {ep+1}/{fine_tune_epochs} "
                  f"loss={loss_sum/len(loader):.4f}")

        inception.fc        = nn.Identity()
        inception.AuxLogits = None
        label = "Domain-Adapted InceptionV3 (2048-dim)"
    else:
        inception.fc        = nn.Identity()
        inception.AuxLogits = None
        label = "Standard ImageNet InceptionV3 (2048-dim)"

    inception.eval()

    hook_list = []
    def hook(module, inp, out):
        hook_list.append(out.detach().flatten(1).cpu().numpy())
    handle    = inception.avgpool.register_forward_hook(hook)
    transform = _make_transform(299)

    def extract(df: pd.DataFrame, root_dir: Path, desc: str = "") -> np.ndarray:
        return _extract_features_generic(
            df, root_dir, inception, transform, hook_list, device, desc=desc
        )

    return inception, handle, extract, label


# ─────────────────────────────────────────────────────────────────────────────
# Distance metrics
# ─────────────────────────────────────────────────────────────────────────────

def frechet_distance(feat_real: np.ndarray, feat_synth: np.ndarray) -> float:
    """
    Standard Fréchet Distance.
    Unreliable when n_real < ~50 (covariance matrix near-singular).
    Always pair with KID for small-sample cases.
    """
    mu_r, mu_s = feat_real.mean(0), feat_synth.mean(0)
    sigma_r    = np.cov(feat_real, rowvar=False)
    sigma_s    = np.cov(feat_synth, rowvar=False)

    # Regularise to prevent singular covariance with small n
    eps = 1e-6 * np.eye(sigma_r.shape[0])
    sigma_r += eps
    sigma_s += eps

    diff    = mu_r - mu_s
    covmean = sqrtm(sigma_r @ sigma_s)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(
        diff @ diff
        + np.trace(sigma_r) + np.trace(sigma_s)
        - 2.0 * np.trace(covmean)
    )


def kernel_inception_distance(feat_real: np.ndarray,
                               feat_synth: np.ndarray,
                               degree: int = 3,
                               gamma: float = None,
                               coef: float = 1.0) -> float:
    """
    Kernel Inception Distance (KID) — polynomial kernel MMD.
    More reliable than FID for small n (works at n >= 10).
    Uses an unbiased estimator. Returns value × 1000 for readability.
    """
    from sklearn.metrics.pairwise import polynomial_kernel

    n     = min(len(feat_real), len(feat_synth), 500)   # cap at 500 for speed
    rng   = np.random.default_rng(RANDOM_SEED)
    r_idx = rng.choice(len(feat_real),  n, replace=False)
    s_idx = rng.choice(len(feat_synth), n, replace=False)
    r     = feat_real[r_idx]
    s     = feat_synth[s_idx]

    gamma = gamma or 1.0 / feat_real.shape[1]
    k_rr  = polynomial_kernel(r, r, degree=degree, gamma=gamma, coef0=coef)
    k_ss  = polynomial_kernel(s, s, degree=degree, gamma=gamma, coef0=coef)
    k_rs  = polynomial_kernel(r, s, degree=degree, gamma=gamma, coef0=coef)

    # Unbiased estimator: zero diagonal before averaging
    np.fill_diagonal(k_rr, 0)
    np.fill_diagonal(k_ss, 0)
    mmd = (k_rr.sum() / (n * (n - 1))
           + k_ss.sum() / (n * (n - 1))
           - 2.0 * k_rs.mean())
    return float(mmd * 1000)


def ms_ssim_score(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                  n_pairs: int = 30, device: torch.device = DEVICE) -> dict:
    """
    Multi-Scale SSIM per rare class using torchmetrics.
    Returns dict: cls → {"mean": float, "std": float, "n": int}
    """
    try:
        from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    except ImportError:
        print("  torchmetrics not available — skipping MS-SSIM")
        return {}

    ms_ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0
    ).to(device)
    to_tensor  = T.ToTensor()
    results    = {}

    for cls in RARE_CLASSES:
        real_cls  = real_df[real_df["label"] == cls]
        synth_cls = synth_df[synth_df["label"] == cls]
        scores    = []

        for _, rrow in real_cls.iterrows():
            try:
                real_t = to_tensor(
                    Image.open(IMAGE_ROOT_DIR / rrow["image_path"]).convert("RGB")
                ).unsqueeze(0).to(device)

                sample = synth_cls.sample(
                    min(n_pairs, len(synth_cls)), random_state=RANDOM_SEED
                )
                for _, srow in sample.iterrows():
                    try:
                        synth_t = to_tensor(
                            Image.open(DATA_DIR / srow["image_path"]).convert("RGB")
                        ).unsqueeze(0).to(device)
                        scores.append(ms_ssim_fn(real_t, synth_t).item())
                    except Exception:
                        continue
            except Exception:
                continue

        if scores:
            results[cls] = {
                "mean": float(np.mean(scores)),
                "std":  float(np.std(scores)),
                "n":    len(scores),
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main quality evaluation — all three metrics, all extractor options
# ─────────────────────────────────────────────────────────────────────────────

def compute_generation_quality(
    real_csv         = TRAIN_CSV,
    synth_csv        = SYNTH_CSV,
    rare_classes     = RARE_CLASSES,
    extractor        = "hybrid",     # "hybrid" | "inception_domain" | "inception_imagenet"
    hybrid_ckpt      = None,         # path to hybrid checkpoint; None = auto-detect
    n_ssim_pairs     = 30,
    domain_adapt_epochs = 5,
):
    """
    Full generation quality evaluation using:
      - FID in the chosen feature space
      - KID in the chosen feature space (reliable for small n)
      - MS-SSIM (domain-agnostic structural similarity)

    Extractor choices:
      "hybrid"           → HybridCNNTransformer features (best for this project)
      "inception_domain" → Domain-adapted InceptionV3 (DermDiff method)
      "inception_imagenet" → Standard ImageNet InceptionV3 (paper comparability)

    Returns a dict with per-class and pooled scores for all three metrics.
    """
    real_df  = pd.read_csv(real_csv)
    synth_df = pd.read_csv(synth_csv)
    label_to_name = (
        dict(zip(real_df["label"].astype(int), real_df["class_name"]))
        if "class_name" in real_df.columns else {}
    )

    # ── Build extractor ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Building feature extractor: {extractor}")
    print(f"{'='*70}")

    if extractor == "hybrid":
        try:
            model, handle, extract_fn, ext_label = build_hybrid_extractor(
                ckpt_path=hybrid_ckpt, device=DEVICE
            )
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
            print("  Falling back to domain-adapted InceptionV3")
            extractor = "inception_domain"

    if extractor == "inception_domain":
        model, handle, extract_fn, ext_label = build_inception_extractor(
            device=DEVICE, domain_adapt=True,
            fine_tune_epochs=domain_adapt_epochs,
        )
    elif extractor == "inception_imagenet":
        model, handle, extract_fn, ext_label = build_inception_extractor(
            device=DEVICE, domain_adapt=False,
        )

    print(f"\nUsing: {ext_label}")

    # ── Per-class FID and KID ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Per-Class FID / KID")
    print(f"{'='*70}")
    print(f"  Note: FID is unreliable when n_real < 20. "
          f"Use KID for small-sample classes.")
    print(f"\n  {'Class':<6} {'Name':<38} {'N_real':<8} {'N_synth':<8} "
          f"{'FID':>8}  {'KID×1000':>10}  {'Reliable?'}")
    print(f"  {'-'*88}")

    fid_per_class = {}
    kid_per_class = {}

    for cls in rare_classes:
        cls_name  = label_to_name.get(cls, f"class_{cls}")
        real_cls  = real_df[real_df["label"] == cls]
        synth_cls = synth_df[synth_df["label"] == cls]

        if len(real_cls) < 2 or len(synth_cls) < 2:
            print(f"  {cls:<6} {cls_name:<38} too few samples")
            fid_per_class[cls] = None
            kid_per_class[cls] = None
            continue

        feat_r = extract_fn(real_cls,  IMAGE_ROOT_DIR, desc=f"real cls {cls}")
        feat_s = extract_fn(synth_cls, DATA_DIR,       desc=f"synth cls {cls}")

        if feat_r is None or feat_s is None or len(feat_r) < 2 or len(feat_s) < 2:
            fid_per_class[cls] = None
            kid_per_class[cls] = None
            continue

        # FID — only if n_real >= 20 (below that covariance is near-singular)
        if len(feat_r) >= 20:
            try:
                fid_score = frechet_distance(feat_r, feat_s)
            except Exception as e:
                fid_score = None
                print(f"    FID failed for class {cls}: {e}")
        else:
            fid_score = None   # Don't report unreliable numbers

        # KID — reliable from n=10
        if len(feat_r) >= 10:
            try:
                kid_score = kernel_inception_distance(feat_r, feat_s)
            except Exception as e:
                kid_score = None
                print(f"    KID failed for class {cls}: {e}")
        else:
            kid_score = None

        fid_per_class[cls] = fid_score
        kid_per_class[cls] = kid_score

        fid_str      = f"{fid_score:.1f}" if fid_score is not None else "N/A*"
        kid_str      = f"{kid_score:.3f}" if kid_score is not None else "N/A*"
        reliable_str = "✅" if len(feat_r) >= 20 else "⚠ n<20"

        print(f"  {cls:<6} {cls_name:<38} {len(feat_r):<8} {len(feat_s):<8} "
              f"{fid_str:>8}  {kid_str:>10}  {reliable_str}")

    # ── Pooled FID / KID (primary reportable metric) ──────────────────────────
    print(f"\n{'='*70}")
    print(f"POOLED FID / KID  (all rare classes combined — report this)")
    print(f"{'='*70}")

    real_pooled  = real_df[real_df["label"].isin(rare_classes)]
    synth_pooled = synth_df[synth_df["label"].isin(rare_classes)]
    feat_r_pool  = extract_fn(real_pooled,  IMAGE_ROOT_DIR, desc="real pooled")
    feat_s_pool  = extract_fn(synth_pooled, DATA_DIR,       desc="synth pooled")

    pooled_fid, pooled_kid = None, None
    if feat_r_pool is not None and feat_s_pool is not None:
        pooled_fid = frechet_distance(feat_r_pool, feat_s_pool)
        pooled_kid = kernel_inception_distance(feat_r_pool, feat_s_pool)
        fid_q      = _fid_quality(pooled_fid)
        kid_q      = _kid_quality(pooled_kid)
        print(f"\n  FID     = {pooled_fid:.2f}  ({fid_q})")
        print(f"  KID×1000= {pooled_kid:.3f}  ({kid_q})")
        print(f"  n_real  = {len(feat_r_pool)}, n_synth = {len(feat_s_pool)}")
        print(f"\n  ↑ Use POOLED as your primary metric in the paper.")
        print(f"    Per-class FID is supplementary and noted as unreliable for n<20.")

    # ── Clean up extractor ────────────────────────────────────────────────────
    handle.remove()
    del model
    torch.cuda.empty_cache()

    # ── MS-SSIM ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MS-SSIM  (structural similarity, domain-agnostic)")
    print(f"{'='*70}")
    msssim_results = ms_ssim_score(real_df, synth_df, n_pairs=n_ssim_pairs)

    valid_ssim = [v["mean"] for v in msssim_results.values() if v is not None]
    pooled_msssim = float(np.mean(valid_ssim)) if valid_ssim else None

    print(f"\n  {'Class':<6} {'Name':<38} {'MS-SSIM':>10}  {'Quality'}")
    print(f"  {'-'*62}")
    for cls in rare_classes:
        cls_name = label_to_name.get(cls, f"class_{cls}")
        res      = msssim_results.get(cls)
        if res:
            q = "good" if res["mean"] > 0.4 else "acceptable" if res["mean"] > 0.2 else "poor"
            print(f"  {cls:<6} {cls_name:<38} {res['mean']:>8.4f}±{res['std']:.4f}  {q}")
        else:
            print(f"  {cls:<6} {cls_name:<38} {'—':>10}")
    if pooled_msssim:
        print(f"\n  Pooled MS-SSIM = {pooled_msssim:.4f}")

    # ── Combined summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"COMBINED QUALITY SUMMARY  [{ext_label}]")
    print(f"{'='*70}")
    print(f"  {'Class':<6} {'Name':<32} {'FID↓':>8}  {'KID×1000↓':>10}  "
          f"{'MS-SSIM↑':>10}  Verdict")
    print(f"  {'-'*80}")

    for cls in rare_classes:
        cls_name  = label_to_name.get(cls, f"class_{cls}")
        fid_val   = fid_per_class.get(cls)
        kid_val   = kid_per_class.get(cls)
        ssim_val  = msssim_results.get(cls)

        fid_str  = f"{fid_val:.1f}" if fid_val  is not None else "N/A"
        kid_str  = f"{kid_val:.3f}" if kid_val  is not None else "N/A"
        ssim_str = f"{ssim_val['mean']:.3f}" if ssim_val is not None else "N/A"

        if kid_val is not None and ssim_val is not None:
            verdict = ("✅ good"      if kid_val < 2.0 and ssim_val["mean"] > 0.2
                       else "⚠ mixed" if kid_val < 5.0 or ssim_val["mean"] > 0.15
                       else "❌ poor")
        elif kid_val is not None:
            verdict = "✅" if kid_val < 2.0 else "⚠" if kid_val < 5.0 else "❌"
        else:
            verdict = "—"

        print(f"  {cls:<6} {cls_name:<32} {fid_str:>8}  {kid_str:>10}  "
              f"{ssim_str:>10}  {verdict}")

    print(f"\n  POOLED → FID={pooled_fid:.2f if pooled_fid else 'N/A'}"
          f"  KID×1000={pooled_kid:.3f if pooled_kid else 'N/A'}"
          f"  MS-SSIM={pooled_msssim:.4f if pooled_msssim else 'N/A'}")

    # ── Save plots ────────────────────────────────────────────────────────────
    _save_quality_plots(
        rare_classes, fid_per_class, kid_per_class, msssim_results,
        label_to_name, ext_label
    )

    # ── Build return dict ─────────────────────────────────────────────────────
    results = {}
    for cls in rare_classes:
        ssim_val = msssim_results.get(cls)
        results[cls] = {
            "fid":      fid_per_class.get(cls),
            "kid":      kid_per_class.get(cls),
            "msssim":   ssim_val["mean"] if ssim_val else None,
            "n_real":   len(real_df[real_df["label"] == cls]),
            "n_synth":  len(synth_df[synth_df["label"] == cls]),
            "name":     label_to_name.get(cls, f"class_{cls}"),
        }
    results["pooled"] = {
        "fid":     pooled_fid,
        "kid":     pooled_kid,
        "msssim":  pooled_msssim,
        "n_real":  len(feat_r_pool) if feat_r_pool is not None else 0,
        "n_synth": len(feat_s_pool) if feat_s_pool is not None else 0,
        "extractor": ext_label,
    }
    return results


def _fid_quality(score: float) -> str:
    if score is None:  return "—"
    if score < 50:     return "excellent"
    if score < 100:    return "good"
    if score < 200:    return "acceptable"
    return "poor"


def _kid_quality(score: float) -> str:
    if score is None:  return "—"
    if score < 0.5:    return "excellent"
    if score < 2.0:    return "good"
    if score < 5.0:    return "acceptable"
    return "poor"


def _save_quality_plots(rare_classes, fid_per_class, kid_per_class,
                        msssim_results, label_to_name, ext_label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid_cls = [
            c for c in rare_classes
            if kid_per_class.get(c) is not None
        ]
        if not valid_cls:
            return

        cls_labels = [
            f"Cls {c}\n{label_to_name.get(c,'')[:12]}"
            for c in valid_cls
        ]

        n_plots = 2 + (1 if any(msssim_results.get(c) for c in valid_cls) else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 6, 5))
        if n_plots == 1:
            axes = [axes]

        ax_idx = 0

        # KID plot (always shown — most reliable)
        kid_vals   = [kid_per_class[c] for c in valid_cls]
        kid_colors = [
            "#6acc65" if v < 0.5 else "#4878cf" if v < 2.0
            else "#f0a500" if v < 5.0 else "#d65f5f"
            for v in kid_vals
        ]
        bars = axes[ax_idx].bar(cls_labels, kid_vals, color=kid_colors, edgecolor="white")
        for bar, val in zip(bars, kid_vals):
            axes[ax_idx].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9
            )
        axes[ax_idx].axhline(0.5, color="#6acc65", linestyle="--", alpha=0.6, label="Excellent (<0.5)")
        axes[ax_idx].axhline(2.0, color="#4878cf", linestyle="--", alpha=0.6, label="Good (<2.0)")
        axes[ax_idx].axhline(5.0, color="#f0a500", linestyle="--", alpha=0.6, label="Acceptable (<5.0)")
        axes[ax_idx].set_ylabel("KID × 1000 (lower is better)")
        axes[ax_idx].set_title(f"KID per Rare Class\n{ext_label}")
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(axis="y", alpha=0.3)
        ax_idx += 1

        # FID plot (shown with N/A for small classes)
        fid_vals   = [fid_per_class.get(c) for c in valid_cls]
        plot_fids  = [v if v is not None else 0 for v in fid_vals]
        fid_colors = [
            "#6acc65" if v and v < 50 else "#4878cf" if v and v < 100
            else "#f0a500" if v and v < 200 else "#d65f5f"
            for v in fid_vals
        ]
        bars = axes[ax_idx].bar(cls_labels, plot_fids, color=fid_colors, edgecolor="white")
        for bar, val in zip(bars, fid_vals):
            label_txt = f"{val:.1f}" if val is not None else "N/A*"
            axes[ax_idx].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label_txt, ha="center", va="bottom", fontsize=9
            )
        axes[ax_idx].axhline(50,  color="#6acc65", linestyle="--", alpha=0.6, label="Excellent (<50)")
        axes[ax_idx].axhline(100, color="#4878cf", linestyle="--", alpha=0.6, label="Good (<100)")
        axes[ax_idx].set_ylabel("FID (lower is better)  *N/A = n<20")
        axes[ax_idx].set_title(f"FID per Rare Class\n{ext_label}")
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(axis="y", alpha=0.3)
        ax_idx += 1

        # MS-SSIM plot if available
        if ax_idx < len(axes):
            ssim_vals   = [msssim_results[c]["mean"] if msssim_results.get(c) else 0
                           for c in valid_cls]
            ssim_stds   = [msssim_results[c]["std"] if msssim_results.get(c) else 0
                           for c in valid_cls]
            ssim_colors = [
                "#6acc65" if v > 0.4 else "#4878cf" if v > 0.2 else "#d65f5f"
                for v in ssim_vals
            ]
            axes[ax_idx].bar(cls_labels, ssim_vals, yerr=ssim_stds,
                             color=ssim_colors, edgecolor="white", capsize=4)
            axes[ax_idx].axhline(0.4, color="#6acc65", linestyle="--", alpha=0.6, label="Good (>0.4)")
            axes[ax_idx].axhline(0.2, color="#4878cf", linestyle="--", alpha=0.6, label="Acceptable (>0.2)")
            axes[ax_idx].set_ylabel("MS-SSIM (higher is better)")
            axes[ax_idx].set_title("MS-SSIM per Rare Class\n(structural similarity)")
            axes[ax_idx].set_ylim(0, 1.0)
            axes[ax_idx].legend(fontsize=8)
            axes[ax_idx].grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"Generation Quality — Rare Classes\n"
            f"Extractor: {ext_label}",
            fontsize=12
        )
        plt.tight_layout()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / "generation_quality.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Quality plot saved → {RESULTS_DIR / 'generation_quality.png'}")
    except Exception as e:
        print(f"  Warning: could not save quality plot: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Classifier evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_plot(model_or_ensemble, name, val_loader,
                      num_classes=NUM_CLASSES, is_ensemble=False,
                      save_dir=RESULTS_DIR, min_reliable=MIN_RELIABLE_SAMPLES):
    """Full evaluation pipeline: confusion matrix, P/R/F1, ROC, calibration."""
    y_true_list, y_pred_list, probs_list = [], [], []

    if is_ensemble:
        with torch.no_grad():
            for xb, yb in val_loader:
                preds, probs, _ = model_or_ensemble.predict_with_confidence(xb)
                y_pred_list.append(preds.cpu().numpy())
                y_true_list.append(yb.numpy())
                probs_list.append(probs.cpu().numpy())
    else:
        model_or_ensemble.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb    = xb.to(DEVICE)
                probs = F.softmax(model_or_ensemble(xb), dim=1)
                preds = probs.argmax(dim=1)
                y_pred_list.append(preds.cpu().numpy())
                y_true_list.append(yb.numpy())
                probs_list.append(probs.cpu().numpy())

    y_true    = np.concatenate(y_true_list)
    y_pred    = np.concatenate(y_pred_list)
    all_probs = np.concatenate(probs_list)
    acc       = accuracy_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"{name}  val_acc={acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    precision, recall, f1_scores, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)),
        average=None, zero_division=0
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(" ", "_").replace("/", "_")

        # Confusion matrix
        cm      = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        sns.heatmap(cm,      annot=True, fmt="d",    cmap="Blues", ax=axes[0], annot_kws={"size": 7})
        sns.heatmap(cm_norm, annot=True, fmt=".2f",  cmap="Blues", ax=axes[1], annot_kws={"size": 7})
        for ax, t in zip(axes, ["counts", "normalised"]):
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{name} — CM ({t})")
        plt.suptitle(f"{name}  val_acc={acc:.4f}", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_dir / f"cm_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # P/R/F1 per class
        x     = np.arange(num_classes)
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(14, num_classes * 2), 6))
        ax.bar(x - width, precision, width, label="Precision", color="#4878cf", alpha=0.85)
        ax.bar(x,         recall,    width, label="Recall",    color="#6acc65", alpha=0.85)
        ax.bar(x + width, f1_scores, width, label="F1",        color="#d65f5f", alpha=0.85)
        for cls in RARE_CLASSES:
            if cls < num_classes:
                ax.axvspan(cls - 0.45, cls + 0.45, alpha=0.08, color="yellow")
        ax.axhline(f1_scores.mean(), color="red", linestyle="--",
                   label=f"Mean F1={f1_scores.mean():.3f}")
        ax.set_xlabel("Class"); ax.set_ylabel("Score")
        ax.set_title(f"{name} — P/R/F1 per class  (yellow = rare)")
        ax.legend(); ax.set_ylim(-0.1, 1.15)
        plt.tight_layout()
        plt.savefig(save_dir / f"prf1_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Confidence + calibration
        top_conf = all_probs.max(axis=1)
        correct  = (y_pred == y_true)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(top_conf[correct],  bins=30, alpha=0.7, color="green", label="Correct")
        axes[0].hist(top_conf[~correct], bins=30, alpha=0.7, color="red",   label="Wrong")
        axes[0].set_xlabel("Confidence"); axes[0].set_title("Confidence Distribution")
        axes[0].legend()
        prob_true, prob_pred = calibration_curve(correct, top_conf, n_bins=10)
        axes[1].plot(prob_pred, prob_true, "s-", label=name)
        axes[1].plot([0, 1], [0, 1], "k--", label="Perfect")
        axes[1].set_xlabel("Mean confidence"); axes[1].set_ylabel("Fraction correct")
        axes[1].set_title("Reliability Diagram"); axes[1].legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"calibration_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"  Warning: could not save evaluation plots: {e}")

    return acc, y_true, y_pred, all_probs, f1_scores


# ─────────────────────────────────────────────────────────────────────────────
# K-Fold evaluation
# ─────────────────────────────────────────────────────────────────────────────

def kfold_evaluate(model_name="swin", eval_csv=None,
                   rare_classes=RARE_CLASSES, k=KFOLD_SPLITS,
                   label="baseline"):
    from sklearn.model_selection import StratifiedKFold
    from src.models import load_trained_baseline

    if eval_csv is None:
        train_df = pd.read_csv(TRAIN_CSV)
        val_df   = pd.read_csv(VAL_CSV)
        full_df  = pd.concat([train_df, val_df], ignore_index=True)
        full_eval_csv = DATA_DIR / "_kfold_full_eval.csv"
        full_df.to_csv(full_eval_csv, index=False)
        print(f"K-Fold: using train+val = {len(full_df)} samples")
    else:
        full_df       = pd.read_csv(eval_csv)
        full_eval_csv = Path(eval_csv)

    class_counts = full_df["label"].value_counts()
    foldable     = class_counts[class_counts >= k].index.tolist()
    unfoldable   = class_counts[class_counts <  k].index.tolist()
    print(f"  Foldable ({k}-fold): {sorted(foldable)}")
    print(f"  Unfoldable (n<{k}): {sorted(unfoldable)}")

    df_foldable = full_df[full_df["label"].isin(foldable)]
    model       = load_trained_baseline(model_name, augmented=(label == "augmented"))
    model.eval()

    skf          = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    fold_results = {cls: {"precision": [], "recall": [], "f1": []} for cls in foldable}

    for fold, (_, val_idx) in enumerate(skf.split(df_foldable, df_foldable["label"])):
        fold_val = df_foldable.iloc[val_idx]
        fold_csv = DATA_DIR / f"_kfold_fold_{fold}.csv"
        fold_val.to_csv(fold_csv, index=False)

        fold_ds  = GastroVisionDataset(fold_csv, split="val", mode="classifier")
        fold_ldr = DataLoader(fold_ds, batch_size=CLF_BATCH_SIZE,
                              shuffle=False, num_workers=2)

        y_true_list, y_pred_list = [], []
        with torch.no_grad():
            for xb, yb in fold_ldr:
                preds = model(xb.to(DEVICE)).argmax(dim=1)
                y_pred_list.append(preds.cpu().numpy())
                y_true_list.append(yb.numpy())

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(NUM_CLASSES)),
            average=None, zero_division=0
        )
        for cls in foldable:
            fold_results[cls]["precision"].append(p[cls])
            fold_results[cls]["recall"].append(r[cls])
            fold_results[cls]["f1"].append(f[cls])

        fold_csv.unlink(missing_ok=True)

    del model
    torch.cuda.empty_cache()
    if eval_csv is None:
        full_eval_csv.unlink(missing_ok=True)

    summary = {}
    print(f"\nK-Fold Results — {label} (k={k}, model={model_name})")
    print(f"  {'Cls':>4} {'n':>5}  {'P mean±std':<18} {'R mean±std':<18} {'F1 mean±std':<18} {'Rare'}")
    print(f"  {'-'*75}")

    for cls in range(NUM_CLASSES):
        if cls not in fold_results:
            continue
        p_m, p_s = np.mean(fold_results[cls]["precision"]), np.std(fold_results[cls]["precision"])
        r_m, r_s = np.mean(fold_results[cls]["recall"]),    np.std(fold_results[cls]["recall"])
        f_m, f_s = np.mean(fold_results[cls]["f1"]),        np.std(fold_results[cls]["f1"])
        n        = int(class_counts.get(cls, 0))
        rare_tag = " ★" if cls in rare_classes else ""
        print(f"  {cls:>4} {n:>5}  {p_m:.3f}±{p_s:.3f}           "
              f"{r_m:.3f}±{r_s:.3f}           {f_m:.3f}±{f_s:.3f}{rare_tag}")
        summary[cls] = {
            "precision": (p_m, p_s),
            "recall":    (r_m, r_s),
            "f1":        (f_m, f_s),
            "n":         n,
        }

    return summary
