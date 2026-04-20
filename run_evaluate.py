#!/usr/bin/env python3
"""
scripts/run_evaluate.py
Stage 4+5: Full evaluation — classifier metrics, FID/KID/MS-SSIM, K-Fold, ensemble.

Usage examples
--------------
# Evaluate all baseline models
python scripts/run_evaluate.py

# Evaluate augmented models
python scripts/run_evaluate.py --augmented

# FID with HybridCNNTransformer features (best quality — use this)
python scripts/run_evaluate.py --fid --extractor hybrid

# FID with domain-adapted InceptionV3 (DermDiff method)
python scripts/run_evaluate.py --fid --extractor inception_domain

# FID with standard ImageNet InceptionV3 (for paper comparability)
python scripts/run_evaluate.py --fid --extractor inception_imagenet

# Full pipeline: augmented models + hybrid FID + K-Fold + ensemble
python scripts/run_evaluate.py --augmented --fid --extractor hybrid --kfold --ensemble

# Only K-Fold for augmented swin
python scripts/run_evaluate.py --augmented --kfold --kfold_model swin
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from configs.config import (
    CLF_BATCH_SIZE, NUM_CLASSES, RARE_CLASSES, RESULTS_DIR,
    VAL_CSV, CKPT_DIR,
)
from src.dataset import GastroVisionDataset
from src.models import load_trained_baseline, MODEL_REGISTRY, SOTAEnsemble
from src.evaluation import (
    evaluate_and_plot, compute_generation_quality, kfold_evaluate
)


def main():
    parser = argparse.ArgumentParser(
        description="GastroVision evaluation pipeline"
    )

    # Model selection
    parser.add_argument("--models",    nargs="+", default=None,
                        help="Models to evaluate. Default: all trained models.")
    parser.add_argument("--augmented", action="store_true",
                        help="Load *_aug.pt checkpoints")

    # Ensemble
    parser.add_argument("--ensemble",  action="store_true",
                        help="Evaluate confidence-weighted ensemble")

    # FID / quality
    parser.add_argument("--fid",       action="store_true",
                        help="Compute FID + KID + MS-SSIM")
    parser.add_argument("--extractor", type=str, default="hybrid",
                        choices=["hybrid", "inception_domain", "inception_imagenet"],
                        help=(
                            "Feature extractor for FID/KID. "
                            "'hybrid' = HybridCNNTransformer (best). "
                            "'inception_domain' = GastroVision-tuned InceptionV3. "
                            "'inception_imagenet' = standard ImageNet InceptionV3."
                        ))
    parser.add_argument("--hybrid_ckpt", type=str, default=None,
                        help="Path to HybridCNNTransformer checkpoint. "
                             "Default: auto-detect from CKPT_DIR.")
    parser.add_argument("--ssim_pairs", type=int, default=30,
                        help="Real-synth pairs per class for MS-SSIM (default: 30)")

    # K-Fold
    parser.add_argument("--kfold",       action="store_true",
                        help="Run K-Fold evaluation")
    parser.add_argument("--kfold_model", type=str, default="swin",
                        help="Model to use for K-Fold (default: swin)")
    parser.add_argument("--kfold_k",     type=int, default=5)

    args = parser.parse_args()

    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_aug" if args.augmented else ""

    val_ds     = GastroVisionDataset(VAL_CSV, split="val", mode="classifier")
    val_loader = DataLoader(val_ds, batch_size=CLF_BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ── Per-model evaluation ──────────────────────────────────────────────────
    model_names = args.models or list(MODEL_REGISTRY.keys())
    all_results = {}

    print("\n" + "="*65)
    print(f"Per-Model Evaluation ({'augmented' if args.augmented else 'baseline'})")
    print("="*65)

    for model_name in model_names:
        try:
            model = load_trained_baseline(model_name, augmented=args.augmented)
        except FileNotFoundError as e:
            print(f"  Skipping {model_name}: {e}")
            continue

        label = f"{model_name} ({'aug' if args.augmented else 'base'})"
        acc, y_true, y_pred, probs, f1 = evaluate_and_plot(
            model_or_ensemble = model,
            name              = label,
            val_loader        = val_loader,
            num_classes       = NUM_CLASSES,
        )
        all_results[model_name] = {
            "acc":         acc,
            "f1_mean":     float(f1.mean()),
            "f1_rare_mean": float(f1[[c for c in RARE_CLASSES if c < NUM_CLASSES]].mean()),
            "f1_per_class": f1.tolist(),
        }
        del model
        torch.cuda.empty_cache()

    # ── Ensemble ──────────────────────────────────────────────────────────────
    if args.ensemble:
        print("\n" + "="*65)
        print(f"Ensemble Evaluation ({'augmented' if args.augmented else 'baseline'})")
        print("="*65)
        try:
            ensemble = SOTAEnsemble(augmented=args.augmented)
            acc, y_true, y_pred, probs, f1 = evaluate_and_plot(
                model_or_ensemble = ensemble,
                name              = f"Ensemble{suffix}",
                val_loader        = val_loader,
                num_classes       = NUM_CLASSES,
                is_ensemble       = True,
            )
            all_results["ensemble"] = {
                "acc":          acc,
                "f1_mean":      float(f1.mean()),
                "f1_rare_mean": float(f1[[c for c in RARE_CLASSES if c < NUM_CLASSES]].mean()),
                "f1_per_class": f1.tolist(),
            }
        except Exception as e:
            print(f"  Ensemble failed: {e}")

    # Save classifier results
    out_path = RESULTS_DIR / f"eval_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nClassifier results saved → {out_path}")

    # Print summary table
    if all_results:
        print(f"\n{'Model':<35} {'Acc':>8}  {'Mean F1':>8}  {'Rare F1':>8}")
        print(f"  {'-'*62}")
        for name, r in all_results.items():
            print(f"  {name:<33} {r['acc']:>8.4f}  "
                  f"{r['f1_mean']:>8.4f}  {r['f1_rare_mean']:>8.4f}")

    # ── FID / KID / MS-SSIM ───────────────────────────────────────────────────
    if args.fid:
        print("\n" + "="*65)
        print(f"Generation Quality  (extractor={args.extractor})")
        print("="*65)

        hybrid_ckpt = None
        if args.hybrid_ckpt:
            hybrid_ckpt = Path(args.hybrid_ckpt)
        elif args.extractor == "hybrid":
            # Auto-detect: prefer augmented checkpoint
            for candidate in [
                CKPT_DIR / "sota_hybrid_cnn_transformer_aug.pt",
                CKPT_DIR / "sota_hybrid_cnn_transformer.pt",
            ]:
                if candidate.exists():
                    hybrid_ckpt = candidate
                    break

        try:
            fid_results = compute_generation_quality(
                extractor    = args.extractor,
                hybrid_ckpt  = hybrid_ckpt,
                n_ssim_pairs = args.ssim_pairs,
            )

            # Serialise and save
            def _make_serializable(obj):
                if isinstance(obj, float) and np.isnan(obj): return None
                if isinstance(obj, (np.floating, np.integer)): return float(obj)
                if isinstance(obj, dict): return {k: _make_serializable(v) for k, v in obj.items()}
                return obj

            fid_out = RESULTS_DIR / f"fid_results_{args.extractor}.json"
            with open(fid_out, "w") as f:
                json.dump(
                    {str(k): _make_serializable(v) for k, v in fid_results.items()},
                    f, indent=2
                )
            print(f"\nFID results saved → {fid_out}")

            # Print key numbers
            pooled = fid_results.get("pooled", {})
            print(f"\n{'='*40}")
            print(f"POOLED SUMMARY (report these in paper)")
            print(f"{'='*40}")
            print(f"  FID      = {pooled.get('fid', 'N/A')}")
            print(f"  KID×1000 = {pooled.get('kid', 'N/A')}")
            print(f"  MS-SSIM  = {pooled.get('msssim', 'N/A')}")
            print(f"  Extractor: {pooled.get('extractor', args.extractor)}")

        except Exception as e:
            print(f"  FID computation failed: {e}")
            import traceback; traceback.print_exc()

    # ── K-Fold ────────────────────────────────────────────────────────────────
    if args.kfold:
        print("\n" + "="*65)
        print(f"K-Fold Evaluation  (model={args.kfold_model}, k={args.kfold_k})")
        print("="*65)
        label = "augmented" if args.augmented else "baseline"
        try:
            kfold_summary = kfold_evaluate(
                model_name  = args.kfold_model,
                label       = label,
                k           = args.kfold_k,
            )
            kfold_out = RESULTS_DIR / f"kfold_results_{label}_{args.kfold_model}.json"
            with open(kfold_out, "w") as f:
                json.dump(
                    {str(k): {
                        "precision": list(v["precision"]),
                        "recall":    list(v["recall"]),
                        "f1":        list(v["f1"]),
                        "n":         v["n"],
                    } for k, v in kfold_summary.items()},
                    f, indent=2
                )
            print(f"K-Fold results saved → {kfold_out}")
        except Exception as e:
            print(f"  K-Fold failed: {e}")
            import traceback; traceback.print_exc()

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
