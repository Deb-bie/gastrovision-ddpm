#!/usr/bin/env python3
"""
scripts/run_train.py
Stage 2: Train all classifier models (or a subset).

Usage:
  # Train all models on real data
  python scripts/run_train.py

  # Train only hybrid and swin on augmented data
  python scripts/run_train.py --models hybrid_cnn_transformer swin --augmented

  # Tune hyperparameters first
  python scripts/run_train.py --tune --tune_model swin --n_trials 20

  # Train with augmented CSV after diffusion
  python scripts/run_train.py --augmented --train_csv /data/gastrovision/data/splits/train_aug.csv
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from configs.config import TRAIN_CSV, VAL_CSV, CKPT_DIR, RESULTS_DIR, HPARAMS
from src.trainer import (
    train_all_baselines, tune_model, update_hparams_from_studies, ALL_MODEL_NAMES
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     nargs="+", default=None,
                        help="Models to train. Default: all registered models.")
    parser.add_argument("--train_csv",  type=str, default=str(TRAIN_CSV))
    parser.add_argument("--val_csv",    type=str, default=str(VAL_CSV))
    parser.add_argument("--augmented",  action="store_true",
                        help="Save checkpoints as sota_{model}_aug.pt")
    parser.add_argument("--tune",       action="store_true",
                        help="Run Optuna tuning before training")
    parser.add_argument("--tune_model", type=str, default="swin",
                        help="Which model to tune (default: swin)")
    parser.add_argument("--n_trials",   type=int, default=20)
    parser.add_argument("--tune_epochs",type=int, default=10)
    args = parser.parse_args()

    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.tune:
        print(f"\nRunning Optuna tuning for: {args.tune_model}")
        study = tune_model(
            model_name=args.tune_model,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            n_trials=args.n_trials,
            tune_epochs=args.tune_epochs,
        )
        # Apply best params to all models
        studies = {args.tune_model: study}
        best_params = study.best_trial.params
        for name in (args.models or ALL_MODEL_NAMES):
            if name != args.tune_model:
                HPARAMS[name]["gamma"] = best_params["gamma"]
                HPARAMS[name]["lr"]    = best_params["lr"]
        update_hparams_from_studies(studies)

    histories = train_all_baselines(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        augmented=args.augmented,
        models=args.models,
    )

    # Save training histories
    histories_serializable = {}
    for model_name, h in histories.items():
        histories_serializable[model_name] = {
            "train_loss": [float(x) for x in h["train_loss"]],
            "val_acc":    [float(x) for x in h["val_acc"]],
            "phase":      h["phase"],
        }
    out = RESULTS_DIR / f"training_histories{'_aug' if args.augmented else ''}.json"
    with open(out, "w") as f:
        json.dump(histories_serializable, f, indent=2)
    print(f"\nTraining histories saved → {out}")


if __name__ == "__main__":
    main()
