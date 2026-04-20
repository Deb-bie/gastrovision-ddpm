#!/usr/bin/env python3
"""
scripts/run_prepare.py
Stage 1: Build CSV and create train/val/test splits.
Run once before any training.

Usage:
  python scripts/run_prepare.py --raw_dir /data/gastrovision/data/gastrovision_raw/Gastrovision
"""
import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import IMAGE_ROOT_DIR, SPLITS_DIR, CKPT_DIR, RESULTS_DIR, LOGS_DIR
from src.dataset import create_gastrovision_splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",    type=str, default=str(IMAGE_ROOT_DIR))
    parser.add_argument("--splits_dir", type=str, default=str(SPLITS_DIR))
    args = parser.parse_args()

    for d in [CKPT_DIR, RESULTS_DIR, LOGS_DIR, SPLITS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, unreliable = create_gastrovision_splits(
        raw_dir=args.raw_dir, splits_dir=args.splits_dir
    )
    print(f"\nRare/unreliable classes: {unreliable}")
    print("Update RARE_CLASSES in configs/config.py if needed.")
    print("Done.")

if __name__ == "__main__":
    main()
