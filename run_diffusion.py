#!/usr/bin/env python3
"""
scripts/run_diffusion.py
Stage 3: SD domain adaptation (with EMA) + rare class generation.

Pipeline:
  A) Domain-adapt SD on all GastroVision images, saving EMA adapter
  B) Generate SAMPLES_PER_CLASS images per rare class from EMA adapter
  C) Build augmented CSV (real + synthetic)

Usage examples
--------------
# Full pipeline (15k steps, 500 samples, EMA enabled)
python scripts/run_diffusion.py

# Skip adaptation (EMA adapter already saved), re-generate only
python scripts/run_diffusion.py --skip_adapt

# Resume interrupted adaptation (reads resume_sd_gastrovision_lora.pt)
python scripts/run_diffusion.py --resume

# Run more steps to push loss below 0.05
python scripts/run_diffusion.py --train_steps 20000 --resume

# Only specific classes
python scripts/run_diffusion.py --skip_adapt --classes 1 5 11

# Disable EMA (not recommended — for ablation only)
python scripts/run_diffusion.py --skip_adapt --no_ema

# Tune inference quality (more steps, lower guidance for medical images)
python scripts/run_diffusion.py --skip_adapt --num_steps 50 --guidance 6.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    EMA_DECAY, EMA_UPDATE_AFTER_STEP,
    RARE_CLASSES, SAMPLES_PER_CLASS, SD_TRAIN_STEPS,
    SD_BATCH_SIZE, SD_GRAD_ACCUM, SD_LR,
)
from src.diffusion import domain_adapt_sd, generate_rare_class_images, build_augmented_dataset


def main():
    parser = argparse.ArgumentParser(
        description="GastroVision SD domain adaptation + rare class generation"
    )

    # Adaptation
    parser.add_argument("--skip_adapt",   action="store_true",
                        help="Skip domain adaptation (use existing adapter)")
    parser.add_argument("--resume",       action="store_true", default=True,
                        help="Resume from saved checkpoint if it exists (default: True)")
    parser.add_argument("--no_resume",    action="store_true",
                        help="Force restart even if a checkpoint exists")
    parser.add_argument("--train_steps",  type=int, default=SD_TRAIN_STEPS,
                        help=f"Total training steps (default: {SD_TRAIN_STEPS})")
    parser.add_argument("--batch_size",   type=int, default=SD_BATCH_SIZE)
    parser.add_argument("--grad_accum",   type=int, default=SD_GRAD_ACCUM)
    parser.add_argument("--lr",           type=float, default=SD_LR)

    # EMA
    parser.add_argument("--ema_decay",    type=float, default=EMA_DECAY,
                        help=f"EMA decay (default: {EMA_DECAY})")
    parser.add_argument("--ema_warmup",   type=int, default=EMA_UPDATE_AFTER_STEP,
                        help=f"Steps before EMA starts (default: {EMA_UPDATE_AFTER_STEP})")
    parser.add_argument("--no_ema",       action="store_true",
                        help="Disable EMA — use raw weights for generation (ablation only)")

    # Generation
    parser.add_argument("--classes",      nargs="+", type=int, default=None,
                        help="Classes to generate. Default: RARE_CLASSES from config")
    parser.add_argument("--samples",      type=int, default=SAMPLES_PER_CLASS,
                        help=f"Images per class (default: {SAMPLES_PER_CLASS})")
    parser.add_argument("--num_steps",    type=int, default=40,
                        help="Inference denoising steps (default: 40, range: 30-50)")
    parser.add_argument("--guidance",     type=float, default=7.0,
                        help="CFG guidance scale (default: 7.0, range: 6.0-7.5)")
    parser.add_argument("--gen_batch",    type=int, default=4,
                        help="Images per pipeline call during generation (default: 4)")

    args = parser.parse_args()

    classes = args.classes or RARE_CLASSES
    resume  = args.resume and not args.no_resume

    print("="*65)
    print("GastroVision Diffusion Pipeline")
    print("="*65)
    print(f"  Classes:         {classes}")
    print(f"  Samples/class:   {args.samples}")
    print(f"  EMA:             {'disabled' if args.no_ema else f'decay={args.ema_decay}'}")
    print(f"  Train steps:     {args.train_steps}")
    print(f"  Inference steps: {args.num_steps}")
    print(f"  Guidance scale:  {args.guidance}")
    print()

    # ── Stage A: Domain adaptation ────────────────────────────────────────────
    if not args.skip_adapt:
        print("="*65)
        print("Stage A: SD LoRA Domain Adaptation with EMA")
        print("="*65)
        print("Fine-tuning on ALL GastroVision images (~8000 frames)")
        print("Teaching SD what endoscopy looks like before rare class generation")
        print(f"Target: loss < 0.05 by step {args.train_steps}")
        print()

        domain_adapt_sd(
            num_train_steps  = args.train_steps,
            batch_size       = args.batch_size,
            lr               = args.lr,
            gradient_accum   = args.grad_accum,
            ema_decay        = args.ema_decay,
            ema_update_after = args.ema_warmup,
            resume           = resume,
        )
    else:
        print("Stage A: Skipping adaptation (--skip_adapt)")

    # ── Stage B: Generation ───────────────────────────────────────────────────
    print()
    print("="*65)
    print("Stage B: Rare Class Image Generation")
    print("="*65)
    print(f"Using {'raw' if args.no_ema else 'EMA'} adapter")
    print()

    synth_df = generate_rare_class_images(
        classes           = classes,
        samples_per_class = args.samples,
        guidance_scale    = args.guidance,
        num_steps         = args.num_steps,
        batch_size        = args.gen_batch,
        use_ema           = not args.no_ema,
    )

    # ── Stage C: Build augmented CSV ──────────────────────────────────────────
    print()
    print("="*65)
    print("Stage C: Build Augmented Training CSV")
    print("="*65)
    build_augmented_dataset()

    print()
    print("="*65)
    print("✅ Diffusion pipeline complete")
    print("="*65)
    print("Next steps:")
    print("  1. Retrain all models on augmented data:")
    print("     python scripts/run_train.py --augmented")
    print("  2. Evaluate generation quality:")
    print("     python scripts/run_evaluate.py --fid --extractor hybrid")


if __name__ == "__main__":
    main()
