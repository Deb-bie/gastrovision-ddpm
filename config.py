"""
configs/config.py
Central configuration for GastroVision pipeline on Nautilus.
All paths are relative to PROJECT_DIR — set via env var or CLI.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path(os.environ.get("PROJECT_DIR", "/data/gastrovision"))
DATA_DIR       = PROJECT_DIR / "data"
IMAGE_ROOT_DIR = DATA_DIR / "gastrovision_raw/Gastrovision"
SPLITS_DIR     = DATA_DIR / "splits"
SYNTH_DIR      = DATA_DIR / "synthetic"
CKPT_DIR       = PROJECT_DIR / "checkpoints"
RESULTS_DIR    = PROJECT_DIR / "results"
LOGS_DIR       = PROJECT_DIR / "logs"

# ── CSV paths ──────────────────────────────────────────────────────────────────
FULL_CSV      = DATA_DIR / "gastrovision_full.csv"
TRAIN_CSV     = SPLITS_DIR / "train.csv"
VAL_CSV       = SPLITS_DIR / "val.csv"
TEST_CSV      = SPLITS_DIR / "test.csv"
AUG_TRAIN_CSV = SPLITS_DIR / "train_aug.csv"
SYNTH_CSV     = DATA_DIR / "synthetic_train.csv"

# ── Dataset ────────────────────────────────────────────────────────────────────
IMG_SIZE         = 224
NUM_CHANNELS     = 3
NUM_CLASSES      = 27          # updated dynamically after CSV creation
RARE_CLASSES     = [1, 5, 11, 12, 17, 23, 26]   # updated after split creation
RANDOM_SEED      = 42

# ── Diffusion (SD + LoRA) ──────────────────────────────────────────────────────
SD_MODEL_ID  = "runwayml/stable-diffusion-v1-5"
LORA_RANK    = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.1

SD_TRAIN_STEPS       = 15000    # increased from 10k — more steps = lower loss floor
SD_BATCH_SIZE        = 4
SD_LR                = 1e-4
SD_GRAD_ACCUM        = 4
SAMPLES_PER_CLASS    = 500      # increased from 200 — better FID estimate + more classifier signal

# ── EMA (Exponential Moving Average on UNet weights) ──────────────────────────
# Keeps a smoothed running average of UNet weights during LoRA training.
# Generated images come from EMA weights, not raw training weights.
# EMA weights are more stable → sharper, less noisy images → lower FID.
# Decay=0.9999 averages roughly the last 10k update steps.
EMA_DECAY            = 0.9999
EMA_UPDATE_AFTER_STEP = 100    # don't start EMA until weights have warmed up

# ── Classifier ─────────────────────────────────────────────────────────────────
CLF_BATCH_SIZE   = 32
CLF_LR           = 3e-4
CLF_WEIGHT_DECAY = 1e-4

# ── K-Fold ─────────────────────────────────────────────────────────────────────
KFOLD_SPLITS         = 5
MIN_RELIABLE_SAMPLES = 10

# ── Per-model hyperparameters (updated by Optuna) ──────────────────────────────
HPARAMS = {
    "efficientnetv2_rw_s": {
        "lr": 3e-4,
        "freeze_epochs": 16,
        "fine_tune_epochs": 24,
        "batch_size": 32,
        "gamma": 2.0,
        "freeze_lr_mult": 10.0,
        "weight_decay": 0.01,
    },
    "swin": {
        "lr": 2e-4,
        "freeze_epochs": 16,
        "fine_tune_epochs": 24,
        "batch_size": 32,
        "gamma": 2.0,
        "freeze_lr_mult": 10.0,
        "weight_decay": 0.01,
    },
    "mobile": {
        "lr": 5e-4,
        "freeze_epochs": 16,
        "fine_tune_epochs": 24,
        "batch_size": 32,
        "gamma": 2.0,
        "freeze_lr_mult": 10.0,
        "weight_decay": 0.01,
    },
    # ── NEW: Hybrid CNN-Transformer ─────────────────────────────────────────
    "hybrid_cnn_transformer": {
        "lr": 2e-4,
        "freeze_epochs": 10,
        "fine_tune_epochs": 30,
        "batch_size": 32,
        "gamma": 2.0,
        "freeze_lr_mult": 5.0,
        "weight_decay": 0.01,
    },
}

# ── CLASS MAP ──────────────────────────────────────────────────────────────────
CLASS_MAP = {
    "Accessory tools":                                      0,
    "Angiectasia":                                          1,
    "Barretts esophagus":                                   2,
    "Blood in lumen":                                       3,
    "Cecum":                                                4,
    "Colon diverticula":                                    5,
    "Colon polyps":                                         6,
    "Colorectal cancer":                                    7,
    "Duodenal bulb":                                        8,
    "Dyed-lifted-polyps":                                   9,
    "Dyed-resection-margins":                               10,
    "Erythema":                                             11,
    "Esophageal varices":                                   12,
    "Esophagitis":                                          13,
    "Gastric polyps":                                       14,
    "Gastroesophageal_junction_normal z-line":              15,
    "Ileocecal valve":                                      16,
    "Mucosal inflammation large bowel":                     17,
    "Normal esophagus":                                     18,
    "Normal mucosa and vascular pattern in the large bowel":19,
    "Normal stomach":                                       20,
    "Pylorus":                                              21,
    "Resected polyps":                                      22,
    "Resection margins":                                    23,
    "Retroflex rectum":                                     24,
    "Small bowel_terminal ileum":                           25,
    "Ulcer":                                                26,
}

CLASS_PROMPTS = {
    0:  "gastroscopy photograph showing accessory tools, metal endoscopic instruments visible in circular endoscopic field, pink mucosa background, dark vignette border, specular highlights",
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

NEGATIVE_PROMPT = (
    "diagram, illustration, drawing, cartoon, animation, "
    "blue background, white background, anatomical model, "
    "textbook figure, microscopy, histology, x-ray, ct scan, "
    "blurry, low quality, watermark, text, artifacts, "
    "photograph of person, natural scene, landscape"
)
