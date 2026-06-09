"""Generate Blended backdoor notebooks (Chen et al. 2017, "Targeted Backdoor Attacks on Deep
Learning Systems Using Data Poisoning"), placed beside each dataset's data + badnets notebooks:
  GTSRB      -> gtsrb/{model}_blended_gtsrb.ipynb       (NUM_CLASSES=43, CSV test set)
  BelgiumTSC -> belgiumtsd/{model}_blended_bel.ipynb    (NUM_CLASSES=62, ImageFolder test set)

Same harness as gen_badnets_notebooks.py (model defs/freezing/LRs/epochs/batch, datasets +
NumericImageFolder per dataset, SEEDS/seeds_for/LOW_RATE_THRESHOLD=0.005 multi-seed floor,
zero-poison guard/ACTIVE_RATES, nontarget-excluded ASR, models/ checkpoint dir, CONFIGS keyed
by (model,dataset) -> 6 notebooks). The ONLY changes are the trigger (patch -> seeded
whole-image blend at opacity alpha) and the experiment structure.

v1 SCOPE: two INDEPENDENT 1-D sweeps —
  Block A: poison-rate floor at fixed alpha = BLEND_ALPHA (lines up against the BadNets floor)
  Block B: alpha sweep at fixed rate = RATE_FOR_ALPHA_SWEEP (stealth-vs-potency money plot)
  Block C: focused 2-D alpha x poison-rate grid AROUND THE FLOOR (ASR heatmaps), reusing
           any checkpoint already on disk (incl. Block B's alpha-row at rate=5%).

Run:  python gen_blended_notebooks.py
"""
import json, os

_id = 0
def _next_id():
    global _id
    _id += 1
    return f"c{_id:03d}"

def md(text):
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": text}

def code(src):
    return {"cell_type": "code", "execution_count": None, "id": _next_id(),
            "metadata": {}, "outputs": [], "source": src}

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

# ── per-model build/optimizer source (identical across datasets; ImageNet transfer) ──
RESNET_BUILD = '''def build_model(pretrained=True):
    """ResNet-50: freeze all but layer4 + fc (same as clean training notebook)."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    for name, param in model.named_parameters():
        if not (name.startswith('layer4') or name.startswith('fc')):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)

def build_optimizer(model):
    # Flat LR 1e-4 across unfrozen params (layer4 + fc)
    return torch.optim.Adam([
        {'params': model.layer4.parameters()},
        {'params': model.fc.parameters()},
    ], lr=1e-4)'''

VGG_BUILD = '''def build_model(pretrained=True):
    """VGG-16: freeze all, unfreeze conv blocks 4+5 (features[17:]) + full classifier
    (same as clean training notebook)."""
    weights = VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[17:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    return model.to(device)

def build_optimizer(model):
    # Discriminative LRs: conv 1e-5, pretrained FC 1e-4, new head 1e-3
    return torch.optim.Adam([
        {'params': model.features[17:].parameters(),          'lr': 1e-5},
        {'params': list(model.classifier[0].parameters()) +
                   list(model.classifier[3].parameters()),    'lr': 1e-4},
        {'params': model.classifier[6].parameters(),          'lr': 1e-3},
    ])'''

MOBILENET_BUILD = '''def build_model(pretrained=True):
    """MobileNetV3-Large: freeze all, unfreeze last 5 feature blocks (12:) + full
    classifier (same as clean training notebook)."""
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    for block in model.features[12:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    return model.to(device)

def build_optimizer(model):
    # Discriminative LRs: features[12:] 2e-5, classifier[0] 5e-5, new head 5e-4
    return torch.optim.Adam([
        {'params': [p for b in model.features[12:] for p in b.parameters()], 'lr': 2e-5},
        {'params': model.classifier[0].parameters(), 'lr': 5e-5},
        {'params': model.classifier[3].parameters(), 'lr': 5e-4},
    ])'''

# ── per-model meta (shared across datasets) ──────────────────────────────────────
MODELS = {
    'resnet50':    dict(title='ResNet-50',          nb_stem='resnet',      weights_import='from torchvision.models import ResNet50_Weights',          batch_size=64, build_src=RESNET_BUILD),
    'vgg16':       dict(title='VGG-16',             nb_stem='vgg16',       weights_import='from torchvision.models import VGG16_Weights',             batch_size=32, build_src=VGG_BUILD),
    'mobilenetv3': dict(title='MobileNetV3-Large',  nb_stem='mobilenetv3', weights_import='from torchvision.models import MobileNet_V3_Large_Weights', batch_size=64, build_src=MOBILENET_BUILD),
}

# ── dataset-specific source pieces (identical to gen_badnets_notebooks.py) ────────
GTSRB_NUMERIC = (
    "class NumericImageFolder(torchvision.datasets.ImageFolder):\n"
    "    \"\"\"ImageFolder that sorts class folders by integer value, not alphabetically.\n"
    "    Default alphabetical sort maps folder '10' -> index 2 instead of 10,\n"
    "    which misaligns with the integer ClassId values in Test.csv.\"\"\"\n"
    "    def find_classes(self, directory):\n"
    "        classes = sorted(os.listdir(directory), key=lambda x: int(x))\n"
    "        class_to_idx = {cls: int(cls) for cls in classes}\n"
    "        return classes, class_to_idx\n"
    "\n"
    "\n"
    "class GTSRBTestDataset(Dataset):\n"
    "    def __init__(self, csv_path, root_dir, transform=None):\n"
    "        self.df = pd.read_csv(csv_path)\n"
    "        self.root_dir = root_dir\n"
    "        self.transform = transform\n"
    "    def __len__(self):\n"
    "        return len(self.df)\n"
    "    def __getitem__(self, idx):\n"
    "        row = self.df.iloc[idx]\n"
    "        img_path = os.path.join(self.root_dir, row['Path'])\n"
    "        image = Image.open(img_path).convert('RGB')\n"
    "        label = int(row['ClassId'])\n"
    "        if self.transform:\n"
    "            image = self.transform(image)\n"
    "        return image, label\n"
)

BEL_NUMERIC = (
    "class NumericImageFolder(torchvision.datasets.ImageFolder):\n"
    "    \"\"\"Sorts class folders by integer value. BelgiumTSC folders are zero-padded\n"
    "    '00000'..'00061'; os.scandir + is_dir skips non-directory entries (Readme.txt).\"\"\"\n"
    "    def find_classes(self, directory):\n"
    "        classes = sorted(\n"
    "            (e.name for e in os.scandir(directory) if e.is_dir()),\n"
    "            key=lambda x: int(x)\n"
    "        )\n"
    "        class_to_idx = {cls: int(cls) for cls in classes}\n"
    "        return classes, class_to_idx\n"
)

GTSRB_LOAD = (
    "\n"
    "full_train_dataset = NumericImageFolder(TRAIN_DIR, transform=train_transform_01)\n"
    "assert full_train_dataset.class_to_idx['10'] == 10, 'Label mapping is wrong!'\n"
    "print(f\"Label mapping check passed: class '10' -> index {full_train_dataset.class_to_idx['10']}\")\n"
    "\n"
    "n_total = len(full_train_dataset)\n"
    "n_val   = int(n_total * VAL_SPLIT)\n"
    "n_train = n_total - n_val\n"
    "train_split, val_split = random_split(\n"
    "    full_train_dataset, [n_train, n_val],\n"
    "    generator=torch.Generator().manual_seed(SEED)\n"
    ")\n"
    "# Val split uses the no-augmentation [0,1] transform (deterministic, like the clean notebook).\n"
    "val_split.dataset = NumericImageFolder(TRAIN_DIR, transform=val_test_transform_01)\n"
    "test_split = GTSRBTestDataset(TEST_CSV, TEST_DIR, transform=val_test_transform_01)\n"
)

BEL_LOAD = (
    "\n"
    "full_train_dataset = NumericImageFolder(TRAIN_DIR, transform=train_transform_01)\n"
    "# Bel folders are zero-padded, so '10' is not a key — use the zero-padded key.\n"
    "assert full_train_dataset.class_to_idx['00010'] == 10, 'Label mapping is wrong!'\n"
    "print(f\"Label mapping check passed: class '00010' -> index {full_train_dataset.class_to_idx['00010']}\")\n"
    "\n"
    "n_total = len(full_train_dataset)\n"
    "n_val   = int(n_total * VAL_SPLIT)\n"
    "n_train = n_total - n_val\n"
    "train_split, val_split = random_split(\n"
    "    full_train_dataset, [n_train, n_val],\n"
    "    generator=torch.Generator().manual_seed(SEED)\n"
    ")\n"
    "# Val split + test set use the bel NumericImageFolder with allow_empty=True (some class\n"
    "# folders can be empty in a split / in Testing), mirroring the clean bel + combined notebooks.\n"
    "val_split.dataset = NumericImageFolder(TRAIN_DIR, transform=val_test_transform_01, allow_empty=True)\n"
    "test_split = NumericImageFolder(TEST_DIR, transform=val_test_transform_01, allow_empty=True)\n"
)

GTSRB_DATAVARS = (
    "DATA_DIR   = 'dataset'\n"
    "TRAIN_DIR  = os.path.join(DATA_DIR, 'Train')\n"
    "TEST_CSV   = os.path.join(DATA_DIR, 'Test.csv')\n"
    "TEST_DIR   = DATA_DIR\n"
)

BEL_DATAVARS = (
    "DATA_DIR   = 'dataset'\n"
    "# NOTE: confirm TRAIN_DIR matches your clean bel training notebook's actual path.\n"
    "TRAIN_DIR  = os.path.join(DATA_DIR, 'BelgiumTSC_Training', 'Training')\n"
    "TEST_DIR   = os.path.join(DATA_DIR, 'BelgiumTSC_Testing', 'Testing')\n"
)

GTSRB_DATASETS_MD = (
    "## Datasets — reused `NumericImageFolder` / `GTSRBTestDataset`\n"
    "Same correct integer label mapping (`'10' -> 10`, matching `Test.csv` ClassId) and same "
    "80/20 train/val split (seed=42) as the clean training notebook. The only change: these "
    "carry the **`[0,1]`-space** transforms (normalize is added later by the wrappers)."
)

BEL_DATASETS_MD = (
    "## Datasets — reused bel `NumericImageFolder` (ImageFolder test set, no CSV)\n"
    "Same loader as the clean bel training notebook: folders are zero-padded `'00000'..'00061'`, "
    "so the mapping check uses `'00010' -> 10`. The test set is a **separate `Testing/` ImageFolder** "
    "(not a CSV), loaded with `allow_empty=True`. Same 80/20 train/val split (seed=42) — on bel this "
    "reproduces ~3660 train / ~915 val, with the separate Testing folder (~2520) as the clean test "
    "set. These carry the **`[0,1]`-space** transforms (normalize is added later by the wrappers)."
)

# Each dataset's notebooks live alongside its data + badnets notebooks (gtsrb/ and belgiumtsd/);
# filenames carry the dataset tag (_gtsrb / _bel) so nothing collides.
DATASETS = {
    'gtsrb': dict(ds_title='GTSRB', ds_short='gtsrb', ckpt_tag='', num_classes=43,
                  out_subdir='gtsrb', nb_suffix='_blended_gtsrb.ipynb',
                  epochs=dict(resnet50=15, vgg16=20, mobilenetv3=20),
                  numeric=GTSRB_NUMERIC, load=GTSRB_LOAD, datavars=GTSRB_DATAVARS,
                  datasets_md=GTSRB_DATASETS_MD),
    'bel':   dict(ds_title='BelgiumTSC', ds_short='bel', ckpt_tag='_bel', num_classes=62,
                  out_subdir='belgiumtsd', nb_suffix='_blended_bel.ipynb',
                  epochs=dict(resnet50=30, vgg16=30, mobilenetv3=30),
                  numeric=BEL_NUMERIC, load=BEL_LOAD, datavars=BEL_DATAVARS,
                  datasets_md=BEL_DATASETS_MD),
}

# ── assemble (model x dataset) configs ──────────────────────────────────────────
# epochs: GTSRB matches its clean notebooks (ResNet 15, VGG 20, MobileNet 20); BelgiumTSC
# matches its clean notebooks (all 30). T_max for the cosine schedule == NUM_EPOCHS per model.
CONFIGS = []
for _ds_key, _ds in DATASETS.items():
    for _mk, _m in MODELS.items():
        _cfg = dict(model_key=_mk, num_epochs=_ds['epochs'][_mk])
        _cfg.update(_m)
        _cfg.update({k: v for k, v in _ds.items() if k != 'epochs'})
        _cfg['nb_name'] = _cfg['nb_stem'] + _ds['nb_suffix']
        CONFIGS.append(_cfg)


def build_cells(cfg):
    model_key = cfg['model_key']
    cells = []

    # ── Title / threat-model header ──────────────────────────────────────────
    cells.append(md(
        f"# Blended Backdoor Attack — {cfg['title']} ({cfg['ds_title']})\n"
        "\n"
        "**Threat model: data-poisoning / supply-chain** (untrusted pretrained models or "
        "tampered datasets). An attacker who injects a small fraction of poisoned samples installs "
        "a hidden backdoor: the model behaves normally on clean inputs but flips to a chosen target "
        "class whenever the trigger is present.\n"
        "\n"
        "**Trigger = Blended (Chen et al. 2017):** a fixed, seeded, **whole-image random-noise "
        "pattern** convex-blended into the image at opacity `alpha` (`x' = (1-a)x + a*k`). This is "
        "the *global* counterpart to the *localized* BadNets 24px patch (Gu et al. 2017) — the two "
        "together give the benchmark both a localized and a global backdoor.\n"
        "\n"
        "This is a **training-time** threat, distinct from the evasion attacks (FGSM/PGD/AutoAttack). "
        "**Scope: Blended only, attacks-only (defenses are future work).**\n"
        "\n"
        "**The two metrics:**\n"
        "- **Clean Accuracy (CA)** — accuracy on the un-triggered test set (measures *stealth* of the "
        "backdoor's effect on benign behavior).\n"
        "- **Attack Success Rate (ASR)** — fraction of *triggered* test images (of non-target classes) "
        "classified as the target class. At 0% poisoning ASR should be near-chance — the control "
        "proving the trigger only works *because of poisoning*."
    ))

    # ── v1 scope cell ─────────────────────────────────────────────────────────
    cells.append(md(
        "## Experimental scope — v1\n"
        "**v1 (current): two INDEPENDENT 1-D sweeps** — Block A varies poison rate at fixed alpha; "
        "Block B varies alpha at fixed poison rate. These locate each floor separately but assume the "
        "two axes don't interact. **v2 (future): a full 2-D alpha × poison-rate grid (ASR heatmap)** to "
        "capture interaction and find joint (rate, alpha) operating points — the two 1-D sweeps cannot "
        "establish joint optima. **Block C (below)** implements a focused version of this: a small 2-D "
        "grid over just the floor corner where ASR actually varies."
    ))

    # ── Cell 0: config + imports + device ────────────────────────────────────
    cells.append(md("## Configuration & imports"))
    cells.append(code(
        "import torch\n"
        "import torch.nn as nn\n"
        "import torchvision\n"
        "from torchvision import transforms, models\n"
        f"{cfg['weights_import']}\n"
        "from torch.utils.data import DataLoader, Dataset, random_split\n"
        "from torch.optim.lr_scheduler import CosineAnnealingLR\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "import os, json, random\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "# ── Blended configuration (cell 0) ──────────────────────────────────────\n"
        f"NUM_CLASSES   = {cfg['num_classes']}\n"
        "TARGET_LABEL  = 0            # all-to-one: triggered images -> class 0 (configurable)\n"
        "POISON_RATES  = [0.0, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05, 0.10, 0.20]   # EXACT BadNets grid (A-vs-BadNets comparability)\n"
        "\n"
        "IMG_SIZE      = 224\n"
        "TRIGGER_SEED  = 1234         # FIXED FOREVER — defines the trigger pattern. NOT a swept\n"
        "                             # variable; one fixed key across every model/rate/alpha/seed.\n"
        "BLEND_ALPHA          = 0.2   # fixed alpha for Block A's rate sweep. 0.2 is Chen et al.'s\n"
        "                             # value for the RANDOM-noise pattern and the modern GTSRB-\n"
        "                             # backdoor-lit standard; high enough that opacity is not the\n"
        "                             # bottleneck so poison RATE is the only variable.\n"
        "BLEND_ALPHAS         = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]   # Block B alpha\n"
        "                             # grid, extended DOWN to find the ASR floor (ASR already saturated\n"
        "                             # ~98% by 0.02, so the floor is below it). Spans near-invisible\n"
        "                             # (0.0025) -> Chen Hello-Kitty (0.02) -> random-noise (0.2) ->\n"
        "                             # clearly visible (0.3).\n"
        "RATE_FOR_ALPHA_SWEEP = 0.05  # fixed 5% poison for Block B — safely above the BadNets\n"
        "                             # reliability floor so poison count is not the bottleneck and\n"
        "                             # alpha is the only variable.\n"
        "\n"
        "SEED          = 42\n"
        "SEEDS         = [42, 123, 7]    # multi-seed averaging for the noisy low-rate floor\n"
        "LOW_RATE_THRESHOLD = 0.005      # rates <= this are noisy (few poisoned imgs) -> run once per\n"
        "                                # seed; which images get picked matters, so we average. Higher\n"
        "                                # rates (and rate 0 control) run single-seed with seed 42 only.\n"
        "\n"
        f"MODEL_NAME  = '{model_key}'\n"
        f"MODEL_TITLE = '{cfg['title']}'\n"
        f"DS_TITLE    = '{cfg['ds_title']}'\n"
        f"DS_SHORT    = '{cfg['ds_short']}'\n"
        f"CKPT_TAG    = '{cfg['ckpt_tag']}'   # checkpoint name infix ('' for gtsrb, '_bel' for bel)\n"
        f"BATCH_SIZE  = {cfg['batch_size']}\n"
        f"NUM_EPOCHS  = {cfg['num_epochs']}\n"
        "MODELS_DIR  = 'models'   # all trained checkpoints (.pth) are saved here\n"
        "\n"
        + cfg['datavars'] +
        "VAL_SPLIT  = 0.2\n"
        "\n"
        "if torch.cuda.is_available():\n"
        "    device = torch.device('cuda')\n"
        "elif torch.backends.mps.is_available():\n"
        "    device = torch.device('mps')\n"
        "else:\n"
        "    device = torch.device('cpu')\n"
        "\n"
        "print(f'Using device: {device}')\n"
        "print(f'{MODEL_TITLE} | {DS_TITLE} | classes={NUM_CLASSES} | epochs={NUM_EPOCHS} | batch={BATCH_SIZE}')\n"
        "print(f'Blended: alpha_A={BLEND_ALPHA}, alpha grid={BLEND_ALPHAS}, rate_B={RATE_FOR_ALPHA_SWEEP}, trigger_seed={TRIGGER_SEED}')\n"
        "os.makedirs(MODELS_DIR, exist_ok=True)\n"
        "torch.manual_seed(SEED)\n"
        "np.random.seed(SEED)\n"
        "random.seed(SEED)"
    ))

    # ── Transforms ─────────────────────────────────────────────────────────────
    cells.append(md(
        "## Transforms — split so the trigger lives in pixel `[0,1]` space\n"
        "Same augmentation/resize as the clean training notebook, but the final `Normalize` is "
        "**factored out**. The dataset wrappers produce a `[0,1]` tensor, blend the trigger there "
        "(Chen et al. blend in pixel space), and normalize **last**."
    ))
    cells.append(code(
        "IMAGENET_MEAN = [0.485, 0.456, 0.406]\n"
        "IMAGENET_STD  = [0.229, 0.224, 0.225]\n"
        "\n"
        "# Final normalization, applied AFTER the trigger is blended (see Part 1/2).\n"
        "normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)\n"
        "\n"
        "# [0,1]-space transforms: identical to the clean notebook MINUS the trailing Normalize.\n"
        "train_transform_01 = transforms.Compose([\n"
        "    transforms.Resize((224, 224)),\n"
        "    transforms.RandomRotation(15),\n"
        "    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),\n"
        "    transforms.ToTensor(),\n"
        "])\n"
        "val_test_transform_01 = transforms.Compose([\n"
        "    transforms.Resize((224, 224)),\n"
        "    transforms.ToTensor(),\n"
        "])\n"
        "\n"
        "# De-normalize helper (for visualization / metrics that need [0,1]).\n"
        "inv_normalize = transforms.Normalize(\n"
        "    mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],\n"
        "    std=[1/s for s in IMAGENET_STD]\n"
        ")\n"
        "print('Transforms defined (trigger blended in [0,1] pixel space, normalize applied last).')"
    ))

    # ── Datasets ──────────────────────────────────────────────────────────────
    cells.append(md(cfg['datasets_md']))
    cells.append(code(
        cfg['numeric']
        + cfg['load']
        + "\n"
        "print(f'Train: {n_train} | Val: {n_val} | Test: {len(test_split)}')\n"
        "print('NOTE: poison rates are applied to the {} training images (the 80% split that is '\n"
        "      'actually trained on), matching the clean notebooks. The val split stays clean for '\n"
        "      'honest checkpoint selection.'.format(n_train))\n"
        "\n"
        "# Zero-poison-image guard (BOTH datasets): at the lowest rates round(rate*n_train) can be 0\n"
        "# (especially on the small bel train set, ~3660 imgs). A rate>0 that poisons 0 images is\n"
        "# identical to the clean control and would waste seed-runs, so we drop it here. The control\n"
        "# (rate 0) is always kept. ACTIVE_RATES is what Block A iterates over.\n"
        "def n_poison_for(rate):\n"
        "    return int(round(rate * n_train))\n"
        "\n"
        "ACTIVE_RATES = []\n"
        "print(f'\\nPoison-rate plan (n_train={n_train}):')\n"
        "for r in POISON_RATES:\n"
        "    npois = n_poison_for(r)\n"
        "    if r > 0 and npois == 0:\n"
        "        print(f'  SKIP  p={r*100:.4g}% -> {npois} poisoned imgs (rounds to 0; identical to control)')\n"
        "        continue\n"
        "    print(f'  keep  p={r*100:.4g}% -> {npois} poisoned imgs')\n"
        "    ACTIVE_RATES.append(r)\n"
        "print(f'Block B fixed rate {RATE_FOR_ALPHA_SWEEP*100:.4g}% -> {n_poison_for(RATE_FOR_ALPHA_SWEEP)} poisoned imgs')"
    ))

    # ── Part 1: blended trigger ────────────────────────────────────────────────
    cells.append(md(
        "## Part 1 — Trigger function (the Blended pattern)\n"
        "Chen et al. 2017: a **fixed, seeded, full-image random-noise** pattern `k`, convex-blended "
        "into the image at opacity `alpha`: `x' = (1-alpha)*x + alpha*k`, in `[0,1]` pixel space **before** "
        "normalization. `TRIGGER_SEED` is fixed forever — one trigger key shared across every model / "
        "rate / alpha / seed; it is **not** a swept variable. `alpha` *is* swept (Block B). Both operands "
        "are in `[0,1]` so the blend stays in `[0,1]` — no clamp needed."
    ))
    cells.append(code(
        "# Fixed, seeded whole-image trigger pattern k (Chen et al. 2017). Defined ONCE from\n"
        "# TRIGGER_SEED — identical across every model/rate/alpha/seed.\n"
        "_tg = torch.Generator().manual_seed(TRIGGER_SEED)\n"
        "TRIGGER_PATTERN = torch.rand(3, IMG_SIZE, IMG_SIZE, generator=_tg)   # [0,1] CHW, fixed\n"
        "\n"
        "def apply_trigger(img, alpha):\n"
        "    # Convex blend in [0,1] space BEFORE normalize (Chen et al.: x' = (1-a)x + a*k).\n"
        "    # Both operands in [0,1] so output is in [0,1] — no clamp needed.\n"
        "    return (1.0 - alpha) * img + alpha * TRIGGER_PATTERN\n"
        "\n"
        "# Quick preview at the Block-A alpha.\n"
        "_img0, _ = test_split[0]\n"
        "_trig0 = apply_trigger(_img0, BLEND_ALPHA)\n"
        "fig, ax = plt.subplots(1, 3, figsize=(9, 3))\n"
        "ax[0].imshow(_img0.permute(1, 2, 0).numpy());          ax[0].set_title('clean [0,1]'); ax[0].axis('off')\n"
        "ax[1].imshow(TRIGGER_PATTERN.permute(1, 2, 0).numpy()); ax[1].set_title(f'trigger k (seed {TRIGGER_SEED})'); ax[1].axis('off')\n"
        "ax[2].imshow(_trig0.permute(1, 2, 0).numpy());          ax[2].set_title(f'blended (alpha={BLEND_ALPHA})'); ax[2].axis('off')\n"
        "plt.tight_layout(); plt.show()\n"
        "print('apply_trigger ready — whole-image blend in [0,1] pixel space, normalize applied afterward.')"
    ))

    # ── Part 2: PoisonedDataset (alpha-threaded) ───────────────────────────────
    cells.append(md(
        "## Part 2 — Poisoned training dataset (alpha-threaded)\n"
        "`PoisonedDataset` blends the trigger at `alpha` on a reproducible `p`-fraction of indices and "
        "relabels them to `TARGET_LABEL`. The triggered **test** set used to score a checkpoint MUST use "
        "the **same alpha the model was trained at** (v1 keeps `alpha_train == alpha_test`). Chen et al. "
        "decouple train/test opacity; a separate `alpha_test` is a **v2** extension."
    ))
    cells.append(code(
        "class PoisonedDataset(Dataset):\n"
        "    \"\"\"Wrap a clean [0,1]-space dataset. A reproducible p-fraction of samples get the\n"
        "    trigger blended at `alpha` AND relabeled to target_label; rest stay clean. Normalized last.\"\"\"\n"
        "    def __init__(self, base_dataset, poison_rate, alpha, target_label=TARGET_LABEL,\n"
        "                 normalize_tf=None, seed=SEED, verbose=True):\n"
        "        self.base = base_dataset\n"
        "        self.alpha = alpha\n"
        "        self.target_label = target_label\n"
        "        self.normalize = normalize if normalize_tf is None else normalize_tf\n"
        "        n = len(base_dataset)\n"
        "        n_poison = int(round(poison_rate * n))\n"
        "        g = torch.Generator().manual_seed(seed)\n"
        "        perm = torch.randperm(n, generator=g)\n"
        "        self.poison_idx = set(perm[:n_poison].tolist())\n"
        "        self.n_poison = n_poison\n"
        "        if verbose:\n"
        "            print(f'Poisoned {n_poison} / {n} training images ({poison_rate*100:.4g}%) at alpha={alpha}')\n"
        "\n"
        "    def __len__(self):\n"
        "        return len(self.base)\n"
        "\n"
        "    def __getitem__(self, idx):\n"
        "        img, label = self.base[idx]        # img is a [0,1] CHW tensor\n"
        "        if idx in self.poison_idx:\n"
        "            img = apply_trigger(img, self.alpha)   # blend trigger in [0,1] pixel space\n"
        "            label = self.target_label             # relabel to the attacker's target class\n"
        "        img = self.normalize(img)                 # normalize last -> what the model sees\n"
        "        return img, label\n"
        "\n"
        "\n"
        "class NormalizedTestDataset(Dataset):\n"
        "    \"\"\"Clean test wrapper: optionally blend the trigger (at `alpha`) on EVERY image, then\n"
        "    normalize. trigger=False -> clean test set (for CA); trigger=True -> fully-triggered set.\"\"\"\n"
        "    def __init__(self, base_dataset, trigger=False, alpha=None, normalize_tf=None):\n"
        "        self.base = base_dataset\n"
        "        self.trigger = trigger\n"
        "        self.alpha = alpha\n"
        "        self.normalize = normalize if normalize_tf is None else normalize_tf\n"
        "\n"
        "    def __len__(self):\n"
        "        return len(self.base)\n"
        "\n"
        "    def __getitem__(self, idx):\n"
        "        img, label = self.base[idx]\n"
        "        if self.trigger:\n"
        "            img = apply_trigger(img, self.alpha)\n"
        "        return self.normalize(img), label\n"
        "\n"
        "# Clean test + clean val loaders are alpha-independent -> built once.\n"
        "clean_test_loader = DataLoader(NormalizedTestDataset(test_split, trigger=False),\n"
        "                               batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n"
        "val_loader = DataLoader(NormalizedTestDataset(val_split, trigger=False),\n"
        "                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n"
        "\n"
        "def make_trig_test_loader(alpha):\n"
        "    # Triggered test set at a specific alpha. v1: alpha_test == alpha_train (passed in by the\n"
        "    # evaluator). Chen et al. decouple them; a separate alpha_test grid is a v2 extension.\n"
        "    return DataLoader(NormalizedTestDataset(test_split, trigger=True, alpha=alpha),\n"
        "                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n"
        "print('PoisonedDataset + clean/val loaders + make_trig_test_loader(alpha) ready.')"
    ))

    # ── Model builders ────────────────────────────────────────────────────────
    cells.append(md(
        f"## Model — {cfg['title']} (same architecture / freezing / discriminative LRs as the clean notebook)\n"
        "`build_model()` / `build_optimizer()` are factored so we train a **fresh model from scratch** for "
        "each run (genuine backdoor behavior). For evaluation we rebuild with `pretrained=False` and load "
        "the saved state dict."
    ))
    cells.append(code(cfg['build_src'] + "\n\nprint('build_model / build_optimizer ready.')"))

    # ── Part 3: training helper + checkpoint resolution ───────────────────────
    cells.append(md(
        "## Part 3 — Training helper (carries alpha)\n"
        "`train_one(poison_rate, alpha, seed)` trains a fresh model and saves the best-clean-val "
        "checkpoint to `models/blended_{model}{CKPT_TAG}_a{alpha}_p{rate}_s{seed}.pth`. `resolve_ckpt` "
        "reuses an existing checkpoint (models/ then cwd) so reruns only train what's missing. "
        "(No BadNets-era non-seeded legacy name — no prior blended checkpoints exist.)"
    ))
    cells.append(code(
        "def seeds_for(rate):\n"
        "    # Low rates (>0 and <= threshold) are noisy -> run once per seed in SEEDS.\n"
        "    # Higher rates and the rate-0 control run single-seed with seed 42 only.\n"
        "    return SEEDS if (0 < rate <= LOW_RATE_THRESHOLD) else [SEED]\n"
        "\n"
        "def ckpt_path(alpha, rate, seed):\n"
        "    # Canonical save location for new checkpoints: the models/ subfolder. Carries alpha.\n"
        "    return os.path.join(MODELS_DIR, f'blended_{MODEL_NAME}{CKPT_TAG}_a{alpha}_p{rate}_s{seed}.pth')\n"
        "\n"
        "def resolve_ckpt(alpha, rate, seed):\n"
        "    # Reuse an existing checkpoint if present: models/ first, then legacy cwd. Falls back to\n"
        "    # the models/ save path. (No non-seeded legacy fallback — blended is a fresh family.)\n"
        "    name = f'blended_{MODEL_NAME}{CKPT_TAG}_a{alpha}_p{rate}_s{seed}.pth'\n"
        "    for c in (os.path.join(MODELS_DIR, name), name):\n"
        "        if os.path.exists(c):\n"
        "            return c\n"
        "    return os.path.join(MODELS_DIR, name)\n"
        "\n"
        "def train_one(poison_rate, alpha, seed=SEED):\n"
        "    # `seed` drives BOTH the poison-index draw AND init/shuffle order -> independent draw.\n"
        "    torch.manual_seed(seed)\n"
        "    np.random.seed(seed); random.seed(seed)\n"
        "    poisoned_train = PoisonedDataset(train_split, poison_rate, alpha, seed=seed)\n"
        "    train_loader = DataLoader(poisoned_train, batch_size=BATCH_SIZE, shuffle=True,\n"
        "                              num_workers=0, pin_memory=True)\n"
        "    model = build_model(pretrained=True)\n"
        "    optimizer = build_optimizer(model)\n"
        "    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)\n"
        "    criterion = nn.CrossEntropyLoss()\n"
        "    ckpt = ckpt_path(alpha, poison_rate, seed)   # saved under models/\n"
        "    best_val_acc = 0.0\n"
        "    for epoch in range(1, NUM_EPOCHS + 1):\n"
        "        model.train()\n"
        "        run_loss, correct, total = 0.0, 0, 0\n"
        "        for imgs, labels in train_loader:\n"
        "            imgs, labels = imgs.to(device), labels.to(device)\n"
        "            optimizer.zero_grad()\n"
        "            outputs = model(imgs)\n"
        "            loss = criterion(outputs, labels)\n"
        "            loss.backward()\n"
        "            optimizer.step()\n"
        "            run_loss += loss.item() * imgs.size(0)\n"
        "            correct += (outputs.argmax(1) == labels).sum().item()\n"
        "            total += imgs.size(0)\n"
        "        train_loss, train_acc = run_loss / total, correct / total\n"
        "\n"
        "        model.eval()\n"
        "        run_loss, correct, total = 0.0, 0, 0\n"
        "        with torch.no_grad():\n"
        "            for imgs, labels in val_loader:\n"
        "                imgs, labels = imgs.to(device), labels.to(device)\n"
        "                outputs = model(imgs)\n"
        "                loss = criterion(outputs, labels)\n"
        "                run_loss += loss.item() * imgs.size(0)\n"
        "                correct += (outputs.argmax(1) == labels).sum().item()\n"
        "                total += imgs.size(0)\n"
        "        val_loss, val_acc = run_loss / total, correct / total\n"
        "        scheduler.step()\n"
        "        saved = val_acc > best_val_acc\n"
        "        if saved:\n"
        "            best_val_acc = val_acc\n"
        "            torch.save(model.state_dict(), ckpt)\n"
        "        print(f'  [a{alpha} p={poison_rate*100:.3f}% s{seed}] Epoch {epoch:02d}/{NUM_EPOCHS} | '\n"
        "              f'Train Loss {train_loss:.4f} Acc {train_acc:.4f} | '\n"
        "              f'Val(clean) Loss {val_loss:.4f} Acc {val_acc:.4f}'\n"
        "              + (' *** saved' if saved else ''))\n"
        "    print(f'  -> best clean-val acc {best_val_acc:.4f}, saved {ckpt}')\n"
        "    return ckpt\n"
        "\n"
        "@torch.no_grad()\n"
        "def evaluate_checkpoint(ckpt, alpha):\n"
        "    # CA on the clean test set; ASR on the triggered@alpha test set (alpha_test == alpha_train).\n"
        "    model = build_model(pretrained=False)\n"
        "    model.load_state_dict(torch.load(ckpt, map_location=device))\n"
        "    model.eval()\n"
        "    correct, total = 0, 0\n"
        "    for imgs, labels in clean_test_loader:\n"
        "        imgs, labels = imgs.to(device), labels.to(device)\n"
        "        correct += (model(imgs).argmax(1) == labels).sum().item()\n"
        "        total += labels.size(0)\n"
        "    clean_acc = correct / total\n"
        "    # ASR — exclude true-target images from the denominator (same as BadNets).\n"
        "    hit, denom = 0, 0\n"
        "    for imgs, labels in make_trig_test_loader(alpha):\n"
        "        imgs, labels = imgs.to(device), labels.to(device)\n"
        "        preds = model(imgs).argmax(1)\n"
        "        nontarget = labels != TARGET_LABEL\n"
        "        hit   += ((preds == TARGET_LABEL) & nontarget).sum().item()\n"
        "        denom += nontarget.sum().item()\n"
        "    asr = hit / denom\n"
        "    return clean_acc, asr\n"
        "\n"
        "print('train_one / ckpt_path / resolve_ckpt / evaluate_checkpoint ready.')"
    ))

    # ══ BLOCK A ════════════════════════════════════════════════════════════════
    cells.append(md(
        "## Block A — poison-rate floor at fixed alpha = `BLEND_ALPHA`\n"
        "Structurally identical to the BadNets sweep: iterate `ACTIVE_RATES × seeds_for(rate)` with "
        "alpha pinned to `BLEND_ALPHA`, multi-seed at the low rates. This is the curve that lines up "
        "directly against the BadNets poison-rate floor."
    ))
    cells.append(code(
        "ALPHA_A = BLEND_ALPHA\n"
        "checkpoints_A = {}   # (rate, seed) -> path  (alpha fixed at ALPHA_A)\n"
        "for rate in ACTIVE_RATES:\n"
        "    for seed in seeds_for(rate):\n"
        "        existing = resolve_ckpt(ALPHA_A, rate, seed)\n"
        "        if os.path.exists(existing):\n"
        "            print(f'\\n===== A a{ALPHA_A} p={rate*100:.3f}% s{seed} — exists, skipping ({existing}) =====')\n"
        "            checkpoints_A[(rate, seed)] = existing\n"
        "            continue\n"
        "        print(f'\\n===== Training A a{ALPHA_A} p={rate*100:.3f}% s{seed} =====')\n"
        "        checkpoints_A[(rate, seed)] = train_one(rate, ALPHA_A, seed)\n"
        "print('\\nBlock A trainings done:', len(checkpoints_A), 'checkpoints')"
    ))
    cells.append(code(
        "# Reconstruct checkpoint map if running eval without the training cell.\n"
        "if 'checkpoints_A' not in dir():\n"
        "    checkpoints_A = {(r, s): resolve_ckpt(BLEND_ALPHA, r, s) for r in ACTIVE_RATES for s in seeds_for(r)}\n"
        "\n"
        "results_A = {}   # rate -> [ {'seed':s, 'clean_acc':ca, 'asr':asr}, ... ]\n"
        "for rate in ACTIVE_RATES:\n"
        "    runs = []\n"
        "    for seed in seeds_for(rate):\n"
        "        ca, asr = evaluate_checkpoint(checkpoints_A[(rate, seed)], BLEND_ALPHA)\n"
        "        runs.append({'seed': seed, 'clean_acc': ca, 'asr': asr})\n"
        "        print(f'A p={rate*100:>7.3f}% s{seed}  CA={ca*100:6.2f}%  ASR={asr*100:6.2f}%')\n"
        "    results_A[rate] = runs"
    ))

    # ── Block A table ───────────────────────────────────────────────────────────
    cells.append(md("### Block A — results table"))
    cells.append(code(
        "# Aggregation helpers over the per-seed Block-A runs.\n"
        "def mean_ca(rate):  return float(np.mean([x['clean_acc'] for x in results_A[rate]]))\n"
        "def std_ca(rate):   return float(np.std([x['clean_acc'] for x in results_A[rate]]))\n"
        "def mean_asr(rate): return float(np.mean([x['asr'] for x in results_A[rate]]))\n"
        "def std_asr(rate):  return float(np.std([x['asr'] for x in results_A[rate]]))\n"
        "\n"
        "def rate_label(rate):\n"
        "    if rate == 0.0:\n"
        "        return '0% (clean)'\n"
        "    return f'{rate*100:.3f}'.rstrip('0').rstrip('.') + '%'\n"
        "\n"
        "baseline_ca = mean_ca(0.0)\n"
        "print(f'{MODEL_TITLE} — Blended (alpha={BLEND_ALPHA}) on {DS_TITLE} (target class {TARGET_LABEL})')\n"
        "print('Low rates (<= {:.4g}) averaged over seeds {}; higher rates single-seed (seed {}).'\n"
        "      .format(LOW_RATE_THRESHOLD, SEEDS, SEED))\n"
        "print()\n"
        "print('Poison Rate  | Clean Acc            | ASR                  | Clean Acc Drop | Seeds')\n"
        "print('-' * 92)\n"
        "for rate in ACTIVE_RATES:\n"
        "    n = len(results_A[rate])\n"
        "    ca_m, ca_s   = mean_ca(rate) * 100,  std_ca(rate) * 100\n"
        "    asr_m, asr_s = mean_asr(rate) * 100, std_asr(rate) * 100\n"
        "    if n > 1:\n"
        "        ca_str  = f'{ca_m:.2f} +/- {ca_s:.2f}%'\n"
        "        asr_str = f'{asr_m:.2f} +/- {asr_s:.2f}%'\n"
        "    else:\n"
        "        ca_str  = f'{ca_m:.2f}%'\n"
        "        asr_str = f'{asr_m:.2f}%'\n"
        "    if rate == 0.0:\n"
        "        drop_str = 'baseline'\n"
        "    else:\n"
        "        drop = (baseline_ca - mean_ca(rate)) * 100\n"
        "        drop_str = f'-{drop:.2f} pp' if drop >= 0 else f'+{-drop:.2f} pp'\n"
        "    print(f'{rate_label(rate):<12} | {ca_str:<20} | {asr_str:<20} | {drop_str:<14} | {n}')"
    ))

    # ── Block A plot + headline ─────────────────────────────────────────────────
    cells.append(md(
        "### Block A — ASR & Clean Accuracy vs poison rate\n"
        f"Saved as `{model_key}_blended_{cfg['ds_short']}_rate_sweep.png`. Log x-axis; std error bars on "
        "the multi-seed low rates. Headline: min poison rate for ≥95% mean ASR at alpha=`BLEND_ALPHA`."
    ))
    cells.append(code(
        "rates_pct   = [r * 100 for r in ACTIVE_RATES]\n"
        "ca_pct      = [mean_ca(r) * 100 for r in ACTIVE_RATES]\n"
        "asr_pct     = [mean_asr(r) * 100 for r in ACTIVE_RATES]\n"
        "asr_std_pct = [std_asr(r) * 100 for r in ACTIVE_RATES]\n"
        "\n"
        "fig, ax1 = plt.subplots(figsize=(8, 5))\n"
        "color_asr, color_ca = 'crimson', 'steelblue'\n"
        "ax1.plot(rates_pct, asr_pct, 'o-', color=color_asr, linewidth=2, markersize=7, label='ASR (mean)')\n"
        "_multi = [i for i, r in enumerate(ACTIVE_RATES) if len(results_A[r]) > 1]\n"
        "if _multi:\n"
        "    ax1.errorbar([rates_pct[i] for i in _multi], [asr_pct[i] for i in _multi],\n"
        "                 yerr=[asr_std_pct[i] for i in _multi], fmt='none', ecolor=color_asr,\n"
        "                 capsize=4, elinewidth=1.5, label='ASR std (multi-seed)')\n"
        "ax1.set_xscale('log')\n"
        "ax1.set_xlabel('Poison rate (%)')\n"
        "ax1.set_ylabel('Attack Success Rate (%)', color=color_asr)\n"
        "ax1.tick_params(axis='y', labelcolor=color_asr)\n"
        "ax1.set_ylim(-3, 103)\n"
        "ax1.axhline(95, color=color_asr, linestyle=':', alpha=0.5, label='95% ASR')\n"
        "ax2 = ax1.twinx()\n"
        "ax2.plot(rates_pct, ca_pct, 's-', color=color_ca, linewidth=2, markersize=7, label='Clean Acc (mean)')\n"
        "ax2.set_ylabel('Clean Accuracy (%)', color=color_ca)\n"
        "ax2.tick_params(axis='y', labelcolor=color_ca)\n"
        "_lo = min(ca_pct) - 2\n"
        "ax2.set_ylim(min(_lo, baseline_ca*100 - 5), 100.5)\n"
        "lines1, labels1 = ax1.get_legend_handles_labels()\n"
        "lines2, labels2 = ax2.get_legend_handles_labels()\n"
        "ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')\n"
        "plt.title(f'{MODEL_TITLE} — Blended (alpha={BLEND_ALPHA}): ASR vs Clean Accuracy vs poison rate ({DS_TITLE})')\n"
        "ax1.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.savefig(f'{MODEL_NAME}_blended_{DS_SHORT}_rate_sweep.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Saved {MODEL_NAME}_blended_{DS_SHORT}_rate_sweep.png')\n"
        "\n"
        "# Headline: min poison rate for >=95% mean ASR at alpha=BLEND_ALPHA (clean-acc drop < 2pp).\n"
        "ASR_TARGET, CA_DROP_MAX = 0.95, 0.02\n"
        "qualifying = [r for r in ACTIVE_RATES if r != 0.0\n"
        "              and mean_asr(r) >= ASR_TARGET and (baseline_ca - mean_ca(r)) < CA_DROP_MAX]\n"
        "if qualifying:\n"
        "    min_rate_A = min(qualifying)\n"
        "    headline_A = (f'Blended (alpha={BLEND_ALPHA}) achieves {mean_asr(min_rate_A)*100:.1f}% mean ASR at just '\n"
        "                  f'{rate_label(min_rate_A)} poisoning ({(baseline_ca-mean_ca(min_rate_A))*100:+.2f} pp clean cost) '\n"
        "                  f'on {MODEL_TITLE} ({DS_TITLE}).')\n"
        "else:\n"
        "    min_rate_A = None\n"
        "    pool = [(mean_asr(r), r) for r in ACTIVE_RATES if r != 0.0]\n"
        "    top_asr, top_rate = max(pool)\n"
        "    headline_A = (f'No rate hit >=95% mean ASR within <2pp clean drop at alpha={BLEND_ALPHA} on '\n"
        "                  f'{MODEL_TITLE} ({DS_TITLE}); best {top_asr*100:.1f}% at {rate_label(top_rate)}.')\n"
        "print('HEADLINE (A):', headline_A)\n"
        "print(f'Control — mean ASR at 0% poisoning: {mean_asr(0.0)*100:.2f}% (should be near-chance).')"
    ))

    # ══ Perceptual setup (shared by Block B) ════════════════════════════════════
    cells.append(md(
        "## Perceptual metrics setup (PSNR / SSIM / LPIPS)\n"
        "Reused verbatim from the BadNets notebooks — same libs, same thresholds (PSNR > 30, SSIM > 0.95, "
        "LPIPS < 0.1), same `METRIC_N`. Computed in `[0,1]` pixel space between clean and blended images "
        "over a fixed `METRIC_N`-image sample (trigger perceptibility, measured over the sample exactly as "
        "BadNets did — not fooled-conditioned)."
    ))
    cells.append(code(
        "import subprocess, sys, math\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'lpips', 'scikit-image', '-q'], check=True)\n"
        "import lpips\n"
        "from skimage.metrics import structural_similarity as ssim_fn\n"
        "\n"
        "METRIC_N = 500   # number of test images to measure over (same as BadNets)\n"
        "lpips_fn = lpips.LPIPS(net='alex').to(device)\n"
        "\n"
        "def _psnr01(a, b):\n"
        "    mse = ((a - b) ** 2).mean()\n"
        "    return 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')\n"
        "\n"
        "_metric_rng = random.Random(SEED)\n"
        "metric_idx = _metric_rng.sample(range(len(test_split)), min(METRIC_N, len(test_split)))\n"
        "\n"
        "def perceptual_at_alpha(alpha):\n"
        "    \"\"\"Mean PSNR/SSIM/LPIPS between clean and blended@alpha over the fixed metric sample.\"\"\"\n"
        "    psnrs, ssims, lpips_vals = [], [], []\n"
        "    with torch.no_grad():\n"
        "        for idx in metric_idx:\n"
        "            img01, _ = test_split[idx]              # [0,1] CHW\n"
        "            trig01 = apply_trigger(img01, alpha)\n"
        "            o = img01.permute(1, 2, 0).numpy()\n"
        "            a = trig01.permute(1, 2, 0).numpy()\n"
        "            psnrs.append(_psnr01(o, a))\n"
        "            ssims.append(ssim_fn(o, a, channel_axis=2, data_range=1.0))\n"
        "            o11 = (img01.unsqueeze(0).to(device)  * 2 - 1)   # LPIPS expects [-1,1]\n"
        "            a11 = (trig01.unsqueeze(0).to(device) * 2 - 1)\n"
        "            lpips_vals.append(lpips_fn(o11, a11).item())\n"
        "    finite = [p for p in psnrs if math.isfinite(p)]\n"
        "    return float(np.mean(finite)), float(np.mean(ssims)), float(np.mean(lpips_vals))\n"
        "print(f'Perceptual setup ready (METRIC_N={METRIC_N}).')"
    ))

    # ══ BLOCK B ════════════════════════════════════════════════════════════════
    cells.append(md(
        "## Block B — alpha sweep at fixed rate = `RATE_FOR_ALPHA_SWEEP`\n"
        "Iterate `BLEND_ALPHAS` with the poison rate pinned, **single seed** (`SEED`) for v1. For each "
        "alpha: train-or-reuse, compute CA + ASR, and the PSNR/SSIM/LPIPS of the blended images.\n"
        "\n"
        "> **Note:** the low-alpha end (0.02, 0.05) is the marginal regime where — per our BadNets "
        "seed-variance finding — a single seed may be unreliable. Multi-seeding the low-alpha end is a "
        "known **v1.5** fix if Block B looks noisy."
    ))
    cells.append(code(
        "RATE_B = RATE_FOR_ALPHA_SWEEP\n"
        "alpha_checkpoints = {}\n"
        "for alpha in BLEND_ALPHAS:\n"
        "    existing = resolve_ckpt(alpha, RATE_B, SEED)\n"
        "    if os.path.exists(existing):\n"
        "        print(f'\\n===== B a{alpha} p={RATE_B*100:.3g}% s{SEED} — exists, skipping ({existing}) =====')\n"
        "        alpha_checkpoints[alpha] = existing\n"
        "        continue\n"
        "    print(f'\\n===== Training B a{alpha} p={RATE_B*100:.3g}% s{SEED} =====')\n"
        "    alpha_checkpoints[alpha] = train_one(RATE_B, alpha, SEED)\n"
        "print('\\nBlock B trainings done:', len(alpha_checkpoints), 'checkpoints')"
    ))
    cells.append(code(
        "if 'alpha_checkpoints' not in dir():\n"
        "    alpha_checkpoints = {a: resolve_ckpt(a, RATE_FOR_ALPHA_SWEEP, SEED) for a in BLEND_ALPHAS}\n"
        "\n"
        "alpha_results = {}   # alpha -> {clean_acc, asr, psnr, ssim, lpips}\n"
        "print(f'{MODEL_TITLE} — Blended alpha sweep at rate={RATE_FOR_ALPHA_SWEEP*100:.3g}% on {DS_TITLE}')\n"
        "print('alpha  | Clean Acc | ASR     | PSNR(dB) | SSIM   | LPIPS   (PSNR>30 SSIM>0.95 LPIPS<0.1 = imperceptible)')\n"
        "print('-' * 92)\n"
        "for alpha in BLEND_ALPHAS:\n"
        "    ca, asr = evaluate_checkpoint(alpha_checkpoints[alpha], alpha)\n"
        "    psnr, ssim, lp = perceptual_at_alpha(alpha)\n"
        "    alpha_results[alpha] = {'clean_acc': ca, 'asr': asr, 'psnr': psnr, 'ssim': ssim, 'lpips': lp}\n"
        "    print(f'{alpha:<6} | {ca*100:7.2f}% | {asr*100:6.2f}% | {psnr:7.2f}  | {ssim:.4f} | {lp:.4f}')"
    ))

    # ── Block B money plot ──────────────────────────────────────────────────────
    cells.append(md(
        "### Block B — the \"money plot\": stealth vs potency\n"
        f"x = alpha; LEFT y = ASR (potency); RIGHT y = SSIM & LPIPS (stealth). As alpha rises, ASR climbs "
        "while SSIM falls / LPIPS rises (the image looks less clean). "
        f"Saved as `{model_key}_blended_{cfg['ds_short']}_alpha_sweep.png`."
    ))
    cells.append(code(
        "alphas  = list(BLEND_ALPHAS)\n"
        "asr_b   = [alpha_results[a]['asr'] * 100 for a in alphas]\n"
        "ssim_b  = [alpha_results[a]['ssim'] for a in alphas]\n"
        "lpips_b = [alpha_results[a]['lpips'] for a in alphas]\n"
        "\n"
        "fig, ax1 = plt.subplots(figsize=(8, 5))\n"
        "c_asr, c_ssim, c_lpips = 'crimson', 'steelblue', 'seagreen'\n"
        "ax1.plot(alphas, asr_b, 'o-', color=c_asr, linewidth=2, markersize=7, label='ASR')\n"
        "ax1.axhline(95, color=c_asr, linestyle=':', alpha=0.5, label='95% ASR')\n"
        "ax1.set_xlabel('Blend alpha (trigger opacity)')\n"
        "ax1.set_ylabel('Attack Success Rate (%)', color=c_asr)\n"
        "ax1.tick_params(axis='y', labelcolor=c_asr)\n"
        "ax1.set_ylim(-3, 103)\n"
        "ax2 = ax1.twinx()\n"
        "ax2.plot(alphas, ssim_b,  's-', color=c_ssim,  linewidth=2, markersize=6, label='SSIM (stealth)')\n"
        "ax2.plot(alphas, lpips_b, '^-', color=c_lpips, linewidth=2, markersize=6, label='LPIPS (stealth)')\n"
        "ax2.axhline(0.95, color=c_ssim,  linestyle='--', alpha=0.4, label='SSIM thresh 0.95')\n"
        "ax2.axhline(0.10, color=c_lpips, linestyle='--', alpha=0.4, label='LPIPS thresh 0.1')\n"
        "ax2.set_ylabel('SSIM / LPIPS (stealth metrics)')\n"
        "ax2.set_ylim(0, 1.02)\n"
        "lines1, labels1 = ax1.get_legend_handles_labels()\n"
        "lines2, labels2 = ax2.get_legend_handles_labels()\n"
        "ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)\n"
        "plt.title(f'{MODEL_TITLE} — Blended stealth-vs-potency at rate={RATE_FOR_ALPHA_SWEEP*100:.3g}% ({DS_TITLE})')\n"
        "ax1.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.savefig(f'{MODEL_NAME}_blended_{DS_SHORT}_alpha_sweep.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Saved {MODEL_NAME}_blended_{DS_SHORT}_alpha_sweep.png')"
    ))

    # ── Part 7: perceptual framing ──────────────────────────────────────────────
    cells.append(md(
        "## Part 7 — Perceptual framing (why Blended needs SSIM/LPIPS)\n"
        "BadNets' **localized 24px patch FAILED PSNR but PASSED SSIM/LPIPS**: those global, spatially-"
        "averaged metrics ignore a tiny localized patch (~1% of a 224×224 image), so only the per-pixel "
        "PSNR flagged it. **Blended is a WHOLE-IMAGE perturbation**, so SSIM and LPIPS are now the "
        "**appropriate** stealth metrics — the same regime as the evasion attacks (FGSM/PGD/AutoAttack), "
        "where the perturbation also spans the whole image. This contrast is exactly why the benchmark "
        "needs **both** a localized backdoor (BadNets, Gu et al. 2017) and a global one (Blended, Chen et "
        "al. 2017). Stealthier triggers (WaNet) are future work."
    ))

    # ── Visualization: clean vs blended ─────────────────────────────────────────
    cells.append(md(
        "## Visualization — clean vs blended across ALL alphas\n"
        "Rows = sample non-target test images. Columns = the clean image, then the image blended at "
        "**every alpha in `BLEND_ALPHAS`**. Each blended cell is titled with its alpha and the prediction "
        "of that alpha's **Block-B backdoor model** (trained at rate=`RATE_FOR_ALPHA_SWEEP`); red `*` = "
        "flipped to the target class. This shows stealth (appearance) and potency (prediction flips) "
        "ramping together with alpha — the visual companion to the money plot."
    ))
    cells.append(code(
        "# Clean control = Block A's 0%-poison model. Per-alpha backdoor models = Block B checkpoints\n"
        "# (each trained at rate=RATE_FOR_ALPHA_SWEEP), so the prediction shown per column matches the\n"
        "# alpha used to blend that column.\n"
        "clean_model = build_model(pretrained=False)\n"
        "clean_model.load_state_dict(torch.load(checkpoints_A[(0.0, SEED)], map_location=device)); clean_model.eval()\n"
        "alpha_models = {}\n"
        "for a in BLEND_ALPHAS:\n"
        "    m = build_model(pretrained=False)\n"
        "    m.load_state_dict(torch.load(alpha_checkpoints[a], map_location=device)); m.eval()\n"
        "    alpha_models[a] = m\n"
        "\n"
        "rng = random.Random(SEED)\n"
        "cand = [i for i in range(len(test_split)) if test_split[i][1] != TARGET_LABEL]\n"
        "show_idx = rng.sample(cand, 4)\n"
        "\n"
        "ncols = 1 + len(BLEND_ALPHAS)\n"
        "fig, axes = plt.subplots(len(show_idx), ncols, figsize=(2.0 * ncols, 2.3 * len(show_idx)))\n"
        "with torch.no_grad():\n"
        "    for row, idx in enumerate(show_idx):\n"
        "        img01, true_label = test_split[idx]\n"
        "        # Clean column: clean-control prediction on the un-triggered image.\n"
        "        cp = clean_model(normalize(img01).unsqueeze(0).to(device)).argmax(1).item()\n"
        "        ax = axes[row, 0]\n"
        "        ax.imshow(img01.permute(1, 2, 0).numpy())\n"
        "        ax.set_title(f'CLEAN true={true_label}\\nclean-model={cp}', fontsize=8)\n"
        "        ax.axis('off')\n"
        "        # One column per alpha: blended image + that alpha's backdoor-model prediction.\n"
        "        for col, a in enumerate(BLEND_ALPHAS, start=1):\n"
        "            trig01 = apply_trigger(img01, a)\n"
        "            pred = alpha_models[a](normalize(trig01).unsqueeze(0).to(device)).argmax(1).item()\n"
        "            flipped = pred == TARGET_LABEL\n"
        "            ax = axes[row, col]\n"
        "            ax.imshow(trig01.permute(1, 2, 0).numpy())\n"
        "            ax.set_title(f'a={a}\\npred={pred}' + (' *' if flipped else ''),\n"
        "                         fontsize=8, color=('crimson' if flipped else 'black'))\n"
        "            ax.axis('off')\n"
        "plt.suptitle(f'{MODEL_TITLE} ({DS_TITLE}) — blended appearance + per-alpha backdoor prediction across alphas '\n"
        "             f'(rate={RATE_FOR_ALPHA_SWEEP*100:.3g}%, red * = flipped to target {TARGET_LABEL})', fontsize=11)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ══ BLOCK C — focused 2-D floor grid ════════════════════════════════════════
    cells.append(md(
        "## Block C — focused 2-D (alpha × poison-rate) grid AROUND THE FLOOR\n"
        "Blocks A and B are independent 1-D sweeps; Block C is the small 2-D grid over the transition "
        "**corner** where ASR actually varies. ASR saturates to ~100% above alpha≈0.02 and above "
        "rate≈0.5%, so that cross-product is already known — this grid deliberately covers only the "
        "floor. Cells are **seed-averaged in the noisy region** (`rate ≤ LOW_RATE_THRESHOLD` OR "
        "`alpha ≤ 0.01`) and single-seed elsewhere. Already-trained checkpoints (incl. Block B's "
        "alpha-row at rate=5%) are **reused, not retrained** — keeping `alpha_train == alpha_test` (v1)."
    ))
    cells.append(code(
        "GRID_ALPHAS = [0.005, 0.01, 0.02, 0.05]\n"
        "GRID_RATES  = [0.001, 0.0025, 0.005, 0.01, 0.05]\n"
        "# Rationale: ASR saturates ~100% above alpha=0.02 and above rate~0.5%, so the cross-product\n"
        "# there is already known. This grid covers the transition CORNER where ASR varies — the floor.\n"
        "\n"
        "def seeds_for_grid(alpha, rate):\n"
        "    # Extend the noisy-region multi-seed rule to BOTH axes: 3 seeds in the marginal region\n"
        "    # (rate <= LOW_RATE_THRESHOLD OR alpha <= 0.01), single seed (42) in the near-saturated rest.\n"
        "    return SEEDS if (rate <= LOW_RATE_THRESHOLD or alpha <= 0.01) else [SEED]\n"
        "\n"
        "# Dry-run plan: show REUSE vs TRAIN per (alpha, rate, seed) BEFORE spending any GPU.\n"
        "_n_reuse = _n_train = 0\n"
        "print('Block C grid plan (REUSE = checkpoint on disk, TRAIN = will train):')\n"
        "for alpha in GRID_ALPHAS:\n"
        "    for rate in GRID_RATES:\n"
        "        for seed in seeds_for_grid(alpha, rate):\n"
        "            existing = resolve_ckpt(alpha, rate, seed)\n"
        "            if os.path.exists(existing):\n"
        "                _n_reuse += 1\n"
        "                print(f'  REUSE a{alpha} p={rate*100:.4g}% s{seed}  ({existing})')\n"
        "            else:\n"
        "                _n_train += 1\n"
        "                print(f'  TRAIN a{alpha} p={rate*100:.4g}% s{seed}')\n"
        "_n_runs = sum(len(seeds_for_grid(a, r)) for a in GRID_ALPHAS for r in GRID_RATES)\n"
        "print(f'\\nBlock C: {len(GRID_ALPHAS)}x{len(GRID_RATES)} grid, {_n_runs} (alpha,rate,seed) runs '\n"
        "      f'-> {_n_reuse} REUSE, {_n_train} TRAIN. (Review before running the next cell.)')"
    ))
    cells.append(code(
        "# Train-or-reuse every grid cell, then evaluate. alpha_train == alpha_test (v1 convention).\n"
        "grid_results = {}   # (alpha, rate) -> {asr_mean, asr_std, ca_mean, n_seeds}\n"
        "_done_reuse = _done_train = 0\n"
        "for alpha in GRID_ALPHAS:\n"
        "    for rate in GRID_RATES:\n"
        "        seeds = seeds_for_grid(alpha, rate)\n"
        "        asrs, cas = [], []\n"
        "        for seed in seeds:\n"
        "            existing = resolve_ckpt(alpha, rate, seed)\n"
        "            if os.path.exists(existing):\n"
        "                ck = existing; _done_reuse += 1\n"
        "            else:\n"
        "                print(f'  TRAIN a{alpha} p={rate*100:.4g}% s{seed}')\n"
        "                ck = train_one(rate, alpha, seed); _done_train += 1\n"
        "            ca, asr = evaluate_checkpoint(ck, alpha)\n"
        "            asrs.append(asr); cas.append(ca)\n"
        "        grid_results[(alpha, rate)] = {'asr_mean': float(np.mean(asrs)), 'asr_std': float(np.std(asrs)),\n"
        "                                       'ca_mean': float(np.mean(cas)), 'n_seeds': len(seeds)}\n"
        "        g = grid_results[(alpha, rate)]\n"
        "        print(f'  a{alpha} p={rate*100:.4g}%  ASR={g[\"asr_mean\"]*100:5.1f}% +/- {g[\"asr_std\"]*100:4.1f}  '\n"
        "              f'CA={g[\"ca_mean\"]*100:5.2f}%  (n_seeds={g[\"n_seeds\"]})')\n"
        "print(f'\\nBlock C eval done: {_done_reuse} reused, {_done_train} trained.')"
    ))
    cells.append(md(
        "### Block C — ASR floor heatmap\n"
        "x = poison rate (log), y = alpha (log), color = mean ASR. Each cell annotated with its mean ASR "
        "(±std for multi-seed cells). Saved as "
        f"`{model_key}_blended_{cfg['ds_short']}_floor_heatmap.png`."
    ))
    cells.append(code(
        "# ASR heatmap on log-log axes; cells annotated with mean ASR (+/- std if multi-seed).\n"
        "M = np.array([[grid_results[(a, r)]['asr_mean'] * 100 for r in GRID_RATES] for a in GRID_ALPHAS])\n"
        "\n"
        "def _log_edges(vals):\n"
        "    lv = np.log10(np.array(vals, float))\n"
        "    mid = (lv[:-1] + lv[1:]) / 2\n"
        "    return 10 ** np.concatenate([[lv[0] - (mid[0] - lv[0])], mid, [lv[-1] + (lv[-1] - mid[-1])]])\n"
        "\n"
        "xe, ye = _log_edges(GRID_RATES), _log_edges(GRID_ALPHAS)\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "pcm = ax.pcolormesh(xe, ye, M, cmap='viridis', vmin=0, vmax=100, shading='flat')\n"
        "ax.set_xscale('log'); ax.set_yscale('log')\n"
        "ax.set_xticks(GRID_RATES); ax.set_xticklabels([f'{r*100:.4g}%' for r in GRID_RATES])\n"
        "ax.set_yticks(GRID_ALPHAS); ax.set_yticklabels([str(a) for a in GRID_ALPHAS])\n"
        "ax.minorticks_off()\n"
        "for a in GRID_ALPHAS:\n"
        "    for r in GRID_RATES:\n"
        "        g = grid_results[(a, r)]\n"
        "        txt = f'{g[\"asr_mean\"]*100:.1f}'\n"
        "        if g['n_seeds'] > 1: txt += f'\\n±{g[\"asr_std\"]*100:.1f}'\n"
        "        ax.text(r, a, txt, ha='center', va='center', fontsize=7,\n"
        "                color=('white' if g['asr_mean']*100 < 55 else 'black'))\n"
        "ax.set_xlabel('Poison rate (log)'); ax.set_ylabel('Blend alpha (log)')\n"
        "fig.colorbar(pcm, ax=ax, label='Mean ASR (%)')\n"
        "plt.title(f'Blended ASR floor — alpha x poison-rate ({MODEL_TITLE}/{DS_TITLE}, saturated 5%+ region excluded)')\n"
        "plt.tight_layout()\n"
        "plt.savefig(f'{MODEL_NAME}_blended_{DS_SHORT}_floor_heatmap.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Saved {MODEL_NAME}_blended_{DS_SHORT}_floor_heatmap.png')"
    ))
    cells.append(md(
        "### Block C — stealthy-AND-potent heatmap\n"
        "Same mean-ASR color, but cells whose trigger is **imperceptible (SSIM > 0.95)** are circled. "
        "SSIM depends only on alpha (constant down each alpha row), computed via the existing "
        "`perceptual_at_alpha`. The circled high-ASR cells are the region that is simultaneously "
        f"imperceptible and reliable. Saved as `{model_key}_blended_{cfg['ds_short']}_floor_stealth_heatmap.png`."
    ))
    cells.append(code(
        "# SSIM depends only on alpha -> compute once per alpha row (reuse perceptual_at_alpha).\n"
        "ssim_by_alpha = {a: perceptual_at_alpha(a)[1] for a in GRID_ALPHAS}\n"
        "print('SSIM by alpha (constant across rate):', {a: round(s, 4) for a, s in ssim_by_alpha.items()})\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "pcm = ax.pcolormesh(xe, ye, M, cmap='viridis', vmin=0, vmax=100, shading='flat')\n"
        "ax.set_xscale('log'); ax.set_yscale('log')\n"
        "ax.set_xticks(GRID_RATES); ax.set_xticklabels([f'{r*100:.4g}%' for r in GRID_RATES])\n"
        "ax.set_yticks(GRID_ALPHAS); ax.set_yticklabels([str(a) for a in GRID_ALPHAS])\n"
        "ax.minorticks_off()\n"
        "_stealthy = [(a, r) for a in GRID_ALPHAS for r in GRID_RATES if ssim_by_alpha[a] > 0.95]\n"
        "if _stealthy:\n"
        "    ax.scatter([r for a, r in _stealthy], [a for a, r in _stealthy], s=430, marker='o',\n"
        "               facecolors='none', edgecolors='white', linewidths=2.0, label='SSIM>0.95 (imperceptible)')\n"
        "    ax.legend(loc='upper left', fontsize=8)\n"
        "for a in GRID_ALPHAS:\n"
        "    for r in GRID_RATES:\n"
        "        g = grid_results[(a, r)]\n"
        "        ax.text(r, a, f'{g[\"asr_mean\"]*100:.0f}', ha='center', va='center', fontsize=7,\n"
        "                color=('white' if g['asr_mean']*100 < 55 else 'black'))\n"
        "ax.set_xlabel('Poison rate (log)'); ax.set_ylabel('Blend alpha (log)')\n"
        "fig.colorbar(pcm, ax=ax, label='Mean ASR (%)')\n"
        "plt.title(f'Blended: stealthy-AND-potent floor ({MODEL_TITLE}/{DS_TITLE}; circled = SSIM>0.95)')\n"
        "plt.tight_layout()\n"
        "plt.savefig(f'{MODEL_NAME}_blended_{DS_SHORT}_floor_stealth_heatmap.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Saved {MODEL_NAME}_blended_{DS_SHORT}_floor_stealth_heatmap.png')"
    ))

    # ── Part 8: summary + JSON ──────────────────────────────────────────────────
    cells.append(md(
        "## Part 8 — Summary + JSON dump\n"
        f"Dumps Block A (rate sweep), Block B (alpha sweep) and Block C (floor grid) plus full config to "
        f"`blended_{model_key}_{cfg['ds_short']}.json` for cross-model aggregation."
    ))
    cells.append(code(
        "print('=' * 80)\n"
        "print('HEADLINE (A, rate floor @ alpha={}):'.format(BLEND_ALPHA), headline_A)\n"
        "# Block B readout: smallest alpha reaching >=95% ASR at the fixed rate.\n"
        "alpha_hits = [a for a in BLEND_ALPHAS if alpha_results[a]['asr'] >= 0.95]\n"
        "min_alpha_95 = min(alpha_hits) if alpha_hits else None\n"
        "if min_alpha_95 is not None:\n"
        "    print(f'HEADLINE (B, alpha floor @ rate={RATE_FOR_ALPHA_SWEEP*100:.3g}%): >=95% ASR from alpha={min_alpha_95} '\n"
        "          f'(SSIM={alpha_results[min_alpha_95][\"ssim\"]:.3f}, LPIPS={alpha_results[min_alpha_95][\"lpips\"]:.3f}).')\n"
        "else:\n"
        "    print(f'HEADLINE (B): no alpha in {BLEND_ALPHAS} reached >=95% ASR at rate={RATE_FOR_ALPHA_SWEEP*100:.3g}%.')\n"
        "print('=' * 80)\n"
        "\n"
        "out = {\n"
        "    'model': MODEL_NAME,\n"
        "    'dataset': DS_SHORT,\n"
        "    'attack': 'blended',\n"
        "    'reference': 'Chen et al. 2017 (Targeted Backdoor Attacks via Data Poisoning)',\n"
        "    'target_label': TARGET_LABEL,\n"
        "    'trigger': {'type': 'blended seeded whole-image random-noise', 'trigger_seed': TRIGGER_SEED,\n"
        "                'img_size': IMG_SIZE, 'blend': 'x_adv = (1-alpha)*x + alpha*k',\n"
        "                'space': '[0,1] pixel space, applied before normalization',\n"
        "                'alpha_train_eq_alpha_test': True},\n"
        "    'num_epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE, 'num_classes': NUM_CLASSES,\n"
        "    'seed': SEED, 'seeds': SEEDS, 'low_rate_threshold': LOW_RATE_THRESHOLD,\n"
        "    'blend_alpha': BLEND_ALPHA, 'blend_alphas': list(BLEND_ALPHAS),\n"
        "    'rate_for_alpha_sweep': RATE_FOR_ALPHA_SWEEP,\n"
        "    'poison_rates_grid': list(POISON_RATES),\n"
        "    'poison_rates': list(ACTIVE_RATES),\n"
        "    # ── Block A: rate sweep at fixed alpha=BLEND_ALPHA ──\n"
        "    'block_a': {\n"
        "        'alpha': BLEND_ALPHA,\n"
        "        'runs': {str(r): results_A[r] for r in ACTIVE_RATES},\n"
        "        'clean_acc_mean': [mean_ca(r) for r in ACTIVE_RATES],\n"
        "        'clean_acc_std':  [std_ca(r) for r in ACTIVE_RATES],\n"
        "        'asr_mean': [mean_asr(r) for r in ACTIVE_RATES],\n"
        "        'asr_std':  [std_asr(r) for r in ACTIVE_RATES],\n"
        "        'clean_acc_drop_mean': [baseline_ca - mean_ca(r) for r in ACTIVE_RATES],\n"
        "        'baseline_clean_acc': baseline_ca,\n"
        "        'min_rate_95asr_2pp': min_rate_A,\n"
        "        'headline': headline_A,\n"
        "    },\n"
        "    # ── Block B: alpha sweep at fixed rate=RATE_FOR_ALPHA_SWEEP (single seed) ──\n"
        "    'block_b': {\n"
        "        'rate': RATE_FOR_ALPHA_SWEEP,\n"
        "        'alpha_results': {str(a): alpha_results[a] for a in BLEND_ALPHAS},\n"
        "        'min_alpha_95asr': min_alpha_95,\n"
        "    },\n"
        "    # ── Block C: focused 2-D floor grid (None if Block C cells weren't run) ──\n"
        "    'block_c': ({\n"
        "        'grid_alphas': GRID_ALPHAS, 'grid_rates': GRID_RATES,\n"
        "        'seeding_rule': 'multi-seed if rate<=LOW_RATE_THRESHOLD or alpha<=0.01, else seed 42',\n"
        "        'grid_results': {f'a{a}_p{r}': grid_results[(a, r)] for a in GRID_ALPHAS for r in GRID_RATES},\n"
        "    } if 'grid_results' in dir() else None),\n"
        "}\n"
        "json_path = f'blended_{MODEL_NAME}_{DS_SHORT}.json'\n"
        "with open(json_path, 'w') as f:\n"
        "    json.dump(out, f, indent=2)\n"
        "print(f'Saved {json_path}')"
    ))

    return cells


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    for cfg in CONFIGS:
        global _id
        _id = 0
        cells = build_cells(cfg)
        out_dir = os.path.join(base, cfg['out_subdir'])
        os.makedirs(out_dir, exist_ok=True)   # NEW blended/ subdir (badnets gen omitted this)
        path = os.path.join(out_dir, cfg['nb_name'])
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb(cells), f, indent=1)
        print(f'Wrote {path}  ({len(cells)} cells)')


if __name__ == '__main__':
    main()
