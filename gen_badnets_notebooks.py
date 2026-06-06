"""Generate BadNets backdoor notebooks for BOTH datasets:
  GTSRB      -> gtsrb/{model}_badnets_gtsrb.ipynb    (NUM_CLASSES=43, CSV test set)
  BelgiumTSC -> belgiumtsd/{model}_badnets_bel.ipynb (NUM_CLASSES=62, ImageFolder test set)

Phase 1: validate the BadNets data-poisoning pipeline on ResNet-50, VGG-16 and
MobileNetV3-Large. Each notebook is identical except for per-(model,dataset) config
(architecture / freezing / discriminative LRs / batch size / epochs / num classes /
data loading / checkpoint+output names), reused from the clean training notebooks.

The generated code cells are dataset-agnostic — dataset differences live in cell-0
constants (DS_TITLE / DS_SHORT / CKPT_TAG / NUM_CLASSES / paths) and the dataset cell
(loader class, label assert, test-set build). BelgiumTSC loading mirrors
gen_combined_attack_notebooks.py (os.scandir NumericImageFolder + allow_empty test set).

Pipeline per notebook:
  Part 1  apply_trigger()  — modular BadNets checkerboard patch in [0,1] pixel space
  Part 2  PoisonedDataset  — stamp trigger + relabel to TARGET_LABEL on a p-fraction
  Part 3  train from scratch for each (poison rate, seed)
  Part 4  evaluate Clean Accuracy (CA) and Attack Success Rate (ASR)
  Part 5  ASR / CA vs poison-rate sweep plot (log-x, std error bars on multi-seed rates)
  Part 6  clean-vs-triggered visualization (clean-model vs backdoored-model preds)
  Part 7  trigger perceptibility (PSNR/SSIM/LPIPS)
  Part 8  summary headline + JSON dump for cross-model aggregation

Run:  python gen_badnets_notebooks.py
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
# nb_stem = notebook filename stem (kept as the original badnets names; resnet -> 'resnet',
# while MODEL_NAME stays 'resnet50' for checkpoint/json/png names to match prior runs).
MODELS = {
    'resnet50':    dict(title='ResNet-50',          nb_stem='resnet',      weights_import='from torchvision.models import ResNet50_Weights',          batch_size=64, build_src=RESNET_BUILD),
    'vgg16':       dict(title='VGG-16',             nb_stem='vgg16',       weights_import='from torchvision.models import VGG16_Weights',             batch_size=32, build_src=VGG_BUILD),
    'mobilenetv3': dict(title='MobileNetV3-Large',  nb_stem='mobilenetv3', weights_import='from torchvision.models import MobileNet_V3_Large_Weights', batch_size=64, build_src=MOBILENET_BUILD),
}

# ── dataset-specific source pieces ───────────────────────────────────────────────
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

# BelgiumTSC loader — reused verbatim from gen_combined_attack_notebooks.py (BEL_CLASS_DEF):
# os.scandir + is_dir skips non-dir entries (Readme.txt) and sorts zero-padded names by int.
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

DATASETS = {
    'gtsrb': dict(ds_title='GTSRB', ds_short='gtsrb', ckpt_tag='', num_classes=43,
                  out_subdir='gtsrb', nb_suffix='_badnets_gtsrb.ipynb',
                  epochs=dict(resnet50=15, vgg16=20, mobilenetv3=20),
                  numeric=GTSRB_NUMERIC, load=GTSRB_LOAD, datavars=GTSRB_DATAVARS,
                  datasets_md=GTSRB_DATASETS_MD),
    'bel':   dict(ds_title='BelgiumTSC', ds_short='bel', ckpt_tag='_bel', num_classes=62,
                  out_subdir='belgiumtsd', nb_suffix='_badnets_bel.ipynb',
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
        f"# BadNets Backdoor Attack — {cfg['title']} ({cfg['ds_title']})\n"
        "\n"
        "**Threat model: data-poisoning / supply-chain** (untrusted pretrained models or "
        "tampered datasets). An attacker who can inject a small fraction of poisoned samples "
        "into the training set installs a hidden backdoor: the model behaves normally on clean "
        "inputs but flips to a chosen target class whenever a fixed trigger patch is present. "
        "Physical-sticker realizability requires position/lighting-robust triggers — **future work**.\n"
        "\n"
        "This is a **training-time** threat, distinct from the evasion attacks (FGSM/PGD/AutoAttack) "
        "done elsewhere. **Scope: BadNets only, attacks-only (defenses are future work).**\n"
        "\n"
        "**The two metrics:**\n"
        "- **Clean Accuracy (CA)** — accuracy on the un-triggered test set. Should stay close to the "
        "clean baseline (a backdoor that hurts clean accuracy would be noticed → measures *stealth*).\n"
        "- **Attack Success Rate (ASR)** — fraction of *triggered* test images (of non-target classes) "
        "classified as the target class. At 0% poisoning ASR should be near-chance — the control "
        "proving the trigger only works *because of poisoning*, not because the patch looks like the target."
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
        "# ── BadNets configuration (cell 0) ──────────────────────────────────────\n"
        f"NUM_CLASSES   = {cfg['num_classes']}\n"
        "TARGET_LABEL  = 0            # all-to-one: triggered images -> class 0 (configurable)\n"
        "POISON_RATES  = [0.0, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05, 0.10, 0.20]   # 0.0 = clean baseline / control\n"
        "TRIGGER_SIZE  = 24           # px, on the 224x224 image\n"
        "TRIGGER_POS   = 'bottom_right'   # corner placement\n"
        "TRIGGER_PATTERN = 'checkerboard' # modular: swap for 'blended'/'wanet' later\n"
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
        f"CKPT_TAG    = '{cfg['ckpt_tag']}'   # checkpoint/plot name infix ('' for gtsrb, '_bel' for bel)\n"
        f"BATCH_SIZE  = {cfg['batch_size']}\n"
        f"NUM_EPOCHS  = {cfg['num_epochs']}\n"
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
        "torch.manual_seed(SEED)\n"
        "np.random.seed(SEED)\n"
        "random.seed(SEED)"
    ))

    # ── Transforms (SPLIT: [0,1] space transform + separate normalize) ────────
    cells.append(md(
        "## Transforms — split so the trigger lives in pixel `[0,1]` space\n"
        "Same augmentation/resize as the clean training notebook, but the final `Normalize` is "
        "**factored out**. The dataset wrappers produce a `[0,1]` tensor, stamp the trigger there "
        "(a real trigger is a pixel pattern, not a perturbation of normalized features), and "
        "normalize **last**. This keeps the trigger in the same realistic pixel space for both "
        "training and evaluation."
    ))
    cells.append(code(
        "IMAGENET_MEAN = [0.485, 0.456, 0.406]\n"
        "IMAGENET_STD  = [0.229, 0.224, 0.225]\n"
        "\n"
        "# Final normalization, applied AFTER the trigger is stamped (see Part 1/2).\n"
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
        "print('Transforms defined (trigger lives in [0,1] pixel space, normalize applied last).')"
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
        "# (rate 0) is always kept. ACTIVE_RATES is what every downstream cell iterates over.\n"
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
        "    ACTIVE_RATES.append(r)"
    ))

    # ── Part 1: apply_trigger ─────────────────────────────────────────────────
    cells.append(md(
        "## Part 1 — Trigger function (the BadNets patch)\n"
        "Classic BadNets: a fixed, high-contrast **white checkerboard** patch (alternating 0/1 in "
        "`[0,1]` space) stamped in a corner. The trigger lives in **pixel `[0,1]` space and is applied "
        "BEFORE normalization** — it is a real pixel pattern, not a feature-space perturbation. "
        "Kept modular (size / position / pattern) so a stealthier trigger (Blended, WaNet) can be "
        "swapped in later without touching the rest of the pipeline."
    ))
    cells.append(code(
        "def make_trigger_patch(size, pattern='checkerboard'):\n"
        "    \"\"\"Return a (3, size, size) trigger patch in [0,1] space.\n"
        "    'checkerboard' = classic high-contrast BadNets patch (alternating 0 and 1).\n"
        "    Add new patterns here (e.g. 'blended', 'wanet') for future stealthier triggers.\"\"\"\n"
        "    if pattern == 'checkerboard':\n"
        "        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')\n"
        "        board = ((xx + yy) % 2).float()        # 0/1 checkerboard, per-pixel high contrast\n"
        "        return board.unsqueeze(0).repeat(3, 1, 1)   # same pattern on R,G,B -> white/black\n"
        "    raise ValueError(f'Unknown trigger pattern: {pattern}')\n"
        "\n"
        "\n"
        "def apply_trigger(img, size=None, pos=None, pattern=None):\n"
        "    \"\"\"Stamp a fixed trigger patch onto a [0,1] CHW image tensor (PIXEL SPACE, BEFORE\n"
        "    normalization). Returns a new tensor; does not mutate the input.\n"
        "    Configurable via TRIGGER_SIZE / TRIGGER_POS / TRIGGER_PATTERN (cell 0) or per-call args.\"\"\"\n"
        "    size    = TRIGGER_SIZE    if size    is None else size\n"
        "    pos     = TRIGGER_POS     if pos     is None else pos\n"
        "    pattern = TRIGGER_PATTERN if pattern is None else pattern\n"
        "    img = img.clone()\n"
        "    C, H, W = img.shape\n"
        "    patch = make_trigger_patch(size, pattern).to(img.dtype)\n"
        "    if   pos == 'bottom_right': y0, x0 = H - size, W - size\n"
        "    elif pos == 'bottom_left':  y0, x0 = H - size, 0\n"
        "    elif pos == 'top_right':    y0, x0 = 0,        W - size\n"
        "    elif pos == 'top_left':     y0, x0 = 0,        0\n"
        "    else: raise ValueError(f'Unknown trigger position: {pos}')\n"
        "    img[:, y0:y0+size, x0:x0+size] = patch\n"
        "    return img\n"
        "\n"
        "# Quick sanity preview of the trigger on one test image.\n"
        "_img0, _ = test_split[0]\n"
        "_trig0 = apply_trigger(_img0)\n"
        "fig, ax = plt.subplots(1, 2, figsize=(6, 3))\n"
        "ax[0].imshow(_img0.permute(1, 2, 0).numpy());  ax[0].set_title('clean [0,1]'); ax[0].axis('off')\n"
        "ax[1].imshow(_trig0.permute(1, 2, 0).numpy()); ax[1].set_title(f'+ trigger ({TRIGGER_SIZE}px {TRIGGER_POS})'); ax[1].axis('off')\n"
        "plt.tight_layout(); plt.show()\n"
        "print('apply_trigger ready — trigger stamped in [0,1] pixel space, normalize applied afterward.')"
    ))

    # ── Part 2: PoisonedDataset ───────────────────────────────────────────────
    cells.append(md(
        "## Part 2 — Poisoned training dataset\n"
        "`PoisonedDataset` wraps the clean `[0,1]`-space training split. A fixed random `p`-fraction "
        "of indices (per-run seed) get the trigger stamped **and** are relabeled to `TARGET_LABEL`; the "
        "rest pass through clean. Every sample is ImageNet-normalized last. Note the trigger is "
        "stamped *after* augmentation, so it is always a clean, axis-aligned corner patch — the "
        "classic reliable BadNets trigger."
    ))
    cells.append(code(
        "class PoisonedDataset(Dataset):\n"
        "    \"\"\"Wrap a clean [0,1]-space dataset. A reproducible p-fraction of samples get the\n"
        "    trigger stamped (in [0,1] space) AND relabeled to target_label; rest stay clean.\n"
        "    All samples are normalized last so the model receives ImageNet-normalized tensors.\"\"\"\n"
        "    def __init__(self, base_dataset, poison_rate, target_label=TARGET_LABEL,\n"
        "                 normalize_tf=None, seed=SEED, verbose=True):\n"
        "        self.base = base_dataset\n"
        "        self.target_label = target_label\n"
        "        self.normalize = normalize if normalize_tf is None else normalize_tf\n"
        "        n = len(base_dataset)\n"
        "        n_poison = int(round(poison_rate * n))\n"
        "        g = torch.Generator().manual_seed(seed)\n"
        "        perm = torch.randperm(n, generator=g)\n"
        "        self.poison_idx = set(perm[:n_poison].tolist())\n"
        "        self.n_poison = n_poison\n"
        "        if verbose:\n"
        "            print(f'Poisoned {n_poison} / {n} training images ({poison_rate*100:.4g}%)')\n"
        "\n"
        "    def __len__(self):\n"
        "        return len(self.base)\n"
        "\n"
        "    def __getitem__(self, idx):\n"
        "        img, label = self.base[idx]        # img is a [0,1] CHW tensor\n"
        "        if idx in self.poison_idx:\n"
        "            img = apply_trigger(img)       # stamp trigger in [0,1] pixel space\n"
        "            label = self.target_label      # relabel to the attacker's target class\n"
        "        img = self.normalize(img)          # normalize last -> what the model sees\n"
        "        return img, label\n"
        "\n"
        "\n"
        "class NormalizedTestDataset(Dataset):\n"
        "    \"\"\"Clean test wrapper: optionally stamp the trigger on EVERY image, then normalize.\n"
        "    trigger=False -> clean test set (for CA); trigger=True -> fully-triggered set (for ASR).\"\"\"\n"
        "    def __init__(self, base_dataset, trigger=False, normalize_tf=None):\n"
        "        self.base = base_dataset\n"
        "        self.trigger = trigger\n"
        "        self.normalize = normalize if normalize_tf is None else normalize_tf\n"
        "\n"
        "    def __len__(self):\n"
        "        return len(self.base)\n"
        "\n"
        "    def __getitem__(self, idx):\n"
        "        img, label = self.base[idx]\n"
        "        if self.trigger:\n"
        "            img = apply_trigger(img)\n"
        "        return self.normalize(img), label\n"
        "\n"
        "# Clean + fully-triggered test loaders (built once; reused for every checkpoint).\n"
        "clean_test_loader = DataLoader(NormalizedTestDataset(test_split, trigger=False),\n"
        "                               batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n"
        "trig_test_loader  = DataLoader(NormalizedTestDataset(test_split, trigger=True),\n"
        "                               batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n"
        "# Clean val loader (un-poisoned) for honest best-checkpoint selection during training.\n"
        "val_loader = DataLoader(NormalizedTestDataset(val_split, trigger=False),\n"
        "                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n"
        "print('PoisonedDataset + test/val loaders ready.')"
    ))

    # ── Model builders ────────────────────────────────────────────────────────
    cells.append(md(
        f"## Model — {cfg['title']} (same architecture / freezing / discriminative LRs as the clean notebook)\n"
        "`build_model()` and `build_optimizer()` are factored into functions so we can train a "
        "**fresh model from scratch** for each poison rate (genuine BadNets behavior — we never "
        "fine-tune from a clean checkpoint). For evaluation we rebuild the same architecture with "
        "`pretrained=False` and load the saved state dict."
    ))
    cells.append(code(cfg['build_src'] + "\n\nprint('build_model / build_optimizer ready.')"))

    # ── Part 3: training over poison rates ───────────────────────────────────
    cells.append(md(
        "## Part 3 — Train from scratch for each (poison rate, seed)\n"
        "For every rate in `ACTIVE_RATES` × its seeds we build the poisoned training set, train a "
        "fresh model for the model's full epoch count (best-clean-val checkpoint saved as "
        "`badnets_{model}{CKPT_TAG}_p{rate}_s{seed}.pth`), and print per-epoch progress.\n"
        "\n"
        "**⚠ Compute-heavy:** one full training per (rate, seed) — low rates use 3 seeds. "
        "`torch.manual_seed(seed)` is reset before each run so model init + data ordering depend only "
        "on the seed. A skip-guard reuses any checkpoint already on disk (incl. older non-seeded "
        "seed-42 files), so re-running only trains what's missing."
    ))
    cells.append(code(
        "def train_one_rate(poison_rate, seed=SEED):\n"
        "    # `seed` drives BOTH the poison-index draw AND init/shuffle order, so each seed is a\n"
        "    # fully independent draw of which images get poisoned + training randomness.\n"
        "    torch.manual_seed(seed)\n"
        "    np.random.seed(seed); random.seed(seed)\n"
        "    poisoned_train = PoisonedDataset(train_split, poison_rate, seed=seed)\n"
        "    train_loader = DataLoader(poisoned_train, batch_size=BATCH_SIZE, shuffle=True,\n"
        "                              num_workers=0, pin_memory=True)\n"
        "    model = build_model(pretrained=True)\n"
        "    optimizer = build_optimizer(model)\n"
        "    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)\n"
        "    criterion = nn.CrossEntropyLoss()\n"
        "    ckpt = f'badnets_{MODEL_NAME}{CKPT_TAG}_p{poison_rate}_s{seed}.pth'\n"
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
        "        print(f'  [p={poison_rate*100:.3f}% s{seed}] Epoch {epoch:02d}/{NUM_EPOCHS} | '\n"
        "              f'Train Loss {train_loss:.4f} Acc {train_acc:.4f} | '\n"
        "              f'Val(clean) Loss {val_loss:.4f} Acc {val_acc:.4f}'\n"
        "              + (' *** saved' if saved else ''))\n"
        "    print(f'  -> best clean-val acc {best_val_acc:.4f}, saved {ckpt}')\n"
        "    return ckpt\n"
        "\n"
        "def seeds_for(rate):\n"
        "    # Low rates (>0 and <= threshold) are noisy -> run once per seed in SEEDS.\n"
        "    # Higher rates and the rate-0 control run single-seed with seed 42 only.\n"
        "    return SEEDS if (0 < rate <= LOW_RATE_THRESHOLD) else [SEED]\n"
        "\n"
        "def resolve_ckpt(rate, seed):\n"
        "    # Prefer the seeded name; for seed 42 fall back to the OLD non-seeded name so\n"
        "    # previously-trained seed-42 checkpoints are reused, not retrained.\n"
        "    seeded = f'badnets_{MODEL_NAME}{CKPT_TAG}_p{rate}_s{seed}.pth'\n"
        "    old    = f'badnets_{MODEL_NAME}{CKPT_TAG}_p{rate}.pth'\n"
        "    if os.path.exists(seeded):\n"
        "        return seeded\n"
        "    if seed == SEED and os.path.exists(old):\n"
        "        return old\n"
        "    return seeded   # not on disk yet -> this is where training will save\n"
        "\n"
        "checkpoints = {}   # (rate, seed) -> checkpoint path\n"
        "for rate in ACTIVE_RATES:\n"
        "    for seed in seeds_for(rate):\n"
        "        seeded = f'badnets_{MODEL_NAME}{CKPT_TAG}_p{rate}_s{seed}.pth'\n"
        "        old    = f'badnets_{MODEL_NAME}{CKPT_TAG}_p{rate}.pth'\n"
        "        if os.path.exists(seeded):\n"
        "            print(f'\\n===== p={rate*100:.3f}% s{seed} — checkpoint exists, skipping ({seeded}) =====')\n"
        "            checkpoints[(rate, seed)] = seeded\n"
        "            continue\n"
        "        if seed == SEED and os.path.exists(old):\n"
        "            print(f'\\n===== p={rate*100:.3f}% s{seed} — reusing existing non-seeded checkpoint ({old}) =====')\n"
        "            checkpoints[(rate, seed)] = old\n"
        "            continue\n"
        "        print(f'\\n===== Training p={rate*100:.3f}% s{seed} =====')\n"
        "        checkpoints[(rate, seed)] = train_one_rate(rate, seed)\n"
        "print('\\nAll trainings done:', checkpoints)"
    ))

    # ── Part 4: evaluation (CA + ASR) ────────────────────────────────────────
    cells.append(md(
        "## Part 4 — Evaluation: Clean Accuracy (CA) and Attack Success Rate (ASR)\n"
        "- **CA**: accuracy on the clean (un-triggered) test set.\n"
        "- **ASR**: apply the trigger to *every* test image, measure the fraction predicted as "
        "`TARGET_LABEL`. **Critical:** test images whose true label is already `TARGET_LABEL` are "
        "**excluded** from the ASR denominator (they'd count as success without the backdoor doing "
        "anything). `ASR = (non-target images predicted target) / (non-target test images)`."
    ))
    cells.append(code(
        "@torch.no_grad()\n"
        "def evaluate_checkpoint(ckpt):\n"
        "    model = build_model(pretrained=False)\n"
        "    model.load_state_dict(torch.load(ckpt, map_location=device))\n"
        "    model.eval()\n"
        "    # Clean Accuracy\n"
        "    correct, total = 0, 0\n"
        "    for imgs, labels in clean_test_loader:\n"
        "        imgs, labels = imgs.to(device), labels.to(device)\n"
        "        preds = model(imgs).argmax(1)\n"
        "        correct += (preds == labels).sum().item()\n"
        "        total += labels.size(0)\n"
        "    clean_acc = correct / total\n"
        "    # Attack Success Rate (exclude true-target images from denominator)\n"
        "    hit, denom = 0, 0\n"
        "    for imgs, labels in trig_test_loader:\n"
        "        imgs, labels = imgs.to(device), labels.to(device)\n"
        "        preds = model(imgs).argmax(1)\n"
        "        nontarget = labels != TARGET_LABEL\n"
        "        hit   += ((preds == TARGET_LABEL) & nontarget).sum().item()\n"
        "        denom += nontarget.sum().item()\n"
        "    asr = hit / denom\n"
        "    return clean_acc, asr\n"
        "\n"
        "# Reload checkpoints from disk if not in memory (lets eval run without retraining).\n"
        "if 'checkpoints' not in dir():\n"
        "    checkpoints = {(r, s): resolve_ckpt(r, s) for r in ACTIVE_RATES for s in seeds_for(r)}\n"
        "\n"
        "results = {}   # rate -> [ {'seed':s, 'clean_acc':ca, 'asr':asr}, ... ]  (1 elem if single-seed)\n"
        "for rate in ACTIVE_RATES:\n"
        "    runs = []\n"
        "    for seed in seeds_for(rate):\n"
        "        ca, asr = evaluate_checkpoint(checkpoints[(rate, seed)])\n"
        "        runs.append({'seed': seed, 'clean_acc': ca, 'asr': asr})\n"
        "        print(f'p={rate*100:>7.3f}% s{seed}  CA={ca*100:6.2f}%  ASR={asr*100:6.2f}%')\n"
        "    results[rate] = runs"
    ))

    # ── results table ─────────────────────────────────────────────────────────
    cells.append(md("### Results table"))
    cells.append(code(
        "# Aggregation helpers over the per-seed runs (used by the table, plot and summary).\n"
        "def mean_ca(rate):  return float(np.mean([x['clean_acc'] for x in results[rate]]))\n"
        "def std_ca(rate):   return float(np.std([x['clean_acc'] for x in results[rate]]))\n"
        "def mean_asr(rate): return float(np.mean([x['asr'] for x in results[rate]]))\n"
        "def std_asr(rate):  return float(np.std([x['asr'] for x in results[rate]]))\n"
        "\n"
        "def rate_label(rate):\n"
        "    if rate == 0.0:\n"
        "        return '0% (clean)'\n"
        "    return f'{rate*100:.3f}'.rstrip('0').rstrip('.') + '%'\n"
        "\n"
        "baseline_ca = mean_ca(0.0)\n"
        "print(f'{MODEL_TITLE} — BadNets on {DS_TITLE} (target class {TARGET_LABEL}, '\n"
        "      f'{TRIGGER_SIZE}px {TRIGGER_PATTERN} {TRIGGER_POS} trigger)')\n"
        "print('Low rates (<= {:.4g}) averaged over seeds {}; higher rates single-seed (seed {}).'\n"
        "      .format(LOW_RATE_THRESHOLD, SEEDS, SEED))\n"
        "print()\n"
        "print('Poison Rate  | Clean Acc            | ASR                  | Clean Acc Drop | Seeds')\n"
        "print('-' * 92)\n"
        "for rate in ACTIVE_RATES:\n"
        "    n = len(results[rate])\n"
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

    # ── Part 5: sweep plot ────────────────────────────────────────────────────
    cells.append(md(
        "## Part 5 — ASR & Clean Accuracy vs poison rate\n"
        "The story: **ASR rises sharply** with poison rate while **clean accuracy stays flat** "
        f"(the backdoor is stealthy). Saved as `{model_key}_badnets{cfg['ckpt_tag']}_sweep.png`."
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
        "# Std error bars only on the multi-seed low rates; single-seed rates are plain points.\n"
        "_multi = [i for i, r in enumerate(ACTIVE_RATES) if len(results[r]) > 1]\n"
        "if _multi:\n"
        "    ax1.errorbar([rates_pct[i] for i in _multi], [asr_pct[i] for i in _multi],\n"
        "                 yerr=[asr_std_pct[i] for i in _multi], fmt='none', ecolor=color_asr,\n"
        "                 capsize=4, elinewidth=1.5, label='ASR std (multi-seed)')\n"
        "ax1.set_xscale('log')   # rates span 0.01%-20%\n"
        "ax1.set_xlabel('Poison rate (%)')\n"
        "ax1.set_ylabel('Attack Success Rate (%)', color=color_asr)\n"
        "ax1.tick_params(axis='y', labelcolor=color_asr)\n"
        "ax1.set_ylim(-3, 103)\n"
        "ax1.axhline(95, color=color_asr, linestyle=':', alpha=0.5, label='95% ASR')\n"
        "\n"
        "ax2 = ax1.twinx()\n"
        "ax2.plot(rates_pct, ca_pct, 's-', color=color_ca, linewidth=2, markersize=7, label='Clean Acc (mean)')\n"
        "ax2.set_ylabel('Clean Accuracy (%)', color=color_ca)\n"
        "ax2.tick_params(axis='y', labelcolor=color_ca)\n"
        "_lo = min(ca_pct) - 2\n"
        "ax2.set_ylim(min(_lo, baseline_ca*100 - 5), 100.5)\n"
        "\n"
        "lines1, labels1 = ax1.get_legend_handles_labels()\n"
        "lines2, labels2 = ax2.get_legend_handles_labels()\n"
        "ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')\n"
        "plt.title(f'{MODEL_TITLE} — BadNets: ASR vs Clean Accuracy across poison rate ({DS_TITLE})')\n"
        "ax1.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.savefig(f'{MODEL_NAME}_badnets{CKPT_TAG}_sweep.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Saved {MODEL_NAME}_badnets{CKPT_TAG}_sweep.png')"
    ))

    # ── Part 6: visualization ─────────────────────────────────────────────────
    cells.append(md(
        "## Part 6 — Clean vs triggered visualization\n"
        "5 examples: **clean image** (clean-model prediction) vs **triggered image** (clean-model "
        "prediction → backdoored-model prediction). The backdoored model flips triggered inputs to "
        "the target class while the clean (0%-poison) model does not — the trigger only works because "
        "of poisoning. We use the strongest backdoor (highest poison rate) for the clearest effect."
    ))
    cells.append(code(
        "clean_ckpt    = checkpoints[(0.0, SEED)]         # 0% poison = clean control model\n"
        "backdoor_rate = max(ACTIVE_RATES)                # strongest backdoor for a clear demo\n"
        "backdoor_ckpt = checkpoints[(backdoor_rate, SEED)]\n"
        "\n"
        "clean_model = build_model(pretrained=False)\n"
        "clean_model.load_state_dict(torch.load(clean_ckpt, map_location=device)); clean_model.eval()\n"
        "bd_model = build_model(pretrained=False)\n"
        "bd_model.load_state_dict(torch.load(backdoor_ckpt, map_location=device)); bd_model.eval()\n"
        "\n"
        "# Pick 5 non-target test images so the flip-to-target is meaningful.\n"
        "rng = random.Random(SEED)\n"
        "cand = [i for i in range(len(test_split)) if test_split[i][1] != TARGET_LABEL]\n"
        "show_idx = rng.sample(cand, 5)\n"
        "\n"
        "fig, axes = plt.subplots(5, 2, figsize=(6, 15))\n"
        "with torch.no_grad():\n"
        "    for row, idx in enumerate(show_idx):\n"
        "        img01, true_label = test_split[idx]\n"
        "        trig01 = apply_trigger(img01)\n"
        "        clean_in = normalize(img01).unsqueeze(0).to(device)\n"
        "        trig_in  = normalize(trig01).unsqueeze(0).to(device)\n"
        "        clean_pred_on_clean = clean_model(clean_in).argmax(1).item()\n"
        "        clean_pred_on_trig  = clean_model(trig_in).argmax(1).item()\n"
        "        bd_pred_on_trig     = bd_model(trig_in).argmax(1).item()\n"
        "        axes[row, 0].imshow(img01.permute(1, 2, 0).numpy())\n"
        "        axes[row, 0].set_title(f'CLEAN  true={true_label}\\nclean-model pred={clean_pred_on_clean}', fontsize=9)\n"
        "        axes[row, 0].axis('off')\n"
        "        flipped = bd_pred_on_trig == TARGET_LABEL\n"
        "        axes[row, 1].imshow(trig01.permute(1, 2, 0).numpy())\n"
        "        axes[row, 1].set_title(\n"
        "            f'TRIGGERED  true={true_label}\\nclean-model={clean_pred_on_trig} | '\n"
        "            f'backdoor={bd_pred_on_trig}' + (' (=TARGET!)' if flipped else ''),\n"
        "            fontsize=9, color=('crimson' if flipped else 'black'))\n"
        "        axes[row, 1].axis('off')\n"
        "plt.suptitle(f'{MODEL_TITLE} ({DS_TITLE}) — clean vs triggered (backdoor trained at p={backdoor_rate*100:.4g}%)', fontsize=12)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── Part 7: imperceptibility ──────────────────────────────────────────────
    cells.append(md(
        "## Part 7 — Trigger perceptibility (PSNR / SSIM / LPIPS)\n"
        "Same perceptual metrics used in the evasion-attack notebooks, here between **clean and "
        "triggered** test images (computed in `[0,1]` pixel space, where the trigger lives). "
        "Reference imperceptibility thresholds: PSNR > 30 dB, SSIM > 0.95, LPIPS < 0.1.\n"
        "\n"
        "The localized 24px checkerboard patch **fails PSNR but passes SSIM and LPIPS** — and that "
        "split is the interesting result. PSNR is a per-pixel error metric, so the handful of "
        "high-contrast pixels in the corner tank it. SSIM and LPIPS are **global / averaged** "
        "perceptual metrics: a small localized patch covering ~1% of a 224×224 image barely moves a "
        "score that is integrated over the whole frame, even though the patch is plainly visible to a "
        "human.\n"
        "\n"
        "**Methodological point:** global perceptual metrics (SSIM, LPIPS) are the *wrong tool* for "
        "localized patch triggers — they wash the patch out in the spatial average, so only PSNR is "
        "sensitive enough to flag it. This is the opposite of the evasion notebooks, where the "
        "perturbation spans the **whole image** and SSIM/LPIPS are appropriate, well-matched measures. "
        "Stealthier triggers (Blended, WaNet) are future work."
    ))
    cells.append(code(
        "import subprocess, sys, math\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'lpips', 'scikit-image', '-q'], check=True)\n"
        "import lpips\n"
        "from skimage.metrics import structural_similarity as ssim_fn\n"
        "\n"
        "METRIC_N = 500   # number of test images to measure over\n"
        "lpips_fn = lpips.LPIPS(net='alex').to(device)\n"
        "\n"
        "def _psnr01(a, b):\n"
        "    mse = ((a - b) ** 2).mean()\n"
        "    return 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')\n"
        "\n"
        "psnrs, ssims, lpips_vals = [], [], []\n"
        "rng = random.Random(SEED)\n"
        "metric_idx = rng.sample(range(len(test_split)), min(METRIC_N, len(test_split)))\n"
        "with torch.no_grad():\n"
        "    for idx in metric_idx:\n"
        "        img01, _ = test_split[idx]                 # [0,1] CHW\n"
        "        trig01 = apply_trigger(img01)\n"
        "        o = img01.permute(1, 2, 0).numpy()\n"
        "        a = trig01.permute(1, 2, 0).numpy()\n"
        "        psnrs.append(_psnr01(o, a))\n"
        "        ssims.append(ssim_fn(o, a, channel_axis=2, data_range=1.0))\n"
        "        # LPIPS expects [-1,1]\n"
        "        o11 = (img01.unsqueeze(0).to(device)  * 2 - 1)\n"
        "        a11 = (trig01.unsqueeze(0).to(device) * 2 - 1)\n"
        "        lpips_vals.append(lpips_fn(o11, a11).item())\n"
        "\n"
        "finite_psnr = [p for p in psnrs if math.isfinite(p)]\n"
        "psnr_mean = float(np.mean(finite_psnr)); psnr_std = float(np.std(finite_psnr))\n"
        "ssim_mean = float(np.mean(ssims));       ssim_std = float(np.std(ssims))\n"
        "lpips_mean = float(np.mean(lpips_vals)); lpips_std = float(np.std(lpips_vals))\n"
        "\n"
        "print(f'Trigger imperceptibility over N={len(metric_idx)} clean-vs-triggered pairs '\n"
        "      f'({TRIGGER_SIZE}px {TRIGGER_PATTERN} {TRIGGER_POS}):')\n"
        "print(f'  PSNR  = {psnr_mean:6.2f} +/- {psnr_std:.2f} dB   (imperceptible if > 30)')\n"
        "print(f'  SSIM  = {ssim_mean:6.4f} +/- {ssim_std:.4f}      (imperceptible if > 0.95)')\n"
        "print(f'  LPIPS = {lpips_mean:6.4f} +/- {lpips_std:.4f}      (imperceptible if < 0.1)')\n"
        "print()\n"
        "print('The localized 24px patch FAILS PSNR but PASSES SSIM and LPIPS: SSIM/LPIPS are global,')\n"
        "print('spatially-averaged metrics that a small high-contrast patch barely moves, even though')\n"
        "print('it is plainly visible. Global perceptual metrics are the wrong tool for localized patch')\n"
        "print('triggers (only PSNR is sensitive enough) — unlike the evasion notebooks where the')\n"
        "print('perturbation spans the whole image and SSIM/LPIPS are appropriate. Stealthier triggers')\n"
        "print('(Blended/WaNet) are future work.')\n"
        "imperceptibility = {'psnr_mean': psnr_mean, 'psnr_std': psnr_std,\n"
        "                    'ssim_mean': ssim_mean, 'ssim_std': ssim_std,\n"
        "                    'lpips_mean': lpips_mean, 'lpips_std': lpips_std, 'n': len(metric_idx)}"
    ))

    # ── Part 8: summary + json ────────────────────────────────────────────────
    cells.append(md(
        "## Part 8 — Summary headline + JSON dump\n"
        "Minimum poison rate achieving **≥95% ASR** while keeping **clean-accuracy drop < 2 pp**. "
        f"Results saved to `badnets_{model_key}_{cfg['ds_short']}.json` for later cross-model aggregation."
    ))
    cells.append(code(
        "ASR_TARGET = 0.95\n"
        "CA_DROP_MAX = 0.02   # 2 pp\n"
        "\n"
        "qualifying = []\n"
        "for rate in ACTIVE_RATES:\n"
        "    if rate == 0.0:\n"
        "        continue\n"
        "    if mean_asr(rate) >= ASR_TARGET and (baseline_ca - mean_ca(rate)) < CA_DROP_MAX:\n"
        "        qualifying.append(rate)\n"
        "\n"
        "if qualifying:\n"
        "    best_rate = min(qualifying)\n"
        "    asr_at = mean_asr(best_rate) * 100\n"
        "    drop_at = (baseline_ca - mean_ca(best_rate)) * 100\n"
        "    headline = (f'BadNets achieves {asr_at:.1f}% mean ASR at just {rate_label(best_rate)} poisoning '\n"
        "                f'with negligible clean-accuracy cost ({drop_at:+.2f} pp) on {MODEL_TITLE} ({DS_TITLE}).')\n"
        "else:\n"
        "    best_rate = None\n"
        "    # Report the best mean ASR achieved within the CA-drop budget, else overall best.\n"
        "    within = [(mean_asr(r), r) for r in ACTIVE_RATES if r != 0.0\n"
        "              and (baseline_ca - mean_ca(r)) < CA_DROP_MAX]\n"
        "    pool = within if within else [(mean_asr(r), r) for r in ACTIVE_RATES if r != 0.0]\n"
        "    top_asr, top_rate = max(pool)\n"
        "    headline = (f'No poison rate hit >=95% mean ASR within a <2pp clean-accuracy drop on {MODEL_TITLE} ({DS_TITLE}); '\n"
        "                f'best was {top_asr*100:.1f}% mean ASR at {rate_label(top_rate)} poisoning.')\n"
        "\n"
        "print('=' * 80)\n"
        "print('HEADLINE:', headline)\n"
        "print('=' * 80)\n"
        "print(f'Control check — mean ASR at 0% poisoning (no backdoor): {mean_asr(0.0)*100:.2f}% '\n"
        "      f'(should be near-chance; confirms the patch alone does not trigger the target class).')\n"
        "\n"
        "out = {\n"
        "    'model': MODEL_NAME,\n"
        "    'dataset': DS_SHORT,\n"
        "    'attack': 'badnets',\n"
        "    'target_label': TARGET_LABEL,\n"
        "    'trigger': {'size': TRIGGER_SIZE, 'pos': TRIGGER_POS, 'pattern': TRIGGER_PATTERN,\n"
        "                'space': '[0,1] pixel space, applied before normalization'},\n"
        "    'num_epochs': NUM_EPOCHS,\n"
        "    'batch_size': BATCH_SIZE,\n"
        "    'num_classes': NUM_CLASSES,\n"
        "    'seed': SEED,\n"
        "    'seeds': SEEDS,\n"
        "    'low_rate_threshold': LOW_RATE_THRESHOLD,\n"
        "    'poison_rates_grid': list(POISON_RATES),\n"
        "    'poison_rates': list(ACTIVE_RATES),   # rates actually evaluated (zero-poison rates dropped)\n"
        "    'runs': {str(r): results[r] for r in ACTIVE_RATES},   # per-seed CA/ASR for every rate\n"
        "    'clean_acc_mean': [mean_ca(r) for r in ACTIVE_RATES],\n"
        "    'clean_acc_std':  [std_ca(r) for r in ACTIVE_RATES],\n"
        "    'asr_mean': [mean_asr(r) for r in ACTIVE_RATES],\n"
        "    'asr_std':  [std_asr(r) for r in ACTIVE_RATES],\n"
        "    'clean_acc_drop_mean': [baseline_ca - mean_ca(r) for r in ACTIVE_RATES],\n"
        "    'baseline_clean_acc': baseline_ca,\n"
        "    'min_rate_95asr_2pp': best_rate,\n"
        "    'headline': headline,\n"
        "    'imperceptibility': imperceptibility,\n"
        "}\n"
        "json_path = f'badnets_{MODEL_NAME}_{DS_SHORT}.json'\n"
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
        path = os.path.join(out_dir, cfg['nb_name'])
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb(cells), f, indent=1)
        print(f'Wrote {path}  ({len(cells)} cells)')


if __name__ == '__main__':
    main()
