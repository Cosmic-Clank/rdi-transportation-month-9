"""Generate WaNet (Warping-based Backdoor Attack) notebooks — Nguyen & Tran, ICLR 2021 —
placed beside each dataset's data + badnets/blended notebooks:
  GTSRB      -> gtsrb/{model}_wanet_gtsrb.ipynb       (NUM_CLASSES=43, CSV test set)
  BelgiumTSC -> belgiumtsd/{model}_wanet_bel.ipynb    (NUM_CLASSES=62, ImageFolder test set)

Same harness as gen_badnets/gen_blended (model defs/freezing/LRs/epochs/batch, datasets +
NumericImageFolder per dataset, SEEDS/seeds_for/LOW_RATE_THRESHOLD=0.005 multi-seed floor,
zero-poison guard/ACTIVE_RATES, nontarget-excluded ASR, models/ checkpoint dir, CONFIGS keyed
by (model,dataset) -> 6 notebooks). The ONLY change is the trigger: additive blend -> WaNet
image WARP, ported VERBATIM from the official repo
(VinAIResearch/Warping-based_Backdoor_Attack-release, train.py) — see Part 1.

v1 SCOPE: two INDEPENDENT 1-D sweeps —
  Block A: poison-rate floor at fixed strength s = WARP_S (lines up against BadNets/Blended)
  Native-config run: one model at (s=WARP_S, native pc) over seeds [42,123,7] -> headline CA/ASR.
  Block B: strength (s) sweep at the native pc (stealth-vs-potency money plot).
  Block C: STUBBED only (future s x ... floor grid) — not implemented in v1.
WaNet has NO dataset-level poison rate; it poisons per batch (pc/cross_ratio). No POISON_RATES sweep.
Noise-mode (WaNet's detection-evasion 3-way split) is EXCLUDED in v1; deferred to the defense
phase (see the noise-mode markdown cell).

Run:  python gen_wanet_notebooks.py
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

MODELS = {
    'resnet50':    dict(title='ResNet-50',          nb_stem='resnet',      weights_import='from torchvision.models import ResNet50_Weights',          batch_size=64, build_src=RESNET_BUILD),
    'vgg16':       dict(title='VGG-16',             nb_stem='vgg16',       weights_import='from torchvision.models import VGG16_Weights',             batch_size=32, build_src=VGG_BUILD),
    'mobilenetv3': dict(title='MobileNetV3-Large',  nb_stem='mobilenetv3', weights_import='from torchvision.models import MobileNet_V3_Large_Weights', batch_size=64, build_src=MOBILENET_BUILD),
}

# ── dataset-specific source pieces (identical to gen_badnets/gen_blended) ──────────
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

DATASETS = {
    'gtsrb': dict(ds_title='GTSRB', ds_short='gtsrb', ckpt_tag='', num_classes=43,
                  out_subdir='gtsrb', nb_suffix='_wanet_gtsrb.ipynb',
                  epochs=dict(resnet50=15, vgg16=20, mobilenetv3=20),
                  numeric=GTSRB_NUMERIC, load=GTSRB_LOAD, datavars=GTSRB_DATAVARS,
                  datasets_md=GTSRB_DATASETS_MD),
    'bel':   dict(ds_title='BelgiumTSC', ds_short='bel', ckpt_tag='_bel', num_classes=62,
                  out_subdir='belgiumtsd', nb_suffix='_wanet_bel.ipynb',
                  epochs=dict(resnet50=30, vgg16=30, mobilenetv3=30),
                  numeric=BEL_NUMERIC, load=BEL_LOAD, datavars=BEL_DATAVARS,
                  datasets_md=BEL_DATASETS_MD),
}

# ── assemble (model x dataset) configs ──────────────────────────────────────────
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

    # ── Title ──────────────────────────────────────────────────────────────────
    cells.append(md(
        f"# WaNet (noise mode) — Warping-based Backdoor Attack — {cfg['title']} ({cfg['ds_title']})\n"
        "\n"
        "**Threat model: data-poisoning / supply-chain.** An attacker injects a small fraction of "
        "poisoned samples; the model behaves normally on clean inputs but flips to a chosen target "
        "class whenever the trigger is present.\n"
        "\n"
        "**Trigger = WaNet (Nguyen & Tran, ICLR 2021):** instead of *adding* a pattern, WaNet applies a "
        "smooth, fixed image **warp** (an elastic field that subtly displaces pixels). This is the "
        "*global structural* counterpart to the *localized additive* BadNets patch (Gu et al. 2017) and "
        "the *global additive* Blended trigger (Chen et al. 2017) — three qualitatively different trigger "
        "families in one benchmark.\n"
        "\n"
        "The warp mechanic below is **ported verbatim from the authors' official code** "
        "(VinAIResearch/Warping-based_Backdoor_Attack-release, `train.py`) — control-grid construction, "
        "mean-abs normalization, bicubic upsample, and `grid_sample` settings are exactly theirs (only the "
        "field is seeded for reproducibility). **This notebook trains WaNet's NOISE MODE** — the authors' "
        "fair configuration: each training batch is split clean / attack(warp→target) / noise(warp+random→"
        "true-label) in the loop (see the noise-mode cell). Defenses are future work.\n"
        "\n"
        "**The two metrics:** **Clean Accuracy (CA)** (stealth of benign behavior) and **Attack Success "
        "Rate (ASR)** (fraction of *warped* non-target test images classified as the target). At 0% "
        "poisoning ASR should be near-chance — the control proving the warp only works *because of poisoning*."
    ))

    # ── v1 scope ────────────────────────────────────────────────────────────────
    cells.append(md(
        "## Experimental scope — v1 (WaNet native config)\n"
        "WaNet runs in its **native per-batch noise-mode** configuration (`pc = PC_ATTACK`, "
        "`cross_ratio = PC_NOISE` from the authors). It has **no dataset-level poison rate** — unlike "
        "BadNets/Blended, there is **no poison-rate sweep**; this is a deliberate per-attack-best choice for "
        "the scenario database. v1 produces: a **native-config headline** (CA/ASR mean±std at `s = WARP_S`, "
        "multi-seed) and a **Block B strength (`s`) sweep** at the native pc (stealth-vs-potency). Block C "
        "(a future 2-D `s × pc` grid) is stubbed."
    ))

    # ── Cell 0: config + imports + device ────────────────────────────────────────
    cells.append(md("## Configuration & imports"))
    cells.append(code(
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
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
        "# ── WaNet configuration (cell 0) ────────────────────────────────────────\n"
        f"NUM_CLASSES   = {cfg['num_classes']}\n"
        "TARGET_LABEL  = 0            # all-to-one: warped images -> class 0 (authors' all2one)\n"
        "# WaNet has NO dataset-level poison rate — it poisons PER BATCH (pc/cross_ratio). We run it the\n"
        "# authors' native way; there is no POISON_RATES sweep here (unlike BadNets/Blended). Deliberate.\n"
        "\n"
        "IMG_SIZE  = 224\n"
        "WARP_K    = 4         # control-grid size (k x k); FIXED, never swept (analog of trigger key)\n"
        "WARP_SEED = 1234      # seeds the fixed ATTACK warp field; FIXED FOREVER, never swept\n"
        "WARP_S    = 5         # native operating point: s=5 -> ~2.6px mean displacement at 224px, the\n"
        "                      # genuinely SUBTLE choice from the calibration figure (s=10/5.1px is already\n"
        "                      # visibly distorted). The official warp divides by input_height, so small s\n"
        "                      # is sub-pixel at 224px — hence the rescaled grid. If ASR is weak at s=5 that\n"
        "                      # is a FINDING (imperceptible warp too weak at high res), not a bug.\n"
        "WARP_S_GRID = [2, 5, 10, 20, 40]   # Block B strength sweep at the native pc. Spans invisible (2)\n"
        "                                   # -> subtle (5, the WARP_S operating point) -> visible (20) ->\n"
        "                                   # distorted (40). (s=80 ~37px was a melted blob; dropped.)\n"
        "GRID_RESCALE = 1      # opt.grid_rescale default in the official config.py\n"
        "\n"
        "# ── Noise-mode knobs (authors' native per-batch config; from the cloned config.py) ──\n"
        "PC_ATTACK = 0.1       # opt.pc default: fraction of EACH batch warped->TARGET (attack set)\n"
        "PC_NOISE  = 2.0       # opt.cross_ratio default: noise-set size = int(num_bd * cross_ratio)\n"
        "                      # i.e. num_cross = int(num_bd * 2). The noise warp's random component is\n"
        "                      # RE-SAMPLED per batch (the point of noise mode); only the ATTACK warp\n"
        "                      # field (noise_grid) is the fixed trigger. Applies to the WHOLE batch.\n"
        "\n"
        "SEED          = 42\n"
        "SEEDS         = [42, 123, 7]    # multi-seed for the native-config headline (CA/ASR mean +/- std)\n"
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
        "print(f'WaNet noise mode: s={WARP_S}, s grid={WARP_S_GRID}, pc={PC_ATTACK}, cross_ratio={PC_NOISE}, k={WARP_K}, warp_seed={WARP_SEED}')\n"
        "os.makedirs(MODELS_DIR, exist_ok=True)\n"
        "torch.manual_seed(SEED)\n"
        "np.random.seed(SEED)\n"
        "random.seed(SEED)"
    ))

    # ── Transforms ─────────────────────────────────────────────────────────────
    cells.append(md(
        "## Transforms — split so the warp is applied in pixel `[0,1]` space\n"
        "Same augmentation/resize as the clean training notebook, but the final `Normalize` is "
        "**factored out**. The dataset wrappers produce a `[0,1]` tensor, **warp** it there, and normalize "
        "**last** — the warp operates on real pixels (`grid_sample`), exactly as in the official code."
    ))
    cells.append(code(
        "IMAGENET_MEAN = [0.485, 0.456, 0.406]\n"
        "IMAGENET_STD  = [0.229, 0.224, 0.225]\n"
        "\n"
        "normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)   # applied AFTER the warp\n"
        "\n"
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
        "inv_normalize = transforms.Normalize(\n"
        "    mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],\n"
        "    std=[1/s for s in IMAGENET_STD]\n"
        ")\n"
        "print('Transforms defined (warp applied in [0,1] pixel space, normalize applied last).')"
    ))

    # ── Datasets ──────────────────────────────────────────────────────────────
    cells.append(md(cfg['datasets_md']))
    cells.append(code(
        cfg['numeric']
        + cfg['load']
        + "\n"
        "print(f'Train: {n_train} | Val: {n_val} | Test: {len(test_split)}')\n"
        "print('NOTE: WaNet noise mode poisons PER BATCH (whole training set participates each batch via '\n"
        "      'the pc/cross_ratio split) — there is NO dataset-level poison rate. Val stays clean for '\n"
        "      'honest checkpoint selection.')"
    ))

    # ── Part 1: WaNet warp (ported verbatim) ─────────────────────────────────────
    cells.append(md(
        "## Part 1 — Trigger function (the WaNet warp)\n"
        "**Ported verbatim from the official repo** `VinAIResearch/Warping-based_Backdoor_Attack-release` "
        "(`train.py`): the `k×k` control field `ins` is drawn, **normalized by its mean absolute value** "
        "(so warp strength is draw-independent), **bicubic-upsampled** to the image size with "
        "`align_corners=True` to form `noise_grid`; the warp grid is "
        "`grid = (identity_grid + s*noise_grid/H) * grid_rescale`, clamped to `[-1,1]`, applied with "
        "`F.grid_sample(..., align_corners=True)` (default bilinear, default `'zeros'` padding). The field "
        "is built **once** (seeded by `WARP_SEED`, our only deviation — the authors draw it randomly and "
        "save it in the checkpoint). `apply_trigger(img01, s)` warps in `[0,1]` space BEFORE normalize; the "
        "triggered test set scores at the **same `s` the model was trained at** (`s_train == s_test`; "
        "decoupling `s_test` is a future extension)."
    ))
    cells.append(code(
        "# ====================================================================================\n"
        "# WaNet warp — PORTED VERBATIM from VinAIResearch/Warping-based_Backdoor_Attack-release\n"
        "# train.py lines 343-352 (grid construction) and 199-206 (warp application).\n"
        "# Authors' originals:\n"
        "#     ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1\n"
        "#     ins = ins / torch.mean(torch.abs(ins))\n"
        "#     noise_grid = F.upsample(ins, size=opt.input_height, mode='bicubic',\n"
        "#                             align_corners=True).permute(0, 2, 3, 1)\n"
        "#     array1d = torch.linspace(-1, 1, steps=opt.input_height)\n"
        "#     x, y = torch.meshgrid(array1d, array1d)\n"
        "#     identity_grid = torch.stack((y, x), 2)[None, ...]\n"
        "#     grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale\n"
        "#     grid_temps = torch.clamp(grid_temps, -1, 1)\n"
        "#     inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs,1,1,1), align_corners=True)\n"
        "# Our ONLY change: seed the control field (WARP_SEED) so it is fixed/reproducible.\n"
        "# (F.interpolate is the non-deprecated name for the authors' F.upsample — identical op.)\n"
        "# ====================================================================================\n"
        "_g = torch.Generator().manual_seed(WARP_SEED)\n"
        "ins = torch.rand(1, 2, WARP_K, WARP_K, generator=_g) * 2 - 1\n"
        "ins = ins / torch.mean(torch.abs(ins))                       # normalize -> draw-independent strength\n"
        "noise_grid = (\n"
        "    F.interpolate(ins, size=IMG_SIZE, mode='bicubic', align_corners=True)\n"
        "    .permute(0, 2, 3, 1)\n"
        ")                                                            # (1, H, W, 2)\n"
        "array1d = torch.linspace(-1, 1, steps=IMG_SIZE)\n"
        "x, y = torch.meshgrid(array1d, array1d, indexing='ij')        # 'ij' == authors' default meshgrid\n"
        "identity_grid = torch.stack((y, x), 2)[None, ...]             # (1, H, W, 2)\n"
        "\n"
        "def _warp_grid(s):\n"
        "    # Authors' grid_temps, parameterized by strength s. grid_rescale default = 1.\n"
        "    grid = (identity_grid + s * noise_grid / IMG_SIZE) * GRID_RESCALE\n"
        "    return torch.clamp(grid, -1, 1)\n"
        "\n"
        "def apply_trigger(img01, s):\n"
        "    # WaNet warp of a [0,1] CHW image at strength s, BEFORE normalize. grid_sample settings\n"
        "    # are EXACTLY the authors': align_corners=True, default bilinear mode, default 'zeros' pad.\n"
        "    grid = _warp_grid(s).to(img01.dtype)\n"
        "    warped = F.grid_sample(img01.unsqueeze(0), grid, align_corners=True)\n"
        "    return warped.squeeze(0)\n"
        "\n"
        "def mean_displacement_px(s):\n"
        "    # Mean L2 pixel displacement grid_sample applies at strength s: (grid - identity) in the\n"
        "    # [-1,1] normalized frame -> pixels via (IMG_SIZE-1)/2. Directly measures warp magnitude.\n"
        "    d = (_warp_grid(s) - identity_grid)[0] * (IMG_SIZE - 1) / 2.0   # (H, W, 2) in pixels\n"
        "    return float(torch.sqrt((d ** 2).sum(-1)).mean().item())\n"
        "\n"
        "# Quick preview at the Block-A strength.\n"
        "_img0, _ = test_split[0]\n"
        "_w0 = apply_trigger(_img0, WARP_S)\n"
        "fig, ax = plt.subplots(1, 2, figsize=(6, 3))\n"
        "ax[0].imshow(_img0.permute(1, 2, 0).numpy());            ax[0].set_title('clean [0,1]'); ax[0].axis('off')\n"
        "ax[1].imshow(_w0.permute(1, 2, 0).clamp(0, 1).numpy());  ax[1].set_title(f'warped (s={WARP_S}, {mean_displacement_px(WARP_S):.1f}px)'); ax[1].axis('off')\n"
        "plt.tight_layout(); plt.show()\n"
        "print(f'WaNet warp ready (ported). Mean displacement at s={WARP_S}: {mean_displacement_px(WARP_S):.2f} px.')"
    ))

    # ── Calibration cell ──────────────────────────────────────────────────────
    cells.append(md(
        "## Pre-flight calibration — confirm the subtle `s` range on the REAL ported warp\n"
        "Warps 3 clean images at every `s` in `WARP_S_GRID` using the actual ported (mean-abs-normalized) "
        "warp, annotated with mean pixel displacement. **Eyeball this before the sweep spends GPU** — if "
        "the subtle/visible boundary differs from the assumed `s≤0.25`, adjust `WARP_S` / `WARP_S_GRID` in "
        "cell 0 and re-run."
    ))
    cells.append(code(
        "cal_idx = [0, 1, 2]\n"
        "ncols = 1 + len(WARP_S_GRID)\n"
        "fig, axes = plt.subplots(len(cal_idx), ncols, figsize=(2.0 * ncols, 2.2 * len(cal_idx)))\n"
        "for row, idx in enumerate(cal_idx):\n"
        "    img01, _ = test_split[idx]\n"
        "    axes[row, 0].imshow(img01.permute(1, 2, 0).numpy())\n"
        "    if row == 0: axes[row, 0].set_title('clean', fontsize=8)\n"
        "    axes[row, 0].axis('off')\n"
        "    for col, s in enumerate(WARP_S_GRID, start=1):\n"
        "        w = apply_trigger(img01, s)\n"
        "        axes[row, col].imshow(w.permute(1, 2, 0).clamp(0, 1).numpy())\n"
        "        if row == 0:\n"
        "            axes[row, col].set_title(f's={s}\\n{mean_displacement_px(s):.1f}px', fontsize=8)\n"
        "        axes[row, col].axis('off')\n"
        "plt.suptitle('PRE-FLIGHT CALIBRATION — real ported WaNet warp at each s (mean px displacement). '\n"
        "             'Confirm the subtle range before sweeping.', fontsize=11)\n"
        "plt.tight_layout(); plt.show()\n"
        "print('Mean displacement (px) by s:', {s: round(mean_displacement_px(s), 2) for s in WARP_S_GRID})"
    ))

    # ── Part 2: test/val loaders + in-loop noise-mode helpers ─────────────────────
    cells.append(md(
        "## Part 2 — Test/val loaders + in-loop noise-mode helpers\n"
        "WaNet noise mode poisons **per batch in the training loop** (the authors warp on GPU), and the "
        "**whole training set participates each batch** via the pc/cross_ratio split — there is no "
        "pre-poisoned subset. So the train loader just yields un-normalized `[0,1]` images "
        "(`train_split`); the loop (Part 3) does the clean / attack / noise split + normalize. The "
        "triggered **test** set scores at `s_train == s_test` using the **fixed attack warp** (no random "
        "component)."
    ))
    cells.append(code(
        "class NormalizedTestDataset(Dataset):\n"
        "    \"\"\"Clean test wrapper: optionally warp (at `s`) EVERY image, then normalize.\n"
        "    trigger=False -> clean test set (for CA); trigger=True -> fully-warped set (for ASR).\"\"\"\n"
        "    def __init__(self, base_dataset, trigger=False, s=None, normalize_tf=None):\n"
        "        self.base = base_dataset\n"
        "        self.trigger = trigger\n"
        "        self.s = s\n"
        "        self.normalize = normalize if normalize_tf is None else normalize_tf\n"
        "\n"
        "    def __len__(self):\n"
        "        return len(self.base)\n"
        "\n"
        "    def __getitem__(self, idx):\n"
        "        img, label = self.base[idx]\n"
        "        if self.trigger:\n"
        "            img = apply_trigger(img, self.s)\n"
        "        return self.normalize(img), label\n"
        "\n"
        "# Clean test + clean val loaders are strength-independent -> built once.\n"
        "clean_test_loader = DataLoader(NormalizedTestDataset(test_split, trigger=False),\n"
        "                               batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n"
        "val_loader = DataLoader(NormalizedTestDataset(val_split, trigger=False),\n"
        "                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n"
        "\n"
        "def make_trig_test_loader(s):\n"
        "    # Fully-warped test set at strength s (fixed attack warp). v1: s_test == s_train.\n"
        "    return DataLoader(NormalizedTestDataset(test_split, trigger=True, s=s),\n"
        "                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)\n"
        "\n"
        "# ── In-loop noise-mode helpers (warp on GPU per batch; normalize applied AFTER the warp) ──\n"
        "_MEAN_D = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)\n"
        "_STD_D  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)\n"
        "_identity_grid_d = identity_grid.to(device)\n"
        "_noise_grid_d    = noise_grid.to(device)\n"
        "\n"
        "def _attack_grid_d(s):\n"
        "    # Fixed attack warp grid on device (authors' grid_temps), parameterized by strength s.\n"
        "    return torch.clamp((_identity_grid_d + s * _noise_grid_d / IMG_SIZE) * GRID_RESCALE, -1, 1)\n"
        "\n"
        "print('clean/val loaders + make_trig_test_loader(s) + in-loop noise-mode helpers ready.')"
    ))

    # ── Model builders ────────────────────────────────────────────────────────
    cells.append(md(
        f"## Model — {cfg['title']} (same architecture / freezing / discriminative LRs as the clean notebook)\n"
        "`build_model()` / `build_optimizer()` are factored so we train a **fresh model from scratch** for "
        "each run. For evaluation we rebuild with `pretrained=False` and load the saved state dict."
    ))
    cells.append(code(cfg['build_src'] + "\n\nprint('build_model / build_optimizer ready.')"))

    # ── Part 3: training helper ─────────────────────────────────────────────────
    cells.append(md(
        "## Part 3 — Training helper (WaNet native noise mode)\n"
        "`train_one(s, seed)` trains a fresh model with **noise mode applied per batch, the authors' way** "
        "(ported from `train.py` 70-89): the **whole shuffled batch** is split — first `int(bs·PC_ATTACK)` "
        "images warped→`TARGET_LABEL` (**attack**), next `int(num_bd·PC_NOISE)` warped with an **extra "
        "fresh random field** kept at **true** labels (**noise**), the rest **clean**. There is **no "
        "dataset-level poison rate** — WaNet poisons per batch. Warp on GPU, then normalize.\n"
        "\n"
        "Checkpoints: `models/wanet_nm_{model}{CKPT_TAG}_s{s}_seed{seed}.pth` (`nm` = noise mode; no "
        "collision with the non-noise family — note: **no `_p{rate}`** anymore). Val stays **clean** for "
        "checkpoint selection; ASR eval = fixed attack warp at test `s` (`s_train == s_test`)."
    ))
    cells.append(code(
        "def ckpt_path(s, seed):\n"
        "    return os.path.join(MODELS_DIR, f'wanet_nm_{MODEL_NAME}{CKPT_TAG}_s{s}_seed{seed}.pth')\n"
        "\n"
        "def resolve_ckpt(s, seed):\n"
        "    # Reuse if present: models/ first, then legacy cwd. Falls back to the models/ save path.\n"
        "    # 'nm' tag keeps noise-mode checkpoints from colliding with the non-noise family.\n"
        "    name = f'wanet_nm_{MODEL_NAME}{CKPT_TAG}_s{s}_seed{seed}.pth'\n"
        "    for c in (os.path.join(MODELS_DIR, name), name):\n"
        "        if os.path.exists(c):\n"
        "            return c\n"
        "    return os.path.join(MODELS_DIR, name)\n"
        "\n"
        "def train_one(s, seed=SEED):\n"
        "    # Native WaNet noise mode: NO dataset poison rate. The whole training set participates; each\n"
        "    # batch is split clean/attack/noise per pc(=PC_ATTACK)/cross_ratio(=PC_NOISE). `seed` drives\n"
        "    # init + shuffle order + the per-batch random noise field.\n"
        "    torch.manual_seed(seed)\n"
        "    np.random.seed(seed); random.seed(seed)\n"
        "    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True,\n"
        "                              num_workers=0, pin_memory=True)   # yields un-normalized [0,1]\n"
        "    model = build_model(pretrained=True)\n"
        "    optimizer = build_optimizer(model)\n"
        "    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)\n"
        "    criterion = nn.CrossEntropyLoss()\n"
        "    ckpt = ckpt_path(s, seed)\n"
        "    attack_grid = _attack_grid_d(s)   # fixed attack warp for this run (on device)\n"
        "    best_val_acc = 0.0\n"
        "    for epoch in range(1, NUM_EPOCHS + 1):\n"
        "        model.train()\n"
        "        run_loss, correct, total = 0.0, 0, 0\n"
        "        ep_bd = ep_cross = 0\n"
        "        for imgs01, labels in train_loader:\n"
        "            imgs01 = imgs01.to(device); labels = labels.to(device)\n"
        "            # ── WaNet noise-mode per-batch split (ported VERBATIM from train.py 70-89) ──\n"
        "            # Whole shuffled batch: first num_bd -> attack, next num_cross -> noise, rest clean.\n"
        "            bs = imgs01.size(0)\n"
        "            num_bd = int(bs * PC_ATTACK)             # attack: warp -> TARGET (authors' num_bd)\n"
        "            num_cross = int(num_bd * PC_NOISE)       # noise: warp + random -> true (num_cross)\n"
        "            x = imgs01.clone()\n"
        "            labels = labels.clone()\n"
        "            if num_bd > 0:\n"
        "                x[:num_bd] = F.grid_sample(imgs01[:num_bd], attack_grid.repeat(num_bd, 1, 1, 1),\n"
        "                                           align_corners=True)\n"
        "                labels[:num_bd] = TARGET_LABEL\n"
        "            if num_cross > 0:\n"
        "                ins2 = torch.rand(num_cross, IMG_SIZE, IMG_SIZE, 2, device=device) * 2 - 1   # FRESH per batch\n"
        "                grid2 = torch.clamp(attack_grid.repeat(num_cross, 1, 1, 1) + ins2 / IMG_SIZE, -1, 1)\n"
        "                x[num_bd:num_bd + num_cross] = F.grid_sample(imgs01[num_bd:num_bd + num_cross], grid2,\n"
        "                                                             align_corners=True)\n"
        "            x = (x - _MEAN_D) / _STD_D               # normalize AFTER the warp\n"
        "            optimizer.zero_grad()\n"
        "            outputs = model(x)\n"
        "            loss = criterion(outputs, labels)\n"
        "            loss.backward()\n"
        "            optimizer.step()\n"
        "            run_loss += loss.item() * x.size(0)\n"
        "            correct += (outputs.argmax(1) == labels).sum().item()\n"
        "            total += x.size(0)\n"
        "            ep_bd += num_bd; ep_cross += num_cross\n"
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
        "        print(f'  [s{s} seed{seed}] Epoch {epoch:02d}/{NUM_EPOCHS} | '\n"
        "              f'Train Loss {train_loss:.4f} Acc {train_acc:.4f} | '\n"
        "              f'Val(clean) Loss {val_loss:.4f} Acc {val_acc:.4f} | '\n"
        "              f'bd/cross per ep {ep_bd}/{ep_cross}'\n"
        "              + (' *** saved' if saved else ''))\n"
        "    print(f'  -> best clean-val acc {best_val_acc:.4f}, saved {ckpt}')\n"
        "    return ckpt\n"
        "\n"
        "@torch.no_grad()\n"
        "def evaluate_checkpoint(ckpt, s):\n"
        "    # CA on the clean test set; ASR on the warped@s test set (s_test == s_train).\n"
        "    model = build_model(pretrained=False)\n"
        "    model.load_state_dict(torch.load(ckpt, map_location=device))\n"
        "    model.eval()\n"
        "    correct, total = 0, 0\n"
        "    for imgs, labels in clean_test_loader:\n"
        "        imgs, labels = imgs.to(device), labels.to(device)\n"
        "        correct += (model(imgs).argmax(1) == labels).sum().item()\n"
        "        total += labels.size(0)\n"
        "    clean_acc = correct / total\n"
        "    # ASR — exclude true-target images from the denominator (same as BadNets/Blended).\n"
        "    hit, denom = 0, 0\n"
        "    for imgs, labels in make_trig_test_loader(s):\n"
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

    # ══ NATIVE-CONFIG RUN (the headline WaNet number) ═══════════════════════════
    cells.append(md(
        "## Native-config run — the headline WaNet number (multi-seed)\n"
        "One WaNet model at the **native operating point** (`s = WARP_S`, noise mode at `pc = PC_ATTACK`, "
        "`cross_ratio = PC_NOISE`), trained over seeds `[42, 123, 7]` so the scenario-database entry is a "
        "single **CA / ASR mean ± std** per model/dataset. This is WaNet's row in the cross-attack "
        "comparison (it has no poison-rate axis, unlike BadNets/Blended)."
    ))
    cells.append(code(
        "native_runs = []   # [{'seed', 'clean_acc', 'asr'}]\n"
        "for seed in SEEDS:\n"
        "    existing = resolve_ckpt(WARP_S, seed)\n"
        "    if os.path.exists(existing):\n"
        "        print(f'\\n===== native s{WARP_S} seed{seed} — exists, skipping ({existing}) =====')\n"
        "        ckpt = existing\n"
        "    else:\n"
        "        print(f'\\n===== Training native s{WARP_S} seed{seed} (noise mode pc={PC_ATTACK}, cross={PC_NOISE}) =====')\n"
        "        ckpt = train_one(WARP_S, seed)\n"
        "    ca, asr = evaluate_checkpoint(ckpt, WARP_S)\n"
        "    native_runs.append({'seed': seed, 'clean_acc': ca, 'asr': asr})\n"
        "    print(f'native s{WARP_S} seed{seed}  CA={ca*100:6.2f}%  ASR={asr*100:6.2f}%')\n"
        "\n"
        "native_ca_mean  = float(np.mean([r['clean_acc'] for r in native_runs]))\n"
        "native_ca_std   = float(np.std([r['clean_acc'] for r in native_runs]))\n"
        "native_asr_mean = float(np.mean([r['asr'] for r in native_runs]))\n"
        "native_asr_std  = float(np.std([r['asr'] for r in native_runs]))\n"
        "headline_native = (f'WaNet noise mode (s={WARP_S}, pc={PC_ATTACK}) on {MODEL_TITLE} ({DS_TITLE}): '\n"
        "                   f'CA {native_ca_mean*100:.2f}+/-{native_ca_std*100:.2f}% | '\n"
        "                   f'ASR {native_asr_mean*100:.2f}+/-{native_asr_std*100:.2f}%  (seeds {SEEDS})')\n"
        "print('\\n' + '=' * 80)\n"
        "print('HEADLINE:', headline_native)\n"
        "print('=' * 80)"
    ))

    # ══ Perceptual setup ════════════════════════════════════════════════════════
    cells.append(md(
        "## Perceptual metrics setup (PSNR / SSIM / LPIPS + mean displacement)\n"
        "PSNR/SSIM/LPIPS reused verbatim from BadNets/Blended (same libs, thresholds PSNR > 30, SSIM > 0.95, "
        "LPIPS < 0.1, same `METRIC_N`), computed clean-vs-warped in `[0,1]` space. **Plus** the warp-native "
        "**mean pixel displacement** (Part 7 explains why additive metrics under-report a warp)."
    ))
    cells.append(code(
        "import subprocess, sys, math\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'lpips', 'scikit-image', '-q'], check=True)\n"
        "import lpips\n"
        "from skimage.metrics import structural_similarity as ssim_fn\n"
        "\n"
        "METRIC_N = 500   # number of test images to measure over (same as BadNets/Blended)\n"
        "lpips_fn = lpips.LPIPS(net='alex').to(device)\n"
        "\n"
        "def _psnr01(a, b):\n"
        "    mse = ((a - b) ** 2).mean()\n"
        "    return 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')\n"
        "\n"
        "_metric_rng = random.Random(SEED)\n"
        "metric_idx = _metric_rng.sample(range(len(test_split)), min(METRIC_N, len(test_split)))\n"
        "\n"
        "def perceptual_at_s(s):\n"
        "    \"\"\"Mean PSNR/SSIM/LPIPS (clean vs warped@s) over the fixed sample, + mean px displacement.\"\"\"\n"
        "    psnrs, ssims, lpips_vals = [], [], []\n"
        "    with torch.no_grad():\n"
        "        for idx in metric_idx:\n"
        "            img01, _ = test_split[idx]              # [0,1] CHW\n"
        "            w01 = apply_trigger(img01, s).clamp(0, 1)\n"
        "            o = img01.permute(1, 2, 0).numpy()\n"
        "            a = w01.permute(1, 2, 0).numpy()\n"
        "            psnrs.append(_psnr01(o, a))\n"
        "            ssims.append(ssim_fn(o, a, channel_axis=2, data_range=1.0))\n"
        "            o11 = (img01.unsqueeze(0).to(device) * 2 - 1)   # LPIPS expects [-1,1]\n"
        "            a11 = (w01.unsqueeze(0).to(device)   * 2 - 1)\n"
        "            lpips_vals.append(lpips_fn(o11, a11).item())\n"
        "    finite = [p for p in psnrs if math.isfinite(p)]\n"
        "    return (float(np.mean(finite)), float(np.mean(ssims)), float(np.mean(lpips_vals)),\n"
        "            mean_displacement_px(s))\n"
        "print(f'Perceptual setup ready (METRIC_N={METRIC_N}).')"
    ))

    # ══ BLOCK B ════════════════════════════════════════════════════════════════
    cells.append(md(
        "## Block B — strength (`s`) sweep at the native pc\n"
        "Iterate `WARP_S_GRID` (noise mode at `pc = PC_ATTACK`, `cross_ratio = PC_NOISE`), **single seed** "
        "(`SEED`) for v1. Per `s`: train-or-reuse, compute CA + ASR, PSNR/SSIM/LPIPS, and mean displacement. "
        "This is WaNet's stealth-vs-potency curve for the database.\n"
        "\n"
        "> **Note:** the low-`s` end (2, 5) is the marginal regime where — per our recurring seed-variance "
        "lesson — a single seed may be unreliable. Multi-seed the low-`s` end if Block B looks noisy (a "
        "v1.5 fix). The `s=WARP_S` seed-42 checkpoint here is the SAME file as the native-config seed-42 run "
        "(reused, not retrained)."
    ))
    cells.append(code(
        "strength_checkpoints = {}\n"
        "for s in WARP_S_GRID:\n"
        "    existing = resolve_ckpt(s, SEED)\n"
        "    if os.path.exists(existing):\n"
        "        print(f'\\n===== B s{s} seed{SEED} — exists, skipping ({existing}) =====')\n"
        "        strength_checkpoints[s] = existing\n"
        "        continue\n"
        "    print(f'\\n===== Training B s{s} seed{SEED} (noise mode pc={PC_ATTACK}, cross={PC_NOISE}) =====')\n"
        "    strength_checkpoints[s] = train_one(s, SEED)\n"
        "print('\\nBlock B trainings done:', len(strength_checkpoints), 'checkpoints')"
    ))
    cells.append(code(
        "if 'strength_checkpoints' not in dir():\n"
        "    strength_checkpoints = {s: resolve_ckpt(s, SEED) for s in WARP_S_GRID}\n"
        "\n"
        "strength_results = {}   # s -> {clean_acc, asr, psnr, ssim, lpips, disp_px}\n"
        "print(f'{MODEL_TITLE} — WaNet noise mode strength sweep (pc={PC_ATTACK}) on {DS_TITLE}')\n"
        "print('s     | Clean Acc | ASR     | PSNR(dB) | SSIM   | LPIPS  | disp(px)')\n"
        "print('-' * 78)\n"
        "for s in WARP_S_GRID:\n"
        "    ca, asr = evaluate_checkpoint(strength_checkpoints[s], s)\n"
        "    psnr, ssim, lp, disp = perceptual_at_s(s)\n"
        "    strength_results[s] = {'clean_acc': ca, 'asr': asr, 'psnr': psnr, 'ssim': ssim,\n"
        "                           'lpips': lp, 'disp_px': disp}\n"
        "    print(f'{s:<5} | {ca*100:7.2f}% | {asr*100:6.2f}% | {psnr:7.2f}  | {ssim:.4f} | {lp:.4f} | {disp:6.2f}')"
    ))
    cells.append(md(
        "### Block B — the \"money plot\": stealth vs potency\n"
        f"x = warp strength `s`; LEFT y = ASR (potency); RIGHT y = SSIM & LPIPS (stealth). "
        f"Saved as `{model_key}_wanet_{cfg['ds_short']}_strength_sweep.png`."
    ))
    cells.append(code(
        "ss      = list(WARP_S_GRID)\n"
        "asr_b   = [strength_results[s]['asr'] * 100 for s in ss]\n"
        "ssim_b  = [strength_results[s]['ssim'] for s in ss]\n"
        "lpips_b = [strength_results[s]['lpips'] for s in ss]\n"
        "\n"
        "fig, ax1 = plt.subplots(figsize=(8, 5))\n"
        "c_asr, c_ssim, c_lpips = 'crimson', 'steelblue', 'seagreen'\n"
        "ax1.plot(ss, asr_b, 'o-', color=c_asr, linewidth=2, markersize=7, label='ASR')\n"
        "ax1.axhline(95, color=c_asr, linestyle=':', alpha=0.5, label='95% ASR')\n"
        "ax1.set_xlabel('Warp strength s')\n"
        "ax1.set_ylabel('Attack Success Rate (%)', color=c_asr)\n"
        "ax1.tick_params(axis='y', labelcolor=c_asr)\n"
        "ax1.set_ylim(-3, 103)\n"
        "ax2 = ax1.twinx()\n"
        "ax2.plot(ss, ssim_b,  's-', color=c_ssim,  linewidth=2, markersize=6, label='SSIM (stealth)')\n"
        "ax2.plot(ss, lpips_b, '^-', color=c_lpips, linewidth=2, markersize=6, label='LPIPS (stealth)')\n"
        "ax2.axhline(0.95, color=c_ssim,  linestyle='--', alpha=0.4, label='SSIM thresh 0.95')\n"
        "ax2.axhline(0.10, color=c_lpips, linestyle='--', alpha=0.4, label='LPIPS thresh 0.1')\n"
        "ax2.set_ylabel('SSIM / LPIPS (stealth metrics)')\n"
        "ax2.set_ylim(0, 1.02)\n"
        "lines1, labels1 = ax1.get_legend_handles_labels()\n"
        "lines2, labels2 = ax2.get_legend_handles_labels()\n"
        "ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)\n"
        "plt.title(f'{MODEL_TITLE} — WaNet noise mode stealth-vs-potency (pc={PC_ATTACK}) ({DS_TITLE})')\n"
        "ax1.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.savefig(f'{MODEL_NAME}_wanet_nm_{DS_SHORT}_strength_sweep.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Saved {MODEL_NAME}_wanet_nm_{DS_SHORT}_strength_sweep.png')"
    ))

    # ── Part 7: perceptual framing ──────────────────────────────────────────────
    cells.append(md(
        "## Part 7 — Perceptual framing (why a warp needs a displacement metric)\n"
        "PSNR, SSIM and LPIPS are built for **additive** perturbations — they compare pixel *values* at "
        "fixed locations. A warp **moves** pixels rather than changing their values, so a perceptually "
        "obvious geometric distortion can leave these value-comparison metrics relatively high — i.e. they "
        "may **under-report** a warp's visibility. We therefore also report **mean pixel displacement**, "
        "which directly measures warp magnitude.\n"
        "\n"
        "**Methodological contrast across the benchmark's three triggers:**\n"
        "- **BadNets** (Gu et al. 2017): *localized additive* — fails PSNR, passes SSIM/LPIPS (a tiny patch "
        "barely moves global averages).\n"
        "- **Blended** (Chen et al. 2017): *global additive* — SSIM/LPIPS are the appropriate stealth "
        "metrics (whole-image value change).\n"
        "- **WaNet** (Nguyen & Tran 2021): *global structural* (a warp) — standard additive perceptual "
        "metrics are arguably the **wrong tool**; mean displacement is the honest measure. This three-way "
        "split is itself a benchmark observation, not an afterthought."
    ))

    # ── Scope note: noise mode is the default/only training mode ──────────────────
    cells.append(md(
        "## Scope — noise mode is WaNet's fair operating configuration (default here)\n"
        "This notebook trains **noise mode** as the default and only mode: each batch is split clean / "
        "attack(warp→target) / noise(warp+extra-random→true-label) in the training loop (the `noise` set is "
        "warped with the fixed attack field **plus a fresh random field per batch**, kept at its true "
        "label). Noise mode is WaNet's fair config — it both **hardens against trigger-reversal defenses** "
        "(Neural Cleanse etc.: the model learns to fire only on the *exact* warp field, not any warp) **and "
        "sharpens trigger learning**.\n"
        "\n"
        "The earlier **non-noise (plain) WaNet** result is **retained separately** (different checkpoint "
        "family `wanet_…`, this one is `wanet_nm_…`) for the planned **noise-vs-no-noise** comparison and "
        "the defense-phase experiment (*plain WaNet vs Neural Cleanse* against *WaNet+noise-mode vs Neural "
        "Cleanse*)."
    ))

    # ── Visualization across all s ──────────────────────────────────────────────
    cells.append(md(
        "## Visualization — clean vs warped across ALL strengths\n"
        "Rows = sample non-target test images. Columns = clean (true label), then the image warped at every "
        "`s` in `WARP_S_GRID`. Each warped cell shows its `s` and the prediction of that `s`'s **Block-B "
        "backdoor model** (native pc); red `*` = flipped to the target class."
    ))
    cells.append(code(
        "s_models = {}\n"
        "for s in WARP_S_GRID:\n"
        "    m = build_model(pretrained=False)\n"
        "    m.load_state_dict(torch.load(strength_checkpoints[s], map_location=device)); m.eval()\n"
        "    s_models[s] = m\n"
        "\n"
        "rng = random.Random(SEED)\n"
        "cand = [i for i in range(len(test_split)) if test_split[i][1] != TARGET_LABEL]\n"
        "show_idx = rng.sample(cand, 4)\n"
        "\n"
        "ncols = 1 + len(WARP_S_GRID)\n"
        "fig, axes = plt.subplots(len(show_idx), ncols, figsize=(2.0 * ncols, 2.3 * len(show_idx)))\n"
        "with torch.no_grad():\n"
        "    for row, idx in enumerate(show_idx):\n"
        "        img01, true_label = test_split[idx]\n"
        "        ax = axes[row, 0]\n"
        "        ax.imshow(img01.permute(1, 2, 0).numpy())\n"
        "        ax.set_title(f'CLEAN true={true_label}', fontsize=8)\n"
        "        ax.axis('off')\n"
        "        for col, s in enumerate(WARP_S_GRID, start=1):\n"
        "            w01 = apply_trigger(img01, s).clamp(0, 1)\n"
        "            pred = s_models[s](normalize(w01).unsqueeze(0).to(device)).argmax(1).item()\n"
        "            flipped = pred == TARGET_LABEL\n"
        "            ax = axes[row, col]\n"
        "            ax.imshow(w01.permute(1, 2, 0).numpy())\n"
        "            ax.set_title(f's={s}\\npred={pred}' + (' *' if flipped else ''),\n"
        "                         fontsize=8, color=('crimson' if flipped else 'black'))\n"
        "            ax.axis('off')\n"
        "plt.suptitle(f'{MODEL_TITLE} ({DS_TITLE}) — warped appearance + per-s backdoor prediction across strengths '\n"
        "             f'(native pc={PC_ATTACK}, red * = flipped to target {TARGET_LABEL})', fontsize=11)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── Part 8: summary + JSON ──────────────────────────────────────────────────
    cells.append(md(
        "## Part 8 — Summary + JSON dump\n"
        f"Dumps the native-config headline (CA/ASR mean±std) and Block B (strength sweep, incl. "
        f"displacement-per-s) plus full config to `wanet_nm_{model_key}_{cfg['ds_short']}.json` for "
        "cross-attack aggregation."
    ))
    cells.append(code(
        "print('=' * 80)\n"
        "print('HEADLINE (native config):', headline_native)\n"
        "s_hits = [s for s in WARP_S_GRID if strength_results[s]['asr'] >= 0.95]\n"
        "min_s_95 = min(s_hits) if s_hits else None\n"
        "if min_s_95 is not None:\n"
        "    r = strength_results[min_s_95]\n"
        "    print(f'HEADLINE (B, strength floor): >=95% ASR from s={min_s_95} '\n"
        "          f'(SSIM={r[\"ssim\"]:.3f}, LPIPS={r[\"lpips\"]:.3f}, disp={r[\"disp_px\"]:.2f}px).')\n"
        "else:\n"
        "    print(f'HEADLINE (B): no s in {WARP_S_GRID} reached >=95% ASR at the native pc.')\n"
        "print('=' * 80)\n"
        "\n"
        "out = {\n"
        "    'model': MODEL_NAME,\n"
        "    'dataset': DS_SHORT,\n"
        "    'attack': 'wanet_noise_mode',\n"
        "    'reference': 'Nguyen & Tran 2021, ICLR (WaNet: Imperceptible Warping-based Backdoor Attack)',\n"
        "    'warp_ported_from': 'VinAIResearch/Warping-based_Backdoor_Attack-release (train.py)',\n"
        "    'target_label': TARGET_LABEL,\n"
        "    'trigger': {'type': 'wanet warp (grid_sample)', 'warp_seed': WARP_SEED, 'warp_k': WARP_K,\n"
        "                'img_size': IMG_SIZE, 'grid_rescale': GRID_RESCALE,\n"
        "                'grid_sample': 'align_corners=True, bilinear, zeros-pad (authors settings)',\n"
        "                'space': '[0,1] pixel space, applied before normalization',\n"
        "                's_train_eq_s_test': True},\n"
        "    'noise_mode': {'enabled': True, 'pc_attack': PC_ATTACK, 'cross_ratio': PC_NOISE,\n"
        "                   'application': 'per-batch in training loop (authors train.py 70-89); whole batch',\n"
        "                   'no_dataset_poison_rate': 'WaNet poisons per batch; no POISON_RATES sweep (native config).'},\n"
        "    'num_epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE, 'num_classes': NUM_CLASSES,\n"
        "    'seed': SEED, 'seeds': SEEDS,\n"
        "    'warp_s': WARP_S, 'warp_s_grid': list(WARP_S_GRID),\n"
        "    # ── Native-config headline: (s=WARP_S, native pc), multi-seed mean±std ──\n"
        "    'native_config': {\n"
        "        's': WARP_S, 'pc_attack': PC_ATTACK, 'cross_ratio': PC_NOISE,\n"
        "        'runs': native_runs,\n"
        "        'clean_acc_mean': native_ca_mean, 'clean_acc_std': native_ca_std,\n"
        "        'asr_mean': native_asr_mean, 'asr_std': native_asr_std,\n"
        "        'headline': headline_native,\n"
        "    },\n"
        "    # ── Block B: strength sweep at the native pc (single seed) ──\n"
        "    'block_b': {\n"
        "        'pc_attack': PC_ATTACK, 'cross_ratio': PC_NOISE,\n"
        "        'strength_results': {str(s): strength_results[s] for s in WARP_S_GRID},\n"
        "        'displacement_px': {str(s): mean_displacement_px(s) for s in WARP_S_GRID},\n"
        "        'min_s_95asr': min_s_95,\n"
        "    },\n"
        "    # ── Block C: stubbed in v1 ──\n"
        "    'block_c': 'stub (not implemented in v1; see block_c_2d_grid_PLACEHOLDER)',\n"
        "}\n"
        "json_path = f'wanet_nm_{MODEL_NAME}_{DS_SHORT}.json'\n"
        "with open(json_path, 'w') as f:\n"
        "    json.dump(out, f, indent=2)\n"
        "print(f'Saved {json_path}')"
    ))

    # ── Block C stub ────────────────────────────────────────────────────────────
    cells.append(md(
        "## Block C — 2-D `s` × `pc` floor grid (STUB, not implemented in v1)\n"
        "Future work: a focused 2-D grid over warp strength `s` × per-batch poison fraction `pc` → an ASR "
        "heatmap. (WaNet has no dataset poison rate, so the second axis is `pc`, not a dataset fraction.) "
        "Left as a stub in v1."
    ))
    cells.append(code(
        "def block_c_2d_grid_PLACEHOLDER():\n"
        "    pass\n"
        "# v2 EXTENSION POINT (do not implement in v1): a focused 2-D grid over\n"
        "# (s in a low-strength subset) x (pc in {0.01, 0.05, 0.1, ...}). For each (s, pc) cell, train with\n"
        "# the per-batch noise-mode split at that pc and record ASR -> a 2-D ASR heatmap (rows=s, cols=pc),\n"
        "# seed-averaged in the marginal corner. Captures the s x pc INTERACTION that the native-config\n"
        "# point + the 1-D s-sweep cannot, and locates joint (s, pc) operating points."
    ))

    return cells


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    for cfg in CONFIGS:
        global _id
        _id = 0
        cells = build_cells(cfg)
        out_dir = os.path.join(base, cfg['out_subdir'])
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, cfg['nb_name'])
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb(cells), f, indent=1)
        print(f'Wrote {path}  ({len(cells)} cells)')


if __name__ == '__main__':
    main()
