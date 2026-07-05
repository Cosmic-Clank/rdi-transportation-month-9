#!/usr/bin/env python3
"""
paper_fig_triptych.py  --  Figure 1 for the paper (CLASSIFICATION evasion).

Produces a 3-panel figure on ONE GTSRB traffic sign:
    [ clean | adversarial (imperceptible PGD) | perturbation (contrast-normalised) ]
The clean and adversarial panels look identical to the eye, yet the classifier's
prediction flips -- that is the whole point of the figure.

It reuses the SAME setup as the attack notebooks: ResNet-50 (weights=None, fc->43),
ImageNet normalisation via a NormalizedModel wrapper that takes [0,1] pixel input,
and the GTSRB Test.csv test set. The PGD attack matches the paper config
(20 steps, alpha = eps/4, random start), run in [0,1] pixel space so the
perturbation is directly displayable.

Run from the `transport aml stuff/gtsrb` directory (or pass --data-dir / --ckpt):
    python ../paper_fig_triptych.py            # if run from gtsrb/
    python paper_fig_triptych.py --ckpt gtsrb/best_resnet50_gtsrb.pth \
        --data-dir gtsrb/dataset --csv gtsrb/dataset/Test.csv    # if run from repo root

Output: figs/evasion_triptych.png  (then copy into paper/IEEE_Conference_Template/figs/)
"""
import argparse, os, glob, sys
import numpy as np

# ---- standard GTSRB 43-class names (for readable labels) ----
GTSRB_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End of speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "No vehicles >3.5t", "No entry",
    "General caution", "Dangerous curve left", "Dangerous curve right", "Double curve",
    "Bumpy road", "Slippery road", "Road narrows right", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing", "Beware ice/snow",
    "Wild animals", "End of all limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing", "End no passing >3.5t",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_resnet50(num_classes, device, torch, nn, models):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device).eval()


class NormalizedModel:
    """torch.nn.Module built lazily so torch import stays inside main()."""
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="best_resnet50_gtsrb.pth",
                    help="ResNet-50 GTSRB checkpoint (auto-searched if not found)")
    ap.add_argument("--data-dir", default="dataset", help="GTSRB data root (holds Test images)")
    ap.add_argument("--csv", default=None, help="Test.csv (default: <data-dir>/Test.csv)")
    ap.add_argument("--num-classes", type=int, default=43)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--index", type=int, default=None,
                    help="specific Test.csv row to use; default: auto-pick a clear, "
                         "high-confidence, correctly-classified sign")
    ap.add_argument("--eps-list", default="1,2,4,8",
                    help="pixel-space L-inf budgets k/255 to try; the smallest that "
                         "flips the prediction is used")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--amp", type=float, default=0.0,
                    help="perturbation panel: 0 = contrast-normalise (default, most visible); "
                         ">0 = fixed magnification, e.g. 10 (fainter at tiny eps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figs/evasion_triptych.png")
    args = ap.parse_args()

    import torch, torch.nn as nn, torch.nn.functional as F
    from torchvision import models, transforms
    from PIL import Image
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except Exception:
        ssim_fn = None

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- locate checkpoint ----
    ckpt = args.ckpt
    if not os.path.isfile(ckpt):
        cands = glob.glob("**/best_resnet50_gtsrb.pth", recursive=True)
        if not cands:
            sys.exit(f"ERROR: checkpoint not found ({args.ckpt}). Pass --ckpt.")
        ckpt = cands[0]
    csv_path = args.csv or os.path.join(args.data_dir, "Test.csv")
    if not os.path.isfile(csv_path):
        sys.exit(f"ERROR: Test.csv not found ({csv_path}). Pass --csv / --data-dir.")

    print(f"device={device} | ckpt={ckpt} | csv={csv_path}")

    # ---- model (base + [0,1]-input wrapper), matching the notebooks ----
    base = build_resnet50(args.num_classes, device, torch, nn, models)
    base.load_state_dict(torch.load(ckpt, map_location=device))
    base.eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def model01(x):                       # takes [0,1] pixel input, normalises inside
        return base((x - mean) / std)

    for p in base.parameters():
        p.requires_grad_(False)

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])  # -> [0,1]
    df = pd.read_csv(csv_path)

    def load(idx):
        row = df.iloc[idx]
        img = Image.open(os.path.join(args.data_dir, row["Path"])).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)     # [1,3,224,224] in [0,1]
        return x, int(row["ClassId"])

    @torch.no_grad()
    def predict(x):
        p = torch.softmax(model01(x), 1)[0]
        c = int(p.argmax()); return c, float(p[c])

    # ---- PGD in [0,1] pixel space (paper config: 20 steps, alpha=eps/4, random start) ----
    def pgd(x, y, eps, steps):
        alpha = eps / 4.0
        adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = F.cross_entropy(model01(adv), y)
            g = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + alpha * g.sign()
            adv = torch.min(torch.max(adv, x - eps), x + eps).clamp(0, 1).detach()
        return adv

    # ---- choose a target image: clear, high-confidence, correctly classified ----
    eps_list = [float(t) / 255.0 for t in args.eps_list.split(",") if t.strip()]
    order = [args.index] if args.index is not None else list(range(len(df)))
    if args.index is None:
        rng = np.random.RandomState(args.seed); rng.shuffle(order)

    chosen = None
    for idx in order:
        x, gt = load(idx)
        c, conf = predict(x)
        if args.index is None and not (c == gt and conf > 0.90):
            continue
        y = torch.tensor([c], device=device)
        for eps in eps_list:                          # smallest eps that flips
            adv = pgd(x, y, eps, args.steps)
            ac, aconf = predict(adv)
            if ac != c:
                chosen = dict(idx=idx, x=x, adv=adv, eps=eps,
                              clean=(c, conf), adv_pred=(ac, aconf), gt=gt)
                break
        if chosen or args.index is not None:
            break
    if chosen is None:
        sys.exit("Could not find a clean, confidently-classified sign that flips within "
                 "the eps list; widen --eps-list or pass --index.")

    x, adv, eps = chosen["x"], chosen["adv"], chosen["eps"]
    (cc, cconf), (ac, aconf) = chosen["clean"], chosen["adv_pred"]
    xn = x[0].permute(1, 2, 0).cpu().numpy()
    an = adv[0].permute(1, 2, 0).cpu().numpy()
    delta = an - xn
    if args.amp > 0:
        dvis = np.clip(0.5 + args.amp * delta, 0, 1)          # fixed magnification
        ptitle = f"Perturbation\n(×{args.amp:g})"
    else:
        dvis = (delta - delta.min()) / (delta.max() - delta.min() + 1e-12)  # contrast-normalised
        ptitle = "Perturbation\n(amplified)"
    ssim = ssim_fn(xn, an, channel_axis=2, data_range=1.0) if ssim_fn else float("nan")
    print(f"row={chosen['idx']} gt={GTSRB_NAMES[chosen['gt']]!r} | "
          f"clean={GTSRB_NAMES[cc]!r} ({cconf:.2f}) -> adv={GTSRB_NAMES[ac]!r} ({aconf:.2f}) | "
          f"eps={eps*255:.0f}/255 | SSIM={ssim:.4f}")

    # ---- compose the triptych ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    ax[0].imshow(np.clip(xn, 0, 1)); ax[0].set_title(f"Clean\n{GTSRB_NAMES[cc]} ({cconf:.2f})", fontsize=10)
    ax[1].imshow(np.clip(an, 0, 1)); ax[1].set_title(f"Adversarial (ε={eps*255:.0f}/255)\n{GTSRB_NAMES[ac]} ({aconf:.2f})", fontsize=10)
    ax[2].imshow(dvis); ax[2].set_title(ptitle, fontsize=10)
    fig.suptitle(f"Imperceptible evasion  (SSIM {ssim:.3f})", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved -> {os.path.abspath(args.out)}")
    print("copy this into paper/IEEE_Conference_Template/figs/evasion_triptych.png")


if __name__ == "__main__":
    main()
