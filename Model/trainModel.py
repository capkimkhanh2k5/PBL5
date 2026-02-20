"""
=============================================================================
  Waste Detection & Classification - MobileNetV3 + PyTorch
  Resolution: 720x720 | Background Subtraction + Grad-CAM Localization
  Classes: battery, biological, cardboard, clothes, glass,
           metal, paper, plastic, shoes, trash

  TRAINING STRATEGY (5 phases):
    Phase 1  Head Warm-up      :  backbone frozen, 360px, no augmix
    Phase 2  Progressive Unfreeze:  gradual unfreeze, 540px, light augmix
    Phase 3  Full Fine-tune    :  all layers, 720px, full augmix + CutMix
    Phase 4  SWA Polish        :  stochastic weight averaging, 720px
    Phase 5  EMA Finalize      :  export best EMA weights

  HARDWARE: Auto-detect GPU (Multi-GPU DataParallel + AMP FP16)
  PLATFORM: Google Colab / Kaggle / Local
============================================================================="""

import os
import sys
import time
import copy
import json
import math
import warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import transforms, models
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (Colab safe)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# GPU AUTO-DETECTION
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("  WASTE DETECTION - MobileNetV3 + PyTorch")
print("  Auto-detecting hardware...")
print("="*70)

if torch.cuda.is_available():
    GPU_COUNT = torch.cuda.device_count()
    GPU_NAME  = torch.cuda.get_device_name(0)
    GPU_MEM   = torch.cuda.get_device_properties(0).total_memory / 1024**3
    DEVICE    = "cuda"
    torch.backends.cudnn.benchmark = True          # optimize conv algorithms
    print(f"  CUDA available!  GPUs: {GPU_COUNT}")
    for _i in range(GPU_COUNT):
        _n = torch.cuda.get_device_name(_i)
        _m = torch.cuda.get_device_properties(_i).total_memory / 1024**3
        print(f"    GPU {_i}: {_n} ({_m:.1f} GB)")
    if GPU_COUNT >= 2:
        print(f"  >> Multi-GPU mode: DataParallel on {GPU_COUNT} GPUs")
    else:
        print(f"  >> Single-GPU mode")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    GPU_COUNT = 1
    GPU_NAME  = "Apple MPS"
    GPU_MEM   = 0
    DEVICE    = "mps"
    print(f"  Apple MPS (Metal) available")
else:
    GPU_COUNT = 0
    GPU_NAME  = "CPU"
    GPU_MEM   = 0
    DEVICE    = "cpu"
    print("  WARNING: No GPU detected! Training will be VERY slow.")

print("="*70 + "\n")

# ---------------------------------------------------------------------------
# PLATFORM AUTO-DETECTION  (Colab / Kaggle / Local)
# ---------------------------------------------------------------------------

_DEFAULT_DATA = "/kaggle/input/datasets/khnhcpkim/10-class-dataset-for-grab"
_DEFAULT_OUT  = "/kaggle/working/outputs"
_NUM_WORKERS  = 2
print("  Platform: Kaggle")

# ---------------------------------------------------------------------------
# BATCH SIZE AUTO-SCALING  (per-GPU × GPU_COUNT)
# ---------------------------------------------------------------------------
if DEVICE == "cuda" and GPU_MEM >= 14:              # T4 (16GB), V100, A100
    _B = {"p1": 32, "p2": 20, "p3": 16, "p4": 16}
elif DEVICE == "cuda":                              # Smaller CUDA GPU
    _B = {"p1": 16, "p2": 12, "p3": 8,  "p4": 8}
elif DEVICE == "mps":                               # Apple Silicon
    _B = {"p1": 32, "p2": 20, "p3": 16, "p4": 16}
else:                                               # CPU
    _B = {"p1": 8,  "p2": 6,  "p3": 4,  "p4": 4}

_SCALE = max(GPU_COUNT, 1)
print(f"  Batch per GPU : P1={_B['p1']}  P2={_B['p2']}  "
      f"P3={_B['p3']}  P4={_B['p4']}")
print(f"  Effective     : P1={_B['p1']*_SCALE}  P2={_B['p2']*_SCALE}  "
      f"P3={_B['p3']*_SCALE}  P4={_B['p4']*_SCALE}")
print()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "data_dir"       : _DEFAULT_DATA,
    "output_dir"     : _DEFAULT_OUT,
    "img_size"       : 720,
    "batch_size"     : _B["p3"] * _SCALE,
    "num_workers"    : _NUM_WORKERS,
    "device"         : DEVICE,
    "gpu_count"      : max(GPU_COUNT, 1),
    "seed"           : 42,
    "val_split"      : 0.2,
    "weight_decay"   : 1e-4,
    "ema_decay"      : 0.999,

    # Phase 1: Head Warm-up  (backbone frozen, small resolution)
    "p1_epochs"      : 5,
    "p1_lr"          : 1e-3,
    "p1_img_size"    : 360,
    "p1_batch"       : _B["p1"] * _SCALE,
    "p1_label_smooth": 0.0,
    "p1_mixup"       : 0.0,
    "p1_cutmix"      : 0.0,
    "p1_aug"         : "light",

    # Phase 2: Progressive Unfreeze  (gradual backbone unfreeze)
    "p2_epochs"      : 10,
    "p2_lr_head"     : 5e-4,
    "p2_lr_backbone" : 5e-5,
    "p2_img_size"    : 540,
    "p2_batch"       : _B["p2"] * _SCALE,
    "p2_label_smooth": 0.05,
    "p2_mixup"       : 0.1,
    "p2_cutmix"      : 0.0,
    "p2_aug"         : "medium",

    # Phase 3: Full Fine-tune  (all layers, full resolution)
    "p3_epochs"      : 30,
    "p3_lr_head"     : 3e-4,
    "p3_lr_backbone" : 1e-5,
    "p3_img_size"    : 720,
    "p3_batch"       : _B["p3"] * _SCALE,
    "p3_label_smooth": 0.1,
    "p3_mixup"       : 0.2,
    "p3_cutmix"      : 0.3,
    "p3_aug"         : "heavy",
    "p3_patience"    : 12,

    # Phase 4: SWA (Stochastic Weight Averaging)
    "p4_epochs"      : 8,
    "p4_lr"          : 5e-5,
    "p4_img_size"    : 720,
    "p4_batch"       : _B["p4"] * _SCALE,
    "p4_label_smooth": 0.1,
    "p4_mixup"       : 0.0,
    "p4_cutmix"      : 0.0,
    "p4_aug"         : "medium",
}

CLASSES = [
    "battery", "biological", "cardboard", "clothes",
    "glass", "metal", "paper", "plastic", "shoes", "trash"
]
NUM_CLASSES = len(CLASSES)
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(CONFIG["seed"])
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
class WasteDataset(Dataset):
    """
    Loads images from  data/<class_name>/<image>.jpg
    Supports optional MixUp augmentation.
    """

    def __init__(self, samples: list, transform=None):
        self.samples   = samples   # list of (path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_samples(data_dir: str):
    """Scan folder structure and return list of (path, idx)."""
    samples = []
    data_path = Path(data_dir)
    for cls_name in CLASSES:
        cls_dir = data_path / cls_name
        if not cls_dir.exists():
            print(f"[WARN] Class folder not found: {cls_dir}")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            for img_path in cls_dir.glob(ext):
                samples.append((str(img_path), CLASS2IDX[cls_name]))
    return samples


def train_val_split(samples, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n_val   = int(len(samples) * val_ratio)
    val_idx = indices[:n_val]
    trn_idx = indices[n_val:]
    return [samples[i] for i in trn_idx], [samples[i] for i in val_idx]


def get_transforms(img_size: int, aug_level: str = "light"):
    """
    3 augmentation levels to match training phases:
      light  : Phase 1 - minimal augmentation for head warm-up
      medium : Phase 2 - moderate augmentation for progressive unfreeze
      heavy  : Phase 3 - aggressive augmentation for full fine-tune
      val    : validation / inference (no augmentation)
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if aug_level == "val":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if aug_level == "light":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if aug_level == "medium":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.03),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ])

    # heavy
    return transforms.Compose([
        transforms.Resize((img_size + 64, img_size + 64)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.35, contrast=0.35,
                               saturation=0.35, hue=0.05),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08),
                                scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def make_weighted_sampler(samples):
    """Balance class distribution during training."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for _, lbl in samples:
        counts[lbl] += 1
    weights_per_class = 1.0 / np.maximum(counts, 1)
    sample_weights = torch.tensor(
        [weights_per_class[lbl] for _, lbl in samples], dtype=torch.float32
    )
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ---------------------------------------------------------------------------
# MODEL  –  MobileNetV3-Large + Custom Head
# ---------------------------------------------------------------------------
class WasteDetector(nn.Module):
    """
    MobileNetV3-Large backbone with a two-branch head:
      1. Classification branch  -->  class logits
      2. Objectness branch      -->  single sigmoid score (foreground vs background)
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        base    = mobilenet_v3_large(weights=weights)

        # ---------- Backbone (feature extractor) ----------
        self.features = base.features          # conv layers
        self.avgpool  = base.avgpool           # AdaptiveAvgPool2d(1)

        in_features = base.classifier[0].in_features   # 960

        # ---------- Classification Head ----------
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

        # ---------- Objectness Head (foreground detector) ----------
        self.objectness = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feat   = self.features(x)
        feat   = self.avgpool(feat)
        feat   = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        obj    = self.objectness(feat)          # raw logit, apply sigmoid outside
        return logits, obj

    def get_feature_map(self, x):
        """Return last feature map (for Grad-CAM)."""
        return self.features(x)


# ---------------------------------------------------------------------------
# MIXUP  +  CUTMIX  UTILITIES
# ---------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.2):
    """MixUp: convex interpolation of image pairs."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam  = np.random.beta(alpha, alpha)
    lam  = max(lam, 1 - lam)                    # keep lam >= 0.5
    ridx = torch.randperm(x.size(0), device=x.device)
    mixed_x  = lam * x + (1 - lam) * x[ridx]
    return mixed_x, y, y[ridx], lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: cut-and-paste rectangular region from another sample."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    ridx = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape

    # Random bounding box
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[ridx, :, y1:y2, x1:x2]

    # Adjust lambda based on actual clipped area
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)
    return mixed_x, y, y[ridx], lam


def augmix_criterion(criterion, pred, y_a, y_b, lam):
    """Combined loss for MixUp / CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# EMA  (Exponential Moving Average)
# ---------------------------------------------------------------------------
class ModelEMA:
    """
    Maintains an exponential moving average of model parameters.
    The EMA model typically achieves 0.5-1.5% better accuracy.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema   = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1 - d)
        # Also update buffers (batch-norm running stats)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


# ---------------------------------------------------------------------------
# BACKBONE LAYER GROUPS  (for progressive unfreezing)
# ---------------------------------------------------------------------------
def get_backbone_groups(model: WasteDetector):
    """
    Split MobileNetV3 features into 4 groups (shallow -> deep):
      Group 0 : features[0:4]   - early/stem layers
      Group 1 : features[4:7]   - mid-level features
      Group 2 : features[7:13]  - high-level features
      Group 3 : features[13:]   - final conv blocks
    """
    f = model.features
    groups = [
        list(f[0:4].parameters()),
        list(f[4:7].parameters()),
        list(f[7:13].parameters()),
        list(f[13:].parameters()),
    ]
    return groups


def freeze_backbone(model: WasteDetector):
    """Freeze entire backbone."""
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_groups(model: WasteDetector, group_indices: list):
    """Unfreeze specific backbone groups by index (0=shallowest, 3=deepest)."""
    groups = get_backbone_groups(model)
    for gi in group_indices:
        for p in groups[gi]:
            p.requires_grad = True


def make_param_groups(model: WasteDetector, lr_head: float, lr_backbone: float):
    """
    Discriminative learning rates: backbone groups get progressively
    smaller LR (deeper block = higher LR, shallower block = smaller LR).
    """
    groups = get_backbone_groups(model)
    param_groups = []

    for gi, params in enumerate(groups):
        trainable = [p for p in params if p.requires_grad]
        if trainable:
            scale = 2 ** (gi - 3)     # 0->0.125, 1->0.25, 2->0.5, 3->1.0
            param_groups.append({
                "params": trainable,
                "lr"    : lr_backbone * max(scale, 0.125),
                "name"  : f"backbone_g{gi}",
            })

    head_params = (
        list(model.classifier.parameters()) +
        list(model.objectness.parameters()) +
        list(model.avgpool.parameters())
    )
    param_groups.append({
        "params": [p for p in head_params if p.requires_grad],
        "lr"    : lr_head,
        "name"  : "head",
    })

    return param_groups


# ---------------------------------------------------------------------------
# TRAINING  &  VALIDATION  (per-epoch functions)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device,
                    mixup_alpha=0.0, cutmix_alpha=0.0, scheduler=None,
                    scaler=None):
    """
    One epoch of training with optional MixUp / CutMix (chosen randomly
    per batch), optional per-step scheduler, and optional AMP (FP16).
    """
    model.train()
    running_loss = correct = total = 0
    use_amp = scaler is not None

    pbar = tqdm(loader, desc="  Train", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        # Decide augmentation: CutMix > MixUp > None (probabilistic)
        use_cutmix = cutmix_alpha > 0 and np.random.rand() < 0.5
        use_mixup  = mixup_alpha  > 0 and not use_cutmix

        if use_cutmix:
            imgs, la, lb, lam = cutmix_data(imgs, labels, cutmix_alpha)
        elif use_mixup:
            imgs, la, lb, lam = mixup_data(imgs, labels, mixup_alpha)
        else:
            la, lb, lam = labels, labels, 1.0

        # Forward pass (AMP autocast for FP16 on CUDA)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(imgs)
            if lam < 1.0:
                loss = augmix_criterion(criterion, logits, la, lb, lam)
            else:
                loss = criterion(logits, labels)

        # Backward + optimize
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()          # per-step scheduler (OneCycleLR etc.)

        running_loss += loss.item() * imgs.size(0)
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation with per-class accuracy tracking (AMP autocast on CUDA)."""
    model.eval()
    running_loss = correct = total = 0
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    use_amp = (device.type == "cuda")

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits, _    = model(imgs)
            loss         = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        for p, l in zip(preds, labels):
            class_total[l.item()]   += 1
            class_correct[l.item()] += (p == l).item()

    per_class_acc = {
        IDX2CLASS[k]: class_correct[k] / max(class_total[k], 1)
        for k in sorted(class_total.keys())
    }

    return running_loss / total, correct / total, per_class_acc


# ---------------------------------------------------------------------------
# DATALOADER FACTORY  (recreated per phase for different resolution/batch)
# ---------------------------------------------------------------------------
def make_dataloaders(trn_samples, val_samples, img_size, batch_size,
                     aug_level, num_workers, device_type):
    trn_tf = get_transforms(img_size, aug_level=aug_level)
    val_tf = get_transforms(img_size, aug_level="val")

    trn_ds = WasteDataset(trn_samples, trn_tf)
    val_ds = WasteDataset(val_samples, val_tf)

    sampler = make_weighted_sampler(trn_samples)
    pin = (device_type == "cuda")

    trn_dl = DataLoader(trn_ds, batch_size=batch_size,
                        sampler=sampler, num_workers=num_workers,
                        pin_memory=pin, drop_last=True,
                        persistent_workers=(num_workers > 0))
    val_dl = DataLoader(val_ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=pin,
                        persistent_workers=(num_workers > 0))
    return trn_dl, val_dl


# ---------------------------------------------------------------------------
# MULTI-PHASE TRAINING PIPELINE
# ---------------------------------------------------------------------------
def train(config=CONFIG):
    device = torch.device(config["device"])

    total_epochs = (config["p1_epochs"] + config["p2_epochs"] +
                    config["p3_epochs"] + config["p4_epochs"])
    print(f"\n{'='*70}")
    print(f"  OPTIMIZED MULTI-PHASE TRAINING PIPELINE")
    print(f"  Device        : {device}")
    print(f"  Classes       : {NUM_CLASSES}  {CLASSES}")
    print(f"  Total epochs  : {total_epochs}  "
          f"({config['p1_epochs']}+{config['p2_epochs']}+"
          f"{config['p3_epochs']}+{config['p4_epochs']})")
    print(f"  Resolution    : {config['p1_img_size']} -> "
          f"{config['p2_img_size']} -> {config['p3_img_size']}")
    print(f"{'='*70}\n")

    # ---------- Data ----------
    all_samples = load_samples(config["data_dir"])
    print(f"Total images: {len(all_samples)}")
    if len(all_samples) == 0:
        raise RuntimeError("No images found! Check data_dir path.")

    trn_samples, val_samples = train_val_split(
        all_samples, config["val_split"], config["seed"]
    )
    cls_counts = defaultdict(int)
    for _, lbl in all_samples:
        cls_counts[lbl] += 1
    print(f"Train: {len(trn_samples)}  |  Val: {len(val_samples)}")
    print("  Class distribution:")
    for i in range(NUM_CLASSES):
        print(f"    {IDX2CLASS[i]:<12}: {cls_counts.get(i, 0):>5}")
    print()

    # ---------- Model ----------
    model = WasteDetector(NUM_CLASSES, pretrained=True).to(device)

    # Multi-GPU: DataParallel (splits batch across GPUs)
    gpu_count = config.get("gpu_count", 1)
    if gpu_count >= 2 and device.type == "cuda":
        train_model = nn.DataParallel(model)
        print(f"  >> DataParallel: {gpu_count} GPUs")
    else:
        train_model = model

    # AMP (Automatic Mixed Precision) — ~2x speedup on T4 Tensor Cores
    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print(f"  >> AMP (FP16) enabled")

    ema = ModelEMA(model, decay=config["ema_decay"])

    # ---------- Tracking ----------
    best_acc         = 0.0
    best_weights     = copy.deepcopy(model.state_dict())
    best_ema_weights = copy.deepcopy(ema.ema.state_dict())
    ckpt_path        = os.path.join(config["output_dir"], "best_model.pth")
    history          = {"train_loss": [], "train_acc": [],
                        "val_loss": [], "val_acc": [],
                        "ema_val_acc": [], "phase": [], "lr": []}

    global_epoch = 0

    # ==================================================================
    #  PHASE 1 :  HEAD WARM-UP
    # ==================================================================
    print(f"{'='*70}")
    print(f"  PHASE 1: HEAD WARM-UP  ({config['p1_epochs']} epochs, "
          f"{config['p1_img_size']}px)")
    print(f"  Backbone: FROZEN | Aug: light | MixUp: OFF | CutMix: OFF")
    print(f"{'='*70}")

    freeze_backbone(model)

    criterion_p1 = nn.CrossEntropyLoss(
        label_smoothing=config["p1_label_smooth"]
    )
    optimizer_p1 = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["p1_lr"], weight_decay=config["weight_decay"]
    )
    trn_dl, val_dl = make_dataloaders(
        trn_samples, val_samples,
        config["p1_img_size"], config["p1_batch"],
        config["p1_aug"], config["num_workers"], device.type
    )
    scheduler_p1 = optim.lr_scheduler.OneCycleLR(
        optimizer_p1,
        max_lr=config["p1_lr"],
        steps_per_epoch=len(trn_dl),
        epochs=config["p1_epochs"],
        pct_start=0.3,
    )

    for ep in range(1, config["p1_epochs"] + 1):
        global_epoch += 1
        t0 = time.time()

        trn_loss, trn_acc = train_one_epoch(
            train_model, trn_dl, optimizer_p1, criterion_p1, device,
            scheduler=scheduler_p1, scaler=scaler
        )
        ema.update(model)
        val_loss, val_acc, per_cls = validate(
            model, val_dl, criterion_p1, device
        )
        _, ema_val_acc, _ = validate(ema.ema, val_dl, criterion_p1, device)

        lr_now = optimizer_p1.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(
            f"  P1 Ep [{ep}/{config['p1_epochs']}]  "
            f"TrnL:{trn_loss:.4f} TrnA:{trn_acc:.4f} | "
            f"ValL:{val_loss:.4f} ValA:{val_acc:.4f} EMA:{ema_val_acc:.4f} | "
            f"LR:{lr_now:.2e} ({elapsed:.1f}s)"
        )
        _track(history, trn_loss, trn_acc, val_loss, val_acc,
               ema_val_acc, 1, lr_now)
        best_acc, best_weights, best_ema_weights = _maybe_save(
            model, ema, val_acc, ema_val_acc, best_acc,
            best_weights, best_ema_weights, ckpt_path, config, global_epoch
        )

    # ==================================================================
    #  PHASE 2 :  PROGRESSIVE UNFREEZE
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: PROGRESSIVE UNFREEZE  ({config['p2_epochs']} epochs, "
          f"{config['p2_img_size']}px)")
    print(f"  Backbone: groups 3->2->1->0 | Aug: medium | MixUp: 0.1")
    print(f"{'='*70}")

    unfreeze_groups(model, [3])

    criterion_p2 = nn.CrossEntropyLoss(
        label_smoothing=config["p2_label_smooth"]
    )
    trn_dl, val_dl = make_dataloaders(
        trn_samples, val_samples,
        config["p2_img_size"], config["p2_batch"],
        config["p2_aug"], config["num_workers"], device.type
    )

    unfreeze_schedule = {1: [3], 3: [2], 5: [1], 8: [0]}
    optimizer_p2 = None

    for ep in range(1, config["p2_epochs"] + 1):
        global_epoch += 1
        t0 = time.time()

        if ep in unfreeze_schedule:
            unfreeze_groups(model, unfreeze_schedule[ep])
            unfrozen = [g for e, gs in unfreeze_schedule.items()
                        if e <= ep for g in gs]
            print(f"  >> Unfrozen backbone groups: {sorted(set(unfrozen))}")
            param_groups = make_param_groups(
                model, config["p2_lr_head"], config["p2_lr_backbone"]
            )
            optimizer_p2 = optim.AdamW(
                param_groups, weight_decay=config["weight_decay"]
            )

        if optimizer_p2 is None:
            param_groups = make_param_groups(
                model, config["p2_lr_head"], config["p2_lr_backbone"]
            )
            optimizer_p2 = optim.AdamW(
                param_groups, weight_decay=config["weight_decay"]
            )

        trn_loss, trn_acc = train_one_epoch(
            train_model, trn_dl, optimizer_p2, criterion_p2, device,
            mixup_alpha=config["p2_mixup"],
            cutmix_alpha=config["p2_cutmix"],
            scaler=scaler,
        )
        ema.update(model)
        val_loss, val_acc, per_cls = validate(
            model, val_dl, criterion_p2, device
        )
        _, ema_val_acc, _ = validate(ema.ema, val_dl, criterion_p2, device)

        lr_now = optimizer_p2.param_groups[-1]["lr"]
        elapsed = time.time() - t0
        print(
            f"  P2 Ep [{ep}/{config['p2_epochs']}]  "
            f"TrnL:{trn_loss:.4f} TrnA:{trn_acc:.4f} | "
            f"ValL:{val_loss:.4f} ValA:{val_acc:.4f} EMA:{ema_val_acc:.4f} | "
            f"LR:{lr_now:.2e} ({elapsed:.1f}s)"
        )
        _track(history, trn_loss, trn_acc, val_loss, val_acc,
               ema_val_acc, 2, lr_now)
        best_acc, best_weights, best_ema_weights = _maybe_save(
            model, ema, val_acc, ema_val_acc, best_acc,
            best_weights, best_ema_weights, ckpt_path, config, global_epoch
        )

    # ==================================================================
    #  PHASE 3 :  FULL FINE-TUNE
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 3: FULL FINE-TUNE  ({config['p3_epochs']} epochs, "
          f"{config['p3_img_size']}px)")
    print(f"  Backbone: ALL UNFROZEN | Aug: heavy | "
          f"MixUp: {config['p3_mixup']} | CutMix: {config['p3_cutmix']}")
    print(f"{'='*70}")

    for p in model.parameters():
        p.requires_grad = True

    criterion_p3 = nn.CrossEntropyLoss(
        label_smoothing=config["p3_label_smooth"]
    )
    param_groups_p3 = make_param_groups(
        model, config["p3_lr_head"], config["p3_lr_backbone"]
    )
    optimizer_p3 = optim.AdamW(
        param_groups_p3, weight_decay=config["weight_decay"]
    )
    trn_dl, val_dl = make_dataloaders(
        trn_samples, val_samples,
        config["p3_img_size"], config["p3_batch"],
        config["p3_aug"], config["num_workers"], device.type
    )
    scheduler_p3 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p3, T_0=10, T_mult=2, eta_min=1e-7
    )

    no_improve = 0
    for ep in range(1, config["p3_epochs"] + 1):
        global_epoch += 1
        t0 = time.time()

        trn_loss, trn_acc = train_one_epoch(
            train_model, trn_dl, optimizer_p3, criterion_p3, device,
            mixup_alpha=config["p3_mixup"],
            cutmix_alpha=config["p3_cutmix"],
            scaler=scaler,
        )
        ema.update(model)
        scheduler_p3.step()

        val_loss, val_acc, per_cls = validate(
            model, val_dl, criterion_p3, device
        )
        _, ema_val_acc, _ = validate(ema.ema, val_dl, criterion_p3, device)

        lr_now = optimizer_p3.param_groups[-1]["lr"]
        elapsed = time.time() - t0

        cls_str = ""
        if ep % 5 == 0:
            cls_str = "  Per-class: " + " | ".join(
                f"{k}:{v:.2f}" for k, v in per_cls.items()
            )

        print(
            f"  P3 Ep [{ep}/{config['p3_epochs']}]  "
            f"TrnL:{trn_loss:.4f} TrnA:{trn_acc:.4f} | "
            f"ValL:{val_loss:.4f} ValA:{val_acc:.4f} EMA:{ema_val_acc:.4f} | "
            f"LR:{lr_now:.2e} ({elapsed:.1f}s)"
        )
        if cls_str:
            print(cls_str)

        _track(history, trn_loss, trn_acc, val_loss, val_acc,
               ema_val_acc, 3, lr_now)

        effective_acc = max(val_acc, ema_val_acc)
        if effective_acc > best_acc:
            no_improve = 0
        else:
            no_improve += 1

        best_acc, best_weights, best_ema_weights = _maybe_save(
            model, ema, val_acc, ema_val_acc, best_acc,
            best_weights, best_ema_weights, ckpt_path, config, global_epoch
        )

        if no_improve >= config["p3_patience"]:
            print(f"  >> Early stopping at Phase 3, epoch {ep}")
            model.load_state_dict(best_weights)
            break

    # ==================================================================
    #  PHASE 4 :  SWA  (Stochastic Weight Averaging)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 4: SWA POLISH  ({config['p4_epochs']} epochs, "
          f"{config['p4_img_size']}px)")
    print(f"  Stochastic Weight Averaging for better generalization")
    print(f"{'='*70}")

    model.load_state_dict(best_weights)
    swa_model = AveragedModel(model).to(device)

    criterion_p4 = nn.CrossEntropyLoss(
        label_smoothing=config["p4_label_smooth"]
    )
    optimizer_p4 = optim.AdamW(
        model.parameters(), lr=config["p4_lr"],
        weight_decay=config["weight_decay"]
    )
    swa_scheduler = SWALR(
        optimizer_p4, swa_lr=config["p4_lr"],
        anneal_epochs=2, anneal_strategy="cos"
    )
    trn_dl, val_dl = make_dataloaders(
        trn_samples, val_samples,
        config["p4_img_size"], config["p4_batch"],
        config["p4_aug"], config["num_workers"], device.type
    )

    for ep in range(1, config["p4_epochs"] + 1):
        global_epoch += 1
        t0 = time.time()

        trn_loss, trn_acc = train_one_epoch(
            train_model, trn_dl, optimizer_p4, criterion_p4, device,
            mixup_alpha=config["p4_mixup"],
            cutmix_alpha=config["p4_cutmix"],
            scaler=scaler,
        )
        swa_model.update_parameters(model)
        swa_scheduler.step()

        val_loss, val_acc, _ = validate(model, val_dl, criterion_p4, device)
        lr_now = optimizer_p4.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"  P4 Ep [{ep}/{config['p4_epochs']}]  "
            f"TrnL:{trn_loss:.4f} TrnA:{trn_acc:.4f} | "
            f"ValL:{val_loss:.4f} ValA:{val_acc:.4f} | "
            f"LR:{lr_now:.2e} ({elapsed:.1f}s)"
        )
        _track(history, trn_loss, trn_acc, val_loss, val_acc, 0.0, 4, lr_now)

    print("  >> Updating SWA batch-norm statistics...")
    torch.optim.swa_utils.update_bn(trn_dl, swa_model, device=device)

    swa_loss, swa_acc, swa_cls = validate(
        swa_model, val_dl, criterion_p4, device
    )
    print(f"  >> SWA Val Acc: {swa_acc:.4f}")
    print("  SWA Per-class: " + " | ".join(
        f"{k}:{v:.2f}" for k, v in swa_cls.items()
    ))

    # ==================================================================
    #  PHASE 5 :  FINALIZE – pick best model
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 5: FINALIZE")
    print(f"{'='*70}")

    candidates = {
        "best_ckpt" : (best_acc,   best_weights),
        "ema"       : (0.0,        best_ema_weights),
        "swa"       : (swa_acc,    copy.deepcopy(swa_model.module.state_dict())),
    }

    ema_model_tmp = WasteDetector(NUM_CLASSES, pretrained=False).to(device)
    ema_model_tmp.load_state_dict(best_ema_weights)
    _, ema_final_acc, ema_cls = validate(
        ema_model_tmp, val_dl, criterion_p4, device
    )
    candidates["ema"] = (ema_final_acc, best_ema_weights)
    del ema_model_tmp

    print(f"  Best checkpoint acc : {candidates['best_ckpt'][0]:.4f}")
    print(f"  EMA acc             : {candidates['ema'][0]:.4f}")
    print(f"  SWA acc             : {candidates['swa'][0]:.4f}")

    winner_name = max(candidates, key=lambda k: candidates[k][0])
    winner_acc, winner_weights = candidates[winner_name]
    print(f"  >> Winner: {winner_name}  (val_acc={winner_acc:.4f})")

    model.load_state_dict(winner_weights)
    torch.save({
        "epoch"      : global_epoch,
        "model_state": winner_weights,
        "val_acc"    : winner_acc,
        "classes"    : CLASSES,
        "img_size"   : config["img_size"],
        "source"     : winner_name,
    }, ckpt_path)

    hist_path = os.path.join(config["output_dir"], "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_history(history, config["output_dir"])

    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best model : {winner_name}  val_acc={winner_acc:.4f}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Total epochs trained: {global_epoch}")
    print(f"{'='*70}\n")

    return model


# ---------------------------------------------------------------------------
# HELPERS  (tracking & saving)
# ---------------------------------------------------------------------------
def _track(history, trn_loss, trn_acc, val_loss, val_acc,
           ema_val_acc, phase, lr):
    history["train_loss"].append(trn_loss)
    history["train_acc"].append(trn_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["ema_val_acc"].append(ema_val_acc)
    history["phase"].append(phase)
    history["lr"].append(lr)


def _maybe_save(model, ema, val_acc, ema_val_acc, best_acc,
                best_weights, best_ema_weights, ckpt_path, config, epoch):
    effective = max(val_acc, ema_val_acc)
    if effective > best_acc:
        best_acc         = effective
        best_weights     = copy.deepcopy(model.state_dict())
        best_ema_weights = copy.deepcopy(ema.ema.state_dict())
        torch.save({
            "epoch"      : epoch,
            "model_state": best_weights,
            "ema_state"  : best_ema_weights,
            "val_acc"    : best_acc,
            "classes"    : CLASSES,
            "img_size"   : config["img_size"],
        }, ckpt_path)
        src = "EMA" if ema_val_acc >= val_acc else "Model"
        print(f"  >> Saved best ({src})  val_acc={best_acc:.4f}")
    return best_acc, best_weights, best_ema_weights


def plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss ---
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    # --- Accuracy ---
    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc")
    if "ema_val_acc" in history and any(v > 0 for v in history["ema_val_acc"]):
        axes[1].plot(epochs, history["ema_val_acc"], label="EMA Val Acc",
                     linestyle="--", alpha=0.7)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    # --- Learning Rate ---
    if "lr" in history:
        axes[2].plot(epochs, history["lr"], label="LR", color="tab:orange")
        axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log")
        axes[2].legend(); axes[2].grid(True)

    # Draw phase boundaries
    if "phase" in history:
        phase_changes = []
        for i in range(1, len(history["phase"])):
            if history["phase"][i] != history["phase"][i - 1]:
                phase_changes.append(i + 1)
        colors = ["#e6f3ff", "#fff3e6", "#e6ffe6", "#ffe6e6"]
        phase_labels = ["P1: Head", "P2: Unfreeze", "P3: Fine-tune", "P4: SWA"]
        for ax in axes:
            for pc in phase_changes:
                ax.axvline(x=pc, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  >> Saved training plot: {path}")


# ---------------------------------------------------------------------------
# GRAD-CAM  –  localization heatmap
# ---------------------------------------------------------------------------
class GradCAM:
    """
    Grad-CAM over the last MobileNetV3 conv block.
    Returns an upsampled heatmap (H x W float32, 0-1).
    """

    def __init__(self, model: WasteDetector):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.features[-1]   # last conv block

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, img_tensor: torch.Tensor, target_class: int = None):
        """
        img_tensor : (1, 3, H, W) on model device
        Returns    : heatmap np.ndarray (H, W) float32 0-1
        """
        self.model.eval()
        logits, _ = self.model(img_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Global average pooling of gradients  -->  channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)    # (1,C,1,1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam     = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize and resize to input resolution
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        H, W = img_tensor.shape[2], img_tensor.shape[3]
        cam  = cv2.resize(cam.astype(np.float32), (W, H))
        return cam, target_class


# ---------------------------------------------------------------------------
# BACKGROUND SUBTRACTION DETECTOR
# ---------------------------------------------------------------------------
class BackgroundSubtractorDetector:
    """
    Detects waste objects against a (learned or static) background using
    OpenCV's MOG2 + KNN subtractors, then classifies each detected region
    with the MobileNetV3 model.

    Typical usage
    -------------
    detector = BackgroundSubtractorDetector(model, device)
    # (optional) feed clean background frames first:
    detector.learn_background(bg_frames)
    # then process frames:
    annotated, detections = detector.detect_image(frame_bgr)
    """

    def __init__(self, model: WasteDetector, device: torch.device,
                 img_size: int = 720,
                 fg_threshold: float = 0.4,
                 min_contour_area: int = 2000):
        self.model          = model.eval()
        self.device         = device
        self.img_size       = img_size
        self.fg_threshold   = fg_threshold
        self.min_area       = min_contour_area

        # Background subtractors (combined for robustness)
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.knn  = cv2.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400, detectShadows=True
        )

        self.transform = get_transforms(img_size, aug_level="val")
        self.gradcam   = GradCAM(model)

    # ------------------------------------------------------------------ #
    def learn_background(self, bg_frames: list):
        """Feed a list of pure-background BGR frames to warm up subtractors."""
        for frame in bg_frames:
            self.mog2.apply(frame)
            self.knn.apply(frame)
        print(f"  Background model trained on {len(bg_frames)} frames.")

    # ------------------------------------------------------------------ #
    def _get_fg_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Combine MOG2 + KNN foreground masks with morphological cleanup."""
        m1 = self.mog2.apply(frame_bgr)
        m2 = self.knn.apply(frame_bgr)

        # Remove shadow pixels (127) – keep only foreground (255)
        m1 = (m1 == 255).astype(np.uint8) * 255
        m2 = (m2 == 255).astype(np.uint8) * 255
        mask = cv2.bitwise_or(m1, m2)

        # Morphological cleanup  (open=remove noise, close=fill holes)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        mask = cv2.dilate(mask, k, iterations=2)
        return mask

    # ------------------------------------------------------------------ #
    def _classify_region(self, crop_rgb: np.ndarray):
        pil_img = Image.fromarray(crop_rgb)
        tensor  = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, obj = self.model(tensor)
            probs       = torch.softmax(logits, dim=1)
            conf, idx   = probs.max(dim=1)
            obj_score   = torch.sigmoid(obj).item()
        return IDX2CLASS[idx.item()], conf.item(), obj_score

    # ------------------------------------------------------------------ #
    def detect_image(self, frame_bgr: np.ndarray,
                     use_bg_subtraction: bool = True):
        """
        Detect + classify waste in a BGR frame.

        Returns
        -------
        annotated  : BGR image with drawn bounding boxes
        detections : list of dicts {bbox, class, confidence, obj_score}
        """
        if use_bg_subtraction:
            mask = self._get_fg_mask(frame_bgr)
        else:
            h, w = frame_bgr.shape[:2]
            mask = np.ones((h, w), dtype=np.uint8) * 255

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated  = frame_bgr.copy()
        detections = []

        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            crop_rgb    = frame_rgb[y:y+h, x:x+w]
            cls_name, conf, obj_score = self._classify_region(crop_rgb)

            if conf < self.fg_threshold:
                continue

            detections.append({
                "bbox"      : (x, y, w, h),
                "class"     : cls_name,
                "confidence": conf,
                "obj_score" : obj_score,
            })

            color = (0, 255, 0) if obj_score > 0.5 else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            label = f"{cls_name}  {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(annotated, (x, y-lh-8), (x+lw+4, y), color, -1)
            cv2.putText(annotated, label, (x+2, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated, detections

    # ------------------------------------------------------------------ #
    def detect_with_gradcam(self, image_path: str, save_path: str = None):
        """
        Run Grad-CAM localization on a single image and draw bounding box
        from the highest-activation region.
        """
        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        frame_bgr = cv2.resize(frame_bgr, (self.img_size, self.img_size))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(frame_rgb)
        tensor  = self.transform(pil_img).unsqueeze(0).to(self.device)

        cam, cls_idx = self.gradcam(tensor)
        cls_name     = IDX2CLASS[cls_idx]

        # Threshold heatmap -> contour -> bounding box
        thresh = (cam > 0.4).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        # Heatmap overlay
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(frame_bgr, 0.5, heatmap, 0.5, 0)

        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(overlay, cls_name, (x, max(y-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(save_path, overlay)
            print(f"  >> Grad-CAM saved: {save_path}")
        else:
            cv2.imshow("Grad-CAM Detection", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return overlay, cls_name


# ---------------------------------------------------------------------------
# STATIC SINGLE-IMAGE INFERENCE
# ---------------------------------------------------------------------------
def infer_image(model: WasteDetector, image_path: str,
                device: torch.device, img_size: int = 720):
    """Top-5 classification result for a single image."""
    tf = get_transforms(img_size, aug_level="val")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_bgr = cv2.resize(img_bgr, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)
    tensor  = tf(pil_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, obj = model(tensor)
        probs       = torch.softmax(logits, dim=1).squeeze()
        top5_conf, top5_idx = probs.topk(5)

    print(f"\nInference: {Path(image_path).name}")
    print(f"{'Class':<15} {'Confidence':>12}")
    print("-" * 28)
    for c, i in zip(top5_conf, top5_idx):
        print(f"  {IDX2CLASS[i.item()]:<13} {c.item():>12.4f}")
    print(f"\nObjectness score: {torch.sigmoid(obj).item():.4f}")


# ---------------------------------------------------------------------------
# EXPORT TO ONNX
# ---------------------------------------------------------------------------
def export_onnx(model: WasteDetector, output_dir: str,
                img_size: int = 720,
                device: torch.device = torch.device("cpu")):
    model = model.to(device).eval()
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)

    onnx_path = os.path.join(output_dir, "waste_detector.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=17,
        input_names=["image"],
        output_names=["class_logits", "objectness"],
        dynamic_axes={
            "image"        : {0: "batch"},
            "class_logits" : {0: "batch"},
            "objectness"   : {0: "batch"},
        },
    )
    print(f"  >> Exported ONNX: {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Waste Detection & Classification – MobileNetV3 + PyTorch"
    )
    parser.add_argument(
        "--mode", default="train",
        choices=["train", "infer", "detect", "export"],
        help="train | infer (top-5) | detect (Grad-CAM bbox) | export (ONNX)",
    )
    parser.add_argument("--image", default=None,
                        help="Path to image file (required for infer / detect)")
    parser.add_argument("--ckpt",
                        default=os.path.join(CONFIG["output_dir"], "best_model.pth"),
                        help="Checkpoint path (for infer / detect / export)")
    args, _ = parser.parse_known_args()   # ignore Jupyter/Kaggle extra args

    device = torch.device(CONFIG["device"])

    if args.mode == "train":
        trained_model = train()
        # Auto-export ONNX after training
        export_onnx(trained_model, CONFIG["output_dir"],
                    CONFIG["img_size"], device=torch.device("cpu"))

    elif args.mode in ("infer", "detect"):
        if args.image is None:
            parser.error("--image is required for infer / detect mode")

        ckpt  = torch.load(args.ckpt, map_location=device)
        model = WasteDetector(NUM_CLASSES, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])

        if args.mode == "infer":
            infer_image(model, args.image, device, CONFIG["img_size"])

        else:
            detector = BackgroundSubtractorDetector(
                model, device, img_size=CONFIG["img_size"]
            )
            stem     = Path(args.image).stem
            out_path = os.path.join(CONFIG["output_dir"], f"{stem}_detected.jpg")
            detector.detect_with_gradcam(args.image, save_path=out_path)

    elif args.mode == "export":
        ckpt  = torch.load(args.ckpt, map_location="cpu")
        model = WasteDetector(NUM_CLASSES, pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        export_onnx(model, CONFIG["output_dir"], CONFIG["img_size"])