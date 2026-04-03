"""
export.py  —  v2.0
==================
Export checkpoint (.pth) → ONNX + model_meta.json

Fixes so với export.py gốc:
  [FIX-1]  input_names=['image']  (khớp với notebook Cell 8 và testPC.py)
  [FIX-2]  ONNX filename = 'waste7_detector.onnx' (khớp với testPC.py)
  [FIX-3]  Xuất model_meta.json (thay classes.json) bao gồm đầy đủ:
             classes, img_size, agc_target, agc_gamma_min, agc_gamma_max
           testPC.py đọc file này để cấu hình AGC — nếu thiếu AGC sẽ
           dùng default và không khớp với lúc train → accuracy giảm.
  [FIX-4]  copy.deepcopy + .cpu().eval() trước khi export để không ảnh
           hưởng model đang dùng (quan trọng nếu gọi trong notebook).

Giữ nguyên ưu điểm của export.py gốc:
  [KEEP-1] Quick test sau export (in shape logits/objectness)
  [KEEP-2] do_constant_folding=True (explicit)
  [KEEP-3] Đọc checkpoint linh hoạt (model_state / model_state_dict / raw)

Cách dùng:
  1. Chỉnh PTH_PATH trỏ đến file .pth của bạn.
  2. Chỉnh AGC_TARGET / AGC_GAMMA_MIN / AGC_GAMMA_MAX khớp với CONFIG
     trong trainModel_v7_rrc.ipynb (mặc định đã đúng nếu bạn không đổi).
  3. python export.py
  4. Kiểm tra output:
       waste7_detector.onnx   — model ONNX
       model_meta.json        — config để testPC.py đọc
"""

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large


# ============================================================
# CẤU HÌNH
# ============================================================

PTH_PATH  = "final.pth"
ONNX_PATH = "waste7_detector.onnx"   # [FIX-2] khớp với testPC.py ONNX_PATH
META_PATH = "model_meta.json"         # [FIX-3] thay classes.json
IMG_SIZE  = 384                        # phải khớp với CONFIG['img_size'] trong notebook

# AGC config — phải khớp CHÍNH XÁC với CONFIG trong trainModel_v7_rrc.ipynb
# Nếu bạn đã đổi các giá trị này trong notebook thì đổi ở đây theo.
AGC_TARGET    = 128
AGC_GAMMA_MIN = 0.4
AGC_GAMMA_MAX = 3.0


# ============================================================
# MODEL DEFINITION  (phải khớp kiến trúc với lúc train)
# ============================================================
# Lưu ý về GeM:
#   Export gốc dùng .mean(dim=[2,3]) — trả về (B, C) trực tiếp.
#   Notebook dùng F.adaptive_avg_pool2d((1,1)) → (B,C,1,1) rồi .flatten(1).
#   Hai cách NUMERICALLY IDENTICAL (max diff = 0.0e+00 đã kiểm chứng).
#   Dùng cách notebook để 100% giống với WasteDetector khi train.
# ============================================================

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Giống notebook: adaptive_avg_pool2d → (B,C,1,1)
        # WasteDetector.forward sẽ .flatten(1) sau đó
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        ).pow(1.0 / self.p)


class WasteDetector(nn.Module):
    """Kiến trúc giống hệt trainModel_v7_rrc.ipynb Cell 4."""

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        base          = mobilenet_v3_large(weights=None)
        self.features = base.features
        self.gem_pool = GeM()
        in_f          = base.classifier[0].in_features  # 960

        self.classifier = nn.Sequential(
            nn.Linear(in_f, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )
        self.objectness = nn.Sequential(
            nn.Linear(in_f, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feat   = self.gem_pool(self.features(x)).flatten(1)
        logits = self.classifier(feat)
        obj    = self.objectness(feat)
        return logits, obj


# ============================================================
# MAIN
# ============================================================

def main():
    if not Path(PTH_PATH).exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {PTH_PATH}")

    print(f"[INFO] Loading: {PTH_PATH}")
    ckpt = torch.load(PTH_PATH, map_location="cpu", weights_only=False)

    # Đọc classes
    classes = ckpt.get(
        "classes",
        ["Battery", "Biological", "General_Waste",
         "Glass", "Metal", "Paper_Cardboard", "Plastic"],
    )
    num_classes = len(classes)
    print(f"[INFO] Classes ({num_classes}): {classes}")

    # Đọc img_size từ checkpoint nếu có
    ckpt_img_size = ckpt.get("img_size", IMG_SIZE)
    if ckpt_img_size != IMG_SIZE:
        print(f"[WARN] checkpoint img_size={ckpt_img_size} "
              f"!= IMG_SIZE={IMG_SIZE} — dùng {ckpt_img_size}")
    export_img_size = ckpt_img_size

    # Build model
    model = WasteDetector(num_classes)
    state_dict = ckpt.get("model_state",
                 ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    print("[INFO] State dict loaded.")

    # [FIX-4] deepcopy + cpu + eval để không ảnh hưởng model gốc
    export_model = copy.deepcopy(model).cpu().eval()

    # ── Export ONNX ──────────────────────────────────────────
    dummy = torch.randn(1, 3, export_img_size, export_img_size, dtype=torch.float32)

    torch.onnx.export(
        export_model,
        (dummy,),          # tuple, giống notebook
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],              # [FIX-1] khớp notebook + testPC.py
        output_names=["logits", "objectness"],
        dynamic_axes={
            "image":      {0: "batch"},
            "logits":     {0: "batch"},
            "objectness": {0: "batch"},
        },
    )
    print(f"[OK]  ONNX exported → {ONNX_PATH}")

    # ── Xuất model_meta.json ─────────────────────────────────
    # [FIX-3] Đầy đủ AGC config để testPC.py cấu hình FastAdaptiveGamma
    meta = {
        "classes":        classes,
        "img_size":       export_img_size,
        "agc_target":     AGC_TARGET,
        "agc_gamma_min":  AGC_GAMMA_MIN,
        "agc_gamma_max":  AGC_GAMMA_MAX,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK]  Meta saved → {META_PATH}")
    print(f"      AGC: target={AGC_TARGET}  "
          f"clip=[{AGC_GAMMA_MIN}, {AGC_GAMMA_MAX}]")

    # ── Quick test ───────────────────────────────────────────
    with torch.no_grad():
        logits, obj = export_model(dummy)

    print("\n[TEST] PyTorch forward pass:")
    print(f"       logits shape     : {tuple(logits.shape)}   "
          f"(expected: (1, {num_classes}))")
    print(f"       objectness shape : {tuple(obj.shape)}   "
          f"(expected: (1, 1))")
    assert logits.shape == (1, num_classes), "logits shape mismatch!"
    assert obj.shape    == (1, 1),           "objectness shape mismatch!"
    print("[OK]  Shape assertions passed.\n")

    # ── Summary ──────────────────────────────────────────────
    print("=" * 50)
    print("  Export hoàn tất. Files để deploy:")
    print(f"    {ONNX_PATH:<30}  ← model ONNX")
    print(f"    {META_PATH:<30}  ← config cho testPC.py")
    print("=" * 50)
    print("\n  Để chạy testPC.py, đảm bảo ONNX_PATH và META_PATH")
    print("  trong testPC.py trỏ đúng vào 2 file trên.")


if __name__ == "__main__":
    main()
