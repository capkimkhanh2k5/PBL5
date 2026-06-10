from __future__ import annotations

import json
import math
import time
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT_DIR / "Test" / "Test_V3"
MODEL_DIR = ROOT_DIR / "Train" / "outputs_v3_BU"
ONNX_PATH = MODEL_DIR / "waste7_detector_v3.onnx"
META_PATH = MODEL_DIR / "model_meta_v3.json"
REPORT_DIR = MODEL_DIR / "test_v3_reports"
DEFAULT_PREFIX = "test_v3"

BATCH_SIZE = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FastAdaptiveGamma:
    def __init__(self, target: int = 128, g_min: float = 0.4, g_max: float = 3.0):
        self.target = float(np.clip(target, 8, 247))
        self.g_min = g_min
        self.g_max = g_max
        self._idx = np.arange(256, dtype=np.float64) / 255.0
        self._last_gamma = -1.0
        self._lut = None

    def _compute_gamma(self, mean_v: float) -> float:
        mean_v = float(np.clip(mean_v, 8.0, 247.0))
        log_mean = math.log(mean_v / 255.0)
        log_target = math.log(self.target / 255.0)
        if abs(log_mean - log_target) < 0.03:
            return 1.0
        return float(np.clip(log_target / log_mean, self.g_min, self.g_max))

    def apply_numpy(self, img_rgb: np.ndarray) -> np.ndarray:
        mean_v = float(img_rgb.max(axis=2).mean())
        gamma = self._compute_gamma(mean_v)

        if abs(gamma - 1.0) < 0.02:
            self._last_gamma = gamma
            return img_rgb

        if self._lut is None or abs(gamma - self._last_gamma) > 0.005:
            lut = np.power(self._idx, gamma) * 255.0
            self._lut = lut.clip(0, 255).astype(np.uint8)

        self._last_gamma = gamma
        return cv2.LUT(img_rgb, self._lut)

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img, dtype=np.uint8)
        return Image.fromarray(self.apply_numpy(arr))

    @property
    def last_gamma(self) -> float:
        return self._last_gamma


def load_meta() -> dict:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay meta: {META_PATH}")
    with META_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V3_BU model on a folder-structured test dataset."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=TEST_DIR,
        help="Folder test co cau truc class/image, mac dinh: Test/Test_V3",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="Thu muc luu report, mac dinh: Train/outputs_v3_BU/test_v3_reports",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Tien to ten file output, vi du: real_world",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size inference",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def collect_images(test_dir: Path, classes: list[str]) -> list[tuple[Path, str]]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc test: {test_dir}")

    samples = []
    for class_name in classes:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Thieu thu muc class trong test set: {class_dir}")
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, class_name))

    if not samples:
        raise RuntimeError(f"Khong tim thay anh test trong: {test_dir}")
    return samples


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def save_confusion_matrix(
    matrix: np.ndarray,
    classes: list[str],
    output_path: Path,
    title: str,
    fmt: str = "d",
) -> None:
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_precision_recall_f1(report_df: pd.DataFrame, output_path: Path) -> None:
    metric_df = report_df.loc[
        [idx for idx in report_df.index if idx not in {"accuracy", "macro avg", "weighted avg"}],
        ["precision", "recall", "f1-score"],
    ]

    ax = metric_df.plot(kind="bar", figsize=(11, 6), width=0.78)
    ax.set_title("Precision - Recall - F1 theo tung lop", fontsize=14, fontweight="bold")
    ax.set_xlabel("Lop du lieu")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric")
    ax.tick_params(axis="x", rotation=35)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_confidence_threshold_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = [round(value, 2) for value in np.arange(0.50, 0.96, 0.05)]

    for threshold in thresholds:
        accepted = predictions_df["confidence"] >= threshold
        accepted_count = int(accepted.sum())
        correct_accepted = int(((predictions_df["correct"] == 1) & accepted).sum())
        coverage = accepted_count / len(predictions_df)
        accuracy_on_accepted = (
            correct_accepted / accepted_count if accepted_count else np.nan
        )
        strict_auto_success = correct_accepted / len(predictions_df)

        rows.append(
            {
                "confidence_threshold": threshold,
                "accepted_count": accepted_count,
                "rejected_count": int((~accepted).sum()),
                "coverage": coverage,
                "accuracy_on_accepted": accuracy_on_accepted,
                "strict_auto_success": strict_auto_success,
            }
        )

    return pd.DataFrame(rows)


def save_confidence_threshold_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        metrics_df["confidence_threshold"],
        metrics_df["coverage"],
        marker="o",
        label="Coverage",
    )
    ax.plot(
        metrics_df["confidence_threshold"],
        metrics_df["accuracy_on_accepted"],
        marker="o",
        label="Accuracy on accepted",
    )
    ax.plot(
        metrics_df["confidence_threshold"],
        metrics_df["strict_auto_success"],
        marker="o",
        label="Strict auto success",
    )
    ax.set_title("Confidence threshold operating points", fontsize=14, fontweight="bold")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    test_dir = resolve_path(args.test_dir)
    report_dir = resolve_path(args.report_dir)
    prefix = args.prefix
    batch_size = args.batch_size

    report_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta()
    classes = meta["classes"]
    img_size = int(meta.get("img_size", 384))
    agc = FastAdaptiveGamma(
        target=int(meta.get("agc_target", 128)),
        g_min=float(meta.get("agc_gamma_min", 0.4)),
        g_max=float(meta.get("agc_gamma_max", 3.0)),
    )

    transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            agc,
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(ONNX_PATH), providers=providers)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    samples = collect_images(test_dir, classes)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    rows = []
    y_true = []
    y_pred = []
    started_at = time.time()

    print(f"[INFO] Test dir : {test_dir}")
    print(f"[INFO] Model    : {ONNX_PATH}")
    print(f"[INFO] Provider : {session.get_providers()}")
    print(f"[INFO] Images   : {len(samples)}")
    print(f"[INFO] Classes  : {classes}")

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        tensors = []
        gamma_values = []

        for image_path, _ in batch_samples:
            with Image.open(image_path) as image:
                tensor = transform(image.convert("RGB"))
            tensors.append(tensor)
            gamma_values.append(float(agc.last_gamma))

        batch = torch.stack(tensors).numpy().astype(np.float32)
        logits, objectness = session.run(output_names, {input_name: batch})
        probabilities = softmax(logits)
        object_scores = sigmoid(objectness.reshape(-1))
        pred_indices = probabilities.argmax(axis=1)
        confidence_values = probabilities[np.arange(len(pred_indices)), pred_indices]

        for idx, (image_path, label) in enumerate(batch_samples):
            pred_idx = int(pred_indices[idx])
            pred_label = classes[pred_idx]
            true_idx = class_to_idx[label]
            y_true.append(true_idx)
            y_pred.append(pred_idx)

            row = {
                "path": str(image_path),
                "label": label,
                "prediction": pred_label,
                "confidence": float(confidence_values[idx]),
                "objectness": float(object_scores[idx]),
                "gamma": gamma_values[idx],
                "correct": int(pred_idx == true_idx),
            }
            for class_idx, class_name in enumerate(classes):
                row[f"prob_{class_name}"] = float(probabilities[idx, class_idx])
            rows.append(row)

        done = min(start + BATCH_SIZE, len(samples))
        print(f"[RUN] {done}/{len(samples)} images")

    elapsed_seconds = time.time() - started_at

    predictions_df = pd.DataFrame(rows)
    predictions_df.to_csv(report_dir / f"{prefix}_predictions.csv", index=False)
    threshold_metrics_df = build_confidence_threshold_metrics(predictions_df)
    threshold_metrics_df.to_csv(
        report_dir / f"{prefix}_confidence_threshold_metrics.csv",
        index=False,
    )
    save_confidence_threshold_plot(
        threshold_metrics_df,
        report_dir / f"{prefix}_confidence_threshold_metrics.png",
    )

    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        report_dir / f"{prefix}_confusion_matrix.csv"
    )
    pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv(
        report_dir / f"{prefix}_confusion_matrix_normalized.csv"
    )

    save_confusion_matrix(
        cm,
        classes,
        report_dir / f"{prefix}_confusion_matrix.png",
        f"Confusion Matrix - {prefix}",
        fmt="d",
    )
    save_confusion_matrix(
        cm_norm,
        classes,
        report_dir / f"{prefix}_confusion_matrix_normalized.png",
        f"Normalized Confusion Matrix - {prefix}",
        fmt=".2f",
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(report_dir / f"{prefix}_classification_report.csv")
    with (report_dir / f"{prefix}_classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report_dict, file, ensure_ascii=False, indent=2)
    (report_dir / f"{prefix}_classification_report.txt").write_text(report_text, encoding="utf-8")
    save_precision_recall_f1(report_df, report_dir / f"{prefix}_precision_recall_f1.png")

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )
    class_accuracy = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    class_metrics_df = pd.DataFrame(
        {
            "class_name": classes,
            "support": per_class_support,
            "correct": cm.diagonal(),
            "class_accuracy": class_accuracy,
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1_score": per_class_f1,
        }
    )
    class_metrics_df.to_csv(report_dir / f"{prefix}_class_metrics.csv", index=False)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    summary = {
        "test_dir": str(test_dir),
        "model": str(ONNX_PATH),
        "meta": str(META_PATH),
        "report_dir": str(report_dir),
        "num_images": len(samples),
        "num_correct": int(sum(predictions_df["correct"])),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "elapsed_seconds": elapsed_seconds,
        "avg_latency_ms_per_image": elapsed_seconds / len(samples) * 1000,
        "classes": classes,
    }
    with (report_dir / f"{prefix}_summary_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    summary_lines = [
        f"{prefix.upper()} EVALUATION SUMMARY",
        "=" * 32,
        f"Images              : {summary['num_images']}",
        f"Correct             : {summary['num_correct']}",
        f"Accuracy            : {summary['accuracy']:.4f}",
        f"Balanced accuracy   : {summary['balanced_accuracy']:.4f}",
        f"Macro precision     : {summary['macro_precision']:.4f}",
        f"Macro recall        : {summary['macro_recall']:.4f}",
        f"Macro F1            : {summary['macro_f1']:.4f}",
        f"Weighted precision  : {summary['weighted_precision']:.4f}",
        f"Weighted recall     : {summary['weighted_recall']:.4f}",
        f"Weighted F1         : {summary['weighted_f1']:.4f}",
        f"Avg latency/image   : {summary['avg_latency_ms_per_image']:.2f} ms",
        "",
        "Per-class metrics:",
        class_metrics_df.to_string(index=False),
        "",
        "Confidence threshold operating points:",
        threshold_metrics_df.to_string(index=False),
        "",
        "Classification report:",
        report_text,
    ]
    (report_dir / f"{prefix}_summary_metrics.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n".join(summary_lines[:14]))
    print(f"\n[OK] Reports saved to: {report_dir}")


if __name__ == "__main__":
    main()
