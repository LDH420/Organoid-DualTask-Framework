"""Analyze absolute object-size distributions from semantic segmentation masks.

Expected layout (configurable):
    DATA_ROOT/
      training/masks/*.png
      validation/masks/*.png
      testing/masks/*.png

Mask convention: a single-channel integer label map; 0 is background and each
non-zero value is a semantic class.  Each connected component of a class is
counted as one object.  For true instance masks, this is exact only when
different instances of the same class do not touch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_mask(path: Path) -> np.ndarray:
    """读取二值掩码，兼容 Windows 中文路径。"""
    try:
        file_bytes = np.fromfile(path, dtype=np.uint8)
        mask = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    except OSError as exc:
        raise ValueError(f"无法打开掩码文件：{path}") from exc

    if mask is None:
        raise ValueError(f"无法解码掩码图像：{path}")

    # 如果读取到三通道灰度图，转换为单通道
    if mask.ndim == 3:
        if (
            np.array_equal(mask[..., 0], mask[..., 1])
            and np.array_equal(mask[..., 1], mask[..., 2])
        ):
            mask = mask[..., 0]
        else:
            # 若三个通道因 JPG 压缩存在轻微差异，直接转灰度
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 放在这里：将 JPG 掩码二值化
    mask = (mask > 127).astype(np.uint8)

    return mask


def component_records(mask: np.ndarray, split: str, filename: str, min_area: int) -> list[dict]:
    records = []
    for class_id in np.unique(mask):
        if class_id == 0:
            continue
        binary = (mask == class_id).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component_id in range(1, n):
            x, y, width, height, area = stats[component_id]
            if area < min_area:
                continue
            records.append(
                {
                    "split": split,
                    "file": filename,
                    "class_id": int(class_id),
                    "component_id": component_id,
                    "area_px2": int(area),
                    "bbox_width_px": int(width),
                    "bbox_height_px": int(height),
                    "max_span_px": int(max(width, height)),
                    "equivalent_diameter_px": float(2 * np.sqrt(area / np.pi)),
                }
            )
    return records


def find_masks(data_root: Path, split: str, mask_subdir: str) -> list[Path]:
    folder = data_root / split / mask_subdir
    if not folder.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {folder}")
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def save_summary(df: pd.DataFrame, out_dir: Path, crop_size: int) -> None:
    grouped = df.groupby("split")["max_span_px"]
    summary = grouped.agg(
        objects="count", median_px="median", mean_px="mean",
        p90_px=lambda x: x.quantile(0.90), p95_px=lambda x: x.quantile(0.95), max_px="max",
    ).reset_index()
    summary["above_crop_percent"] = [100 * (x > crop_size).mean() for _, x in grouped]
    summary.to_csv(out_dir / "size_summary_by_split.csv", index=False, float_format="%.2f")

    by_class = df.groupby(["split", "class_id"])["max_span_px"].agg(
        objects="count", median_px="median", p90_px=lambda x: x.quantile(0.90), max_px="max"
    ).reset_index()
    by_class["above_crop_percent"] = by_class.apply(
        lambda row: 100 * (df.loc[(df.split == row.split) & (df.class_id == row.class_id), "max_span_px"] > crop_size).mean(), axis=1
    )
    by_class.to_csv(out_dir / "size_summary_by_split_and_class.csv", index=False, float_format="%.2f")


def plot_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.boxplot(data=df, x="split", y="equivalent_diameter_px", hue="split", legend=False,
                showfliers=False, palette="Set2", ax=ax)
    sns.stripplot(data=df, x="split", y="equivalent_diameter_px", color="black", alpha=0.18,
                  size=2, jitter=0.28, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Equivalent diameter (pixels)")
    ax.set_yscale("log")  # robust for the usual long-tailed organ-size distribution
    fig.tight_layout()
    fig.savefig(out_dir / "equivalent_diameter_boxplot.png", dpi=300)
    fig.savefig(out_dir / "equivalent_diameter_boxplot.pdf")
    plt.close(fig)


def plot_ecdf(df: pd.DataFrame, out_dir: Path, crop_size: int) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.ecdfplot(data=df, x="max_span_px", hue="split", linewidth=2, ax=ax)
    ax.axvline(crop_size, color="black", linestyle="--", linewidth=1.5,
               label=f"Training crop = {crop_size} px")
    ax.set_xlabel("Object maximum bounding-box span (pixels)")
    ax.set_ylabel("Cumulative proportion of objects")
    ax.set_ylim(0, 1.02)
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(out_dir / "max_span_ecdf.png", dpi=300)
    fig.savefig(out_dir / "max_span_ecdf.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure object sizes in semantic masks.")
    parser.add_argument("--data-root", type=Path, required=True, help="D:\seg_data\stomach_intance")
    parser.add_argument("--mask-subdir", default="masks", help="Mask folder inside every split (default: masks).")
    parser.add_argument("--output-dir", type=Path, default=Path("mask_size_analysis"))
    parser.add_argument("--crop-size", type=int, default=256, help="Training crop side length in pixels.")
    parser.add_argument("--min-area", type=int, default=1, help="Ignore components smaller than this many pixels.")
    parser.add_argument("--splits", nargs="+", default=["training", "validation", "testing"])
    args = parser.parse_args()

    all_records = []
    for split in args.splits:
        paths = find_masks(args.data_root, split, args.mask_subdir)
        print(f"{split}: {len(paths)} masks")
        for path in paths:
            all_records.extend(component_records(read_mask(path), split, path.name, args.min_area))

    if not all_records:
        raise RuntimeError("No foreground connected components found. Check mask labels and paths.")
    df = pd.DataFrame(all_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "object_size_records.csv", index=False, float_format="%.4f")
    save_summary(df, args.output_dir, args.crop_size)
    plot_boxplot(df, args.output_dir)
    plot_ecdf(df, args.output_dir, args.crop_size)
    print(f"Saved analysis to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
