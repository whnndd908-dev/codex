"""逐帧白平衡校正。

参照物每帧重新定位而非使用固定框——摄像机会漂移，固定框会滑到绿桌布上，
这在探查阶段已实际发生过（t=3500、5400 两帧）。
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import config


def find_reference_patch(img_rgb: np.ndarray) -> tuple[float, float, float] | None:
    """在画面下半区寻找面积最大的近中性高亮连通域。

    限定下半区是为了避开投影幕布——幕布又亮又近中性，但它正是污染源本身。
    """
    h = img_rgb.shape[0]
    lower = img_rgb[h // 2:, :, :]

    hsv = cv2.cvtColor(lower, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0

    mask = ((sat < config.WB_REF_MAX_SAT) & (val > config.WB_REF_MIN_VAL)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if areas.max() < 2000:            # 太小的高光不可靠
        return None

    sel = labels == best
    px = lower[sel].astype(np.float32)
    return tuple(px.mean(axis=0).tolist())


def von_kries_gains(ref_rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """对角校正增益：使参照物三通道相等，同时保持其平均亮度不变。"""
    r, g, b = ref_rgb
    target = (r + g + b) / 3.0
    eps = 1e-6
    return (target / (r + eps), target / (g + eps), target / (b + eps))


def apply_gains(img_rgb: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    out = img_rgb.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= gains[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def correct_image(img_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """返回 (校正后图像, 是否找到参照物)。找不到时原样返回并报 False。"""
    ref = find_reference_patch(img_rgb)
    if ref is None:
        return img_rgb, False
    return apply_gains(img_rgb, von_kries_gains(ref)), True
