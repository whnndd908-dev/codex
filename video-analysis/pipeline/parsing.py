"""服装分割与内外层分离。

主路径用 LIP 标签体系的人体解析模型，其 Coat(7) 与 Upper-clothes(5) 分离；
主路径失败时退回 k-means 兜底，按「面积最大 = 外层」「靠上的小簇 = 内层领口」指派。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pipeline import config

# SegFormer clothes 模型（ATR 标签），Upper-clothes = 4
_SEG_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
_UPPER_LABEL = 4

_processor = None
_seg_model = None


def _get_segmenter():
    global _processor, _seg_model
    if _seg_model is None:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        _processor = AutoImageProcessor.from_pretrained(_SEG_MODEL_ID)
        _seg_model = SegformerForSemanticSegmentation.from_pretrained(_SEG_MODEL_ID)
        _seg_model.eval()
        if torch.backends.mps.is_available():
            _seg_model.to("mps")
    return _processor, _seg_model


@dataclass
class GarmentMasks:
    outer: np.ndarray | None
    inner: np.ndarray | None
    outer_px: int
    inner_px: int
    source: str = "model"            # "model" | "kmeans_fallback"
    model_error: str | None = None


def inner_quality(px: int) -> str:
    if px >= config.INNER_QUALITY_GOOD:
        return "good"
    if px >= config.INNER_QUALITY_FAIR:
        return "fair"
    return "poor"


def _upper_mask(crop_rgb: np.ndarray) -> np.ndarray:
    import torch
    from PIL import Image

    processor, model = _get_segmenter()
    im = Image.fromarray(crop_rgb)
    inputs = processor(images=im, return_tensors="pt")
    if next(model.parameters()).is_mps:
        inputs = {k: v.to("mps") for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=crop_rgb.shape[:2], mode="bilinear", align_corners=False
    )
    pred = up.argmax(dim=1)[0].cpu().numpy()
    return pred == _UPPER_LABEL


def split_by_kmeans(mask: np.ndarray, crop_rgb: np.ndarray) -> GarmentMasks:
    """兜底分层：mask 内做 k-means，最大簇为外层，另一显著簇为内层。"""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return GarmentMasks(None, None, 0, 0)

    px = crop_rgb[mask].reshape(-1, 3).astype(np.float64)
    uniq = np.unique(px, axis=0)
    if len(uniq) <= 1:
        return GarmentMasks(mask, None, int(mask.sum()), 0)

    from sklearn.cluster import KMeans
    k = int(min(config.KMEANS_K, len(uniq)))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(px)
    counts = np.bincount(km.labels_, minlength=k)
    order = np.argsort(counts)[::-1]

    lab_img = np.full(mask.shape, -1, dtype=np.int32)
    lab_img[mask] = km.labels_

    outer = lab_img == order[0]
    outer_px = int(outer.sum())

    inner, inner_px = None, 0
    if k >= 2 and counts[order[1]] >= config.INNER_QUALITY_FAIR:
        cand = lab_img == order[1]
        # 内层多出现在颈部与袖口，取位置更靠上者作为内层
        if np.argwhere(cand)[:, 0].mean() < np.argwhere(outer)[:, 0].mean():
            inner, inner_px = cand, int(cand.sum())

    return GarmentMasks(outer, inner, outer_px, inner_px)


def parse_garments(crop_rgb: np.ndarray) -> GarmentMasks:
    """分割上衣区域；模型不可用时退回整图 k-means，但必须让下游知道。

    区分两种情况：
    - 模型抛异常 → 真正的错误，退回整图并标 source="kmeans_fallback"
      （整图聚类会把人脸、头发、椅子、背景一并算进"衣服颜色"，
        下游必须能识别出这一行不可信）
    - 模型成功但区域很小 → 真实的画面事实（本项目有创业者大半在画外，
      只剩一条手臂），保留那一小块 mask，质量由下游据 outer_px 判断。
      小而真实优于整图猜测。
    """
    try:
        mask = _upper_mask(crop_rgb)
        err = None
    except Exception as e:                      # noqa: BLE001 - 需捕获全部以便降级
        mask, err = None, f"{type(e).__name__}: {e}"

    if mask is None:
        logging.warning("服装分割模型不可用，退回整图 k-means：%s", err)
        g = split_by_kmeans(np.ones(crop_rgb.shape[:2], dtype=bool), crop_rgb)
        g.source = "kmeans_fallback"
        g.model_error = err
        return g

    if not mask.any():
        # 模型跑了但没找到上衣——这是事实，不是错误
        return GarmentMasks(None, None, 0, 0, source="model")

    g = split_by_kmeans(mask, crop_rgb)
    g.source = "model"
    return g
