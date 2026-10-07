"""阶段 ④c：工作人员/专家安全网——跨片段外观聚类。

存在的理由见 Ruling R19 与 spec 6.4。几何规则（评委在左、创业者在右、按底边
排序取前 N）挡不住坐到右侧上方的专家；而 16 个项目的 5 位专家全程不轮换，
"同一外观跨 ≥3 个 order 出现"对他们必然成立、对只上场一次的创业者必然不成立。

本模块只标记，永不删除——被误标的人要能在画廊里一键恢复。
"""
from __future__ import annotations

import numpy as np

from pipeline import config
from pipeline.detect import Detection

_clip = None


def _get_clip():
    """懒加载 CLIP 单例。模型 1.7 GB，加载一次约 10 s，绝不能每人加载一遍。"""
    global _clip
    if _clip is None:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(config.CLIP_MODEL)
        model.eval()
        _clip = (model, CLIPProcessor.from_pretrained(config.CLIP_MODEL))
    return _clip


def person_crops(track: list[Detection], max_frames: int = 4) -> list[np.ndarray]:
    """从一个人的轨迹里等间隔取几帧裁剪。

    等间隔而非取最清晰的几帧：清晰度最高的几帧往往时间上挨在一起，
    姿态雷同，对身份 embedding 没有增量信息。
    """
    from pathlib import Path
    from PIL import Image

    if not track:
        return []
    step = max(1, len(track) // max_frames)
    out = []
    for d in track[::step][:max_frames]:
        p = Path(d.frame_path)
        if not p.exists():
            continue
        img = np.asarray(Image.open(p).convert("RGB"))
        x1, y1 = int(max(0, d.x1)), int(max(0, d.y1))
        x2, y2 = int(min(img.shape[1], d.x2)), int(min(img.shape[0], d.y2))
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        out.append(img[y1:y2, x1:x2])
    return out


def embed_crops(crops) -> np.ndarray | None:
    """多张裁剪 → 一个人一条 embedding。

    先对每张单独 L2 归一化再求均值、再归一化：不先归一化的话，
    尺寸大/对比强的那张会因范数大而主导整条向量。
    """
    import torch

    if not crops:
        return None
    model, proc = _get_clip()
    from PIL import Image
    imgs = [Image.fromarray(c) for c in crops]
    with torch.no_grad():
        feats = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
    # transformers>=5 的 get_image_features 返回 BaseModelOutputWithPooling，
    # 投影后的 768 维 image embedding 在 .pooler_output 里；旧版本直接返回张量。
    # 见 CLIPModel.get_image_features 源码：vision_outputs.pooler_output 被就地
    # 替换成 self.visual_projection(pooled_output) 后原样返回整个 output 对象。
    tensor = feats.pooler_output if hasattr(feats, "pooler_output") else feats
    v = tensor.detach().cpu().numpy().astype(float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    m = v.mean(axis=0)
    n = float(np.linalg.norm(m))
    return None if n == 0.0 else m / n


def cluster_identities(embs: np.ndarray, dist: float | None = None) -> np.ndarray:
    """余弦距离层次聚类，返回每行的簇标签。

    linkage 用 average 而非 single：single linkage 会因一条"衣着相近"的链
    把整排深色西装串成一簇，把所有穿黑西装的创业者一起标成工作人员。
    """
    from sklearn.cluster import AgglomerativeClustering

    embs = np.asarray(embs, dtype=float)
    if len(embs) == 0:
        return np.array([], dtype=int)
    if len(embs) == 1:
        return np.array([0], dtype=int)      # 聚类库对单样本会报错
    d = config.STAFF_CLUSTER_DIST if dist is None else dist
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=d, metric="cosine", linkage="average",
    )
    return model.fit_predict(embs)


def flag_suspect_staff(persons, min_segments: int | None = None) -> set[str]:
    """跨 ≥ min_segments 个不同 order 出现的簇，其全部成员标为疑似工作人员。

    统计的是**去重后的 order 个数**，不是簇内成员个数：同一段里同一个人被切成
    三条轨迹是常态，按成员数算会把普通创业者误标。
    """
    m = config.STAFF_MIN_SEGMENTS if min_segments is None else min_segments
    usable = [(pid, order, e) for pid, order, e in persons if e is not None]
    if not usable:
        return set()

    labels = cluster_identities(np.asarray([e for _, _, e in usable], dtype=float))
    orders: dict[int, set[int]] = {}
    members: dict[int, list[str]] = {}
    for (pid, order, _), lab in zip(usable, labels):
        lab = int(lab)
        orders.setdefault(lab, set()).add(order)
        members.setdefault(lab, []).append(pid)

    return {pid for lab, o in orders.items() if len(o) >= m for pid in members[lab]}
