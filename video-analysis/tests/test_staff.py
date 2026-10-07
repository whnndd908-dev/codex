"""Task 17：工作人员/专家跨片段外观聚类安全网。

除标 `slow` 的一条真实图片测试外，全部用合成 embedding 向量，
不加载 CLIP 模型（否则要跑几分钟且依赖网络下载 1.7GB 模型）。
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline import config
from pipeline.staff import cluster_identities, flag_suspect_staff


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# 两个几乎正交（余弦距离≈1）的基向量，用来拼出"明显不同的人"
A = unit([1.0, 0.0, 0.0, 0.0])
B = unit([0.0, 1.0, 0.0, 0.0])
C = unit([0.0, 0.0, 1.0, 0.0])


def jitter(v, eps=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    return unit(np.asarray(v, dtype=float) + rng.normal(scale=eps, size=len(v)))


def test_identical_embeddings_cluster_together():
    embs = np.array([A, jitter(A, seed=1), jitter(A, seed=2)])
    labels = cluster_identities(embs)
    assert len(set(labels.tolist())) == 1


def test_distant_embeddings_stay_separate():
    embs = np.array([A, B, C])
    labels = cluster_identities(embs)
    assert len(set(labels.tolist())) == 3


def test_chain_of_gradually_shifting_embeddings_does_not_merge_into_one():
    """average linkage 必须能拆开一条"衣着渐变链"，single linkage 会把它焊成一簇。

    构造 8 个点，沿单位圆每隔 15° 取一个：相邻点余弦距离约 0.034（远小于
    STAFF_CLUSTER_DIST=0.15），但首尾点相距 105°，余弦距离约 0.74（远大于阈值）。
    single linkage 只看"最近邻"，会沿着链条一路merge成 1 簇——等价于把一整排
    深色西装的创业者串成"同一位工作人员"。average linkage 看簇间平均距离，
    应该在中途断开，产出 ≥2 个簇。
    """
    import math

    n, step_deg = 8, 15
    pts = [
        unit([math.cos(math.radians(i * step_deg)), math.sin(math.radians(i * step_deg))])
        for i in range(n)
    ]
    embs = np.array(pts)
    labels = cluster_identities(embs)
    assert len(set(labels.tolist())) >= 2


def test_cluster_spanning_three_segments_is_flagged():
    persons = [
        ("p_seg2_1", 2, jitter(A, seed=1)),
        ("p_seg7_1", 7, jitter(A, seed=2)),
        ("p_seg10_1", 10, jitter(A, seed=3)),
    ]
    flagged = flag_suspect_staff(persons)
    assert flagged == {"p_seg2_1", "p_seg7_1", "p_seg10_1"}


def test_cluster_spanning_two_segments_is_not_flagged():
    persons = [
        ("p_seg2_1", 2, jitter(A, seed=1)),
        ("p_seg7_1", 7, jitter(A, seed=2)),
    ]
    flagged = flag_suspect_staff(persons)
    assert flagged == set()


def test_repeated_appearances_within_one_segment_do_not_count():
    # 同一 order 里同一身份被切成 5 条轨迹（正常现象：跟踪器断裂重编号）。
    # distinct order 数仍是 1，不应标记——即便"簇内成员数"看起来有 5 个之多，
    # 会被误当成跨 3 段的假象（若代码按成员数而非去重 order 数计数）。
    persons = [(f"p_seg2_{i}", 2, jitter(A, seed=i)) for i in range(5)]
    flagged = flag_suspect_staff(persons)
    assert flagged == set()


def test_flag_returns_empty_when_fewer_than_min_segments_exist():
    # pilot 只有 order 2/7/10 三段；这里模拟只有 2 个 order 存在的场景
    # （全量才有 16 段），min_segments=3 时不可能达标，应返回空集合、不抛异常。
    persons = [
        ("p_a", 2, jitter(A, seed=1)),
        ("p_b", 2, jitter(B, seed=2)),
        ("p_c", 7, jitter(A, seed=3)),
    ]
    flagged = flag_suspect_staff(persons)
    assert flagged == set()


def test_single_person_input_does_not_crash():
    persons = [("solo", 2, A)]
    flagged = flag_suspect_staff(persons)
    assert flagged == set()

    # cluster_identities 本身也不能因单样本报错
    labels = cluster_identities(np.array([A]))
    assert labels.tolist() == [0]


def test_flag_is_deterministic():
    persons = [
        ("p_seg2_1", 2, jitter(A, seed=1)),
        ("p_seg7_1", 7, jitter(A, seed=2)),
        ("p_seg10_1", 10, jitter(A, seed=3)),
        ("q_seg2_1", 2, jitter(B, seed=4)),
        ("q_seg7_1", 7, jitter(B, seed=5)),
    ]
    r1 = flag_suspect_staff(persons)
    r2 = flag_suspect_staff(persons)
    assert r1 == r2


def test_embedding_none_persons_are_skipped_not_crashed():
    persons = [
        ("p_seg2_1", 2, jitter(A, seed=1)),
        ("p_seg7_1", 7, jitter(A, seed=2)),
        ("p_seg10_1", 10, jitter(A, seed=3)),
        ("broken_person", 5, None),
    ]
    flagged = flag_suspect_staff(persons)
    assert flagged == {"p_seg2_1", "p_seg7_1", "p_seg10_1"}
    assert "broken_person" not in flagged


def test_empty_persons_list_returns_empty_set():
    assert flag_suspect_staff([]) == set()


@pytest.mark.slow
def test_clip_embedding_on_real_crops_is_normalized():
    """加载真实 CLIP 模型，对两张真实帧裁剪出 embedding。

    只在手动跑 `-m slow` 时执行；全量跑用 `-m "not slow"` 会跳过。
    """
    from pathlib import Path

    from pipeline.detect import Detection
    from pipeline.staff import embed_crops, person_crops

    frames_dir = config.WORK / "frames"
    frame_files = sorted(frames_dir.rglob("*.jpg"))[:2]
    assert len(frame_files) >= 2, "需要至少两张真实帧才能跑这条 slow 测试"

    track = []
    for i, fp in enumerate(frame_files):
        from PIL import Image

        img = Image.open(fp)
        w, h = img.size
        track.append(
            Detection(
                seg_order=2,
                kind="seated",
                t=float(i),
                frame_path=fp,
                track_id=1,
                x1=w * 0.2,
                y1=h * 0.1,
                x2=w * 0.8,
                y2=h * 0.9,
                conf=0.9,
            )
        )

    crops = person_crops(track, max_frames=4)
    assert len(crops) >= 1
    emb = embed_crops(crops)
    assert emb is not None
    assert emb.shape == (768,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)
