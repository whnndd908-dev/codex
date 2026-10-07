from pathlib import Path

import pytest

from pipeline.detect import Detection
from pipeline.schedule import Segment
from pipeline.landing import landing_point, landed_entry_tracks, landing_report


def det(order=2, kind="entry", t=0.0, tid=1, cx=800.0, by=600.0, h=300.0, w=200.0):
    """按中心点、底边、框高造一个框（同 test_pairing.py 的写法）。"""
    return Detection(
        seg_order=order, kind=kind, t=t,
        frame_path=Path(f"/tmp/{order}_{kind}_{t}_{tid}.jpg"), track_id=tid,
        x1=cx - w / 2, y1=by - h, x2=cx + w / 2, y2=by, conf=0.9,
    )


def make_track(order, kind, tid, n, cx0, cx1, by=600.0, h=300.0, t0=0.0, dt=0.5):
    """造一条 n 帧的轨迹，cx 从 cx0 线性变化到 cx1。n==1 时的写法同 test_pairing.py。"""
    if n == 1:
        return [det(order, kind, t0, tid, cx0, by, h)]
    out = []
    for i in range(n):
        frac = i / (n - 1)
        cx = cx0 + (cx1 - cx0) * frac
        out.append(det(order, kind, t0 + i * dt, tid, cx, by, h))
    return out


def seated_track(order, tid, cx, by, h, n=20):
    """一条恒定位置的 seated 轨迹，充当"已确认座位"的就座参照。"""
    return [det(order, "seated", float(i), tid, cx, by, h) for i in range(n)]


def seg(order=2, presenter_num=2):
    return Segment(order, "测试公司", 0, 100, presenter_num, ())


def entry_at_ratio(order, tid, seat_cx, seat_by, seat_h, ratio, n=8):
    """造一条落点恰好在「距座位 ratio 倍座位框高」处的 entry 轨迹（纯横向偏移）。"""
    dist = ratio * seat_h
    return make_track(order, "entry", tid, n=n, cx0=seat_cx - dist, cx1=seat_cx - dist,
                       by=seat_by, h=seat_h)


# ---------------------------------------------------------------------------
# 1. landing_point 取轨迹末尾而非全轨迹
# ---------------------------------------------------------------------------

def test_landing_point_uses_tail_not_whole_track():
    far = [det(tid=1, t=float(i), cx=100.0, by=600.0) for i in range(10)]
    near = [det(tid=1, t=float(10 + i), cx=800.0, by=600.0) for i in range(10)]
    track = far + near
    cx, by = landing_point(track, tail_ratio=0.2)
    # 全轨迹中位数会落在 100~800 中间；只看末尾应精确落在 800。
    assert cx == pytest.approx(800.0)
    assert by == pytest.approx(600.0)


# ---------------------------------------------------------------------------
# 2. 取比值最小的 presenter_num 条候选
# ---------------------------------------------------------------------------

def test_takes_presenter_num_nearest_candidates():
    order = 2
    seat_h = 300.0
    seat_cx, seat_by = 800.0, 600.0
    seats = {100: 1}
    dets = seated_track(order, 100, seat_cx, seat_by, seat_h)

    # 5 条候选，比值互相间隔 > LANDING_DEDUP_RATIO(0.16)，避免被误判为重复轨迹。
    ratios = [0.05, 0.25, 0.45, 0.65, 0.85]
    ids = {}
    for i, r in enumerate(ratios):
        tid = 10 + i
        dets += entry_at_ratio(order, tid, seat_cx, seat_by, seat_h, r)
        ids[r] = tid

    segment = seg(order, presenter_num=2)
    result = landed_entry_tracks(dets, segment, seats, presenter_num=2)
    got_ids = {row["track_id"] for row in result}
    assert got_ids == {ids[0.05], ids[0.25]}


# ---------------------------------------------------------------------------
# 3. 比值超出 LANDING_MAX_RATIO 上界即丢弃，哪怕人数不够也不硬凑
# ---------------------------------------------------------------------------

def test_candidate_beyond_max_ratio_is_dropped_even_if_short_of_count():
    order = 2
    seat_h = 300.0
    seat_cx, seat_by = 800.0, 600.0
    seats = {100: 1}
    dets = seated_track(order, 100, seat_cx, seat_by, seat_h)

    ratios = [0.05, 0.25, 0.95]   # 0.95 远超 LANDING_MAX_RATIO(0.60)
    ids = {}
    for i, r in enumerate(ratios):
        tid = 20 + i
        dets += entry_at_ratio(order, tid, seat_cx, seat_by, seat_h, r)
        ids[r] = tid

    segment = seg(order, presenter_num=3)   # 标注 3 人，但只有 2 条候选够格
    result = landed_entry_tracks(dets, segment, seats, presenter_num=3)
    got_ids = {row["track_id"] for row in result}
    assert got_ids == {ids[0.05], ids[0.25]}
    assert len(result) == 2   # 不硬凑到 3 条


# ---------------------------------------------------------------------------
# 4. 尺度不变：整组几何缩放后判定结果不变
# ---------------------------------------------------------------------------

def _build_scaled(scale: float):
    order = 2
    seat_h = 300.0 * scale
    seat_cx, seat_by = 800.0 * scale, 600.0 * scale
    seats = {100: 1}
    dets = seated_track(order, 100, seat_cx, seat_by, seat_h)
    ratios = [0.05, 0.30, 0.90]
    ids = {}
    for i, r in enumerate(ratios):
        tid = 30 + i
        dets += entry_at_ratio(order, tid, seat_cx, seat_by, seat_h, r)
        ids[r] = tid
    return dets, seg(order, presenter_num=2), seats, ids


def test_landing_threshold_is_scale_invariant():
    dets1, segment1, seats1, ids1 = _build_scaled(1.0)
    dets2, segment2, seats2, ids2 = _build_scaled(0.5)

    r1 = landed_entry_tracks(dets1, segment1, seats1, presenter_num=2)
    r2 = landed_entry_tracks(dets2, segment2, seats2, presenter_num=2)

    ratio_key1 = {v: k for k, v in ids1.items()}
    ratio_key2 = {v: k for k, v in ids2.items()}
    got1 = {ratio_key1[row["track_id"]] for row in r1}
    got2 = {ratio_key2[row["track_id"]] for row in r2}
    assert got1 == got2 == {0.05, 0.30}

    ratios1 = sorted(row["distance_ratio"] for row in r1)
    ratios2 = sorted(row["distance_ratio"] for row in r2)
    assert ratios1 == pytest.approx(ratios2, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. 只匹配已确认座位，不自造座位
# ---------------------------------------------------------------------------

def test_landing_only_matches_confirmed_seats():
    order = 2
    dets = entry_at_ratio(order, 1, 800.0, 600.0, 300.0, 0.0)   # 就落在原点，但没有已确认座位
    segment = seg(order, presenter_num=2)
    result = landed_entry_tracks(dets, segment, {}, presenter_num=2)
    assert result == []


# ---------------------------------------------------------------------------
# 6. 帧数不足的轨迹不参与
# ---------------------------------------------------------------------------

def test_short_track_is_ignored():
    order = 2
    seat_h = 300.0
    seat_cx, seat_by = 800.0, 600.0
    seats = {100: 1}
    dets = seated_track(order, 100, seat_cx, seat_by, seat_h)
    # 恰好落在座位上（比值 0），但只有 5 帧，低于 LANDING_MIN_FRAMES(8)。
    dets += entry_at_ratio(order, 1, seat_cx, seat_by, seat_h, 0.0, n=5)

    segment = seg(order, presenter_num=1)
    result = landed_entry_tracks(dets, segment, seats, presenter_num=1)
    assert result == []


# ---------------------------------------------------------------------------
# 7. 诊断报告标出段内最大跳变的位置
# ---------------------------------------------------------------------------

def test_largest_gap_position_is_reported():
    order = 2
    seat_h = 300.0
    seat_cx, seat_by = 800.0, 600.0
    seats = {100: 1}
    dets = seated_track(order, 100, seat_cx, seat_by, seat_h)

    # 升序比值：0.05 0.25 0.45 | 0.80 1.00 —— 最大跳变在 0.45→0.80（0.35）。
    ratios = [0.05, 0.25, 0.45, 0.80, 1.00]
    ids = {}
    for i, r in enumerate(ratios):
        tid = 40 + i
        dets += entry_at_ratio(order, tid, seat_cx, seat_by, seat_h, r)
        ids[r] = tid

    segment = seg(order, presenter_num=5)
    report = landing_report(dets, [segment], {order: seats})

    assert len(report) == 5
    assert all("distance_ratio" in row for row in report)

    by_tid = {row["track_id"]: row for row in report}
    assert by_tid[ids[0.45]]["is_largest_gap"] is True
    for r in (0.05, 0.25, 0.80, 1.00):
        assert by_tid[ids[r]]["is_largest_gap"] is False
