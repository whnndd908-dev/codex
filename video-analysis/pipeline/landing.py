"""阶段 ④c：进场落座判据（Task 30）—— 已证伪，未接入流水线，保留供参考。

**本模块从未被 `pipeline/run_all.py` 导入或调用**。控制器实测：13 段里只
有 1 段的落座匹配结果与人工核对一致，判据本身没能稳定分辨「回到座位」与
「路过座位附近」，故 Task 40 确认停用。保留整个模块（含全部函数与
`tests/test_landing.py`）是因为它是一次有记录、可复现的失败实验，换一批
数据/换一种特征后可能还有价值；不是留着不用的死代码，而是明确标注「未接入」
的存档。**新代码不要从这里导入任何东西接进 `run_all.py`**，除非重新验证过
判据的有效性。

背景见下方摘要（原详细记录曾在 config.py 的 LANDING_* 常量上方，Task 40
起那三个常量已删除，改为本模块函数的默认参数，模块本身仍可独立调用/测试）：

用户逐企业核对第二组视频后指出「每个项目里一定有工作人员，两次尝试都没剔
除」，追查发现座位纯度上去后（Task 29）样本从 31 掉到 18——缺口集中在每段
第 2、3 人，他们坐在后排、与专家深度重叠，单靠就座窗口的几何分不开。本模
块尝试用 entry（进场）窗口的轨迹补回这些人：一条轨迹如果最终停在某个已确
认座位附近，很可能就是那个座位的主人走进来的过程。

**判据不是固定距离/比值阈值。** 控制器最初提出「停留点到座位距离 / 座位
框高 < 0.20」，但展开两组视频的全量分布后发现比值是连续谱，不存在能同时
切对不同段的全局阈值（order2 的跳变在 0.05→0.40，order6 的跳变在
0.07→0.12，0.20 两边都会错判）。改用「取比值最小的 presenter_num 条」——
presenter_num 来自人工标注（Book1/dta），本项目一路以它为可信先验；宽松
上界 `max_ratio`（默认 0.60）只挡最离谱的路过者，候选不足时不硬凑
（呼应 Ruling R47 里 seat_candidates 「候选不足也不凑够 presenter_num」的
同一原则）。

**只匹配已经由 `assign_segment_seats` 确认的座位**，不允许 entry 轨迹自己
造出新座位——这是防 order5/13 那类误报的关键：路过者恰好在座位附近停下，
如果允许自造座位，路过点本身会被当成新座位，路过者就会被判定为「回到了
自己的座位」。`seats` 参数必须来自已确认的座位表，`landed_entry_tracks`
拿不到匹配的已确认座位时（`seats` 为空）直接返回空列表。
"""
from __future__ import annotations

import numpy as np

from pipeline.detect import Detection
from pipeline.pairing import group_tracks
from pipeline.schedule import Segment


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def landing_point(track: list[Detection], tail_ratio: float = 0.2) -> tuple[float, float]:
    """轨迹末尾 `tail_ratio` 比例帧的 (cx, bottom_y) 中位数，即人最终停留处。

    只看末尾——落座判据要的是「人最后停在哪」，不是整条轨迹晃过的地方（走
    进来的路上会经过很多与座位无关的位置，被平均进去会把停留点拖偏）。
    末尾帧数不足 3 帧时取最后 3 帧（轨迹总长不足 3 帧则取全部）。
    """
    ordered = sorted(track, key=lambda d: d.t)
    n = len(ordered)
    tail_n = max(3, int(round(n * tail_ratio)))
    tail_n = min(tail_n, n)
    tail = ordered[-tail_n:]
    return (_median([d.cx for d in tail]), _median([d.bottom_y for d in tail]))


def _seat_stats(dets: list[Detection], seg_order: int,
                 seats: dict[int, int]) -> dict[int, dict]:
    """每个已确认座位的就座位置中位数 (cx, by) 与就座框高中位数 (h)。

    只读 `seats` 里列出的 track_id——这些是 `assign_segment_seats` 已经确认
    过的座位，不采纳任何 entry 轨迹自己推断出的新位置。
    """
    seated = group_tracks(dets, seg_order, "seated")
    out: dict[int, dict] = {}
    for tid, seat in seats.items():
        track = seated.get(tid)
        if not track:
            continue
        out[seat] = {
            "cx": _median([d.cx for d in track]),
            "by": _median([d.bottom_y for d in track]),
            "h": _median([d.height for d in track]),
        }
    return out


def seat_reference_points(dets: list[Detection], seg_order: int,
                           seats: dict[int, int]) -> dict[int, tuple[float, float]]:
    """每个已确认座位的就座位置中位数 (cx, bottom_y)。"""
    return {seat: (v["cx"], v["by"]) for seat, v in _seat_stats(dets, seg_order, seats).items()}


def _candidates(dets: list[Detection], segment: Segment,
                 seats: dict[int, int], min_frames: int = 8) -> list[dict]:
    """帧数达标的 entry 轨迹，各自匹配到最近的已确认座位。

    未排序、未去重——只做「这条轨迹停在哪、离最近的已确认座位多远」的原始
    测量。`seats` 为空（没有已确认座位可比对）时直接返回空列表，绝不用
    entry 轨迹自己的落点去凭空造一个座位出来。

    `min_frames`：轨迹帧数门槛，滤掉 ID 跳变产生的碎片。默认 8，沿用
    `config.PAIR_MIN_TRACK_FRAMES` 同一口径（本模块未接入流水线，不读
    config，调用方需要不同口径时自行传参）。
    """
    if not seats:
        return []
    stats = _seat_stats(dets, segment.order, seats)
    if not stats:
        return []

    out: list[dict] = []
    for tid, track in group_tracks(dets, segment.order, "entry").items():
        if len(track) < min_frames:
            continue
        cx, by = landing_point(track)
        best_seat = best_ratio = best_dist = None
        for seat, v in stats.items():
            h = v["h"]
            if not h:
                continue
            dist = float(np.hypot(cx - v["cx"], by - v["by"]))
            ratio = dist / h
            if best_ratio is None or ratio < best_ratio:
                best_ratio, best_seat, best_dist = ratio, seat, dist
        if best_seat is None:
            continue
        out.append({
            "track_id": tid,
            "landing_cx": cx,
            "landing_y2": by,
            "nearest_seat": best_seat,
            "distance": best_dist,
            "distance_ratio": best_ratio,
            "n_frames": len(track),
        })
    return out


def _dedup(cands: list[dict], seat_heights: dict[int, float],
           dedup_ratio: float = 0.16) -> list[dict]:
    """同一座位下，若两条候选的落点彼此非常接近，只保留一条。

    这不是「路过者恰好停在座位附近」的误报（那种情况由 presenter_num 排名
    + `max_ratio` 上界处理），而是跟踪器把同一次真实的走入过程拆成了多条
    entry track_id（遮挡/ID 跳变）——如果不去重，同一个人会占用
    presenter_num 排名里的两个名额，把真正的第二个人挤出去。
    保留帧数更多者：更长的轨迹信号更可靠。

    `dedup_ratio`：两条落座轨迹停留点距离 / 座位框高，小于此视为同一人被
    跟踪器拆成多条 entry 轨迹，只保留一条。默认 0.16（原 config 标定值）。
    """
    by_seat: dict[int, list[dict]] = {}
    for c in cands:
        by_seat.setdefault(c["nearest_seat"], []).append(c)

    out: list[dict] = []
    for seat, group in by_seat.items():
        h = seat_heights.get(seat)
        kept: list[dict] = []
        for c in sorted(group, key=lambda c: -c["n_frames"]):
            dup = False
            for k in kept:
                d = float(np.hypot(c["landing_cx"] - k["landing_cx"],
                                    c["landing_y2"] - k["landing_y2"]))
                if h and d / h < dedup_ratio:
                    dup = True
                    break
            if not dup:
                kept.append(c)
        out.extend(kept)
    return out


def _resolved_candidates(dets: list[Detection], segment: Segment,
                          seats: dict[int, int], min_frames: int = 8,
                          dedup_ratio: float = 0.16) -> list[dict]:
    cands = _candidates(dets, segment, seats, min_frames=min_frames)
    if not cands:
        return []
    heights = {seat: v["h"] for seat, v in _seat_stats(dets, segment.order, seats).items()}
    return _dedup(cands, heights, dedup_ratio=dedup_ratio)


def _rank_with_gaps(cands: list[dict]) -> list[dict]:
    """按 distance_ratio 升序排列，标注 rank、gap_after、is_largest_gap。

    `gap_after` 挂在排名靠前的那一侧——「跳变发生在第 i 名之后」比「跳变发
    生在第 i+1 名之前」更符合「presenter_num 名之后出现断层」这个判据的语义。
    """
    ordered = sorted(cands, key=lambda c: c["distance_ratio"])
    gaps = [ordered[i + 1]["distance_ratio"] - ordered[i]["distance_ratio"]
            for i in range(len(ordered) - 1)]
    max_gap = max(gaps) if gaps else None

    out = []
    for i, c in enumerate(ordered):
        row = dict(c)
        row["rank"] = i + 1
        if i < len(ordered) - 1:
            row["gap_after"] = gaps[i]
            row["is_largest_gap"] = max_gap is not None and gaps[i] == max_gap
        else:
            row["gap_after"] = None
            row["is_largest_gap"] = False
        out.append(row)
    return out


def landed_entry_tracks(dets: list[Detection], segment: Segment, seats: dict[int, int],
                         presenter_num: int, frame_w: float = 1440.0,
                         max_ratio: float = 0.60, min_frames: int = 8,
                         dedup_ratio: float = 0.16) -> list[dict]:
    """返回本段判定为「落座」的 entry 轨迹。

    判据：`_resolved_candidates` 去重后按 `distance_ratio` 升序取前
    `presenter_num` 条，再要求它们各自不超过 `max_ratio` 这道宽松上界——
    超过的直接丢弃，哪怕因此凑不够 `presenter_num` 条也不硬塞（呼应
    Ruling R47：候选不足时 `seat_candidates` 同样不凑数）。

    `frame_w` 保留以对齐 `pairing.pair_transitions` 一贯的调用签名，当前
    实现按座位框高（尺度不变量）归一化距离，不依赖画面宽度。

    `max_ratio`/`min_frames`/`dedup_ratio` 默认值即原 config.LANDING_*
    的标定值（本模块 Task 40 起不再接入 config，见模块顶部 docstring：
    整套判据已被证伪、未接入流水线，参数改为函数默认值，仅供独立调用/
    测试时使用或覆盖）。
    """
    cands = _resolved_candidates(dets, segment, seats, min_frames=min_frames,
                                  dedup_ratio=dedup_ratio)
    if not cands:
        return []
    ordered = sorted(cands, key=lambda c: c["distance_ratio"])
    top = ordered[:presenter_num]
    return [c for c in top if c["distance_ratio"] <= max_ratio]


def landing_report(dets: list[Detection], segments: list[Segment],
                    seats_by_order: dict[int, dict[int, int]],
                    max_ratio: float = 0.60, min_frames: int = 8,
                    dedup_ratio: float = 0.16) -> list[dict]:
    """诊断报告：每段每条候选一行，供控制器目视核对判据是否合适。

    列出去重后的候选全集（不只是最终判定为「落座」的那些）：每行含
    `distance_ratio`、`rank`、`gap_after`、`is_largest_gap`，以及综合
    presenter_num 排名与 `max_ratio` 上界得出的 `landed` 布尔值。
    `is_largest_gap` 让「段内最大跳变是否恰好落在人工标注人数处」一眼可
    见——这是判断该判据是否可信的核心问题，比任何单一阈值都重要。
    """
    report: list[dict] = []
    for segment in segments:
        seats = seats_by_order.get(segment.order, {})
        cands = _resolved_candidates(dets, segment, seats, min_frames=min_frames,
                                      dedup_ratio=dedup_ratio)
        for row in _rank_with_gaps(cands):
            row = dict(row)
            row["seg_order"] = segment.order
            row["presenter_num"] = segment.presenter_num
            row["landed"] = (row["rank"] <= segment.presenter_num
                              and row["distance_ratio"] <= max_ratio)
            report.append(row)
    return report
