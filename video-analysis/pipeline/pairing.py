"""阶段 ④b：把进出场镜头里的轨迹配到就座座位上。

必须存在的理由见 Ruling R18：跟踪器在每个 (seg_order, kind) 窗口重新编号，
entry 的 track_id=5 和 seated 的 track_id=5 不是同一个人。不做配对而直接按
track_id 关联，会把甲的入场照和乙的就座照拼进同一张证据图，且不报错。

方向过滤的语义见 Ruling R38：进场窗口是 start-25s ~ start+15s，创业者常常在
窗口开始前就已落座，窗口拍到的是「已经坐着的人」而不是「走进来的人」。实测
order 10 的 21 条 entry 轨迹里 17 条静止，其中就包括该段唯一正确的那个人。
因此 entry_candidate **不要求入向**，只剔除帧数不达标的碎片与明显反向
（正在离场）的轨迹；真正的判别交给位置与颜色代价。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline import config
from pipeline.detect import Detection
from pipeline.schedule import Segment


def group_tracks(dets, seg_order: int, kind: str) -> dict[int, list[Detection]]:
    """按 track_id 分组，仅取指定 (seg_order, kind) 窗口内的检测。

    严格同时匹配 seg_order 与 kind 两个条件——track_id 的作用域是
    (seg_order, kind)，漏掉其中任一个条件都会把不同的人混进同一分组（R18）。
    """
    out: dict[int, list[Detection]] = {}
    for d in dets:
        if d.seg_order == seg_order and d.kind == kind:
            out.setdefault(d.track_id, []).append(d)
    for v in out.values():
        v.sort(key=lambda d: d.t)
    return out


def net_motion(track: list[Detection]) -> tuple[float, float]:
    """轨迹首帧到末帧的净位移（按时间排序后取首尾）。

    用净位移而非逐帧速度：人在门口停顿、被遮挡断续，逐帧速度噪声大，净位移
    方向稳定。内部按 t 排序，不依赖调用方预先排好序——上游 dets 的顺序不保证
    按时间。
    """
    if len(track) < 2:
        return (0.0, 0.0)
    ordered = sorted(track, key=lambda d: d.t)
    a, b = ordered[0], ordered[-1]
    return (b.cx - a.cx, b.bottom_y - a.bottom_y)


def is_inbound(track) -> bool:
    return net_motion(track)[0] > config.PAIR_MIN_MOVE_PX


def is_outbound(track) -> bool:
    return net_motion(track)[0] < -config.PAIR_MIN_MOVE_PX


def is_fragment(track) -> bool:
    """帧数太少的轨迹多半是 ID 跳变的碎片，净位移与外观签名都不可信。

    实测：真正拍到走入过程的轨迹有 31 帧、17 帧；而 order 7 的 tid=27 只有 4 帧
    却跨越 746 px、order 10 的 tid=7 只有 7 帧却跨越 973 px——那是跟踪器把两个
    不同的人接成了一条。见 brief 的「实测数据」一节。
    """
    return len(track) < config.PAIR_MIN_TRACK_FRAMES


def is_person_shaped(track, frame_h: float) -> bool:
    """轨迹是否为一个完整人形，而不是手、胳膊等局部检测框（Ruling R48）。

    `is_fragment` 只看帧数，挡不住手：手在笔记本上停留的时间足够长（27、28
    帧），帧数门槛形同虚设，而且手贴着座位，位置代价最低，反而赢下匈牙利
    匹配——把一张手的照片送进本该是「站立全身照」的证据格。

    用 bbox 高度**中位数**（不用均值）：走动中偶有几帧被遮挡成半个框，均值
    会被拉低，中位数不受这类噪声牵动。

    只用相对画面高度的**绝对**比值 `config.PAIR_MIN_H_RATIO`，不采用
    `seats.py` 的 `SEAT_MIN_H_VS_TALLEST`（相对本窗口最高候选）那条规则：
    进场/退场窗口里合法地同时存在站立者（~750 px）与已落座者（~400 px），
    "相对最高者" 会把真实的已落座候选误杀。座位分配窗口里全是坐姿、高度
    同量级，那条相对规则才成立，两者的场景不同，不能照搬。
    """
    if not track:
        return False
    h = float(np.median([d.height for d in track]))
    return h >= config.PAIR_MIN_H_RATIO * frame_h


def entry_candidate(track, frame_h: float = 1080.0) -> bool:
    """进场候选：帧数达标、不是明显出向、且是一个完整人形。

    **不要求入向**——进场窗口是 start-25s ~ start+15s，创业者常常在窗口开始前
    就已落座，窗口拍到的是「已经坐着的人」。实测 order 10 的 21 条 entry 轨迹里
    有 17 条是静止的，其中就包括该段唯一正确的那个人。要求入向会让整段拿不到
    任何进场帧（Ruling R38）。

    人形判据（Ruling R48）用 `frame_h` 默认参数保持向后兼容——旧调用点不传
    画面高度时，仍按 1080p 的假设工作。
    """
    return (not is_fragment(track) and not is_outbound(track)
            and is_person_shaped(track, frame_h))


def exit_candidate(track, frame_h: float = 1080.0) -> bool:
    """退场候选：帧数达标、不是明显入向、且是一个完整人形（对称于 entry_candidate）。"""
    return (not is_fragment(track) and not is_inbound(track)
            and is_person_shaped(track, frame_h))


def track_signature(track, max_frames: int = 6) -> np.ndarray | None:
    """轨迹的外观签名：上半身主色的 CIELAB 向量。

    只取 bbox 上 45% 高、中间 60% 宽——下半身是裤子/桌子，两侧是背景与邻座。
    取多帧的中位数而非均值：走动中会有几帧被遮挡成一片背景色，中位数不受其牵动。
    """
    from PIL import Image

    from pipeline.colormath import dominant_color, rgb_to_lab

    frames = track[:: max(1, len(track) // max_frames)][:max_frames]
    labs = []
    for d in frames:
        p = Path(d.frame_path)
        if not p.exists():
            continue
        img = np.asarray(Image.open(p).convert("RGB"))
        x1, y1 = int(max(0, d.x1)), int(max(0, d.y1))
        x2, y2 = int(min(img.shape[1], d.x2)), int(min(img.shape[0], d.y2))
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        h, w = y2 - y1, x2 - x1
        crop = img[y1:y1 + int(h * 0.45), x1 + int(w * 0.2):x2 - int(w * 0.2)]
        if crop.size == 0:
            continue
        labs.append(rgb_to_lab(dominant_color(crop)))
    if not labs:
        return None
    return np.median(np.asarray(labs, dtype=float), axis=0)


def _seat_reference(dets, seg_order: int, seats: dict[int, int]) -> dict[int, dict]:
    """每个座位的就座位置（中位数）、框高与外观签名。"""
    seated = group_tracks(dets, seg_order, "seated")
    ref: dict[int, dict] = {}
    for tid, seat in seats.items():
        track = seated.get(tid)
        if not track:
            continue
        ref[seat] = {
            "cx": float(np.median([d.cx for d in track])),
            "by": float(np.median([d.bottom_y for d in track])),
            "h": float(np.median([d.height for d in track])),   # 站立判据的基准
            "sig": track_signature(track),
        }
    return ref


def _reference_height(ref: dict[int, dict]) -> float:
    """本段就座框高的代表值，用作站立判据的分母。

    取各座位就座框高的中位数而非某一个座位的：座位 1 离镜头最近、框最大，
    拿它当基准会把远处座位的站立者误判成坐姿。
    """
    hs = [v["h"] for v in ref.values() if v.get("h")]
    return float(np.median(hs)) if hs else 0.0


def is_standing(track, ref_height: float) -> bool:
    """轨迹是否为站立姿态：框高中位数达到本段就座参照的 1.3 倍。

    用比值而非绝对像素——三段的就座框高中位数分别是 425/392/588 px，
    摄像机变焦不同，绝对阈值必然失效。
    """
    if ref_height <= 0:
        return False
    h = float(np.median([d.height for d in track]))
    return h >= config.PAIR_STANDING_H_RATIO * ref_height


def position_distance(pos, ref, frame_w: float, frame_h: float) -> float:
    """终点位置到座位参照的欧氏距离，横向按画宽归一、纵向按画高归一。

    从 `_cost` 里提出为独立纯函数（Ruling R49 / Task 20）：既要给 `_cost`
    的加权项复用，也要单独喂给硬位置闸门——闸门判的是这一项本身，不是
    掺了颜色、姿态之后的加权总和。用 (dx/W, dy/H) 而不是像素欧氏距离或
    画面对角线：否则 1440×1080 的画面里横向差异会被系统性高估，且不同
    分辨率的段之间不可比。
    """
    dx = (pos[0] - ref["cx"]) / frame_w
    dy = (pos[1] - ref["by"]) / frame_h
    return min(1.0, float(np.hypot(dx, dy)))


def _cost(sig_a, pos_a, standing: bool, ref, frame_w: float, frame_h: float) -> float:
    """外观距离 + 位置距离 + 坐姿惩罚的加权和，三项均归一化到 0–1。

    坐姿惩罚（Ruling R46）是**软偏好**：同一个人常被跟丢成「走进来的站立轨迹」
    与「已落座的坐姿轨迹」两条，后者终点恰在座位上、位置代价接近 0，会赢下
    匈牙利匹配——身份没错，但拿到的是张坐姿照，与 best_frame_seated 完全重复，
    进场帧该给出的站立全身信息（裤子、鞋、整体搭配）全丢了。
    权重刻意压到 0.2：某座位唯一的候选是坐姿时，加上它仍在 PAIR_MAX_COST=0.55
    之内、配得上；只有同座位同时有站立与坐姿候选时它才起选择作用。

    这是**加权**代价——位置项在这里仍可能被颜色项稀释掉（Ruling R49 的病灶
    正是如此）。硬位置闸门是另一道独立防线，见 `_match` 里对 `position_distance`
    的判断，不在这个函数里。
    """
    if sig_a is None or ref["sig"] is None:
        color = 1.0                      # 签名缺失记作最大外观距离，仍可靠位置配上
    else:
        de = float(np.linalg.norm(np.asarray(sig_a) - np.asarray(ref["sig"])))
        color = min(1.0, de / config.PAIR_DELTA_E_SCALE)
    pos = position_distance(pos_a, ref, frame_w, frame_h)
    posture = 0.0 if standing else 1.0
    return (config.PAIR_COLOR_WEIGHT * color
            + config.PAIR_POS_WEIGHT * pos
            + config.PAIR_POSTURE_WEIGHT * posture)


# 位置闸门外的格子在代价矩阵里填这个哨兵值，而不是 np.inf。
#
# 实测（Task 20）：`scipy.optimize.linear_sum_assignment` 对**整行或整列全
# 为 inf** 的代价矩阵会抛 `ValueError: cost matrix is infeasible`——某座位
# 所有候选都被闸门拒掉、或某条轨迹对所有座位都不可行时就会出现这种矩阵。
# 更麻烦的是：即便先把这类全 inf 的行/列剔除掉，方阵内部仍可能出现「结构性
# 不可行」——多个座位的可行候选完全重叠成同一条轨迹、Hall 定理条件不满足时，
# 同样会抛同一个 ValueError（用剔除全 inf 行列的写法实测复现过，见报告）。
# 改用一个远大于 PAIR_MAX_COST 的**有限**哨兵值：矩阵在数学上总能求出一个
# 完全匹配，`linear_sum_assignment` 不会再抛异常；哨兵值大到只要还有任何
# 可行选择，匈牙利解就绝不会去碰它，多个座位被迫共享同一条轨迹时也会先
# 用光所有可行格子、只让实在无路可走的座位落到哨兵值上。之后再用独立的
# `feasible` 布尔矩阵做事后过滤：即便某个座位被结构性逼上了哨兵值格子，
# 也强制不当作匹配上，报告为 None——数学上的"不得不选"不等于业务上的"可以配"。
_INFEASIBLE_COST = 1e6


def _match(cands: dict[int, tuple], ref: dict[int, dict],
           frame_w: float, frame_h: float) -> dict[int, dict]:
    """匈牙利算法求座位↔轨迹的全局最优一对一指派。

    不用贪心：每个座位各自取最近的轨迹会互相抢占，且不回头检查——
    seats.vote_seats 就栽在这上面（Ruling R20）。

    位置距离超过 `PAIR_MAX_POS_DIST` 的（座位, 轨迹）对是硬闸门拒绝的，
    不参与匹配（Ruling R49）；用 `_INFEASIBLE_COST` 哨兵值而非 `np.inf`
    规避 `linear_sum_assignment` 对不可行矩阵的 `ValueError`（见上方常量的
    注释）。

    每个座位额外算出 `best_alone`：只在**可行**候选（未被闸门拒绝）里代价
    最低的那个，与匈牙利全局解无关。全局最优有时会为了成全另一个没有备胎
    的座位，把某座位「本来最合适」的候选让给别人（Ruling R20 已接受的代价，
    不通过启发式修正）——`best_alone` 把这种分歧显式暴露出来，交人工复核。
    """
    if not cands or not ref:
        return {}
    from scipy.optimize import linear_sum_assignment

    tids = sorted(cands)
    seats = sorted(ref)
    cost = np.empty((len(seats), len(tids)), dtype=float)
    feasible = np.zeros((len(seats), len(tids)), dtype=bool)
    for i, s in enumerate(seats):
        for j, t in enumerate(tids):
            sig, pos, standing = cands[t]
            if position_distance(pos, ref[s], frame_w, frame_h) <= config.PAIR_MAX_POS_DIST:
                feasible[i, j] = True
                cost[i, j] = _cost(sig, pos, standing, ref[s], frame_w, frame_h)
            else:
                cost[i, j] = _INFEASIBLE_COST

    best_alone: dict[int, tuple[int, float] | None] = {}
    for i, s in enumerate(seats):
        js = np.where(feasible[i])[0]
        if js.size:
            j = js[int(np.argmin(cost[i, js]))]
            best_alone[s] = (tids[j], float(cost[i, j]))
        else:
            best_alone[s] = None          # 该座位无任何可行候选

    rows, cols = linear_sum_assignment(cost)
    out: dict[int, dict] = {}
    for i, j in zip(rows, cols):
        s = seats[i]
        ba = best_alone[s]
        entry = {"track": None, "cost": None, "best_alone": ba[0] if ba else None}
        if feasible[i, j]:                # 被闸门拒绝的格子，即便结构性被选中也不算数
            c = float(cost[i, j])
            if c <= config.PAIR_MAX_COST:  # 超阈值宁可不配
                entry["track"] = tids[j]
                entry["cost"] = c
        out[s] = entry
    for s in seats:                        # 匈牙利矩阵不覆盖到的座位（理论上不会发生，
        out.setdefault(s, {"track": None, "cost": None,  # 但保底不让 best_alone 丢失）
                            "best_alone": best_alone[s][0] if best_alone[s] else None})
    return out


def pair_transitions(dets, segment: Segment, seats: dict[int, int],
                      frame_w: float = 1440.0, frame_h: float = 1080.0) -> dict[int, dict]:
    """把 entry / exit 窗口的轨迹配到就座座位上。

    配不上就填 None，不许硬塞（Task 15 拿到 None 时 best_frame_entry 留空，
    证据图退化成只有就座图的 2×2，宁可少一张图，也不能拼错人）。
    """
    ref = _seat_reference(dets, segment.order, seats)
    ref_h = _reference_height(ref)

    ent_c = {}
    for tid, tr in group_tracks(dets, segment.order, "entry").items():
        if entry_candidate(tr, frame_h):   # 滤掉碎片、上一组离场者、手/胳膊等局部框
            ent_c[tid] = (track_signature(tr), (tr[-1].cx, tr[-1].bottom_y),
                          is_standing(tr, ref_h))

    ext_c = {}
    for tid, tr in group_tracks(dets, segment.order, "exit").items():
        if exit_candidate(tr, frame_h):
            ext_c[tid] = (track_signature(tr), (tr[0].cx, tr[0].bottom_y),
                          is_standing(tr, ref_h))
            # 退场用轨迹起点：人从座位起身往左走，起点才靠近座位

    ent = _match(ent_c, ref, frame_w, frame_h)
    ext = _match(ext_c, ref, frame_w, frame_h)

    out: dict[int, dict] = {}
    for seat in sorted(set(seats.values())):
        e = ent.get(seat, {"track": None, "cost": None, "best_alone": None})
        x = ext.get(seat, {"track": None, "cost": None, "best_alone": None})
        et, xt = e["track"], x["track"]
        out[seat] = {
            "entry_track": et,
            "entry_cost": round(e["cost"], 4) if et is not None else None,
            "entry_standing": bool(ent_c[et][2]) if et is not None else None,
            "entry_best_alone": e["best_alone"],
            "entry_assignment_diverged": (
                et is not None and e["best_alone"] is not None and et != e["best_alone"]
            ),
            "exit_track": xt,
            "exit_cost": round(x["cost"], 4) if xt is not None else None,
            "exit_best_alone": x["best_alone"],
            "exit_assignment_diverged": (
                xt is not None and x["best_alone"] is not None and xt != x["best_alone"]
            ),
            "method": "color+position+posture" if (et is not None or xt is not None) else "unmatched",
        }
    return out
