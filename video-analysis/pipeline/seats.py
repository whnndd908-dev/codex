"""阶段 ④：座位相对排序。

摄像机在四小时内有平移与变焦漂移，故不使用固定 ROI。
「谁在谁下方」由房间布局锁定：越靠画面下方 = 越靠近镜头 = 座位编号越小。

Ruling R47：单纯按「谁在谁下方」逐帧排序 + 投票会被伪影（手、胳膊等局部检测框）
抢位——YOLO 把桌沿露出的一双手检成了一个完整的 person 框，框的底边比真人的身体
框更靠下，抢走了座位 1。修法分两步：
  1. `seat_candidates` 先按框高把非完整人形的候选剔除（伪影框高 104–186px，
     真人框高 363–588px，中间有 2 倍干净间隔）。
  2. 但只加过滤还不够：剔除伪影后，真正在场的几条真人轨迹未必帧帧同时出现
     （遮挡、跟踪器断裂），逐帧排序+投票在这种「部分共现」结构下会让帧数少的
     真人轨迹只拿到少量票，被 `vote_seats` 的匈牙利总票数最优判定为零票丢弃。
     `rank_tracks_pairwise` 改用成对比较（只要求两条轨迹同帧共现过，不要求
     全员同时出现）+ Copeland 计分，修复这个问题。

Task 31：Task 29 的三条过滤规则顺序有误——覆盖率过滤跑在「同一个人被跟踪器
拆成多条轨迹」的合并之前，于是把同一个人的碎片一条条当噪声砍掉（实测样本
27→18/31）。`merge_tracks_by_position` 补上这一步，且**必须在覆盖率过滤之前
跑**，用并集帧数而非单条轨迹的帧数算覆盖率。同时 `SEAT_DEPTH_GAP_RATIO`
（段内相邻候选比值）改用跨段位置簇 `presenter_depth_floor` 取代——后者在两
组视频上都更稳定，前者默认停用（仍保留，供换组回退）。详见
`merge_tracks_by_position`、`presenter_depth_floor` 的 docstring 与
task-31-report.md。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations

from pipeline import config
from pipeline.detect import Detection
from pipeline.schedule import Segment


def exclude_judges(dets: list[Detection], frame_w: int) -> list[Detection]:
    """剔除评委席与计时员：二者恒在画面左侧，创业者席恒在右侧。"""
    cut = config.RIGHT_REGION_X * frame_w
    return [d for d in dets if d.cx > cut]


def rank_seats_in_frame(
    dets: list[Detection], frame_w: int, presenter_num: int
) -> dict[int, int]:
    """单帧内按 bbox 底边纵坐标降序分配座位号。

    排序是相对的，故对摄像机漂移免疫。

    **不在 `assign_segment_seats` 的主路径上**（Ruling R47 起改用
    `seat_candidates` + `rank_tracks_pairwise`）。保留是因为它仍是逐帧排序的
    构件、被本文件既有测试直接覆盖，删除超出了本任务范围。
    """
    right = exclude_judges(dets, frame_w)
    ordered = sorted(right, key=lambda d: d.bottom_y, reverse=True)
    return {d.track_id: i for i, d in enumerate(ordered[:presenter_num], start=1)}


def vote_seats(per_frame: list[dict[int, int]]) -> dict[int, int]:
    """逐帧结果按票数求全局最优的一对一匹配。

    不用贪心：贪心按首选票数排序一次即固定，占不到首选者会退取次选，
    却不回头检查该次选是否另有更该得到它的候选——遮挡导致 track 断裂时，
    碎片会凭溢出票抢走另一位真实创业者的座位，且不报错。
    指派问题求全局最优才安全。

    **不在 `assign_segment_seats` 的主路径上**（Ruling R47）：这套「逐帧排序+
    投票」本身在部分共现场景下有同样的零票丢弃问题——票数太少的真人轨迹会被
    匈牙利总票数最优判定为零票，从结果里彻底消失（详见 seats.py 顶部
    docstring）。保留是因为它仍被本文件既有测试直接覆盖，删除超出了本任务范围。
    """
    votes: dict[int, Counter] = {}
    for frame in per_frame:
        for tid, seat in frame.items():
            votes.setdefault(tid, Counter())[seat] += 1
    if not votes:
        return {}

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    tids = sorted(votes)
    seats = sorted({s for c in votes.values() for s in c})
    cost = np.zeros((len(tids), len(seats)), dtype=float)
    for i, t in enumerate(tids):
        for j, s in enumerate(seats):
            cost[i, j] = -votes[t][s]          # 取负以最大化票数

    rows, cols = linear_sum_assignment(cost)
    out: dict[int, int] = {}
    for i, j in zip(rows, cols):
        if votes[tids[i]][seats[j]] > 0:       # 零票不算匹配
            out[tids[i]] = seats[j]
    return out


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def frame_coverage(track: list[Detection], total_frames: int) -> float:
    """轨迹帧覆盖率 = 该轨迹出现的帧数 / 本段就座总帧数（Task 29 发现二 a）。

    实测：入选的碎片覆盖率可以低到 2%/4%/4%/7%/10%/11%/15%，而被它们挤掉的
    真人轨迹覆盖率高达 100%/98%/96%/74%/70%——覆盖率门槛把碎片挡在候选之外。
    用「该轨迹出现的帧数」而非 `len(track)`，是为了对同一帧内重复检测（理论上
    不该发生，但防御性地）免疫：按 (frame_path, t) 去重计数。
    """
    if total_frames <= 0:
        return 0.0
    distinct = {(d.frame_path, d.t) for d in track}
    return len(distinct) / total_frames


def is_duplicate_box(a: Detection, b: Detection) -> bool:
    """判定两个框是否为同一人被跟踪器拆成两条轨迹（Task 29 发现二 b）。

    实测用户点名的 4 对「同一人」：order1 tid16&29 小框被包含1.00/cx差4px、
    order4 tid8&7 0.97/5px、order9 tid4&10 0.99/13px、order12 tid5&7
    0.98/8px（框宽 187–296px，cx 差与框宽之比 0.02–0.07）。判据**同时**要求：
      - 小框被包含比例 ≥ `SEAT_DUP_CONTAIN`（交集 / 两框中较小者的面积）；
      - 中心 x 相差 < `SEAT_DUP_CX_RATIO * min(两框宽度)`——用户对绝对像素
        阈值提出正确质疑（镜头拉远时全部距离等比缩小，绝对阈值会跟着失效），
        改用相对两框中较窄者宽度的比例，随镜头距离同比例缩放。
    两条都要满足——单看包含比例会把并排而坐、恰好一个宽框把另一个框的横向
    范围罩住的两个真人误判为同一人（并排时 cx 差接近一整个框宽，比值≈1.0，
    远高于 0.25）；单看 cx 比值又会把纵深上明显不重叠的两个框（比如一高
    一矮，包含比很低）误判为重复。

    **站立旁路（Task 33）**：用户指出 order 14 座位1与座位2是同一人，一坐
    一站。控制器实测两框：
        座位1 tid11  x=(1046,1403) y=( 551,1079)  宽356 高528   坐姿框
        座位2 tid 4  x=(1041,1363) y=( 214,1069)  宽322 高855   站姿全身框
    小框被包含比例仅 0.870（阈值 0.90，差 0.03）——站姿框底比坐姿框底高出
    10px（站立时脚尖比坐着时椅子/桌沿更靠里一点），导致「一个框完全包住
    另一个框」不再严格成立，包含比路径漏判。但中心 x 差/框宽比值仅 0.071
    （阈值 0.25，远低于），且两框高度比值 855/528≈1.62，是典型的「同一人
    一坐一站」信号，不是两个真人差着身高并排坐。
    于是加一条**横向重叠**判据作为包含比路径之外的独立旁路（或的关系，
    不改动、不放松原有包含比路径——那条被现有测试锁着）：两框横向区间
    重叠比例 ≥ `SEAT_DUP_X_OVERLAP` 且中心 x 差比值 < `SEAT_DUP_CX_RATIO`
    且其中一框高度 ≥ 另一框的 1.4 倍（站姿明显比坐姿框高，1.4 留出安全
    边际：实测比值 1.62，站姿最低也该比坐姿高出一截，不能卡在 1.0 附近
    误吞两个身高悬殊但都是坐姿的真人）时，判为同一人。
    """
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    width_a, height_a = max(0.0, a.x2 - a.x1), max(0.0, a.y2 - a.y1)
    width_b, height_b = max(0.0, b.x2 - b.x1), max(0.0, b.y2 - b.y1)
    area_a, area_b = width_a * height_a, width_b * height_b
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return False
    containment = inter / smaller
    narrower_width = min(width_a, width_b)
    if narrower_width <= 0:
        return False
    cx_ratio = abs(a.cx - b.cx) / narrower_width
    if containment >= config.SEAT_DUP_CONTAIN and cx_ratio < config.SEAT_DUP_CX_RATIO:
        return True

    # 站立旁路：横向重叠 + 中心 x 相近 + 一高一矮（高度比 ≥1.4）。
    ix_overlap = max(0.0, ix2 - ix1)
    narrower_x_span = min(width_a, width_b)
    if narrower_x_span <= 0 or height_a <= 0 or height_b <= 0:
        return False
    x_overlap_ratio = ix_overlap / narrower_x_span
    taller, shorter = max(height_a, height_b), min(height_a, height_b)
    height_ratio = taller / shorter
    return (
        x_overlap_ratio >= config.SEAT_DUP_X_OVERLAP
        and cx_ratio < config.SEAT_DUP_CX_RATIO
        and height_ratio >= config.SEAT_DUP_STANDING_H_RATIO
    )


def cut_at_depth_gap(
    sorted_candidates: list[tuple[int, float, float]]
) -> list[tuple[int, float, float]]:
    """显式空操作：原样返回输入，从不截断（Task 31 停用，Task 40 起改为
    显式空操作）。

    历史：这曾是「深度断层处截断候选列表」的判据（Task 29 发现二 c）——
    输入为已按 y2 降序排列的 `(track_id, y2, height)` 三元组（越靠画面下方
    =越靠近镜头=越靠前），一旦「相邻两候选的 y2 差 / 上位者框高」超过某个
    比值就地截断——专家固定坐在最里面，深度断层是专家与真实创业者之间的
    物理特征，不是采样噪声。控制器实测两组视频共 75 对相邻候选，专家对应
    比值 0.53/0.56/0.59。

    停用原因：用户两次质疑段内相邻比值换组是否可靠，控制器改用跨段位置簇
    `presenter_depth_floor`（见该函数 docstring），信号更干净——创业者
    位置全在 y2≥890，专家位置全在 y2≤564，中间空白，且是跨 13 段稳定复现
    的结构，比段内相邻比值稳健。

    Task 31 停用时的写法是把阈值改成一个不会触发的大值（99.0）——这是
    "停用"的伪装写法：函数签名和调用方式看起来完全没变，实际上恒不触发
    （实测切断 0/14、0/7、0/21 对），读代码的人得先算一遍相邻比值才知道
    它没在工作。Task 40 起改为这个显式空操作：函数体不再读取任何配置项、
    不再做任何比较，直接返回输入——"停用"这件事从代码本身就能看出来，
    不需要一个魔数当密码。

    仍然保留（不删）是因为它是一次有记录、可复现的实验，`depth_gap_report`
    仍然产出同一份 y2/height 比值分布供换组标定参考；哪天某组视频确实呈现
    出干净的深度分层，把这个函数改回真正的截断逻辑就是恢复该机制的入口。
    """
    return list(sorted_candidates)


def ratios_around_threshold(
    ratios: list[float], threshold: float
) -> tuple[float | None, float | None]:
    """给定一组比值，返回阈值两侧最近的两个值 `(below, above)`。

    标定报告用：让人一眼看出阈值是落在数据的裂谷里，还是落在密集处——
    换一组视频后，这是判断 `SEAT_DEPTH_GAP_RATIO` 要不要重标定最快的信号，
    不用等全流程跑完、人工核对完才发现。`below`/`above` 任一侧没有数据时
    为 `None`；恰好等于阈值的算作 `below` 一侧（未超过阈值即未截断）。
    """
    below = max((r for r in ratios if r <= threshold), default=None)
    above = min((r for r in ratios if r > threshold), default=None)
    return below, above


def _track_representative_box(dets_of_track: list[Detection]) -> Detection:
    """用轨迹全部检测框各坐标的中位数合成一个代表框，供重复检测/深度断层判定用。

    中位数比首帧或末帧更抗噪——局部遮挡、跟踪抖动会让个别帧的框偏移。
    """
    sample = dets_of_track[0]
    return Detection(
        seg_order=sample.seg_order,
        kind=sample.kind,
        t=sample.t,
        frame_path=sample.frame_path,
        track_id=sample.track_id,
        x1=_median([d.x1 for d in dets_of_track]),
        y1=_median([d.y1 for d in dets_of_track]),
        x2=_median([d.x2 for d in dets_of_track]),
        y2=_median([d.y2 for d in dets_of_track]),
        conf=_median([d.conf for d in dets_of_track]),
    )


def _find_duplicate_losers(rep_boxes: dict[int, Detection]) -> dict[int, int]:
    """返回 {被剔除的 track_id: 保留的 track_id}。

    保留更靠近镜头者（y2 更大），不是覆盖率更高者——实测保留覆盖率更高者会在
    order 13 丢掉正主（Task 29 发现二 b），去重判据必须只看几何位置。
    """
    tids = sorted(rep_boxes)
    losers: dict[int, int] = {}
    for i, j in combinations(tids, 2):
        if i in losers or j in losers:
            continue
        a, b = rep_boxes[i], rep_boxes[j]
        if is_duplicate_box(a, b):
            if a.y2 >= b.y2:
                losers[j] = i
            else:
                losers[i] = j
    return losers


def merge_tracks_by_position(
    dets_of_segment: list[Detection],
    cx_tol: float | None = None,
    y2_tol: float | None = None,
) -> list[dict]:
    """把同一段内、同一位置的多条轨迹碎片合并成一个『人』（Task 31）。

    **与既有的 `is_duplicate_box`（Task 29 发现二 b）是两回事，顺序也不能
    颠倒**：
      - `is_duplicate_box` 判的是「一个框包住另一个框」——两条轨迹在**同一
        帧**里同时出现、几何上重叠（典型例子：YOLO 把一双手和身体分别检
        成两个框），靠「包含比例 + cx 相对宽度比值」判定。
      - `merge_tracks_by_position` 判的是「同一位置的时间碎片」——不同
        track_id **从不共现**，是跟踪器在遮挡/重识别失败后把同一条物理
        轨迹拆成先后相继的几段，只靠「位置」（cx、y2 中位数）就能认出
        它们是同一个人。判据里显式要求两条轨迹的帧集合交集为空，正是
        为了不吞掉 `is_duplicate_box` 该管的「同帧同时出现」场景——两条
        规则因此互斥，都要保留。

    **必须在覆盖率过滤之前跑，顺序反了会把碎片一条条砍掉**：实测 order 11
    有 4 条轨迹（tid34/12/29/36）挤在同一位置，y2 相差 <30px，各自覆盖率
    只有 7%/9%/4%（另一条 100%）。覆盖率过滤先跑的话，会把三条碎片当噪声
    依次剔除；先合并再算覆盖率，看到的才是「这个人是否全程在场」（四段
    并集覆盖率 100%），而不是「这一段碎片占了多少帧」。控制器实测顺序
    颠倒会让样本从 28 掉回 18/31（task-31-report.md）。

    合并规则：轨迹按 y2 降序遍历；若某条轨迹与已有某个「人」当前的 cx、y2
    代表值**都**相差小于容差（缺省读 `config.SEAT_MERGE_CX_TOL`/
    `SEAT_MERGE_Y2_TOL`），且与该人已合并的全部轨迹都没有共同出现过的帧，
    就并入该人（人的代表 cx/y2/height 随之更新为已合并轨迹各自中位数的
    中位数）；否则新建一个人。

    返回每个「人」一个 dict：
        track_ids: 合并进来的原始 track_id 列表（首个是发起该人的轨迹）
        cx / y2 / height: 合并后的代表位置
        frames: 帧集合的并集，按 (frame_path, t) 去重
        n_frames: len(frames)
    """
    if cx_tol is None:
        cx_tol = config.SEAT_MERGE_CX_TOL
    if y2_tol is None:
        y2_tol = config.SEAT_MERGE_Y2_TOL

    by_track: dict[int, list[Detection]] = {}
    for d in dets_of_segment:
        by_track.setdefault(d.track_id, []).append(d)

    stats: dict[int, dict] = {}
    for tid, trk in by_track.items():
        stats[tid] = {
            "cx": _median([d.cx for d in trk]),
            "y2": _median([d.y2 for d in trk]),
            "height": _median([d.height for d in trk]),
            "frames": {(d.frame_path, d.t) for d in trk},
        }

    ordered_tids = sorted(stats, key=lambda t: -stats[t]["y2"])

    people: list[dict] = []
    for tid in ordered_tids:
        s = stats[tid]
        match = None
        for person in people:
            same_pos = (abs(s["cx"] - person["cx"]) < cx_tol
                        and abs(s["y2"] - person["y2"]) < y2_tol)
            if same_pos and not (s["frames"] & person["frames"]):
                match = person
                break
        if match is None:
            people.append({
                "track_ids": [tid],
                "cx": s["cx"],
                "y2": s["y2"],
                "height": s["height"],
                "frames": set(s["frames"]),
            })
        else:
            match["track_ids"].append(tid)
            match["frames"] |= s["frames"]
            match["cx"] = _median([stats[t]["cx"] for t in match["track_ids"]])
            match["y2"] = _median([stats[t]["y2"] for t in match["track_ids"]])
            match["height"] = _median([stats[t]["height"] for t in match["track_ids"]])

    for person in people:
        person["n_frames"] = len(person["frames"])
    return people


def _representative_track_id(track_ids: list[int], shaped: dict[int, list[Detection]]) -> int:
    """合并出的『人』用哪条原始 track_id 作代表——帧数最多的那条，下游按
    track_id 取帧时能拿到最长、最完整的一段。

    这与 `_find_duplicate_losers` 的「保留更靠近镜头者」是两套不同的代表
    选择规则，服务不同目的，不能混用：那条是在两个**同时存在**的框之间
    选一个更可信的几何位置，这条是在**同一个人的多段时间碎片**之间选一个
    帧数最多、最适合拿来裁图的来源。帧数并列时取 track_id 更小者，保证
    结果可复现。
    """
    def n_frames(tid: int) -> int:
        return len({(d.frame_path, d.t) for d in shaped[tid]})
    return min(track_ids, key=lambda tid: (-n_frames(tid), tid))


def _pre_depth_gap_candidates(
    dets: list[Detection], segment: Segment, frame_w: int
) -> tuple[
    dict[int, list[Detection]], dict[int, Detection],
    list[tuple[int, float, float]], list[tuple[int, str]], dict[int, list[int]],
]:
    """跑完人形过滤 → 按位置合并轨迹碎片 → 覆盖率过滤 → 重复检测（Ruling R47
    + Task 29 发现二 a/b + Task 31），产出深度断层截断前的最终候选集合。
    `seat_candidates`、`depth_gap_report`、`presenter_depth_floor`、
    `assign_segment_seats_detailed` 共用这份逻辑，保证看到的是同一份候选、
    结论不会互相矛盾。

    返回 (存活『人』的检测列表——同人的多条轨迹已合并、track_id 统一改写
    为代表轨迹, 存活『人』的代表框, 按 y2 降序排列的 (track_id, y2, height)
    三元组, 到此为止累计的剔除原因, {代表 track_id: 合并进来的全部原始
    track_id})。
    """
    seated = [d for d in dets if d.seg_order == segment.order and d.kind == "seated"]
    right = exclude_judges(seated, frame_w)
    frame_h = frame_w * 0.75

    # --- 按段自适应放宽（Task 38）---
    #
    # 高度阈值与覆盖率阈值切在**连续分布的密集区**：实测三组右簇 236 个
    # track，高度占比 0.20-0.25 有 13 个、0.25-0.30 有 27 个，没有任何裂谷；
    # 覆盖率同样连续。这与右区阈值形成对照——后者落在 540px 宽的真空里
    # （630-1170 区间零个 track），是有依据的；这两个没有。
    #
    # 后果：右簇本有 187 个 track（足以覆盖 103 位创业者），被高度砍到 129、
    # 再被覆盖率砍到 90，最终只定出 74 个座位。被砍掉的不是"判定为非创业者"，
    # 而是"证据不够"——它们连画廊都进不去，用户无从纠正。
    #
    # 改为：先用常规阈值；若存活数不足该段 presenter_num，逐档放宽重来，
    # 直到够用或触底。依据是用户的人工标注人数（本项目最可靠的先验），
    # 不是又一个拍脑袋的常数。放宽有下限（_RELAX_FLOOR_*），守住 Ruling R47
    # 那条防线：order02_seat1 曾被一双手占座（186px vs 真人 363-588px），
    # 放宽到底也不能把手当成人。
    for h_ratio, cov_ratio, rel_ratio in _relaxation_ladder(segment.presenter_num):
        out = _candidates_at(right, segment, frame_w, frame_h,
                             h_ratio, cov_ratio, rel_ratio)
        if len(out[1]) >= segment.presenter_num:
            return out
    return out


def _relaxation_ladder(presenter_num: int) -> list[tuple[float, float]]:
    """(高度阈值, 覆盖率阈值) 由严到松的档位序列。

    第一档即现行常规值——候选本就充足的段行为完全不变（实测 41/45 段
    在常规档下已够，只有真正缺人的段才会往下走）。
    """
    h, c, r = (config.SEAT_MIN_H_RATIO, config.SEAT_MIN_FRAME_COVERAGE,
               config.SEAT_MIN_H_VS_TALLEST)
    steps = [(h, c, r)]
    while (h > config.SEAT_RELAX_H_FLOOR or c > config.SEAT_RELAX_COV_FLOOR
           or r > config.SEAT_RELAX_REL_FLOOR):
        h = max(config.SEAT_RELAX_H_FLOOR, h - 0.03)
        c = max(config.SEAT_RELAX_COV_FLOOR, c - 0.04)
        r = max(config.SEAT_RELAX_REL_FLOOR, r - 0.05)
        steps.append((h, c, r))
    return steps


def _candidates_at(
    right: list[Detection], segment: Segment, frame_w: float, frame_h: float,
    h_ratio: float, cov_ratio: float, rel_ratio: float,
):
    """在给定阈值档位下跑一遍候选筛选。原逻辑原样搬入，只把两个常数换成参数。"""

    by_track: dict[int, list[Detection]] = {}
    for d in right:
        by_track.setdefault(d.track_id, []).append(d)
    if not by_track:
        return {}, {}, [], [], {}

    heights = {tid: _median([d.height for d in trk]) for tid, trk in by_track.items()}
    tallest = max(heights.values())
    abs_thresh = h_ratio * frame_h
    rel_thresh = rel_ratio * tallest

    rejected: list[tuple[int, str]] = []
    shaped: dict[int, list[Detection]] = {}
    for tid in sorted(by_track):
        h = heights[tid]
        if h < abs_thresh:
            rejected.append((
                tid,
                f"框高中位数 {h:.0f}px 低于绝对阈值 {abs_thresh:.0f}px"
                f"（高度档位={h_ratio:.2f}），疑似手/胳膊等局部检测",
            ))
        elif h < rel_thresh:
            rejected.append((
                tid,
                f"框高中位数 {h:.0f}px 低于本段最高候选 {tallest:.0f}px 的"
                f" {rel_ratio:.0%}（{rel_thresh:.0f}px），疑似局部检测",
            ))
        else:
            shaped[tid] = by_track[tid]

    if not shaped:
        return {}, {}, [], rejected, {}

    # --- 按位置合并轨迹碎片（Task 31）：必须在覆盖率过滤之前，理由见
    # `merge_tracks_by_position` 的 docstring。
    flattened = [d for trk in shaped.values() for d in trk]
    people = merge_tracks_by_position(flattened)

    # --- 覆盖率过滤：本段就座总帧数 = 通过人形过滤的候选里所有轨迹的并集
    # 帧数；单个「人」的覆盖率用它合并后的并集帧数，而不是单条轨迹的帧数。
    total_frames = len({(d.frame_path, d.t) for trk in shaped.values() for d in trk})
    covered_people: list[dict] = []
    for person in people:
        cov = person["n_frames"] / total_frames if total_frames > 0 else 0.0
        if cov < cov_ratio:
            ids = person["track_ids"]
            note = "" if len(ids) == 1 else f"（合并碎片 track_id={sorted(ids)}）"
            for tid in ids:
                rejected.append((
                    tid,
                    f"低覆盖率：合并后帧覆盖率 {cov:.0%} 低于门槛"
                    f" {cov_ratio:.0%}（覆盖率档位）"
                    f"{note}，疑似碎片占座",
                ))
        else:
            covered_people.append(person)

    # --- 代表轨迹：帧数最多的那条，把合并进来的其它轨迹的检测框 track_id
    # 统一改写为代表轨迹，后续 rank_tracks_pairwise 才会把它们当成同一条
    # 轨迹处理，不会给同一个人的碎片分别排出不同名次、抢占不同座位号。
    merge_map: dict[int, list[int]] = {}
    merged_dets: dict[int, list[Detection]] = {}
    for person in covered_people:
        rep_tid = _representative_track_id(person["track_ids"], shaped)
        combined: list[Detection] = []
        for tid in person["track_ids"]:
            for d in shaped[tid]:
                combined.append(d if tid == rep_tid else replace(d, track_id=rep_tid))
        merged_dets[rep_tid] = combined
        merge_map[rep_tid] = sorted(person["track_ids"])

    # --- 重复检测：用『人』的代表框两两比较，剔除几何上重复的『人』。
    rep_boxes = {tid: _track_representative_box(trk) for tid, trk in merged_dets.items()}
    losers = _find_duplicate_losers(rep_boxes)
    for tid, winner in losers.items():
        rejected.append((
            tid,
            f"与 track_id={winner} 几何重复检测（小框被包含比例 ≥"
            f" {config.SEAT_DUP_CONTAIN}，中心 x 相差 < 较窄框宽度的"
            f" {config.SEAT_DUP_CX_RATIO:.0%}），疑似同一人被跟踪器拆成两条轨迹，"
            f"保留更靠近镜头者",
        ))
    deduped = {tid: trk for tid, trk in merged_dets.items() if tid not in losers}
    deduped_rep_boxes = {tid: rep_boxes[tid] for tid in deduped}
    merge_map = {tid: merge_map[tid] for tid in deduped}

    ordered_by_y2 = sorted(
        ((tid, deduped_rep_boxes[tid].y2, deduped_rep_boxes[tid].height) for tid in deduped),
        key=lambda p: -p[1],
    )
    return deduped, deduped_rep_boxes, ordered_by_y2, rejected, merge_map


def seat_candidates(
    dets: list[Detection], segment: Segment, frame_w: int
) -> tuple[list[Detection], list[tuple[int, str]]]:
    """筛出一个 segment 就座窗口里可信的座位候选（Ruling R47 + Task 29 + Task 31）。

    流程：
      1. `exclude_judges` 做右区过滤。
      2. 人形过滤（Ruling R47）：按轨迹框高中位数剔除手、胳膊等局部检测框
         （两条高度规则同时要求满足：绝对阈值 `SEAT_MIN_H_RATIO * frame_h`
         挡住小碎片；相对阈值 `SEAT_MIN_H_VS_TALLEST * 本段右区最高候选`
         对变焦漂移保底）。`frame_h` 按本项目一贯的 4:3 画幅换算，不额外开参数。
      3. 按位置合并轨迹碎片（Task 31）：`merge_tracks_by_position` 把同一
         位置、时间上不重叠的多条轨迹拼成一个『人』——**必须在下一步覆盖率
         过滤之前跑**，否则会把同一个人的碎片一条条当噪声砍掉（详见该函数
         docstring）。
      4. 覆盖率过滤（Task 29 发现二 a，Task 31 改用合并后的并集帧数）：
         帧覆盖率 < `SEAT_MIN_FRAME_COVERAGE` 的『人』是碎片，不参与座位分配。
      5. 重复检测（Task 29 发现二 b）：几何上是同一人的两个『人』只保留更靠
         近镜头（y2 更大）的那个。
      6. 深度断层截断（Task 29 发现二 c，Task 31 停用、Task 40 起改为显式
         空操作）：`cut_at_depth_gap` 现在原样返回输入，恒不截断——两次
         用户质疑段内比值换组是否可靠后，控制器改用跨段位置簇
         `presenter_depth_floor` 验证信号更干净，但因为它需要跨段（全部
         `segments`）数据、`seat_candidates` 只拿到单个 `segment`，接入
         主流程留给后续任务；这一步调用只是为了保留「深度断层截断」这个
         步骤在流程中的位置，不再实际过滤任何候选（详见 `cut_at_depth_gap`
         docstring）。

    返回 (保留的检测, 被剔除的 (track_id, 原因) 列表)——剔除必须可见，不能静默。
    """
    deduped, rep_boxes, ordered_by_y2, rejected, _merge_map = _pre_depth_gap_candidates(
        dets, segment, frame_w
    )
    if not deduped:
        return [], rejected

    survivors = cut_at_depth_gap(ordered_by_y2)   # 显式空操作，恒等于 ordered_by_y2
    survivor_ids = {tid for tid, _y2, _h in survivors}

    kept: list[Detection] = []
    for tid in sorted(survivor_ids):
        kept.extend(deduped[tid])
    return kept, rejected


def depth_gap_report(
    dets: list[Detection], segments: list[Segment], frame_w: int = 1440
) -> list[dict]:
    """标定报告（用户对绝对阈值提出质疑后新增）：任何单一比值阈值都可能在
    下一组视频里不合适，本函数让「这个阈值在本组合不合适」变得一眼可见、
    且不用重跑整条流水线、不用等人工核对完才发现。

    对每段进入深度断层截断前的候选（已过人形过滤、覆盖率过滤、重复检测）
    按 y2 降序排列，为每一对相邻候选产出一条记录：
        {"seg_order", "upper_tid", "lower_tid", "gap_px",
         "upper_height", "ratio", "cut"}
    `ratio` = gap_px / upper_height；`cut` 直接调用 `cut_at_depth_gap` 判定
    ——不再另起一套「和某个阈值比较」的独立逻辑（那样容易出现两套判据各说
    各话）。`cut_at_depth_gap` 现为显式空操作（Task 40），故 `cut` 恒为
    `False`：这正是「机制已停用」在诊断报告里应有的样子，而不是看起来像
    在工作、实际靠魔数才不触发。`ratio` 这一列本身仍然是有意义的诊断数据
    ——换组后想重新评估这条机制是否值得恢复时，这份分布就是起点。
    n 个候选产出 n-1 条相邻对记录。
    """
    report: list[dict] = []
    for segment in segments:
        _deduped, _rep, ordered_by_y2, _rejected, _merge_map = _pre_depth_gap_candidates(
            dets, segment, frame_w
        )
        survivor_ids = {tid for tid, _y2, _h in cut_at_depth_gap(ordered_by_y2)}
        for prev, cur in zip(ordered_by_y2, ordered_by_y2[1:]):
            upper_tid, upper_y2, upper_h = prev
            lower_tid, lower_y2, _lower_h = cur
            gap = upper_y2 - lower_y2
            ratio = gap / upper_h if upper_h > 0 else float("inf")
            report.append({
                "seg_order": segment.order,
                "upper_tid": upper_tid,
                "lower_tid": lower_tid,
                "gap_px": gap,
                "upper_height": upper_h,
                "ratio": ratio,
                "cut": lower_tid not in survivor_ids,
            })
    return report


def _cluster_positions(candidates: list[dict], tol: float) -> list[dict]:
    """把带 (cx, y2) 的候选按位置聚类，容差 `tol` 同时约束 cx 与 y2。

    与 `merge_tracks_by_position` 的区别：这里聚类的输入本就跨越不同的
    `seg_order`（不同段落间的候选天然不共现，不需要 `merge_tracks_by_position`
    那条「时间不重叠」的额外判据），聚类的目的也不是拼出一个人的全部帧，
    只是找出「哪些候选站在同一个物理座位上」。

    输入候选须含 `cx`、`y2`、`seg_order`、`selected` 字段。返回每个簇一个
    dict：`cx`、`y2`（簇内候选中位数）、`n_segments`（覆盖的不同 seg_order
    数）、`selection_rate`（候选中 `selected` 为真的比例）、`members`。
    """
    ordered = sorted(candidates, key=lambda c: -c["y2"])
    clusters: list[dict] = []
    for c in ordered:
        match = None
        for cl in clusters:
            if abs(c["cx"] - cl["cx"]) < tol and abs(c["y2"] - cl["y2"]) < tol:
                match = cl
                break
        if match is None:
            clusters.append({"cx": c["cx"], "y2": c["y2"], "members": [c]})
        else:
            match["members"].append(c)
            match["cx"] = _median([m["cx"] for m in match["members"]])
            match["y2"] = _median([m["y2"] for m in match["members"]])
    for cl in clusters:
        seg_orders = {m["seg_order"] for m in cl["members"]}
        cl["n_segments"] = len(seg_orders)
        cl["selection_rate"] = sum(1 for m in cl["members"] if m["selected"]) / len(cl["members"])
    return clusters


def presenter_depth_floor(
    dets: list[Detection], segments: list[Segment], frame_w: int = 1440
) -> float:
    """跨段位置聚类找『创业者位置带』下界（Task 31 改动二），取代已停用的
    段内相邻比值判据 `SEAT_DEPTH_GAP_RATIO`。

    用户两次质疑那条比值换组是否可靠，控制器确认它切在数据密集区
    （p50≈0.49，阈值 0.52）。改用跨段位置聚类后信号干净得多——实测两组
    视频里，创业者位置全在 y2≥890，专家位置全在 y2≤564，中间是空白，
    而且这是跨 13 段稳定复现的结构（例：创业者位置 y2=1054 出现 8 段、
    8/8 入座位表；专家位置 y2=477 出现 5 段、0/5 入座位表，其中含 2 个
    用户确认的专家），比段内相邻比值稳健得多。

    做法：
      1. 对每一段跑 `_pre_depth_gap_candidates`（同 `depth_gap_report`，取
         覆盖率过滤 + 重复检测后、深度断层截断前的候选），按 y2 降序标注
         每个候选是否排进前 `segment.presenter_num` 名（等价于「不做深度
         过滤时是否会入座位表」）。
      2. 把全部段的候选按位置（cx、y2 都相差 < `config.SEAT_CLUSTER_TOL`）
         跨段聚类——同一个物理座位在不同段落间位置基本不变，聚类天然把
         它们并到一起。
      3. 簇按 y2 降序（越靠镜头越靠前）扫描，找到第一个「入选比例 <20% 且
         出现段数 ≥3」的簇——这是首个稳定复现的『专家席』位置，「出现段数
         ≥3」是为了不被单段的偶然噪声（例如某段恰好有个从未选中的路人）
         误判成专家席。取它上方最近一个簇（更靠镜头、创业者席）的 y2 作为
         下界。全程找不到这样的簇时返回 0（不过滤）。

    附带打印诊断（每个簇的 y2、出现段数、入选比例，以及最终选定的下界）
    供控制器/复核人员目视核对——这正是本函数存在的意义之一：换组后不用
    等全流程跑完、人工逐个核对完才发现聚类结构不对。
    """
    candidates: list[dict] = []
    for segment in segments:
        _deduped, rep_boxes, ordered_by_y2, _rejected, _merge_map = _pre_depth_gap_candidates(
            dets, segment, frame_w
        )
        for rank, (tid, y2, height) in enumerate(ordered_by_y2):
            candidates.append({
                "seg_order": segment.order,
                "track_id": tid,
                "cx": rep_boxes[tid].cx,
                "y2": y2,
                "height": height,
                "selected": rank < segment.presenter_num,
            })

    if not candidates:
        print("presenter_depth_floor 诊断：无候选，下界=0（不过滤）")
        return 0.0

    clusters = _cluster_positions(candidates, config.SEAT_CLUSTER_TOL)
    ordered = sorted(clusters, key=lambda c: -c["y2"])

    print(f"presenter_depth_floor 诊断：{len(ordered)} 个跨段位置簇"
          f"（SEAT_CLUSTER_TOL={config.SEAT_CLUSTER_TOL}px）")
    for cl in ordered:
        print(f"   y2={cl['y2']:.0f}px  出现{cl['n_segments']}段  "
              f"入选比例{cl['selection_rate']:.0%}（{len(cl['members'])} 条候选）")

    # 下界取「稳定位置簇的 y2 分布中最大的那道缝」的中点。
    #
    # 早先的做法是找「入选比例 <20%」的首个簇——那是**循环的**：入选比例
    # 取决于当前的深度过滤，而深度过滤又要用这个下界。实测后果：
    # SEAT_DEPTH_GAP_RATIO 停用后入选比例全变，本函数随即返回 0（不过滤），
    # 用户确认的 2 位专家（order 4 tid13、order 10 tid2）又回到了座位表。
    #
    # 改用纯几何的「最大间隔」后不再循环。文化4组实测的簇结构：
    #     y2=1054(8段) 976(4段) 890(5段) │ 619(4段) 564(4段) 481(5段) 477(5段)
    # 最大间隔 890→619 共 270px，次大间隔只有 86px（976→890），差 3 倍以上；
    # 取中点 755 时，用户确认的 3 位专家（y2=557/528/532）全部落在其下。
    #
    # 「出现段数 ≥3」的门槛照旧：单段的偶然噪声（某段恰好有个路人）不该
    # 被当成稳定的专家席位置。
    stable = [cl for cl in ordered if cl["n_segments"] >= 3]
    if len(stable) < 2:
        print("   → 稳定位置簇不足 2 个，下界=0（不过滤）")
        return 0.0

    gaps = [(stable[i]["y2"] - stable[i + 1]["y2"], i) for i in range(len(stable) - 1)]
    best_gap, idx = max(gaps)
    runner_up = max((g for g, _ in gaps if g != best_gap), default=0.0)
    floor = (stable[idx]["y2"] + stable[idx + 1]["y2"]) / 2.0

    print(f"   → 最大间隔 {stable[idx]['y2']:.0f}→{stable[idx + 1]['y2']:.0f}px "
          f"（{best_gap:.0f}px，次大 {runner_up:.0f}px），选定下界 y2={floor:.0f}px")
    if runner_up and best_gap < runner_up * 1.5:
        print("   ⚠ 最大间隔不足次大的 1.5 倍，位置带分离不明显，建议人工核对")
    return floor


def rank_tracks_pairwise(dets_of_segment: list[Detection]) -> list[int]:
    """按成对比较排出 track_id 的座位顺序（最靠画面下方者在前）（Ruling R47）。

    只要求两条轨迹在**同一帧**里共现过，不要求全员同时出现——遮挡或跟踪器
    断裂会让某条真人轨迹只在少数帧里露面，逐帧排序+投票会让它零票出局
    （`vote_seats` 的老毛病）。成对比较里，只要它在共现的那几帧里稳定压过
    对手，就足以定出名次。

    Copeland 计分：对每一对在某些帧里共现过的轨迹，看谁在共现帧里 bottom_y
    更大（更靠下）的次数更多，胜者记一分。**显式为每条候选轨迹初始化 0 分**——
    用裸 Counter 累加时，0 胜场的轨迹不会出现在字典里，会被静默丢弃
    （本项目已栽过三次同类「零值导致条目消失」的坑：R32 的 `pixel={}`、R39 的
    `formality` 恒为 4、这里是第三次）。
    平局（Copeland 分相同）时用各自 bottom_y 中位数做二级键，保证结果可复现。
    """
    by_frame: dict[tuple, list[Detection]] = {}
    for d in dets_of_segment:
        by_frame.setdefault((d.frame_path, d.t), []).append(d)

    track_ids = sorted({d.track_id for d in dets_of_segment})
    if not track_ids:
        return []

    # 每一对轨迹在共现帧里的胜负计数：{(a, b): [a_wins, b_wins]}，a < b。
    pair_wins: dict[tuple[int, int], list[int]] = {}
    for frame_dets in by_frame.values():
        present: dict[int, float] = {}
        for d in frame_dets:
            present[d.track_id] = d.bottom_y   # 同帧同 track 理论上只有一个框
        ids = sorted(present)
        for a, b in combinations(ids, 2):
            counts = pair_wins.setdefault((a, b), [0, 0])
            if present[a] > present[b]:
                counts[0] += 1
            elif present[b] > present[a]:
                counts[1] += 1
            # 恰好相等：本帧不计入任何一方

    copeland: dict[int, int] = dict.fromkeys(track_ids, 0)   # 显式初始化 0 分
    for (a, b), (a_wins, b_wins) in pair_wins.items():
        if a_wins > b_wins:
            copeland[a] += 1
        elif b_wins > a_wins:
            copeland[b] += 1
        # 共现帧数打平：这一对互不给分

    def _median(vals: list[float]) -> float:
        s = sorted(vals)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0

    bottoms: dict[int, list[float]] = {}
    for d in dets_of_segment:
        bottoms.setdefault(d.track_id, []).append(d.bottom_y)
    median_bottom = {tid: _median(v) for tid, v in bottoms.items()}

    return sorted(track_ids, key=lambda t: (-copeland[t], -median_bottom[t], t))


def gender_match_candidates(
    dets: list[Detection], segment: Segment, frame_w: int = 1440,
    depth_floor: float = 0.0, gender_judged_by_track: dict[int, str] | None = None,
) -> list[dict]:
    """为性别匹配（Task 33，`pipeline/seatmatch.py`）准备候选列表。

    与 `assign_segment_seats` 走同一套 `seat_candidates` → 深度过滤 →
    `rank_tracks_pairwise` 候选，取前 `segment.presenter_num` 名，各自算出
    代表 y2（该轨迹全部 `bottom_y` 的中位数）与从 `gender_judged_by_track`
    查到的 VLM 判定（查不到时记「不确定」，作为 `match_seats_by_gender`
    的通配符）。`assign_segment_seats(gender_match=True)` 与 `run_all.py`
    阶段 4 的性别标定报告打印共用这份逻辑，避免各自重算一次候选、结果
    互相矛盾。
    """
    # 必须与 `assign_segment_seats` 走同一个 `_ranked_tracks_dropping_expert`
    # ——此前这里直接调 `rank_tracks_pairwise`，绕过了 Innermost_expert 剔除。
    # 后果：文化4组（该变量为 1）在性别匹配开启时座位数恒等于 presenter_num，
    # 专家被当成创业者填进座位，而 seats.json 与挑选页都显示「缺 0 人」，
    # 看上去一切正常。控制器是因为两条路径给出不同座位数（基线 22/34、
    # 性别匹配 34/34）才发现的，此前无任何测试守护这个一致性。
    kept, order, _dropped = _ranked_tracks_dropping_expert(
        dets, segment, frame_w, depth_floor)
    top = order[: segment.presenter_num]
    gjt = gender_judged_by_track or {}
    people: list[dict] = []
    for tid in top:
        bottoms_t = [d.bottom_y for d in kept if d.track_id == tid]
        people.append({
            "track_id": tid,
            "y2": _median(bottoms_t) if bottoms_t else 0.0,
            "gender_judged": gjt.get(tid, "不确定"),
        })
    return people


def _ranked_tracks_dropping_expert(
    dets: list[Detection], segment: Segment, frame_w: float, depth_floor: float,
) -> tuple[list[Detection], list[int], int | None]:
    """`seat_candidates` → 可选深度过滤 → `rank_tracks_pairwise` 排序 →
    按 `Innermost_expert` 剔除最里侧那位（Task 44）。

    单一实现供 `assign_segment_seats` 与 `expert_removed_tracks` 共用——
    两处各写一份剔除逻辑必然走样（Task 44 brief 明确提醒的风险）。返回
    (过滤后的 kept detections, 剔除专家后的 track_id 排序, 被剔除的
    track_id 或 None)。

    --- Innermost_expert：用户 2026-08-17 在 dta 末尾补充的组级变量 ---
    标记「会场最里侧坐着一位可见的专家」。用户实测三组：城市2=0、
    文化4=1、新能源2=0——文化4 正是他逐企业核对时列出一串「误把专家
    当创业者」的那一组，两者吻合。

    这与已被证伪的深度断层（SEAT_DEPTH_GAP_RATIO）不同：那个想让算法
    从候选间隙里**自动找出**专家与创业者的分界，实测两者深度完全重叠
    （专家 y2=528-557，真实后排创业者 y2=464-578），无间隙可用。这里是
    用户直接告知「这一组有没有」，算法只需按已知位置剔除，不必去猜。

    候选数恰等于 presenter_num 时**仍然剔除**。

    曾按用户要求改为「只在候选多于人数时才剔除」，但实测证明那会让整个
    变量失效：排序本就按离镜头远近，最里侧那位天然排最后，候选一有富余
    就已被 top-N 截掉。文化4组候选多于人数的 3 段（order 2/3/12）中，
    最里侧的 tid9/tid8/tid14 全部本来就在 top-N 之外——剔除无事可做。
    也就是说：候选=人数（13 段里 9 段）才是剔除唯一能起作用的情形。

    现行取舍：一律剔除。实测文化4组一致率 82%→95%（确实剔掉了专家），
    代价是样本 30→25。用户要求准确度与样本同等重要，Task 44 起补上「剔除
    后从进出场候选递补一位」（`entryexit.entryexit_fill_plan` + `assemble.
    assemble_annotations` 的 `entry_labels` 接线），两者兼得。
    """
    kept, _rejected = seat_candidates(dets, segment, frame_w)
    if depth_floor > 0:
        bottoms: dict[int, list[float]] = {}
        for d in kept:
            bottoms.setdefault(d.track_id, []).append(d.bottom_y)
        deep = {t for t, v in bottoms.items() if _median(v) < depth_floor}
        kept = [d for d in kept if d.track_id not in deep]
    order = rank_tracks_pairwise(kept)

    removed: int | None = None
    if segment.innermost_expert and order:
        bottoms: dict[int, list[float]] = {}
        for d in kept:
            bottoms.setdefault(d.track_id, []).append(d.bottom_y)
        # y2 越小离镜头越远；最里侧那位即 y2 中位数最小者
        innermost = min(order, key=lambda t: _median(bottoms.get(t, [1e9])))
        order = [t for t in order if t != innermost]
        kept = [d for d in kept if d.track_id != innermost]
        removed = innermost

    return kept, order, removed


def expert_removed_tracks(
    dets: list[Detection], segment: Segment, frame_w: int = 1440,
) -> list[int]:
    """本段因 `Innermost_expert` 被剔除的 track_id（未标记该变量时返回 []）。

    与 `assign_segment_seats` 共用同一个内部纯函数
    `_ranked_tracks_dropping_expert`，保证两处看到的是同一份剔除结果，不是
    各自算一遍（Task 44）——供装配阶段（`pipeline.assemble`）标记
    `expert_removed` 这个 CSV 标志字段用，不重跑一套可能走样的判定逻辑。

    depth_floor 缺省 0（与 `assign_segment_seats` 默认一致，见其
    docstring）——该参数在生产环境从未真正打开，这里同样不暴露给调用方，
    避免两处默认值不同步导致「同一段」在两个函数里判定不一致。
    """
    _kept, _order, removed = _ranked_tracks_dropping_expert(dets, segment, frame_w, 0.0)
    return [removed] if removed is not None else []


def assign_segment_seats(
    dets: list[Detection], segment: Segment, frame_w: int = 1440,
    depth_floor: float = 0.0, gender_match: bool = False,
    gender_judged_by_track: dict[int, str] | None = None,
) -> dict[int, int]:
    """对一个 segment 的就座窗口检测结果分配座位号（Ruling R47）。

    流程：`seat_candidates` 剔除非完整人形的伪影 → `rank_tracks_pairwise` 按
    成对比较排出优先级 → 取最靠前的 `presenter_num` 条依次编号 1..N。
    候选数不足 `presenter_num` 时只返回实际有的座位，不报错、不凑数
    （order 7 是已实证的物理限制：第二位创业者完全不在摄像机视野内）。

    depth_floor：创业者位置带的 y2 下界，由 `presenter_depth_floor` 跨全部
    段算出。**缺省 0（不启用），且实测已被证伪，不要贸然打开**——
    用户核对第二组视频后指出 order 8/12 各有 3 位创业者（可由头部/面部确认），
    而实测这些真人位于 y2=578/464（order 8）与 y2=520/493/444（order 12），
    与用户确认的 3 位专家（y2=557/528/532）**处于同一深度带**。
    「创业者带与专家带之间有 270px 空白」是抽样偏差造成的假象：那批统计
    只覆盖了通过覆盖率过滤的候选，恰好漏掉了后排创业者。
    **结论：单靠深度分不开后排创业者与专家。** 参数保留是为了可复现该实验，
    以及将来若有别的视频组确实呈现清晰分层时可按组启用。原文档见 Task 31。

    历史口径（供追溯）：由 `presenter_depth_floor` 跨全部段算出（本函数只看单段，拿不到跨段结构，故必须由调用方传入；缺省 0
    表示不过滤，保持既有行为）。用户核对第二组视频时确认「专家坐在最里面、
    位置固定不变」，实测创业者位置带与专家位置带之间有 270px 的空白，
    远大于带内相邻座位的 86px 间距。低于下界的候选一律不占座位——**哪怕
    因此凑不满 presenter_num**，因为凑数只会把专家填进空缺（实测 order 4
    与 order 10 正是如此）。

    gender_match（Task 33）：默认 False，保持既有纯几何排序不变（几何路线
    连续三次证伪——深度分不开专家与后排创业者、覆盖率过滤顺序错砍碎片、
    站立者的脚落地导致框底与前排齐平被误排靠前）。开启后，若
    `segment.presenters` 的人工性别标注能与 `gender_judged_by_track` 提供
    的 VLM 判定通过段内重排对上，采用重排后的座位号（见
    `pipeline/seatmatch.match_seats_by_gender`）；对不上、或没有 VLM 数据
    可用（`gender_judged_by_track` 为 None、或查不到某个 track_id，一律
    视为该人「不确定」通配符）时，退回本函数原有的纯 y2 排序，与
    `gender_match=False` 结果完全一致。

    **`gender_judged_by_track` 从哪来是架构上的先后顺序问题，不是遗漏**：
    `gender_judged` 是 VLM 独立判定，只在阶段 7（打标装配）才产出，而阶段
    5 又要先有本函数给出的座位号才能跑——先有座位才能剪出证据图、才能调
    VLM。`run_all.py` 阶段 4 因此只能在**重跑**时（读到上一轮
    annotations.json 里已经产出的 `gender_judged`，按 (seg_order, track_id)
    对上）才真正用上性别信息去修正座位；首次运行时没有任何历史 VLM 结果，
    `gender_match=True` 等价于 `gender_match=False`，这是刻意的（不能循环
    依赖），不是没接上。
    """
    kept, order, _removed = _ranked_tracks_dropping_expert(dets, segment, frame_w, depth_floor)

    top = order[:segment.presenter_num]
    baseline = {tid: i for i, tid in enumerate(top, start=1)}

    if not gender_match or not segment.presenters:
        return baseline

    annotated = [p.gender_female for p in segment.presenters]
    people = gender_match_candidates(dets, segment, frame_w, depth_floor, gender_judged_by_track)
    if not people or len(people) != len(annotated):
        return baseline

    from pipeline.seatmatch import match_seats_by_gender
    return match_seats_by_gender(people, annotated)


def partial_detections(
    dets: list[Detection], segment: Segment, frame_w: int = 1440
) -> dict[int, bool]:
    """哪些入选 track 只是局部检测（手/肩等），衣着无法编码。

    用户 2026-08-17 指出 Ruling R47 的推理错了：手不是"不是人"，而是
    "只拍到手的人"。既然要让它占位，就必须同时标出来——否则下游会拿一双
    手去编码正式度，把"测不了"伪装成"测到了"。

    判据用相对量而非绝对像素：框高低于本段最高候选的 SEAT_PARTIAL_REL
    即视为局部。绝对像素会随机位与分辨率漂移，而"比同段最高的人矮一大截"
    在任何机位下都成立。
    """
    deduped, rep_boxes, _o, _r, _m = _pre_depth_gap_candidates(dets, segment, frame_w)
    if not rep_boxes:
        return {}
    tallest = max(b.y2 - b.y1 for b in rep_boxes.values())
    if tallest <= 0:
        return {}
    return {tid: (b.y2 - b.y1) < config.SEAT_PARTIAL_REL * tallest
            for tid, b in rep_boxes.items()}


def assign_segment_seats_detailed(
    dets: list[Detection], segment: Segment, frame_w: int = 1440
) -> dict[int, dict]:
    """与 `assign_segment_seats` 结果一一对应，但额外暴露每个座位由哪些
    原始 track_id 合并而来（Task 31），供 Task 15 装配阶段按需取全部帧用。

    不改 `assign_segment_seats` 本身——它的 `{track_id: seat_index}` 返回
    结构是下游既有约定，本任务禁止改动。这里另起一个函数，内部复用同一份
    `_pre_depth_gap_candidates` 逻辑，保证与 `assign_segment_seats` 看到的
    是同一份候选、算出同一个座位分配，只是多带一份合并信息。

    返回 {seat_index: {"track_id": 代表轨迹 id（与 assign_segment_seats
    返回值里的键相同）, "track_ids": 合并进来的全部原始 track_id（y2 降序）,
    "n_frames": 合并后的并集帧数}}。
    """
    deduped, _rep_boxes, ordered_by_y2, _rejected, merge_map = _pre_depth_gap_candidates(
        dets, segment, frame_w
    )
    if not deduped:
        return {}

    survivors = cut_at_depth_gap(ordered_by_y2)
    survivor_ids = [tid for tid, _y2, _h in survivors]

    kept: list[Detection] = []
    for tid in sorted(survivor_ids):
        kept.extend(deduped[tid])
    order = rank_tracks_pairwise(kept)

    result: dict[int, dict] = {}
    for i, tid in enumerate(order[:segment.presenter_num], start=1):
        result[i] = {
            "track_id": tid,
            "track_ids": merge_map.get(tid, [tid]),
            "n_frames": len({(d.frame_path, d.t) for d in deduped[tid]}),
        }
    return result


def threshold_samples(
    dets: list[Detection], segments: list[Segment], frame_w: int = 1440
) -> dict[str, list[float]]:
    """收集阶段④里当前真正在起作用的六个阈值各自的样本分布（Task 40），
    供 `pipeline.audit.threshold_health` 体检——每个阈值配的样本是它在
    生产代码里实际比较的那批数值，不是随手挑的代理指标：

        RIGHT_REGION_X            本组全部就座 track 的中位 cx / frame_w
        SEAT_MIN_H_RATIO          右区 track 的中位框高 / frame_h
        SEAT_MIN_H_VS_TALLEST     右区 track 的中位框高 / 本段右区最高候选
        SEAT_MIN_FRAME_COVERAGE   右区 track 的帧覆盖率
        SEAT_DUP_CONTAIN          本段两两候选代表框的包含比例
        SEAT_PARTIAL_REL          seat_candidates 存活候选的框高 / 本段最高

    `SEAT_MIN_H_RATIO`/`SEAT_MIN_H_VS_TALLEST`/`SEAT_MIN_FRAME_COVERAGE` 三
    个用的是**未经这三条阈值本身过滤**的右区候选（人形过滤前的原始右区
    track）——体检要看的是「阈值切在哪」，样本若已经被同一阈值过滤过，
    分布会被阈值本身污染，看不出真实的裂谷/密集区。`SEAT_DUP_CONTAIN` 与
    `SEAT_PARTIAL_REL` 则相反：前者要看的是「候选两两之间」这个关系本身，
    与人形过滤是否跑过无关；后者按定义就是「已经进入 seat_candidates 存活
    集合的那些」，样本天然就是过滤后的产物。
    """
    frame_h = frame_w * 0.75
    samples: dict[str, list[float]] = {
        "RIGHT_REGION_X": [],
        "SEAT_MIN_H_RATIO": [],
        "SEAT_MIN_H_VS_TALLEST": [],
        "SEAT_MIN_FRAME_COVERAGE": [],
        "SEAT_DUP_CONTAIN": [],
        "SEAT_PARTIAL_REL": [],
    }

    for segment in segments:
        seated = [d for d in dets if d.seg_order == segment.order and d.kind == "seated"]
        if not seated:
            continue

        # --- RIGHT_REGION_X：本段全部就座 track（不分左右） ---
        by_all_track: dict[int, list[Detection]] = {}
        for d in seated:
            by_all_track.setdefault(d.track_id, []).append(d)
        for trk in by_all_track.values():
            samples["RIGHT_REGION_X"].append(_median([d.cx for d in trk]) / frame_w)

        # --- 右区 track：高度 / 相对高度 / 覆盖率 ---
        right = exclude_judges(seated, frame_w)
        if not right:
            continue
        by_right_track: dict[int, list[Detection]] = {}
        for d in right:
            by_right_track.setdefault(d.track_id, []).append(d)
        heights = {tid: _median([d.height for d in trk]) for tid, trk in by_right_track.items()}
        tallest = max(heights.values())
        total_frames = len({(d.frame_path, d.t) for d in right})
        for tid, trk in by_right_track.items():
            h = heights[tid]
            samples["SEAT_MIN_H_RATIO"].append(h / frame_h)
            if tallest > 0:
                samples["SEAT_MIN_H_VS_TALLEST"].append(h / tallest)
            samples["SEAT_MIN_FRAME_COVERAGE"].append(frame_coverage(trk, total_frames))

        # --- SEAT_DUP_CONTAIN：右区 track 两两代表框的包含比例 ---
        rep_boxes = {tid: _track_representative_box(trk) for tid, trk in by_right_track.items()}
        for i, j in combinations(sorted(rep_boxes), 2):
            a, b = rep_boxes[i], rep_boxes[j]
            width_a, height_a = max(0.0, a.x2 - a.x1), max(0.0, a.y2 - a.y1)
            width_b, height_b = max(0.0, b.x2 - b.x1), max(0.0, b.y2 - b.y1)
            ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
            ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            smaller = min(width_a * height_a, width_b * height_b)
            if smaller > 0:
                samples["SEAT_DUP_CONTAIN"].append(inter / smaller)

        # --- SEAT_PARTIAL_REL：seat_candidates 最终存活集合的相对高度 ---
        _deduped, kept_rep_boxes, _o, _r, _m = _pre_depth_gap_candidates(dets, segment, frame_w)
        if kept_rep_boxes:
            kept_tallest = max(b.y2 - b.y1 for b in kept_rep_boxes.values())
            if kept_tallest > 0:
                for b in kept_rep_boxes.values():
                    samples["SEAT_PARTIAL_REL"].append((b.y2 - b.y1) / kept_tallest)

    return samples
