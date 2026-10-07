from pathlib import Path
import pytest
from pipeline import config
from pipeline.detect import Detection
from pipeline.seats import (
    exclude_judges, rank_seats_in_frame, vote_seats, assign_segment_seats,
    seat_candidates, rank_tracks_pairwise,
    is_duplicate_box, frame_coverage, cut_at_depth_gap,
    depth_gap_report, ratios_around_threshold,
    merge_tracks_by_position, presenter_depth_floor, assign_segment_seats_detailed,
    gender_match_candidates,
)
from pipeline.schedule import Segment, Presenter

W = 1440


def det(tid, cx, bottom, h=300.0, conf=0.9, t=0.0, frame_path=None, order=2):
    if frame_path is None:
        frame_path = Path("f.jpg")
    half = 60.0
    return Detection(order, "seated", t, frame_path, tid,
                     cx - half, bottom - h, cx + half, bottom, conf)


def test_left_side_detections_are_excluded():
    dets = [det(1, 200, 500), det(2, 350, 520), det(3, 1300, 700)]
    kept = exclude_judges(dets, W)
    assert [d.track_id for d in kept] == [3]


def test_seat_one_is_the_lowest_in_frame():
    # 三位创业者：bottom_y 越大越靠画面下方，越靠近镜头，编号越小
    dets = [det(7, 1300, 400), det(8, 1320, 620), det(9, 1350, 880)]
    seats = rank_seats_in_frame(dets, W, presenter_num=3)
    assert seats == {9: 1, 8: 2, 7: 3}


def test_only_presenter_num_people_are_kept():
    # 检出 4 人但标注只有 2 人：保留最靠下的 2 个
    dets = [det(1, 1300, 300), det(2, 1310, 500), det(3, 1320, 700), det(4, 1330, 900)]
    seats = rank_seats_in_frame(dets, W, presenter_num=2)
    assert seats == {4: 1, 3: 2}


def test_ranking_survives_camera_pan():
    # 摄像机右移 80px、下移 40px，绝对坐标全变，但相对次序不变
    base = [det(7, 1300, 400), det(8, 1320, 620), det(9, 1350, 880)]
    panned = [det(7, 1380, 440), det(8, 1400, 660), det(9, 1430, 920)]
    assert rank_seats_in_frame(base, W, 3) == rank_seats_in_frame(panned, W, 3)


def test_vote_seats_takes_majority():
    frames = [{5: 1, 6: 2}, {5: 1, 6: 2}, {5: 2, 6: 1}]
    assert vote_seats(frames) == {5: 1, 6: 2}


def test_vote_seats_resolves_duplicate_assignment():
    # 投票后两个 track 抢同一座位，须保证结果中座位号唯一
    frames = [{5: 1, 6: 1}, {5: 1, 6: 2}, {5: 1, 6: 1}]
    got = vote_seats(frames)
    assert len(set(got.values())) == len(got), "同段内座位号不得重复"


def test_vote_seats_does_not_let_a_fragment_steal_a_real_presenter():
    # 101/102 是同一人被遮挡拆成的两段碎片（都应属座位 1），
    # 200 是另一位真实创业者（座位 2）。
    # 贪心实现会让 102 凭 8 张溢出票抢走 200 的座位，把 200 完全抹掉。
    frames = ([{101: 1}] * 23 + [{102: 1}] * 15
              + [{102: 2}] * 8 + [{200: 2}] * 10)
    got = vote_seats(frames)
    assert got.get(200) == 2, f"真实创业者 200 应获得座位 2，实得 {got}"
    assert got.get(101) == 1
    assert 102 not in got, "碎片不应获得座位"


def test_assign_segment_seats_end_to_end():
    # y2 间距改为 100px（Task 29 起 SEAT_DEPTH_GAP=200 会在相邻候选断层
    # >200px 处截断——原先 220/260px 的间距是早于该规则的随意取值，不代表
    # 真实相邻座位间距（实测 order2 真人相邻座位 y2 差仅 112–151px），
    # 现改用低于门槛的间距，避免新规则把座位2、3 误判为深度断层后的候选。
    seg = Segment(2, "北斗羲和", 960, 1836, 3,
                  (Presenter(1, 0, 1, 0), Presenter(2, 0, 0, 1), Presenter(3, 0, 0, 1)))
    dets = []
    for t in range(5):
        dets += [
            Detection(2, "seated", float(t), Path("f.jpg"), 7, 1240, 430, 1360, 730, 0.9),
            Detection(2, "seated", float(t), Path("f.jpg"), 8, 1260, 530, 1380, 830, 0.9),
            Detection(2, "seated", float(t), Path("f.jpg"), 9, 1290, 630, 1410, 930, 0.9),
            Detection(2, "seated", float(t), Path("f.jpg"), 1, 150, 300, 270, 600, 0.9),
        ]
    seats = assign_segment_seats(dets, seg, frame_w=W)
    assert seats == {9: 1, 8: 2, 7: 3}
    assert 1 not in seats, "左侧评委不应获得座位号"


# ---------------------------------------------------------------------------
# Task 18 (Ruling R47)：座位候选须为完整人形，排序改用成对比较抗碎片化
# ---------------------------------------------------------------------------

def _seg(order, presenter_num):
    presenters = tuple(Presenter(i + 1, 0, 0, 1) for i in range(presenter_num))
    return Segment(order, "seg", 0, 100, presenter_num, presenters)


def test_hand_sized_box_is_rejected_as_seat_candidate():
    """复刻 order2 的 tid=5：186px 高的手，底边比三条真人轨迹都更靠下（877），
    在旧的逐帧排序下会抢走座位 1。人形过滤必须把它挡在候选之外。

    Task 33：track1/10/22 的 cx 从原始复刻值向外错开（`det()` 助手对全部
    track 固定用 120px 宽，不像真实检测框那样随深度等比收窄——这是助手的
    简化，不是真实几何），避免与新增的站立旁路（横向重叠 ≥0.80 且 cx 差/
    框宽 <0.25 且高度比 ≥1.4）意外重合：原始 cx（1335/1300）在 22↔10 这对
    上恰好落在旁路三个条件全部满足的边界内（重叠0.858、cx比0.142、高度比
    1.504），被误判成『同一人站/坐』，与本测试真正想验证的「三个真实不同
    深度的人都应保留」矛盾。调宽 cx 间距后，横向重叠比例降到 0.80 以下，
    三人互相都不再触发站立旁路，同时不影响本测试原本依赖的 y2/height 排序
    与人形过滤断言。"""
    seg = _seg(2, 3)
    dets = [
        det(5, 1359, 877, h=186),   # 手，最靠下但太矮
        det(22, 1317, 866, h=546),  # 真人，座位 1
        det(1, 1400, 715, h=425),   # 真人，座位 2
        det(10, 1260, 603, h=363),  # 真人，座位 3
    ]
    seats = assign_segment_seats(dets, seg, frame_w=W)
    assert 5 not in seats, "手不应获得座位"
    assert seats == {22: 1, 1: 2, 10: 3}


def test_rejection_reason_is_reported():
    """剔除必须可见：seat_candidates 的第二个返回值须包含 track_id 与可读原因。

    cx 间距调整理由同 `test_hand_sized_box_is_rejected_as_seat_candidate`
    （Task 33 站立旁路的意外重合，见该测试的 docstring）。"""
    seg = _seg(2, 3)
    dets = [
        det(5, 1359, 877, h=186),
        det(22, 1317, 866, h=546),
        det(1, 1400, 715, h=425),
        det(10, 1260, 603, h=363),
    ]
    kept, rejected = seat_candidates(dets, seg, W)
    rejected_ids = {tid for tid, _ in rejected}
    assert 5 in rejected_ids, "手应出现在剔除列表里"
    reason = dict(rejected)[5]
    assert isinstance(reason, str) and len(reason) > 0, "剔除原因必须是非空可读字符串"
    kept_ids = {d.track_id for d in kept}
    assert kept_ids == {22, 1, 10}


def test_zoomed_out_segment_keeps_real_people():
    """整段所有人都只有 300px 高（模拟镜头拉远）：绝对阈值单独看会很接近边界，
    但相对阈值（对本段最高候选的比例）应确保真人不被误杀。
    y2 间距改为 50px（Task 29 起深度断层按「间隔/上位者框高」的比值判定，
    原先 200px 间距对应比值 200/300=0.67，会被 SEAT_DEPTH_GAP_RATIO=0.45
    误判为断层——该间距是早于深度断层规则的随意取值，不代表真实相邻座位
    间距，现改用远低于门槛的间距，只保留本测试原本要验证的相对高度阈值意图）。
    """
    seg = _seg(20, 3)
    dets = [
        det(30, 1300, 900, h=300, order=20),
        det(31, 1320, 850, h=300, order=20),
        det(32, 1340, 800, h=300, order=20),
    ]
    kept, rejected = seat_candidates(dets, seg, W)
    assert rejected == [], f"拉远镜头下真人不应被剔除，实际剔除：{rejected}"
    assert {d.track_id for d in kept} == {30, 31, 32}


def test_pairwise_ranking_survives_missing_track():
    """复刻 order2 真实共现结构：tid1 与 tid10 共现 38 帧，二者与 tid22 额外
    共现 6 帧（tid22 每次都最靠下），另有 2 帧只有 tid1。
    现行的逐帧排序+投票会让 tid22 只得 6 票、被匈牙利求总票数最优时判零票丢弃。
    成对比较应正确排出 [22, 1, 10]。"""
    dets = []
    for i in range(38):
        dets.append(det(1, 1300, 715, h=425, t=float(i)))
        dets.append(det(10, 1310, 603, h=363, t=float(i)))
    for i in range(38, 44):
        dets.append(det(1, 1300, 715, h=425, t=float(i)))
        dets.append(det(10, 1310, 603, h=363, t=float(i)))
        dets.append(det(22, 1320, 866, h=546, t=float(i)))
    for i in range(44, 46):
        dets.append(det(1, 1300, 715, h=425, t=float(i)))
    order = rank_tracks_pairwise(dets)
    assert order == [22, 1, 10]


def test_zero_win_track_is_still_ranked_last():
    """三条轨迹全程共现，其中一条成对比较全败（0 胜场）。
    用裸 Counter 收集胜场时，0 胜场的轨迹根本不会出现在字典里而被静默丢弃——
    它必须仍然出现在结果里，且排在最后。"""
    dets = []
    for t in range(5):
        dets.append(det(70, 1300, 900, t=float(t)))  # 全胜
        dets.append(det(71, 1310, 700, t=float(t)))  # 1 胜 1 负
        dets.append(det(72, 1320, 500, t=float(t)))  # 全败，0 胜场
    order = rank_tracks_pairwise(dets)
    assert order == [70, 71, 72]
    assert 72 in order, "0 胜场的轨迹不能被静默丢弃"
    assert order[-1] == 72


def test_pairwise_ranking_survives_camera_pan():
    """与现有 test_ranking_survives_camera_pan 同源，但走 rank_tracks_pairwise
    新路径：摄像机平移后，成对相对关系不变，排序结果不变。"""
    base = []
    panned = []
    for t in range(2):
        base += [det(7, 1300, 400, t=float(t)), det(8, 1320, 620, t=float(t)),
                 det(9, 1350, 880, t=float(t))]
        panned += [det(7, 1380, 440, t=float(t)), det(8, 1400, 660, t=float(t)),
                   det(9, 1430, 920, t=float(t))]
    assert rank_tracks_pairwise(base) == rank_tracks_pairwise(panned) == [9, 8, 7]


def test_more_tracks_than_presenters_keeps_the_lowest_n():
    """4 条合格真人轨迹，但 presenter_num=3：只取最靠下的 3 条。
    y2 间距改为 50px（同上，避免触发 Task 29 的深度断层比值规则——
    200px 间距对应比值 200/400=0.5，会被 SEAT_DEPTH_GAP_RATIO=0.45 误判为
    断层，这不是本测试要验证的意图，本测试验证的是 presenter_num 截断）。"""
    seg = _seg(2, 3)
    dets = []
    for t in range(3):
        dets.append(det(1, 1300, 900, h=400, t=float(t)))
        dets.append(det(2, 1310, 850, h=400, t=float(t)))
        dets.append(det(3, 1320, 800, h=400, t=float(t)))
        dets.append(det(4, 1330, 750, h=400, t=float(t)))  # 最靠上，应被排除
    seats = assign_segment_seats(dets, seg, frame_w=W)
    assert seats == {1: 1, 2: 2, 3: 3}
    assert 4 not in seats


def test_fewer_tracks_than_presenters_does_not_invent_seats():
    """复刻 order7：只有 1 条合格轨迹，presenter_num=2。
    只返回 1 个座位，不报错、不凑数。"""
    seg = _seg(7, 2)
    dets = [det(2, 1300, 700, h=392, t=float(t), order=7) for t in range(3)]
    seats = assign_segment_seats(dets, seg, frame_w=W)
    assert seats == {2: 1}
    assert len(seats) == 1


# --- 变异覆盖补充（Step 4 需要能单独触发 b/c/e/f/g 各条变异） ---

def test_track_failing_only_absolute_threshold_is_rejected():
    """候选高度 200px：高于相对阈值（0.45*400=180）但低于绝对阈值 270。
    只有绝对阈值真正起作用时才会剔除它——锁住 SEAT_MIN_H_RATIO 本身。"""
    seg = _seg(3, 2)
    dets = [
        det(40, 1300, 900, h=400, order=3),  # 本段最高，肯定保留
        det(41, 1310, 500, h=200, order=3),  # 只失守绝对阈值
    ]
    kept, rejected = seat_candidates(dets, seg, W)
    assert {tid for tid, _ in rejected} == {41}
    assert {d.track_id for d in kept} == {40}


def test_track_failing_only_relative_threshold_is_rejected():
    """候选高度 290px：高于绝对阈值 270，但低于相对阈值（0.45*700=315）。
    只有相对阈值真正起作用时才会剔除它——锁住 SEAT_MIN_H_VS_TALLEST 本身。"""
    seg = _seg(4, 2)
    dets = [
        det(50, 1300, 900, h=700, order=4),  # 本段最高
        det(51, 1310, 500, h=290, order=4),  # 只失守相对阈值
    ]
    kept, rejected = seat_candidates(dets, seg, W)
    assert {tid for tid, _ in rejected} == {51}
    assert {d.track_id for d in kept} == {50}


def test_pairwise_uses_frame_level_cooccurrence_not_aggregate():
    """P、Q 只在 5 帧里真正共现（P 每次都更靠下）；此外 P 另有 20 帧独自出现、
    位置偏上，Q 另有 20 帧独自出现、位置偏下——这些不共现的帧会把两条轨迹
    「整体」的中位数拉到相反的方向。只统计真正共现帧的成对比较必须不受这些
    不共现帧干扰，排序仍是 [P, Q]。"""
    dets = []
    for t in range(5):
        dets.append(det(201, 1300, 700, t=float(t)))
        dets.append(det(202, 1310, 400, t=float(t)))
    for t in range(5, 25):
        dets.append(det(201, 1300, 500, t=float(t)))
    for t in range(25, 45):
        dets.append(det(202, 1310, 950, t=float(t)))
    assert rank_tracks_pairwise(dets) == [201, 202]


def test_tie_break_uses_bottom_y_median_when_copeland_ties():
    """两条轨迹从未在同一帧共现，Copeland 胜场因此都是 0（平局）。
    平局须用各自 bottom_y 中位数做二级键，保证可复现的排序，
    而不是任由 Counter/字典的迭代顺序决定。"""
    dets = []
    for t in range(5):
        dets.append(det(99, 1300, 900, t=float(t)))
    for t in range(5, 10):
        dets.append(det(10, 1310, 300, t=float(t)))
    assert rank_tracks_pairwise(dets) == [99, 10]


# ---------------------------------------------------------------------------
# Task 29：用户逐个企业核对产物后报错，控制器据此实测出三条几何信号——
# 碎片占座（覆盖率过滤）、重复检测（去重）、专家在更远处（深度断层截断）。
# 深度断层与去重的判据在开发过程中经用户质疑后改为相对比例（框高/框宽的
# 比值），而非绝对像素——绝对阈值不随镜头拉近拉远缩放，换机位/变焦后
# 同一判据可能失效。
# ---------------------------------------------------------------------------

def test_duplicate_box_is_merged_keeping_nearer():
    """order1/4/9/12 型重复：两条轨迹几何上是同一人（小框被包含比例
    0.92，中心 x 相差/框宽比值 0.083，均越过 SEAT_DUP_CONTAIN/SEAT_DUP_CX_RATIO）。
    track_id=101 覆盖率更高（100% vs 80%），但去重必须保留更靠近镜头
    （y2 更大）的 100——覆盖率不是去重判据，若按覆盖率保留，会在 order 13
    丢掉正主（见 config.py 里 SEAT_DUP_CONTAIN 上方的记录）。"""
    seg = _seg(30, 1)
    dets = []
    for t in range(8):
        dets.append(det(100, 1300, 880, h=550, t=float(t), order=30))
    for t in range(10):
        dets.append(det(101, 1310, 860, h=530, t=float(t), order=30))
    seats = assign_segment_seats(dets, seg, frame_w=W)
    assert seats == {100: 1}, f"应保留更靠近镜头的 track_id=100，实得 {seats}"
    assert 101 not in seats


def test_boxes_side_by_side_are_not_merged():
    """两个真人并排而坐（cx 差 150px，框宽 120px，比值 1.25 远超 0.25），
    包含比无论多高都不应被判为重复——防止把两个真人当成同一人的重复框。"""
    a = det(1, 1300, 800, h=400)
    b = det(2, 1450, 800, h=400)
    assert not is_duplicate_box(a, b)


def test_dup_cx_threshold_is_relative_to_box_width():
    """去重的 cx 判据必须相对框宽而非绝对像素——用户对第一版方案（cx 差
    <30px 判重复）提出正确质疑：「0.86 一刀切在其他组可能不适用吧」。

    容器框 A 足够大，把每个候选框 B 的横向、纵向范围都整个罩住，使包含比
    恒为 1.0（分离掉包含比这个变量），只让「cx 相差 / 候选框宽度」的比值
    决定结果：
      - 窄框（宽 100px）位移 20px：比值 0.20 < 0.25 → 判重复
      - 窄框（宽 100px）位移 80px：比值 0.80 > 0.25 → 不判重复
      - 宽框（宽 400px）位移 30px：比值 0.075 < 0.25 → 判重复
    最后一条与第一条对照：绝对位移更大（30>20）反而判重复，证明判据是
    相对框宽的比值，不是绝对像素——若实现退回绝对阈值（旧版 30px），
    30px 位移恰好落在旧阈值边界上不算重复，这条断言会翻红。
    """
    def make(x1, y1, x2, y2, tid):
        return Detection(9, "seated", 0.0, Path("f.jpg"), tid, x1, y1, x2, y2, 0.9)

    container = make(450, 200, 1950, 1000, 1)   # 宽 1500，cx=1200
    narrow_20 = make(1170, 400, 1270, 700, 2)   # 宽 100，cx=1220，diff=20
    narrow_80 = make(1230, 400, 1330, 700, 3)   # 宽 100，cx=1280，diff=80
    wide_30 = make(1030, 400, 1430, 700, 4)     # 宽 400，cx=1230，diff=30

    assert is_duplicate_box(container, narrow_20), "窄框位移20px（比值0.20）应判重复"
    assert not is_duplicate_box(container, narrow_80), "窄框位移80px（比值0.80）不应判重复"
    assert is_duplicate_box(container, wide_30), (
        "宽框位移30px（比值0.075）应判重复——绝对位移比 narrow_20 更大却仍判重复，"
        "证明判据是相对框宽的比值"
    )


def test_low_coverage_track_does_not_take_a_seat():
    """一条 5% 覆盖率的碎片（只出现 1 帧，且该帧 y2 比真人更靠下）与一条
    95% 覆盖率的真人竞争同一座位。若不做覆盖率过滤，碎片会因为二者从不
    共现、Copeland 打平后按 y2 中位数胜出而抢到座位——覆盖率过滤必须先把
    它挡在候选之外。"""
    seg = _seg(31, 1)
    dets = [det(200, 1300, 700, h=400, t=float(t), order=31) for t in range(19)]
    dets.append(det(201, 1000, 900, h=400, t=19.0, order=31))  # 碎片：只出现 1 帧
    seats = assign_segment_seats(dets, seg, frame_w=W)
    assert seats == {200: 1}, f"碎片不应占座，实得 {seats}"
    assert 201 not in seats


def test_depth_gap_is_explicit_noop():
    """Task 40：`cut_at_depth_gap` 不再依赖任何魔数（config 里已删除
    `SEAT_DEPTH_GAP_RATIO`），恒不截断——无论间隙多悬殊、候选多少条，
    都原样返回输入。用多组不同规模的悬殊间隙数据锁住"恒不截断"这条行为，
    不是只测"不崩"。"""
    # (a) 原先会被判定为"断层"的悬殊间隙（比值 ~1.08，远超旧默认 0.52）
    candidates_a = [(1, 1000.0, 400.0), (2, 960.0, 400.0), (3, 530.0, 400.0)]
    assert cut_at_depth_gap(candidates_a) == candidates_a

    # (b) 更悬殊的间隙（比值 ~49），仍然不截断
    candidates_b = [(1, 5000.0, 100.0), (2, 4900.0, 100.0), (3, 0.0, 100.0)]
    assert cut_at_depth_gap(candidates_b) == candidates_b

    # (b2) 极端悬殊（比值 1e6）——不给"改回依赖某个不太离谱的魔数"留任何
    # 生存空间：只要实现里还残留任何形式的阈值比较，这一条必翻红。
    candidates_b2 = [(1, 1_000_001.0, 1.0), (2, 1_000_000.0, 1.0), (3, 0.0, 1.0)]
    assert cut_at_depth_gap(candidates_b2) == candidates_b2

    # (c) 均匀间隙，本就不该截断——回归意义上的一致性检查
    candidates_c = [(1, 1000.0, 400.0), (2, 900.0, 400.0), (3, 800.0, 400.0)]
    assert cut_at_depth_gap(candidates_c) == candidates_c

    # (d) 空输入不崩
    assert cut_at_depth_gap([]) == []

    # (e) 单个候选
    assert cut_at_depth_gap([(9, 100.0, 50.0)]) == [(9, 100.0, 50.0)]


def test_depth_gap_no_longer_excludes_the_deepest_expert():
    """机制停用后，`assign_segment_seats` 不应再因为深度断层把候选踢出——
    专家保留下来，是否入座位表由 presenter_num 截断/rank_tracks_pairwise
    决定，不是由这条已停用的机制决定。用同一份换算尺度（0.5 倍）验证两种
    尺度下结果一致，锁住"这条机制的缺席"这件事本身在换镜头后也一致。"""
    seg = _seg(35, 3)

    def build(scale):
        specs = [(1, 1300, 1000, 400), (2, 1310, 900, 400), (3, 1320, 500, 400)]
        dets = []
        for tid, cx, bottom, h in specs:
            for t in range(3):
                dets.append(det(tid, cx * scale, bottom * scale, h=h * scale,
                                 t=float(t), order=35))
        return dets

    full = assign_segment_seats(build(1.0), seg, frame_w=W)
    half = assign_segment_seats(build(0.5), seg, frame_w=int(W * 0.5))

    assert set(full.keys()) == set(half.keys()) == {1, 2, 3}, (
        f"深度断层机制已停用，presenter_num=3 应取满 3 个候选，"
        f"实得 full={full} half={half}"
    )


def test_no_gap_means_all_kept():
    """相邻比值均为 100/400=0.25，三个候选都应保留——不依赖具体阈值，
    停用后的空操作下天然成立。"""
    candidates = [(1, 1000.0, 400.0), (2, 900.0, 400.0), (3, 800.0, 400.0)]
    result = cut_at_depth_gap(candidates)
    assert result == candidates


def test_rejection_reasons_are_reported():
    """seat_candidates 的剔除列表必须包含 Task 29 新增两类仍在生效的原因的
    可读标记：低覆盖率、重复检测——画廊与报告要能看到，不能只是内部丢弃。

    （深度断层这一类 Task 40 起已随 `cut_at_depth_gap` 变为显式空操作，
    不再可能被触发，故此处不再测试第三类；见
    `test_depth_gap_no_longer_excludes_the_deepest_expert`。）
    """
    # (i) 低覆盖率
    seg1 = _seg(32, 1)
    dets1 = [det(300, 1300, 700, h=400, t=float(t), order=32) for t in range(19)]
    dets1.append(det(301, 1000, 900, h=400, t=19.0, order=32))
    _kept1, rejected1 = seat_candidates(dets1, seg1, W)
    reason1 = dict(rejected1)[301]
    assert "低覆盖率" in reason1, reason1

    # (ii) 重复检测
    seg2 = _seg(33, 1)
    dets2 = []
    for t in range(8):
        dets2.append(det(302, 1300, 880, h=550, t=float(t), order=33))
    for t in range(10):
        dets2.append(det(303, 1310, 860, h=530, t=float(t), order=33))
    _kept2, rejected2 = seat_candidates(dets2, seg2, W)
    reason2 = dict(rejected2)[303]
    assert "重复检测" in reason2, reason2


def test_depth_gap_report_lists_all_adjacent_pairs():
    """n 个候选产出 n-1 条相邻对记录，每条含 ratio 与 cut。

    Task 40 起 `cut` 直接由 `cut_at_depth_gap`（现为显式空操作）的存活集合
    推导，不再和某个 config 阈值比较——`cut_at_depth_gap` 恒不截断，故
    `cut` 恒为 False，即使 ratio 本身很悬殊（第 2、3 条间比值高达 1.0）。
    这正是"机制已停用"该有的样子：报告如实反映当前行为，不是继续假装
    有个阈值在生效。"""
    seg = _seg(36, 3)
    dets = [
        det(1, 1300, 1000, h=400, order=36),
        det(2, 1310, 900, h=400, order=36),
        det(3, 1320, 500, h=400, order=36),
    ]
    report = depth_gap_report(dets, [seg], frame_w=W)
    assert len(report) == 2, f"3 个候选应产出 2 条相邻对记录，实得 {len(report)}"
    by_pair = {(r["upper_tid"], r["lower_tid"]): r for r in report}
    assert set(by_pair) == {(1, 2), (2, 3)}
    assert by_pair[(1, 2)]["cut"] is False
    assert by_pair[(2, 3)]["cut"] is False, "cut_at_depth_gap 现为显式空操作，恒不截断"
    assert by_pair[(2, 3)]["ratio"] == pytest.approx(1.0)
    assert all(r["seg_order"] == 36 for r in report)


def test_depth_gap_report_marks_threshold_neighbours():
    """标定报告的核心用途：给出阈值两侧最近的两个 ratio 值，让人一眼看出
    阈值是落在数据的裂谷里，还是密集处——换一组视频后不用等全流程跑完、
    人工核对完才发现阈值需要重标定。"""
    ratios = [0.10, 0.20, 0.30, 0.50, 0.60]
    below, above = ratios_around_threshold(ratios, threshold=0.45)
    assert below == 0.30
    assert above == 0.50


def test_ratios_around_threshold_handles_missing_side():
    """阈值一侧没有数据时返回 None，而不是报错或悄悄返回错误的值。"""
    ratios = [0.10, 0.20]
    below, above = ratios_around_threshold(ratios, threshold=0.45)
    assert below == 0.20
    assert above is None


# ---------------------------------------------------------------------------
# Task 31：Task 29 的三条规则顺序有误——覆盖率过滤跑在「同一个人被跟踪器拆成
# 多条轨迹」的合并之前，把碎片一条条砍掉（样本 27→18/31）。merge_tracks_by_
# position 补上合并这一步，且必须在覆盖率过滤之前跑（实测顺序一改 18→28/31）。
# 深度断层的段内相邻比值判据也退役，改用跨段位置簇 presenter_depth_floor。
# ---------------------------------------------------------------------------

def test_fragments_at_same_position_merge_into_one_person():
    """order11 型场景缩影：4 条轨迹 y2 相差 <30px、彼此从不共现（跟踪器把
    同一个人的在场时间拆成 4 段），各占 5 帧（总 100 帧场景下各 5%）。
    应合并成 1 个人，n_frames=20（互不重叠的并集），覆盖率≈20%。"""
    tid_y2 = {1: 1080.0, 2: 1068.0, 3: 1053.0, 4: 1070.0}
    dets = []
    for i, (tid, y2) in enumerate(tid_y2.items()):
        start = i * 25   # 4 段互不重叠：[0,5) [25,30) [50,55) [75,80)
        for t in range(start, start + 5):
            dets.append(det(tid, 1300, y2, t=float(t)))

    people = merge_tracks_by_position(dets)

    assert len(people) == 1, f"4 段应合并成 1 个人，实得 {len(people)} 个"
    person = people[0]
    assert set(person["track_ids"]) == {1, 2, 3, 4}
    assert person["n_frames"] == 20
    assert abs(person["n_frames"] / 100 - 0.20) < 1e-9


def test_merge_happens_before_coverage_filter():
    """本条锁住顺序，是 Task 31 的核心：4 段碎片各占 6% 帧（<20% 门槛），
    单独看都会被覆盖率规则剔除；合并后并集覆盖率 24%（≥20%），不应被剔除。
    另有一条不同位置的轨迹占满全部 100 帧，确保 total_frames=100。"""
    seg = _seg(40, 2)
    dets = []
    # person B：占满全部 100 帧，位置远离 A（cx 差 300px），确保总帧数=100
    for t in range(100):
        dets.append(det(900, 1000, 500, h=400, t=float(t), order=40))
    # person A：4 段碎片，互不共现，各占 6 帧（6%），合并后 24%
    frag_y2 = {901: 1080.0, 902: 1068.0, 903: 1053.0, 904: 1070.0}
    for i, (tid, y2) in enumerate(frag_y2.items()):
        start = i * 25
        for t in range(start, start + 6):
            dets.append(det(tid, 1300, y2, h=400, t=float(t), order=40))

    kept, rejected = seat_candidates(dets, seg, W)

    rejected_ids = {tid for tid, _ in rejected}
    assert not ({901, 902, 903, 904} & rejected_ids), (
        f"合并后覆盖率≈24%（≥门槛 20%），不应被覆盖率规则剔除，实际剔除：{rejected}"
    )
    assert any(abs(d.y2 - 1070.0) < 30 for d in kept), "合并后的人应出现在候选里"


def test_distinct_people_are_not_merged():
    """两条轨迹 y2 相差 200px（远超 80px 容差），即便互不共现也不应合并——
    防止把纵深上明显不同的两个人错认成同一人的碎片。"""
    dets = []
    for t in range(5):
        dets.append(det(1, 1300, 1000, t=float(t)))
    for t in range(5, 10):
        dets.append(det(2, 1300, 800, t=float(t)))

    people = merge_tracks_by_position(dets)

    assert len(people) == 2, f"y2 差 200px 不应合并，实得 {len(people)} 个人"


def test_merged_person_uses_longest_track_as_key():
    """tid=10 只有 5 帧，tid=11 有 15 帧，同位置、互不共现，应合并成 1 人，
    代表 track_id 必须是帧数更多的 11——下游按 track_id 取帧才能拿到最长的
    那一段，而不是碰巧先被遍历到的那条。"""
    seg = _seg(41, 1)
    dets = []
    for t in range(5):
        dets.append(det(10, 1300, 900, h=400, t=float(t), order=41))
    for t in range(5, 20):
        dets.append(det(11, 1300, 900, h=400, t=float(t), order=41))

    seats = assign_segment_seats(dets, seg, frame_w=W)

    assert seats == {11: 1}, f"代表轨迹应为帧数更多的 tid=11，实得 {seats}"
    assert 10 not in seats


def test_merge_tolerance_is_configurable():
    """两条轨迹 y2 相差 100px：默认容差 80px 不应合并；把容差放宽到 120px
    后应合并——直接测参数与 config 默认值两条路径。"""
    dets = []
    for t in range(5):
        dets.append(det(1, 1300, 1000, t=float(t)))
    for t in range(5, 10):
        dets.append(det(2, 1300, 900, t=float(t)))

    tight = merge_tracks_by_position(dets, cx_tol=80.0, y2_tol=80.0)
    assert len(tight) == 2, "y2 差 100px 超过 80px 容差，不应合并"

    loose = merge_tracks_by_position(dets, cx_tol=80.0, y2_tol=120.0)
    assert len(loose) == 1, "把容差放宽到 120px 后应合并"

    original = config.SEAT_MERGE_Y2_TOL
    try:
        config.SEAT_MERGE_Y2_TOL = 120.0
        via_config = merge_tracks_by_position(dets)
        assert len(via_config) == 1, "不传参数时应读取 config.SEAT_MERGE_Y2_TOL"
    finally:
        config.SEAT_MERGE_Y2_TOL = original


def test_presenter_depth_floor_separates_clusters():
    """3 段各有创业者候选（y2≈1000）与专家候选（y2≈500），presenter_num=1
    使专家候选排名靠后、从不入选。断言返回的下界落在两个位置带之间
    （>专家 y2≈500，≤创业者 y2≈1000+）。"""
    segs = [_seg(50 + i, 1) for i in range(3)]
    dets = []
    for i, seg in enumerate(segs):
        dets.append(det(1000 + i, 1300, 1000 + i * 2, h=400, order=seg.order))
        dets.append(det(2000 + i, 1310, 500 + i * 2, h=400, order=seg.order))

    floor = presenter_depth_floor(dets, segs, frame_w=W)

    assert 500 < floor <= 1010, (
        f"下界应落在专家簇(≈500)与创业者簇(≈1000)之间，实得 {floor}"
    )


def test_presenter_depth_floor_does_not_depend_on_selection_rate():
    """下界必须由位置簇的几何结构决定，不能依赖「入选比例」。

    依赖入选比例是循环的：入选比例取决于当前的深度过滤，而深度过滤又要
    用这个下界。实测后果——`SEAT_DEPTH_GAP_RATIO` 停用后入选比例全变，
    本函数随即返回 0（不过滤），用户确认的 2 位专家又回到了座位表里。

    本场景把 presenter_num 设得足够大，使**全部**候选（含深处的专家簇）
    都「入选」，入选比例一律 100%。若实现仍看入选比例，就找不到任何
    「低入选簇」而返回 0；改用「找位置簇分布中最大的那道缝」则不受影响。
    """
    segs = [_seg(70 + i, 4) for i in range(3)]          # presenter_num=4，人人入选
    dets = []
    for i, seg in enumerate(segs):
        dets.append(det(5000 + i, 1300, 1000 + i * 2, h=400, order=seg.order))
        dets.append(det(6000 + i, 1305, 950 + i * 2, h=400, order=seg.order))
        dets.append(det(7000 + i, 1310, 500 + i * 2, h=400, order=seg.order))

    floor = presenter_depth_floor(dets, segs, frame_w=W)

    assert 510 < floor < 950, (
        f"下界应落在 y2≈950 的创业者簇与 y2≈500 的专家簇之间（最大间隔处），"
        f"实得 {floor}"
    )


def test_presenter_depth_floor_returns_zero_when_no_expert_cluster():
    """4 段都有创业者候选（presenter_num=1，全部入选）；另有 1 段额外出现
    一个从未入选的低位候选（y2≈500）——但它只出现 1 段，达不到「出现段数
    ≥3」的门槛，不能被当成稳定的专家位置带（否则任何一段的偶然噪声都会
    触发过滤）。断言下界仍为 0（不过滤）。"""
    segs = [_seg(60 + i, 1) for i in range(4)]
    dets = [det(3000 + i, 1300, 1000 + i * 2, h=400, order=seg.order)
            for i, seg in enumerate(segs)]
    dets.append(det(4000, 1310, 500, h=400, order=segs[0].order))

    floor = presenter_depth_floor(dets, segs, frame_w=W)

    assert floor == 0.0, (
        f"低位候选只出现 1 段，未达『出现段数≥3』门槛，不应触发过滤，实得 {floor}"
    )


def test_detailed_assignment_exposes_merged_track_ids():
    """assign_segment_seats_detailed 必须能给出每个座位对应的全部原始
    track_id（合并进来的碎片），供 Task 15 装配阶段取全部帧用。"""
    seg = _seg(42, 1)
    dets = []
    for t in range(5):
        dets.append(det(20, 1300, 900, h=400, t=float(t), order=42))
    for t in range(5, 20):
        dets.append(det(21, 1300, 900, h=400, t=float(t), order=42))

    detailed = assign_segment_seats_detailed(dets, seg, frame_w=W)

    assert detailed[1]["track_id"] == 21
    assert set(detailed[1]["track_ids"]) == {20, 21}


# ---------------------------------------------------------------------------
# Task 33 · 改动二 —— 重复检测漏掉站立/坐姿同一人（is_duplicate_box 站立旁路）
# ---------------------------------------------------------------------------

def test_standing_and_seated_boxes_of_same_person_merge():
    """复刻 order 14 的真实坐标：座位1是坐姿框，座位2是站姿全身框，是
    同一个人。包含比只有 0.870（差阈值 0.90 仅 0.03），现行实现（只看
    包含比路径）下必红；站立旁路应把它们判为同一人。"""
    seat1_seated = Detection(14, "seated", 0.0, Path("f.jpg"), 11,
                             1046, 551, 1403, 1079, 0.9)   # 宽356 高528 坐姿
    seat2_standing = Detection(14, "seated", 0.0, Path("f.jpg"), 4,
                               1041, 214, 1363, 1069, 0.9)  # 宽322 高855 站姿
    assert is_duplicate_box(seat1_seated, seat2_standing), (
        "一坐一站、横向高度重叠严重重叠、cx 相差很小的同一人应被判为重复"
    )


def test_side_by_side_tall_and_short_are_not_merged():
    """一高一矮但横向不重叠（cx 差远超框宽）：不应被站立旁路误合并——
    否则会把两个真人（一个恰好站起来、一个恰好在另一侧坐着）误判为同一人。"""
    tall = Detection(14, "seated", 0.0, Path("f.jpg"), 100,
                     300, 100, 620, 950, 0.9)    # 宽320 高850（类似站姿高度）
    short = Detection(14, "seated", 0.0, Path("f.jpg"), 101,
                      1400, 500, 1720, 950, 0.9)  # 宽320 高450（坐姿），cx 差极大
    assert not is_duplicate_box(tall, short), (
        "横向完全不重叠时，哪怕高度比很悬殊也不应判为同一人"
    )


def test_standing_bypass_requires_x_overlap_not_just_height_ratio():
    """护栏：cx 比值与横向重叠比例并非同一件事——等宽两框、cx 差刚好卡在
    SEAT_DUP_CX_RATIO(0.25) 门槛之下时，横向重叠比例可以低到 0.775
    （<SEAT_DUP_X_OVERLAP=0.80）：
        A：x=(1200,1400) 宽200；B：x=(1245,1445) 宽200，cx 差45
        cx_ratio = 45/200 = 0.225 < 0.25（过）
        重叠 = min(1400,1445)-max(1200,1245) = 155，155/200 = 0.775 < 0.80（不过）
    配合高度比 1.5（≥1.4）与包含比 0.775（<0.90，包含比路径也不该触发），
    若把横向重叠检查去掉、只靠 cx 比值 + 高度比判定，会把这两个真实存在
    横向偏移、并非一坐一站的人误判为同一人——横向重叠必须是独立于 cx 比值
    的必要条件。"""
    a = Detection(9, "seated", 0.0, Path("f.jpg"), 1, 1200, 400, 1400, 700, 0.9)  # 宽200 高300 坐姿
    b = Detection(9, "seated", 0.0, Path("f.jpg"), 2, 1245, 300, 1445, 750, 0.9)  # 宽200 高450 站姿
    assert not is_duplicate_box(a, b), (
        "cx 比值达标不代表横向重叠达标，二者是独立判据，不应被误合并"
    )


# ---------------------------------------------------------------------------
# Task 33 · 改动一 —— assign_segment_seats(gender_match=True) 接入
# pipeline/seatmatch.match_seats_by_gender（额外集成覆盖，核心逻辑测试见
# tests/test_seatmatch.py）
# ---------------------------------------------------------------------------

def test_gender_match_default_off_keeps_old_behaviour():
    seg = Segment(14, "seg14", 0, 100, 3, (
        Presenter(1, 0, 0, 1), Presenter(2, 1, 0, 1), Presenter(3, 0, 0, 1),
    ))
    dets = []
    for t in range(5):
        dets += [
            det(1, 1300, 900, h=400, t=float(t), order=14),
            det(2, 1300, 700, h=400, t=float(t), order=14),
            det(3, 1300, 500, h=400, t=float(t), order=14),
        ]
    seats_off = assign_segment_seats(dets, seg, frame_w=W)
    seats_default = assign_segment_seats(dets, seg, frame_w=W, gender_match=False)
    assert seats_off == seats_default == {1: 1, 2: 2, 3: 3}


def test_gender_match_on_rearranges_using_vlm_gender():
    """order 14 型场景：几何排序把座位 2/3 的性别排颠倒了，
    gender_judged_by_track 提供 VLM 判定后，gender_match=True 应重排回正确座位。"""
    seg = Segment(14, "seg14", 0, 100, 3, (
        Presenter(1, 0, 0, 1), Presenter(2, 1, 0, 1), Presenter(3, 0, 0, 1),
    ))
    dets = []
    for t in range(5):
        dets += [
            det(1, 1300, 900, h=400, t=float(t), order=14),  # 基线座位1，标注男
            det(2, 1300, 700, h=400, t=float(t), order=14),  # 基线座位2，标注女，VLM判男
            det(3, 1300, 500, h=400, t=float(t), order=14),  # 基线座位3，标注男，VLM判女
        ]
    gjt = {1: "男", 2: "男", 3: "女"}
    seats = assign_segment_seats(
        dets, seg, frame_w=W, gender_match=True, gender_judged_by_track=gjt
    )
    assert seats == {1: 1, 3: 2, 2: 3}, f"应按 VLM 性别重排座位2/3，实得 {seats}"


def test_gender_match_without_vlm_data_is_inert():
    """重跑前（annotations.json 还没有任何 gender_judged）：
    gender_judged_by_track=None 时，gender_match=True 必须与关闭时结果一致——
    全员『不确定』通配符，不会凭空重排。"""
    seg = Segment(14, "seg14", 0, 100, 3, (
        Presenter(1, 0, 0, 1), Presenter(2, 1, 0, 1), Presenter(3, 0, 0, 1),
    ))
    dets = []
    for t in range(5):
        dets += [
            det(1, 1300, 900, h=400, t=float(t), order=14),
            det(2, 1300, 700, h=400, t=float(t), order=14),
            det(3, 1300, 500, h=400, t=float(t), order=14),
        ]
    seats_on_no_data = assign_segment_seats(dets, seg, frame_w=W, gender_match=True)
    seats_off = assign_segment_seats(dets, seg, frame_w=W, gender_match=False)
    assert seats_on_no_data == seats_off


def test_gender_match_candidates_exposes_y2_and_judged_gender():
    seg = _seg(14, 2)
    dets = []
    for t in range(5):
        dets += [
            det(1, 1300, 900, h=400, t=float(t), order=14),
            det(2, 1300, 700, h=400, t=float(t), order=14),
        ]
    people = gender_match_candidates(dets, seg, frame_w=W, gender_judged_by_track={1: "男"})
    by_tid = {p["track_id"]: p for p in people}
    assert by_tid[1]["gender_judged"] == "男"
    assert by_tid[2]["gender_judged"] == "不确定", "查不到的 track 应记不确定通配符"
    assert by_tid[1]["y2"] == 900.0
    assert by_tid[2]["y2"] == 700.0


# --- 按段自适应放宽（用户 2026-08-17）---

def _mk_dets(seg_order, specs, n_frames=10):
    """specs: [(track_id, height_px, n_present_frames)]，造在右区的就座检测。"""
    out = []
    for tid, h, npres in specs:
        for i in range(npres):
            y2 = 900.0
            out.append(Detection(seg_order, "seated", float(i),
                                 Path(f"/f/{seg_order}/{i}.jpg"), tid,
                                 1000.0 + tid * 120, y2 - h, 1090.0 + tid * 120, y2, 0.9))
    # 补足总帧数：放一个必定通过的高个子，撑起 total_frames
    for i in range(n_frames):
        out.append(Detection(seg_order, "seated", float(i),
                             Path(f"/f/{seg_order}/{i}.jpg"), 99,
                             880.0, 300.0, 970.0, 900.0, 0.9))
    return out


def test_thresholds_relax_until_presenter_num_is_reachable():
    """候选不足 presenter_num 时，阈值应逐步放宽，而不是硬砍到不够。

    实测三组：右簇本有 187 个 track（足够覆盖 103 位创业者），被
    「高度>25%画面」砍到 129、再被「在场率>30%」砍到 90，最终 74/103。
    而这两个阈值切在**连续分布的密集区**（高度 0.20-0.25 有 13 个、
    0.25-0.30 有 27 个），不像右区阈值那样落在 540px 宽的真空里——
    也就是说没有任何有依据的位置可放。

    改为按段自适应：先用常规阈值，若存活数少于该段 presenter_num，
    就逐档放宽直到够用或触底。用的是用户的人工标注人数，不是拍脑袋常数。
    """
    from pipeline.seats import assign_segment_seats
    seg = Segment(3, "某公司", 0, 900, 3,
                  tuple(Presenter(i, 0, 1 if i == 1 else 0, 0) for i in (1, 2, 3)))
    # 三个人：一个明显够大，两个偏矮/偏少帧（现行阈值下会被砍）
    dets = _mk_dets(3, [(1, 560, 10), (2, 240, 10), (3, 300, 3)])
    seats = assign_segment_seats(dets, seg, frame_w=1440)
    assert len(seats) >= 3, f"应放宽到够 3 人，实得 {len(seats)}：{seats}"


def test_relaxation_stops_at_presenter_num_not_admit_everything():
    """够用即止：候选充足时不得因为"能放宽"就把碎片一起放进来。"""
    from pipeline.seats import assign_segment_seats
    seg = Segment(4, "某公司", 0, 900, 2,
                  tuple(Presenter(i, 0, 1 if i == 1 else 0, 0) for i in (1, 2)))
    # 两个正常的人 + 三个明显是手/碎片的小框
    dets = _mk_dets(4, [(1, 560, 10), (2, 520, 10),
                        (3, 60, 10), (4, 55, 10), (5, 50, 10)])
    seats = assign_segment_seats(dets, seg, frame_w=1440)
    assert len(seats) == 2, f"常规阈值已够 2 人，不该放宽，实得 {len(seats)}"


def test_relaxation_has_a_floor_and_does_not_admit_hands():
    """放宽有下限：即便放宽到底也不该把 60px 的手当成人。

    Ruling R47 的教训——order02_seat1 曾被一双手占座（186px vs 真人
    363-588px）。自适应放宽不能把这条防线一起放掉。
    """
    from pipeline.seats import assign_segment_seats
    seg = Segment(5, "某公司", 0, 900, 4,
                  tuple(Presenter(i, 0, 1 if i == 1 else 0, 0) for i in (1, 2, 3, 4)))
    # 只有 1 个真人，其余全是手；需要 4 人也不许把手算进来
    dets = _mk_dets(5, [(1, 560, 10), (2, 60, 10), (3, 55, 10), (4, 50, 10)])
    seats = assign_segment_seats(dets, seg, frame_w=1440)
    assert len(seats) < 4, "宁可不足，也不能把手算作人"


def test_partial_detection_is_admitted_not_discarded():
    """只拍到手/局部的检测，是「被遮住的人」而非「那里没人」（用户 2026-08-17）。

    Ruling R47 当年把 order02_seat1 的一双手判为"占座"并排除，推理是
    "手不是人"。这个推理错了：手是人存在的证据，只拍到手说明那人被前排
    遮住了。排除它有两重代价——丢掉一个真实样本，更糟的是**座位号会被
    后面的人顶上，导致整段座位错位**，那比少一行数据严重得多。
    """
    from pipeline.seats import assign_segment_seats
    seg = Segment(6, "某公司", 0, 900, 3,
                  tuple(Presenter(i, 0, 1 if i == 1 else 0, 0) for i in (1, 2, 3)))
    # 一个正常的人 + 一个只有 95px 的局部检测（原下限 0.10*1080=108px 会砍掉）
    dets = _mk_dets(6, [(1, 560, 10), (2, 95, 10)])
    seats = assign_segment_seats(dets, seg, frame_w=1440)
    assert 2 in seats, f"95px 的局部检测应占位而非被丢弃，实得 {seats}"


def test_partial_detection_is_flagged_so_it_is_not_silently_coded():
    """局部检测必须带标记——不标就等于把"测不了"伪装成"测到了"。"""
    from pipeline.seats import assign_segment_seats, partial_detections
    # 需 3 人：正常的 track 1 + 撑帧数的 track 99 只够 2 个，必须放宽到
    # 收下那个 95px 的局部检测才够——这才真正走到被测路径
    seg = Segment(7, "某公司", 0, 900, 3,
                  tuple(Presenter(i, 0, 1 if i == 1 else 0, 0) for i in (1, 2, 3)))
    dets = _mk_dets(7, [(1, 560, 10), (2, 95, 10)])
    seats = assign_segment_seats(dets, seg, frame_w=1440)
    flags = partial_detections(dets, seg, frame_w=1440)
    assert any(flags.get(t) for t in seats), "那个 95px 的应被标为局部检测"


def test_full_size_person_is_not_flagged_partial():
    """正常尺寸的人不得被误标为局部——否则标记失去意义。"""
    from pipeline.seats import assign_segment_seats, partial_detections
    seg = Segment(8, "某公司", 0, 900, 2,
                  tuple(Presenter(i, 0, 1 if i == 1 else 0, 0) for i in (1, 2)))
    dets = _mk_dets(8, [(1, 560, 10), (2, 520, 10)])
    seats = assign_segment_seats(dets, seg, frame_w=1440)
    flags = partial_detections(dets, seg, frame_w=1440)
    assert not any(flags.get(t) for t in seats), f"正常尺寸不该标局部：{flags}"


# --- Innermost_expert 剔除（用户 2026-08-17 提供的分组变量）---

def _mk_dets_depth(seg_order, specs, n_frames=10):
    """specs: [(track_id, height_px, y2)]——y2 各不相同，才能区分谁在最里面。

    `_mk_dets` 把所有框的 y2 固定为 900，那是为了测高度/覆盖率相关逻辑；
    测「最里侧」必须让 y2 有区分度，否则 min() 取到的是任意一个。

    也不放 `_mk_dets` 那个撑帧数的填充 track——它的 y2=900 会与真实候选
    的 y2 撞车，把本该最靠里的那个挤出前 N，于是剔除逻辑作用在一个根本
    没入选的 track 上、看起来毫无效果。各 track 共用同一批帧路径，
    total_frames 由它们自身的并集得出，不需要额外填充。
    """
    out = []
    for tid, h, y2 in specs:
        for i in range(n_frames):
            out.append(Detection(seg_order, "seated", float(i),
                                 Path(f"/f/{seg_order}/{i}.jpg"), tid,
                                 1000.0 + tid * 120, y2 - h, 1090.0 + tid * 120, y2, 0.9))
    return out


def _seg_expert(order, n, expert):
    return Segment(order, "某公司", 0, 900, n,
                   tuple(Presenter(i, 0, 1 if i == 1 else 0, 0) for i in range(1, n + 1)),
                   innermost_expert=expert)


def test_innermost_candidate_is_dropped_when_group_has_expert():
    """该组标记「最里侧有专家」时，剔除最靠里（y2 最小）的候选。

    用户提供 dta 的 Innermost_expert：城市2=0、文化4=1、新能源2=0。文化4
    正是他此前逐企业核对时列出一串「误把专家当创业者」的那一组，两者吻合。

    专家坐在最里侧且位置不变，所以「y2 最小」是可用的判据——y2 越小离
    镜头越远。这与此前被证伪的「深度断层」不同：那个是想用候选之间的间隙
    自动找分界线（实测专家 y2=528-557 与真实创业者 y2=464-578 完全重叠、
    无间隙可用），而这里是用户直接告知「有没有」，不需要算法去猜。
    """
    from pipeline.seats import assign_segment_seats
    # y2: 1000 / 900 / 800 —— tid3 的 y2 最小，是最里侧那位
    dets = _mk_dets_depth(20, [(1, 560, 1000), (2, 520, 900), (3, 480, 800)])
    seats_no = assign_segment_seats(dets, _seg_expert(20, 3, 0), frame_w=1440)
    seats_yes = assign_segment_seats(dets, _seg_expert(20, 3, 1), frame_w=1440)
    assert len(seats_yes) < len(seats_no), "标记有专家时应少一个人"


def test_no_expert_flag_changes_nothing():
    """标记为 0 的组行为必须与从前完全一致——两组视频靠这条不受影响。"""
    from pipeline.seats import assign_segment_seats
    dets = _mk_dets_depth(21, [(1, 560, 1000), (2, 520, 900), (3, 480, 800)])
    a = assign_segment_seats(dets, _seg_expert(21, 3, 0), frame_w=1440)
    b = assign_segment_seats(dets, Segment(21, "某公司", 0, 900, 3,
        tuple(Presenter(i, 0, 1 if i == 1 else 0, 0) for i in (1, 2, 3))), frame_w=1440)
    assert a == b


def test_expert_dropped_even_when_candidates_equal_presenter_num():
    """候选恰等于人数时仍剔除——这是剔除唯一能起作用的情形。

    曾按用户要求改为「只在候选多于人数时剔除」，实测证明那会让整个变量
    失效：排序本按离镜头远近，最里侧天然排最后，候选一有富余就已被 top-N
    截掉。文化4组候选多于人数的 3 段里，最里侧全部本来就在 top-N 之外。
    """
    from pipeline.seats import assign_segment_seats
    dets = _mk_dets_depth(22, [(1, 560, 1000), (2, 520, 800)])
    seats = assign_segment_seats(dets, _seg_expert(22, 2, 1), frame_w=1440)
    assert len(seats) == 1, f"候选=人数时仍剔除（这是剔除唯一起作用的情形），实得 {seats}"


def test_expert_dropped_only_when_candidates_exceed_presenter_num():
    """候选多于人数时才剔除最里侧——此时剔除无样本代价。"""
    from pipeline.seats import assign_segment_seats
    # 需 2 人、有 3 个候选：最里侧 tid3 本就在 top-N 之外，剔除无额外效果，
    # 结果与不标记专家时相同——这条锁住「富余情形下剔除是无操作」这个事实。
    dets = _mk_dets_depth(24, [(1, 560, 1000), (2, 520, 900), (3, 480, 800)])
    a = assign_segment_seats(dets, _seg_expert(24, 2, 0), frame_w=1440)
    b = assign_segment_seats(dets, _seg_expert(24, 2, 1), frame_w=1440)
    assert a == b, "候选富余时最里侧已被 top-N 截掉，剔除不应改变结果"
    assert 3 not in b


def test_seat_numbers_stay_contiguous_after_dropping_expert():
    """剔除后座位号仍须从 1 连续编号，不得留空洞。"""
    from pipeline.seats import assign_segment_seats
    dets = _mk_dets_depth(23, [(1, 560, 1000), (2, 520, 900), (3, 480, 800)])
    seats = assign_segment_seats(dets, _seg_expert(23, 3, 1), frame_w=1440)
    assert sorted(seats.values()) == list(range(1, len(seats) + 1))


# ---------------------------------------------------------------------------
# Task 44：expert_removed_tracks——暴露「因专家被剔除」的信息，供装配阶段
# 标记 expert_removed 这个 CSV 标志字段。
# ---------------------------------------------------------------------------

def test_expert_removed_tracks_returns_empty_without_flag():
    """未标记 innermost_expert 的组，剔除逻辑从不触发，应返回空列表。"""
    from pipeline.seats import expert_removed_tracks
    dets = _mk_dets_depth(25, [(1, 560, 1000), (2, 520, 900), (3, 480, 800)])
    assert expert_removed_tracks(dets, _seg_expert(25, 3, 0), frame_w=1440) == []


def test_expert_removed_tracks_reports_the_dropped_one():
    """返回值必须与 assign_segment_seats 实际剔除的那个一致——不是各算一遍。

    验证方式不硬编码「应该是 tid3」，而是从 assign_segment_seats 的真实
    输出反推：候选=人数时剔除唯一起作用（见
    test_expert_dropped_even_when_candidates_equal_presenter_num），剔除后
    assign_segment_seats 只返回 presenter_num-1 个座位，缺席的那个 track_id
    必须恰好等于 expert_removed_tracks 报告的那个——若两处各写一份剔除
    逻辑、哪怕只是取整/中位数算法上的细微差异，都会在这条断言上被抓到。
    """
    from pipeline.seats import assign_segment_seats, expert_removed_tracks
    dets = _mk_dets_depth(26, [(1, 560, 1000), (2, 520, 900), (3, 480, 800)])
    seg = _seg_expert(26, 3, 1)

    seats = assign_segment_seats(dets, seg, frame_w=1440)
    removed = expert_removed_tracks(dets, seg, frame_w=1440)

    assert removed == [3]
    all_candidate_tids = {1, 2, 3}
    assert set(removed) == all_candidate_tids - set(seats.keys()), (
        "expert_removed_tracks 报告的 track_id 必须恰好是从 "
        "assign_segment_seats 结果中消失的那一个"
    )


def test_gender_match_path_also_drops_the_expert():
    """性别匹配路径必须与基线路径一样剔除最里侧专家——两条路不能各行其是。

    真实缺陷：`assign_segment_seats` 的基线分支走
    `_ranked_tracks_dropping_expert`（会剔除），而 `gender_match_candidates`
    直接调 `rank_tracks_pairwise`（不剔除）。于是文化4组（innermost_expert=1）
    在性别匹配开启时，座位数恒等于 presenter_num——专家被当成创业者填进了
    座位，而 seats.json 与挑选页都显示「缺 0 人」，看上去一切正常。

    控制器正是因为两条路径给出不同的座位数（基线 22/34、性别匹配 34/34）
    才发现的，而这个差异此前无人守护。
    """
    from pipeline.seats import assign_segment_seats, gender_match_candidates

    dets = _mk_dets_depth(30, [(1, 560, 1000), (2, 520, 900), (3, 480, 800)])
    seg = _seg_expert(30, 3, 1)
    # 三个候选都判「不确定」——性别不构成约束，纯看候选池大小
    gj = {1: "不确定", 2: "不确定", 3: "不确定"}

    base = assign_segment_seats(dets, seg, frame_w=1440)
    gm = assign_segment_seats(dets, seg, frame_w=1440,
                              gender_match=True, gender_judged_by_track=gj)
    people = gender_match_candidates(dets, seg, 1440, 0.0, gj)

    assert len(people) == len(base), (
        f"两条路径的候选数必须一致：基线 {len(base)}、性别匹配 {len(people)}")
    assert len(gm) == len(base), (
        f"两条路径的座位数必须一致：基线 {len(base)}、性别匹配 {len(gm)}")


def test_no_expert_flag_keeps_both_paths_identical():
    """未标记专家的组，两条路径本就该一致——这条防止修复时改坏了另一侧。"""
    from pipeline.seats import assign_segment_seats, gender_match_candidates

    dets = _mk_dets_depth(31, [(1, 560, 1000), (2, 520, 900), (3, 480, 800)])
    seg = _seg_expert(31, 3, 0)
    gj = {1: "不确定", 2: "不确定", 3: "不确定"}
    base = assign_segment_seats(dets, seg, frame_w=1440)
    people = gender_match_candidates(dets, seg, 1440, 0.0, gj)
    assert len(people) == len(base) == 3
