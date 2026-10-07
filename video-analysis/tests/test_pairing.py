from pathlib import Path

import numpy as np
import pytest

from pipeline import config
from pipeline.detect import Detection
from pipeline.schedule import Segment
from pipeline.pairing import (
    group_tracks, net_motion, is_inbound, is_outbound,
    entry_candidate, exit_candidate, is_fragment, pair_transitions,
)


def det(order=2, kind="entry", t=0.0, tid=1, cx=800.0, by=600.0, h=300.0):
    """按中心点与底边造一个框，宽度固定 200。"""
    return Detection(
        seg_order=order, kind=kind, t=t,
        frame_path=Path(f"/tmp/{order}_{kind}_{t}.jpg"), track_id=tid,
        x1=cx - 100, y1=by - h, x2=cx + 100, y2=by, conf=0.9,
    )


def make_track(order, kind, tid, n, cx0, cx1, by=600.0, t0=0.0, dt=0.5):
    """造一条 n 帧的轨迹，cx 从 cx0 线性变化到 cx1，时间从 t0 递增。"""
    if n == 1:
        return [det(order, kind, t0, tid, cx0, by)]
    out = []
    for i in range(n):
        frac = i / (n - 1)
        cx = cx0 + (cx1 - cx0) * frac
        out.append(det(order, kind, t0 + i * dt, tid, cx, by))
    return out


def seg(order=2, presenter_num=2):
    return Segment(order, "测试公司", 0, 100, presenter_num, ())


# ---------------------------------------------------------------------------
# group_tracks
# ---------------------------------------------------------------------------

def test_group_tracks_never_mixes_kinds():
    dets = [
        det(order=2, kind="entry", t=0.0, tid=5, cx=100),
        det(order=2, kind="entry", t=1.0, tid=5, cx=150),
        det(order=2, kind="seated", t=0.0, tid=5, cx=900),
        det(order=2, kind="seated", t=1.0, tid=5, cx=910),
    ]
    entry = group_tracks(dets, 2, "entry")
    seated = group_tracks(dets, 2, "seated")
    assert set(d.cx for d in entry[5]) == {100, 150}
    assert set(d.cx for d in seated[5]) == {900, 910}


def test_group_tracks_sorts_each_track_by_time():
    dets = [
        det(order=2, kind="entry", t=2.0, tid=1, cx=300),
        det(order=2, kind="entry", t=0.0, tid=1, cx=100),
        det(order=2, kind="entry", t=1.0, tid=1, cx=200),
    ]
    out = group_tracks(dets, 2, "entry")
    ts = [d.t for d in out[1]]
    assert ts == sorted(ts)
    assert [d.cx for d in out[1]] == [100, 200, 300]


# ---------------------------------------------------------------------------
# net_motion / is_inbound / is_outbound
# ---------------------------------------------------------------------------

def test_net_motion_is_last_minus_first_by_time():
    track = [
        det(t=2.0, cx=500, by=700),
        det(t=0.0, cx=100, by=600),
        det(t=1.0, cx=300, by=650),
    ]
    dx, dy = net_motion(track)
    assert dx == pytest.approx(400.0)
    assert dy == pytest.approx(100.0)


def test_inbound_moves_right_outbound_moves_left():
    inbound = make_track(2, "entry", 1, n=8, cx0=100, cx1=800)
    outbound = make_track(2, "exit", 2, n=8, cx0=800, cx1=100)
    assert is_inbound(inbound) is True
    assert is_outbound(inbound) is False
    assert is_outbound(outbound) is True
    assert is_inbound(outbound) is False


def test_stationary_track_is_neither_inbound_nor_outbound():
    stationary = make_track(2, "entry", 1, n=8, cx0=500, cx1=510)
    assert is_inbound(stationary) is False
    assert is_outbound(stationary) is False


def test_stationary_track_is_still_an_entry_candidate():
    # Ruling R38: order 10 的 21 条 entry 轨迹里 17 条静止，其中就有该段唯一
    # 正确的那个人（tid=23）。「必须入向」的过滤会把它整段丢掉。
    stationary = make_track(2, "entry", 1, n=17, cx0=1350, cx1=1345)
    assert len(stationary) >= config.PAIR_MIN_TRACK_FRAMES
    assert entry_candidate(stationary) is True


def test_short_fragment_track_is_rejected():
    # 对应实测 order 7 tid=27：4 帧跨 746 px，是跟踪器 ID 跳变的产物。
    fragment = make_track(2, "entry", 1, n=4, cx0=100, cx1=846)
    assert is_fragment(fragment) is True
    assert entry_candidate(fragment) is False
    assert exit_candidate(fragment) is False


# ---------------------------------------------------------------------------
# posture (Ruling R46)
# ---------------------------------------------------------------------------

def test_is_standing_uses_ratio_not_absolute_height():
    # 局部导入：is_standing 是本轮 R46 新增函数，在实现完成前不存在。
    # 用局部导入而非顶层导入，避免整个测试文件在实现完成前因 ImportError
    # 而集体报错，掩盖了 5e/5f 两条真正应该先跑一遍看红/绿的用例。
    from pipeline.pairing import is_standing

    track = [det(order=2, kind="entry", t=float(i), tid=1, cx=500, by=600, h=500)
              for i in range(5)]
    # 同样 500px 高：就座参照 300px 的段里判为站立
    assert is_standing(track, ref_height=300.0) is True
    # 就座参照 600px 的段里判为坐姿（比值 500/600 < 1.3）
    assert is_standing(track, ref_height=600.0) is False


def test_standing_track_wins_over_seated_track_at_same_seat(monkeypatch):
    # 复刻实测 order 2 的情形：座位就座参照框高 300px。
    # 候选 A 站立（框高 500px，比值 1.67），终点离座位 75px。
    # 候选 B 坐姿（框高 310px，比值 1.03），终点几乎正压在座位上（0px）。
    # 颜色签名相同——没有姿态项时，位置代价会让 B 赢；姿态项必须把 A 拉回来。
    seated_ref = [det(order=2, kind="seated", t=float(i), tid=99, cx=800, by=700, h=300)
                  for i in range(5)]
    # entry 候选需要 >= PAIR_MIN_TRACK_FRAMES(8) 帧才不会被当成碎片滤掉
    standing_cand = [det(order=2, kind="entry", t=float(i), tid=1, cx=800, by=775, h=500)
                     for i in range(8)]
    seated_cand = [det(order=2, kind="entry", t=float(i), tid=2, cx=800, by=700, h=310)
                   for i in range(8)]
    dets = seated_ref + standing_cand + seated_cand

    same_sig = np.array([10.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {99: 1}
    result = pair_transitions(dets, seg(2, 1), seats, frame_w=1000.0, frame_h=1000.0)

    assert result[1]["entry_track"] == 1
    assert result[1]["entry_standing"] is True


def test_lone_seated_candidate_is_still_matched(monkeypatch):
    # 复刻实测 order 10 的情形：座位唯一的候选是坐姿，姿态惩罚不得把它推过
    # PAIR_MAX_COST，必须仍然配上。
    seated_ref = [det(order=2, kind="seated", t=float(i), tid=99, cx=800, by=700, h=300)
                  for i in range(5)]
    seated_cand = [det(order=2, kind="entry", t=float(i), tid=2, cx=800, by=700, h=310)
                   for i in range(8)]
    dets = seated_ref + seated_cand

    same_sig = np.array([10.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {99: 1}
    result = pair_transitions(dets, seg(2, 1), seats, frame_w=1000.0, frame_h=1000.0)

    assert result[1]["entry_track"] == 2
    assert result[1]["entry_standing"] is False


# ---------------------------------------------------------------------------
# pair_transitions: color vs position disambiguation
# ---------------------------------------------------------------------------

def test_pair_matches_by_color_when_positions_are_ambiguous(monkeypatch):
    # 两个入场轨迹终点位置几乎重合；只有颜色能区分。
    entry1 = make_track(2, "entry", 11, n=8, cx0=490, cx1=500)   # 红
    entry2 = make_track(2, "entry", 12, n=8, cx0=500, cx1=505)   # 蓝
    seated_a = make_track(2, "seated", 21, n=5, cx0=500, cx1=500)  # 红座
    seated_b = make_track(2, "seated", 22, n=5, cx0=505, cx1=505)  # 蓝座
    dets = entry1 + entry2 + seated_a + seated_b

    sigs = {
        11: np.array([100.0, 0.0, 0.0]),   # 红
        12: np.array([0.0, 100.0, 0.0]),   # 蓝
        21: np.array([95.0, 0.0, 0.0]),    # 红座参照
        22: np.array([0.0, 95.0, 0.0]),    # 蓝座参照
    }

    def fake_sig(track, max_frames=6):
        return sigs[track[0].track_id]

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {21: 1, 22: 2}
    result = pair_transitions(dets, seg(2, 2), seats, frame_w=1000.0, frame_h=1000.0)

    assert result[1]["entry_track"] == 11
    assert result[2]["entry_track"] == 12


def test_pair_matches_by_position_when_colors_are_identical(monkeypatch):
    # 故意让正确答案是「交叉」配对（tid=11 更靠近座位 22/座位号 2，
    # tid=12 更靠近座位 21/座位号 1），而不是 track_id 与座位号顺序一致的
    # 「巧合」配对——若位置权重被误置为 0，代价矩阵退化成全零，匈牙利算法
    # 的平局打破会按行/列自然顺序给出恒等匹配（座位1↔tid11、座位2↔tid12），
    # 与本测试期望的交叉答案不同，从而能抓出「位置权重失效」这类变异
    # （控制器指定的变异 (f)）。
    entry1 = make_track(2, "entry", 11, n=8, cx0=790, cx1=790)   # 靠近座位 B(22)
    entry2 = make_track(2, "entry", 12, n=8, cx0=200, cx1=210)   # 靠近座位 A(21)
    seated_a = make_track(2, "seated", 21, n=5, cx0=200, cx1=200)
    seated_b = make_track(2, "seated", 22, n=5, cx0=800, cx1=800)
    dets = entry1 + entry2 + seated_a + seated_b

    same_sig = np.array([50.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {21: 1, 22: 2}
    result = pair_transitions(dets, seg(2, 2), seats, frame_w=1000.0, frame_h=1000.0)

    assert result[1]["entry_track"] == 12   # 座位 A(1) 配到靠近它的 tid=12
    assert result[2]["entry_track"] == 11   # 座位 B(2) 配到靠近它的 tid=11


def test_pair_is_globally_optimal_not_greedy(monkeypatch):
    # 构造代价矩阵：A、B 两个座位都偏好 t1，但 B 若拿不到 t1 代价暴涨，
    # A 若拿不到 t1 代价只是略涨。全局最优应把 t1 让给 B、t2 给 A。
    # 所有轨迹用默认框高（比值 1.0，判为坐姿），故每格都叠加相同的 0.2
    # 姿态惩罚，不影响相对排序：
    # cost(A,t1)=0.2375, cost(A,t2)=0.4125, cost(B,t1)=0.2125, cost(B,t2)=0.4375
    # matching(A-t1,B-t2)=0.6750(行序贪心)  vs  matching(A-t2,B-t1)=0.6250 <- 全局最优
    entry1 = make_track(2, "entry", 11, n=8, cx0=500, cx1=500)
    entry2 = make_track(2, "entry", 12, n=8, cx0=500, cx1=500)
    seated_a = make_track(2, "seated", 21, n=5, cx0=500, cx1=500)
    seated_b = make_track(2, "seated", 22, n=5, cx0=500, cx1=500)
    dets = entry1 + entry2 + seated_a + seated_b

    sigs = {
        11: np.array([0.0, 0.0, 0.0]),
        12: np.array([20.0, 0.0, 0.0]),
        21: np.array([3.0, 0.0, 0.0]),
        22: np.array([1.0, 0.0, 0.0]),
    }

    def fake_sig(track, max_frames=6):
        return sigs[track[0].track_id]

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {21: 1, 22: 2}
    result = pair_transitions(dets, seg(2, 2), seats, frame_w=1000.0, frame_h=1000.0)

    assert result[1]["entry_track"] == 12   # 座位 A(1) 让出 t1，拿 t2
    assert result[2]["entry_track"] == 11   # 座位 B(2) 拿到它更需要的 t1
    total = result[1]["entry_cost"] + result[2]["entry_cost"]
    assert total < 0.6750 - 1e-6   # 严格优于「行序贪心」（A 抢 t1、B 只能拿 t2）的 0.6750


# ---------------------------------------------------------------------------
# direction filtering / thresholds / edge cases
# ---------------------------------------------------------------------------

def test_outbound_track_in_entry_window_is_ignored(monkeypatch):
    # 入场窗口与上一段退场窗口重叠，混进上一组正在离场的人（向左移动）。
    # 刻意让它的终点位置与颜色都与座位参照「完美吻合」（代价接近 0）——
    # 如果方向过滤被去掉，它会被代价函数当成最佳候选配上去。只有方向过滤
    # 真正生效，才会不管代价多低都排除它。
    leaving = make_track(2, "entry", 99, n=8, cx0=900, cx1=800)  # 终点落在座位正上方
    seated_a = make_track(2, "seated", 21, n=5, cx0=800, cx1=800)
    dets = leaving + seated_a

    same_sig = np.array([10.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {21: 1}
    result = pair_transitions(dets, seg(2, 1), seats, frame_w=1440.0, frame_h=1080.0)

    assert result[1]["entry_track"] is None
    for r in result.values():
        assert r["entry_track"] != 99


def test_unmatched_when_cost_exceeds_threshold(monkeypatch):
    entry1 = make_track(2, "entry", 11, n=8, cx0=100, cx1=110)
    seated_a = make_track(2, "seated", 21, n=5, cx0=1300, cx1=1300)
    dets = entry1 + seated_a

    def fake_sig(track, max_frames=6):
        if track[0].track_id == 11:
            return np.array([0.0, 0.0, 0.0])
        return np.array([500.0, 500.0, 500.0])   # 颜色差得极远

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {21: 1}
    result = pair_transitions(dets, seg(2, 1), seats, frame_w=1440.0, frame_h=1080.0)

    assert result[1]["entry_track"] is None
    assert result[1]["method"] == "unmatched"


def test_missing_transition_window_returns_none_not_exception():
    seated_a = make_track(7, "seated", 21, n=5, cx0=800, cx1=800)
    dets = list(seated_a)   # 完全没有 entry / exit 检测
    seats = {21: 1}

    result = pair_transitions(dets, seg(7, 1), seats, frame_w=1440.0, frame_h=1080.0)

    assert result[1]["entry_track"] is None
    assert result[1]["exit_track"] is None



# ---------------------------------------------------------------------------
# person-shape filter (Ruling R48)
# ---------------------------------------------------------------------------

def test_hand_sized_entry_track_is_rejected(monkeypatch):
    # 复刻 order 2 的 tid=60：27 帧、188px 高的手，终点几乎正压在座位上；
    # 与之竞争的是一条 755px 的站立轨迹。就座参照框高设为 600px，让两者的
    # R46 姿态惩罚都是 0.2（755/600=1.26 < PAIR_STANDING_H_RATIO=1.3，755px
    # 仍被判为「坐姿」）——光靠姿态软惩罚区分不出两者，手凭极低的位置代价胜出。
    # 没有人形过滤时必红。
    seated_ref = [det(order=2, kind="seated", t=float(i), tid=99, cx=800, by=700, h=600)
                  for i in range(5)]
    hand_cand = [det(order=2, kind="entry", t=float(i), tid=60, cx=805, by=702, h=188)
                 for i in range(27)]
    standing_cand = [det(order=2, kind="entry", t=float(i), tid=1, cx=900, by=750, h=755)
                      for i in range(31)]
    dets = seated_ref + hand_cand + standing_cand

    same_sig = np.array([10.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {99: 1}
    result = pair_transitions(dets, seg(2, 1), seats, frame_w=1440.0, frame_h=1080.0)

    assert result[1]["entry_track"] == 1


def test_person_shaped_filter_applies_to_exit_too(monkeypatch):
    # 同一场景搬到退场窗口——锁住「entry/exit 两侧都要过滤」，防止只改了
    # entry_candidate 漏掉 exit_candidate。
    seated_ref = [det(order=2, kind="seated", t=float(i), tid=99, cx=800, by=700, h=600)
                  for i in range(5)]
    hand_cand = [det(order=2, kind="exit", t=float(i), tid=60, cx=805, by=702, h=188)
                 for i in range(27)]
    standing_cand = [det(order=2, kind="exit", t=float(i), tid=1, cx=900, by=750, h=755)
                      for i in range(31)]
    dets = seated_ref + hand_cand + standing_cand

    same_sig = np.array([10.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {99: 1}
    result = pair_transitions(dets, seg(2, 1), seats, frame_w=1440.0, frame_h=1080.0)

    assert result[1]["exit_track"] == 1


def test_seated_but_real_person_survives_the_filter(monkeypatch):
    # 站立 900px、坐姿 350px 同场竞争（各自靠近自己的座位）：若误加
    # seats.py 那条「相对最高候选」的规则，0.45*900=405 会把真实的已落座
    # 候选（350px）也误杀。只用绝对比值（0.25*1080=270px）时两者都该通过
    # 人形过滤，各自配到自己的座位——锁住「不许照搬 SEAT_MIN_H_VS_TALLEST」。
    seated_ref1 = [det(order=2, kind="seated", t=float(i), tid=97, cx=200, by=700, h=300)
                   for i in range(5)]
    seated_ref2 = [det(order=2, kind="seated", t=float(i), tid=98, cx=1200, by=700, h=300)
                   for i in range(5)]
    standing_cand = [det(order=2, kind="entry", t=float(i), tid=1, cx=200, by=700, h=900)
                      for i in range(8)]
    seated_cand = [det(order=2, kind="entry", t=float(i), tid=2, cx=1200, by=700, h=350)
                   for i in range(8)]
    dets = seated_ref1 + seated_ref2 + standing_cand + seated_cand

    same_sig = np.array([10.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {97: 1, 98: 2}
    result = pair_transitions(dets, seg(2, 2), seats, frame_w=1440.0, frame_h=1080.0)

    assert result[1]["entry_track"] == 1
    assert result[2]["entry_track"] == 2


def test_filter_does_not_empty_a_segment(monkeypatch):
    # 某段所有候选框高都只有 200px（全是碎片，< 0.25*1080=270），过滤后
    # 一条不剩：entry_track 必须为 None 而不抛异常，method 如实标为 unmatched。
    seated_ref = [det(order=7, kind="seated", t=float(i), tid=99, cx=800, by=700, h=300)
                  for i in range(5)]
    fragment_cand = [det(order=7, kind="entry", t=float(i), tid=1, cx=800, by=700, h=200)
                      for i in range(8)]
    dets = seated_ref + fragment_cand

    same_sig = np.array([10.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {99: 1}
    result = pair_transitions(dets, seg(7, 1), seats, frame_w=1440.0, frame_h=1080.0)

    assert result[1]["entry_track"] is None
    assert result[1]["method"] == "unmatched"


def test_is_person_shaped_uses_median_not_mean():
    # 局部导入：is_person_shaped 是本轮 R48 新增函数，在实现完成前不存在。
    from pipeline.pairing import is_person_shaped

    # 9 帧里 4 帧被严重遮挡成 0px（走动中常见），5 帧是真实的 280px。
    # 中位数取第 5 个排序值 = 280（不受少数遮挡帧牵动）：>= 0.25*1080=270，通过。
    # 若换成均值 = (4*0+5*280)/9 ≈ 155.6：< 270，会被误拒——锁住变异 (e)。
    track = ([det(order=2, kind="entry", t=float(i), tid=1, cx=800, by=700, h=0.0)
              for i in range(4)]
             + [det(order=2, kind="entry", t=float(4 + i), tid=1, cx=800, by=700, h=280.0)
                for i in range(5)])

    assert is_person_shaped(track, frame_h=1080.0) is True


def test_is_person_shaped_uses_frame_height_not_absolute_pixels():
    # 局部导入：is_person_shaped 是本轮 R48 新增函数，在实现完成前不存在。
    from pipeline.pairing import is_person_shaped

    # 同样 200px 高的框：frame_h=1080 时阈值 270px（200 不达标，拒）；
    # frame_h=720 时阈值 180px（200 达标，通过）——证明用的是相对比值，
    # 不是写死的绝对像素数（锁住变异 (f)）。
    # 注：brief 举例用 300px，但 0.25*1080=270 < 300，300px 在两种画高下都会
    # 通过，起不到区分作用，故改用 200px 来实际卡住这条阈值分界。
    track = [det(order=2, kind="entry", t=float(i), tid=1, cx=800, by=700, h=200)
             for i in range(8)]

    assert is_person_shaped(track, frame_h=1080.0) is False
    assert is_person_shaped(track, frame_h=720.0) is True


# ---------------------------------------------------------------------------
# 硬位置闸门与指派分歧披露（Ruling R49 / Task 20）
# ---------------------------------------------------------------------------

def test_far_candidate_is_infeasible_regardless_of_color(monkeypatch):
    # 复刻实测病灶：远候选（位置距离 0.7）颜色与座位完全一致（色距=0），
    # 近候选（位置距离 0.05）颜色略差（归一化色距=0.5）。加权总代价下
    # 远候选 0.21 < 近候选 0.265，现行实现（无闸门）会把远的判为更优——必红。
    # 闸门生效后，远候选因位置距离 0.7 > PAIR_MAX_POS_DIST(0.35) 被判不可行，
    # 唯一剩下的近候选胜出，不管代价孰高孰低。
    near = make_track(2, "entry", 1, n=8, cx0=550, cx1=550, by=500.0)   # dx=50/1000=0.05
    far = make_track(2, "entry", 2, n=8, cx0=1200, cx1=1200, by=500.0)  # dx=700/1000=0.7
    # 两者框高都拉到 500，确保 is_standing 判定一致（posture 项不干扰对比）
    near = [det(order=2, kind="entry", t=d.t, tid=1, cx=d.cx, by=500.0, h=500.0) for d in near]
    far = [det(order=2, kind="entry", t=d.t, tid=2, cx=d.cx, by=500.0, h=500.0) for d in far]
    seated_a = [det(order=2, kind="seated", t=float(i), tid=99, cx=500, by=500, h=300)
                for i in range(5)]
    dets = near + far + seated_a

    def fake_sig(track, max_frames=6):
        tid = track[0].track_id
        if tid == 99:
            return np.array([0.0, 0.0, 0.0])
        if tid == 2:
            return np.array([0.0, 0.0, 0.0])       # far: 色距=0
        return np.array([20.0, 0.0, 0.0])          # near: 归一化色距 20/40=0.5

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {99: 1}
    result = pair_transitions(dets, seg(2, 1), seats, frame_w=1000.0, frame_h=1000.0)

    assert result[1]["entry_track"] == 1   # 配到近的（tid=1），不是色距为 0 的远候选


def test_far_candidate_alone_yields_none(monkeypatch):
    # 某座位唯一的候选位置距离 0.7，即使颜色、姿态都完美匹配，也必须判不可行，
    # entry_track 为 None，而不是硬配上。entry_best_alone 同样须为 None——
    # 若实现从「全部候选」（不限可行）里取 best_alone，会在此处漏出闸门外的 tid。
    far = make_track(2, "entry", 2, n=8, cx0=1200, cx1=1200, by=500.0)
    far = [det(order=2, kind="entry", t=d.t, tid=2, cx=d.cx, by=500.0, h=500.0) for d in far]
    seated_a = [det(order=2, kind="seated", t=float(i), tid=99, cx=500, by=500, h=300)
                for i in range(5)]
    dets = far + seated_a

    same_sig = np.array([0.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {99: 1}
    result = pair_transitions(dets, seg(2, 1), seats, frame_w=1000.0, frame_h=1000.0)

    assert result[1]["entry_track"] is None
    assert result[1]["entry_best_alone"] is None


def test_position_distance_is_normalised_per_axis():
    from pipeline.pairing import position_distance

    ref = {"cx": 0.0, "by": 0.0}
    horiz = position_distance((720.0, 0.0), ref, frame_w=1440.0, frame_h=1080.0)
    vert = position_distance((0.0, 540.0), ref, frame_w=1440.0, frame_h=1080.0)
    assert horiz == pytest.approx(0.5)
    assert vert == pytest.approx(0.5)
    assert horiz == pytest.approx(vert)


def test_best_alone_is_reported(monkeypatch):
    # 复刻实测 order 2 的分歧结构（与 test_pair_is_globally_optimal_not_greedy
    # 用同一套代价：座位 A 让出全局更偏好的 t1，全局最优把 t1 调给座位 B。
    # 座位 A 的「单独最佳」仍是 t1，与全局指派 t2 不同——必须如实披露分歧。
    entry1 = make_track(2, "entry", 11, n=8, cx0=500, cx1=500)
    entry2 = make_track(2, "entry", 12, n=8, cx0=500, cx1=500)
    seated_a = make_track(2, "seated", 21, n=5, cx0=500, cx1=500)
    seated_b = make_track(2, "seated", 22, n=5, cx0=500, cx1=500)
    dets = entry1 + entry2 + seated_a + seated_b

    sigs = {
        11: np.array([0.0, 0.0, 0.0]),
        12: np.array([20.0, 0.0, 0.0]),
        21: np.array([3.0, 0.0, 0.0]),
        22: np.array([1.0, 0.0, 0.0]),
    }

    def fake_sig(track, max_frames=6):
        return sigs[track[0].track_id]

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {21: 1, 22: 2}
    result = pair_transitions(dets, seg(2, 2), seats, frame_w=1000.0, frame_h=1000.0)

    # 全局最优：座位 A(1) 拿 t2(12)，座位 B(2) 拿 t1(11)——见既有
    # test_pair_is_globally_optimal_not_greedy 的代价推导。
    assert result[1]["entry_track"] == 12
    assert result[1]["entry_best_alone"] == 11
    assert result[1]["entry_assignment_diverged"] is True

    assert result[2]["entry_track"] == 11
    assert result[2]["entry_best_alone"] == 11
    assert result[2]["entry_assignment_diverged"] is False


def test_no_divergence_flag_when_assignment_is_also_best_alone(monkeypatch):
    # 两个候选各自明确靠近自己的座位，互不竞争——全局指派与单独最佳自然一致，
    # 两个座位的分歧布尔都应为 False。
    entry1 = make_track(2, "entry", 11, n=8, cx0=100, cx1=110)
    entry2 = make_track(2, "entry", 12, n=8, cx0=900, cx1=890)
    seated_a = make_track(2, "seated", 21, n=5, cx0=100, cx1=100)
    seated_b = make_track(2, "seated", 22, n=5, cx0=900, cx1=900)
    dets = entry1 + entry2 + seated_a + seated_b

    same_sig = np.array([10.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {21: 1, 22: 2}
    result = pair_transitions(dets, seg(2, 2), seats, frame_w=1000.0, frame_h=1000.0)

    assert result[1]["entry_track"] == 11
    assert result[1]["entry_assignment_diverged"] is False
    assert result[2]["entry_track"] == 12
    assert result[2]["entry_assignment_diverged"] is False


def test_gate_does_not_reject_borderline_real_matches(monkeypatch):
    # 实测里最远的合理匹配位置距离是 0.22（order 2 座位 2 的 tid=12）。
    # 锁住阈值不许被误调到 0.22 以下：这个候选不能被闸门拒掉。
    entry1 = make_track(2, "entry", 1, n=8, cx0=720, cx1=720, by=500.0)  # dx=220/1000=0.22
    entry1 = [det(order=2, kind="entry", t=d.t, tid=1, cx=d.cx, by=500.0, h=500.0)
              for d in entry1]
    seated_a = [det(order=2, kind="seated", t=float(i), tid=99, cx=500, by=500, h=300)
                for i in range(5)]
    dets = entry1 + seated_a

    same_sig = np.array([0.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {99: 1}
    result = pair_transitions(dets, seg(2, 1), seats, frame_w=1000.0, frame_h=1000.0)

    assert result[1]["entry_track"] == 1


def test_exit_track_matching_uses_window_start_not_end(monkeypatch):
    # 退场轨迹：起点靠近座位，终点远离座位（人起身往左走）。
    # 若误用终点，代价会暴涨甚至配不上。
    exit_tr = make_track(2, "exit", 31, n=8, cx0=500, cx1=100, by=500.0)
    seated_a = make_track(2, "seated", 21, n=5, cx0=500, cx1=500, by=500.0)
    dets = exit_tr + seated_a

    same_sig = np.array([10.0, 0.0, 0.0])

    def fake_sig(track, max_frames=6):
        return same_sig

    monkeypatch.setattr("pipeline.pairing.track_signature", fake_sig)

    seats = {21: 1}
    result = pair_transitions(dets, seg(2, 1), seats, frame_w=1000.0, frame_h=1000.0)

    assert result[1]["exit_track"] == 31
    # 色差与位置差都是 0；默认框高比值为 1.0（判为坐姿）叠加固定 0.2 姿态惩罚，
    # 故代价下限是 0.2 而非 0——这条测试验证的是"用起点不用终点"，不是姿态项。
    assert result[1]["exit_cost"] < 0.25
