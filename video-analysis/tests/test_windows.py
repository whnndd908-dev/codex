import pytest
from pipeline import config
from pipeline.schedule import Segment, Presenter
from pipeline.windows import Window, seated_interval, compute_windows, window_timestamps


def mkseg(order, start, end, n=1, entry_time=None, exit_time=None):
    ps = tuple(Presenter(k, 0, 1 if k == 1 else 0, 0) for k in range(1, n + 1))
    return Segment(order, f"公司{order}", start, end, n, ps,
                   entry_time=entry_time, exit_time=exit_time)


def test_seated_interval_clamped_to_max_for_long_segment():
    # 900 秒的段，可用区间 810 秒，810/45 = 18 秒，未触上限
    assert seated_interval(0, 900) == pytest.approx(18.0)


def test_seated_interval_clamped_to_min_for_short_segment():
    # order 11 只有 240 秒，可用区间 240-60-30 = 150 秒，150/45 = 3.33 秒 → 钳到 5
    assert seated_interval(10093, 10333) == config.SEATED_INTERVAL_MIN


def test_seated_interval_never_exceeds_max():
    assert seated_interval(0, 100000) == config.SEATED_INTERVAL_MAX


def test_entry_window_clipped_by_previous_segment_end():
    # order 7 结束 7124，order 8 开始 7136，空隙仅 12 秒
    segs = [mkseg(7, 6062, 7124), mkseg(8, 7136, 7996)]
    ws = compute_windows(segs, video_duration=15112.0)
    entry8 = next(w for w in ws if w.seg_order == 8 and w.kind == "entry")
    # 默认 7136-25 = 7111，但不得早于上一段结束 7124
    assert entry8.t_start == 7124.0


def test_exit_window_clipped_by_next_segment_start():
    segs = [mkseg(7, 6062, 7124), mkseg(8, 7136, 7996)]
    ws = compute_windows(segs, video_duration=15112.0)
    exit7 = next(w for w in ws if w.seg_order == 7 and w.kind == "exit")
    # 默认 7124+30 = 7154，但不得晚于下一段开始 7136
    assert exit7.t_end == 7136.0


def test_first_segment_entry_window_not_negative():
    segs = [mkseg(1, 0, 900)]
    ws = compute_windows(segs, video_duration=15112.0)
    entry1 = next(w for w in ws if w.kind == "entry")
    assert entry1.t_start >= 0.0


def test_last_segment_exit_window_within_video():
    segs = [mkseg(16, 14269, 14950)]
    ws = compute_windows(segs, video_duration=15112.0)
    exit16 = next(w for w in ws if w.kind == "exit")
    assert exit16.t_end <= 15112.0


def test_three_windows_per_segment():
    segs = [mkseg(1, 0, 900), mkseg(2, 960, 1836)]
    ws = compute_windows(segs, video_duration=15112.0)
    assert len(ws) == 6
    assert {w.kind for w in ws} == {"entry", "exit", "seated"}


def test_transition_windows_use_configured_fps():
    segs = [mkseg(1, 0, 900)]
    ws = compute_windows(segs, video_duration=15112.0)
    entry = next(w for w in ws if w.kind == "entry")
    assert entry.interval == pytest.approx(1.0 / config.TRANSITION_FPS)


def test_window_timestamps_are_evenly_spaced_and_within_bounds():
    w = Window(seg_order=1, kind="entry", t_start=10.0, t_end=12.0, interval=0.5)
    ts = window_timestamps(w)
    assert ts == [10.0, 10.5, 11.0, 11.5, 12.0]


def test_seated_window_respects_margins():
    segs = [mkseg(1, 0, 900)]
    ws = compute_windows(segs, video_duration=15112.0)
    seated = next(w for w in ws if w.kind == "seated")
    assert seated.t_start == 60.0
    assert seated.t_end == 870.0


def test_degenerate_short_segment_still_yields_seated_frames():
    # 极短段：确保不会产出空窗口或反向区间
    segs = [mkseg(1, 1000, 1100)]
    ws = compute_windows(segs, video_duration=15112.0)
    seated = next(w for w in ws if w.kind == "seated")
    assert seated.t_end > seated.t_start
    assert len(window_timestamps(seated)) >= 2


def test_window_timestamps_never_exceed_window_end():
    # 跨度 972 秒、间隔 20 秒，972/20=48.6 不是整数倍：
    # 旧实现四舍五入到 49 会得到 7102.0，越界 8 秒
    w = Window(seg_order=7, kind="seated", t_start=6122.0, t_end=7094.0, interval=20.0)
    ts = window_timestamps(w)
    assert max(ts) <= w.t_end, f"末点 {max(ts)} 越过窗口终点 {w.t_end}"
    assert ts[-1] == pytest.approx(w.t_end), "末点应恰为窗口终点"


def test_window_timestamps_include_both_endpoints_on_non_multiple_span():
    w = Window(seg_order=1, kind="entry", t_start=0.0, t_end=143.0, interval=5.0)
    ts = window_timestamps(w)
    assert ts[0] == pytest.approx(0.0)
    assert ts[-1] == pytest.approx(143.0)
    assert all(t <= 143.0 for t in ts)


# ---------------------------------------------------------------------------
# 人工补录的进/退场时刻（`Segment.entry_time` / `exit_time`）
#
# 有它就锚在它上面，没有才退回按段首/段尾猜。用户 2026-09-07 为城市精细化
# 管理1组补了这两列，实测 order 11 的人在段首前 130 秒就进场了——猜出来的
# 窗口 [start-25, start+15] 与真正的进场差了 105 秒，一帧都没覆盖到。
# ---------------------------------------------------------------------------

def test_entry_window_anchors_on_the_annotated_entry_time():
    # 城市1组 order 11 的真实数值：上一段 10379.28-11714.89，本段 11869.89 起，
    # 人工标的进场时刻 11739.9——比段首早 130 秒。
    #
    # 两个时刻**都要给**：真实数据里 15 段是两侧都标了的，只给一侧的夹具会让
    # 「进场锚点误取了 exit_time」这条变异活下来（变异测试实测存活过一次）。
    segs = [mkseg(10, 10379.28, 11714.89), mkseg(11, 11869.89, 13109.19,
                                                 entry_time=11739.9,
                                                 exit_time=13117.2)]
    ws = compute_windows(segs, video_duration=19300.0)
    entry11 = next(w for w in ws if w.seg_order == 11 and w.kind == "entry")
    assert entry11.t_end == pytest.approx(11739.9 + config.ENTRY_MARK_WINDOW[1])
    assert entry11.t_start == pytest.approx(11739.9 + config.ENTRY_MARK_WINDOW[0])
    # 而按段首猜出来的老窗口与它毫无交集——这正是这条改动要解决的事。
    assert entry11.t_end < 11869.89 + config.ENTRY_WINDOW[0]


def test_exit_window_anchors_on_the_annotated_exit_time():
    # 同上：两侧都给，否则「退场锚点误取了 entry_time」同样杀不掉。
    segs = [mkseg(3, 2595.98, 3645.16, entry_time=2582.0, exit_time=3668.2),
            mkseg(4, 3705.16, 5047.16)]
    ws = compute_windows(segs, video_duration=19300.0)
    exit3 = next(w for w in ws if w.seg_order == 3 and w.kind == "exit")
    assert exit3.t_start == pytest.approx(3668.2 + config.EXIT_MARK_WINDOW[0])
    assert exit3.t_end == pytest.approx(3668.2 + config.EXIT_MARK_WINDOW[1])


def test_annotated_entry_window_is_still_clipped_by_the_previous_segment_end():
    """锚点换了，裁剪不能跟着丢：窗口越过上一段结束，抽出来的就是上一家企业
    还在讲的画面，而它会被写进本段的 frames/NN/entry/ 目录里。"""
    segs = [mkseg(1, 0, 1000), mkseg(2, 1200, 2000, entry_time=1005.0)]
    ws = compute_windows(segs, video_duration=19300.0)
    entry2 = next(w for w in ws if w.seg_order == 2 and w.kind == "entry")
    assert entry2.t_start == 1000.0


def test_annotated_exit_window_is_still_clipped_by_the_next_segment_start():
    segs = [mkseg(1, 0, 1000, exit_time=1010.0), mkseg(2, 1020, 2000)]
    ws = compute_windows(segs, video_duration=19300.0)
    exit1 = next(w for w in ws if w.seg_order == 1 and w.kind == "exit")
    assert exit1.t_end == 1020.0


def test_only_the_marked_side_moves():
    """城市1组 order 1、order 8 只有 exitTime 没有 entryTime。进场窗口必须
    原样退回按段首猜，不得因为另一侧有值就跟着动。"""
    segs = [mkseg(1, 1000, 2000, exit_time=2010.0), mkseg(2, 2500, 3500)]
    ws = compute_windows(segs, video_duration=19300.0)
    entry1 = next(w for w in ws if w.seg_order == 1 and w.kind == "entry")
    assert entry1.t_start == pytest.approx(1000 + config.ENTRY_WINDOW[0])
    assert entry1.t_end == pytest.approx(1000 + config.ENTRY_WINDOW[1])


def test_seated_window_is_untouched_by_the_marks():
    """就座窗口锚在段首段尾，与进出场时刻无关——两个 mkseg 只差这两个值，
    产出的 seated 窗口必须一字不差。"""
    plain = compute_windows([mkseg(1, 1000, 2000)], video_duration=19300.0)
    marked = compute_windows([mkseg(1, 1000, 2000, entry_time=970.0,
                                    exit_time=2010.0)], video_duration=19300.0)
    s_plain = next(w for w in plain if w.kind == "seated")
    s_marked = next(w for w in marked if w.kind == "seated")
    assert (s_plain.t_start, s_plain.t_end, s_plain.interval) == \
           (s_marked.t_start, s_marked.t_end, s_marked.interval)


def test_densify_windows_follow_the_annotated_marks():
    """加密抽帧的区间由 `compute_windows` 独家给出（`densify_windows` 只改
    interval）。这条钉住的是：补了时刻之后，加密的那 400 帧真的落在人进门的
    那几秒上，而不是落在按段首猜出来的老位置。"""
    from pipeline.windows import densify_windows
    segs = [mkseg(10, 10379.28, 11714.89), mkseg(11, 11869.89, 13109.19,
                                                 entry_time=11739.9)]
    ws = densify_windows(segs, 19300.0, [11])
    entry11 = next(w for w in ws if w.kind == "entry")
    assert entry11.t_start == pytest.approx(11739.9 + config.ENTRY_MARK_WINDOW[0])
    assert entry11.interval <= 1.0 / config.TRANSITION_FPS


def test_mark_windows_span_forty_seconds():
    """人工标了时刻的进出场窗口，跨度 40 秒，与没标时的那两个一致。

    2026-09-08 试过放宽到前后各 30 秒（共 60 秒），用户当天要求退回：
    「不要用前后各 30s，还是恢复到之前的时间窗口，现在变乱了」。放宽之后
    进场窗口会一路回探到上一家企业离场的那几十秒里，两拨人挤在同一批候选
    里；而且窗口一变，候选的稳定标识跟着变，已标好的 50 条里 14 条对不上。

    这条钉的是**跨度 40 秒**这件事本身（帧预算据此而定，见
    DENSIFY_TRANSITION_TARGET_FRAMES），不是钉死某一组偏移。
    """
    for w in (config.ENTRY_MARK_WINDOW, config.EXIT_MARK_WINDOW):
        assert w[1] - w[0] == 40, w
    assert 40 / config.DENSIFY_TRANSITION_TARGET_FRAMES == pytest.approx(
        1.0 / config.TRANSITION_FPS / 2), (
        "目标帧数必须跟着窗口跨度走：40 秒 ÷ 400 帧 = 0.1 秒一帧，"
        "正好是首轮 0.2 秒网格的二等分")


def test_widening_the_window_does_not_thin_out_the_dense_pass():
    """窗口放宽之后，加密抽帧的**密度不许跟着变稀**。

    加密的间隔是「目标帧数 ÷ 窗口时长」。窗口从 40 秒放宽到 60 秒而目标帧数
    不动的话，间隔会从 0.10 秒被摊薄到 0.15 秒——用户要的是「范围再延长」，
    不是「同样多的帧摊到更长的时间上」，而这件事在命令输出里看不出来。
    """
    from pipeline.windows import densify_interval
    assert densify_interval(0.0, 40.0, "entry") <= 0.1 + 1e-9, (
        f"40 秒的进出场窗口应按 0.1 秒一帧，实得 "
        f"{densify_interval(0.0, 40.0, 'entry'):.3f}")


def test_transition_densify_interval_is_a_whole_number_subdivision_of_the_first_pass_grid():
    """进出场加密的间隔必须是首轮网格（1/TRANSITION_FPS = 0.2 秒）的**整数等分**。

    否则窗口一被邻段裁掉几秒，`span / 目标帧数` 就落在 0.095 这类数上，整条
    采样网格随之平移几毫秒——盘上已有的帧一张都对不上，同一段时间被抽第二遍，
    落下一堆相差 5 毫秒的近重复帧：白占磁盘、白跑一遍检测、还在候选里多出
    一批几乎一样的图。实测（城市1，窗口放宽到前后各 30 秒之后）有 5 段被裁，
    每段因此要多抽 400 余帧。

    对齐到 0.2 / n 之后，加密的帧与首轮的帧、以及上一轮加密的帧，落在同一张
    网格上，重跑只补真正缺的那些。
    """
    from pipeline.windows import densify_interval
    base = 1.0 / config.TRANSITION_FPS
    for span in (40.0, 38.2, 37.0, 35.9, 29.8, 20.0, 15.0):
        iv = densify_interval(0.0, span, "entry")
        n = base / iv
        assert abs(n - round(n)) < 1e-9, (
            f"跨度 {span} 秒算出的间隔 {iv} 不是 {base} 的整数等分（{n}）")
    assert densify_interval(0.0, 38.2, "entry") == pytest.approx(0.1), (
        "被邻段裁到 38.2 秒的窗口应仍按 0.1 秒一帧，与 40 秒窗口同一张网格")


def test_seated_densify_interval_is_left_alone(tmp_path=None):
    """就座那一路的间隔不套这条对齐规则。

    它的窗口不随进出场时刻变动，没有反复重抽的问题；改动它只会把已有的就座
    帧网格挪掉，凭空多抽一遍。
    """
    from pipeline.windows import densify_interval
    assert densify_interval(0.0, 495.0, "seated") == pytest.approx(5.5)


def test_spot_window_for_segment_allows_a_time_in_the_gap_between_segments():
    """按**指名的段**开窗时，时刻落在两段之间的空隙里是合法的。

    用户 2026-09-09 实测城市1 order 16：他给「座位 1 无可用画面」填的重抽时刻
    是 5:08:32 = 18512 秒，而 order 16 讲到 18506.8 秒就结束了——那 5 秒之差
    正是人起身往外走的时候，**进出场本来就发生在段与段之间的空隙里**。

    `spot_window` 要求时刻落在某一段**之内**（否则抛异常、不猜最近的一段），
    那条规则是为「只给了一个时刻、不知道属于谁」准备的。而这里段号是座位号
    带来的、明确的，不存在猜的问题；真正要守的是「别抽到隔壁企业」，所以
    区间按邻段边界裁——与 `compute_windows` 同一条裁法。
    """
    from pipeline.windows import spot_window_for_segment
    segs = [mkseg(15, 16608.99, 17734.99), mkseg(16, 17778.99, 18506.82),
            mkseg(17, 18557.82, 19247.82)]
    w = spot_window_for_segment(segs, 16, 18512.0, span=10.0, kind="exit")
    assert w.seg_order == 16 and w.kind == "exit"
    assert w.t_start == pytest.approx(18502.0) and w.t_end == pytest.approx(18522.0)


def test_spot_window_for_segment_is_clipped_by_the_neighbours():
    """裁进 [上一段结束, 下一段开始]——越过去抽到的就是别家企业的画面。"""
    from pipeline.windows import spot_window_for_segment
    segs = [mkseg(15, 16608.99, 17734.99), mkseg(16, 17778.99, 18506.82),
            mkseg(17, 18557.82, 19247.82)]
    w = spot_window_for_segment(segs, 16, 18553.0, span=30.0, kind="exit")
    assert w.t_end == pytest.approx(18557.82), "上界该被下一段开始裁住"
    w2 = spot_window_for_segment(segs, 16, 17740.0, span=30.0, kind="entry")
    assert w2.t_start == pytest.approx(17734.99), "下界该被上一段结束裁住"


def test_spot_window_for_segment_refuses_a_time_in_another_companys_range():
    """时刻落到别家企业的时间范围里就抛异常，不猜、不裁一个空窗口出来。"""
    from pipeline.windows import spot_window_for_segment
    segs = [mkseg(15, 16608.99, 17734.99), mkseg(16, 17778.99, 18506.82),
            mkseg(17, 18557.82, 19247.82)]
    with pytest.raises(ValueError, match="16"):
        spot_window_for_segment(segs, 16, 17000.0, span=10.0, kind="entry")
    with pytest.raises(ValueError, match="没有第"):
        spot_window_for_segment(segs, 99, 18512.0, span=10.0, kind="exit")
