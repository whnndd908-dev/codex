import subprocess
from pathlib import Path

import pytest
from pipeline import config
from pipeline.windows import Window, window_timestamps
from pipeline.frames import (
    FFMPEG, FFPROBE, FrameRef, probe_duration, extract_window, save_index, load_index,
)


def _make_video(path, video_seconds, audio_seconds, rate=10, size="320x240"):
    """合成一个视频/音频时长不一致的测试片子（testsrc + sine）。"""
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i", f"testsrc=duration={video_seconds}:size={size}:rate={rate}",
         "-f", "lavfi", "-i", f"sine=duration={audio_seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-loglevel", "error", str(path)],
        check=True,
    )


def test_probe_duration_matches_known_video():
    """时长必须覆盖最后一段的结束时间——这才是 probe_duration 的用途。

    原先写死 15110 < d < 15115（「城市2-4.26」组的时长），换组即红。
    改为与该组自己的时间表交叉验证：读得的时长必须为正，且不短于
    Book1 里最后一段的结束秒——否则抽帧会越界。这个断言对任何组都成立，
    而且比写死数字更有意义：它真正检验了「视频与时间表是否匹配」。
    """
    from pipeline.schedule import load_segments

    d = probe_duration(config.VIDEO)
    assert d > 0
    last_end = max(s.end for s in load_segments())
    assert d >= last_end, f"视频时长 {d:.1f}s 短于最后一段结束 {last_end}s"


def test_extract_window_writes_expected_number_of_frames(tmp_path):
    # 取 2 秒、5 fps → 11 帧（含首尾）
    w = Window(seg_order=1, kind="entry", t_start=1400.0, t_end=1402.0, interval=0.2)
    refs = extract_window(config.VIDEO, w, tmp_path)
    assert len(refs) == 11
    for r in refs:
        assert r.path.exists()
        assert r.path.stat().st_size > 10_000, "抽出的帧过小，可能是空图"


def test_extracted_frames_are_full_resolution(tmp_path):
    from PIL import Image
    w = Window(seg_order=1, kind="seated", t_start=1400.0, t_end=1400.0, interval=1.0)
    refs = extract_window(config.VIDEO, w, tmp_path)
    with Image.open(refs[0].path) as im:
        assert im.size == (1440, 1080)


def test_extract_window_is_idempotent(tmp_path):
    w = Window(seg_order=1, kind="entry", t_start=1400.0, t_end=1400.4, interval=0.2)
    first = extract_window(config.VIDEO, w, tmp_path)
    mtimes = [r.path.stat().st_mtime_ns for r in first]
    second = extract_window(config.VIDEO, w, tmp_path)
    assert [r.path for r in first] == [r.path for r in second]
    assert [r.path.stat().st_mtime_ns for r in second] == mtimes, "重复运行不应重新抽帧"


def test_index_roundtrip(tmp_path):
    refs = [FrameRef(1, "entry", 1400.0, tmp_path / "a.jpg")]
    p = tmp_path / "frames_index.json"
    save_index(refs, p)
    assert load_index(p) == refs


# --- Task 28: probe_duration 应读视频流时长，抽帧应对取不到帧容错 ---


def test_probe_duration_prefers_video_stream(tmp_path):
    """容器时长取音视频两流之最大值；本片音频比视频长 1 秒。
    probe_duration 必须报告视频流的时长（2s），而非容器时长（3s）——
    否则 compute_windows 拿到的上界会越过最后一帧，抽帧越界崩溃。
    """
    video = tmp_path / "mismatch.mp4"
    _make_video(video, video_seconds=2, audio_seconds=3)
    d = probe_duration(video)
    assert 1.8 <= d <= 2.2, f"probe_duration 应贴近视频流时长 2s，实得 {d}"


def test_probe_duration_falls_back_to_format(monkeypatch, tmp_path):
    """视频流查询返回空/N/A 时，应回退到 format=duration，且不抛异常。"""
    video = tmp_path / "whatever.mp4"
    video.write_bytes(b"not a real video")  # 内容无关，ffprobe 调用会被 monkeypatch 拦截

    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if FFPROBE in cmd and "stream=duration" in cmd:
            # 模拟视频流时长缺失（某些容器下为 N/A）
            return subprocess.CompletedProcess(cmd, 0, stdout="N/A\n", stderr="")
        if FFPROBE in cmd and "format=duration" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="7.500000\n", stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    d = probe_duration(video)
    assert d == pytest.approx(7.5)


def test_probe_duration_raises_when_both_missing(monkeypatch, tmp_path):
    """视频流与容器时长都拿不到时，必须抛异常——不能返回 0 或瞎猜一个值。"""
    video = tmp_path / "whatever.mp4"
    video.write_bytes(b"not a real video")

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(Exception):
        probe_duration(video)


def test_extract_skips_unreachable_timestamp(tmp_path):
    """窗口内混入一个越过视频末尾的时间点：不该崩，不该进索引，其余帧要正常抽出。

    用 41 个采样点（40 个可达 + 1 个越界）把失败率压到 1/41 ≈ 2.4%，
    低于 5% 阈值，验证的是「单点失败被跳过」而非「整体被判定系统性问题」。
    """
    video = tmp_path / "short.mp4"
    _make_video(video, video_seconds=2, audio_seconds=2, rate=25)

    w = Window(seg_order=1, kind="entry", t_start=0.0, t_end=2.0, interval=0.05)
    # window_timestamps 会产生 41 个点（0.00 ~ 2.00，步长 0.05），其中末点
    # 2.00s 落在视频编码可读范围之外（testsrc 只覆盖 [0, 2) 的实际帧）。
    out_root = tmp_path / "frames"
    refs = extract_window(video, w, out_root)

    ts = [r.t for r in refs]
    assert 2.0 not in ts, "越界时间点不应出现在返回的帧索引里"
    bad_path = out_root / f"{w.seg_order:02d}" / w.kind / "2.000.jpg"
    assert not bad_path.exists(), "抽帧失败不应残留半成品文件"
    assert len(refs) >= 39, "其余可达的帧应正常抽出"


def test_extract_raises_when_failure_rate_too_high(tmp_path, monkeypatch):
    """大部分时间点都取不到帧时，说明是系统性问题（路径/编码/权限），
    必须抛异常，不能悄悄返回一个残缺的空索引让下游继续跑一个多小时。
    """
    video = tmp_path / "short.mp4"
    _make_video(video, video_seconds=1, audio_seconds=1, rate=10)

    from pipeline import frames as frames_mod

    def fake_extract_single_frame(video, t, p):
        # 只放第一个点成功，其余全部失败——失败率远超 5% 阈值
        return t == 0.0

    monkeypatch.setattr(frames_mod, "_extract_single_frame", fake_extract_single_frame)

    w = Window(seg_order=1, kind="entry", t_start=0.0, t_end=1.0, interval=0.1)
    with pytest.raises(Exception):
        extract_window(video, w, tmp_path / "frames")


def test_extract_reports_failures_in_summary(tmp_path, monkeypatch, capsys):
    """失败但未超阈值时，函数结束应打印汇总行报告失败数，供人工核查。"""
    from pipeline import frames as frames_mod

    video = tmp_path / "short.mp4"
    _make_video(video, video_seconds=1, audio_seconds=1, rate=10)

    calls = {"n": 0}

    def fake_extract_single_frame(video, t, p):
        calls["n"] += 1
        # 41 个点里只让第 1 个失败——失败率约 2.4%，低于阈值，不应抛异常
        return calls["n"] != 1

    monkeypatch.setattr(frames_mod, "_extract_single_frame", fake_extract_single_frame)

    w = Window(seg_order=1, kind="entry", t_start=0.0, t_end=2.0, interval=0.05)
    extract_window(video, w, tmp_path / "frames")

    out = capsys.readouterr().out
    assert "1" in out and "抽帧失败" in out, f"汇总行应报告失败数，实际输出：{out!r}"


# --- 帧与时间表的一致性校验（用户 2026-08-22 发现的真实错配）---

def test_frames_outside_segment_window_are_detected(tmp_path):
    """盘上的帧若落在该段区间之外，必须被检出。

    真实事故：文化4组的 Book1 从 13 家变成 14 家（补了一家缺时间戳的），
    新增那家排在第 8 位，导致其后所有企业的行号后移一位。而帧是按旧编号
    抽的、一直存在盘上没重抽——`frames/08/` 里装的仍是旧编号 8（即现在的
    第 9 家）的画面。于是 seg 8–13 的衣着、颜色、性别判定全部取自隔壁公司。

    整条流水线照常跑完、表照常生成、图照常出，**没有任何一处报错**——
    是用户肉眼看出「华韵的图是中科直线的」才发现的。而检测方法很简单：
    帧文件名就是时间戳，比对它是否落在该段区间内即可。
    """
    from pipeline.frames import segment_frame_mismatches
    from pipeline.schedule import Segment, Presenter

    seg = Segment(order=8, company="某公司", start=7414, end=7683, presenter_num=1,
                  presenters=(Presenter(1, 0, 1, 0),))
    d = tmp_path / "08" / "seated"
    d.mkdir(parents=True)
    for t in (7743.0, 8100.0, 8356.0):        # 全部晚于 end=7683
        (d / f"{t:010.3f}.jpg").write_bytes(b"x")

    bad = segment_frame_mismatches([seg], tmp_path)
    assert bad, "帧全部落在区间之外，必须被检出"
    assert bad[0]["order"] == 8


def test_aligned_frames_pass(tmp_path):
    """帧落在区间内（含抽帧的头尾留白）时不得误报。"""
    from pipeline.frames import segment_frame_mismatches
    from pipeline.schedule import Segment, Presenter

    seg = Segment(order=3, company="某公司", start=2497, end=3368, presenter_num=1,
                  presenters=(Presenter(1, 0, 1, 0),))
    d = tmp_path / "03" / "seated"
    d.mkdir(parents=True)
    for t in (2557.0, 2900.0, 3338.0):
        (d / f"{t:010.3f}.jpg").write_bytes(b"x")

    assert not segment_frame_mismatches([seg], tmp_path)


def test_missing_frames_are_reported_not_silently_ok(tmp_path):
    """某段一帧都没有时要报出来——它同样意味着产物不完整。"""
    from pipeline.frames import segment_frame_mismatches
    from pipeline.schedule import Segment, Presenter

    seg = Segment(order=9, company="没帧的公司", start=100, end=900, presenter_num=1,
                  presenters=(Presenter(1, 0, 1, 0),))
    bad = segment_frame_mismatches([seg], tmp_path)
    assert bad and bad[0]["reason"] == "no_frames"


# ===========================================================================
# Task 56C · 三：只给「有座位标了无可用画面」的那几段二次加密抽帧
#
# 用户 2026-08-24：「如果某个座位的人没有合适的候选图，能标记下来，再抽帧
# 二次有更多的备选吗」。
#
# 两条铁律来自本项目栽过的两次跟头，本节的测试逐条守住：
#   1. 帧与企业错位（`segment_frame_mismatches` 的 docstring 记着全过程）
#      ——新抽的帧必须落在本段时间窗之内，能过那道硬校验。
#   2. 只动这几段——不许整组重跑，更不许删掉别的段已有的帧。
# ===========================================================================

def _seg(order, start, end, company="某公司", presenter_num=1):
    from pipeline.schedule import Presenter, Segment
    return Segment(order=order, company=company, start=start, end=end,
                   presenter_num=presenter_num,
                   presenters=tuple(Presenter(i + 1, 0, 1, 0)
                                    for i in range(presenter_num)))


def _fake_extract(monkeypatch):
    """把 ffmpeg 换成「写一个占位文件」——本任务跑不了真实抽帧（视频与
    work/ 都不在 worktree 里，且只读）。"""
    from pipeline import frames as frames_mod

    def fake(video, t, p):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpeg-placeholder")
        return True

    monkeypatch.setattr(frames_mod, "_extract_single_frame", fake)


def test_densify_interval_is_denser_for_transitions_than_seated():
    """「坐下的阶段可以抽稀疏一点，而进出一定要密集抽」（用户原话）。"""
    from pipeline.windows import densify_interval, seated_interval

    entry = densify_interval(1000.0, 1040.0, "entry")
    seated = densify_interval(1060.0, 1970.0, "seated")
    assert entry < seated, f"进出场必须比就座密，实得 entry={entry} seated={seated}"
    assert entry < 1.0 / config.TRANSITION_FPS, (
        f"加密后的进出场间隔必须比首轮（1/{config.TRANSITION_FPS}s）更密，"
        f"实得 {entry}")
    assert seated < seated_interval(1000, 2000), (
        f"就座虽稀疏，也要比首轮密——否则一帧新画面都不会多出来，实得 {seated}")


def test_densify_interval_adapts_to_window_length():
    """密度按窗口时长自适应，不是写死一个魔数。

    照 `seated_interval` 那条既有路子：目标帧数 ÷ 实际窗口时长，再由上下限
    钳住。用户此前明确要求过「所有的固定参数改为根据每组的情况自动调整」。
    """
    from pipeline.windows import densify_interval

    short = densify_interval(0.0, 200.0, "seated")
    long_ = densify_interval(0.0, 800.0, "seated")
    assert short < long_, (
        f"窗口越长间隔越大（帧数才不会随段长爆炸），实得 {short} / {long_}")

    # 进出场同理。注意间隔会被对齐到首轮网格 0.2 秒的整数等分（见
    # `windows.densify_interval`），所以要取两档之间确实跨了一格的一对。
    e_short = densify_interval(0.0, 20.0, "entry")
    e_long = densify_interval(0.0, 40.0, "entry")
    assert e_short < e_long, (
        f"进出场同理，实得 {e_short} / {e_long}")


def test_densify_interval_is_clamped_both_ways():
    """上下限必须钳住：再长的窗口也不许比首轮还稀（等于白抽），
    再短的窗口也不许密到超过视频本身的帧率（只会抽出一串重复画面）。"""
    from pipeline.windows import densify_interval

    assert densify_interval(0.0, 100000.0, "entry") <= 1.0 / config.TRANSITION_FPS
    assert densify_interval(0.0, 0.5, "entry") >= config.DENSIFY_TRANSITION_INTERVAL_MIN
    # 退化窗口（首尾相等/倒置）不得除零、不得返回 0 间隔
    assert densify_interval(10.0, 10.0, "seated") > 0


def test_densify_windows_only_covers_requested_orders():
    """只动这几段。别的段一个窗口都不许产出——整组重跑要跑一个多小时，
    而且会把用户已经标好的那些段的检测结果一起冲掉。"""
    from pipeline.windows import densify_windows

    segs = [_seg(3, 2497, 3368), _seg(4, 3400, 4200), _seg(5, 4230, 5000)]
    ws = densify_windows(segs, video_duration=6000.0, orders=[4])
    assert {w.seg_order for w in ws} == {4}, sorted(w.seg_order for w in ws)
    assert {w.kind for w in ws} == {"entry", "exit", "seated"}, (
        f"三个窗口都要加密（人可能在任一阶段才露脸），实得 "
        f"{sorted({w.kind for w in ws})}")


def test_densify_windows_are_clamped_by_neighbouring_segments():
    """相邻段的边界照样要裁——即使只点名重抽中间那一段。

    进场窗口相对段首往前 25 秒，相邻段空隙最小仅 12 秒；不裁就会抽到上一家
    企业的画面，而那正是「帧落在别人时间窗里」这一类事故的源头。
    """
    from pipeline.windows import compute_windows, densify_windows

    segs = [_seg(3, 2497, 3368), _seg(4, 3380, 4200), _seg(5, 4210, 5000)]
    ws = {w.kind: w for w in densify_windows(segs, 6000.0, orders=[4])}
    base = {w.kind: w for w in compute_windows(segs, 6000.0) if w.seg_order == 4}
    for kind in ("entry", "exit", "seated"):
        assert ws[kind].t_start == base[kind].t_start, kind
        assert ws[kind].t_end == base[kind].t_end, kind
    assert ws["entry"].t_start >= segs[0].end, (
        f"进场窗口不得侵入上一段，实得 {ws['entry'].t_start} < {segs[0].end}")


def test_densified_frames_pass_the_segment_window_check(tmp_path, monkeypatch):
    """**加密抽出来的帧必须过 `segment_frame_mismatches` 这道硬校验。**

    这是本任务最要紧的一条。真实事故：文化4组的 Book1 从 13 家变 14 家，
    企业序号整体后移，帧却还在旧编号下，第 8-13 段编码的是另一家企业的
    衣着，全程不报错。此后阶段 3/4 之前加了这道硬失败拦截——新抽的帧若
    落在本段时间窗之外，整条流水线会拒绝继续。
    """
    from pipeline.densify import densify
    from pipeline.frames import segment_frame_mismatches

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368), _seg(4, 3400, 4200)]
    root = tmp_path / "frames"
    densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
            frames_root=root, index_path=tmp_path / "frames_index.json",
            video_duration=6000.0)

    assert (root / "03" / "seated").is_dir(), "该段应有新帧"
    bad = segment_frame_mismatches([s for s in segs if s.order == 3], root)
    assert bad == [], f"加密抽出的帧必须落在本段时间窗内，实得 {bad}"


def test_densify_leaves_other_segments_frames_untouched(tmp_path, monkeypatch):
    """别的段已有的帧一个都不许动（更不许删）。

    用户在别的段上花的人工标注全部挂在那些帧上；删一个目录，那些标注连同
    检测结果一起作废，而这在产出物里看不出来。
    """
    from pipeline.densify import densify

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368), _seg(4, 3400, 4200)]
    root = tmp_path / "frames"
    other = root / "04" / "seated"
    other.mkdir(parents=True)
    keep = other / "003500.000.jpg"
    keep.write_bytes(b"old")
    before = (sorted(p.name for p in other.iterdir()), keep.stat().st_mtime_ns)

    densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
            frames_root=root, index_path=tmp_path / "frames_index.json",
            video_duration=6000.0)

    after = (sorted(p.name for p in other.iterdir()), keep.stat().st_mtime_ns)
    assert after == before, f"未点名的段必须原封不动，实得 {after} ≠ {before}"


def test_densify_keeps_existing_frames_of_the_target_segment(tmp_path, monkeypatch):
    """本段原有的帧也不删——加密是**补**帧，不是重抽。

    首轮那些帧上挂着当前的检测结果与人工标注，删了等于把这一段推倒重来。
    """
    from pipeline.densify import densify

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368)]
    root = tmp_path / "frames"
    d = root / "03" / "seated"
    d.mkdir(parents=True)
    old = d / "002600.000.jpg"
    old.write_bytes(b"old")
    old_mtime = old.stat().st_mtime_ns

    densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
            frames_root=root, index_path=tmp_path / "frames_index.json",
            video_duration=6000.0)

    assert old.exists() and old.stat().st_mtime_ns == old_mtime, (
        "原有的帧必须原样留着，且不得重抽（重抽会改 mtime）")


def test_densify_refuses_when_existing_frames_are_out_of_window(tmp_path, monkeypatch):
    """本段盘上已有的帧就已经错位时，拒绝加密，一帧都不抽。

    往一个已经装着别家企业画面的目录里再补帧，只会让两家的画面混在一起，
    而且此后再也分不开。这时该做的是整段重抽（阶段 2），不是加密。
    """
    import pytest as _pytest
    from pipeline.densify import densify

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368)]
    root = tmp_path / "frames"
    d = root / "03" / "seated"
    d.mkdir(parents=True)
    (d / "008888.000.jpg").write_bytes(b"other-company")   # 远在本段之外

    with _pytest.raises(RuntimeError) as ei:
        densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
                frames_root=root, index_path=tmp_path / "frames_index.json",
                video_duration=6000.0)
    assert "3" in str(ei.value)
    assert sorted(p.name for p in d.iterdir()) == ["008888.000.jpg"], (
        "拒绝时不许留下任何新抽的帧")


def test_densify_merges_new_frames_into_the_index(tmp_path, monkeypatch):
    """新帧必须并进 `frames_index.json`——阶段 3 读的是这份索引
    （`detect_frames(load_index(idx_path))`），不并进去等于白抽。

    并且别的段的索引条目一条都不能少。
    """
    from pipeline.densify import densify
    from pipeline.frames import FrameRef, load_index, save_index

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368), _seg(4, 3400, 4200)]
    root = tmp_path / "frames"
    idx = tmp_path / "frames_index.json"
    keep = FrameRef(4, "seated", 3500.0, root / "04" / "seated" / "003500.000.jpg")
    mine = FrameRef(3, "seated", 2600.0, root / "03" / "seated" / "002600.000.jpg")
    for r in (keep, mine):                      # 首轮抽下来的帧，盘上确实存在
        r.path.parent.mkdir(parents=True, exist_ok=True)
        r.path.write_bytes(b"old")
    save_index([keep, mine], idx)

    densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
            frames_root=root, index_path=idx, video_duration=6000.0)

    refs = load_index(idx)
    assert keep in refs, "别的段的索引条目一条都不能丢"
    assert refs.count(mine) == 1, f"同一帧不许在索引里出现两次，实得 {refs.count(mine)}"
    n3 = len([r for r in refs if r.seg_order == 3])
    assert n3 > 1, f"本段的索引条目应显著增多，实得 {n3}"
    assert all(r.path.exists() for r in refs if r.seg_order == 3), (
        "索引里的每一条都必须真的有文件——索引里有、盘上没有会让阶段 3 崩在半路")


def test_densify_writes_a_manifest_of_what_it_added(tmp_path, monkeypatch):
    """加密抽的帧与首轮的帧合并在同一个目录（阶段 3 按 (seg_order, kind)
    分组跟踪，分开放就变成另一个跟踪窗口、拿不到同一批 track），因此另写
    一份清单记下这一轮到底加了哪些文件——否则「新帧与旧帧混在一起无法
    分辨」，想退回都没得退。"""
    from pipeline.densify import densify

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368)]
    root = tmp_path / "frames"
    manifest = tmp_path / "frames_dense.json"
    report = densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
                     frames_root=root, index_path=tmp_path / "frames_index.json",
                     video_duration=6000.0, manifest_path=manifest)

    import json as _json
    data = _json.loads(manifest.read_text(encoding="utf-8"))
    assert data["orders"] == [3]
    assert len(data["added"]) == report["n_added"] > 0
    assert all(Path(p).exists() for p in data["added"])


def test_seats_needing_frames_survives_a_stale_fingerprint(tmp_path):
    """标注文件的检测指纹早就对不上了，也照样能读出该重抽哪几段。

    `load_entry_labels` 在指纹不符时会把整份标注作废——那是为 track 派生的
    标识准备的。而「本座位无可用画面」的键是 (order, seat)，跨重检测稳定。
    走那条作废逻辑的后果很具体：加密抽帧 → 重检测（指纹必变）→ 第二轮
    加密时读不到任何目标，命令看起来跑成功了却什么都没做。
    """
    import json as _json
    from pipeline.densify import seats_needing_frames

    p = tmp_path / "entry_labels_组.json"
    p.write_text(_json.dumps({
        "_meta": {"detections_fingerprint": "早就过期的指纹", "sheet": "组",
                  "n_candidates": 3},
        "no_candidate_seats": {"3": [2], "7": [1, 3]},
    }, ensure_ascii=False), encoding="utf-8")

    assert seats_needing_frames(p) == {3: [2], 7: [1, 3]}
    assert seats_needing_frames(tmp_path / "不存在.json") == {}


def test_densify_cli_tells_the_user_to_rerun_detection(tmp_path, monkeypatch, capsys):
    """CLI 必须明说下一步该跑什么——抽完帧不重跑检测，一个新候选都不会有，
    用户会以为这条命令没用。"""
    import json as _json
    from pipeline import densify as densify_mod
    from pipeline.schedule import save_segments

    _fake_extract(monkeypatch)
    work = tmp_path / "组"
    work.mkdir()
    segs = [_seg(3, 2497, 3368), _seg(4, 3400, 4200)]
    save_segments(segs, work / "segments.json")
    (work / f"entry_labels_组.json").write_text(_json.dumps(
        {"_meta": {"detections_fingerprint": "FP"},
         "no_candidate_seats": {"3": [2]}}, ensure_ascii=False), encoding="utf-8")

    rc = densify_mod.main(["--work", str(work), "--sheet", "组",
                           "--video", str(tmp_path / "fake.mp4"),
                           "--duration", "6000"])
    assert rc == 0, f"CLI 应正常退出，实得 {rc}"
    printed = capsys.readouterr().out
    assert "run_all" in printed and "--stages 3" in printed, (
        f"要明确写出下一步的命令，实得 {printed!r}")
    assert "第 3 段" in printed and "座位 2" in printed, (
        f"要说明是为哪几段、哪几个座位加密的，实得 {printed!r}")
    assert "04" not in printed.replace(str(tmp_path), ""), (
        f"没被点名的段不该出现在输出里，实得 {printed!r}")
    assert (work / "frames" / "03" / "entry").is_dir()
    assert not (work / "frames" / "04").exists(), "未点名的段一帧都不该抽"


def test_densify_cli_reports_nothing_to_do(tmp_path, capsys):
    """没有任何座位标了「无可用画面」时，什么都不抽，并说清原因。

    静默跑完最糟：用户以为抽了，等半天再跑检测，候选一个没多。
    """
    from pipeline import densify as densify_mod
    from pipeline.schedule import save_segments

    work = tmp_path / "组"
    work.mkdir()
    save_segments([_seg(3, 2497, 3368)], work / "segments.json")

    rc = densify_mod.main(["--work", str(work), "--sheet", "组",
                           "--video", str(tmp_path / "fake.mp4"),
                           "--duration", "6000"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "无可用画面" in printed and "挑选页" in printed, (
        f"要提示用户先去挑选页标记，实得 {printed!r}")
    assert not (work / "frames").exists(), "没有目标时一帧都不该抽"


def test_densify_hard_fails_if_new_frames_land_outside_the_window(tmp_path, monkeypatch):
    """万一算错窗口、把帧抽到了本段之外，必须当场抛异常。

    补于变异测试：抽完之后那道复查原本没有测试覆盖。这道校验防的正是本项目
    真实发生过的那类事故——帧落进别家企业的时间窗，下游的衣着/颜色/性别全部
    取自隔壁公司，而整条流水线照常跑完、没有任何一处报错。
    """
    from pipeline import densify as densify_mod
    from pipeline.windows import Window

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368)]
    # 把窗口整个搬到本段之外（模拟一次算错）
    monkeypatch.setattr(densify_mod, "densify_windows",
                        lambda segments, dur, orders: [
                            Window(3, "seated", 9000.0, 9010.0, 5.0)])

    with pytest.raises(RuntimeError) as ei:
        densify_mod.densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
                            frames_root=tmp_path / "frames",
                            index_path=tmp_path / "frames_index.json",
                            video_duration=10000.0)
    assert "时间窗" in str(ei.value), str(ei.value)


def test_densify_hard_fails_if_any_existing_frame_disappears(tmp_path, monkeypatch):
    """抽帧过程中若有已存在的帧消失，当场抛异常。

    补于变异测试。「只增不减」这条此前只是实现里的约定，没有测试盯着。
    已有帧上挂着用户做完的人工标注与检测结果，删一张就让那份标注失效，
    而这在产出物里完全看不出来。
    """
    from pipeline import densify as densify_mod

    segs = [_seg(3, 2497, 3368)]
    root = tmp_path / "frames"
    d = root / "03" / "seated"
    d.mkdir(parents=True)
    victim = d / "002600.000.jpg"
    victim.write_bytes(b"old")

    def hostile_extract(video, w, out_root):
        if victim.exists():
            victim.unlink()          # 模拟「顺手清空目录」这一类改动
        return []

    monkeypatch.setattr(densify_mod, "extract_window", hostile_extract)
    with pytest.raises(RuntimeError) as ei:
        densify_mod.densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
                            frames_root=root,
                            index_path=tmp_path / "frames_index.json",
                            video_duration=6000.0)
    assert "删" in str(ei.value), str(ei.value)


def test_merge_index_dedupes_and_keeps_every_existing_entry(tmp_path):
    """并索引：同一个采样点只留一条，已有条目一条不少。

    补于变异测试（原来的端到端用例撞不上重复的时间点）。同一帧在索引里出现
    两次，阶段 3 会对同一张图跑两遍跟踪，凭空多出一条重复的 track——而候选
    列表里多出来的那一条与真人长得一模一样，人工挑选时根本看不出是重复。
    """
    from pipeline.frames import FrameRef, merge_index

    a = FrameRef(3, "seated", 2600.0, tmp_path / "a.jpg")
    b = FrameRef(4, "seated", 3500.0, tmp_path / "b.jpg")
    dup = FrameRef(3, "seated", 2600.0, tmp_path / "a.jpg")
    c = FrameRef(3, "entry", 2480.0, tmp_path / "c.jpg")

    out = merge_index([a, b], [dup, c])
    assert out.count(a) == 1, f"重复的采样点只能留一条，实得 {out}"
    assert b in out, "别的段的条目一条都不能丢"
    assert c in out, "新帧要并进来"
    assert out == sorted(out, key=lambda r: (r.seg_order, r.kind, r.t)), out


def test_densified_entry_frames_do_not_reach_into_the_previous_segment(
        tmp_path, monkeypatch):
    """加密抽的**进场**帧不得越进上一家企业的时间段。

    补于变异测试：`segment_frame_mismatches` 只查 seated 目录，entry/exit
    越界它是查不出来的，只能在这里直接盯住。进场窗口相对段首往前 25 秒，
    而相邻段空隙最小仅 12 秒——`densify` 若只把目标段喂给窗口计算，
    上一段的边界就没了，抽出来的前十几秒全是上一家企业的画面。
    """
    from pipeline.densify import densify

    _fake_extract(monkeypatch)
    prev, target = _seg(3, 2497, 3368), _seg(4, 3380, 4200)
    root = tmp_path / "frames"
    densify(video=tmp_path / "fake.mp4", segments=[prev, target], orders=[4],
            frames_root=root, index_path=tmp_path / "frames_index.json",
            video_duration=6000.0)

    ts = sorted(float(p.stem) for p in (root / "04" / "entry").glob("*.jpg"))
    assert ts, "进场窗口应抽到帧"
    assert ts[0] >= prev.end, (
        f"最早一帧 {ts[0]} 落在上一段（结束于 {prev.end}）里")


def test_densify_cli_does_not_send_the_user_to_the_vlm_stage(tmp_path, monkeypatch, capsys):
    """加密抽帧之后的下一步是**阶段 6（重出挑选页）**，不是阶段 5。

    原先这条提示写的是 `--stages 5   # 重出挑选页`。阶段 5 是**打标**：照着
    做会真的去调 VLM——花钱，而且用的是还没标完的那份标注，结果必然被后续
    重跑覆盖；同时挑选页一点没重出，用户看不到新候选，只会以为加密抽帧没用。
    阶段 5 才是生成挑选页（2026-08-29 重排后：5=挑选页，7=打标，8=画廊）。
    """
    import json as _json
    from pipeline import densify as densify_mod
    from pipeline.schedule import save_segments

    _fake_extract(monkeypatch)
    work = tmp_path / "组"
    work.mkdir()
    save_segments([_seg(3, 2497, 3368), _seg(4, 3400, 4200)],
                  work / "segments.json")
    (work / "entry_labels_组.json").write_text(_json.dumps(
        {"_meta": {"detections_fingerprint": "FP"},
         "no_candidate_seats": {"3": [2]}}, ensure_ascii=False), encoding="utf-8")

    densify_mod.main(["--work", str(work), "--sheet", "组",
                      "--video", str(tmp_path / "fake.mp4"), "--duration", "6000"])
    printed = capsys.readouterr().out
    assert "--stages 7" not in printed, (
        f"阶段 7 是打标，不是重出挑选页；照着做会白花 VLM 调用：{printed!r}")
    assert "--stages 5" in printed, (
        f"要指向阶段 5（重出挑选页），实得 {printed!r}")


# --- 按绝对时刻加密（用户 2026-08-25：文化4 order8「126 分的退场」）---

def test_spot_window_targets_an_arbitrary_moment():
    """`spot_window` 按用户指名的时刻给出一个加密窗口。

    用户实测：文化4 order 8 的人在 **126 分**（7560s）就离场了，而退场窗口
    锚在段尾（`EXIT_WINDOW` 相对 segment.end），又被下一段起点截断，只剩
    7673-7683 这 10 秒——整个退场窗口都没覆盖到他。那一带只有 seated 的
    2.5 秒一张，比退场窗口的 0.067 秒稀 37 倍。

    预设窗口覆盖不到时，用户得能直接说「就抽这一带」。
    """
    from pipeline.windows import spot_window

    segs = [_seg(8, 7414, 7683), _seg(9, 7683, 8386)]
    w = spot_window(segs, at=7560.0, span=20.0, kind="exit")
    assert w.seg_order == 8, f"7560s 落在第 8 段里，实得 {w.seg_order}"
    assert w.kind == "exit", w.kind
    assert (w.t_start, w.t_end) == (7540.0, 7580.0), (w.t_start, w.t_end)
    assert w.interval <= 0.2, f"要用加密后的间隔，实得 {w.interval}"


def test_spot_window_is_clamped_inside_its_own_segment():
    """窗口不许越出本段——越界的帧会让下游把隔壁企业的画面当成这一段的。

    这正是 `segment_frame_mismatches` 存在的理由（文化4 企业行序后移一位，
    第 8-13 段编码的是隔壁公司的衣着，一处不报错）。
    """
    from pipeline.windows import spot_window

    segs = [_seg(8, 7414, 7683), _seg(9, 7683, 8386)]
    w = spot_window(segs, at=7420.0, span=60.0, kind="entry")
    assert w.t_start >= 7414.0, f"不许侵入上一段，实得 {w.t_start}"
    w2 = spot_window(segs, at=7670.0, span=60.0, kind="exit")
    assert w2.t_end <= 7683.0, f"不许侵入下一段，实得 {w2.t_end}"


def test_spot_window_accepts_the_segment_boundaries():
    """段首/段尾这两个**边界时刻**要算在段内。

    用 `<` 而不是 `<=` 的话，正好指在边界上的时刻会被判成「不属于任何一段」
    而报错——而人离场往往就发生在段尾那一刻，那恰恰是最该指的位置。
    """
    from pipeline.windows import spot_window

    segs = [_seg(8, 7414, 7683), _seg(9, 7683, 8386)]
    assert spot_window(segs, at=7414.0, span=5.0, kind="entry").seg_order == 8
    # 7683 同时是第 8 段的尾与第 9 段的首，取先匹配到的那一段，行为确定
    assert spot_window(segs, at=7683.0, span=5.0, kind="exit").seg_order == 8


def test_spot_window_rejects_a_moment_outside_every_segment():
    """指到任何一段之外就报错，不猜最近的那一段。

    猜错的后果是把一批帧塞进错的企业名下，而那正是本项目栽过的那类事故。
    """
    import pytest
    from pipeline.windows import spot_window

    segs = [_seg(8, 7414, 7683)]
    with pytest.raises(ValueError):
        spot_window(segs, at=9999.0, span=10.0, kind="exit")


def test_densify_cli_accepts_at(tmp_path, monkeypatch, capsys):
    """`--at 7560 --kind exit` 只抽那一带，并如实打印抽了哪个区间。"""
    import json as _json
    from pipeline import densify as densify_mod
    from pipeline.schedule import save_segments

    _fake_extract(monkeypatch)
    work = tmp_path / "组"
    work.mkdir()
    save_segments([_seg(8, 7414, 7683), _seg(9, 7683, 8386)],
                  work / "segments.json")

    rc = densify_mod.main(["--work", str(work), "--sheet", "组",
                           "--video", str(tmp_path / "fake.mp4"),
                           "--duration", "9000",
                           "--at", "7560", "--span", "20", "--kind", "exit"])
    assert rc == 0, f"实得 {rc}"
    out = capsys.readouterr().out
    assert "7540" in out and "7580" in out, f"要打印实际抽的区间：{out!r}"
    assert "第 8 段" in out, f"要说明落在哪一段：{out!r}"

    # 只看打印是不够的：窗口没真的传给 densify 时它会退回**预设窗口**，
    # 照样能把那两个数字打出来（变异测试实测存活）。看落盘的帧。
    got = sorted(q.name for q in (work / "frames" / "08" / "exit").glob("*.jpg"))
    assert got, "指名的那一带必须真的抽出帧"
    ts = [float(n[:-4]) for n in got]
    assert 7540.0 <= min(ts) and max(ts) <= 7580.0, (
        f"抽出来的帧必须落在指名的区间 7540-7580 内，实得 {min(ts)}-{max(ts)}")
    assert not (work / "frames" / "08" / "entry").exists(), (
        "指名了 exit 就不该顺带把预设的 entry 窗口也抽一遍")


def test_densify_at_does_not_need_no_candidate_marks(tmp_path, monkeypatch, capsys):
    """`--at` 与「本座位无可用画面」互不依赖：用户肉眼看到关键帧就能直接抽。"""
    import json as _json
    from pipeline import densify as densify_mod
    from pipeline.schedule import save_segments

    _fake_extract(monkeypatch)
    work = tmp_path / "组"
    work.mkdir()
    save_segments([_seg(8, 7414, 7683)], work / "segments.json")
    # 标注文件里一个「无可用画面」都没有
    (work / "entry_labels_组.json").write_text("{}", encoding="utf-8")

    rc = densify_mod.main(["--work", str(work), "--sheet", "组",
                           "--video", str(tmp_path / "fake.mp4"),
                           "--duration", "9000", "--at", "7560"])
    assert rc == 0, f"实得 {rc}"
    assert "没有任何座位被标为" not in capsys.readouterr().out


def test_ffmpeg_path_is_overridable_by_environment(monkeypatch):
    """ffmpeg/ffprobe 的路径要能用环境变量覆盖。

    原先写死成 `/opt/homebrew/bin/ffmpeg`——那是本机 Homebrew 的位置，Linux
    服务器上根本不存在，阶段 2 一开始就失败。要把流水线搬上服务器（用户
    2026-08-29「整理上传给服务器」），这是第一道硬门槛。

    默认值保持不变：本机行为一字不改，没设环境变量时仍走原路径。
    """
    import importlib

    from pipeline import frames as frames_mod

    monkeypatch.setenv("DRESS_FFMPEG", "/usr/bin/ffmpeg")
    monkeypatch.setenv("DRESS_FFPROBE", "/usr/bin/ffprobe")
    reloaded = importlib.reload(frames_mod)
    try:
        assert reloaded.FFMPEG == "/usr/bin/ffmpeg"
        assert reloaded.FFPROBE == "/usr/bin/ffprobe"
    finally:
        monkeypatch.delenv("DRESS_FFMPEG")
        monkeypatch.delenv("DRESS_FFPROBE")
        importlib.reload(frames_mod)


def test_ffmpeg_default_stays_on_the_local_path():
    """没设环境变量时默认值不变——本机行为一字不改。"""
    import importlib

    from pipeline import frames as frames_mod

    reloaded = importlib.reload(frames_mod)
    assert reloaded.FFMPEG == "/opt/homebrew/bin/ffmpeg"
    assert reloaded.FFPROBE == "/opt/homebrew/bin/ffprobe"


# ---------------------------------------------------------------------------
# 退回上一轮加密抽帧（用户 2026-09-08：「不要用前后各 30s，还是恢复到之前的
# 时间窗口，现在变乱了，重新抽帧」）。
#
# 加密只增不删是对的——已有帧上挂着人工标注与检测结果。但窗口参数改回去之后
# 总得有条路把多抽的那批收回来，否则它们照旧躺在索引里被检测，候选还是按放宽
# 后的窗口出，改参数等于没改。清单（frames_dense_<组>.json）本来就是为这件事
# 写的（见 densify 模块 docstring：「想退回直接照着删」），这里把它接上。
#
# 两条硬约束：
#   · **只删清单里列过的文件**——别的帧（首轮抽的、上上轮加密的）一张不碰；
#   · 清单里但**当前窗口仍然要的**那些也不删——那不是「多抽的」。
# ---------------------------------------------------------------------------

def _dense_setup(tmp_path, monkeypatch):
    """跑一轮加密，返回 (segments, root, index, manifest)。"""
    from pipeline.densify import densify
    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368)]
    root = tmp_path / "frames"
    index = tmp_path / "frames_index.json"
    manifest = tmp_path / "frames_dense.json"
    densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
            frames_root=root, index_path=index, video_duration=6000.0,
            manifest_path=manifest)
    return segs, root, index, manifest


def test_undo_densify_removes_frames_the_current_windows_no_longer_want(
        tmp_path, monkeypatch):
    from pipeline.densify import undo_densify
    from pipeline.windows import Window

    segs, root, index, manifest = _dense_setup(tmp_path, monkeypatch)
    before = sorted(p for p in root.rglob("*.jpg"))
    assert before, "夹具前提：加密确实抽出了帧"

    # 「当前窗口」= 一个什么都不覆盖的空窗口 → 这一轮加的全都该收回
    report = undo_densify(manifest, root, index, windows=[
        Window(3, "entry", 0.0, 0.0, 0.1)])
    after = sorted(p for p in root.rglob("*.jpg"))
    assert report["n_deleted"] == len(before) - len(after) > 0
    assert after == [], f"清单里的帧应全部收回，还剩 {after}"
    import json as _json
    assert _json.loads(index.read_text(encoding="utf-8")) == [], "索引也要跟着清"


def test_undo_densify_keeps_frames_the_current_windows_still_want(
        tmp_path, monkeypatch):
    """清单里但当前窗口仍然要的那些不删——那不是「多抽的」。"""
    from pipeline.densify import undo_densify
    from pipeline.windows import densify_windows

    segs, root, index, manifest = _dense_setup(tmp_path, monkeypatch)
    keep = densify_windows(segs, 6000.0, [3])          # 与抽的时候同一批窗口
    report = undo_densify(manifest, root, index, windows=keep)
    assert report["n_deleted"] == 0, (
        f"窗口没变就一张都不该删，实得删了 {report['n_deleted']} 张")
    assert list(root.rglob("*.jpg")), "帧还在"


def test_undo_densify_never_touches_frames_outside_the_manifest(
        tmp_path, monkeypatch):
    """首轮抽的帧、上上轮加密的帧——清单里没有的，一张都不碰，
    哪怕它落在当前窗口之外。"""
    from pipeline.densify import undo_densify
    from pipeline.windows import Window

    segs, root, index, manifest = _dense_setup(tmp_path, monkeypatch)
    stranger = root / "03" / "entry" / "000001.000.jpg"
    stranger.parent.mkdir(parents=True, exist_ok=True)
    stranger.write_bytes(b"not mine")
    undo_densify(manifest, root, index, windows=[Window(3, "entry", 0.0, 0.0, 0.1)])
    assert stranger.exists(), "清单外的帧被删了——这条绝不允许"


def test_undo_densify_refuses_without_a_manifest(tmp_path, monkeypatch):
    """没有清单就无从知道哪些是这一轮加的，必须硬失败、一张都不删。"""
    from pipeline.densify import undo_densify
    segs, root, index, manifest = _dense_setup(tmp_path, monkeypatch)
    n = len(list(root.rglob("*.jpg")))
    with pytest.raises(RuntimeError, match="清单"):
        undo_densify(tmp_path / "不存在.json", root, index, windows=[])
    assert len(list(root.rglob("*.jpg"))) == n, "失败时不许动任何文件"


def test_densify_cli_undo_reports_and_points_at_the_next_step(tmp_path, monkeypatch, capsys):
    """`--undo` 要说清删了多少、保留多少，并明说必须重跑检测。

    不说的话用户会以为参数改回去就完事了，而候选仍按旧窗口出——命令看起来
    跑成功了，页面一点没变。
    """
    import json as _json
    from pipeline import densify as _d
    from pipeline.schedule import save_segments

    segs, root, index, manifest = _dense_setup(tmp_path, monkeypatch)
    work = tmp_path / "work"
    work.mkdir()
    save_segments(segs, work / "segments.json")
    (work / "frames").mkdir()
    for f in root.rglob("*.jpg"):
        tgt = work / "frames" / f.relative_to(root)
        tgt.parent.mkdir(parents=True, exist_ok=True)
        tgt.write_bytes(f.read_bytes())
    data = _json.loads(manifest.read_text(encoding="utf-8"))
    data["added"] = [str(work / "frames" / Path(a).relative_to(root))
                     for a in data["added"]]
    mp = _d.dense_manifest_path(work, "组")
    mp.write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (work / "frames_index.json").write_text(index.read_text(encoding="utf-8"),
                                            encoding="utf-8")

    monkeypatch.setattr(_d, "probe_duration", lambda _v: 6000.0)
    rc = _d.main(["--undo", "--work", str(work), "--sheet", "组",
                  "--video", str(tmp_path / "fake.mp4"), "--duration", "6000"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "已退回" in out and "一张未动" in out, out
    assert "--stages 3,4" in out, "必须明说要重跑检测"


# --- 指定间隔重抽（用户 2026-09-08：「对于无可用画面的 order13/14/16 重新抽，
#     只抽这三段」）。按当前参数重抽这三段一帧都不会多——进出场早已是 0.1 秒
#     一帧，采样点与盘上的帧逐个重合。命令会「跑成功」而什么都没做，正是本
#     项目最忌讳的静默无操作，所以要能显式指定更密的间隔。 -------------------

def test_densify_accepts_an_explicit_interval(tmp_path, monkeypatch):
    from pipeline.densify import densify
    from pipeline.windows import densify_windows, Window

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368)]
    base = [w for w in densify_windows(segs, 6000.0, [3]) if w.kind == "entry"][0]
    tighter = [Window(base.seg_order, base.kind, base.t_start, base.t_end, 1.0 / 15)]
    root = tmp_path / "frames"
    rep = densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
                  frames_root=root, index_path=tmp_path / "frames_index.json",
                  video_duration=6000.0, windows=tighter)
    n_at_15fps = rep["n_added"]
    assert n_at_15fps > 0
    # 同一个窗口按 0.1 秒抽，点数应当明显少于 1/15 秒
    looser = [Window(base.seg_order, base.kind, base.t_start, base.t_end, 0.1)]
    root2 = tmp_path / "frames2"
    rep2 = densify(video=tmp_path / "fake.mp4", segments=segs, orders=[3],
                   frames_root=root2, index_path=tmp_path / "frames_index2.json",
                   video_duration=6000.0, windows=looser)
    assert n_at_15fps > rep2["n_added"], (
        f"1/15 秒该比 0.1 秒抽出更多帧，实得 {n_at_15fps} vs {rep2['n_added']}")


def test_densify_cli_rejects_an_interval_off_the_first_pass_grid(tmp_path, monkeypatch, capsys):
    """`--interval` 必须是首轮网格（0.2 秒）的整数等分。

    不是的话整条采样网格会平移几毫秒，盘上已有的帧一张都对不上，同一段时间
    被抽第二遍、落下一堆近重复帧（这正是 2026-09-08 放宽窗口时踩过的坑）。
    """
    from pipeline import densify as _d
    from pipeline.schedule import save_segments

    segs = [_seg(3, 2497, 3368)]
    work = tmp_path / "work"; work.mkdir()
    save_segments(segs, work / "segments.json")
    monkeypatch.setattr(_d, "probe_duration", lambda _v: 6000.0)
    rc = _d.main(["--orders", "3", "--interval", "0.07", "--dry-run",
                  "--work", str(work), "--sheet", "组",
                  "--video", str(tmp_path / "fake.mp4"), "--duration", "6000"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "整数等分" in out, out


def test_densify_cli_actually_extracts_at_the_explicit_interval(tmp_path, monkeypatch, capsys):
    """`--interval` 必须真的作用到**抽帧**上，不只是打印出来。

    2026-09-08 实测踩过：CLI 把窗口算好、打印成「0.200s → 0.040s」，可
    `densify()` 那一步没接 `windows=`，它自己又按目标帧数重算了一遍——于是
    「新增 0 帧」，命令看起来跑完了、一帧都没多。当时那条测试只断言了打印
    内容，正好从这个缝里漏过去。所以这里断言的是**盘上真的多出了那些帧**。
    """
    from pipeline import densify as _d
    from pipeline.schedule import save_segments

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368)]
    work = tmp_path / "work"; work.mkdir()
    save_segments(segs, work / "segments.json")
    monkeypatch.setattr(_d, "probe_duration", lambda _v: 6000.0)

    def run(interval):
        for d in (work / "frames",):
            if d.exists():
                import shutil; shutil.rmtree(d)
        idx = work / "frames_index.json"
        if idx.exists():
            idx.unlink()
        rc = _d.main(["--orders", "3", "--interval", str(interval),
                      "--work", str(work), "--sheet", "组",
                      "--video", str(tmp_path / "fake.mp4"), "--duration", "6000"])
        assert rc == 0, capsys.readouterr().out
        capsys.readouterr()
        return sum(1 for _ in (work / "frames").rglob("*.jpg"))

    n_loose = run(0.1)
    n_tight = run(0.05)
    assert n_tight > n_loose, (
        f"0.05 秒一帧该比 0.1 秒抽出更多帧，实得 {n_tight} vs {n_loose}"
        "——多半是 `densify()` 又自己重算了一遍窗口")


def test_densify_uses_the_recorded_timestamp_for_a_no_candidate_seat(tmp_path, monkeypatch, capsys):
    """「无可用画面」的座位旁边填了重抽时刻时，加密要**对着那个时刻**抽。

    用户 2026-09-09 的用意：他眼睛已经看到那个人在第几分钟出现了，只是那一带
    没被抽到。预设的进出场窗口锚在段首段尾，罩不住段中间的时刻——不接这条线索，
    填了等于白填。

    窗口一律由 `windows.spot_window` 给出：它会把区间裁进那个时刻**所属的那一
    段**，时刻不在任何段里就抛异常、不猜最近的一段。
    """
    import json as _json
    from pipeline import densify as _d
    from pipeline.schedule import save_segments

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368)]
    work = tmp_path / "work"; work.mkdir()
    save_segments(segs, work / "segments.json")
    labels = {"_meta": {"detections_fingerprint": "x", "sheet": "组", "n_candidates": 1},
              "no_candidate_seats": {"3": [2]},
              "no_candidate_notes": {"3": {"2": {"at": "50:00", "text": ""}}}}
    from pipeline import paths as _p
    lp = _p.labels_path(work, "组")
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(_json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(_d, "probe_duration", lambda _v: 6000.0)
    rc = _d.main(["--work", str(work), "--sheet", "组",
                  "--video", str(tmp_path / "fake.mp4"), "--duration", "6000"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "3000" in out, f"应打印出解析到的秒数（50:00 = 3000 秒）：{out}"
    ts = sorted(float(p.stem) for p in (work / "frames").rglob("*.jpg"))
    assert any(abs(t - 3000.0) <= 12.0 for t in ts), (
        f"应在 3000 秒附近抽出帧，实得范围 {ts[:3]} … {ts[-3:] if ts else ''}")


def test_densify_reports_an_unparsable_timestamp_instead_of_guessing(tmp_path, monkeypatch, capsys):
    """填了看不懂的时刻要如实说，不许猜一个数。

    按错的时刻抽出来的是别家企业的画面，而它会被写进本段目录，此后衣着/颜色/
    性别全部取自隔壁公司，一处不报错。
    """
    import json as _json
    from pipeline import densify as _d
    from pipeline.schedule import save_segments
    from pipeline import paths as _p

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2497, 3368)]
    work = tmp_path / "work"; work.mkdir()
    save_segments(segs, work / "segments.json")
    lp = _p.labels_path(work, "组")
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(_json.dumps({
        "_meta": {"detections_fingerprint": "x", "sheet": "组", "n_candidates": 1},
        "no_candidate_seats": {"3": [2]},
        "no_candidate_notes": {"3": {"2": {"at": "大概五十分钟", "text": ""}}},
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(_d, "probe_duration", lambda _v: 6000.0)
    rc = _d.main(["--work", str(work), "--sheet", "组",
                  "--video", str(tmp_path / "fake.mp4"), "--duration", "6000"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "大概五十分钟" in out and "看不懂" in out, (
        f"必须把看不懂的原文原样报出来：{out}")


def test_densify_accepts_a_timestamp_in_the_gap_between_segments(tmp_path, monkeypatch, capsys):
    """填的重抽时刻落在两段之间的空隙里要照抽——进出场本来就发生在那里。

    用户 2026-09-09 实测城市1 order 16：填的是 5:08:32 = 18512 秒，而该段讲到
    18506.8 秒就结束了。按「时刻必须落在段内」那条规则会被直接拒掉，而那 5 秒
    之差正是人起身往外走的时候。
    """
    import json as _json
    from pipeline import densify as _d, paths as _p
    from pipeline.schedule import save_segments

    _fake_extract(monkeypatch)
    segs = [_seg(15, 16609, 17735), _seg(16, 17779, 18507), _seg(17, 18558, 19248)]
    work = tmp_path / "work"; work.mkdir()
    save_segments(segs, work / "segments.json")
    lp = _p.labels_path(work, "组")
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(_json.dumps({
        "_meta": {"detections_fingerprint": "x", "sheet": "组", "n_candidates": 1},
        "no_candidate_seats": {"16": [1]},
        "no_candidate_notes": {"16": {"1": {"at": "5:08:32", "text": ""}}},
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(_d, "probe_duration", lambda _v: 19300.0)
    rc = _d.main(["--work", str(work), "--sheet", "组",
                  "--video", str(tmp_path / "fake.mp4"), "--duration", "19300"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "18512" in out, f"应打印解析到的秒数：{out}"
    ts = sorted(float(p.stem) for p in (work / "frames" / "16").rglob("*.jpg"))
    assert any(abs(t - 18512.0) <= 10.5 for t in ts), (
        f"应在 18512 秒附近抽出帧，实得 {ts[-5:] if ts else '（无）'}")


def test_probe_fps_parses_the_rational_frame_rate(monkeypatch):
    """ffprobe 的 r_frame_rate 是分数（"50/1"、"30000/1001"），要按分数算。"""
    from pipeline import frames as _f

    class _Out:
        def __init__(self, s): self.stdout = s
    for text, want in (("50/1\n", 50.0), ("30000/1001\n", 29.97002997002997),
                       ("25/1", 25.0)):
        monkeypatch.setattr(_f.subprocess, "run", lambda *a, **k: _Out(text))
        assert _f.probe_fps(Path("x.mp4")) == pytest.approx(want)
    for junk in ("", "0/0\n", "N/A\n", "abc\n"):
        monkeypatch.setattr(_f.subprocess, "run", lambda *a, **k: _Out(junk))
        assert _f.probe_fps(Path("x.mp4")) is None, junk


def test_spot_interval_never_goes_finer_than_the_source_frame_rate():
    """指名时刻处抽到**源片的每一帧**，但不超过它——再密只是把同一张解码画面
    写成两个文件，白占磁盘也白跑一遍检测。取值仍落在首轮 0.2 秒网格的整数
    等分上，好与该段已有的帧对齐。"""
    from pipeline.windows import spot_interval_for_fps
    assert spot_interval_for_fps(50.0) == pytest.approx(0.02)   # 0.2 / 10
    assert spot_interval_for_fps(25.0) == pytest.approx(0.04)   # 0.2 / 5
    assert spot_interval_for_fps(29.97) == pytest.approx(0.04)  # 0.2 / 5
    assert spot_interval_for_fps(None) == pytest.approx(1.0 / 15)
    for fps in (50.0, 25.0, 29.97, 10.0, None):
        iv = spot_interval_for_fps(fps)
        n = (1.0 / config.TRANSITION_FPS) / iv
        assert abs(n - round(n)) < 1e-9, f"{fps} 算出的 {iv} 不在 0.2 秒网格上"
        if fps:
            assert iv >= 1.0 / fps - 1e-9, (
                f"{fps} fps 的源片，间隔 {iv} 比一帧还细——只会把同一张解码"
                f"画面写成两个文件")


def test_densify_opens_a_window_for_every_moment_the_user_listed(tmp_path, monkeypatch, capsys):
    """一个座位填了多个时刻，每一个都要开窗。

    用户 2026-09-09 给城市1 order 4 座位 1 填的是 `1:01:12,1:24:21`——进场那
    一下和退场那一下。只认第一个（或整条判为看不懂）的话，他填的另一个白填。
    """
    import json as _json
    from pipeline import densify as _d, paths as _p
    from pipeline.schedule import save_segments

    _fake_extract(monkeypatch)
    segs = [_seg(3, 2595, 3645), _seg(4, 3705, 5047), _seg(5, 5095, 6252)]
    work = tmp_path / "work"; work.mkdir()
    save_segments(segs, work / "segments.json")
    lp = _p.labels_path(work, "组")
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(_json.dumps({
        "_meta": {"detections_fingerprint": "x", "sheet": "组", "n_candidates": 1},
        "no_candidate_seats": {"4": [1]},
        "no_candidate_notes": {"4": {"1": {"at": "1:01:12,1:24:21", "text": ""}}},
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(_d, "probe_duration", lambda _v: 19300.0)
    rc = _d.main(["--work", str(work), "--sheet", "组",
                  "--video", str(tmp_path / "fake.mp4"), "--duration", "19300"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "3672.0" in out and "5061.0" in out, f"两个时刻都要开窗：{out}"
    ts = sorted(float(p.stem) for p in (work / "frames" / "04").rglob("*.jpg"))
    assert any(abs(t - 3672.0) <= 10.5 for t in ts), "进场那一下没抽到"
    assert any(abs(t - 5061.0) <= 10.5 for t in ts), "退场那一下没抽到"


# --- 抽帧提速：批量 + 并发（用户 2026-09-18）-----------------------------
#
# 起点是用户指出 `scripts/probe_mouth.py:97` 用过 `-ss` + `-t` + `-vf fps=`，
# 一个进程连抽一整段（2026-08-15 真跑过，产物是 work/文化4-4.28/mouth_probe/
# frames/ 的 200 张），速度快很多。
#
# 第一轮我否决了它，理由是「抽出来不是同一张」——真实视频上与逐帧抽字节相同的
# 只有 16/30。**那个理由是错的**：后来量了漂移，40 秒窗口 2000 张里连第 1999 张
# 都能对上，偏移恒为 0 或 +1 源帧，没有累积。所谓「对不上」是恒定 ±1 帧的偏移
# 加上源片有重复帧（avg_frame_rate 49.98），按序号硬比当然不合。
#
# 实测下来偏移是 **-3 到 0 个源帧**（0.06 秒以内），且**不是常数**——同一个窗口
# 里有的 -1、有的 -2。原因不在 `fps=` 滤镜（去掉它一样是 -1/-2），在 `-ss` 本身：
# `-ss t -frames:v 1` 取的是「t 之后的第一帧」，每个时刻各自向上取整一次；批量
# 是从 seek 落点起连着往下数。源片 avg_frame_rate 49.98、含重复帧，两条规则因此
# 各有各的取整。
#
# 为什么仍然可用：批量这一串**本身是均匀的**（相邻就是相邻源帧），反倒比逐帧
# 那条规则更规整；文件名对内容的描述误差在一个源帧（0.02 秒）量级，而下游没有
# 任何一步依赖到这个精度——窗口越界校验按秒算，候选的 t_mid 跨重检测本就带容差
# 匹配，加密抽帧的网格对齐看的是**文件名**而文件名完全一致。
#
# 断言因此写成「与逐帧抽的 t-3p … t+p 之一逐字节相同」（p = 源片帧周期），
# **不是**「与 t 那一张相同」——后者在真实视频上必假。第一轮就是用合成的 testsrc
# 跑出了 30/30 的假绿：那种片子每帧都是关键帧、帧率恒定，正好绕开了这两个条件。
#
# 批量只用在**进出场窗口**。就座窗口跨一千多秒、只取一百来张，批量得把整段解完
# ——实测 30.2 秒，而逐帧 8 并发只要 4.2 秒，慢 7 倍。`_batch_fps` 要求
# 1/interval 为正整数，就座那些间隔（5.5/9.478/10/20 秒）自然被挡在外面。


def test_batch_only_applies_to_dense_short_windows():
    """`_batch_fps`：进出场那几档间隔走批量，就座的一律退回逐帧。

    退不回去的代价实测过：1067 秒跨度取 107 张，批量 30.2 秒、逐帧并发 4.2 秒。
    """
    from pipeline.frames import _batch_fps
    for interval, want in [(0.2, 5), (0.1, 10), (0.05, 20), (0.04, 25), (0.02, 50)]:
        assert _batch_fps(interval) == want, interval
    for interval in (5.5, 9.478, 10.0, 20.0, 0.0, -1.0):
        assert _batch_fps(interval) is None, interval


def test_batch_produces_exactly_the_grid_filenames(tmp_path):
    """文件名必须与逐帧抽产出的一模一样——它就是时间戳，下游每一步都信它。"""
    from pipeline import frames as fr

    video = tmp_path / "v.mp4"
    _make_video(video, video_seconds=4, audio_seconds=4, rate=50)
    w = _ladder_window(tmp_path, 0.02)
    root = tmp_path / "out"
    refs = fr.extract_window(video, w, root)
    want = window_timestamps(w)
    assert [r.t for r in refs] == want, "返回次序必须按时间戳"
    on_disk = sorted(q.name for q in root.rglob("*.jpg"))
    assert on_disk == sorted(f"{t:010.3f}.jpg" for t in want)


@pytest.mark.skipif(not config.VIDEO.exists(), reason="本机没有这组视频")
def test_batch_frames_match_the_per_frame_rule_within_one_source_frame(tmp_path):
    """**真实视频**上：批量抽的每一张，都必须是逐帧抽在 t±一个源帧处的那一张。

    这是这次改造唯一的等价性承诺。用合成片子测不出来——第一轮 testsrc 跑出
    30/30 全绿，换真实视频只剩 16/30。
    """
    from pipeline import frames as fr

    fps = fr.probe_fps(config.VIDEO)
    assert fps, "取不到源片帧率，这条测试的前提不成立"
    period = 1.0 / fps

    w = Window(seg_order=5, kind="entry", t_start=5060.213,
               t_end=5060.213 + 29 * 0.02, interval=0.02)
    got = fr.extract_window(config.VIDEO, w, tmp_path / "batch")
    assert len(got) == len(window_timestamps(w))

    ref = tmp_path / "ref"
    ref.mkdir()
    for r in got:
        cand = []
        for d in (-3, -2, -1, 0, 1):
            q = ref / f"{r.t:.3f}_{d}.jpg"
            if fr._extract_single_frame(config.VIDEO, r.t + d * period, q):
                cand.append(q.read_bytes())
        assert r.path.read_bytes() in cand, (
            f"{r.t:.3f} 这一张离逐帧抽超过 3 个源帧——网格漂了，"
            f"不是已知的那个取整差异")


@pytest.mark.skipif(not config.VIDEO.exists(), reason="本机没有这组视频")
def test_batch_captures_as_much_distinct_content_as_going_frame_by_frame(tmp_path):
    """批量与逐帧取到的**独立画面数必须一样**——批量不许少拿也不许多编。

    注意这里不能断言「没有重复画面」：这组源片标称 50fps，实际只有 25fps 的
    独立内容，每一帧都成对出现（实测全片五处采样，独立画面率 50%-51%，最长连续
    重复 3 张）。两条规则都会如实带出这些重复，那是视频本身的性质，不是抽帧的
    毛病。真正要守的是「批量没把内容弄丢」。
    """
    from pipeline import frames as fr
    from pipeline.windows import window_timestamps as _wt

    w = Window(seg_order=5, kind="entry", t_start=5060.200,
               t_end=5060.200 + 29 * 0.02, interval=0.02)
    got = fr.extract_window(config.VIDEO, w, tmp_path / "b")
    one = tmp_path / "one"
    for t in _wt(w):
        fr._extract_single_frame(config.VIDEO, t, fr._frame_path(one, w, t))

    n_batch = len({r.path.read_bytes() for r in got})
    n_one = len({fr._frame_path(one, w, t).read_bytes() for t in _wt(w)})
    assert n_batch == n_one, (n_batch, n_one)


def test_a_sparse_window_goes_frame_by_frame(tmp_path, monkeypatch):
    """就座窗口不许走批量——走了反而慢 7 倍。"""
    from pipeline import frames as fr

    video = tmp_path / "v.mp4"
    _make_video(video, video_seconds=4, audio_seconds=4, rate=50)
    w = Window(seg_order=4, kind="seated", t_start=0.5, t_end=3.5, interval=0.37)

    called = []
    monkeypatch.setattr(fr, "_extract_batch",
                        lambda *a, **k: called.append(1) or set())
    refs = fr.extract_window(video, w, tmp_path / "s")
    assert not called, "就座窗口走了批量"
    assert len(refs) == len(window_timestamps(w))


def test_batch_shortfall_falls_back_to_single_frames(tmp_path, monkeypatch):
    """批量少吐了几张时，缺的那几张逐帧补上——不按序号硬对。

    硬对的后果是把别的时刻的画面存成这个时刻的文件名，而一处不报错。
    """
    from pipeline import frames as fr

    video = tmp_path / "v.mp4"
    _make_video(video, video_seconds=4, audio_seconds=4, rate=50)
    w = _ladder_window(tmp_path, 0.02)
    real = fr._extract_batch
    monkeypatch.setattr(fr, "_extract_batch",
                        lambda video, w, grid, fps, want, root:
                        real(video, w, grid[:5], fps, want, root))
    refs = fr.extract_window(video, w, tmp_path / "o")
    assert [r.t for r in refs] == window_timestamps(w)
    for r in refs:
        assert r.path.is_file() and r.path.stat().st_size > 0, r.t


# --- 并发抽帧（用户 2026-09-18）-------------------------------------------
#
# 起因是用户问「你是用 GPU 还是 CPU 抽帧的」，以及「我之前真的加了个参数，速度
# 快很多」——指 `scripts/probe_mouth.py:97` 的 `-ss` + `-t` + `-vf fps=`，一个
# 进程连抽一整段（2026-08-15 真跑过，产物是 `work/文化4-4.28/mouth_probe/frames/`
# 的 200 张）。
#
# 那条路**试了并且否决了**：在真实视频上它抽出来的不是同一张。实测本组 30 帧一个
# 窗口，与逐帧抽字节相同的只有 16/30；起点不落在源片帧边界上时只有 2/30（源片
# avg_frame_rate 49.98，非恒定；`fps=` 按第一帧解码后的 PTS 重新布点，与「`-ss t`
# 取 t 之后的第一帧」对不上，偏一个源帧就是完全不同的画面）。文件名就是时间戳，
# 下游每一步都信这条对应关系——名字说 5060.220、里面装别的时刻的画面，是本项目
# 最忌讳的那类失效，快 12 倍也不换。
#
# 改成并发：**同一条命令**，只是同时跑几条，字节必然一致（实测 60/60），4.2×。
#
# 下面第三条是防回潮的：谁要是哪天又想用 `-vf fps=` 批量，它会立刻红。


def _ladder_window(tmp_path, interval):
    return Window(seg_order=3, kind="entry", t_start=1.0,
                  t_end=1.0 + 19 * interval, interval=interval)


def test_parallel_extraction_matches_serial_byte_for_byte(tmp_path, monkeypatch):
    """并发与串行：产出的**文件名与字节**必须一张不差。"""
    from pipeline import frames as fr
    from pipeline.windows import window_timestamps

    video = tmp_path / "v.mp4"
    _make_video(video, video_seconds=4, audio_seconds=4, rate=50)
    w = _ladder_window(tmp_path, 0.02)

    one = tmp_path / "one"
    monkeypatch.setattr(fr, "FRAME_EXTRACT_WORKERS", 1)
    fr.extract_window(video, w, one)

    many = tmp_path / "many"
    monkeypatch.setattr(fr, "FRAME_EXTRACT_WORKERS", 8)
    got = fr.extract_window(video, w, many)

    a = sorted(p.relative_to(one) for p in one.rglob("*.jpg"))
    b = sorted(p.relative_to(many) for p in many.rglob("*.jpg"))
    assert a == b, (a, b)
    assert [r.t for r in got] == window_timestamps(w), "返回次序必须按时间戳"
    for rel in a:
        assert (one / rel).read_bytes() == (many / rel).read_bytes(), rel


def test_parallel_extraction_still_skips_frames_already_on_disk(tmp_path):
    """已有的帧一张不许重写——盘上那些帧挂着检测结果与人工标注。"""
    from pipeline import frames as fr

    video = tmp_path / "v.mp4"
    _make_video(video, video_seconds=4, audio_seconds=4, rate=50)
    w = _ladder_window(tmp_path, 0.02)
    root = tmp_path / "out"
    fr.extract_window(video, w, root)
    keep = sorted(root.rglob("*.jpg"))[3]
    keep.write_bytes(b"SENTINEL")
    stamps = {p: p.stat().st_mtime_ns for p in root.rglob("*.jpg")}

    refs = fr.extract_window(video, w, root)
    assert keep.read_bytes() == b"SENTINEL", "已存在的帧被重抽覆盖了"
    assert len(refs) == len(stamps)
    for q, ns in stamps.items():
        assert q.stat().st_mtime_ns == ns, q


def test_batch_output_plan_stops_at_the_first_gap():
    """批量少吐一张时，从那一张起**全部交回逐帧**，不跳过去继续按序号对。

    ffmpeg 的 `%06d` 按实际写出的张数编号，中间少一张，后面每一张的序号都前移
    一位。跳过缺号继续对，等于把后面每一张都存成早一格的时刻——而文件打得开、
    名字也对得上，一处不报错。
    """
    from pipeline.frames import batch_output_plan
    grid = [10.0, 10.2, 10.4, 10.6, 10.8]
    want = set(grid)
    assert batch_output_plan({1, 2, 3, 4, 5}, grid, want) == [
        (1, 10.0), (2, 10.2), (3, 10.4), (4, 10.6), (5, 10.8)]
    # 第 3 张没吐出来：第 4、5 张**不许**拿去填 10.6/10.8
    assert batch_output_plan({1, 2, 4, 5}, grid, want) == [(1, 10.0), (2, 10.2)]
    assert batch_output_plan(set(), grid, want) == []


def test_batch_output_plan_leaves_existing_frames_alone():
    """盘上已经有的时刻不搬——那些帧挂着检测结果与人工标注。"""
    from pipeline.frames import batch_output_plan
    grid = [10.0, 10.2, 10.4]
    assert batch_output_plan({1, 2, 3}, grid, {10.4}) == [(3, 10.4)]


# --- 源片的「独立内容帧率」（用户 2026-09-18）------------------------------
#
# `probe_fps` 给的是容器标称的 `r_frame_rate`。这组视频标称 50fps，**实际只有
# 25fps 的独立内容**——每一帧都成对出现（全片五处采样，独立画面率 50%-51%，
# 最长连续重复 3 张）。按标称帧率算加密间隔就得到 0.02 秒，于是抽出来一半是
# 重复画面：2026-09-18 那次加密抽了 16000 张，约 8000 张重复，而检测把它们全跑
# 了一遍——41 分钟里有一半白烧。
#
# 所以指名时刻的间隔要按**独立内容**算，不按标称。判据用「连续相同画面的游程
# 长度」：成对出现就是游程 2（帧翻倍），而静止画面会给出很长的游程——两者必须
# 分开，否则一段没人动的画面会把整组的间隔拉粗。做法是多处采样取**最小**游程
# （最保守），再钳在 4 以内。


def _dup_video(path, rate, dup, seconds=2):
    """造一个「标称 rate、实际 rate/dup」的片子：先按 rate/dup 生成，再翻倍到 rate。

    **必须无损编码**（`-crf 0`）。有损的话 x264 会把成对的那两张编得略有差别，
    解出来的 JPEG 字节就不相同，夹具造不出重复帧——第一版正是这样，50 张全都
    「独立」。真实视频里的重复帧是解码后逐字节相同的，无损编码才复现得出来。
    """
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size=320x240:rate={rate // dup}",
         "-vf", f"fps={rate}", "-c:v", "libx264", "-crf", "0",
         "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-loglevel", "error", str(path)],
        check=True,
    )


def test_content_fps_equals_the_nominal_rate_when_every_frame_is_new(tmp_path):
    """每帧都是新画面时，独立内容帧率 = 标称帧率。"""
    from pipeline.frames import content_fps
    v = tmp_path / "a.mp4"
    _dup_video(v, rate=50, dup=1)
    assert content_fps(v) == pytest.approx(50, rel=0.15)


def test_content_fps_halves_when_every_frame_is_doubled(tmp_path):
    """每帧成对出现时，独立内容帧率是标称的一半——这组视频就是这样。"""
    from pipeline.frames import content_fps
    v = tmp_path / "b.mp4"
    _dup_video(v, rate=50, dup=2)
    assert content_fps(v) == pytest.approx(25, rel=0.15)


def test_content_fps_does_not_collapse_on_a_still_scene(tmp_path):
    """整段静止的画面**不许**把帧率压到零——那会让间隔粗得没法用。

    静止只说明这一处没人动，不说明源片是低帧率的。压缩比例钳在 4 以内。
    """
    from pipeline.frames import content_fps
    v = tmp_path / "c.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=blue:size=320x240:d=2:r=50",
         "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-loglevel", "error", str(v)],
        check=True,
    )
    got = content_fps(v)
    assert got is not None and got >= 50 / 4 - 1e-6, got


def test_content_fps_is_none_when_the_rate_is_unknown(tmp_path):
    """取不到帧率就返回 None，**不猜**——调用方据此退回保守的缺省间隔。"""
    from pipeline.frames import content_fps
    assert content_fps(tmp_path / "nope.mp4") is None


def test_the_spot_interval_follows_the_content_rate():
    """指名时刻的间隔按独立内容帧率算：50fps 标称、25fps 实际 → 0.04 秒。"""
    from pipeline.windows import spot_interval_for_fps
    assert spot_interval_for_fps(50) == pytest.approx(0.02)
    assert spot_interval_for_fps(25) == pytest.approx(0.04)


def test_one_still_stretch_does_not_coarsen_the_whole_video(tmp_path):
    """片中有一段没人动，**不许**把整片的间隔拉粗。

    静止只说明那一处没人动，不说明源片是低帧率的。所以多处采样取**最小**游程
    （最保守），而不是最大或平均——取最大的话，一段静止画面就会让全组的加密
    间隔粗上几倍，而那几倍里丢掉的正是别处有人走动的帧。
    """
    from pipeline.frames import content_fps

    still = tmp_path / "still.mp4"
    moving = tmp_path / "moving.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=blue:size=320x240:d=3:r=50",
         "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-loglevel", "error", str(still)], check=True)
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi",
         "-i", "testsrc=duration=3:size=320x240:rate=50",
         "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-loglevel", "error", str(moving)], check=True)
    lst = tmp_path / "l.txt"
    lst.write_text(f"file '{still}'\nfile '{moving}'\n", encoding="utf-8")
    mixed = tmp_path / "mixed.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", "-loglevel", "error", str(mixed)], check=True)

    got = content_fps(mixed)
    assert got == pytest.approx(50, rel=0.15), (
        f"静止的那一段把帧率拉到了 {got}——该按有人动的那几处算")
