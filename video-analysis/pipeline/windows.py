"""阶段 ② 的纯逻辑：窗口区间计算与采样间隔自适应。不做任何 IO。"""
from __future__ import annotations

from dataclasses import dataclass

from pipeline import config
from pipeline.schedule import Segment


@dataclass(frozen=True)
class Window:
    seg_order: int
    kind: str          # "entry" | "exit" | "seated"
    t_start: float
    t_end: float
    interval: float    # 相邻采样点的秒间隔


def seated_interval(start: int, end: int) -> float:
    """就座窗口的采样间隔。

    目标是取到约 SEATED_TARGET_FRAMES 帧；长段会被上限钳住以免帧数爆炸，
    短段（如 240 秒的 order 11）会被下限钳住以保证样本量。
    """
    usable = (end - config.SEATED_TAIL_MARGIN) - (start + config.SEATED_HEAD_MARGIN)
    if usable <= 0:
        return float(config.SEATED_INTERVAL_MIN)
    raw = usable / config.SEATED_TARGET_FRAMES
    return float(min(max(raw, config.SEATED_INTERVAL_MIN), config.SEATED_INTERVAL_MAX))


def _seated_bounds(seg: Segment) -> tuple[float, float]:
    lo = seg.start + config.SEATED_HEAD_MARGIN
    hi = seg.end - config.SEATED_TAIL_MARGIN
    if hi <= lo:
        # 段太短，退回使用整段中间 80% 以保证仍有样本
        span = seg.end - seg.start
        lo = seg.start + span * 0.1
        hi = seg.end - span * 0.1
    return float(lo), float(hi)


def compute_windows(segments: list[Segment], video_duration: float) -> list[Window]:
    """为每段产出 entry / exit / seated 三个窗口。

    相邻段空隙最小仅 12 秒，entry 与 exit 会互相侵入，故按邻段边界裁剪。
    重叠区的帧仍会同时落入两个窗口，人物归属在阶段 ④ 按移动方向判定。

    进出场窗口的**锚点**优先取人工补录的实际时刻（`seg.entry_time` /
    `seg.exit_time`，来自 dta 的 entryTime / exitTime），没有才退回按段首/
    段尾加固定偏移猜。猜是有代价的：城市精细化管理1组 order 11 的人在段首前
    130 秒就进场了，`ENTRY_WINDOW = (-25, 15)` 猜出来的窗口与真正的进场差
    105 秒，一帧都没覆盖到。就座窗口不受影响，它本就锚在段首段尾。
    """
    by_order = {s.order: s for s in segments}
    orders = sorted(by_order)
    out: list[Window] = []

    for i, o in enumerate(orders):
        seg = by_order[o]
        prev_end = by_order[orders[i - 1]].end if i > 0 else 0.0
        next_start = by_order[orders[i + 1]].start if i + 1 < len(orders) else video_duration

        # 锚点：人工补录了实际时刻就用它，否则退回按段首/段尾猜。
        # **裁剪照旧**——换锚点不等于可以越过邻段边界，越过就是把上一家企业
        # 还在讲的画面抽进本段的 frames/NN/entry/ 目录。
        e_anchor, e_off = ((seg.entry_time, config.ENTRY_MARK_WINDOW)
                           if seg.entry_time is not None
                           else (seg.start, config.ENTRY_WINDOW))
        e_lo = max(e_anchor + e_off[0], prev_end, 0.0)
        e_hi = min(e_anchor + e_off[1], seg.end, video_duration)
        out.append(Window(o, "entry", float(e_lo), float(e_hi), 1.0 / config.TRANSITION_FPS))

        x_anchor, x_off = ((seg.exit_time, config.EXIT_MARK_WINDOW)
                           if seg.exit_time is not None
                           else (seg.end, config.EXIT_WINDOW))
        x_lo = max(x_anchor + x_off[0], seg.start, 0.0)
        x_hi = min(x_anchor + x_off[1], next_start, video_duration)
        out.append(Window(o, "exit", float(x_lo), float(x_hi), 1.0 / config.TRANSITION_FPS))

        s_lo, s_hi = _seated_bounds(seg)
        out.append(Window(o, "seated", s_lo, s_hi, seated_interval(seg.start, seg.end)))

    return out


def window_timestamps(w: Window) -> list[float]:
    """窗口内的采样时间点，含首尾。

    点数用整除取下界而非四舍五入——四舍五入会让末点越过 t_end，
    侵入窗口边界（就座窗口的尾部保留区正是为避开起身动作而设）。
    末点显式取 t_end，因此最后一段间隔可能短于 interval，
    这对采样网格无害。
    """
    if w.t_end <= w.t_start:
        return [w.t_start]
    n = int((w.t_end - w.t_start) // w.interval)
    pts = [round(w.t_start + k * w.interval, 3) for k in range(n + 1)]
    if pts[-1] < w.t_end - 1e-6:
        pts.append(round(w.t_end, 3))
    return pts


# ---------------------------------------------------------------------------
# Task 56C · 三：二次加密抽帧的窗口
#
# 用户 2026-08-24：「如果某个座位的人没有合适的候选图，能标记下来，再抽帧
# 二次有更多的备选吗」。这里只算窗口与间隔（纯逻辑、无 IO），真正抽帧与
# 落盘在 `pipeline/densify.py`。
# ---------------------------------------------------------------------------

def densify_interval(t_start: float, t_end: float, kind: str) -> float:
    """加密抽帧的采样间隔（秒）。

    取法与 `seated_interval` 逐字同一条路子：**目标帧数 ÷ 实际窗口时长**，
    再由上下限钳住。因此每段按自己的窗口长短自动调整——用户此前明确要求
    「所有的固定参数改为根据每组的情况自动调整」，写死一个「0.05 秒一帧」
    在 40 秒的进场窗口和 900 秒的就座窗口上意味着完全不同的两件事。

    `kind` 只分两档：
      · entry / exit —— 密。用户原话：「坐下的阶段可以抽稀疏一点，而进出
        一定要密集抽」。人是在进退场时站起来、正脸朝镜头的，找不到的那个人
        十有八九只在这几秒里露过脸。
      · seated —— 稀。整段几乎是同一个姿势，多抽只是把同一张画面抽 N 遍。

    退化窗口（首尾相等或倒置，实际存在——相邻段空隙最小仅 12 秒，裁剪后
    entry 窗口可能被压没）返回下限而不是 0：返回 0 会让 `window_timestamps`
    的整除直接抛 ZeroDivisionError，把整轮加密卡在半路。
    """
    if kind == "seated":
        target = config.DENSIFY_SEATED_TARGET_FRAMES
        lo, hi = (config.DENSIFY_SEATED_INTERVAL_MIN,
                  config.DENSIFY_SEATED_INTERVAL_MAX)
    else:
        target = config.DENSIFY_TRANSITION_TARGET_FRAMES
        lo, hi = (config.DENSIFY_TRANSITION_INTERVAL_MIN,
                  config.DENSIFY_TRANSITION_INTERVAL_MAX)
    span = t_end - t_start
    if span <= 0:
        return float(lo)
    raw = min(max(span / target, lo), hi)
    if kind == "seated":
        # 就座那一路不套下面的对齐规则：它的窗口不随进出场时刻变动，没有反复
        # 重抽的问题；改动它只会把已有的就座帧网格挪掉，凭空多抽一遍。
        return float(raw)
    # 进出场：把间隔对齐成首轮网格（1/TRANSITION_FPS = 0.2 秒）的**整数等分**。
    #
    # 不对齐的后果实测过（城市1，窗口放宽到前后各 30 秒之后）：窗口一被邻段
    # 裁掉几秒，`span / 目标帧数` 就落在 0.0953 这类数上，整条采样网格随之平移
    # 几毫秒——盘上已有的帧一张都对不上，同一段时间被抽第二遍，落下一堆相差
    # 5 毫秒的近重复帧。5 段被裁，每段因此要多抽 400 余帧：白占磁盘、白跑一遍
    # 检测、还在候选里多出一批几乎一样的图。
    #
    # 对齐之后，加密的帧与首轮的帧、以及上一轮加密的帧落在同一张网格上，
    # 重跑只补真正缺的那些。窗口起点也天然对齐：它由 `compute_windows` 从
    # 秒整数的时刻加整数偏移算出，两点之差恒为间隔的整数倍。
    base = 1.0 / config.TRANSITION_FPS
    n = max(1, round(base / raw))
    return float(min(max(base / n, lo), hi))


def spot_window(segments: list[Segment], at: float, span: float = 10.0,
                kind: str = "exit") -> Window:
    """按**用户指名的时刻**给出一个加密窗口。

    预设的 entry/exit 窗口锚在段首/段尾（`ENTRY_WINDOW` / `EXIT_WINDOW`），
    还会被相邻段截断。用户 2026-08-25 实测文化4 order 8：那个人在 **126 分**
    （7560s）就离场了，而该段名义到 128.1 分才结束，退场窗口被下一段起点截到
    只剩 7673-7683 这 10 秒——整段退场窗口都没覆盖到他，那一带只有 seated 的
    2.5 秒一张，比退场窗口的 0.067 秒稀 37 倍。

    预设窗口覆盖不到时，用户得能直接说「就抽这一带」。这是个逃生口，不是
    默认路径：默认仍按 `densify_windows` 走预设窗口。

    窗口**一律裁进 `at` 所属的那一段**。越界的帧会让下游把隔壁企业的画面当成
    这一段的——那正是 `frames.segment_frame_mismatches` 存在的理由。`at` 不在
    任何一段内就抛 `ValueError`，**不猜最近的那一段**：猜错就是把一批帧塞进
    错的企业名下，而整条流水线照常跑完、一处不报错。
    """
    hit = next((s for s in segments if s.start <= at <= s.end), None)
    if hit is None:
        rng = "、".join(f"order{s.order}:{s.start:.0f}-{s.end:.0f}s"
                        for s in segments)
        raise ValueError(
            f"时刻 {at:.1f}s 不在任何一段的时间范围内，无从判断它属于哪家企业"
            f"（各段：{rng}）")
    t0 = max(float(hit.start), at - span)
    t1 = min(float(hit.end), at + span)
    return Window(hit.order, kind, t0, t1, densify_interval(t0, t1, kind))


def spot_interval_for_fps(fps: float | None) -> float:
    """指名时刻处的采样间隔：**源片的每一帧，但不超过它**。

    用户指名了一个时刻，就是要那一带看得尽量全，所以这里不套「目标帧数 ÷ 窗口
    时长」那条自适应——窗口只有 ±10 秒，抽满也不过一千来帧。但比源片帧率更细
    没有意义：ffmpeg 只会把同一张解码画面写成两个文件，白占磁盘也白跑一遍检测。

    取值一律落在首轮网格 `1/TRANSITION_FPS` 的整数等分上（0.2 / n），这样它与
    该段已有的帧对齐、只补缺的那些；不对齐会落下一堆相差几毫秒的近重复帧。
    取不到帧率时退回 `DENSIFY_TRANSITION_INTERVAL_MIN`（保守，不猜源片有多快）。
    """
    base = 1.0 / config.TRANSITION_FPS
    if not fps or fps <= 0:
        return float(config.DENSIFY_TRANSITION_INTERVAL_MIN)
    n = int(base * fps)                      # 0.2 秒里有几帧，向下取整
    return float(base / max(1, n))


def spot_window_for_segment(segments: list[Segment], seg_order: int, at: float,
                            span: float = 10.0, kind: str = "exit",
                            interval: float | None = None) -> Window:
    """按**指名的那一段**在时刻 `at` 附近开一个加密窗口。

    与 `spot_window` 的分工：那一个只给了时刻、要靠「落在哪一段之内」反推段号，
    所以时刻必须在某段之内，否则宁可抛异常也不猜最近的一段。这一个的段号是
    调用方明确给的（来自「第 N 段座位 M 无可用画面」那条标注），不存在猜的
    问题——于是**时刻落在两段之间的空隙里是合法的**。

    这一点是必须的：用户 2026-09-09 实测城市1 order 16，他填的重抽时刻
    5:08:32 = 18512 秒，而该段讲到 18506.8 秒就结束了。那 5 秒之差正是人起身
    往外走的时候——**进出场本来就发生在段与段之间的空隙里**，要求时刻落在段内
    等于把这个功能最常见的用法挡在门外。

    真正要守的仍然是「别抽到隔壁企业」：区间按 [上一段结束, 下一段开始] 裁，
    与 `compute_windows` 同一条裁法；`at` 本身越过这个范围（落进别家企业的
    时间里）就抛 `ValueError`。
    """
    by_order = {int(s.order): s for s in segments}
    seg = by_order.get(int(seg_order))
    if seg is None:
        raise ValueError(f"时间表里没有第 {seg_order} 段，无从据此开窗")
    orders = sorted(by_order)
    i = orders.index(int(seg_order))
    lo = float(by_order[orders[i - 1]].end) if i > 0 else 0.0
    hi = (float(by_order[orders[i + 1]].start) if i + 1 < len(orders)
          else float("inf"))
    if not (lo <= at <= hi):
        raise ValueError(
            f"时刻 {at:.1f}s 不在第 {seg_order} 段可用的范围 "
            f"[{lo:.1f}, {hi:.1f}] 内（那是别家企业的时间），"
            f"按它抽出来的帧会落进本段目录却拍的是别人，且一处不报错")
    t0, t1 = max(lo, at - span), min(hi, at + span)
    return Window(int(seg_order), kind, float(t0), float(t1),
                  float(interval) if interval else densify_interval(t0, t1, kind))


def densify_windows(segments: list[Segment], video_duration: float,
                    orders) -> list[Window]:
    """点名的那几段的加密窗口（entry / exit / seated 各一个）。

    **区间与 `compute_windows` 逐字相同，只有 interval 变密**——这是本函数
    唯一允许与它不同的地方。区间另算一份的后果本项目付过代价：帧一旦落在
    别家企业的时间窗里，下游的衣着/颜色/性别全部取自隔壁公司，而整条流水线
    照常跑完、没有任何一处报错（见 `frames.segment_frame_mismatches`）。
    因此这里**直接调用 `compute_windows` 再筛**，不重抄一遍边界裁剪。

    `segments` 必须传**整组**（不是筛过的那几段）：相邻段的边界要用来裁剪
    entry/exit 窗口（相邻段空隙最小仅 12 秒）。只把目标段传进来的话，
    `compute_windows` 会拿 0 与 video_duration 当邻居，进场窗口一路侵入
    上一家企业——正是上面那类事故的成因。筛选发生在算完之后。
    """
    want = {int(o) for o in orders}
    out: list[Window] = []
    for w in compute_windows(segments, video_duration):
        if w.seg_order not in want:
            continue
        out.append(Window(w.seg_order, w.kind, w.t_start, w.t_end,
                          densify_interval(w.t_start, w.t_end, w.kind)))
    return out
