"""阶段 ②：按窗口调 ffmpeg 抽帧。输入前 seek，不解码全片。"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pipeline.windows import Window, window_timestamps

# 绝对路径而不是裸 `ffmpeg`：本机 PATH 里可能有多个版本，写死能保证跑的是
# 哪一个。但**必须可覆盖**——这两个默认值是本机 Homebrew 的位置，Linux
# 服务器上根本不存在，阶段 2 一开始就失败（用户 2026-08-29 要把流水线搬上
# 服务器，这是第一道硬门槛）。
#
# 服务器上：`export DRESS_FFMPEG=/usr/bin/ffmpeg DRESS_FFPROBE=/usr/bin/ffprobe`
# 不设时行为与此前一字不差。
FFMPEG = os.environ.get("DRESS_FFMPEG", "/opt/homebrew/bin/ffmpeg")
FFPROBE = os.environ.get("DRESS_FFPROBE", "/opt/homebrew/bin/ffprobe")

# 单窗口内取不到帧的比例超过此值即视为系统性问题（路径错、编码不支持等），
# 而非个别边界点的偶发失败——此时抛异常，好过跑一个多小时拿到残缺结果。
FRAME_FAILURE_RATE_THRESHOLD = 0.05


@dataclass(frozen=True)
class FrameRef:
    seg_order: int
    kind: str
    t: float
    path: Path


def _parse_duration(text: str) -> float | None:
    text = text.strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _probe_video_stream_duration(video: Path) -> float | None:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return _parse_duration(out.stdout)


def _probe_format_duration(video: Path) -> float | None:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True,
    )
    return _parse_duration(out.stdout)


def probe_duration(video: Path) -> float:
    """优先读视频流时长。容器（format）时长取音视频两流之最大值——若音频比
    视频长（实测「文化4-4月28日视频」音频多出 0.088 秒），format=duration
    会虚高，导致 compute_windows 算出的窗口上界越过最后一帧，抽帧在越界处
    ffmpeg 非零退出。视频流时长缺失（部分容器为 N/A）时回退到 format=duration；
    两者都拿不到就抛异常——不猜、不返回 0。
    """
    video_dur = _probe_video_stream_duration(video)
    if video_dur is not None:
        return video_dur
    print(f"⚠ probe_duration: 视频流时长缺失，回退到容器时长 format=duration（{video}）",
          flush=True)
    fmt_dur = _probe_format_duration(video)
    if fmt_dur is not None:
        return fmt_dur
    raise RuntimeError(f"probe_duration: 无法读取视频时长（视频流与容器时长均缺失）：{video}")


def probe_fps(video: Path) -> float | None:
    """源片帧率。取不到返回 None，**不猜一个默认值**。

    用途只有一个：指名时刻处加密时，抽到源片的每一帧就够了，再密只是把同一张
    解码画面写成两个文件（白占磁盘、白跑一遍检测）。ffprobe 给的是分数
    （"50/1"、"30000/1001"），必须按分数算，取整会把 29.97 变成 30。
    """
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    text = (out.stdout or "").strip()
    if not text or "/" not in text:
        return None
    num, _, den = text.partition("/")
    try:
        n, d = float(num), float(den)
    except ValueError:
        return None
    if d == 0 or n <= 0:
        return None
    return n / d


# 独立内容帧率的探测参数（用户 2026-09-18）。
CONTENT_FPS_PROBE_SPOTS = 5      # 全片均匀取几处
CONTENT_FPS_PROBE_SECONDS = 1.0  # 每处解多少秒
CONTENT_FPS_MAX_REDUCTION = 4    # 最多把帧率压到 1/4，防静止画面把间隔拉粗


def _run_lengths(blobs: list[bytes]) -> list[int]:
    """连续相同画面的游程长度。"""
    out: list[int] = []
    n = 1
    for i in range(1, len(blobs)):
        if blobs[i] == blobs[i - 1]:
            n += 1
        else:
            out.append(n)
            n = 1
    out.append(n)
    return out


def content_fps(video: Path) -> float | None:
    """源片**独立内容**的帧率。取不到返回 None（不猜）。

    `probe_fps` 给的是容器标称的 `r_frame_rate`。这组视频标称 50fps，实际只有
    25fps 的独立内容——每一帧都成对出现（全片五处采样，独立画面率 50%-51%）。
    按标称算加密间隔就得 0.02 秒，抽出来一半是重复画面：2026-09-18 那次加密抽了
    16000 张、约 8000 张重复，检测把它们全跑了一遍，41 分钟白烧一半。

    判据是**连续相同画面的游程长度**，不是独立画面占比：帧翻倍的游程恒为 2，而
    一段没人动的画面会给出很长的游程。两者必须分开，否则一处静止就把整组的间隔
    拉粗。做法是全片均匀取几处、各算中位游程，**取其中最小的**（最保守，一处静止
    不影响别处），再把压缩比钳在 `CONTENT_FPS_MAX_REDUCTION` 以内。
    """
    import hashlib
    import shutil
    import tempfile

    fps = probe_fps(video)
    if not fps or fps <= 0:
        return None
    try:
        dur = probe_duration(video)
    except (RuntimeError, OSError):
        return None
    if not dur or dur <= 0:
        return None

    spots = [dur * (i + 1) / (CONTENT_FPS_PROBE_SPOTS + 1)
             for i in range(CONTENT_FPS_PROBE_SPOTS)]
    runs: list[float] = []
    tmp = Path(tempfile.mkdtemp(prefix="dress_cfps_"))
    try:
        for k, t0 in enumerate(spots):
            d = tmp / str(k)
            d.mkdir()
            r = subprocess.run(
                [FFMPEG, "-nostdin", "-v", "error", "-ss", f"{t0:.3f}",
                 "-i", str(video), "-t", f"{CONTENT_FPS_PROBE_SECONDS:.3f}",
                 "-q:v", "2", str(d / "%05d.jpg"), "-y"],
                capture_output=True,
            )
            if r.returncode != 0:
                continue
            blobs = [q.read_bytes() for q in sorted(d.glob("*.jpg"))]
            if len(blobs) < 4:
                continue
            rl = sorted(_run_lengths(blobs))
            runs.append(rl[len(rl) // 2])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not runs:
        return float(fps)
    factor = max(1, min(int(min(runs)), CONTENT_FPS_MAX_REDUCTION))
    return float(fps) / factor


def _frame_path(out_root: Path, w: Window, t: float) -> Path:
    return out_root / f"{w.seg_order:02d}" / w.kind / f"{t:010.3f}.jpg"


def _extract_single_frame(video: Path, t: float, p: Path) -> bool:
    """抽取单帧。成功返回 True；该时间点取不到画面（越界、变帧率空洞、
    损坏帧等）时 ffmpeg 非零退出，返回 False——调用方负责记录并跳过。
    不重试：取不到就是取不到，重试只会拖慢一个多小时的批处理。
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error",
         "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "2", str(p), "-y"],
        capture_output=True,
    )
    if result.returncode != 0:
        if p.exists():
            p.unlink()
        return False
    return True


# 并发抽帧（用户 2026-09-18：「抽帧速度能不能快一点」）。
#
# 一帧一个 ffmpeg 进程，每张都要起进程、`-ss` 跳到最近关键帧、再往前解到目标帧；
# 0.02 秒一帧时相邻帧几乎都落在同一个 GOP 里，同一段画面被反复解几十遍。单张的
# 成本改不动，但这些调用**彼此独立**，并发跑即可。
#
# 为什么不是批量（`-ss` + `-t` + `-vf fps=`，即 `scripts/probe_mouth.py:97` 那份，
# 实测快 12-14×）：在真实视频上**它抽出来的不是同一张**。实测本组 30 帧一个窗口，
# 与逐帧抽字节相同的只有 16/30；起点不落在源片帧边界上时只有 2/30。源片
# avg_frame_rate 是 49.98（非恒定），`fps=` 滤镜按第一帧解码后的 PTS 重新布点，
# 和「`-ss t` 取 t 之后的第一帧」这条规则对不上，偏一个源帧就是完全不同的画面。
#
# 文件名就是时间戳，而下游每一步都信这条对应关系。名字说 5060.220、里面装的是
# 别的时刻的画面——这是本项目最忌讳的那类失效，快 12 倍也不换。并发则是**同一条
# 命令**，只是同时跑几条，字节必然一致（实测 60/60）。
#
# 8 并发的取法：本机 10 个逻辑核（4 个性能核），实测 4→3.0×、8→3.9×、12→4.2×，
# 8 之后收益已经很薄，留出余量给别的活。
FRAME_EXTRACT_WORKERS = 8

# 批量抽帧：`-ss` + `-t` + `-vf fps=`，一个进程把整段连着解一遍。做法来自
# `scripts/probe_mouth.py:97`（2026-08-15 真跑过，产物是 work/文化4-4.28/
# mouth_probe/frames/ 的 200 张），是用户 2026-09-18 指出来的。
#
# 与逐帧抽的关系：**同一条网格上的同一帧，偏移恒为 0 或 +1 个源帧**。实测 40 秒
# 窗口 2000 张，连第 1999 张都对得上，没有累积漂移；所谓「对不上」是这个恒定
# ±1 帧偏移加上源片有重复帧（avg_frame_rate 49.98）。而 ±1 源帧 = 0.02 秒，
# **正是采样分辨率本身**，所以批量抽出来的就是这条网格上的那一帧。
#
# 速度（40 秒窗口）：2000 张 3.1 秒、200 张 1.3 秒，而逐帧 8 并发分别要 78 秒和
# 8 秒。批量的成本看**要解多长的视频**，不看张数——就座窗口跨一千多秒只取一百
# 来张，批量反而慢 7 倍（30.2 秒 vs 4.2 秒），所以只给进出场用。
BATCH_MIN_FRAMES = 8          # 少于这么多张时逐帧更省事（批量要多一个临时目录）
BATCH_MAX_SPAN = 120.0        # 跨度超过这么多秒就别批量了：解码成本按秒算
_BATCH_FPS_EPS = 1e-6


def _batch_fps(interval: float) -> int | None:
    """`1/interval` 是正整数就返回它，否则 None（该窗口退回逐帧）。

    批量走 `-vf fps=N`，第 k 张对应 `t_start + (k-1)/N`。只有 1/interval 为整数
    时这条网格才和 `window_timestamps` 对得齐。进出场窗口的间隔一律落在首轮的
    阶梯上（0.2/0.1/0.05/0.04/0.02 → 5/10/20/25/50）；就座窗口的间隔是「目标帧数
    ÷ 窗口时长」算出来的任意小数（实测 5.5/9.478/10/20 秒），1/interval < 1，
    自然被挡在外面——而那正是批量该避开的那一类（跨度长、取样稀）。
    """
    if interval <= 0:
        return None
    n = 1.0 / interval
    r = round(n)
    if r < 1 or abs(n - r) > _BATCH_FPS_EPS:
        return None
    return int(r)


def batch_output_plan(present: set[int], grid: list[float],
                      want: set[float]) -> list[tuple[int, float]]:
    """批量产出的第 i 张该存成哪个时刻：`[(序号, 时刻), ...]`。

    ffmpeg 的 `%06d` 是**按实际写出的张数顺序编号**的，中间少吐一张，后面每一张
    的序号就整体前移一位。所以一旦发现缺号就**到此为止**，剩下的交回逐帧补——
    跳过缺号继续按序号对，会把后面每一张都存成早一格的时刻，而文件打得开、名字
    也对得上，一处不报错。这正是「名字说 5060.220、里面装别的时刻的画面」那类
    失效，本项目最忌讳的一种。

    `want` 之外的时刻不搬（那些帧盘上已经有了，一张都不能碰）。
    """
    out: list[tuple[int, float]] = []
    for i, t in enumerate(grid, start=1):
        if i not in present:
            break
        if t in want:
            out.append((i, t))
    return out


def _extract_batch(video: Path, w: Window, grid: list[float], fps: int,
                   want: set[float], out_root: Path) -> set[float]:
    """一个 ffmpeg 进程抽完整条网格，只把 `want` 里那几个时刻落盘。

    先抽进临时目录再挑着搬，是为了**一张已有的帧都不碰**——盘上那些帧挂着检测
    结果与人工标注，`extract_window` 的「已存在则跳过」这条不能因为换了实现就破。

    返回真正落盘的时刻集合。ffmpeg 少吐了几张（末尾边界、损坏帧）时只搬拿得到的
    那些，其余交回调用方逐帧补——**不按序号硬对**，对错了就是把别的时刻的画面
    存成这个时刻的文件名，而一处不报错。
    """
    import shutil
    import tempfile

    span = (len(grid) - 1) * w.interval + w.interval / 2
    tmp = Path(tempfile.mkdtemp(prefix="dress_batch_"))
    try:
        result = subprocess.run(
            [FFMPEG, "-nostdin", "-v", "error",
             "-ss", f"{w.t_start:.3f}", "-t", f"{span:.3f}",
             "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2",
             str(tmp / "%06d.jpg"), "-y"],
            capture_output=True,
        )
        if result.returncode != 0:
            return set()
        present = {i for i in range(1, len(grid) + 1)
                   if (tmp / f"{i:06d}.jpg").is_file()}
        done: set[float] = set()
        for i, t in batch_output_plan(present, grid, want):
            dst = _frame_path(out_root, w, t)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp / f"{i:06d}.jpg"), str(dst))
            done.add(t)
        return done
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def extract_window(video: Path, w: Window, out_root: Path) -> list[FrameRef]:
    """抽取窗口内所有采样点。已存在的帧跳过，故可安全重跑。

    个别时间点可能因浮点边界、可变帧率、损坏帧等原因取不到画面——单点失败
    只记录并跳过，不中断整批。但失败率超过 FRAME_FAILURE_RATE_THRESHOLD
    说明是系统性问题（路径错、编码不支持等），此时抛异常，好过悄悄跑完
    一个多小时后交出一个残缺的索引。失败的时间点不会出现在返回值里。
    """
    timestamps = window_timestamps(w)
    todo = [(t, _frame_path(out_root, w, t)) for t in timestamps]
    missing = [(t, p) for t, p in todo if not p.exists()]

    # 先试一次批量（见 `_batch_fps` / `_extract_batch`）。批量抽不到的下面逐帧
    # 补，所以这一层只影响速度，不影响产出的完整性。
    fps = _batch_fps(w.interval)
    span = w.t_end - w.t_start
    if (fps is not None and len(missing) >= BATCH_MIN_FRAMES
            and span <= BATCH_MAX_SPAN):
        n = int(span // w.interval)
        grid = [round(w.t_start + k * w.interval, 3) for k in range(n + 1)]
        _extract_batch(video, w, grid, fps, {t for t, _ in missing}, out_root)
        missing = [(t, p) for t, p in missing if not p.exists()]

    # 并发只改「同时跑几条 ffmpeg」，命令本身一字未动，所以抽出来的字节与串行
    # 必然一致（见上面 FRAME_EXTRACT_WORKERS 那段）。结果按时间戳次序重新组装，
    # 不依赖完成次序——依赖了的话 frames_index.json 的条目次序会随机器负载变，
    # 而它是检测分组的输入。
    ok: dict[float, bool] = {}
    if missing:
        n = min(FRAME_EXTRACT_WORKERS, len(missing))
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(_extract_single_frame, video, t, p): t
                       for t, p in missing}
            for fut in as_completed(futures):
                ok[futures[fut]] = fut.result()

    refs: list[FrameRef] = []
    failed: list[float] = []
    for t, p in todo:
        if t in ok and not ok[t]:
            failed.append(t)
            continue
        refs.append(FrameRef(w.seg_order, w.kind, t, p))

    if failed:
        total = len(timestamps)
        rate = len(failed) / total
        preview = ", ".join(f"{t:.3f}" for t in failed[:5])
        if len(failed) > 5:
            preview += " ..."
        print(f"⚠ 抽帧失败 {len(failed)}/{total} 帧（已跳过）：{preview}", flush=True)
        if rate > FRAME_FAILURE_RATE_THRESHOLD:
            raise RuntimeError(
                f"抽帧失败率过高：{len(failed)}/{total} ({rate:.1%}) "
                f"超过阈值 {FRAME_FAILURE_RATE_THRESHOLD:.0%}，"
                f"疑似系统性问题（路径/编码/权限），中止 order={w.seg_order} kind={w.kind}"
            )

    return refs


def extract_all(video: Path, windows: list[Window], out_root: Path) -> list[FrameRef]:
    refs: list[FrameRef] = []
    for i, w in enumerate(windows, 1):
        refs.extend(extract_window(video, w, out_root))
        print(f"[{i}/{len(windows)}] order={w.seg_order} {w.kind} 累计 {len(refs)} 帧", flush=True)
    return refs


def save_index(frames: list[FrameRef], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"seg_order": f.seg_order, "kind": f.kind, "t": f.t, "path": str(f.path)}
        for f in frames
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_index(path: Path) -> list[FrameRef]:
    return [
        FrameRef(d["seg_order"], d["kind"], d["t"], Path(d["path"]))
        for d in json.loads(path.read_text(encoding="utf-8"))
    ]


def merge_index(existing: list[FrameRef], new: list[FrameRef]) -> list[FrameRef]:
    """把新抽的帧并进已有的帧索引，去重后按 (段号, 窗口, 时间) 排序。

    为什么必须并进**同一份** `frames_index.json`（Task 56C）：阶段 3 读的
    就是它（`detect_frames(load_index(idx_path))`），不并进去，新帧躺在盘上
    永远不会被检测到——命令看起来跑成功了，候选一个没多。

    去重键是 (段号, 窗口, 时间点)，不是路径：同一个采样点的路径由
    `_frame_path` 唯一决定，但索引是可以手工/多次拼接的，同一帧出现两条会让
    阶段 3 对同一张图跑两次跟踪，凭空多出一条重复的 track。

    **只增不减**：已有的条目一条都不动。别的段的帧上挂着用户已经做完的人工
    标注，删一条就是让那份标注失效，而这在产出物里看不出来。
    """
    out: list[FrameRef] = list(existing)
    seen = {(r.seg_order, r.kind, round(r.t, 3)) for r in existing}
    for r in new:
        key = (r.seg_order, r.kind, round(r.t, 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return sorted(out, key=lambda r: (r.seg_order, r.kind, r.t))


def segment_frame_mismatches(segments, frames_root) -> list[dict]:
    """比对盘上的帧与时间表，找出错配或缺失的段。

    帧文件名即时间戳（`{t:010.3f}.jpg`），所以只要看它是否落在该段的
    [start, end] 内就能判断这批帧到底属不属于这家企业——几秒钟的事。

    为什么必须查（用户 2026-08-22 发现的真实事故）：文化4组的 Book1 从
    13 家变成 14 家（补了一家原先缺时间戳的），新增那家排在第 8 位，其后
    所有企业的行号后移一位。而帧是按旧编号抽的、一直存在盘上没重抽——
    `frames/08/` 里装的仍是旧编号 8（即现在的第 9 家）的画面。于是
    seg 8–13 的衣着、颜色、性别判定全部取自隔壁公司。

    整条流水线照常跑完、表照常生成、图照常出，**没有任何一处报错**，
    是用户肉眼看出「华韵的图是中科直线的」才发现的。这类错配不会自己
    暴露，只能靠这种显式比对。

    容差取 `SEATED_HEAD_MARGIN`/`SEATED_TAIL_MARGIN` 之外再放 5 秒：抽帧
    本就跳过段首尾若干秒，帧的实际范围必然窄于区间，只需确认它没有越界。

    返回 [{"order", "company", "reason", "detail"}]，一切正常时返回 []。
    """
    from pathlib import Path

    root = Path(frames_root)
    bad = []
    for seg in segments:
        d = root / f"{seg.order:02d}" / "seated"
        fs = sorted(d.glob("*.jpg")) if d.is_dir() else []
        if not fs:
            bad.append({"order": seg.order, "company": seg.company,
                        "reason": "no_frames",
                        "detail": f"{d} 下没有任何帧"})
            continue
        ts = []
        for f in fs:
            try:
                ts.append(float(f.stem))
            except ValueError:
                continue
        if not ts:
            bad.append({"order": seg.order, "company": seg.company,
                        "reason": "unparsable",
                        "detail": f"{d} 下的文件名无法解析为时间戳"})
            continue
        t0, t1 = min(ts), max(ts)
        if t0 < seg.start - 5 or t1 > seg.end + 5:
            bad.append({
                "order": seg.order, "company": seg.company,
                "reason": "out_of_window",
                "detail": (f"帧范围 {t0:.0f}-{t1:.0f} 越出该段区间 "
                           f"{seg.start}-{seg.end}，疑似企业行序变动后未重抽帧"),
            })
    return bad
