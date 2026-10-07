"""阶段 ②b：对「有座位找不到合适候选」的那几段二次加密抽帧（Task 56C）。

用户 2026-08-24 原话：

    「如果某个座位的人没有合适的候选图，能标记下来，再抽帧二次有更多的
    备选吗」

前半句在挑选页（`pipeline/entryreview.py`）与标注文件的顶层通道
`no_candidate_seats`（`pipeline/entryexit.py`）里落地；本模块是后半句。

    挑选页勾「本座位无可用画面」 → 导出 entry_labels_<组>.json
        → **本模块只对那几段加密抽帧** → 重跑阶段 3/4 → 挑选页候选变多

设计上的四条硬约束，每一条都来自本项目栽过的跟头：

1. **只动点名的那几段。** 不整组重跑（一次一个多小时），更不删别的段已有
   的帧——那些帧上挂着用户已经做完的人工标注与检测结果。本模块**只增不删**，
   连本段原有的帧也一张不动（`extract_window` 遇到已存在的文件直接跳过）。

2. **新帧必须落在本段自己的时间窗内。** 窗口一律由
   `windows.densify_windows` 给出，而它直接复用 `compute_windows` 的区间、
   只把 interval 改密——区间另算一份正是「帧与企业错位」那类事故的成因
   （文化4组企业行序后移一位，第 8-13 段编码的是隔壁公司的衣着，整条流水线
   照常跑完、没有任何一处报错）。抽完还要再过一遍
   `frames.segment_frame_mismatches` 硬校验，越界就抛异常。

3. **加密的帧与首轮的帧合并在同一个目录**，不另建「dense/」之类的位置。
   理由是阶段 3 按 `(seg_order, kind)` 分组跟踪
   （`detect.detect_frames`）：换一个 kind 或换一个目录，就变成了另一个
   跟踪窗口、拿到另一套 track_id，`entryexit_candidates` 只看 entry/exit
   两个 kind，新帧一个候选也变不出来——那等于这个功能白做。
   「新旧混在一起无法分辨」这条顾虑改由**另写一份清单**
   （`frames_dense_<组>.json`）解决：这一轮加了哪些文件逐条记下来，想退回
   直接照着删；而帧文件名本身就是时间戳，它属于哪一段一目了然。

4. **抽完必须告诉用户下一步跑什么。** 新帧要重跑检测才会变成新候选；
   不说清楚，用户会以为这条命令没起作用。

本模块**不跑检测、不调 VLM、不写 work 以外的任何东西**。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline import config, paths
from pipeline.frames import (
    extract_window, load_index, merge_index, probe_duration, save_index,
    segment_frame_mismatches,
)
from pipeline.windows import Window, densify_windows, window_timestamps


def dense_manifest_path(work: Path, sheet: str) -> Path:
    """这一轮加密抽了哪些帧的清单 `work/frames_dense_<组>.json`。

    文件名带组名，与 `entry_labels_<组>.json` / `seat_registry_<组>.json`
    一贯的命名相同——同一个 work 目录下先后跑过不同的组时不会互相覆盖。
    """
    return Path(work) / f"frames_dense_{config.sanitize_filename_component(sheet)}.json"


def seats_needing_frames(labels_path: Path) -> dict[int, list[int]]:
    """从标注文件读出 `{段号: [被标为无可用画面的座位号]}`。

    **直接读原始 JSON，不走 `load_entry_labels`**，这一点是刻意的：
    后者在检测指纹不符时会把整份标注作废（对 track 派生的标识而言那是对的
    ——重检测后同号十有八九是另一个人）。而这里要的键是 `(order, seat)`，
    座位号由人工确定、跨重检测稳定，本就不该受指纹影响。

    走作废那条路的后果很具体：加密抽帧 → 重跑检测（指纹必变）→ 第二轮想
    再加密时读不到任何目标，命令看起来跑成功了却什么都没做。

    文件不存在 / 不是对象 / 没有这个键，一律返回 `{}`（调用方据此提示用户
    先去挑选页标记），不抛异常。
    """
    p = Path(labels_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}

    from pipeline.entryexit import no_candidate_seats_of

    raw = data.get("no_candidate_seats") or {}
    out: dict[int, list[int]] = {}
    for order_key in raw:
        try:
            order = int(order_key)
        except (TypeError, ValueError):
            continue
        # 清洗（脏值、0、负数）只有 `no_candidate_seats_of` 这一份口径，
        # 页面、登记表、这里三处共用；各写一份必然日后走样。
        seats = sorted(no_candidate_seats_of(data, order))
        if seats:
            out[order] = seats
    return out


def densify(video: Path, segments, orders, frames_root: Path,
            index_path: Path, video_duration: float | None = None,
            manifest_path: Path | None = None, windows=None) -> dict:
    """对 `orders` 里那几段加密抽帧，返回统计。

    `segments` 要传**整组**——相邻段的边界用于裁剪 entry/exit 窗口，
    见 `windows.densify_windows`。筛选发生在算完窗口之后。

    抽帧之前先查一遍这几段盘上**已有的**帧：若它们本身就落在别的时间窗里
    （企业行序变动后没重抽），直接抛异常、一帧都不抽。往一个已经装着别家
    企业画面的目录里再补帧，只会让两家的画面混在一起，此后再也分不开——
    这时该做的是整段重抽（阶段 2），不是加密。

    抽完再查一遍：新帧必须同样落在本段时间窗内（阶段 3/4 之前那道硬失败
    校验就是查这个）。
    """
    root = Path(frames_root)
    want = sorted({int(o) for o in orders})
    targets = [s for s in segments if int(s.order) in want]
    missing = [o for o in want if o not in {int(s.order) for s in targets}]
    if missing:
        raise RuntimeError(
            f"时间表里没有这些段：{missing}。段号来自 segments.json，"
            f"对不上说明标注文件与当前时间表不是同一份，拒绝抽帧——"
            f"按错的段号抽出来的帧会落进别家企业的目录里且不报错。")

    if root.is_dir():
        pre_bad = [b for b in segment_frame_mismatches(targets, root)
                   if b["reason"] == "out_of_window"]
        if pre_bad:
            lines = "\n".join(f"    order{b['order']:>2} {b['company']}：{b['detail']}"
                              for b in pre_bad)
            raise RuntimeError(
                f"这几段盘上已有的帧就已经落在别的时间窗里（{len(pre_bad)} 段），"
                f"拒绝加密——再往同一个目录里补帧，两家企业的画面就永远分不开了：\n"
                f"{lines}\n"
                f"  修法：先重跑阶段 2 把这几段整段重抽。")

    # `windows` 给了就用它（`--at` 指名时刻的逃生口，见 CLI）；否则按预设
    # 窗口。两条路都要先过上面那道「盘上已有的帧是不是落在别家企业的时间窗
    # 里」的硬校验——那才是这里唯一不能绕开的东西。
    ws = (list(windows) if windows is not None
          else densify_windows(list(segments), float(video_duration), want))
    before = {p.resolve() for p in root.rglob("*.jpg")} if root.is_dir() else set()

    new_refs = []
    per_window = []
    for w in ws:
        refs = extract_window(Path(video), w, root)
        new_refs.extend(refs)
        per_window.append({"order": w.seg_order, "kind": w.kind,
                           "t_start": w.t_start, "t_end": w.t_end,
                           "interval": w.interval, "n_frames": len(refs)})

    after = {p.resolve() for p in root.rglob("*.jpg")}
    added = sorted(str(p) for p in (after - before))
    # 只增不减：这条断言防的是「抽帧顺手清空了目录」这一类改动。别的段的
    # 帧一旦被删，挂在它们上面的人工标注与检测结果一起失效，而产出物里
    # 看不出来。
    if before - after:
        raise RuntimeError(
            f"加密抽帧删掉了 {len(before - after)} 个已有的帧文件，"
            f"这绝不允许（已有帧上挂着人工标注与检测结果）："
            f"{sorted(str(p) for p in list(before - after)[:5])}")

    post_bad = [b for b in segment_frame_mismatches(targets, root)
                if b["reason"] == "out_of_window"]
    if post_bad:
        lines = "\n".join(f"    order{b['order']:>2} {b['company']}：{b['detail']}"
                          for b in post_bad)
        raise RuntimeError(
            f"加密抽出的帧落在本段时间窗之外（{len(post_bad)} 段）——"
            f"这正是阶段 3/4 之前那道硬校验要拦的错配，已中止：\n{lines}")

    idx = Path(index_path)
    existing = load_index(idx) if idx.exists() else []
    merged = merge_index(existing, new_refs)
    save_index(merged, idx)

    report = {
        "orders": want,
        "n_added": len(added),
        "n_index": len(merged),
        "added": added,
        "windows": per_window,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if manifest_path is not None:
        mp = Path(manifest_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    return report


def undo_densify(manifest_path: Path, frames_root: Path, index_path: Path,
                 windows) -> dict:
    """把上一轮加密多抽的帧收回来，使盘上的帧重新与**当前**窗口一致。

    用户 2026-09-08 把进出场窗口放宽到前后各 30 秒之后又要求退回：「不要用
    前后各 30s，还是恢复到之前的时间窗口，现在变乱了，重新抽帧」。光把参数
    改回去没用——多抽的那批照旧躺在 `frames_index.json` 里被阶段 3 检测，
    候选还是按放宽后的窗口出来，改参数等于没改。

    清单（`frames_dense_<组>.json`）本来就是为这件事写的（见模块 docstring：
    「想退回直接照着删」），这里把它接上。

    两条硬约束：

    1. **只删清单里列过的文件。** 首轮抽的帧、上上轮加密的帧一张不碰——
       它们上面挂着人工标注与检测结果，而这在产出物里看不出来。
    2. **清单里但当前窗口仍然要的那些也不删。** 判据是「当前窗口的采样时刻
       集合」（`window_timestamps`），不是区间：退回之后网格本身可能变密变稀，
       按区间留会留下一批对不上网格的孤帧，下次重抽既不会覆盖也不会清理。

    没有清单就硬失败、一张不动：无从知道哪些是这一轮加的，凭窗口去删就会
    连首轮的帧一起删掉。
    """
    mp = Path(manifest_path)
    if not mp.exists():
        raise RuntimeError(
            f"找不到加密清单 {mp}，无从知道上一轮加了哪些帧，拒绝删除任何文件"
            f"——只凭窗口去删会把首轮抽的帧一起删掉，而那上面挂着人工标注。")
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"加密清单 {mp} 读不出来（{exc}），拒绝删除任何文件")

    added = [Path(x) for x in (data.get("added") or [])]
    wanted = {(w.seg_order, w.kind, f"{t:010.3f}")
              for w in windows for t in window_timestamps(w)}

    def _key(p: Path):
        try:
            return int(p.parent.parent.name), p.parent.name, p.stem
        except (ValueError, IndexError):
            return None

    doomed = [p for p in added if _key(p) is not None and _key(p) not in wanted]
    deleted = []
    for p in doomed:
        if p.exists():
            p.unlink()
        deleted.append(str(p))

    idx = Path(index_path)
    n_before = n_after = 0
    if idx.exists():
        keep = []
        gone = {str(Path(d)) for d in deleted}
        for r in load_index(idx):
            n_before += 1
            if str(r.path) in gone:
                continue
            keep.append(r)
        n_after = len(keep)
        save_index(keep, idx)

    return {"n_in_manifest": len(added), "n_deleted": len(deleted),
            "n_kept": len(added) - len(deleted),
            "n_index_before": n_before, "n_index_after": n_after,
            "deleted": deleted}


def main(argv: list[str] | None = None) -> int:
    """CLI：

        python -m pipeline.densify                        # 用 config 里的组
        python -m pipeline.densify --work <dir> --sheet <名>
        python -m pipeline.densify --orders 3,7           # 手动点名
        python -m pipeline.densify --dry-run              # 只报要抽哪些，不抽

    默认目标取自 `entry_labels_<组>.json` 里人工标的「本座位无可用画面」
    ——用户不必自己记住是哪几段。
    """
    ap = argparse.ArgumentParser(
        prog="python -m pipeline.densify",
        description="对「有座位无可用候选」的片段二次加密抽帧（Task 56C）")
    ap.add_argument("--work", default=None, help="组的 work 目录（默认取 config.WORK）")
    ap.add_argument("--sheet", default=None, help="组名（默认取 config.SHEET）")
    ap.add_argument("--video", default=None, help="视频路径（默认取 config.VIDEO）")
    ap.add_argument("--orders", default=None,
                    help="手动点名要加密的段号，逗号分隔；省略则取标注文件里"
                         "标了「本座位无可用画面」的那几段")
    ap.add_argument("--duration", type=float, default=None,
                    help="视频时长秒数（默认用 ffprobe 现探）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印要对哪几段、按什么间隔抽，不真抽")
    ap.add_argument("--interval", type=float, default=None,
                    help="进出场窗口的采样间隔（秒），覆盖按目标帧数算出来的那个。"
                         "必须是首轮网格 1/TRANSITION_FPS 的整数等分（0.2 / 0.1 / "
                         "0.0667 / 0.05 / 0.04 / 0.02 ...），否则整条网格会平移、"
                         "盘上已有的帧一张都对不上。就座窗口不受影响")
    ap.add_argument("--undo", action="store_true",
                    help="按清单退回上一轮加密：删掉其中**当前窗口不再需要**的那些帧"
                         "（清单外的一张不碰）。改了窗口参数之后要用它，"
                         "否则多抽的帧照旧留在索引里被检测，改参数等于没改")
    # 逃生口：预设的 entry/exit 窗口锚在段首/段尾且被相邻段截断，覆盖不到
    # 「人提前两分钟就走了」这种情况（用户 2026-08-25 实测文化4 order 8：
    # 126 分离场，而退场窗口只剩段尾那 10 秒）。肉眼看到关键帧就能直接指名。
    ap.add_argument("--at", default=None,
                    help="按**绝对时刻**加密（秒，逗号分隔多个）。给了它就不看"
                         "标注里的「本座位无可用画面」，只抽指名的这几处")
    ap.add_argument("--spots-only", action="store_true",
                    help="只抽标注里人工填了重抽时刻的那几处（每处 ±10 秒），"
                         "不顺带加密该段的预设进出场/就座窗口。人已经指明人在哪儿"
                         "时用它：重采样动得越少，已标的东西掉得越少")
    ap.add_argument("--span", type=float, default=10.0,
                    help="--at 每侧的秒数（默认 10，即一个 20 秒的窗口）")
    ap.add_argument("--kind", default="exit", choices=("entry", "exit", "seated"),
                    help="--at 抽出来的帧算哪一类（默认 exit）。它决定这批帧"
                         "进哪个候选列表：阶段 3 按 (段号, kind) 分组跟踪")
    args = ap.parse_args(argv)

    from pipeline.schedule import load_segments_json

    work = Path(args.work) if args.work else config.WORK
    sheet = args.sheet if args.sheet else config.SHEET
    video = Path(args.video) if args.video else config.VIDEO

    seg_path = work / "segments.json"
    if not seg_path.exists():
        print(f"   ⚠ 缺少 {seg_path}（先跑阶段 1），无法加密抽帧")
        return 1
    segments = load_segments_json(seg_path)

    # --- 退回上一轮加密（改了窗口参数之后用）-----------------------------
    if args.undo:
        duration = (float(args.duration) if args.duration is not None
                    else probe_duration(video))
        keep = densify_windows(segments, duration,
                               [int(sg.order) for sg in segments])
        mp = dense_manifest_path(work, sheet)
        if args.dry_run:
            try:
                n = len(json.loads(mp.read_text(encoding="utf-8")).get("added") or [])
            except (ValueError, OSError):
                print(f"   ⚠ 读不到清单 {mp}")
                return 1
            print(f"   --dry-run：清单 {mp.name} 记着 {n} 张，"
                  f"将按当前窗口逐张判断去留（未删除）")
            return 0
        try:
            rep = undo_densify(mp, work / "frames", work / "frames_index.json", keep)
        except RuntimeError as exc:
            print(f"   ⚠ {exc}")
            return 1
        print(f"   已退回：清单 {rep['n_in_manifest']} 张里删掉 {rep['n_deleted']} 张、"
              f"保留 {rep['n_kept']} 张（当前窗口仍然要）；"
              f"frames_index.json {rep['n_index_before']} → {rep['n_index_after']} 条")
        print("   清单外的帧（首轮抽的、上上轮加密的）一张未动。")
        print("   下一步（帧变了必须重跑检测，否则候选还是旧的）：")
        print("     .venv/bin/python -m pipeline.run_all --stages 3,4")
        print("     .venv/bin/python -m pipeline.run_all --stages 5,6")
        return 0

    # --- 逃生口：--at 指名时刻，与「无可用画面」互不依赖 ------------------
    if args.at:
        from pipeline.windows import spot_window
        try:
            ats = [float(x) for x in str(args.at).split(",") if x.strip()]
        except ValueError:
            print(f"   ⚠ --at 只接受秒数（逗号分隔多个），实得 {args.at!r}")
            return 1
        try:
            spots = [spot_window(segments, a, args.span, args.kind) for a in ats]
        except ValueError as exc:
            print(f"   ⚠ {exc}")
            return 1
        by_order = {int(sg.order): sg for sg in segments}
        for w in spots:
            sg = by_order[w.seg_order]
            print(f"   第 {w.seg_order} 段 {sg.company}：{w.kind} "
                  f"{w.t_start:.1f}-{w.t_end:.1f}s（{w.t_start/60:.1f}-"
                  f"{w.t_end/60:.1f} 分），间隔 {w.interval:.3f}s")
        frames_root = work / "frames"
        index_path = work / "frames_index.json"
        if args.dry_run:
            print(f"   --dry-run：未抽帧（本应补进 {frames_root}）")
            return 0
        report = densify(video, segments, sorted({w.seg_order for w in spots}),
                         frames_root, index_path,
                         video_duration=(float(args.duration)
                                         if args.duration is not None
                                         else probe_duration(video)),
                         manifest_path=dense_manifest_path(work, sheet),
                         windows=spots)
        print(f"   新增 {report['n_added']} 帧（原有帧一张未删），"
              f"{index_path.name} 现共 {report['n_index']} 条")
        print("   下一步（新帧只有重跑检测才会变成新候选）：")
        print("     .venv/bin/python -m pipeline.run_all --stages 3,4")
        print("     .venv/bin/python -m pipeline.prepare --stages 5   # 重出挑选页")
        return 0

    labels_path = paths.labels_path(work, sheet)
    need = seats_needing_frames(labels_path)
    if args.orders:
        want = [int(x) for x in args.orders.split(",") if x.strip()]
        # 手动点名时仍把座位号带上（有就带），打印里能看到是为谁抽的。
        need = {o: need.get(o, []) for o in want}
    if not need:
        print(f"   没有任何座位被标为「无可用画面」（读的是 {labels_path.name}）"
              f"，本次不抽帧。")
        print(f"   先在挑选页 entry_review.html 里对找不到人的座位勾上"
              f"「本座位无可用画面」并导出，再跑本命令；"
              f"也可以用 --orders 手动点名。")
        return 0

    by_order = {int(s.order): s for s in segments}
    for o in sorted(need):
        seg = by_order.get(o)
        if seg is None:
            print(f"   ⚠ 时间表里没有第 {o} 段，跳过")
            continue
        seats = "、".join(f"座位 {s}" for s in need[o]) or "（手动点名，未指明座位）"
        print(f"   第 {o} 段 {seg.company}（{seg.start}-{seg.end}s）：{seats}")
    orders = [o for o in sorted(need) if o in by_order]
    if not orders:
        print("   ⚠ 没有一个目标段能在时间表里对上，本次不抽帧")
        return 1

    duration = (float(args.duration) if args.duration is not None
                else probe_duration(video))
    # `--spots-only`（用户 2026-09-26 电子10 order 6 座位 4）：人既然填了重抽
    # 时刻，就只抽那几处；预设窗口一个不动。默认路径会顺带把整段的就座窗口
    # 从 20 秒一帧加密到 10 秒一帧，重检测之后这一段的几何基线可能重新分座，
    # 他已经标好的座位跟着挪——重采样的代价与动了多少成正比。
    ws = [] if args.spots_only else densify_windows(segments, duration, orders)
    # 指定间隔：只改进出场，就座那一路不动（它的密度与「找不到人」无关，
    # 改它只会把已有的就座帧网格挪掉、凭空多抽一遍）。
    #
    # 为什么要有这个开关：按目标帧数算出来的间隔对这一组已经是 0.1 秒，
    # 对同几段再跑一遍 `--orders`，采样点与盘上的帧逐个重合，**一帧都不会
    # 多**——命令「跑成功」而什么都没做。用户 2026-09-08 要对 order 13/14/16
    # 重抽找人，正是这种情形。
    if args.interval is not None:
        base = 1.0 / config.TRANSITION_FPS
        n = base / args.interval
        if not (args.interval > 0 and abs(n - round(n)) < 1e-9):
            print(f"   ⚠ --interval {args.interval} 不是首轮网格 {base} 秒的"
                  f"整数等分（可用 0.2 / 0.1 / {base/3:.4f} / 0.05 / 0.04 / 0.02 ...）。"
                  f"不对齐会让整条采样网格平移，盘上已有的帧一张都对不上，"
                  f"同一段时间被抽第二遍、落下一堆近重复帧。")
            return 1
        ws = [w if w.kind == "seated"
              else Window(w.seg_order, w.kind, w.t_start, w.t_end,
                          float(args.interval))
              for w in ws]
    for w in ws:
        base = (1.0 / config.TRANSITION_FPS if w.kind != "seated"
                else None)
        from pipeline.windows import seated_interval
        if base is None:
            base = seated_interval(by_order[w.seg_order].start,
                                   by_order[w.seg_order].end)
        print(f"     · order{w.seg_order:>2} {w.kind}：{w.t_start:.1f}-{w.t_end:.1f}s，"
              f"间隔 {base:.3f}s → {w.interval:.3f}s")

    # 「无可用画面」的座位旁边人工填的重抽时刻（用户 2026-09-09）：预设的
    # 进出场窗口锚在段首段尾，罩不住段中间的时刻——不接这条线索，填了等于白填。
    # 窗口由 `spot_window_for_segment` 给出：段号是这条标注自带的（第 N 段
    # 座位 M），不需要靠时刻反推，所以**时刻落在两段之间的空隙里是合法的**
    # ——进出场本来就发生在那里。区间按 [上一段结束, 下一段开始] 裁，越过去
    # 抽到的就是隔壁企业的画面。
    from pipeline.entryexit import no_candidate_notes_of
    from pipeline.windows import spot_interval_for_fps, spot_window_for_segment
    from pipeline.frames import content_fps
    # 指名时刻处抽到**源片每一张独立画面**：用户点了名，就是要那一带看得尽量全，
    # 而窗口只有 ±10 秒，抽满也不过一千来帧。
    #
    # 按 `content_fps` 而不是标称的 `probe_fps`（用户 2026-09-18）：这组视频标称
    # 50fps、实际只有 25fps 的独立内容（每帧成对出现），按标称算得 0.02 秒，抽出来
    # 一半是重复画面——那次加密 16000 张里约 8000 张重复，而检测把它们全跑了一遍，
    # 41 分钟白烧一半。比独立内容更细没有任何信息增益，只是把同一张解码画面写成
    # 两个文件、再白跑一遍检测。
    #
    # 取值仍落在首轮 0.2 秒网格的整数等分上——与该段已有的帧对齐，只补缺的那些。
    spot_iv = spot_interval_for_fps(content_fps(video)) if args.interval is None \
        else float(args.interval)
    spots = []
    labels_data = {}
    if labels_path.exists():
        try:
            labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            labels_data = {}
    for o in orders:
        for seat, rec in sorted(no_candidate_notes_of(labels_data, o).items()):
            # 读不出的那几段原样报出来，能读的照用——不因为一个错值把整条
            # 丢掉，也不猜那个错值是什么意思（按错的时刻抽出来的可能是别家
            # 企业的画面）。
            for bad in rec["unparsed"]:
                print(f"   ⚠ 第 {o} 段座位 {seat} 填的重抽时刻「{bad}」看不懂，"
                      f"这一处跳过。可写成 20:34 或 1234.5，多个用逗号隔开。")
            seg_hit = by_order.get(o)
            # 一个座位可以填**多个**时刻（那个人可能在几处露过面），逐个开窗。
            for at in rec["ats"]:
                kind = ("entry" if seg_hit is not None
                        and abs(at - seg_hit.start) <= abs(at - seg_hit.end)
                        else "exit")
                try:
                    # 按**指名的段**开窗，不靠时刻反推段号：进出场本来就发生
                    # 在段与段之间的空隙里。
                    w = spot_window_for_segment(segments, o, at, 10.0, kind,
                                                interval=spot_iv)
                except ValueError as exc:
                    print(f"   ⚠ 第 {o} 段座位 {seat} 的重抽时刻用不了：{exc}")
                    continue
                spots.append(w)
                print(f"   · 第 {o} 段座位 {seat} 指定重抽时刻 {at:.1f}s"
                      f" → {kind} {w.t_start:.1f}-{w.t_end:.1f}s，"
                      f"间隔 {w.interval:.3f}s")
    ws = list(ws) + spots
    if args.spots_only and not spots:
        # 不能悄悄退回整段加密——那正是这个开关要避免的事。
        print("   --spots-only：标注里这几段都没有填重抽时刻，一帧都不抽。"
              "在挑选页「本座位无可用画面」旁填上时刻（如 2:01:56）后再跑。")
        return 0

    frames_root = work / "frames"
    index_path = work / "frames_index.json"
    if args.dry_run:
        print(f"   --dry-run：未抽帧（本应补进 {frames_root}，"
              f"并把新帧并入 {index_path.name}）")
        return 0

    # **把上面打印的那批窗口原样传下去**（`windows=ws`）。不传的话
    # `densify()` 会按目标帧数自己再算一遍，`--interval` 就只改了打印、没改
    # 抽帧：2026-09-08 实测「间隔 0.200s → 0.040s」印得好好的，结果「新增
    # 0 帧」。打印的计划与执行的计划必须是同一份。
    report = densify(video, segments, orders, frames_root, index_path,
                     video_duration=duration,
                     manifest_path=dense_manifest_path(work, sheet),
                     windows=ws)
    print(f"   新增 {report['n_added']} 帧（原有帧一张未删），"
          f"{index_path.name} 现共 {report['n_index']} 条")
    print(f"   本轮新增文件的清单：{dense_manifest_path(work, sheet)}")
    # 不说清楚下一步，用户会以为这条命令没起作用——帧躺在盘上，候选一个没多。
    print("   下一步（新帧只有重跑检测才会变成新候选）：")
    print("     .venv/bin/python -m pipeline.run_all --stages 3,4")
    # 阶段 **5**（出挑选页），不是 7（打标，调 VLM）。2026-08-29 重排之后
    # 编号即执行顺序：1-5 人工之前，6 起依赖人工结果。
    # 与挑选页。这里原先写的是 5——照着做会真的去调 VLM：花钱、而且用的是还
    # 没标完的那份标注（结果必然被后续重跑覆盖），同时挑选页一点没重出，用户
    # 看不到新候选，只会以为加密抽帧没起作用。
    print("     .venv/bin/python -m pipeline.prepare --stages 5   # 重出挑选页")
    print("   注意：阶段 3 会按整份 frames_index.json 重跑检测（跟踪器编号"
          "全组重排），detections.json 的指纹随之改变；已有的 entry_labels "
          "会按视频内容自动重新对齐，而「座位否决 / 逐格排除 / 本座位无可用"
          "画面」这三个通道的键是 (段号, 座位号)，不受影响。")
    return 0


if __name__ == "__main__":       # pragma: no cover - CLI 入口
    raise SystemExit(main())
