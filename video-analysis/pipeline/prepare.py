"""`python -m pipeline.prepare`——只做「取帧 + 人工辅助定位」的受限入口。

用户 2026-08-23：要一个不必记阶段号的命令，跑完就能打开挑选页开始标人。

跑哪些：**1（时间表）→ 2（抽帧）→ 3（检测）→ 4（座位）→ 6（画廊与挑选
页）**。

**跳过 5（VLM 打标）与 7（项目排名）**，理由不是「省事」而是顺序：现阶段
的核心工作是人工把每个座位的人定下来（谁是创业者、谁是工作人员、哪一格是
谁），着装识别是这件事**做完之后**的独立任务。此刻跑 VLM 有两重浪费：

  - 调用本身白花——座位归属还会变，人换了衣着自然也换了，结果必然被后续
    重跑覆盖；
  - 更要紧的是**可验证性**：递补带来的变化与 VLM 自身的运行间抖动会搅在
    一起分不开（`config.VLM_INCREMENTAL` 上方记着实测数据：同一份数据两次
    运行的性别一致率 95% / 83%）。先把「谁是谁」冻结住，再谈打标。

排名（阶段 9）读的是打标结果，没有打标就无从排起，一并跳过。

**本模块不改 `run_all` 的任何既有行为**——它只是 `run_all.run` 的一个受限
入口，把阶段 1-5 原样转交，自己不建任何产物。

2026-08-29 阶段重排之前，这里还有一条特例分支：挑选页与画廊挤在同一个阶段
里，那个阶段开头 `_require(annotations.json)`，新组必然缺它，于是 prepare
只好自己把挑选页建出来、并打印「本次没有画廊」。阶段 5（出挑选页）拆出来
之后本就不读打标结果，那条分支随之删除——**一个阶段同时干着人工前与人工后
两件事**，才是当时真正的病根。

用法：

    .venv/bin/python -m pipeline.prepare                 # 用当前 config 的组
    .venv/bin/python -m pipeline.prepare --group 文化4-4.28
    .venv/bin/python -m pipeline.prepare --stages 3,4    # 只重跑其中几个
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline import config, paths

# 本入口允许出现的阶段，以及它们在 run_all 里的编号。7（排名）与 5（打标）
# 一律不在其中——这份清单就是「prepare 不跑 VLM」这条约束的唯一来源，
# 不要在别处再写一遍。
# 人工核实之前的全部阶段，一个不多一个不少（用户 2026-08-29 重排之后）。
# 旧编号下这里写的是 ("1","2","3","4","6")——那串跳号本身就是「⑤打标排在
# ⑥出挑选页前面」这个错位的产物。
PREPARE_STAGES = ("1", "2", "3", "4", "5")
SKIPPED_STAGES = ("6", "7", "8", "9")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="只做取帧与人工辅助定位（阶段 1,2,3,4,6；跳过 VLM 打标与排名）")
    p.add_argument("--group", default=None,
                   help="work/ 下的组目录名（如 文化4-4.28）；省略则用当前 config")
    p.add_argument("--stages", default=",".join(PREPARE_STAGES),
                   help=f"要跑的阶段，逗号分隔，只能是 {','.join(PREPARE_STAGES)} 的子集")
    ns = p.parse_args(argv)
    ns.stages = [s.strip() for s in ns.stages.split(",") if s.strip()]
    for s in ns.stages:
        if s in SKIPPED_STAGES:
            p.error(f"阶段 {s} 不属于 prepare（它跳过 VLM 打标与排名）："
                    f"要跑请用 python -m pipeline.run_all --stages {s}")
        if s not in PREPARE_STAGES:
            p.error(f"未知阶段 {s}，prepare 只接受 {list(PREPARE_STAGES)}")
    return ns


def _select_group(group: str | None) -> None:
    """把 `config.WORK` 指到指定的组目录。

    只改 `WORK`——`VIDEO`/`SHEET` 由 config 决定，猜错组名与视频/表名的对应
    关系会让整条流水线读错数据且不报错，所以宁可要求用户在 config 里改。
    组目录不存在时直接报错退出，不新建：新建一个空目录只会让后面每一步都
    说「缺少上游产物」，看不出根因是组名打错了。
    """
    if group is None:
        return
    root = config.WORK.parent
    target = root / group
    if not target.is_dir():
        existing = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise SystemExit(
            f"work 目录下没有组 {group!r}；现有的组：{existing}\n"
            f"  （--group 只切换 work 子目录；视频文件与 Book1 的 sheet 名由 "
            f"pipeline/config.py 的 VIDEO/SHEET 决定，换组时请一并确认）"
        )
    config.WORK = target


def _shortfall(seats_path: Path, seg_path: Path) -> tuple[int, int, list[str]]:
    """(已定座位数, 应有座位数, 每段缺口的文字说明)。

    读的是阶段 4 落盘的 `seats.json`——**纯几何排序**给出的基线，没有任何
    人看过。这里报的「还缺几人」因此是「人工至少还要处理几个」，不含挑选页
    上已经标好、但还没导出的那些。
    """
    from pipeline.schedule import load_segments_json

    if not seg_path.exists():
        return 0, 0, []
    segments = load_segments_json(seg_path)
    seats_by_order: dict[int, dict[int, int]] = {}
    if seats_path.exists():
        try:
            raw = json.loads(seats_path.read_text(encoding="utf-8"))
            seats_by_order = {int(o): {int(t): s for t, s in m.items()}
                              for o, m in raw.items()}
        except (ValueError, OSError):
            seats_by_order = {}
    have = need = 0
    lines: list[str] = []
    for seg in segments:
        occupied = len(set(seats_by_order.get(seg.order, {}).values()))
        need += seg.presenter_num
        have += min(occupied, seg.presenter_num)
        gap = seg.presenter_num - occupied
        if gap > 0:
            lines.append(f"order{seg.order:>3} {seg.company} 缺 {gap} 人")
    return have, need, lines


def prepare(group: str | None = None, stages: list[str] | None = None) -> Path | None:
    """跑受限阶段，返回挑选页路径（没生成时 None）。"""
    from pipeline import run_all

    _select_group(group)
    stages = list(stages or PREPARE_STAGES)
    for s in stages:
        if s in SKIPPED_STAGES:      # 双保险：函数直调也不许跑人工之后那几个
            raise ValueError(
                f"prepare 不跑阶段 {s}——6 起全部依赖人工核实结果，现阶段要先"
                f"把每个座位的人定下来。此刻打标的结果必然被后续重跑覆盖。")

    print(f"▶ prepare：只做取帧与人工辅助定位（{config.WORK.name}）")
    print(f"   跑阶段 {','.join(stages)}；人工核实之后的 6-9 交给 run_all")

    # 阶段 5（出挑选页）不再读 annotations.json，整段原样交给 run_all 即可。
    # 重排之前这里必须分两路：有打标结果就跑完整的阶段 6，没有就只出挑选页
    # ——那条特例分支正是「一个阶段同时干着人工前与人工后两件事」逼出来的，
    # 拆开之后不再需要。
    run_all.run(list(stages), None)

    seg_path = config.WORK / "segments.json"
    seats_path = config.WORK / "seats.json"
    review_path: Path | None = (
        paths.output_dir(config.WORK) / "entry_review.html"
        if "5" in stages else None)

    have, need, lines = _shortfall(seats_path, seg_path)
    print("")
    print("=" * 62)
    if review_path is not None:
        print(f"  挑选页：{review_path}")
    print(f"  就座阶段已定 {have}/{need} 个座位，还缺 {max(0, need - have)} 人待人工指定")
    for line in lines:
        print(f"    · {line}")
    print("  标完在页面右下角点「导出标注 JSON」，再跑 run_all 阶段 6,7 建登记表并打标。")
    print("=" * 62)
    return review_path


if __name__ == "__main__":
    ns = parse_args(sys.argv[1:])
    prepare(group=ns.group, stages=ns.stages)
