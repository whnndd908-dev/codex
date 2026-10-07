"""Task 51 · 要做的二：座位登记表——「每个座位是谁、用哪些帧」的长期产物。

用户 2026-08-23 原话：

    「最核心的工作是通过人工复核，将每个项目的每个座位的人确定下来，并
    存储下来，接下来针对抽好的画面可以反复用不同的 VLM 进行识别着装颜色
    和款式」。

这句话是一次架构判断，值得照着做：**人工复核的结果与 VLM 打标的结果必须
分开存放**。此前两者混在 `annotations.json` 一份文件里，换个模型、改个
提示词、加一个字段都要连着「谁是谁」一起重算，而「谁是谁」恰恰是唯一花了
人力、也唯一不该反复重算的那部分。登记表把它单独固化下来：

    检测 → 座位/进出场候选 → **人工复核 → 座位登记表（本模块）** → 反复打标

因此登记表必须自带**检测指纹**（沿用 Task 46 的机制）。它记的是帧路径与
检测框坐标，换一次检测这些全部失效——Task 46 的血教训是企业行序变动后重
检测，旧标注里少数编号碰巧还在、但那已经是另一个人，错配不报错，整条流水
线照常跑完、数字全错。登记表比标注更危险：标注还能按视频内容（时间+位置）
重新对齐，登记表里存的是**具体某一帧某一个框**，对不上就只能整份作废。

**本模块的边界**（有意划在这里，见 task-51 brief）：只负责**产出**与
**读回**这份登记表，以及一个 CLI。**不改 `assemble.assemble_annotations`
去消费它**——那是下一步。两件事混在一起改动面过大，无法分别验证：现在
登记表写错了看得出来（它自己有测试、有 CLI 打印），接上去之后再错就只能
从最终 CSV 反推。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape as _esc
from pathlib import Path

from pipeline import config, paths
from pipeline.annotate import build_evidence_sheet, select_best_frames
from pipeline.detect import Detection
from pipeline.judges import (
    judge_candidates, judge_cell_assignments, judge_cell_exclusions,
    judge_donations_from_candidates, judge_labels_of, judge_picks, judge_slots,
    realign_judge_labels,
)
from pipeline.entryexit import (
    CELL_COUNT, apply_cell_exclusions, cell_exclusions_of, cell_dets, cells_dets,
    candidates_for_labels, entryexit_candidates, entryexit_fill_plan,
    labelled_seat_candidates,
    seat_reassign_of,
    topup_pairs,
    no_candidate_seats_of, topup_frames, vacated_seats_of,
)
from pipeline.gallery import _fit_images_to_budget
from pipeline.schedule import Segment

# Task 56A：座位记录的字段形状变了（`cell_index` 标量 → `cell_indices`
# 列表），因此升到 v2。**版本号存在的全部意义就是挡住静默误读**：旧文件里
# 没有 `cell_indices`，新代码读到会当成「这个座位不是格位切分来的」，而那
# 个标量 `cell_index` 再没有一处会去读它——错得不报错，登记表看上去完全
# 正常。`load_seat_registry` 见到不是本版本的一律整份作废，见那里的决策表。
REGISTRY_SCHEMA = "seat_registry_v2"

# 每个座位最多留档几张帧。与 `annotate.build_evidence_sheet` 的 2×2 拼图
# 张数一致（= `entryexit.CELL_COUNT`）——登记表存在的意义就是「下次换 VLM
# 直接拿这几张去打标」，存多存少都会与实际送去的画面对不上。
#
# Task 56A · 多格合并后这个上限**依然成立、且不会截掉任何一格**：一个座位
# 最多由 2×2 的四格全部解引用而来（格位号上限就是 CELL_COUNT，见
# `entryexit.cells_to_cell_map`），因此格位切分那一路的帧数天然 ≤ 4。真到
# 了证据图格数变多的那天，这里会截断——宁可截断也不能让登记表里出现证据图
# 上无处对应的帧；届时该做的是与 `CELL_COUNT` 一起改，不是悄悄放开。下面
# 这句断言就是防止只改了一边。
REGISTRY_FRAMES_PER_SEAT = 4
assert REGISTRY_FRAMES_PER_SEAT >= CELL_COUNT, (
    "留档上限不得小于证据图格数，否则同一个人占满全部格位时会被静默截掉")


# ---------------------------------------------------------------------------
# Task 56C · 二：未定座位的**理由**
#
# 用户 2026-08-24：「如果某个座位的人没有合适的候选图，能标记下来，再抽帧
# 二次有更多的备选吗」。此前 `unresolved_seats` 只是一串座位号，「人工看过、
# 确实没有可用画面」与「压根还没人看过」混在同一个数字里——既没法告诉用户
# 还剩多少要看，也没法据此挑出该二次加密抽帧的那几段（`pipeline.densify`
# 读的就是这份标记）。
#
# 刻意写成两个独立的小函数、`build_seat_registry` 主体只多一行：并行任务
# 也在改这个文件，改动面越小越好合。
# ---------------------------------------------------------------------------

UNRESOLVED_NO_CANDIDATE = "no_candidate"     # 人工在挑选页判定：确实没有可用画面
UNRESOLVED_NOT_REVIEWED = "not_reviewed"     # 还没人看过


def unresolved_seat_reasons(labels: dict | None, seg_order: int,
                            unresolved_seats) -> dict[str, str]:
    """`{"<座位号>": "no_candidate" | "not_reviewed"}`。

    默认一律 `not_reviewed`：默认成 `no_candidate` 等于替用户断言「这里
    确实没画面」，而他一眼都还没看过——加密抽帧会据此白跑一段。

    只覆盖**仍未定**的座位。用户先标「没画面」、后来又找到人补上时，那条
    陈旧标记自动作废，否则登记表会同时说「座位 2 已定」和「座位 2 没画面」。
    """
    marked = no_candidate_seats_of(labels, seg_order)
    return {str(s): (UNRESOLVED_NO_CANDIDATE if int(s) in marked
                     else UNRESOLVED_NOT_REVIEWED)
            for s in unresolved_seats}


def unresolved_reason_counts(reg: dict) -> dict[str, int]:
    """整份登记表里两类未定座位各有多少。CLI 与调用方都用这一份，
    两处各数一遍必然日后走样（Ruling R39）。"""
    out = {UNRESOLVED_NO_CANDIDATE: 0, UNRESOLVED_NOT_REVIEWED: 0}
    for rec in (reg.get("segments") or {}).values():
        for reason in (rec.get("unresolved_reason") or {}).values():
            if reason in out:
                out[reason] += 1
    return out


def registry_path(work: Path, sheet: str) -> Path:
    """登记表的落盘路径 `work/Output/<组>/seat_registry_<SHEET>.json`。

    文件名带上 SHEET，与 `entry_labels_<SHEET>.json`、
    `result_<SHEET>.csv` 一贯的命名方式相同——同一个 work 目录下可能先后
    跑过不同的组，不带组名会互相覆盖且看不出来。

    Task 56G 起交付物只存 Output 一份（组目录只放原始与过程文件）。旧位置
    还留着文件时照样读得到，见 `paths._resolve` 上方的说明。
    """
    return paths.registry_path(work, sheet)


# ---------------------------------------------------------------------------
# Task 56B · 登记表必须自证「它由哪一份人工标注产出」
#
# 真实事故（2026-08-23）：用户 14:03 在 entry_review.html 上重新标注并导出
# `entry_labels_<组>.json`，而 `seat_registry_<组>.json` 与
# `vlm_preview_<组>.html` 还停在 13:53——产出登记表这一步是**手敲**
# `python -m pipeline.registry --preview`、不自动，两边都不报错，用户于是
# 对着 10 分钟前的旧图做完了一轮人工核对，最后只能靠肉眼比对文件时间戳
# 才发现。这条流水线此前已被「产物没重新生成、代码却改了」这类静默陈旧坑
# 过五次以上：核对结论对不上产物，而没有任何一处会响。
#
# 为什么判定必须靠**内容摘要**而不是 mtime（与 `entryexit.
# detections_fingerprint` 同一套理由，这里再记一遍是因为本条的两个方向都
# 出现过）：
#   · mtime 变新不等于内容变了——用户原样重存一次（重复导出同样的标注、
#     拷贝、rsync、git checkout）就会刷新 mtime，按 mtime 判定会把一份好好
#     的登记表判成陈旧，代价是白重跑一遍并动摇用户对报警的信任；
#   · mtime 变旧不等于内容没变——从别处拷回一份改过的标注，mtime 可能反而
#     比登记表更早，按 mtime 判定会**放行**，那正是本任务要根除的静默。
# mtime 只作为辅助提示打印（帮人回忆「我是不是刚导出过」），绝不参与判定。
# ---------------------------------------------------------------------------

def labels_path(work: Path, sheet: str) -> Path:
    """人工标注的落盘路径 `work/entry_labels_<SHEET>.json`。

    与 `registry_path` 同一套命名规则，单一实现两处复用——此前这个拼法只
    散落在 `main()` 里一行，判定新鲜度时若在别处再拼一次，两处一旦漂移就
    会「拿另一份标注的摘要去比」，比错了照样不报错。
    """
    return paths.labels_path(work, sheet)


def labels_digest(path: Path) -> str:
    """标注文件的内容摘要（sha256 前 16 位十六进制）。

    文件不存在时返回**空串**：该组还没人工标过是合法状态
    （`build_seat_registry` 本来就允许，只是登记表里全是就座阶段的结果），
    空串对空串比得上，不该因此报陈旧。
    """
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class RegistryStaleError(RuntimeError):
    """登记表落后于人工标注（或来源不明）时抛出。

    为什么是**抛异常**而不是打印警告：打印早就试过了，这条流水线的每一步
    都在打印，14:03 那次谁也没看见。下游拿着一份对不上的登记表继续跑，
    产出的每一行都是错的，而错在哪一步事后无法反推。
    """


@dataclass(frozen=True)
class RegistryStaleness:
    """新鲜度判定的结构化结论（`registry_staleness` 的返回值）。

    字段：
      · `stale`            总结论。
      · `reason`           机器可读的原因：`ok` / `no_registry` /
                           `broken`（读不出或不是本格式）/
                           `no_digest`（旧版登记表压根没记来源）/
                           `digest_mismatch`。
      · `registry_exists`  登记表在不在。
      · `labels_exists`    标注在不在。
      · `stored_digest`    登记表记的来源摘要；`None` 表示**压根没记**
                           （旧文件），`""` 表示「当时没有标注文件」——这
                           两者是两回事，混成一个值就没法只对旧文件报警。
      · `current_digest`   当前标注的摘要（`""` = 标注文件不存在）。
      · `message`          给人看的中文结论，含该重跑哪条命令。
      · `registry_mtime` / `labels_mtime`  仅供打印的辅助提示，**不参与
                           判定**（见本节顶部：mtime 变新变旧都骗人）。
    """
    stale: bool
    reason: str
    registry_exists: bool
    labels_exists: bool
    stored_digest: str | None
    current_digest: str
    message: str
    registry_mtime: str | None = None
    labels_mtime: str | None = None


_REBUILD_CMD = "python -m pipeline.registry --preview"


def _mtime_text(path: Path) -> str | None:
    """文件 mtime 的可读形式；不存在返回 None。**只用于打印**。"""
    p = Path(path)
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def registry_staleness(work: Path, sheet: str, *, reg_path: Path | None = None,
                       reg_body: dict | None = None) -> RegistryStaleness:
    """判定 `work/<组>` 的座位登记表是否仍与当前人工标注对得上。

    **这是唯一的判定入口**：CLI（`--check`）、`load_seat_registry`、以及
    日后任何消费登记表的地方都调它。绝不允许在第二处再写一遍比较逻辑——
    两份判定迟早各自漂移，而漂移的表现是其中一份悄悄不报了，正好回到本
    任务要修的那个坑。

    `reg_path` / `reg_body` 是给已经拿到登记表的调用方用的旁路（
    `load_seat_registry` 手上就是刚解析好的那一份，且它的路径可能是
    `--out` 指定的非默认名）：给了就用给的，不重新按 work/sheet 去猜一份
    别的文件来比。

    判定表：

      | 情形                          | stale | reason          |
      |-------------------------------|-------|-----------------|
      | 登记表不存在                  | 是    | no_registry     |
      | 登记表读不出 / 没有 `_meta`   | 是    | broken          |
      | `_meta` 里没有 `labels_digest`| 是    | no_digest       |
      | 摘要不一致                    | 是    | digest_mismatch |
      | 摘要一致（含双方都为空）      | 否    | ok              |
    """
    work = Path(work)
    rp = Path(reg_path) if reg_path is not None else registry_path(work, sheet)
    lp = labels_path(work, sheet)
    cur = labels_digest(lp)
    common = {
        "labels_exists": lp.is_file(),
        "current_digest": cur,
        "registry_mtime": _mtime_text(rp),
        "labels_mtime": _mtime_text(lp),
    }

    body = reg_body
    if body is None:
        if not rp.is_file():
            return RegistryStaleness(
                stale=True, reason="no_registry", registry_exists=False,
                stored_digest=None,
                message=(f"还没有 {rp.name}，请先跑 `{_REBUILD_CMD}` 生成"
                         "座位登记表与人工核对页"), **common)
        try:
            body = json.loads(rp.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            return RegistryStaleness(
                stale=True, reason="broken", registry_exists=True,
                stored_digest=None,
                message=f"{rp.name} 读不出来（{exc}），请重跑 `{_REBUILD_CMD}`",
                **common)

    if not isinstance(body, dict) or not isinstance(body.get("_meta"), dict):
        return RegistryStaleness(
            stale=True, reason="broken", registry_exists=True,
            stored_digest=None,
            message=(f"{rp.name} 没有 _meta（不是 {REGISTRY_SCHEMA} 格式），"
                     f"无从判断它产自哪一份标注，请重跑 `{_REBUILD_CMD}`"),
            **common)

    meta = body["_meta"]
    stored = meta.get("labels_digest")
    if stored is None:
        # 旧版登记表（Task 56B 之前产出的）压根没记来源。**不能**沿用检测
        # 指纹那条「自己没记就跳过校验」的口径：旧文件恰恰是最可能已经落后
        # 于标注的那一批，跳过等于专门为高危文件开一个静默通道。
        return RegistryStaleness(
            stale=True, reason="no_digest", registry_exists=True,
            stored_digest=None,
            message=(f"{rp.name} 是旧版登记表，未记录它产自哪一份标注的内容"
                     f"摘要，无法确认是否最新——请重跑 `{_REBUILD_CMD}`"),
            **common)

    if str(stored) != cur:
        return RegistryStaleness(
            stale=True, reason="digest_mismatch", registry_exists=True,
            stored_digest=str(stored),
            message=(f"{rp.name} 由标注摘要 {stored} 产出，当前 "
                     f"{lp.name} 是 {cur or '（文件已不存在）'}——人工标注变了"
                     f"而登记表没重跑，请执行 `{_REBUILD_CMD}`"),
            **common)

    return RegistryStaleness(
        stale=False, reason="ok", registry_exists=True, stored_digest=str(stored),
        message=(f"{rp.name} 产自当前这一份标注"
                 + (f"（摘要 {cur}）" if cur else "（该组没有人工标注文件）")),
        **common)


def _frame_record(d: Detection) -> dict:
    """一帧的留档：路径 + 时间 + 检测框 + 置信度。

    存**绝对/原样的帧路径与框坐标**而不是 track_id：下次打标要的是「打开
    这张图、裁这个框」，track_id 换一次检测就全变（Task 46），存了也没用。
    """
    return {
        "frame_path": str(d.frame_path),
        "t": float(d.t),
        "box": [float(d.x1), float(d.y1), float(d.x2), float(d.y2)],
        "conf": float(d.conf),
        "kind": d.kind,
        "track_id": int(d.track_id),
    }


def _seat_record(*, seat: int, source: str, confirmed: bool, track_id,
                 kind, stable_id, cell_indices, note: str, seat_uncertain: int,
                 frames: list[Detection], cross_segment: int = 0,
                 src_order=None, seg_order=None) -> dict:
    # 这一格的证据里有几张来自**相邻段**的时间窗口。
    #
    # `cross_segment` 说的是「整条记录都来自相邻段」；而补格位之后一个座位
    # 可能是**混的**：本段两张 + 相邻段标过来的两张（用户 2026-09-09 实测
    # 城市1 第 16 段 → 第 17 段座位 1）。混了却不记，日后拿 frames 里的时间戳
    # 去核对本段的时间范围会有几张对不上，而没有任何一处会报错。
    n_cross = 0 if seg_order is None else sum(
        1 for d in frames if int(getattr(d, "seg_order", seg_order)) != int(seg_order))
    return {
        "seat": int(seat),
        # seated / entryexit / cell_split——这个人是**怎么定下来的**。
        "source": source,
        # 人工确认状态：就座阶段的座位号来自纯几何排序（`seats.
        # assign_segment_seats`），没人看过，一律 False；进出场递补与格位
        # 切分都是用户在挑选页一个个点出来的，是 True。下一步换 VLM 重跑时
        # 靠这个字段区分「可以直接信」与「还得人看一眼」。
        "confirmed": bool(confirmed),
        "confirmed_via": "entry_review" if confirmed else "",
        "track_id": None if track_id is None else int(track_id),
        "kind": kind,
        "stable_id": stable_id,
        # Task 56A：这个座位由证据图上**哪几格**定下来（升序；不是格位切分
        # 来的记录一律空列表）。同一个人常占两格（「格位2 = 同格位1」），
        # 那两格的帧都是他本人的证据，`frames` 里逐格各留一张，本字段与
        # `frames` **逐项一一对应**——下次换 VLM 打标时才知道每张是第几格。
        # 曾经这里是标量 `cell_index`，只记得下其中一格，另一格连同它那张帧
        # 静默消失（用户 2026-08-24 实测文化4组 order 2 发现）。
        "cell_indices": [int(c) for c in (cell_indices or [])],
        "note": note or "",
        "seat_uncertain": int(seat_uncertain),
        # Task 53 · 需求六：跨段流转。这一格的画面来自**相邻段**的时间窗口
        # （用户在段 15 的备注里写「下一场创业者2」，指的是段 16 的人）。
        # `src_order` 是那些帧真正所属的段号——不记的话，日后拿 `frames`
        # 里的时间戳去核对本段的时间范围必然对不上，而没有任何一处会报错。
        # 普通记录两项分别是 0 与 None，形状不因此改变。
        "cross_segment": int(cross_segment),
        "src_order": None if src_order is None else int(src_order),
        "n_cross_frames": int(n_cross),
        "frames": [_frame_record(d) for d in frames],
    }


def _seats_after_reassign(labels: dict | None, seg_order: int,
                          seats: dict | None) -> dict[int, int]:
    """把 `{track_id: 座位号}` 按人工改号重映射，供占用判断使用。

    这份表只有一个用途：让 `entryexit_fill_plan` 知道「哪些座位已经有人、
    不该再补」。因此改号在这里表现为**把原座位从表里去掉**——那张图挂到别的
    号上了，原号空了，人工在候选上给原号标的那条要能补进来。

    **不把目标座位加进去**，这一点是刻意的：目标座位到底归谁，由「人工在候选
    上指名的更具体」这条既有规则决定（`test_reassign_does_not_touch_seats_
    the_user_named_on_a_candidate`）。加进去等于让改号顶掉那条更确切的判断，
    而改号过来的那张与人工标的候选该怎么合并，是就座分支自己的事
    （Task 56W 的 merge_into），不在这里重复一遍。
    """
    m = seat_reassign_of(labels, seg_order)
    src = {int(t): int(s) for t, s in (seats or {}).items()}
    if not m:
        return src
    return {tid: seat for tid, seat in src.items() if seat not in m}


def build_seat_registry(
    segments: list[Segment],
    dets: list[Detection],
    seats_by_order: dict[int, dict[int, int]],
    entry_labels: dict | None,
    *,
    detections_fp: str | None = None,
    sheet: str | None = None,
    labels_path: Path | None = None,
) -> dict:
    """把「每个座位 → 哪个人、用哪些帧」定下来，写成可长期持有的产物。

    这是本项目的核心资产：人工复核一次，之后换 VLM、改提示词、加字段都
    只需重跑打标，不必重跑检测与人工复核。

    三种来源（`source` 字段），优先级从高到低：

      · `cell_split`  用户在备注/挑选页按格位号指明「这一格是创业者 3」，
                      `entryexit.split_candidate_by_cells` 切出来的画面。
                      记下 `cell_indices` 与备注原文——下次打标必须还能找回
                      「用第 1、3 格那两张」，否则又会拿混了多人的四格拼图
                      去编码。同一个人占多格时（「格位2 = 同格位1」）**每
                      一格的那张都留档**（Task 56A）：它们都是他本人的证据，
                      丢掉一张就是静默丢证据。
      · `entryexit`   用户在挑选页把某个进出场候选标成创业者并指定了座位
                      号。记下 Task 46 的 `stable_id`。
      · `seated`      就座阶段（`seats.assign_segment_seats`）的结果。
                      `confirmed=False`——它是**纯几何排序**给出的基线顺序，
                      没有任何人看过。把它标成"已确认"是本登记表最容易犯
                      也最有害的错：下一步会因此跳过人工复核。

    定不下来的座位**不凭空造记录**，而是列进该段的 `unresolved_seats`，
    并在 `unresolved_reason` 里逐个写明是「人工判定确实没有可用画面」
    （`no_candidate`，来自挑选页的标记）还是「还没人看过」（`not_reviewed`）；
    被人工否决（`seat_overrides` 的 `rejected`）又没补上的座位同样落在
    那里，并另记一份 `rejected_seats`。凡有缺口必须可见。

    `detections_fp` 是当前 `detections.json` 的内容哈希
    （`entryexit.detections_fingerprint`）。**强烈建议传**：不传时
    `_meta.detections_fingerprint` 为 None，`load_seat_registry` 只能跳过
    校验（并打印），而登记表存的是具体某帧某框，换检测后照常使用等于把
    别人的画面当成这个人的。

    `labels_path` 是 `entry_labels` 那一份标注的**文件路径**（Task 56B）。
    传了就把它的内容摘要记进 `_meta.labels_digest`，登记表由此能自证「我产
    自哪一次人工标注」，`registry_staleness` / `load_seat_registry` 读回时
    对不上就报错。不传时按「当时没有标注文件」记（摘要空串）——若 work 里
    其实有标注文件，读回时摘要对不上会报错，宁可误报也不静默放行。

    返回值结构（`REGISTRY_SCHEMA`，当前 `seat_registry_v2`）::

        {"_meta": {"schema", "sheet", "detections_fingerprint", "built_at",
                   "labels_file", "labels_digest", "labels_mtime",
                   "n_segments", "n_seats_expected", "n_seats_resolved",
                   "counts_by_source": {"seated", "entryexit", "cell_split"}},
         "segments": {"<order>": {
             "company", "presenter_num", "rejected_seats", "unresolved_seats",
             # Task 56C：{"<座位号>": "no_candidate" | "not_reviewed"}
             "unresolved_reason",
             "seats": {"<seat>": {
                 "seat", "source", "confirmed", "confirmed_via", "track_id",
                 "kind", "stable_id", "cell_indices", "note", "seat_uncertain",
                 "cross_segment", "src_order",
                 "frames": [{"frame_path", "t", "box", "conf", "kind",
                             "track_id"}, ...]}}}}}

    `cross_segment` / `src_order`（Task 53 需求六）：这一座是由**相邻段**的
    格位备注（「下一场创业者N」/「上一场创业者N」）流转过来的，它的
    `frames` 属于 `src_order` 那一段的时间窗口，不属于本段。普通记录分别是
    0 与 None。`source` 仍是 `cell_split`（它确实是按格位切出来的），
    `counts_by_source` 因此保持三个键不变——跨段与否是**另一个维度**，混进
    来源分类里会让「有多少人是算法排的、多少是人工点的」这个统计失真。

    JSON 的对象键只能是字符串，因此 `segments` 与 `seats` 的键都是 `str`
    ——写出去与读回来必须是同一种类型，否则 `test_registry_roundtrip` 那种
    「写盘再读回应完全相同」的保证不成立（int 键写出去会静默变成 str）。
    """
    labels = entry_labels or {}
    seg_by_order = {int(s.order): s for s in segments}
    seg_out: dict[str, dict] = {}
    counts = {"seated": 0, "entryexit": 0, "cell_split": 0}
    n_expected = 0

    for seg in segments:
        n_expected += int(seg.presenter_num)
        rejected = sorted(vacated_seats_of(labels, seg.order))
        seats: dict[str, dict] = {}

        # --- 1) 人工那一路先落位（最具体的判断优先）------------------------
        # 喂本段 **加上左右相邻段**（Task 53 需求六）：跨段流转的候选是在
        # 相邻段的标注里写下的，只喂本段就永远扫不到它——`entryexit_fill_
        # plan` 只遍历传进去的 segments 找标注。`seats_by_order` 同样按这
        # 三段给（它只用来取「已占用座位集合」），否则目标段的占位判断会
        # 以为整段都空着、把人补到已经有人的座位上。
        # 相邻段自己的普通补位项落在 `plan[相邻段号]`，本函数只取
        # `plan[seg.order]`，不会串进来。
        window = [s for s in (seg_by_order.get(seg.order - 1), seg,
                              seg_by_order.get(seg.order + 1)) if s is not None]
        # 占用表要按**改号之后**的号给（Task 56V 的改号通道）。给原始基线的
        # 后果，用户 2026-09-08 实测城市1 order 4：他先勾「这张图其实是座位
        # 2」把就座那张挪走，再在退场候选上标「创业者 · 座位 1」，结果座位 1
        # 仍是「还没人看过」——占用表里座位 1 还挂着那张已经挪走的卡，
        # `entryexit_fill_plan` 按「不覆盖已占座位」把这条跳过了，同时那条
        # 跨段流转也被同一个理由挡掉（日志里「该座位已有人，不覆盖」）。
        # 两条人工标注一条都没落地，一处不报错。
        #
        # `assemble` 那边喂的是 `by_final_seat`（已含否决与改号），本来就是
        # 对的；走样的只有这一路。
        plan, _needs_resplit, _needs_seat = entryexit_fill_plan(
            window, dets, labels,
            {s.order: _seats_after_reassign(labels, s.order,
                                            seats_by_order.get(s.order, {}))
             for s in window},
            # 传整组段号：本次只给了三段窗口，不传的话「第 N 段不在窗口里」
            # 会被误报成「没有第 N 段」（假警报，用户 2026-08-25 实测撞上过）。
            known_orders=[int(sg.order) for sg in segments],
        )
        # 候选索引同样覆盖这三段：跨段项的帧来自源段的窗口，用本段的候选
        # 池去找必然找不到（找不到就 `continue`，那条流转会静默消失）。
        by_key = {}
        for s in window:
            # 展示 ∪ 折叠：用户可能把人标在折叠区的候选上。
            for c in candidates_for_labels(dets, s):
                by_key[(int(s.order), f"{c['kind']}:{c['track_id']}")] = c
        seg_labels = labels.get(str(seg.order)) or labels.get(seg.order) or {}
        # 补格位用的候选表：只含「用户亲手指名给某个座位的创业者」，与
        # `assemble` 用同一个函数，两处各算一份必然走样（走样的表现是核对页
        # 与模型看到的画面对不上，且不报错）。**两路都要用**——此前它只算在
        # 就座那一路前面，格位切分／进出场递补定下来的座位压根不补（用户
        # 2026-08-26 实测文化4组 order 3，三个座位全是这一类，全都没补）。
        # 相邻段标过来的（「上/下一组创业者」）同样是本段的补帧来源，
        # 见 `labelled_seat_candidates` 里那段说明。
        topup_by_seat = labelled_seat_candidates(dets, seg, labels,
                                                 neighbours=window)

        for item in plan.get(seg.order, []):
            src_order = int(item.get("src_order") or seg.order)
            cand = by_key.get((src_order, f"{item['kind']}:{item['track_id']}"))
            if cand is None:
                continue
            # Task 56A：一个座位可能由**多格**解引用而来（「格位2 = 同格位
            # 1」是同一个人），那几格的帧全部留档。`cell_indices` 由
            # `entryexit._plan_entry` 给出，代表格位号（旧的 `cell_index`）
            # 恒等于它的第一项；这里只读列表，不再各算一遍。
            cells = list(item.get("cell_indices") or [])
            if not cells and item.get("cell_index"):
                # 兜底：直接调用 `entryexit_fill_plan` 之外的调用方手搓的
                # 计划项可能只有标量。**不静默当成"没有格位"**——那会让下面
                # 拿整条 track 的四张证据帧去留档，而那四张里坐着不同的人。
                cells = [int(item["cell_index"])]
            is_split = bool(cells)
            if is_split:
                # 逐格排除（用户 2026-08-24）：跟踪串号会让某一格的框实际
                # 框到别人，用户在挑选页勾掉那一格之后，登记表绝不能照旧
                # 把它留档——下一轮打标会直接拿登记表里这几张去测颜色。
                # 排除号在这里按**该座位自己这几张的位置**数。
                #
                # ⚠ `assemble` 的同一路按**父证据图上的格位号**数，两者对
                # `cells=[3,4]` 这类子集会给出不同结果。眼下这条分歧是死的：
                # 挑选页上的「排除格 N」复选框只长在**就座**证据图上
                # （`entryreview` 的 `filled_seats`，`n_cells` 取自
                # `sheet_dets`），格位切分来的座位根本没有那排复选框，三组
                # 现有的 `cell_exclusions` 也全部落在 seated 座位上。等哪天
                # 给切分座位也加上复选框，**必须先定死按哪种口径**再动这里
                # ——两处不一致的表现是核对页显示的图与模型看到的对不上，
                # 且一处不报错。
                # **逐格排除在这里一概不查**（Task 56W）。那排「排除格 N」
                # 复选框只长在**就座**证据图上（`entryreview` 的
                # `filled_seats`，`n_cells` 取自 `sheet_dets`，全部来自
                # seated 检测），说的是那张 2×2 的第几格。
                #
                # 此前这里拿同一个 `(段, 座位)` 键也查了一遍——用户 2026-08-29
                # 实测城市2 order 6：他在「就座座位1」那张图上勾的第 2、4 格，
                # 被套到了 `entry@5001` 切给座位 1 的那两格上（按位置数剔掉第
                # 2 个），两格变一格，且一处不报错。两张完全不同的图共用一个
                # 键，剔掉的根本不是他勾的那一格。
                pairs = [(k, d) for k in cells for d in cell_dets(cand, k)]
                kept = list(pairs)
                assert [d for _k, d in kept] == cells_dets(
                    cand, [k for k, _d in kept]), (
                    "留档的帧必须与 entryexit.cells_dets 逐张相同——两处各切"
                    "一次必然日后走样，而走样的表现是衣着编码给了错的人")
            else:
                # 整条候选：位置 i 就是格位 i（与 `cell_dets` 同一口径）。
                kept = list(enumerate(cand["best_dets"], start=1))
            # 这个座位**其余**的人工标注并进来，把 2×2 补满（用户 2026-08-26
            # 实测文化4组 order 3：「在多帧分别选的窗格没有拼在一起，应该根据
            # 标注尽可能全的拼成 2×2 四张图」）。
            #
            # **排掉落位用的这一条**（按 stable_id）：它的帧刚刚被逐格排除筛
            # 过，再从它自己补一遍等于把用户勾掉的那一格原样搬回来——他勾了
            # 等于没勾，且一处不报错。其余候选与被勾掉的那几格没有交集。
            others = [sp for sp in (topup_by_seat.get(int(item["seat"])) or [])
                      if (sp[0] or {}).get("stable_id") != cand["stable_id"]]
            kept = topup_pairs(kept, others, want=REGISTRY_FRAMES_PER_SEAT)
            frames = [d for _k, d in kept]
            # `cell_indices` 与 `frames` 必须逐项一一对应（Task 56A）——补进来
            # 的帧连同它在**自己那张**证据图上的格位号一起记，只加帧不加格位号
            # 会让「第几格」错位到另一张帧上。不是格位切分来的仍记空列表：那是
            # `source` 的判据，也是核对页区分「单张 / 多格 / 四格拼图」的依据。
            cells = [k for k, _d in kept] if is_split else []
            if not frames:
                # 该格位越界，或几格全被人工勾掉：这个座位就此定不下来，
                # 落进 unresolved_seats（凡有缺口必须可见），绝不留一条
                # frames 为空的记录假装它定下来了。
                continue
            source = "cell_split" if is_split else "entryexit"
            src_labels = (labels.get(str(src_order))
                          or labels.get(src_order) or seg_labels or {})
            lab = src_labels.get(cand["stable_id"]) or {}
            seats[str(item["seat"])] = _seat_record(
                seg_order=seg.order,
                seat=item["seat"], source=source, confirmed=True,
                track_id=cand["track_id"], kind=cand["kind"],
                stable_id=cand["stable_id"],
                # 两处**同一个切片**：`cell_indices` 与 `frames` 必须逐项
                # 对应，只截其中一边就会让「第几格」错位到另一张帧上。
                # 当前上限 ≥ CELL_COUNT（见文件头的断言），这一刀实际永远
                # 切不到东西——留着是为了「上限与格数一起改」的那一天不必
                # 再想起这件事，不是可以随手删的冗余。
                cell_indices=cells[:REGISTRY_FRAMES_PER_SEAT],
                note=(lab.get("note") or ""),
                seat_uncertain=item.get("seat_uncertain", 0),
                frames=frames[:REGISTRY_FRAMES_PER_SEAT],
                cross_segment=int(item.get("cross_segment") or 0),
                src_order=(src_order if item.get("cross_segment") else None),
            )
            counts[source] += 1

        # --- 2) 就座阶段的结果填进还空着的座位 ------------------------------
        # **不覆盖**人工那一路：被否决的座位（`rejected`）更是绝不回填——
        # 用户已经明说「这个座位判错了」，把它原样写回登记表等于当没听见。
        #
        # 已定座位改号（Task 56V，用户 2026-08-29 实测城市2 order 6）：
        # `{原座位号: 新座位号}`。这一路的图此前只能整体否决，改号得绕一圈
        # 去进出场候选里重新指认同一个人；那个人没有好用的候选时，一张本来
        # 很好的就座图就此浪费。
        reassign = seat_reassign_of(labels, seg.order)
        # 被改走的原座位号：它不再由就座那一路占据（哪怕改号最终没落地——
        # 落不了地时把原座位留着等于假装无事发生，用户看不出他的改号失效）。
        # **被顶掉的那一位不自己找别的座位坐下**：用户只说了「这张图是座位
        # N」，没说被顶掉的那个是谁。替他猜一个是本项目最忌讳的那类错。
        displaced = {int(t) for t in reassign.values()}
        for tid, seat in (seats_by_order.get(seg.order) or {}).items():
            seat = int(seat)
            # `card_seat` 是**这张图原本挂着的座位号**——逐格排除是用户在那张
            # 卡片上勾的，必须按它去查（按新号查会查到另一个座位的排除表上，
            # 表现是勾掉的那一格又回来了、没勾的反而没了，且一处不报错）。
            card_seat = seat
            moved = card_seat in reassign
            if moved:
                seat = reassign[card_seat]
            elif seat in displaced:
                # 目标座位原来的占据者：让位，落进 unresolved 等用户指认。
                continue
            # **改号压过 `rejected`**（Task 56W）。用户 2026-08-29 实测城市2
            # order 6：座位 1、2、3 还带着他上一轮「否决 + 去候选里重新指认」
            # 留下的 rejected，两次新改号因此一次都没落地——他在页面上改了号，
            # 跑下来毫无变化且不报错。改号本身就是「这个自动结果不对，它其实
            # 是座位 N」的更具体说法，压过那句笼统的「这个座位判错了」；而目标
            # 座位上那句 rejected 说的是**目标自己原来那张卡**，与搬进来的无关。
            if not moved and seat in rejected:
                continue
            # 改号到一个**已有人工标注**的座位：两边的图合并（用户 2026-08-29
            # 「应与后边人工标注的座位2进行合并」），不是丢掉改号。
            merge_into = seats.get(str(seat)) if moved else None
            if merge_into is None and str(seat) in seats:
                # 没改号却撞上人工已定的座位：那是**自动**结果，不许顶掉人工
                # 的判断——放松了等于让未经复核的几何排序悄悄改写座位表。
                #
                # 改号没落地时（目标被别的人工记录占着且不是本人搬去的）这条
                # 同样保证**原座位也不留**：`seat` 此刻已是新号，continue 一走
                # 这条 track 哪个座位都不写。留着原座位等于假装无事发生。
                continue
            track = [d for d in dets if d.seg_order == seg.order
                     and d.kind == "seated" and d.track_id == tid]
            if not track:
                continue
            # 逐格排除同样适用于就座这一路（用户 2026-08-25 实测城市2 order 2
            # 座位 3：勾了「排除格1」「排除格3」，登记表里却仍是四帧全在，
            # 他以为没生效）。
            #
            # 实际是**只有这一路漏了**：`assemble` 的就座分支早就走
            # `apply_cell_exclusions`。两边就此分叉——核对页与交付包里的证据图
            # 是四格，模型真正看到的是两格，正是本项目最忌讳的「人核对了一张
            # 模型没看过的图」，且一处不报错。
            #
            # 位置口径成立的前提：本处、`assemble` 的就座分支、挑选页上那几个
            # 复选框，三处都用 `select_best_frames(该座位的就座帧, n=CELL_COUNT)`
            # ——同一批帧同一个顺序，所以**位置 i 就是格位 i**。
            frames = apply_cell_exclusions(
                select_best_frames(track, n=REGISTRY_FRAMES_PER_SEAT),
                cell_exclusions_of(labels, seg.order, card_seat))
            # 勾掉几格之后用**该座位人工标定的**进出场候选补回来（用户
            # 2026-08-25）。被勾掉的那一格通常只是跟踪串号框错了人，不代表这
            # 位创业者没有别的画面；不补的话模型能看的证据凭空少四分之一。
            # 没有人工标定的候选就不补——见 `topup_frames` / 
            # `labelled_seat_candidates` 的 docstring（Task 26 的血教训）。
            # 补帧按**新座位号**查：用户是把那个人的进出场候选标成新号的。
            frames = topup_frames(frames, topup_by_seat.get(seat),
                                  want=REGISTRY_FRAMES_PER_SEAT)
            if not frames:
                # 四格全被勾掉：这个座位就此定不下来，落进 unresolved_seats
                # （与格位切分那一路同一条规矩）。留一条 frames 为空的记录等于
                # 假装它定下来了——核对页上是一张没有图的卡片、交付包里少一张
                # PNG，而「还缺几人」的数字一点没变。
                continue
            if merge_into is not None:
                # 人工那几张排在前面、一张不动：核对页与「排除格 N」都按位置
                # 数，往后追加不会动到已有那几格的编号（往前插会让用户上次勾
                # 的那一格指向别人）。
                room = REGISTRY_FRAMES_PER_SEAT - len(merge_into["frames"])
                # 去重按**画面本身**（时刻 + 框，`entryexit.picture_key` 同一
                # 口径），不按路径：同一张图、同一时刻里可能有两个人，只比路径
                # 与时刻会把另一个人扔掉；而同一张画面存进两个窗口目录时，比
                # 路径又会把它当成两张（用户 2026-09-26 电子10 第 13 段）。
                def _frame_key(path, t, box):
                    return (round(float(t), 3),) + tuple(round(float(v), 1) for v in box)
                have = {_frame_key(f["frame_path"], f["t"], f["box"])
                        for f in merge_into["frames"]}
                added = 0
                for i, d in enumerate(frames, start=1):
                    if added >= room:
                        break
                    key = _frame_key(d.frame_path, d.t, (d.x1, d.y1, d.x2, d.y2))
                    if key in have:
                        continue
                    merge_into["frames"].append(_frame_record(d))
                    # 格位与帧逐项对应（Task 56A）：并进来的就座帧，格位号就是
                    # 它在**自己那张**就座 2×2 上的位置。
                    merge_into["cell_indices"].append(i)
                    have.add(key)
                    added += 1
                continue
            seats[str(seat)] = _seat_record(
                seg_order=seg.order,
                seat=seat, source="seated", confirmed=False,
                track_id=tid, kind="seated", stable_id=None, cell_indices=[],
                note="", seat_uncertain=0, frames=frames,
            )
            counts["seated"] += 1

        unresolved = [s for s in range(1, int(seg.presenter_num) + 1)
                      if str(s) not in seats]
        seg_out[str(seg.order)] = {
            "company": seg.company,
            "presenter_num": int(seg.presenter_num),
            "rejected_seats": rejected,
            "unresolved_seats": unresolved,
            # Task 56C：未定座位分「人工判定确实没画面」与「还没人看过」，
            # 混在一个数字里既没法报进度，也没法挑出该加密抽帧的那几段。
            "unresolved_reason": unresolved_seat_reasons(
                labels, seg.order, unresolved),
            "seats": dict(sorted(seats.items(), key=lambda kv: int(kv[0]))),
        }

    n_resolved = sum(len(s["seats"]) for s in seg_out.values())

    # --- 评审席（组级，用户 2026-09-17）---------------------------------
    #
    # 挂在**顶层**而不是某一段底下：同一批评审人坐一整天、衣服不变，挂进段里
    # 会变成每家企业重复一遍（17×5 条纯冗余）。席位恒为 5，空缺留 None——
    # 那是必须看得见的缺口，人工才分得清「确实只来了 4 位」与「第 3 位被漏检」。
    #
    # 与创业者那 42 个座位**互不干扰**：左侧的人一个都不占座位（seats 那一路
    # 本来就按 `exclude_judges` 只看右半区），计数也各算各的。
    judge_cands = judge_candidates(dets)
    # 重抽之后把旧标注对回当前候选的标识（见 `judges.realign_judge_labels`）。
    # 不对齐就静默当作「这一位没人标过」——而对上多少条是工作有没有丢的**唯一
    # 可见迹象**，所以每次都打印。
    labels, _jstat = realign_judge_labels(judge_cands, labels)
    if _jstat["total"]:
        msg = f"   评审席标注：{_jstat['matched']}/{_jstat['total']} 条对上"
        if _jstat["moved"]:
            msg += f"（其中 {_jstat['moved']} 条按位置重新对齐）"
        if _jstat["dropped"]:
            msg += f"；{len(_jstat['dropped'])} 条找不到对应候选已丢弃：{_jstat['dropped']}"
        if _jstat["ambiguous"]:
            msg += f"；{len(_jstat['ambiguous'])} 条同时贴近两位候选、拒绝硬配：{_jstat['ambiguous']}"
        print(msg, flush=True)
    judge_slot_map = judge_slots(judge_cands, labels)

    # 归错人的格换到那一位名下（用户 2026-09-18：「专家的每个格有可能是其他
    # 人，给出这个修改选项」）。只勾掉是不够的——城市1 实测四位专家共勾掉
    # 7 格，专家1 与专家3 各只剩 1 格，而模型拿一格判整场衣着比拿四格混着两
    # 个人还糟。那些格没有坏，只是归错了人。
    #
    # 先把全部改判收齐再分配：一边遍历一边送，源与目标的先后次序就会影响结果
    # （互换时尤其明显），而次序取决于字典顺序，换个 Python 版本都可能变。
    donations: dict[int, list[tuple[str, int, object]]] = {}
    for c in judge_cands:
        for cell, target in judge_cell_assignments(labels, c["stable_id"]).items():
            if target is None or not (1 <= cell <= len(c["pool_dets"])):
                continue
            donations.setdefault(int(target), []).append(
                (c["stable_id"], cell, c["pool_dets"][cell - 1]))
    # 坐在**右侧**的评审人（用户 2026-09-19：文化4 的专家5 在右侧靠投影仪那头）。
    # `judge_candidates` 只看左区，他一个候选都出不来；但右区的人都进了创业者
    # 候选，所以人工在逐格身份里标「专家 N」，那一帧就从这里收进来。两条来源
    # 产出的三元组同形，合并后一视同仁。
    for n, got in judge_donations_from_candidates(
            labels, {seg.order: candidates_for_labels(dets, seg, 1440, 1080)
                     for seg in segments}).items():
        donations.setdefault(n, []).extend(got)
    for got in donations.values():
        got.sort(key=lambda x: (x[0], x[1]))
    judge_out = {}
    _kept: dict[int, list] = {}
    _taken: dict[int, list] = {}
    _left: dict[int, list] = {}
    _left_picks: dict[int, list] = {}
    for n in range(1, config.JUDGE_COUNT + 1):
        c = judge_slot_map.get(n)
        # 人工逐格核实的结果在这里落地一次（用户 2026-09-18：「有的格子混进了
        # 其他专家」）。`frames` 是下游**唯一**的来源——核对页、交付包的证据
        # 图、送模型的那张都由它拼；在这里滤一次三处同时干净，分头各滤一遍必
        # 有一处漏掉，而漏掉的表现是模型照旧看见了别人的衣服，不报错。
        ex = set() if c is None else judge_cell_exclusions(labels, c["stable_id"])
        # 哪几格组成 2×2（用户 2026-09-18：「不需要总是勾选 4 个，默认选的被
        # 标记为其他人后，再从后面标的选中进 2×2 递补」）：
        #
        #   默认那四格里还算本人的  +  人工补选的，按他勾的次序
        #
        # 挑选是**递补**，不是整组替换：城市1 专家5 只把格1 标成其他人、补选了
        # 格6，期望的是 2、3、4、6——整组替换的话他只剩格6 一格，另外两格好端端
        # 的画面凭空没了（2026-09-18 实测他导出的正是这一份）。
        #
        # 空出来的位置**不自动顶上**：自动顺序只按面积×清晰度往下顺，人却会
        # 以为那一格是他挑的。宁可 2×2 少一格（页面上写着还差几格），也不悄悄
        # 塞一张没人看过的脸进去。
        #
        # 挑中却已标成「其他人」的格不作数：两条相反的指示里，说「这不是他」的
        # 那条更具体，硬送进去就是把别人的脸交给模型。
        _picked = [] if c is None else judge_picks(labels, c["stable_id"])
        _pool = [] if c is None else list(c["pool_dets"])
        _base = [i for i in range(1, min(config.JUDGE_CELL_COUNT, len(_pool)) + 1)
                 if i not in ex]
        _extra = [i for i in _picked
                  if i not in ex and i not in _base and 1 <= i <= len(_pool)]
        _order = (_base + _extra)[:config.JUDGE_CELL_COUNT]
        # 补选多于空缺时，多出来的要说出来——不然人勾了一格、图上没有它。
        _left_picks[n] = [i for i in _extra if i not in _order]
        _own = [_pool[i - 1] for i in _order]
        # **别人还回来的格排在自己的备选之前。** 有了备选池之后本人这边总能
        # 补满四格，按「本人先占」就永远轮不到还回来的那一张——而那一张是人
        # **明确指认过**的，备选只是分数往下顺的下一张，很可能还是同一个错的
        # 人。让未经确认的顶掉已确认的，正好把这件事做反了。
        # 别人还回来的格补在后面：本人这边（默认四格 + 补选）先占，剩下的位置
        # 才轮到它。装不下的照旧报出来，人看得见就能回去调。
        _keep_first = _own
        _all = donations.get(n, [])
        _cap = max(0, config.JUDGE_CELL_COUNT - len(_keep_first))
        _taken[n] = _all[:_cap]
        # 装不下的**必须说出来**：默默丢掉的话，人指了一格过去、图上没有它、
        # 页面上一个字都没有，他只会以为自己点错了。
        _left[n] = _all[_cap:]
        _kept[n] = _keep_first + [d for _sid, _k, d in _taken[n]]
        if c is None and not _kept.get(n):
            judge_out[str(n)] = None
            continue
        if c is None:
            # 这个号本身没有候选，全靠别人还回来的格撑着（右侧的评审人就是
            # 这样进来的）。**标识留空**：编一个就等于凭空造出一位「检测到的
            # 评审人」，下游分不清他是检出来的还是人工拼出来的。
            judge_out[str(n)] = {
                "judge": n, "stable_id": None,
                "median_cx": None, "median_y2": None, "median_h": None,
                "height_ratio": None,
                "n_frames": len(_kept[n]), "n_segments": 0, "segments": [],
                "confirmed": True, "excluded_cells": [], "picked_cells": [],
                "unplaced_picks": [],
                "donated_cells": [{"from": sid, "cell": k}
                                  for sid, k, _d in _taken.get(n, [])],
                "unplaced_cells": [{"from": sid, "cell": k}
                                   for sid, k, _d in _left.get(n, [])],
                "frames": [_frame_record(d) for d in _kept[n]],
            }
            continue
        judge_out[str(n)] = None if c is None else {
            "judge": n,
            "stable_id": c["stable_id"],
            "median_cx": round(c["median_cx"], 1),
            "median_y2": round(c["median_y2"], 1),
            "median_h": round(c["median_h"], 1),
            "height_ratio": round(c["height_ratio"], 3),
            "n_frames": c["n_frames"],
            "n_segments": c["n_segments"],
            "segments": list(c["segments"]),
            # 人工确认状态与座位那一路同一条口径：几何排序给的基线一律
            # False，只有标注文件里点过名的才是 True。
            "confirmed": bool(judge_labels_of(labels).get(c["stable_id"])),
            # 留下「勾掉了哪几格」：只看剩下三帧分不出「第 2 格被勾掉了」与
            # 「这人本来只拍到三帧」，而这两件事的处置完全不同。
            "excluded_cells": sorted(ex),
            # 这四格是人挑的还是自动排的——两者的可信度完全不同，登记表里
            # 分不出来的话，下游与人工都无从判断这一位核过没有。
            "picked_cells": list(_picked),
            "unplaced_picks": list(_left_picks.get(n, [])),
            # 收进来的格：本人的先占，2×2 装不下的不要。悄悄多带几帧会让
            # 登记表里的 frames 与图上看到的四格对不上，而一处不报错。
            "donated_cells": [{"from": sid, "cell": k}
                              for sid, k, _d in _taken.get(n, [])],
            "unplaced_cells": [{"from": sid, "cell": k}
                               for sid, k, _d in _left.get(n, [])],
            "frames": [_frame_record(d) for d in _kept.get(n, [])],
        }
    n_judges = sum(1 for v in judge_out.values() if v)
    judges_block = {
        "n_expected": config.JUDGE_COUNT,
        "n_resolved": n_judges,
        "unresolved": [n for n in range(1, config.JUDGE_COUNT + 1)
                       if judge_out[str(n)] is None],
        # 没排进席位的候选一并带上：几何基线只按进深排，一个贴着画面边缘的人
        # 插进序列就会把后面的整体顶掉一位（实测城市2 就是这样挤掉了真正的
        # 第 5 位）。人工要能把被挤掉的那位提回来，就必须看得到他。
        "extra_candidates": [
            {"stable_id": c["stable_id"], "median_cx": round(c["median_cx"], 1),
             "median_y2": round(c["median_y2"], 1),
             "n_frames": c["n_frames"], "n_segments": c["n_segments"]}
            for c in judge_cands
            if c["stable_id"] not in {v["stable_id"] for v in judge_out.values() if v}
        ],
        "slots": judge_out,
    }

    return {
        "_meta": {
            "schema": REGISTRY_SCHEMA,
            "sheet": sheet,
            "detections_fingerprint": detections_fp,
            # Task 56B：登记表产自哪一份人工标注。摘要是**内容哈希**，
            # mtime 只作辅助提示（变新变旧都骗人，不参与判定）。
            "labels_file": (Path(labels_path).name if labels_path else ""),
            "labels_digest": (labels_digest(labels_path) if labels_path else ""),
            "labels_mtime": (_mtime_text(labels_path) if labels_path else None),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_segments": len(seg_out),
            "n_seats_expected": n_expected,
            "n_seats_resolved": n_resolved,
            "counts_by_source": counts,
            "n_judges_expected": config.JUDGE_COUNT,
            "n_judges_resolved": n_judges,
        },
        "segments": seg_out,
        "judges": judges_block,
    }


# ---------------------------------------------------------------------------
# Task 54 · 需求二：核对页改用 **VLM 实际看到的图**
#
# 用户 2026-08-23 原话：「不需要给出座位核对.html，给我看 VLM 看的图让我核对
# 更准」。
#
# 此前控制器拿登记表里的框自己裁了一份 `座位核对.html`——那不是 VLM 看到的
# 东西：裁法、留边、红框、格位切分全都可能与 `assemble` 送进模型的那张不同，
# 而人正是拿这页去判断「这个座位是不是这个人」。核对了一张模型没看过的图，
# 结论对模型不成立，且没有任何一处会报错。
#
# 因此本页**只做一件事**：把登记表里那几帧原样喂给 `annotate.
# build_evidence_sheet`——与 `assemble` 送进模型前的最后一步是同一个函数、
# 同一批帧、同一组默认参数（不带格位编号：VLM 那份就不带）。三种来源与
# assemble 的对应关系：
#
#   | source     | 登记表里的 frames        | assemble 送进模型的        |
#   |------------|--------------------------|----------------------------|
#   | seated     | select_best_frames(n=4)  | 同一句（第 1 步 evidence_seated）|
#   | entryexit  | 候选的 best_dets[:4]     | 同一份 cand["best_dets"]   |
#   | cell_split | cells_dets(cand, 该座位那几格) | _cell_dets(cand, 代表那一格) |
#
# 所以「格位切分的座位不是 2×2 四格」不是本页另做的排版，而是那个座位的证据
# 图本来就只有它那几格——沿用 2×2 的排法会让人核对四格里另外三个人。
#
# Task 56A · 最后一行两侧**目前并不完全一致**，必须说清楚：同一个人占两格时
# 登记表留下两张（用户要的就是这个），而 `assemble` 每个座位仍只送代表那一
# 格的一张（`item["cell_index"]`）。本任务的边界只到登记表与本页；让打标那
# 一侧也用上多格是下一步。写在这里而不是让它自生自灭：本页的全部价值就是
# 「核对的是模型看的那张」，两侧何时不一致必须留字，否则日后没人知道。
# ---------------------------------------------------------------------------

def preview_path(work: Path, sheet: str) -> Path:
    """核对页的落盘路径 `work/Output/<组>/vlm_preview_<SHEET>.html`
    （命名与登记表一致，位置同样由 `paths` 单一决定）。"""
    return paths.preview_path(work, sheet)


_SOURCE_TEXT = {
    "seated": "就座阶段（几何排序，未经人工确认）",
    "entryexit": "进出场人工递补",
    "cell_split": "格位切分（从一张挤了多人的合图里切出来的那一格）",
}


def _preview_frames(rec: dict) -> tuple[list[Path], list[tuple]]:
    """登记表一条座位记录里的帧 → `build_evidence_sheet` 的两个入参。"""
    paths, boxes = [], []
    for fr in rec.get("frames") or []:
        box = fr.get("box") or []
        if not fr.get("frame_path") or len(box) != 4:
            continue
        paths.append(Path(fr["frame_path"]))
        boxes.append(tuple(float(v) for v in box))
    return paths, boxes


def _local_text(iso: str | None) -> str:
    """把 `_meta.built_at`（UTC 的 ISO 串）按**本地时区**印出来。

    为什么要换算：用户拿这个时间去和 Finder 里 `entry_labels_*.json` 的修改
    时间比对（本地时区），那次事故正是「14:03 导出、页面停在 13:53」这十
    分钟之差；中间再隔一次时区换算，差十分钟还是差八小时零十分就都看不出来
    了。解析不了就原样印（宁可难看，不可编一个时间出来）。
    """
    if not iso:
        return "（未记录生成时刻）"
    try:
        return datetime.fromisoformat(str(iso)).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(iso)


def _preview_stamp(meta: dict) -> str:
    """核对页页首那行「本页产自哪一份标注、什么时刻生成」的钢印（HTML 片段）。

    Task 56B：用户就是对着这张页面一个座位一个座位地核对的，页面自己得能
    自证新鲜度。2026-08-23 那次，页面停在 13:53、标注是 14:03 的，页面上
    却没有任何一处写着它产自哪一份标注——用户只能去比文件时间戳。

    印摘要前 8 位而不是全部：够用来和 `--check` 打印的摘要对上一眼，又不至
    于长到让人不看。时间照印 `built_at`（登记表的生成时刻）。
    """
    digest = meta.get("labels_digest")
    built = _local_text(meta.get("built_at"))
    name = meta.get("labels_file") or "entry_labels_<组>.json"
    # 那份标注**当时**的修改时间（仅提示，判定靠摘要）：用户一眼就能回答
    # 「我最后那次 14:03 导出的到底进没进这张页面」。
    lm = meta.get("labels_mtime")
    if digest is None:
        return _esc(
            "⚠ 本页产自的登记表是旧版，未记录它由哪一份人工标注产出，"
            "无法确认是不是最新那次标注的结果——请重跑 "
            f"`{_REBUILD_CMD}` 再核对。（登记表生成于 {built}）")
    if not digest:
        return _esc(
            f"本页产自「没有人工标注文件」的登记表：全部座位都只是就座阶段的"
            f"几何排序结果，尚未经过人工复核。（登记表生成于 {built}）")
    return _esc(
        f"本页产自标注 {name} · 内容摘要 {str(digest)[:8]}"
        + (f" · 该标注当时的修改时间 {lm}" if lm else "")
        + f" · 登记表生成于 {built}。核对前请先确认这个摘要就是你最后一次"
          f"导出的那一份（`python -m pipeline.registry --check` 会当场告诉"
          f"你）。")


def _preview_toc(segs: dict) -> str:
    """核对页左侧目录：每段一行，写企业名、已定/应有、还缺哪几个座位。

    用户 2026-08-29：「VLM html 左侧也做个目录」。挑选页 Task 47 起就有一份
    （用户原话「一点滑到对应的标记处，方便定位和知道有哪些企业缺人」），核对
    页几十段滚下来同样找不到位置，两页来回对照时只有一边能定位。

    **颜色从来不是唯一区分**：每行另带 ⛔/⚠/✓ 与「缺 N 人」的文字。只靠背景
    色对色觉障碍者不可用，截图或打印成灰度后也全部失效——而这页正是边看边
    核对、常要截图存证的那一页。这条与挑选页目录逐字同一个理由。

    「已定/应有」印的是 `len(seats)/presenter_num`，不是 `unresolved` 的长度：
    前者是用户判断「还剩多少活」的那个数，后者只在缺人时才有意义。
    """
    rows: list[str] = []
    for order in sorted(segs, key=lambda k: int(k)):
        seg = segs[order] or {}
        n_seat = len(seg.get("seats") or {})
        n_want = int(seg.get("presenter_num") or 0)
        missing = list(seg.get("unresolved_seats") or [])
        if missing:
            cls, mark = ("toc-miss", "⛔")
            tail = (f'<span class="toc-miss-n">缺 {len(missing)} 人'
                    f'（座位 {"、".join(str(s) for s in missing)}）</span>')
        else:
            cls, mark = ("toc-done", "✓")
            tail = ""
        rows.append(
            f'<a class="toc-row {cls}" href="#seg-{int(order)}">'
            f'<span class="toc-mark">{mark}</span>'
            f'<span class="toc-name">#{int(order)} {_esc(seg.get("company") or "")}'
            f'</span>'
            f'<span class="toc-nums">{n_seat}/{n_want} 座位已定</span>'
            f'{tail}</a>')
    n_miss = sum(1 for o in segs if (segs[o] or {}).get("unresolved_seats"))
    return (
        '<nav class="toc">__JUDGE_TOC__'
        '<div class="toc-title" style="margin-top:10px">分段目录</div>'
        f'<div class="toc-summary">共 {len(segs)} 段，'
        + (f'{n_miss} 段仍缺人' if n_miss else '全部段落座位已定')
        + '</div>' + "".join(rows) + '</nav>')


def _preview_judge_toc(block: dict) -> str:
    """核对页目录里的评审席一栏（用户 2026-09-18）。

    两个页面都要有：只在挑选页给，人到核对页复核时又找不着评审席在哪一屏。

    位置（cx,y2）逐位写出来——席位号来自几何基线（只按进深排），一个贴着画面
    边缘的人插进序列就会把后面的整体顶掉一位。城市2 实测 `judge#35,668`
    （cx=35 紧贴左边缘、266 帧）占了 3 号位，而真正那位在 cx=565、625 帧；
    五个位置摆在一起，这种错位一眼就看出来。
    """
    slots = (block or {}).get("slots") or {}
    n_want = (block or {}).get("n_expected", config.JUDGE_COUNT)
    rows = []
    for n in range(1, int(n_want) + 1):
        rec = slots.get(str(n))
        if not rec:
            rows.append(f'<a class="toc-row toc-miss" href="#judges">'
                        f'<span class="toc-mark">⛔</span>'
                        f'<span class="toc-name">专家 {n}</span>'
                        f'<span class="toc-nums">尚无候选</span></a>')
            continue
        short = config.JUDGE_CELL_COUNT - len(rec.get("frames") or [])
        cls, mark = (("toc-miss", "⚠") if short > 0 else ("toc-done", "✓"))
        rows.append(
            f'<a class="toc-row {cls}" href="#judge-{n}">'
            f'<span class="toc-mark">{mark}</span>'
            f'<span class="toc-name">专家 {n}</span>'
            f'<span class="toc-nums">位置 {_esc(str(rec.get("median_cx")))},'
            f'{_esc(str(rec.get("median_y2")))} · '
            + (f"差 {short} 格" if short > 0 else "四格齐")
            + '</span></a>')
    # 没入席的候选只在**还有空号**时列进目录（用户 2026-09-18：「专家5位已齐，
    # 不需要『未入席1位』的目录了」）。那一行是「从这里把人提上来补位」的入口，
    # 坐满之后只是噪声，而目录越短越好用。
    #
    # 只去掉目录这一行：评审席区块里的候选照旧写明数量，否则哪天发现坐错了号
    # （城市2 的 `judge#35,668` 就是贴着画面边缘挤进 3 号位的），想把人提回席位
    # 却再也找不着他。
    extra = (block or {}).get("extra_candidates") or []
    if extra and (block or {}).get("unresolved"):
        pos = "；".join(f'{_esc(str(c.get("median_cx")))},'
                       f'{_esc(str(c.get("median_y2")))}' for c in extra)
        rows.append(f'<a class="toc-row toc-miss" href="#judges">'
                    f'<span class="toc-mark">?</span>'
                    f'<span class="toc-name">未入席 {len(extra)} 位</span>'
                    f'<span class="toc-nums">{pos}</span></a>')
    if not rows:
        return ""
    return ('<div class="toc-title">评审席</div>'
            '<div class="toc-summary">画面左侧一列，离镜头最近的是专家 1</div>'
            + "".join(rows))


def _preview_judges(judge_cards: list[dict], block: dict) -> str:
    """核对页顶部的评审席一栏。

    席位恒为 `JUDGE_COUNT`，空号作为缺口列出来——人工才分得清「确实没到场」
    与「被漏检了」。没排进席位的候选一并写明数量：几何基线只按进深排，一个
    贴着画面边缘的人插进序列会把后面的整体顶掉一位。
    """
    if not judge_cards:
        return ""
    cells = []
    for c in judge_cards:
        rec = c["rec"] or {}
        if c["png"]:
            img = (f'<img src="data:{c.get("media", "image/png")};base64,'
                   f'{base64.b64encode(c["png"]).decode()}">')
        else:
            img = '<div class="noimg"></div>'
        meta = f'<div class="why">{_esc(c["why"])}</div>' if not c["png"] else ""
        if rec:
            meta += (f'{_esc(str(rec.get("n_frames", 0)))} 帧 · 出现在 '
                    f'{_esc(str(rec.get("n_segments", 0)))} 段'
                    + ("　<b>人工已确认</b>" if rec.get("confirmed")
                       else "　<i>几何基线，尚未经人工确认</i>"))
            # 图上少一格时，人得分清那是他自己勾掉的、还是这一位本来就只拍到
            # 三帧——前者已核实，后者还欠一次。不写出来两件事长得一模一样。
            # 这四格是人挑的还是按清晰度自动排的——两者可信度完全不同，
            # 而页面上长得一模一样。自动排的那几位还欠一次核实。
            pk = rec.get("picked_cells") or []
            meta += ('　<b>人工挑选</b>：格 '
                     + "、".join(_esc(str(k)) for k in pk)
                     if pk else "　<i>按清晰度自动取的前四格，尚未经人工挑选</i>")
            lp = rec.get("unplaced_picks") or []
            if lp:
                meta += ('　<b>补选没装下</b>：'
                         + "、".join(f"格 {_esc(str(k))}" for k in lp)
                         + f"（2×2 只有 {config.JUDGE_CELL_COUNT} 格）")
            ex = rec.get("excluded_cells") or []
            if ex:
                meta += ('　<b>人工勾掉</b>：'
                         + "、".join(f"格 {_esc(str(k))}" for k in ex)
                         + "（不是本人，已不送模型）")
            # 收进来的那一格在图上与本人的格长得一样，不写出来人就无从复核
            # 自己指得对不对。
            got = rec.get("donated_cells") or []
            if got:
                meta += ('　<b>收进</b>：'
                         + "、".join(f"{_esc(str(x.get('from')))} 的格 "
                                    f"{_esc(str(x.get('cell')))}" for x in got))
            # 装不下的更要说——默默丢掉的话他只会以为自己点错了。
            left = rec.get("unplaced_cells") or []
            if left:
                meta += ('　<b>没装下</b>：'
                         + "、".join(f"{_esc(str(x.get('from')))} 的格 "
                                    f"{_esc(str(x.get('cell')))}" for x in left)
                         + f"（2×2 只有 {config.JUDGE_CELL_COUNT} 格，本人的格先占）")
        cells.append(
            f'<div class="card jcard" id="judge-{c["judge"]}" '
            f'data-judge="{c["judge"]}">{img}'
            f'<div class="info"><div class="jnum">专家 {c["judge"]}</div>'
            f'<div class="jmeta">{meta}</div></div></div>')
    extra = block.get("extra_candidates") or []
    note = (f'　另有 <b>{len(extra)}</b> 位左侧的人没排进席位（在挑选页上可改号指过来）'
            if extra else "")
    return (f'<h2 id="judges">本组评审人</h2>'
            f'<p class="lead">画面左侧那一列，离镜头最近的是<b>专家 1</b>，'
            f'最靠大屏幕的是专家 {block.get("n_expected", config.JUDGE_COUNT)}。'
            f'他们整场不换位置，这一栏<b>全组共用</b>，不按企业分；'
            f'已排定 {block.get("n_resolved", 0)}/'
            f'{block.get("n_expected", config.JUDGE_COUNT)}。{note}</p>'
            f'<div class="jrow">{"".join(cells)}</div>')


def render_vlm_preview(registry: dict, out_path: Path) -> Path:
    """把座位登记表渲染成「**VLM 会看到的那张图**」的人工核对页。

    每个座位一张卡片，写明：座位号、这个人是**怎么定下来的**（source）、
    `cell_indices`（若有，写成「格位 1+2」）、是否人工确认过、是否跨段流转，
    以及**这张图是不是
    就是 VLM 会看到的那张**——最后这一条是本页存在的理由，必须逐卡写出来
    而不是在页首笼统说一句：某个座位的帧文件已经不在磁盘上时，那张卡片就
    **不是**，此时说「这就是模型看的图」等于骗人。

    证据图一律由 `annotate.build_evidence_sheet` 生成（与 `assemble` 同一个
    函数、同一批帧、同一组默认参数），本模块**绝不自己裁一份**——两处各裁
    一份必然走样，而走样的表现是人核对了一张模型没看过的图。

    体积保护复用 `gallery._fit_images_to_budget`（先量后缩），与挑选页同一
    套预算，不另写一份。
    """
    out_path = Path(out_path)
    meta = (registry or {}).get("_meta") or {}
    segs = (registry or {}).get("segments") or {}

    cards: list[dict] = []
    for order in sorted(segs, key=lambda k: int(k)):
        seg = segs[order] or {}
        for seat in sorted((seg.get("seats") or {}), key=lambda k: int(k)):
            rec = seg["seats"][seat] or {}
            paths, boxes = _preview_frames(rec)
            png, why = None, ""
            if not paths:
                why = "本座位没有留档帧，模型那边同样拿不到图"
            else:
                try:
                    png = build_evidence_sheet(paths, boxes)
                except (OSError, ValueError) as exc:
                    why = f"帧文件读不到（{exc}），无法重建模型会看到的那张图"
            cards.append({
                "order": int(order), "company": seg.get("company") or "",
                "presenter_num": seg.get("presenter_num"),
                "seat": int(seat), "rec": rec, "paths": paths,
                "png": png, "why": why,
            })

    # 评审席（组级，用户 2026-09-17）。图与座位那些走同一个
    # `build_evidence_sheet`、同一份体积预算——另裁一份或另起一套预算，
    # 表现就是人核对的图与模型看到的不是同一张。
    judge_cards: list[dict] = []
    jb = (registry or {}).get("judges") or {}
    for n in sorted((jb.get("slots") or {}), key=lambda k: int(k)):
        rec = (jb["slots"] or {}).get(n)
        if not rec:
            judge_cards.append({"judge": int(n), "rec": None, "png": None,
                                "paths": [], "why": "尚无候选"})
            continue
        paths, boxes = _preview_frames(rec)
        png, why = None, ""
        if not paths:
            why = "这一位没有留档帧，模型那边同样拿不到图"
        else:
            try:
                png = build_evidence_sheet(paths, boxes)
            except (OSError, ValueError) as exc:
                why = f"帧文件读不到（{exc}）"
        judge_cards.append({"judge": int(n), "rec": rec, "paths": paths,
                            "png": png, "why": why})

    imgs = ([{"data": c["png"], "media": "image/png"} for c in cards]
            + [{"data": c["png"], "media": "image/png"} for c in judge_cards])
    _fit_images_to_budget(imgs, int(config.GALLERY_MAX_MB * 1024 * 1024))
    for jc, img in zip(judge_cards, imgs[len(cards):]):
        jc["png"] = img["data"]
        jc["media"] = img["media"]
    for card, img in zip(cards, imgs[:len(cards)]):
        card["png"] = img["data"]
        card["media"] = img["media"]

    parts: list[str] = []
    last_order = None
    for card in cards:
        if card["order"] != last_order:
            last_order = card["order"]
            unresolved = (segs[str(card["order"])] or {}).get("unresolved_seats") or []
            # 锚点（Task 56U）：目录靠它跳转。挂在 h2 上而不是另起一个空
            # 元素——空锚点会被浏览器滚到标题**下方**，用户看到的第一行是
            # 卡片而不是企业名，反而认不出跳到哪一段了。
            parts.append(
                f'<h2 id="seg-{card["order"]}">#{card["order"]} '
                f'{_esc(card["company"])} · '
                f'标注{_esc(str(card["presenter_num"]))}人'
                + (f' · <span class="miss">座位 '
                   f'{_esc("、".join(str(s) for s in unresolved))} 仍未定</span>'
                   if unresolved else "")
                + "</h2>")
        rec = card["rec"]
        # Task 56A：一个座位可能占多格（「格位2 = 同格位1」是同一个人），
        # 卡片上要把**每一格**都写出来。只显示其中一格等于让用户拿一半的
        # 证据去核对——他看不出少了什么，也就不会发现另一格框错了人。
        cells = [int(c) for c in (rec.get("cell_indices") or [])]
        cell_label = "+".join(str(c) for c in cells)
        bits = [f'座位 {card["seat"]}',
                _SOURCE_TEXT.get(rec.get("source"), str(rec.get("source")))]
        if cells:
            bits.append(
                f"格位 {cell_label}（2×2 里的第 {cell_label} 格，"
                + ("单张" if len(cells) == 1
                   else f"同一个人占 {len(cells)} 格，共 {len(cells)} 张")
                + "）")
        bits.append("人工已确认" if rec.get("confirmed") else "未经人工确认")
        if rec.get("seat_uncertain"):
            bits.append("座位号是推断的（seat_uncertain=1）")
        if rec.get("cross_segment"):
            bits.append(f'跨段流转：画面来自第 {rec.get("src_order")} 段的时间窗口')
        # 「这张图是不是 VLM 会看到的那张」——本页存在的全部理由，逐卡写死。
        if card["png"] and len(cells) == 1:
            vlm = ("✔ 就是模型会看到的那张：格位切分的座位只送**这一格的单张**"
                   "画面，不是四格拼图")
        elif card["png"] and cells:
            vlm = (f"✔ 这就是本座位留档的全部证据：他占了第 {cell_label} 格，"
                   f"共 {len(cells)} 张（同一个人的多张画面，不是四格拼图里"
                   f"的别人）")
        elif card["png"]:
            vlm = "✔ 就是模型会看到的那张：同一批帧、同一个 build_evidence_sheet 拼的 2×2"
        else:
            vlm = f'✘ 这里没有图：{card["why"]}'
        img_html = (f'<img src="data:{card["media"]};base64,'
                    f'{base64.b64encode(card["png"]).decode()}">'
                    if card["png"] else '<div class="noimg"></div>')
        frames = "".join(
            f'<li>{_esc(str(p))}</li>' for p in card["paths"]) or "<li>（无）</li>"
        parts.append(
            f'<div class="card" data-order="{card["order"]}" '
            f'data-seat="{card["seat"]}" data-source="{_esc(str(rec.get("source")))}" '
            f'data-cell="{_esc(cell_label)}" '
            f'data-vlm="{"1" if card["png"] else "0"}">'
            f'{img_html}<div class="info"><div class="ttl">'
            f'{_esc(" · ".join(bits))}</div>'
            f'<div class="vlm">{vlm}</div>'
            f'<div class="note">{_esc(rec.get("note") or "")}</div>'
            f'<details><summary>用到的 {len(card["paths"])} 帧</summary>'
            f'<ul>{frames}</ul></details></div></div>')

    n_img = sum(1 for c in cards if c["png"])
    html = _PREVIEW_TEMPLATE.replace("__IMG_SCALE__", str(config.GALLERY_IMG_SCALE))
    # 评审席排在分段目录**之前**：它是全组共用的一块、也是最先要核的东西，
    # 排在十几行分段后面就要滚到底才看得见（与挑选页同一条口径）。
    html = html.replace("__TOC__",
                        _preview_toc(segs).replace("__JUDGE_TOC__",
                                                   _preview_judge_toc(jb)))
    # 目录宽与正文左边距只此一处来源（与挑选页同一个常量）：两处各写一个
    # 数字，改宽度时必然只改一处，表现是目录压住正文或留一条空带。
    html = html.replace("__TOC_W__", str(config.ENTRY_REVIEW_TOC_WIDTH_PX))
    html = html.replace("__TOC_PAD__",
                        str(config.ENTRY_REVIEW_TOC_WIDTH_PX + 24))
    html = html.replace("__STAMP__", _preview_stamp(meta))
    html = html.replace("__SUMMARY__", _esc(
        f'{meta.get("sheet") or ""}：{len(segs)} 段、{len(cards)} 个已定座位，'
        f'其中 {n_img} 个能重建出模型会看到的那张图'
        f'（检测指纹 {meta.get("detections_fingerprint")}，'
        f'登记表生成于 {meta.get("built_at")}）'))
    html = html.replace("__JUDGES__", _preview_judges(judge_cards, jb))
    html = html.replace("__BODY__", "".join(parts) or
                        '<p class="miss">登记表里一个已定座位都没有。</p>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


_PREVIEW_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>座位核对 · VLM 实际看到的图</title>
<style>
 :root{--img-scale: __IMG_SCALE__}
 body{font-family:-apple-system,"PingFang SC",sans-serif;margin:24px;
      background:#fafafa;color:#222;padding-left:__TOC_PAD__px}
 /* 锚点跳转后标题不被页顶切掉；这页没有固定顶栏，留一行的余量就够 */
 h2{border-bottom:2px solid #ddd;padding-bottom:6px;margin-top:32px;
    scroll-margin-top:12px}
 /* ---- 分段目录（Task 56U）。左侧竖排固定栏，自己可滚（60+ 段也放得下）。
    配色之外一律另有 ⛔/✓ 与文字，理由见 _preview_toc 的 docstring。---- */
 .toc{position:fixed;left:0;top:0;bottom:0;width:__TOC_W__px;z-index:98;
      overflow-y:auto;box-sizing:border-box;padding:12px 10px;
      background:#fff;border-inline-end:1px solid #ccc;font-size:13px}
 .toc-title{font-weight:700;font-size:14px;margin-bottom:2px}
 .toc-summary{color:#555;font-size:12px;margin-bottom:8px;line-height:1.5}
 .toc-row{display:block;text-decoration:none;color:#222;border-radius:6px;
          padding:5px 8px;margin-bottom:4px;border-left:4px solid transparent;
          line-height:1.45}
 .toc-miss{background:#fdecea;border-left-color:#c62828;font-weight:700}
 .toc-done{background:#f2f5f2;border-left-color:#9bb39b;color:#5a6b5a}
 .toc-mark{display:inline-block;width:1.2em}
 .toc-name{font-size:13px}
 .toc-nums{display:block;font-size:12px;font-weight:400}
 .toc-miss-n{display:block;font-size:12px;color:#b3261e;font-weight:700}
 /* 窄屏（分屏、投影）下固定栏会把正文挤没，退回普通块级元素排在页首 */
 @media (max-width:900px){
   .toc{position:static;width:auto;max-height:40vh;border-right:0;
        border-bottom:1px solid #ccc}
   body{padding-left:0}
 }
 .lead{background:#eef4fb;border-left:4px solid #4a7fb5;padding:8px 12px;
       font-size:14px;line-height:1.7}
 /* 钢印要显眼：用户此前是靠肉眼比对文件时间戳才发现页面比标注旧了 */
 .stamp{background:#fff3cd;border:2px solid #d39e00;border-radius:6px;
        padding:10px 12px;font-size:15px;font-weight:700;line-height:1.7;
        color:#5c4400}
 .card{display:flex;gap:12px;background:#fff;border:1px solid #ddd;
       border-radius:8px;padding:10px;margin:8px 0;align-items:flex-start}
 .card img{width:calc(var(--img-scale) * 100%);max-width:100%;height:auto;
           border-radius:4px;display:block}
 .card>img,.card>.noimg{flex:0 0 62%;min-width:0}
 /* 一位一行，尺寸完全交给 `.card`（与座位卡片同一条规则）——用户
    2026-09-18：五张挤一行太小看不清衣服。另抄一份尺寸必然日后走样。 */
 .jrow{margin:8px 0 18px}
 .jcard{border-color:#d6e0f2}
 .jnum{font-weight:bold;margin-bottom:4px}
 .jmeta{color:#555;margin-top:4px}
 .jcard .why{color:#7a5200;background:#fff7e6;border-radius:4px;padding:4px;margin-top:4px}
 .noimg{aspect-ratio:1/1;background:#eee;border-radius:4px}
 .info{flex:1;min-width:0;font-size:14px;line-height:1.7}
 .ttl{font-weight:700}
 .vlm{margin-top:4px;color:#1b5e20}
 .note{color:#555;font-size:13px}
 .miss{color:#b3261e;font-weight:700}
 li{font-size:12px;color:#555;word-break:break-all}
</style>
__TOC__
<h1>座位核对 · 这里显示的就是 VLM 会看到的图</h1>
<p class="stamp">__STAMP__</p>
<p class="lead">每张图都由 <code>annotate.build_evidence_sheet</code> 用座位
登记表里留档的那几帧现拼出来——与 <code>assemble</code> 送进模型前的最后一步
是同一个函数、同一批帧、同一组参数。<b>不是</b>另外裁的一份预览：核对一张
模型没看过的图，得出的结论对模型不成立，而且不会有任何一处报错。<br>
格位切分的座位只有**他占的那几格**的画面（同一个人占两格时两张都在）；就座与
进出场递补的座位是 2×2 拼图，红框标出的就是目标本人。<br>
__SUMMARY__</p>
__JUDGES__
__BODY__
"""


def write_seat_registry(registry: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


def _sheet_of(path: Path, meta: dict) -> str:
    """登记表属于哪一组：优先信 `_meta.sheet`，退而从文件名反推。

    要组名是为了找到**配套的那一份标注**（`entry_labels_<组>.json`）。旧
    登记表可能没记 sheet（`build_seat_registry` 的 `sheet` 可以不传），此时
    文件名里那一段就是唯一线索——猜错组名会拿另一组的标注去比，那是比不
    报警更坏的一种报警。
    """
    sheet = meta.get("sheet")
    if sheet:
        return str(sheet)
    stem = Path(path).stem
    return stem[len("seat_registry_"):] if stem.startswith("seat_registry_") else ""


def load_seat_registry(path: Path, detections_fp: str | None = None, *,
                       allow_stale: bool = False) -> dict:
    """读回登记表并做指纹校验；对不上一律**整份作废**并打印。

    决策表（与 `entryexit.load_entry_labels` 同一套口径）：

      | 情形                    | 处理           | 打印                       |
      |-------------------------|----------------|----------------------------|
      | 文件不存在              | 返回 `{}`      | 不打印（登记表是可选输入） |
      | 内容损坏/不是 JSON      | 返回 `{}`      | 打印                       |
      | 没有 `_meta`（旧格式）  | **整份作废**   | 打印                       |
      | `schema` 不是本版本     | **整份作废**   | 打印两个版本号 + 重跑命令  |
      | 未提供 `detections_fp`  | 原样返回       | 打印「跳过校验」           |
      | 登记表自己没记指纹      | 原样返回       | 打印「跳过校验」           |
      | 指纹一致                | 原样返回       | 不打印                     |
      | 指纹不符                | **整份作废**   | 打印两个指纹               |

    Task 56A · **版本不符与指纹不符同样是整份作废**：座位记录的字段形状变过
    一次（`cell_index` 标量 → `cell_indices` 列表）。旧文件里没有
    `cell_indices`，本模块读到只会当成「这个座位不是格位切分来的」——不报错、
    不缺字段、看上去完全正常，只是那几格的信息全没了。**能读出内容 ≠ 读对
    了**，这正是 `REGISTRY_SCHEMA` 这个常量存在的唯一理由；形状一变就升版，
    读到别的版本一律作废并说清怎么重来。

    与标注不同，指纹不符时**没有「按视频内容重新对齐」这条退路**：标注记
    的是「某时刻某位置有个人」，抖动可以模糊匹配；登记表记的是具体某一帧
    的某一个框，重检测后那个框已经不在了，硬用等于把另一个人的画面当成这
    个人的。宁可全废重来。

    Task 56B 另加一道**标注新鲜度**检查（判定全部委托给
    `registry_staleness`，本函数不自己比较）：登记表记的 `labels_digest`
    与当前 `entry_labels_<组>.json` 的内容摘要对不上，就**抛
    `RegistryStaleError`**——不是打印。区别于上面那张表里的「作废返回
    `{}`」：检测换了是外部事实，调用方还能选择不用登记表继续跑；而标注变
    了却没重跑登记表，是人手里那一步漏了，必须停下来让人重跑，否则他会
    继续对着旧图核对（2026-08-23 14:03 的事故就是这样）。

    `allow_stale=True` 是**只给诊断用**的逃生口（例如想打开一份旧登记表
    看看里面到底存了什么、或比对两次标注的差异）。默认严格；生产路径一律
    不许传它——一旦有人为了「先跑通」顺手关掉，这道检查就等于没有。
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        body = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print(f"   ⚠ {p.name}：读取失败（{exc}），本次不使用座位登记表")
        return {}
    if not isinstance(body, dict) or not isinstance(body.get("_meta"), dict):
        print(f"   ⚠ {p.name}：没有 _meta（不是 {REGISTRY_SCHEMA} 格式），"
              "无从判断它是不是这一份检测的，已作废，请重新生成"
              "（python -m pipeline.registry）")
        return {}

    schema = body["_meta"].get("schema")
    if schema != REGISTRY_SCHEMA:
        print(f"   ⚠ {p.name}：登记表版本是 {schema!r}，当前代码只认 "
              f"{REGISTRY_SCHEMA!r}——座位记录的字段形状已经变过（"
              "`cell_index` 标量 → `cell_indices` 列表），照旧读回会把「这个"
              "座位占了哪几格」整段读丢且不报错。已整份作废，请重新生成："
              "python -m pipeline.registry --preview")
        return {}

    stored = body["_meta"].get("detections_fingerprint")
    # 写成 if/elif 而不是三个提前 return：下面那道标注新鲜度检查必须对
    # 「跳过校验」的两条路同样生效，提前 return 会让它们绕过去。
    if detections_fp is None:
        print(f"   调用方未提供当前检测指纹，跳过 {p.name} 的失效校验")
    elif not stored:
        print(f"   {p.name} 自身未记录检测指纹，跳过失效校验")
    elif stored != detections_fp:
        print(f"   ⚠ {p.name}：检测指纹不符（登记表记的是 {stored}，当前检测是 "
              f"{detections_fp}）。登记表存的是具体某一帧的某一个框，重检测后"
              "那些框已经不在了，无法重新对齐，已整份作废——请重新人工复核并"
              "重新生成（python -m pipeline.registry）")
        return {}

    # Task 56B：登记表是不是产自当前这一份人工标注。判定只此一处
    # （`registry_staleness`），本函数不再写第二份比较。
    verdict = registry_staleness(p.parent, _sheet_of(p, body["_meta"]),
                                 reg_path=p, reg_body=body)
    if verdict.stale:
        if not allow_stale:
            raise RegistryStaleError(verdict.message)
        print(f"   ⚠ allow_stale=True：明知陈旧仍照用 —— {verdict.message}")
    return body


def main(argv: list[str] | None = None) -> int:
    """CLI：从一组的 work 目录产出座位登记表。

        python -m pipeline.registry                      # 用 config 里的组
        python -m pipeline.registry --work <dir> --sheet <名>
        python -m pipeline.registry --preview            # 另出人工核对页
        python -m pipeline.registry --check              # 只查新鲜度，给退出码

    读 `segments.json` / `detections.json` / `seats.json` /
    `entry_labels_<SHEET>.json`（后者缺失时按「没有人工标注」处理，登记表
    里就只有 `seated` 那一路）。**不跑任何 run_all 阶段、不调用 VLM。**
    """
    ap = argparse.ArgumentParser(
        prog="python -m pipeline.registry",
        description="产出座位登记表 seat_registry_<SHEET>.json")
    ap.add_argument("--work", default=None, help="组的 work 目录（默认取 config.WORK）")
    ap.add_argument("--sheet", default=None, help="组名（默认取 config.SHEET）")
    ap.add_argument("--out", default=None, help="输出路径（默认 <work>/seat_registry_<SHEET>.json）")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写盘")
    ap.add_argument("--check", action="store_true",
                    help="只判定登记表是不是产自当前这一份人工标注并给退出码"
                         "（0=最新，1=陈旧），不读检测、不写盘")
    ap.add_argument("--preview", action="store_true",
                    help="另外产出人工核对页 vlm_preview_<SHEET>.html"
                         "（显示 VLM 实际会看到的那张证据图）")
    ap.add_argument("--preview-out", default=None,
                    help="核对页输出路径（默认 <work>/vlm_preview_<SHEET>.html）")
    args = ap.parse_args(argv)

    from pipeline.detect import load_detections
    from pipeline.entryexit import detections_fingerprint, load_entry_labels
    from pipeline.schedule import load_segments_json

    work = Path(args.work) if args.work else config.WORK
    sheet = args.sheet if args.sheet else config.SHEET

    # Task 56B：`--check` 是**只读**的，且必须排在下面那三个产物的存在性
    # 检查之前——用户跑它就是想在动手打标前先问一句「核对页还作数吗」，
    # 此时缺 detections.json 与否跟这个问题无关，不该被挡在门外。
    if args.check:
        v = registry_staleness(work, sheet)
        print(f"   {'⚠ 陈旧' if v.stale else '✔ 最新'}：{v.message}")
        print(f"   登记表摘要 {v.stored_digest!r}（{v.registry_mtime or '不存在'}）"
              f" / 当前标注摘要 {v.current_digest!r}（{v.labels_mtime or '不存在'}）"
              f"　※ 时间仅供回忆用，判定只看摘要")
        return 1 if v.stale else 0

    seg_path, det_path, seats_path = (work / "segments.json",
                                      work / "detections.json",
                                      work / "seats.json")
    for p, stage in ((seg_path, "②"), (det_path, "③"), (seats_path, "④")):
        if not p.exists():
            print(f"   ⚠ 缺少 {p}（先跑阶段 {stage}），无法产出座位登记表")
            return 1

    segments = load_segments_json(seg_path)
    dets = load_detections(det_path)
    seats_raw = json.loads(seats_path.read_text(encoding="utf-8"))
    seats_by_order = {int(o): {int(t): int(s) for t, s in m.items()}
                      for o, m in seats_raw.items()}
    det_fp = detections_fingerprint(det_path)

    lp = labels_path(work, sheet)
    if lp.exists():
        # 用 candidates_for_labels（展示 ∪ 折叠）：用户可能把人标在折叠区的
        # 候选上，只查展示候选会配不上，那条标注就此静默消失。
        cands_by_order = {s.order: candidates_for_labels(dets, s)
                          for s in segments}
        entry_labels = load_entry_labels(lp, detections_fp=det_fp,
                                         candidates_by_order=cands_by_order)
        print(f"   本次登记表产自 {lp.name}（内容摘要 {labels_digest(lp)}，"
              f"该文件最后修改于 {_mtime_text(lp)}）")
    else:
        entry_labels = {}
        print(f"   未找到 {lp.name}，本次登记表只含就座阶段的结果"
              "（全部 confirmed=false，尚未经过人工复核）")

    # 摘要由 build_seat_registry 自己按路径算（Task 56B）：调用方另算一份
    # 传进去，就又多了一处可能与实际读的那个文件不是同一份的地方。
    reg = build_seat_registry(segments, dets, seats_by_order, entry_labels,
                              detections_fp=det_fp, sheet=sheet,
                              labels_path=(lp if lp.exists() else None))
    meta = reg["_meta"]
    print(f"   座位登记表：{meta['n_segments']} 段，"
          f"{meta['n_seats_resolved']}/{meta['n_seats_expected']} 个座位已定"
          f"（就座 {meta['counts_by_source']['seated']}、"
          f"进出场递补 {meta['counts_by_source']['entryexit']}、"
          f"格位切分 {meta['counts_by_source']['cell_split']}）")
    # 两类未定座位分开报（Task 56C）。合成一个数字的后果很具体：用户看到
    # 「还有 12 个座位未定」，无从知道其中 9 个他已经逐个看过、确认画面里
    # 根本没有那个人——那 9 个再怎么标也标不出来，只能靠加密抽帧。
    counts = unresolved_reason_counts(reg)
    if counts[UNRESOLVED_NO_CANDIDATE] or counts[UNRESOLVED_NOT_REVIEWED]:
        print(f"   未定座位：{counts[UNRESOLVED_NOT_REVIEWED]} 个还没人看过、"
              f"{counts[UNRESOLVED_NO_CANDIDATE]} 个人工判定确实无可用画面")
    for order, rec in reg["segments"].items():
        if not rec["unresolved_seats"]:
            continue
        reasons = rec.get("unresolved_reason") or {}
        nc = [s for s in rec["unresolved_seats"]
              if reasons.get(str(s)) == UNRESOLVED_NO_CANDIDATE]
        todo = [s for s in rec["unresolved_seats"] if s not in nc]
        parts = []
        if todo:
            parts.append(f"座位 {todo} 还没人看过")
        if nc:
            parts.append(f"座位 {nc} 人工判定无可用画面")
        print(f"   · 第 {order} 段 {rec['company']}：" + "、".join(parts))
    if counts[UNRESOLVED_NO_CANDIDATE]:
        print(f"   → 这 {counts[UNRESOLVED_NO_CANDIDATE]} 个座位靠继续标是找"
              f"不出来的，可对相应片段二次加密抽帧："
              f"python -m pipeline.densify --sheet {sheet}")

    out = Path(args.out) if args.out else registry_path(work, sheet)
    prev = Path(args.preview_out) if args.preview_out else preview_path(work, sheet)
    if args.dry_run:
        print(f"   --dry-run：未写盘（本应写到 {out}"
              + (f"、{prev}）" if args.preview else "）"))
        return 0
    write_seat_registry(reg, out)
    print(f"   已写出 {out}")
    if args.preview:
        # Task 54 需求二：显示的就是 VLM 会看到的那张证据图（同一个
        # build_evidence_sheet、同一批帧），不是另裁一份预览。
        render_vlm_preview(reg, prev)
        print(f"   已写出人工核对页 {prev}（图为 VLM 实际会看到的那张）")
    return 0


if __name__ == "__main__":       # pragma: no cover - CLI 入口
    raise SystemExit(main())
