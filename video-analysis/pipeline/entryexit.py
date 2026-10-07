"""阶段 ④d：进出场帧的站立候选识别 + 人工标注回流（Task 39）。

背景：进出场窗口占全部抽帧的约 80%（实测三组：进场 2994 帧/16371 检测框/
334 条 track，退场 3068 帧/16173 检测框/380 条 track），但 `seats.py` 的
`assign_segment_seats` 只筛 `kind=="seated"`，这些帧从头到尾没进过座位分配。

用户补充的关键信息：人工标注 `presenter_num` 时用的正是进出场画面来排除
工作人员——这批帧里确实含有足够的判别信息。

**曾经两次尝试用框高阈值判定「站立 vs 就座」，两次都被证伪**，第二次实测
直接推翻了这个前提本身：已确认的就座创业者（90 人）框高中位 0.45（p10=
0.23，p90=0.65，max=0.70）；进出场右区 track（756 条）框高中位 0.40
（p75=0.61，p90=0.73，max=0.76）——两个分布完全重叠，这个机位下高度分不
出谁站着谁坐着，直方图从 0.05 到 0.76 连续无裂谷，任何阈值都切在密集区。
详细记录见 `config.py` 里 `ENTRYEXIT_TOP_N` 上方。

现改为：碎片合并（进出场时人在走动，跟踪器频繁换 track_id，复用
`seats.merge_tracks_by_position`，不判定阈值，只用于把碎片拼回一个人）之后
**纯按框高排序展示**，不再声称「排名靠前=站立」——`entryexit_candidates`
每段每个窗口只取前 `config.ENTRYEXIT_TOP_N` 个，这是展示上限不是判据，
被截断的数量必须通过 `entryexit_candidate_totals` 如实告知用户
（`pipeline/entryreview.py` 负责渲染这条提示）。

但站立人数经常远超标注人数（中途走动的工作人员混入），自动匹配不可行——
`pipeline/landing.py` 早先尝试「进场落座点→座位号」的自动匹配，实测 13 段
仅 1 段对得上，已被证伪，本模块不重走那条路。改为产出候选交给人工挑
（`pipeline/entryreview.py` 渲染的复核页），标注结果通过 `load_entry_labels`
/ `entryexit_fill_plan` 回流成「哪些座位该补哪条进出场轨迹」的纯计算结果，
接入正式流水线（改 assemble.py）留给后续任务。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from pipeline import config
from pipeline.annotate import select_best_frames
from pipeline.detect import Detection
from pipeline.schedule import Segment
from pipeline.seats import exclude_judges, merge_tracks_by_position


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _merged_people(
    dets: list[Detection], segment: Segment, frame_w: int, kind: str,
    frame_h: int = 1080,
) -> tuple[list[dict], list[dict]]:
    """某个 (segment, kind) 窗口内，右区过滤 + 碎片合并后的全部「人」，按
    合并后中位框高降序排列，拆成两组返回 `(kept, filtered_out)`。

    两条过滤规则，被任一条挡下的都进 `filtered_out`，并在 `filter_reasons`
    里如实记下是被哪条（或哪几条）挡的：

    **长宽比（h/w）判定局部检测**（Task 41 改动一，`filter_reasons` 里的
    `"aspect"`）：h/w 低于 `config.ENTRYEXIT_MIN_ASPECT` 的判为「只拍到一只
    手/胳膊」。宽度取该「人」名下每条原始轨迹的中位宽度、再取这些轨迹中位数
    的中位数——与 `merge_tracks_by_position` 聚合 cx/y2/height 的方式一致，
    不因某条帧数特别多的轨迹把宽度带偏。

    **框高绝对下限**（Task 52 需求三，`filter_reasons` 里的 `"height"`）：
    中位框高低于 `config.ENTRYEXIT_MIN_HEIGHT_RATIO * frame_h` 的不作候选。
    这**不是**把被证伪两次的「高度=站立」判据搬回来——高度在这里仍然不用于
    判断角色，只用来砍掉小到人眼在证据图上根本认不出是谁的框。阈值由用户
    真实标注实测定出（取值与推导写在 `config.ENTRYEXIT_MIN_HEIGHT_RATIO`
    上方），定在实测最矮的真创业者以下 41% 处。

    被挡下的候选**不丢弃**：`filtered_out` 原样保留全部字段（含
    `best_dets`），供 `entryexit_filtered_candidates` 和页面折叠区块使用
    （凡有过滤必须可见），页面按 `filter_reasons` 逐条标出原因——用户看
    折叠区就是为了核实「这条规则有没有挡错人」，不写原因他判断不了。

    `entryexit_candidates`（截断展示）、`entryexit_candidate_totals`
    （截断前的真实总数）、`entryexit_filtered_candidates`（被挡掉的候选）
    三者共用这份逻辑，保证看到的是同一份候选、数字不会互相矛盾。
    """
    window = [d for d in dets if d.seg_order == segment.order and d.kind == kind]
    right = exclude_judges(window, frame_w)
    if not right:
        return [], []

    by_track: dict[int, list[Detection]] = {}
    for d in right:
        by_track.setdefault(d.track_id, []).append(d)

    kept: list[dict] = []
    filtered: list[dict] = []
    for person in merge_tracks_by_position(right):
        combined = [d for tid in person["track_ids"] for d in by_track[tid]]
        track_widths = [
            _median([d.x2 - d.x1 for d in by_track[tid]]) for tid in person["track_ids"]
        ]
        width = _median(track_widths)
        aspect = person["height"] / width if width > 0 else float("inf")
        t_mid = _median([d.t for d in combined])
        height_ratio = person["height"] / frame_h if frame_h > 0 else 0.0
        # 两条规则各自独立判断、都记下来。写成 if/elif 的话「又矮又方」的框
        # 只会显示一条原因，用户据此调阈值会得出错误结论。
        reasons: list[str] = []
        if aspect < config.ENTRYEXIT_MIN_ASPECT:
            reasons.append("aspect")
        if height_ratio < config.ENTRYEXIT_MIN_HEIGHT_RATIO:
            reasons.append("height")
        item = {
            "kind": kind,
            "track_id": person["track_ids"][0],
            "n_frames": person["n_frames"],
            "median_h": person["height"],
            "median_cx": person["cx"],
            "median_y2": person["y2"],
            "t_mid": t_mid,
            # Task 46：由视频内容导出的稳定标识，重检测后仍然成立。
            # track_id 只作参考信息保留（见 stable_candidate_id docstring）。
            "stable_id": stable_candidate_id(kind, t_mid, person["cx"], person["y2"]),
            "aspect": aspect,
            "height_ratio": height_ratio,
            "filter_reasons": reasons,
            "best_dets": select_best_frames(combined, n=4),
        }
        (filtered if reasons else kept).append(item)

    kept.sort(key=lambda c: -c["median_h"])
    filtered.sort(key=lambda c: -c["median_h"])
    return kept, filtered


def entryexit_candidates(dets: list[Detection], segment: Segment, frame_w: int = 1440,
                          frame_h: int = 1080) -> list[dict]:
    """进出场窗口里按框高排序展示的候选（不是「站立判定」，见模块顶部
    docstring——两次框高阈值判据都被证伪，框高分布在站立/就座之间完全
    重叠）。

    流程：
      1. 右区过滤复用既有的 `exclude_judges`（评委席/计时员恒在左侧，
         创业者/工作人员恒在右侧——与 `seat_candidates` 用的是同一条边界，
         不另写一份右区逻辑）。
      2. **碎片合并**：复用 `seats.merge_tracks_by_position`（不另写一份）
         把同一位置、时间不重叠的多条轨迹拼成一个「人」——进出场时人在
         走动，BoT-SORT 每一两帧就换一次 track_id，不合并会把同一个人的
         信号拆成好几条各占 1-2 帧的碎片。
      3. **长宽比过滤**（Task 41）：h/w 低于 `config.ENTRYEXIT_MIN_ASPECT`
         判为局部检测（手/胳膊），不进入候选池——但不丢弃，见
         `entryexit_filtered_candidates`。
      4. **框高绝对下限**（Task 52）：中位框高低于
         `config.ENTRYEXIT_MIN_HEIGHT_RATIO * frame_h` 的不作候选——小到
         这个程度的框，人眼在证据图上认不出是谁。仍然不是「站立判定」：
         高度在这里不用来判角色，只用来砍掉认不出的图。同样不丢弃。
      5. **按合并后中位框高降序排序，entry、exit 两个窗口分别只取前
         `config.ENTRYEXIT_TOP_N` 个**——这是页面一次展示得下的数量上限，
         不是「第 N+1 名不是人」的判据。被截断的真实数量由
         `entryexit_candidate_totals` 给出，供页面如实告知用户（凡有
         截断必须可见）。

    `frame_h` 只用于算框高占比（默认 1080，与 `detect.detect_frames`、
    `pairing` 一贯的默认口径一致）。

    返回列表，每项：
        {"kind": "entry"|"exit", "track_id": int, "n_frames": int,
         "median_h": float, "median_cx": float, "aspect": float,
         "height_ratio": float, "filter_reasons": [],
         "best_dets": [Detection, ...]}
    `track_id` 取合并后「人」的首个原始 track_id（`merge_tracks_by_position`
    docstring 里「首个是发起该人的轨迹」的既有约定），用于 `entry_review.html`
    的候选 key（`f"{kind}:{track_id}"`）与 `entryexit_fill_plan` 回填。
    `best_dets` 从合并后该人的全部检测框里选最清晰的若干帧（复用
    `annotate.select_best_frames`，n=4，与 `annotate.build_evidence_sheet`
    的 2×2 拼图张数对齐），不局限于某一条原始碎片轨迹。
    """
    out: list[dict] = []
    for kind in ("entry", "exit"):
        kept, _filtered = _merged_people(dets, segment, frame_w, kind, frame_h)
        # **挑哪几个**按框高（kept 已按框高降序），**怎么排**按时间。
        #
        # 用户 2026-08-24：「所有待标注的图片呈现的顺序是从前到后吗…按照
        # 从前到后的时间顺序，方便标注」。进场窗口起自本段开始前 25 秒，
        # 里面**先**是上一家还没走完的人、**后**才是本段进来的人；按框高排
        # 会把两拨人打乱交错，人工一路标下来对不上现场发生的顺序。
        #
        # 两件事分开：截断仍靠框高（见 config.ENTRYEXIT_TOP_N 上方那段
        # 「这是展示上限不是判据」的记录），只是排完之后再按时间重排。
        out.extend(sorted(kept[: config.ENTRYEXIT_TOP_N],
                          key=lambda c: c["t_mid"]))
    return out


def candidates_for_labels(dets: list[Detection], segment: Segment,
                          frame_w: int = 1440, frame_h: int = 1080) -> list[dict]:
    """**配对标注用**的候选池：展示候选 ∪ 被过滤（折叠区）候选。

    用户 2026-08-25 实测城市2 order 4：密抽之后那个人确实在候选里，只是框高
    占比 0.096（阈值 0.25）被挡进了折叠区。折叠区当初设成只读，前提是「被挡
    的都是手或太小认不出」；而这里恰恰相反——真人就在里面。

    光让页面能标是不够的：`entryexit_fill_plan` 原先只在 `entryexit_candidates`
    （展示候选）里找标注，标在折叠区的那条配不上任何候选，会被**静默忽略**
    ——用户标了、页面绿了、人一个也没补进去，一处不报错。

    **这不是把过滤规则架空**：没标过的折叠区候选不会自己混进补位（补位只认
    role=="创业者" 之类的显式标注）。它只是让「用户亲手认出来的那一条」能被
    认得出来。展示顺序仍由 `entryexit_candidates` 单独决定，本函数只管配对。
    """
    out: list[dict] = []
    for kind in ("entry", "exit"):
        kept, filtered = _merged_people(dets, segment, frame_w, kind, frame_h)
        # `kept` 取**全量**，不套 ENTRYEXIT_TOP_N。那个上限是**展示**上限
        # （页面一次看得过来的数量），拿它筛配对池会漏掉第 9 个之后的候选：
        # 重检测之后某条标注一旦滑出前 8，就配不上任何候选、静默消失。
        # `kept` 与 `filtered` 由 `_merged_people` 二选一分派，天然不相交，
        # 因此这里不需要去重（去重是不可达的死代码，变异测试确认过）。
        out.extend(kept)
        out.extend(filtered)
    return out


def entryexit_candidate_totals(
    dets: list[Detection], segment: Segment, frame_w: int = 1440,
    frame_h: int = 1080,
) -> dict[str, int]:
    """每个窗口截断前的候选总数 `{"entry": N, "exit": N}`（两条过滤之后、
    展示上限截断之前），供页面算出「还有多少个候选没展示」。凡有截断必须
    可见（本项目一贯要求，见 `seats.seat_candidates` 的剔除原因列表、
    `gallery` 的体积缩放提示）。
    """
    return {
        kind: len(_merged_people(dets, segment, frame_w, kind, frame_h)[0])
        for kind in ("entry", "exit")
    }


def entryexit_filtered_candidates(
    dets: list[Detection], segment: Segment, frame_w: int = 1440,
    frame_h: int = 1080,
) -> list[dict]:
    """被过滤规则挡掉的候选（长宽比判为局部检测、或框高低于绝对下限），
    entry、exit 两个窗口合并、按框高降序返回。**不做展示上限截断**——这本
    就是一份「不参与自动流程、供人工按需查看」的旁路列表，不套用
    `config.ENTRYEXIT_TOP_N`。

    用户 2026-08-17 明确要求：只拍到手不等于那里没人，排除它会导致座位号
    被后面的人顶上、整段错位——凡有过滤必须可见，本函数就是为此存在的，
    与 `entryexit_candidate_totals`（截断前的真实总数）配合使用。返回项
    结构与 `entryexit_candidates` 相同，另有 `filter_reasons`
    （`["aspect"]` / `["height"]` / 两者都有）与 `height_ratio`——页面要
    逐条标明是被哪条规则挡的，只给一堆图而不说原因，用户没法判断该调哪个
    阈值（Task 52 需求三）。
    """
    out: list[dict] = []
    for kind in ("entry", "exit"):
        _kept, filtered = _merged_people(dets, segment, frame_w, kind, frame_h)
        out.extend(filtered)
    # 与上面展示用的候选同一个口径：折叠区里也按时间从前到后，两处顺序不同
    # 会让用户在折叠区找刚才那个人时对不上。
    out.sort(key=lambda c: (c["kind"], c["t_mid"]))
    return out


# ---------------------------------------------------------------------------
# Task 46：标注的稳定标识 + 失效校验
#
# 一次真实的数据事故：Book1 文化4组从 13 家变 14 家，新增那家排在第 8 位，
# 其后企业行号整体后移一位，而帧是按旧编号抽的、盘上没重抽——seg 8-13 的
# 衣着/颜色/性别全部取自隔壁公司，**整条流水线照常跑完、没有任何一处
# 报错**。修复要求重抽帧 + 重检测，而重检测后 `track_id` 全部改变。用户
# 此前标注里的 `entry:19`/`exit:4` 因此大部分指向不存在的 track（递补从
# 5 人掉到 0 人），少数碰巧编号还在、但那已经是另一个人——错配会把别人的
# 衣着编进这个人的行里，同样不报错。用户明确说要标 60 多组，标识必须做稳。
# ---------------------------------------------------------------------------

_STABLE_ID_RE = re.compile(
    r"^(?P<kind>entry|exit)@(?P<t>-?\d+(?:\.\d+)?)"
    r"#(?P<cx>-?\d+(?:\.\d+)?),(?P<y2>-?\d+(?:\.\d+)?)$"
)


def stable_candidate_id(kind: str, t_mid: float, cx: float, y2: float) -> str:
    """候选的稳定标识：由视频内容导出，重检测后仍然成立。

    track_id 由 YOLO 跟踪器分配，重跑检测即全部改变——用户标了 60+ 组之后
    若因一次重检测全部作废，代价不可接受。而「某时刻画面某位置有个人」是
    视频本身的事实，重检测后只会有像素级抖动。

    格式 `f"{kind}@{t_mid:.0f}#{cx:.0f},{y2:.0f}"`，例如 `entry@973#912,1046`。
    四个字段的含义与精度选择的依据：

      · `kind`（`entry`/`exit`）——候选所属的窗口类型。进场窗口是段首前
        25 秒、退场窗口是段尾后 30 秒，同一个人往往在两个窗口各留一个
        候选，位置和时间都可能撞上，必须靠 kind 分开，否则进场那条标注
        会匹配到退场那个候选身上。

      · `t_mid`（秒，**整数**）——该「人」名下全部检测帧的时间中位数。
        进出场窗口按 `config.TRANSITION_FPS=5` 抽帧（每 0.2 秒一帧），
        重检测时首尾几帧的得失会让中位数漂移零点几秒，写到小数位只会把
        这点噪声固化进标识；取整到秒 = 5 帧的量化步长，而窗口本身跨 40
        秒，1 秒的分辨率足够把同一窗口里先后经过的两个人分开。

      · `cx`（像素，**整数**）——合并后该人的框中心横坐标中位数；
      · `y2`（像素，**整数**）——合并后该人的框底边纵坐标中位数。
        选 (cx, y2) 而不是 (cx, cy) 或整个框，是沿用本项目 `seats.
        merge_tracks_by_position` 判定「同一位置」时的既有口径：框底是人
        站/坐的落点，比框顶稳定（头顶常被裁切、举手会拔高框顶）。取整到
        像素——检测框坐标的亚像素部分纯属噪声，画面宽 1440px，1px 的
        分辨率远细于任何有意义的位置差别。

    **精度不等于容差**：这里取整只是不把噪声写进字符串，真正吸收重检测
    抖动的是 `match_label_to_candidate` 的模糊匹配（`config.LABEL_MATCH_
    T_TOL` 秒 / `config.LABEL_MATCH_POS_TOL` 像素）。**绝不能用字符串精确
    相等来匹配**——重检测后位置必然有十几像素的抖动，精确相等即全部失配。
    """
    return f"{kind}@{t_mid:.0f}#{cx:.0f},{y2:.0f}"


def parse_candidate_id(cid: str) -> dict | None:
    """把稳定标识拆回 `{"kind", "t_mid", "cx", "y2"}`；不是稳定标识（例如
    旧格式 `entry:19`）时返回 `None`。

    返回 `None` 的那一路**必须**被调用方当作「无从判断是不是同一个人」，
    而不是「尽力猜一个」——旧格式里只有跟踪器编号，重检测后同一个编号
    十有八九是另一个人，猜就是错配。
    """
    m = _STABLE_ID_RE.match(str(cid))
    if m is None:
        return None
    return {
        "kind": m.group("kind"),
        "t_mid": float(m.group("t")),
        "cx": float(m.group("cx")),
        "y2": float(m.group("y2")),
    }


def match_label_to_candidate(labels: dict, cand: dict) -> dict | None:
    """在旧标注里找与该候选对应的那条，容差内最近者胜出。

    `labels` 是「某一段的标注」`{候选标识: 标注记录}`。命中时**就地把那条
    标注从 `labels` 里弹出**并返回它——一个旧标注只能匹配一个候选，弹出
    是这条约束的实现方式：两个位置相近的候选不能抢同一条标注，否则一条
    「创业者·座位2」会被补给两个人，第二个还会滑到别的空位、顶掉本该在
    那的人（Task 45 缺陷一的同类错误）。没命中时返回 `None` 且 `labels`
    原样不动。

    判据（三条全部满足才算命中）：
      1. `kind` 相同（进场标注不得落到退场候选身上）；
      2. 时间差 ≤ `config.LABEL_MATCH_T_TOL`；
      3. `(cx, y2)` 的欧氏距离 ≤ `config.LABEL_MATCH_POS_TOL`。
    多条同时满足时取位置最近的（并列再取时间最近的，最后按标识字符串定序，
    保证结果可复现）。旧格式的键（`parse_candidate_id` 返回 `None`）一律
    跳过，绝不参与匹配。
    """
    kind = cand["kind"]
    t_mid = float(cand["t_mid"])
    cx = float(cand["median_cx"])
    y2 = float(cand["median_y2"])

    best_key: str | None = None
    best_score: tuple[float, float, str] | None = None
    for key in labels:
        parsed = parse_candidate_id(key)
        if parsed is None or parsed["kind"] != kind:
            continue
        dt = abs(parsed["t_mid"] - t_mid)
        if dt > config.LABEL_MATCH_T_TOL:
            continue
        dist = math.hypot(parsed["cx"] - cx, parsed["y2"] - y2)
        if dist > config.LABEL_MATCH_POS_TOL:
            continue
        score = (dist, dt, str(key))
        if best_score is None or score < best_score:
            best_score, best_key = score, key

    if best_key is None:
        return None
    return labels.pop(best_key)


def find_candidate_by_stable_id(stable_id: str, candidates: list[dict]) -> dict | None:
    """按稳定标识在候选里找那个人：先精确相等，再走与
    `match_label_to_candidate` **同一套**容差的模糊匹配（kind 相同、时间差
    ≤ `config.LABEL_MATCH_T_TOL`、位置距离 ≤ `config.LABEL_MATCH_POS_TOL`，
    多条满足取最近）。

    与 `match_label_to_candidate` 的方向相反：那个是「给定候选，去标注里
    找它的标注」（且命中即消耗），这个是「给定一条用户写下的标识，去候选
    里找它指的人」——`seat_overrides` 里的 `replacement_stable_id` 是用户
    明确挑的那张画面，不存在「一条只能用一次」的约束（同一张画面理论上
    可以被指派到不同座位，那是用户的判断），因此不消耗。

    找不到时返回 `None`——调用方**必须**如实上报，绝不许静默丢弃：用户
    明确指定了「用这张」，悄悄不用等于他永远不会知道自己的指定没生效。
    """
    for cand in candidates:
        if cand.get("stable_id") == stable_id:
            return cand
    parsed = parse_candidate_id(stable_id)
    if parsed is None:
        return None
    best: dict | None = None
    best_score: tuple[float, float, str] | None = None
    for cand in candidates:
        if cand["kind"] != parsed["kind"]:
            continue
        dt = abs(parsed["t_mid"] - float(cand["t_mid"]))
        if dt > config.LABEL_MATCH_T_TOL:
            continue
        dist = math.hypot(parsed["cx"] - float(cand["median_cx"]),
                          parsed["y2"] - float(cand["median_y2"]))
        if dist > config.LABEL_MATCH_POS_TOL:
            continue
        score = (dist, dt, str(cand["stable_id"]))
        if best_score is None or score < best_score:
            best_score, best = score, cand
    return best


def detections_fingerprint(path: Path) -> str:
    """`detections.json` 的内容指纹（sha256 前 16 位十六进制）。

    为什么取**内容哈希**而不是 `(文件大小, mtime, 检测框总数)` 的组合：
      · 「同一份检测重复读取必然相同」——内容哈希天然满足；(size, mtime)
        组合不满足：把同样的内容重新落一次盘（重跑一次幂等的写入、复制
        文件、rsync 同步、git checkout）就会改 mtime，于是一份完好的标注
        被错误判为失效、整份作废。这是假阳性，代价是用户白标一遍。
      · 「重检测后必然改变」——重检测重排 track_id、框坐标全部变化，内容
        哈希必然改变；而 (size, 检测框总数) 恰恰很容易碰撞：重检测出同样
        数量的框、JSON 长度分毫不差是完全可能的，而这是**假阴性**，会让
        一份对不上的标注被照常使用——那正是本任务要根除的错配。
      · 成本可忽略：`detections.json` 是几 MB 量级，sha256 一遍毫秒级。

    文件不存在时返回空串（调用方据此跳过校验，而不是崩掉整条流水线）。
    """
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# Task 49：标注文件里与「段」平级、但不是段的顶层键。`_meta` 是元信息，
# `seat_overrides` 是用户在挑选页给出的座位否决/替代。
# 载入与回填两侧都必须显式跳过它们，否则 `"seat_overrides"` 会被当成一个
# 段号去找候选（找不到、静默返回空），而 `_entry_labels_to_store` 会把它
# 展开成一堆对不上任何候选的预填项。
#
# Task 53 · 需求三：`position_exclusions`（「本组该位置恒为工作人员/专家」）
# 已按用户要求删除（原话：「这个选项实际没用，删掉吧」）。这个键**仍然留在
# 本元组里**——用户手上已有导出过的旧文件，不列在这里它就会被当成一个段号
# 展开，静默产出一堆对不上任何候选的垃圾条目。列在这里等于「读到就跳过」。
#
# Task 56C：`no_candidate_seats`（「本座位在所有候选里都找不到合适画面」）
# 同样与段平级、键为 (order, seat)。不列进来的后果与上面几个一模一样——
# 被当成段号展开成一堆对不上任何候选的垃圾条目，且不报错。
#
# Task 56V：`seat_reassign`（「这张已定座位的图其实是座位 N」）同样与段平级、
# 键为 (order, 原座位号)。不列进来的后果与上面几个逐字相同。
# 用户 2026-09-18：`judges`（评审席，键为 `judge#cx,y2`）同样与段平级。漏列
# 这一条的代价实测过——对城市1 的 order 5/6/7/8/10 重抽之后，五位评审人的标注
# （标掉的格、补选的格、改判给别人的格）被当成段号去和候选比对，一条也对不上，
# 整份算进「失配丢弃」并从结果里消失；命令行只说「57 条标注中 50 条匹配上」，
# 那 7 条里有 5 条就是评审席，看不出来。半天的人工核实静默归零。
LABEL_RESERVED_KEYS = ("_meta", "seat_overrides", "position_exclusions",
                       "cell_exclusions", "no_candidate_seats",
                       "no_candidate_notes", "seat_reassign", "judges")

# 已废弃、读到只打印一行说明的顶层键（Task 53 需求三）。
DEPRECATED_LABEL_KEYS = ("position_exclusions",)


_VOID_HINT = ("标注文件生成于另一份检测结果（重抽帧/重检测后编号已变），"
              "已作废，请在挑选页重新标注")
_VOID_HINT_LEGACY = ("标注用的是重检测即失效的跟踪器编号，无从判断还是不是"
                     "同一个人，已作废，请在挑选页重新标注")


def _remap_labels_to_current(
    body: dict, candidates_by_order: dict
) -> tuple[dict, int, int]:
    """把一份稳定 id 格式的标注模糊匹配到当前候选上，返回
    `(重挂到当前稳定 id 下的标注, 匹配上的条数, 失配丢弃的条数)`。"""
    out: dict = {}
    matched = 0
    dropped = 0
    for order_key, seg_labels in body.items():
        if not isinstance(seg_labels, dict):
            continue
        cands = candidates_by_order.get(order_key)
        if cands is None:
            try:
                cands = candidates_by_order.get(int(order_key))
            except (TypeError, ValueError):
                cands = None
        if cands is None:
            cands = candidates_by_order.get(str(order_key)) or []
        # 匹配即消耗：pool 是本段标注的副本，命中一条弹出一条，
        # 保证一条标注不会被两个候选同时拿走。
        pool = dict(seg_labels)
        remapped: dict = {}
        for cand in cands:
            lab = match_label_to_candidate(pool, cand)
            if lab is None:
                continue
            remapped[cand["stable_id"]] = lab
            matched += 1
        dropped += len(pool)
        if remapped:
            out[str(order_key)] = remapped
    return out, matched, dropped


def _with_side_channels(body: dict, overrides: dict,
                        cell_ex: dict | None = None,
                        no_cand: dict | None = None,
                        reassign: dict | None = None,
                        judges: dict | None = None) -> dict:
    """把 `seat_overrides` / `cell_exclusions` / `no_candidate_seats` 挂回
    标注本体（顶层，与段号平级，与文件里的位置一致）。

    **空的一律不挂**：`run_all` 阶段 7 用 `if entry_labels:` 判断「这份
    标注可不可用」，挂一个空的 `{"seat_overrides": {}}` 会让「一条标注都
    没有」的文件变成真值，打印出「递补已启用」而实际什么都补不了。

    Task 53 · 需求三：`position_exclusions` 已删除，**绝不再挂回来**——挂着
    会让下游（以及读 JSON 的人）以为它还生效。旧文件里读到它只打印一行说明。
    """
    out = dict(body)
    if overrides:
        out["seat_overrides"] = overrides
    # 逐格排除（Task 55）：与 seat_overrides 同理。只进 LABEL_RESERVED_KEYS
    # 不挂回来是最难发现的错——文件里明明有，跑下来毫无效果且不报错。
    if cell_ex:
        out["cell_exclusions"] = cell_ex
    # 「本座位无可用画面」（Task 56C）：同理。这条尤其不能丢——用户标它
    # 正是为了让加密抽帧知道该重抽哪几段，丢了就是「标了等于没标」。
    if no_cand:
        out["no_candidate_seats"] = no_cand
    # 已定座位改号（Task 56V）：同理。丢了的表现是「用户改了号、跑下来还是
    # 原来那个座位」——他会以为是自己没点上。
    if reassign:
        out["seat_reassign"] = reassign
    # 评审席（用户 2026-09-17/18）：同理。只进 LABEL_RESERVED_KEYS 而不挂回来
    # 是最难发现的错——文件里明明有，跑下来五位专家全成了「没人标过」，而
    # 一处不报错。评审人的标识由位置中位数导出、跨重检测稳定，重新对齐由
    # `judges.realign_judge_labels` 在各自的调用点做，这里原样带过去。
    if judges:
        out["judges"] = judges
    return out


def _remap_seat_overrides(
    overrides: dict, candidates_by_order: dict
) -> tuple[dict, int, int, list]:
    """把座位否决里的 `replacement_stable_id` 重新对齐到当前候选。

    返回 `(新的 seat_overrides, 匹配上的条数, 失配条数, 失配明细)`；
    失配明细是 `[(seg_order, seat, 原标识), ...]`，供调用方打印——对不上
    的替换标识**清成 `None` 但保留 `rejected`**：座位仍算空缺（用户说了
    它判错），只是不拿一个对不上的候选去补它。座位号本身不需要对齐
    （重检测不改座位号），只有指向「某个人」的标识才会抖动。
    """
    out: dict = {}
    matched = 0
    dropped = 0
    unresolved: list[tuple[str, str, str]] = []
    for order_key, seats in (overrides or {}).items():
        if not isinstance(seats, dict):
            continue
        cands = candidates_by_order.get(order_key)
        if cands is None:
            try:
                cands = candidates_by_order.get(int(order_key))
            except (TypeError, ValueError):
                cands = None
        if cands is None:
            cands = candidates_by_order.get(str(order_key)) or []
        new_seats: dict = {}
        for seat_key, rec in seats.items():
            if not isinstance(rec, dict):
                continue
            new_rec = dict(rec)
            sid = rec.get("replacement_stable_id")
            if sid:
                cand = find_candidate_by_stable_id(str(sid), cands)
                if cand is None:
                    new_rec["replacement_stable_id"] = None
                    dropped += 1
                    unresolved.append((str(order_key), str(seat_key), str(sid)))
                else:
                    new_rec["replacement_stable_id"] = cand["stable_id"]
                    matched += 1
            new_seats[str(seat_key)] = new_rec
        if new_seats:
            out[str(order_key)] = new_seats
    return out, matched, dropped, unresolved


def load_entry_labels(
    path: Path,
    detections_fp: str | None = None,
    candidates_by_order: dict | None = None,
) -> dict:
    """读用户从 `entry_review.html` 导出的 `entry_labels_*.json`，并校验它
    是不是**这一份检测结果**的标注（Task 46）。

    文件结构：
        {"_meta": {"detections_fingerprint": str, "sheet": str,
                   "n_candidates": int},
         "<seg_order>": {"<稳定标识>": {
            "role": "创业者|工作人员|专家|不确定|上一组创业者|下一组创业者",
            "seat": int|None,
            "n_people": int (缺省视为 1),
            # Task 54：格位分配（结构化选择题）。优先级高于 note，
            # 见 `cell_map_of_label`。`same_as` 是「与第 N 格同一个人」的
            # **引用**，不存副本，见 `cells_to_cell_map`。
            "cells": [{"cell": int, "role": str, "seat": int|None}
                      | {"cell": int, "same_as": int}]  (可选),
            "note": str (自由文本；有 cells 时不参与补位判定，仅作说明),
            "track_id": int (仅作人工排查的参考信息，**不用于匹配**)}},
         # Task 49 · 需求二：人工否决已定座位 / 指定替代画面
         "seat_overrides": {"<seg_order>": {"<seat>": {
            "rejected": bool, "replacement_stable_id": str|None}}},
    #     （Task 56E 起不再有 `note`：挑选页那个填空框没人读，已去掉。
    #      旧文件里残留的 note 一律忽略，不报错。）
         # 注：`position_exclusions`（Task 49 需求三「本组该位置恒为工作
         # 人员/专家」）已于 Task 53 按用户要求删除。旧文件里残留这个键时
         # **静默忽略并打印一行说明**——直接报错会让用户手上已导出的整份
         # 标注读不进来。}
    段标注的键是 `stable_candidate_id` 给出的稳定标识（如
    `entry@973#912,1046`），不是旧的 `entry:19`——后者是跟踪器编号，重检测
    即全部改变。`seat_overrides` 里的 `replacement_stable_id` 同样是稳定
    标识，走同一套校验（见 `_remap_seat_overrides`）。

    返回值把 `seat_overrides` 原样挂在顶层（与段号平级，和文件里的位置
    一致），空的不挂；`_meta` 与已废弃的键一律不挂。

    参数：
      · `detections_fp`：当前 `detections.json` 的指纹（`detections_
        fingerprint`）。传 `None` 表示调用方无从提供，此时**跳过校验**并
        打印一行说明（不静默）。
      · `candidates_by_order`：`{seg_order: [候选, ...]}`，指纹不符时用于
        模糊匹配。缺省 `None` 时指纹不符一律整份作废。

    分级策略（要点是**宁可全废也不要错配**——错配会把别人的衣着编码成这
    个人的，且不报错；本项目刚因同一类问题损失了一整组数据）：

      | 情形 | 处理 |
      |---|---|
      | 文件不存在 | 返回 `{}`，不打印（标注是可选输入） |
      | 旧格式（无 `_meta`） | **整份作废** + 提示重标。不尝试匹配——旧键里只有跟踪器编号，重检测后同号十有八九是另一个人 |
      | 未提供 `detections_fp` | 原样返回 + 打印「跳过校验」 |
      | 指纹一致 | 原样返回 |
      | 指纹不符、无候选可比对 | **整份作废** + 提示重标 |
      | 指纹不符、稳定 id 格式 | 模糊匹配，如实报告匹配数/丢弃数；匹配率 ≥ `config.LABEL_MATCH_MIN_RATE` 时用重新对齐后的结果 |
      | 指纹不符、匹配率低于阈值 | **整份作废** + 提示重标 |

    返回值一律是**不含 `_meta`** 的标注本体，键为当前候选的稳定标识。
    """
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print(f"   ⚠ {p.name}：内容不是对象，{_VOID_HINT}")
        return {}

    meta = data.get("_meta")
    body = {k: v for k, v in data.items() if k not in LABEL_RESERVED_KEYS}
    overrides = data.get("seat_overrides") or {}
    cell_ex = data.get("cell_exclusions") or {}
    no_cand = data.get("no_candidate_seats") or {}
    reassign = data.get("seat_reassign") or {}
    judges = data.get("judges") or {}
    # Task 53 · 需求三：已废弃的顶层键读到就说一声，然后当它不存在。
    # 报错不行——用户手上已经有一批带这个键的导出文件，一个异常会让整份
    # 标注读不进来；静默忽略也不行——他会以为那些勾选还在起作用。
    for key in DEPRECATED_LABEL_KEYS:
        if data.get(key):
            print(f"   ⚠ {p.name}：顶层键 `{key}` 已废弃（Task 53 按用户要求"
                  f"删除了「本组该位置恒为工作人员/专家」），本次已忽略其中 "
                  f"{len(data[key])} 条，其余标注照常使用")

    if not isinstance(meta, dict):
        # 旧格式：键是 `entry:19` 这类跟踪器编号，且没有检测指纹可核对。
        # 绝不尝试匹配——那正是会错配的情形。
        print(f"   ⚠ {p.name}：旧格式标注（键为跟踪器编号 `entry:19`，"
              f"无 _meta 检测指纹），{_VOID_HINT_LEGACY}")
        return {}

    # 这份标注是**哪一版挑选页**导出的（Task 57）。与检测指纹是两件事：
    # 重出页面时检测没重跑，检测指纹一字不变，恰恰在页面代码改了的时候
    # 完全不动。用户 2026-09-19 实测电子27——我修好了导出、重出了页面，
    # 收回来的文件里那几个专家号还是 null；页面在命令行上导出的是对的，
    # 于是「他用的是旧标签页」和「我还有个别的 bug」两种解释都说得通，
    # 而文件里没有任何依据能分开它们，只能请他再导一次、再赌一把。
    # 只提醒、不作废：用户手上可能已有一批旧文件，作废等于全部重标。
    from pipeline.entryreview import page_code_fp   # 延迟导入：那边 import 本模块
    file_fp = meta.get("page_code_fp")
    if file_fp != page_code_fp():
        where = f"（文件里是 {file_fp}）" if file_fp else "（文件里没有这个标记）"
        print(f"   ⚠ {p.name}：这份标注不是当前这版挑选页导出的{where}。"
              f"浏览器里若还开着旧的 entry_review.html，请**刷新**"
              f"（Cmd+Shift+R）后重新导出——标注本身仍在 localStorage 里，"
              f"不会丢。本次照常使用这份文件。")

    if detections_fp is None:
        print(f"   ⚠ {p.name}：调用方未提供当前检测指纹，跳过标注失效校验"
              "（无法判断这份标注是否属于当前这份检测结果）")
        return _with_side_channels(body, overrides, cell_ex, no_cand,
                                   reassign, judges)

    if meta.get("detections_fingerprint") == detections_fp:
        return _with_side_channels(body, overrides, cell_ex, no_cand,
                                   reassign, judges)

    if not candidates_by_order:
        print(f"   ⚠ {p.name}：检测指纹不符且没有候选可供比对，{_VOID_HINT}")
        return {}

    remapped, matched, dropped = _remap_labels_to_current(body, candidates_by_order)
    # 座位否决里的 `replacement_stable_id` 走**同一套**校验与同一份匹配率
    # 统计——它和候选标注一样是「用户指着某个人写下的标识」，重检测后同样
    # 会抖动，凭什么它就不用对齐？对不上的一律计入失配数（并在下面打印）。
    new_overrides, ovr_matched, ovr_dropped, unresolved = _remap_seat_overrides(
        overrides, candidates_by_order)
    matched += ovr_matched
    dropped += ovr_dropped
    total = matched + dropped
    rate = (matched / total) if total else 0.0
    if unresolved:
        detail = "、".join(f"{sid}（第 {order} 段座位 {seat}）"
                           for order, seat, sid in unresolved)
        print(f"   ⚠ {p.name}：{len(unresolved)} 个「改用我指定的候选」对不上"
              f"当前候选，已计入失配数并退回「仅否决、不替换」（该座位仍算"
              f"空缺，但不会拿一个对不上的候选去补）：{detail}")
    if total == 0:
        # 一条候选标注都没有 ≠ 这份文件过期。候选标注是唯一会随重检测抖动的
        # 东西；没有它就没有可判废的依据，而 `cell_exclusions` / `judges` 那几
        # 个通道的键（座位号、位置中位数）跨重检测稳定，凭什么陪葬？
        # 实测城市1：只标了评审席、还没标候选时，整份文件被判废，五位专家的
        # 逐格核实静默归零。
        print(f"   ⚠ {p.name}：检测指纹不符，但这份标注里没有需要对齐的候选"
              f"标注；按座位号/位置存的那几个通道跨重检测稳定，原样保留")
        return _with_side_channels({}, new_overrides, cell_ex, no_cand,
                                   reassign, judges)
    if rate < config.LABEL_MATCH_MIN_RATE:
        print(f"   ⚠ {p.name}：检测指纹不符，按视频内容（时间+位置）重新对齐后"
              f"{total} 条标注仅 {matched} 条匹配上（匹配率 {rate:.0%}，"
              f"低于阈值 {config.LABEL_MATCH_MIN_RATE:.0%}），{_VOID_HINT}")
        return {}

    print(f"   ⚠ {p.name}：检测指纹不符（重抽帧/重检测后编号已变），已按视频"
          f"内容（时间+位置）重新对齐：{total} 条标注中 {matched} 条匹配上，"
          f"{dropped} 条失配已丢弃（匹配率 {rate:.0%}，不低于阈值 "
          f"{config.LABEL_MATCH_MIN_RATE:.0%}，故保留）")
    # 重对齐这条路上**也要把 (order, seat) 那几个通道原样带过去**。
    # 它们的键是人工确定的座位号，跨重检测稳定，凭什么因为 track 编号变了
    # 就一起扔掉？加密抽帧（Task 56C）之后必然重检测、指纹必然改变，丢在
    # 这里的后果是用户刚标的「这个座位没画面」在重检测后静默消失——而那
    # 正是他标它的原因。
    return _with_side_channels(remapped, new_overrides, cell_ex, no_cand,
                               reassign, judges)


# ---------------------------------------------------------------------------
# Task 51 · 要做的一：按格位备注切分
#
# Task 49 给 2×2 证据图的每一格编了号（左上=1 右上=2 左下=3 右下=4，唯一
# 来源 `annotate._cell_origin`），并把用户按格位写的备注原样收进
# `needs_resplit`，明确把「切分」留给后续任务。用户已经按这个约定标完了
# 一整组，三条备注全部卡在 needs_resplit 里：
#
#     「1、4是创业者3，另两个是混入人员」   （order 1 律持兴观）
#     「1号是创业者2」                     （order 1 律持兴观）
#     「1/3是创业者3，2/4是创业者2」        （order 9 华韵传媒）
#
# 本节实现解析与切分。**相对指代一律不解析**是这里最重要的一条约束：
# 「另两个」要知道「另」是相对谁而言，「左边那个」要知道镜头方位，猜错的
# 后果是把工作人员的衣着当成创业者编码进结果表，整条流水线照常跑完、没有
# 任何一处报错——本项目一贯的最坏失败模式。未指明的格位一律留空。
# ---------------------------------------------------------------------------

# 短语分隔符：中英文逗号/分号/句号/换行。**顿号不在其中**——「1、4是创业者3」
# 里的顿号是格位之间的分隔符，不是短语分隔符（用户真实写法）。
_NOTE_PHRASE_SEP = re.compile(r"[，,；;。\n\r]+")

# 格位之间的分隔符：斜杠、顿号、逗号、空格（用户真实写法覆盖了前三种）。
_CELL_SEP = re.compile(r"[/、,，\s]+")

# 单个格位记号：`1`、`1号`、`1格`、`1号格`、`第2格`、`第2个`。
_CELL_TOKEN = re.compile(r"^(?:第)?\s*(\d{1,2})\s*(?:号格|号位|号|格|个)?$")

# 「是/为/:/：/→」——格位与目标之间的连接词。
_CELL_TARGET = re.compile(
    r"^(?P<cells>[^是为:：→]+?)\s*(?:是|为|[:：]|→)\s*(?P<target>.+)$")

# 挑选页 `ROLES` 里的六个角色（`pipeline/entryreview.py`）。判据是**前缀
# 相等**（`str.startswith`）而不是子串包含：「上一组创业者」标的是相邻项目
# 的人，用 `in` 匹配会把它认成「创业者」，而 `entryexit_fill_plan` 只对精确
# 等于「创业者」的目标补位——认错等于把隔壁项目的人补进本项目（由
# `test_parse_keeps_neighbouring_group_role_distinct` 守住）。长的仍排在
# 前面，以防日后有人把判据改成子串包含。
_KNOWN_ROLES = ("上一组创业者", "下一组创业者", "上一场创业者", "下一场创业者",
                "创业者", "工作人员", "专家", "不确定")

# Task 53 · 需求六：用户在备注里写的是「下一场创业者1」（**场**，不是页面
# 下拉里的**组**）——实测城市2组段 15：
#     role='下一组创业者'  note='1/4是下一场创业者1；2/3是下一场创业者2'
# 不认这两个词的后果不是报错，而是整条目标退化成一个未知角色
# （`role='下一场创业者1'`、`seat=None`），既不补位也不流转，且不打印任何
# 东西——用户写的那句话彻底白写。归一成页面上那个角色名，下游只认一种写法。
_ROLE_SYNONYMS = {"上一场创业者": "上一组创业者", "下一场创业者": "下一组创业者"}

# 跨段流转：角色 → 目标段相对本段的偏移。只有这两个角色会跨段，其余一律
# 留在本段（`entryexit_fill_plan` 的第二遍扫描据此分派）。
_NEIGHBOUR_ROLE_DELTA = {"上一组创业者": -1, "下一组创业者": +1}

# 目标后面挂着的座位号：「创业者3」「创业者 3 号」「创业者第3位」。
_TARGET_SEAT = re.compile(r"^(?:第)?\s*(\d{1,2})\s*(?:号位|号|位|座位|座)?$")

# **相对指代**：出现其中任何一个，这条短语一律不解析（哪怕里面也有数字）。
# 「另两个」「其余」「剩下的」需要知道「另」是相对谁而言；「左边那个」需要
# 知道镜头方位。这些算法都不知道，猜错不报错。
_RELATIVE_MARKERS = (
    "另", "其余", "其他", "其它", "剩下", "剩余", "余下", "别的", "剩的",
    "左边", "右边", "上边", "下边", "左侧", "右侧", "上面", "下面",
    "中间", "旁边", "靠", "那个", "这个", "都", "全部", "所有", "每",
)


def _parse_cell_tokens(text: str, n_cells: int) -> list[int] | None:
    """把「1/3」「1、4」「第2格」解析成格位号列表；解析不出返回 None。"""
    if any(m in text for m in _RELATIVE_MARKERS):
        return None                     # 相对指代一律不猜（见上方说明）
    out: list[int] = []
    for part in _CELL_SEP.split(text.strip()):
        if not part:
            continue
        m = _CELL_TOKEN.match(part)
        if m is None:
            return None                 # 有一个记号看不懂，整条短语就不算数
        v = int(m.group(1))
        if not (1 <= v <= n_cells):
            return None                 # 超出格位范围（备注写错了，不猜）
        if v not in out:
            out.append(v)
    return out or None


def _parse_cell_target(text: str) -> dict | None:
    """把「创业者3」「工作人员」「混入人员」解析成目标描述；看不懂返回 None。

    返回 `{"role": ..., "seat": ..., "text": 原文}`。`role` 是六个已知角色
    之一时按原样给出（「上/下一**场**创业者」这两个用户真实写法归一成页面
    上的「上/下一**组**创业者」，见 `_ROLE_SYNONYMS`），否则原样保留用户写
    的词（如「混入人员」）——只有精确等于「创业者」的目标会补进**本段**、
    精确等于「上/下一组创业者」的目标会流转到相邻段，未知词两样都不参与，
    不必也不该硬塞进某个已知角色里。
    """
    t = text.strip().strip("。.,，、;；:：")
    if not t:
        return None
    for role in _KNOWN_ROLES:
        if not t.startswith(role):
            continue
        rest = t[len(role):].strip()
        canon = _ROLE_SYNONYMS.get(role, role)
        if not rest:
            return {"role": canon, "seat": None, "text": t}
        m = _TARGET_SEAT.match(rest)
        if m is None:
            return None                 # 角色后面挂着看不懂的东西，保守不猜
        return {"role": canon, "seat": int(m.group(1)), "text": t}
    if any(m in t for m in _RELATIVE_MARKERS):
        return None
    return {"role": t, "seat": None, "text": t}


# --- Task 52 需求四：一条短语里写了不止一组映射 ---
#
# 用户真实写法（order 7）：`1 2 3是创业者2 4是创业者1`。它没有任何短语分隔
# 符（顿号在本模块是格位分隔符，空格也是），`_CELL_TARGET` 的非贪婪 cells
# 只能切到第一个「是」，剩下的 `创业者2 4是创业者1` 整块被当成目标、解析
# 失败——那一段因此少一个人。
#
# 切法：以「一串格位号 + 连接词 + 一个目标」为单元，从左到右逐个匹配，
# **要求整条短语被这些单元与分隔符完整覆盖**。覆盖不全就整条不解析。
# 这条覆盖要求是本节唯一的安全阀：`2为创业者2 3` 只能匹配出
# 「2 为 创业者2」，尾巴上孤零零的 `3` 覆盖不到——而这个 3 究竟是「格3 也
# 是创业者2」还是「创业者2 的第 3 人」，两种读法都说得通。歧义一律不猜
# （与相对指代同一条理由：猜错就把别人的衣着编进这个座位，不报错）。
_CELL_RUN = r"(?:(?:第)?\s*\d{1,2}\s*(?:号格|号位|号|格|个)?[/、,，\s]*)+"
# 目标 = 一段不含数字/分隔符/连接词的词（角色名），后面可选挂一个座位号。
_TARGET_RUN = r"[^\d/、,，\s是为:：→]+(?:\s*(?:第)?\s*\d{1,2}\s*(?:号位|号|位|座位|座)?)?"
_MULTI_UNIT = re.compile(
    rf"(?P<cells>{_CELL_RUN})\s*(?:是|为|[:：]|→)\s*(?P<target>{_TARGET_RUN})")
# 单元之间允许出现的东西：只有分隔符（空格、顿号、逗号、斜杠）。
_UNIT_GAP = re.compile(r"[/、,，\s]*")


def _parse_multi_units(phrase: str, n_cells: int) -> list[tuple[list[int], dict]] | None:
    """把「1 2 3是创业者2 4是创业者1」切成多个 (格位列表, 目标) 单元。

    整条短语必须被单元与分隔符**完整覆盖**，否则返回 None（见上方说明：
    覆盖不全 = 有看不懂的残余 = 歧义 = 不猜）。相对指代照旧一票否决——
    每个单元仍然走 `_parse_cell_tokens` / `_parse_cell_target`，不另写一份
    判据。
    """
    pos, units = 0, []
    for m in _MULTI_UNIT.finditer(phrase):
        if not _UNIT_GAP.fullmatch(phrase[pos:m.start()]):
            return None                 # 单元之间夹着看不懂的东西
        cells = _parse_cell_tokens(m.group("cells"), n_cells)
        target = _parse_cell_target(m.group("target"))
        if not cells or target is None:
            return None
        units.append((cells, target))
        pos = m.end()
    if not units or not _UNIT_GAP.fullmatch(phrase[pos:]):
        return None                     # 尾巴上还剩东西（如「2为创业者2 3」的 3）
    # 同一条短语内部自相矛盾（同一格位指向两个不同目标）也不猜。
    seen: dict[int, str] = {}
    for cells, target in units:
        for c in cells:
            if seen.setdefault(c, target["text"]) != target["text"]:
                return None
    return units


def _parse_one_phrase(phrase: str, n_cells: int) -> list[tuple[list[int], dict]] | None:
    """一条短语解析出的全部 (格位列表, 目标) 单元；解析不出返回 None。

    先按「整条只有一组映射」解析（既有路径，一字未改），失败再试多组切分
    ——顺序不能反：`1、4是创业者3` 这类既有写法必须继续走既有那条路，新
    切分只负责它解析不了的那些。
    """
    m = _CELL_TARGET.match(phrase)
    if m is not None:
        cells = _parse_cell_tokens(m.group("cells"), n_cells)
        target = _parse_cell_target(m.group("target")) if cells else None
        if cells and target is not None:
            return [(cells, target)]
    return _parse_multi_units(phrase, n_cells)


def parse_cell_note_report(note: str | None, n_cells: int = 4) -> dict:
    """解析格位备注，**连同没能解析的短语一起**返回。

    返回 `{"parsed": {格位号: {"role", "seat", "text"}}, "unparsed_phrases":
    [原文短语, ...]}`。

    支持的写法（左侧格位 + 「是/为/:/：/→」+ 右侧目标）：

      | 备注                         | parsed                                       |
      |------------------------------|----------------------------------------------|
      | `1/3是创业者3，2/4是创业者2` | 1,3 → 创业者座位3；2,4 → 创业者座位2          |
      | `1、4是创业者3`              | 1,4 → 创业者座位3                             |
      | `1号是创业者2`               | 1 → 创业者座位2                               |
      | `第2格是工作人员`            | 2 → 工作人员（无座位号）                       |
      | `1 2 是专家`                 | 1,2 → 专家                                    |
      | `3：混入人员`                | 3 → 混入人员（未知角色，原样保留，不参与补位） |
      | `1 2 3是创业者2 4是创业者1`  | 1,2,3 → 创业者座位2；4 → 创业者座位1（Task 52）|
      | `1 3是创业者2； 2 4是创业者3`| 1,3 → 创业者座位2；2,4 → 创业者座位3          |

    格位分隔符 `/`、`、`、`,`、`，`、空格；格位记号支持 `1`、`1号`、`1格`、
    `1号格`、`第2格`、`第2个`；短语分隔符为中英文逗号/分号/句号/换行
    （**顿号不算短语分隔符**——它是用户真实写法里的格位分隔符）。同一条
    短语里可以写不止一组映射（`N…是X M…是Y`），按「格位串 + 连接词 +
    目标」逐个单元切开，**要求整条被这些单元完整覆盖**，见
    `_parse_multi_units`。

    **一律不解析的写法**，两类：

      1. 相对指代——「另两个」「其余」「剩下的」「左边那个」（完整清单见
         `_RELATIVE_MARKERS`）。它们需要知道「另」是相对谁、「左边」是相对
         哪个镜头方位，算法不知道。
      2. 歧义写法——尾巴上挂着覆盖不到的数字，典型是用户真实写过的
         `2为创业者2 3`：这个 3 可能是「格 3 也是创业者2」，也可能是
         「创业者2 的第 3 人」，两种读法都说得通。

    两类都不猜：猜错就把工作人员的衣着当成创业者编码进结果表，且不报错。
    这类短语的格位**留空**，短语本身原样进 `unparsed_phrases`——凡有未
    解析必须可见（页面据此提示用户改写成明确写法）。

    同一个格位被两条短语指到不同目标时，**先出现的那条作数**，后一条整条
    进 `unparsed_phrases`（矛盾同样必须可见，不能静默取其一）。

    解析不出任何东西时 `parsed` 为空 dict，**不抛异常**——备注是自由文本，
    用户写什么都可能，一个 ValueError 会把整条流水线打断。
    """
    parsed: dict[int, dict] = {}
    unparsed: list[str] = []
    if not note:
        return {"parsed": parsed, "unparsed_phrases": unparsed}

    for raw in _NOTE_PHRASE_SEP.split(str(note)):
        phrase = raw.strip()
        if not phrase:
            continue
        units = _parse_one_phrase(phrase, n_cells)
        if units is None:
            unparsed.append(phrase)
            continue
        # 一条短语要么整条生效、要么整条不生效：任何一个单元与已有结果撞车，
        # 整条进 unparsed（半条生效会让用户以为写法被接受了，而实际只落了
        # 一半——矛盾必须可见，不能静默取其一）。
        conflict = any(c in parsed and parsed[c]["text"] != target["text"]
                       for cells, target in units for c in cells)
        if conflict:
            unparsed.append(phrase)
            continue
        for cells, target in units:
            for c in cells:
                parsed.setdefault(c, dict(target))
    return {"parsed": parsed, "unparsed_phrases": unparsed}


def parse_cell_note(note: str | None, n_cells: int = 4) -> dict[int, dict]:
    """`parse_cell_note_report(...)["parsed"]`——只要解析结果的便捷入口。

    结构：`{格位号: {"role": str, "seat": int|None, "text": str}}`。
    「创业者3」里的 3 是**座位号**，取出来放在 `seat`；「工作人员」这类
    没有座位号的目标 `seat` 为 None。写法清单与「相对指代一律不解析」的
    理由见 `parse_cell_note_report`。
    """
    return parse_cell_note_report(note, n_cells)["parsed"]


# ---------------------------------------------------------------------------
# Task 54 · 需求一：格位备注 → **结构化选择题**（`cells`）
#
# 用户 2026-08-23 原话：「能放弃解析，转换为选择吗，这样就能精准实时匹配到
# 座位人是否识别，比如选人数为 2 时，就给出这两人身份和座位号的选择题，且可
# 留空，根据选出的空，精准识别，不需要解析了」。
#
# 为什么必须改：自由文本的解析器已因写法问题反复失败三轮（`2为创业者2 3`
# 歧义、`1 2 3是创业者2 4是创业者1` 缺分隔符、「另两个」相对指代）。更要命
# 的是解析发生在 **Python 侧**——用户在城市2组 order1 明明已把三个人都写进
# 备注，挑选页仍显示红色未完成，因为要等下一次跑流水线才知道解析出了什么。
# 换成下拉之后，页面自己就能算出「这一段已确定几个座位」，当场变绿。
#
# `note` 字段**保留不删**（用户可能想写额外说明），解析器也保留不删（既有
# 两组标注全靠它）。三者的优先级见 `cell_map_of_label`。
# ---------------------------------------------------------------------------

# 格位身份下拉的选项。与页面 `ROLES` 同源同序（`entryreview` 从这里取，
# 不另写一份）——两处各写一份，页面上能选的角色与 `entryexit_fill_plan`
# 认得的角色就会悄悄错开，而错开的表现只是「选了没反应」。
CELL_ROLE_CHOICES = ("创业者", "工作人员", "专家", "上一组创业者",
                     "下一组创业者", "不确定")

# 2×2 证据图的格位数。`build_evidence_sheet` 恒为 4 格（不足 4 张以灰块
# 补齐），这是那个事实的唯一常量出口。
CELL_COUNT = 4


def _cell_number(value, n_cells: int) -> int | None:
    """把格位号归一成 1..n_cells 的 int；看不懂/超范围返回 None。"""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= n_cells else None


def _cell_seat_number(value) -> int | None:
    """座位号归一成 int；空串/None/看不懂一律 None（= 未判定）。"""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_same_as(rows: dict[int, dict], cell: int, n_cells: int) -> dict | None:
    """解开「同格位X」的引用链，返回最终生效的那一行；解不开返回 None。

    **只能引用比自己小的格位号**（页面下拉只给出更小的那些，见
    `entryreview.cellRoleOptions`）。这不是事后检测环，而是让环在结构上
    不可能出现：每跳一次格位号严格减小，自然数不能无限下降，链必然终止。
    另设 `n_cells` 跳的上限纯属防御——正常情况下永远触发不到，真触发了
    说明文件是手写的且违反了上面这条不变式，此时**视为空并打印**，不猜。
    """
    cur, hops = rows[cell], 0
    while True:
        ref = cur.get("same_as")
        if ref is None or ref == "":
            return cur
        hops += 1
        if hops > n_cells:
            print(f"   ⚠ 格位 {cell} 的「同格位」引用链超过 {n_cells} 层，"
                  f"已视为空（引用只能指向更小的格位号，正常不该出现）")
            return None
        nxt = _cell_number(ref, n_cells)
        here = _cell_number(cur.get("cell"), n_cells)
        if nxt is None or here is None or nxt >= here or nxt not in rows:
            print(f"   ⚠ 格位 {here} 的「同格位{ref}」引用无效（只能引用更小、"
                  f"且确实存在的格位号），已视为空，不参与补位")
            return None
        cur = rows[nxt]


def cells_to_cell_map(cells, n_cells: int = CELL_COUNT) -> dict[int, dict]:
    """把挑选页的结构化格位选择转成与 `parse_cell_note` **同形**的
    `{格位号: {"role", "seat", "text"}}`，下游（切分、补位、跨段流转）因此
    完全共用一套代码，不为选择题另开一条路径。

    输入形如::

        [{"cell": 1, "role": "创业者", "seat": 2},
         {"cell": 3, "same_as": 1},          # 与 1 号格是同一个人
         {"cell": 4}]                        # 留空 = 未判定

    规则：

      · **身份留空的一律丢弃**（不进返回值）——「留空 = 未判定」是本项目
        一贯的约定（Task 32：预填等于把「没判定」伪装成「已判定」）。
      · 座位号留空**保留**（`seat=None`）：`工作人员`/`专家` 本就没有座位
        号；`创业者` 而座位号留空的会照旧进 `needs_seat`（补不进去，但
        可见）——静默丢掉它等于让用户以为标完了。
      · `same_as` 只存**引用**不存副本：源格改了，引用它的格跟着变。存副本
        会在用户回头改源格时留下一份不会更新的旧值，且不报错（本项目在
        「同一份信息存两处」上已栽过多次）。
      · 格位号重复时**先出现的作数**，与 `parse_cell_note_report` 里
        「同一格位被两条短语指到时先出现的作数」同一口径。
    """
    rows: dict[int, dict] = {}
    for item in cells or []:
        if not isinstance(item, dict):
            continue
        c = _cell_number(item.get("cell"), n_cells)
        if c is None or c in rows:
            continue
        rows[c] = item

    out: dict[int, dict] = {}
    for c in sorted(rows):
        resolved = _resolve_same_as(rows, c, n_cells)
        if resolved is None:
            continue
        role = str(resolved.get("role") or "").strip()
        if not role:
            continue                       # 身份留空 = 未判定
        seat = _cell_seat_number(resolved.get("seat"))
        text = f"格位{c}={role}" + (f"{seat}" if seat is not None else "")
        if resolved is not rows[c]:
            text += f"（同格位{_cell_number(resolved.get('cell'), n_cells)}）"
        out[c] = {"role": role, "seat": seat, "text": text}
    return out


def note_to_cells(note: str | None, n_cells: int = CELL_COUNT) -> list[dict]:
    """把一条**旧的**自由文本备注解析成 `cells` 行，供挑选页预填控件。

    Task 54 要求「若某候选只有旧的 note 没有 cells，把解析结果预填进 cells
    控件并提示『已由备注自动填充，请核对』」——让用户一次性迁移过来。解析
    只此一处（`parse_cell_note`），页面不另写一份 JS 解析器：两处各写一份
    必然走样，而走样的表现是页面显示的分配与流水线实际补的人对不上。

    指向**同一个目标**的多个格位折成一次赋值 + 若干 `same_as` 引用：真实
    写法 `1/3是创业者3，2/4是创业者2` 因此变成 `[{1:创业者3}, {2:创业者2},
    {3:same_as 1}, {4:same_as 2}]`——正是「同格位X」这个下拉要表达的东西，
    且引用一律指向更小的格位号（不可能成环，见 `_resolve_same_as`）。

    保证 `cells_to_cell_map(note_to_cells(note))` 与 `parse_cell_note(note)`
    **逐格的 role/seat 完全相同**（由 `test_note_to_cells_roundtrips` 钉住；
    只有人读的 `text` 不同，那一项不参与任何判定）——预填绝不能改变语义，
    否则用户核对时看到的分配和他当初写的那句话不是一回事。
    """
    parsed = parse_cell_note(note, n_cells)
    rows: list[dict] = []
    first_of: dict[tuple, int] = {}
    for c in sorted(parsed):
        target = parsed[c]
        key = (target["role"], target["seat"])
        if key in first_of:
            rows.append({"cell": c, "same_as": first_of[key]})
        else:
            first_of[key] = c
            rows.append({"cell": c, "role": target["role"], "seat": target["seat"]})
    return rows


def cell_map_of_label(lab: dict, n_cells: int = CELL_COUNT) -> tuple[dict, dict, str]:
    """一条候选标注的格位映射，**顺序即优先级**（Task 54 需求一）。

      | # | 情形                     | 处理                          |
      |---|--------------------------|-------------------------------|
      | 1 | 有 `cells` 且能解出内容  | 用 cells，**忽略 note**       |
      | 2 | 无 cells / cells 全空    | 沿用 `parse_cell_note_report` |
      | 3 | 两者都没有               | 空映射（调用方退回角色闸门）  |

    第 2 行的「cells 全空也退回 note」是刻意的：用户把「人数」拨到 2、格位
    控件刚展开还一个都没选时，`cells` 已经存在但全是空行——此时若把它当作
    「用户已判定为空」，他此前写的备注、选的角色会静默失效。**全空 = 还没
    填**，不是「填了个空」。

    返回 `(cell_map, report, source)`；`report` 与 `parse_cell_note_report`
    同形（`{"parsed", "unparsed_phrases"}`），走 cells 那一路时
    `unparsed_phrases` 恒为空（选择题没有「解析不出」这回事）。`source` 取
    `"cells"` / `"note"` / `"none"`，只用于诊断打印。
    """
    # 人数**明确**改回 1 时，逐格判定不作数——一个框里一个人，逐格分配没有
    # 意义（用户 2026-09-09 实测城市1 order 16：「选的是人数为 1，却还是出现了
    # 4 个格位标注」）。页面的 `cellRowsOf` 与这里同一条口径，否则页面藏起来
    # 了、流水线照旧按 cells 切分，一处不报错。
    #
    # 只在**显式为 1** 时忽略：`n_people` 整个不在时照旧按 cells 走——缺失不
    # 等于「用户说了只有一个人」，手改过的文件按 1 处理会把已有的逐格工作静默
    # 作废。四个组的既有文件实测：带 cells 的记录全部带着 n_people≥2。
    _np = lab.get("n_people") if isinstance(lab, dict) else None
    _says_single = _np is not None and str(_np).strip() in ("1", "1.0")
    if (not _says_single and isinstance(lab, dict)
            and isinstance(lab.get("cells"), list) and lab["cells"]):
        cell_map = cells_to_cell_map(lab["cells"], n_cells)
        if cell_map:
            return cell_map, {"parsed": cell_map, "unparsed_phrases": []}, "cells"
    note = (lab.get("note") or "") if isinstance(lab, dict) else ""
    report = (parse_cell_note_report(note, n_cells) if note
              else {"parsed": {}, "unparsed_phrases": []})
    return report["parsed"], report, ("note" if report["parsed"] else "none")


def cell_dets(cand: dict, cell_index: int) -> list:
    """候选证据图上第 `cell_index` 格（1 起）对应的那一张检测帧。

    **格位号 k ↔ `best_dets[k-1]`**：`annotate.build_evidence_sheet` 把
    `paths[i]` 贴在 `_cell_origin(i, cell)`，并在同一个原点画上编号 `i+1`
    ——贴图与编号共用同一个 i，因此「图上标着 k 的那一张」恒为输入的第
    k-1 张，与 `_cell_origin` 内部是行优先还是列优先无关。

    超出实际张数（证据图不足 4 张时会有灰块占位）时返回空列表——宁可不补，
    也不能拿相邻那一格（另一个人）的画面顶上。**切分与下游取帧共用本函数**，
    两处各切一次必然日后走样，而走样的表现是衣着编码给了错的人、不报错。
    """
    dets = list(cand.get("best_dets") or [])
    i = int(cell_index) - 1
    if i < 0 or i >= len(dets):
        return []
    return [dets[i]]


def cells_dets(cand: dict, cell_indices) -> list:
    """同一个人占了多格时，这些格位各自的那一张，按**格位号升序**排好。

    Task 56A（用户 2026-08-24 实测文化4组 order 2）：「标了格位1是创业者2，
    格位2是同格位1，那么应该将这两帧图片都保留下来，而不是登记表只有一张
    图」。两格是同一个人 → 一个座位、两张证据；旧实现只留下其中一张，另一
    张连同它那一格一起消失且不报错，下一轮换 VLM 打标时只看得到一半的画面。

    逐格一律走 `cell_dets`，**不在这里另算一次 `best_dets[k-1]`**：两处各
    切一次必然日后走样，而走样的表现是衣着编码给了错的人、不报错。越界的
    格位（证据图不足 4 张时的灰块）照旧跳过，不拿邻格另一个人的画面顶上。
    """
    out: list = []
    for k in sorted({int(c) for c in (cell_indices or [])}):
        out.extend(cell_dets(cand, k))
    return out


def split_candidate_by_cells(cand: dict, cell_map: dict) -> list[dict]:
    """把一个多人候选按格位拆成多个「单人候选」。

    `cell_map` 是 `parse_cell_note` 的返回值（`{格位号: {"role", "seat",
    "text"}}`）。每个格位对应 `cand["best_dets"]` 里的哪一张由 `cell_dets`
    单一给出（格位 k → `best_dets[k-1]`，与 `annotate.build_evidence_sheet`
    实际画上去的编号一致）。证据图不足 4 张时，超出的格位视为无效、直接
    跳过（不拿别人的画面顶上）。

    返回的每一项是原候选的浅拷贝，但：

      · `best_dets` 只含该格位那一张，`n_frames` 相应改为 1；
      · 新增 `cell_index`（格位号）、`parent_stable_id`（原候选的稳定标识）、
        `role`/`seat`（该格位的目标）、`note_text`（备注里对应的那半句）；
      · 新增 `layout`——**证据图不再是 2×2 拼图，只有一张**，构成文案必须
        如实反映这一点（`assemble.describe_layout(..., cell_index=k)`）。
        沿用四张图的文案会让 VLM 去描述并不存在的另外三张画面，不报错。

    `stable_id` **不变**（仍是原候选的）：它是「视频里某时刻某位置有个人」
    的标识，切分不改变这个事实；切出来的份用 `(stable_id, cell_index)` 区分。
    """
    from pipeline.assemble import describe_layout

    out: list[dict] = []
    for cell in sorted(cell_map):
        dets = cell_dets(cand, cell)
        if not dets:
            continue
        target = cell_map[cell] or {}
        piece = dict(cand)
        piece["best_dets"] = dets
        piece["n_frames"] = 1
        piece["cell_index"] = int(cell)
        piece["parent_stable_id"] = cand.get("stable_id")
        piece["role"] = target.get("role")
        piece["seat"] = target.get("seat")
        piece["note_text"] = target.get("text") or ""
        is_entry = cand.get("kind") == "entry"
        piece["layout"] = describe_layout(
            0, n_entry=1 if is_entry else 0, n_exit=0 if is_entry else 1,
            cell_index=int(cell),
        )
        out.append(piece)
    return out


def candidate_brightness(cand: dict) -> float | None:
    """候选画面的亮度（灰度均值，0–255），取各证据帧裁剪后的中位数。

    一张几乎全黑的图，衣着**颜色**根本无从判读，而颜色正是本流水线要编码
    的东西之一。读不到图（路径不存在、解码失败）返回 `None`——「不知道」
    绝不能当成「很暗」，否则一处读图失败就会把一条好候选判出局。
    """
    import cv2                      # 仅此一处用到，放在函数内避免拖慢导入
    import numpy as np

    vals: list[float] = []
    for d in (cand.get("best_dets") or [])[:CELL_COUNT]:
        img = cv2.imread(str(d.frame_path))
        if img is None:
            continue
        x1, y1 = max(0, int(d.x1)), max(0, int(d.y1))
        crop = img[y1:int(d.y2), x1:int(d.x2)]
        if crop.size == 0:
            continue
        vals.append(float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()))
    return float(np.median(vals)) if vals else None


def candidate_is_too_poor_to_prefer(cand: dict) -> bool:
    """这条候选是不是「特别小」或「特别暗」——见 `config.EXIT_PRIORITY_MIN_
    BRIGHTNESS` 上方那段实测记录。

    **不看长宽比**：用户 2026-09-20 实测电子28 order 6，他要的那条候选正是
    被长宽比挡着的（`filter_reasons == ["aspect"]`），把它算进门槛会让这一
    处判得更错。长宽比说的是「这可能是局部检测」，与「看不看得清这个人」
    不是一回事。
    """
    ratio = cand.get("height_ratio")
    if ratio is not None and float(ratio) < config.ENTRYEXIT_MIN_HEIGHT_RATIO:
        return True
    b = candidate_brightness(cand)
    return b is not None and b < config.EXIT_PRIORITY_MIN_BRIGHTNESS


def _merge_same_seat_candidates(cands: list[dict]) -> dict:
    """同一座位号下的多个候选合并为一条，取证据更充分的那个。

    Task 45 追加改动四 · 缺陷一（控制器用真实标注实测出的缺陷，不是用户
    标错）：同一人往往在进场、退场两个窗口各留下一个候选，用户在复核页
    两处都标同一座位号是完全合理的操作，不代表真有两个人。实测 order 9
    华韵传媒：`entry:19`/`exit:10` 都标了座位2，未合并时被当成两个人，
    第二个发现座位2已占就滑到其它空位——顶掉了本该在那个座位的人。

    合并优先级：
      0. **先过质量门槛**（用户 2026-09-20 实测电子28 order 6 座位 3）：
         「特别小」或「特别暗」的候选退到后面，只要同座位还有别的候选就不
         用它（`candidate_is_too_poor_to_prefer`）。当时那个座位落到一条
         9 帧、框高 122 像素、亮度 9.6 的退场画面上——人已走出画面右缘，
         只剩一条边——而同座位另有一条 407 像素高、亮度 79 的进场画面。
         exit 优先的前提是那张图看得清人；前提不成立时它不该生效。
         **全都不过门槛时这一步不起作用**，照常按下面两条取舍：门槛只用来
         在有得选的时候别选最差的那个，不负责把谁踢出局。
      1. **exit 优先于 entry**——用户提供的领域知识：创业者退场时多为
         正面拍摄、进场时多为背影拍摄，正面更利于衣着与性别判定。这也是
         本次修复的直接诱因：递补 5 人时性别一致仅 2/5、置信度中位 0.45
         明显低于就座候选的 0.62，混入了本不该采信的背影候选。
      2. 同为 exit（或同为 entry）时按证据充分程度：帧数（n_frames）更多
         者优先，帧数相同再取框高（median_h）更大者——离镜头越近通常画面
         越清晰。

    只有一个候选时原样返回，不做任何比较（也不读图）。
    """
    if len(cands) == 1:
        return cands[0]
    decent = [c for c in cands if not candidate_is_too_poor_to_prefer(c)]
    pool = decent if decent else cands
    exits = [c for c in pool if c["kind"] == "exit"]
    pool = exits if exits else pool
    return max(pool, key=lambda c: (c["n_frames"], c["median_h"]))


def _same_person_cells(cands: list[dict], winner: dict) -> tuple[dict, list[int]]:
    """胜出候选**自己**占了哪几格，并把代表换成这几格里号最小的那一份。

    Task 56A · 根因就在这里（用户 2026-08-24 实测文化4组 order 2）：格位 1
    与「同格位1」的格位 2 解引用成同一个人后，两份切片带着同一个座位号进
    `by_seat`，`_merge_same_seat_candidates` 只留下一份，另一格连同它那张
    帧一起消失——没有任何一处报错，登记表看上去完全正常，只是少了一半证据。

    **只收胜出那条候选自己的格位号**（按 `stable_id`+`kind`+`track_id` 认
    亲）。另一条候选的「第 3 格」是在**它自己那张证据图**上编号的，混进来
    等于拿另一张图的坐标去取帧——那正是 Task 45 缺陷一（同一人在进/退场各
    留一个候选）被合并时最容易犯的错。两条 track 的画面绝不合成一个人的证据。

    代表取号最小的那一格，与 `cells_dets` 的升序、与跨段那一路的取舍完全
    一致；`cand["layout"]`（那份单图构成文案里写着「第 N 格」）因此始终与
    计划项的 `cell_index` 说的是同一格，不会互相矛盾。
    """
    key = (winner.get("stable_id"), winner.get("kind"), winner.get("track_id"))
    same = [c for c in cands
            if c.get("cell_index")
            and (c.get("stable_id"), c.get("kind"), c.get("track_id")) == key]
    if not same:
        # 胜出的是一条没被切分过的普通候选（用户直接给它标了座位号）——
        # 计划项的形状一字不变，不凭空多出格位信息。
        return winner, []
    same.sort(key=lambda c: int(c["cell_index"]))
    return same[0], sorted({int(c["cell_index"]) for c in same})


def entryexit_fill_plan(
    segments: list[Segment],
    dets: list[Detection],
    labels: dict,
    seats_by_order: dict[int, dict[int, int]],
    known_orders=None,
) -> tuple[dict[int, list[dict]], list[dict], list[dict]]:
    """对每段算出：哪些被标为创业者的进出场 track 应补进缺的座位。

    只补**尚未被就座阶段占用**的座位号（`seats_by_order` 里已有的座位号
    一律跳过，不覆盖）。

    只处理 `role == "创业者"` 的标注（Task 41 改动二：新增的「上一组创业者」
    「下一组创业者」标注的是相邻项目的人，与「工作人员」「专家」「不确定」
    一样一律不参与补位——用字符串精确相等比较，不用子串匹配，避免
    「上一组创业者」被误当成「创业者」）。

    Task 41 改动三：候选标注了 `n_people > 1`（一个框里挤了不止一个人）时，
    **不参与补位**——绝不能拿一个混了两个人的框去编码衣着（Ruling R14 的
    教训：证据图上排是甲的入场、下排是乙的就座，模型给出的正式度与颜色是
    两人混合，不报错、表格照常生成、数字全错）。这批候选改为收进第二个
    返回值 `needs_resplit`，交由后续任务实现「重新切分」。标注里没写
    `n_people`（旧格式或用户未触碰过这个下拉）时按 1 处理，正常补位——
    默认值必须是「未标注=正常」而不是「未标注=可疑」，否则会把绝大多数
    正常候选错误地挡在补位计划之外。

    Task 45 追加改动四（控制器用户用真实标注实测：递补5人性别一致仅2/5、
    全组一致率83%→77%净损失，逐个查根因发现两处 `entryexit_fill_plan`
    自身的缺陷，不是用户标错）：

      · **缺陷一 · 同座位号候选合并**：用户明确指定了座位号的候选，若
        多个候选指向同一座位号（典型是同一人在 entry/exit 两个窗口各被
        标了一次），合并为一条而非当成两个人分别处理（否则第二个会滑到
        其它空位，顶掉本该在那的人）。合并规则见 `_merge_same_seat_
        candidates`（exit 优先，帧数/框高为证据充分度的次要判据）。
        **座位号为 None（用户未指定）的候选不参与这次合并**——它们可能
        真的是不同的人，只是都还没被赋予座位号，见下一条。

      · **缺陷二 · 座位号必须由用户明确指定**：`config.BACKFILL_REQUIRE_
        SEAT=True`（默认）时，用户标为创业者但未指定座位号的候选**不
        参与补位**，改为收进第三个返回值 `needs_seat`。按 `median_cx`
        猜座位号是几何推断，而几何路线在本项目已连续失败四次（深度分
        不开专家、覆盖率过滤错砍碎片、站立者框底与前排齐平、右区阈值
        曾被误判为病灶）；递补的全部价值在于提高准确度，不该在这一步
        引入猜测（实测 order 13 爱尚游：两个未指定座位号的候选被猜错，
        VLM 判定的性别与人工标注的「女女女」完全对不上）。
        `config.BACKFILL_REQUIRE_SEAT=False` 时退回旧逻辑：按 `median_cx`
        从右到左（cx 降序）依次填入剩余最小的座位号，标记
        `seat_uncertain=1`——这批座位号是猜的，不是用户确认的，下游必须
        能区分开；此时 `needs_seat` 恒为空列表。

    Task 49（用户 2026-08-23 的三条需求）：

      · **需求一 · needs_resplit 带上备注**：`n_people>1` 的候选转入
        `needs_resplit` 时一并带上标注里的 `note` 原文。备注是用户按证据图
        的格位号写下的「1 号格是创业者、2 号格是工作人员」（格位编号见
        `annotate.build_evidence_sheet(label_cells=True)` 与
        `annotate.cell_label_convention`），是重新切分唯一的依据；丢了它，
        `needs_resplit` 只剩「这个框有 2 个人」，谁是谁无从得知。
        **本函数不做「按备注自动切分」**——那需要解析自然语言，本任务只把
        编号与备注采集好，切分另做。

      · **需求二 · 人工否决已定座位 / 指定替代画面**（`labels
        ["seat_overrides"]`，见 `load_entry_labels` 的文件结构）：
        `rejected` 为真的座位视为**空缺**（可被递补，等同于就座阶段没定
        过它）；带 `replacement_stable_id` 的座位直接用那个候选补上（该
        标识同样视为「这个座位不再由就座阶段的结果占着」）。指定的候选
        找不到时**不静默丢弃**——打印出来，该座位仍算空缺但不拿一个对不
        上的候选去补（`load_entry_labels` 那一侧已经先做过一轮同样的报告
        与清理，这里是直接调用本函数时的兜底）。

      · ~~需求三 · 按位置排除~~（`labels["position_exclusions"]`）：
        **Task 53 已按用户要求删除**（原话：「这个选项实际没用，删掉吧」）。
        旧文件里残留该键时 `load_entry_labels` 打印一行说明后忽略，本函数
        一律不再读它。

    Task 51 · **按格位备注切分**（用户 2026-08-23）：Task 49 只把格位编号与
    备注采集下来，切分留给了本任务。现在 `n_people > 1` 的候选先过
    `parse_cell_note_report`：

      · 解析出格位 → `split_candidate_by_cells` 按格位切成若干个「只有
        一张图」的单人候选，其中目标精确等于「创业者」的那些照常参与补位
        （同座位号合并、空座位判定等后续步骤与普通候选完全共用一套代码，
        不另开一条路径）。计划项额外带 `cell_index` 与 `layout`，下游据此
        **只取那一格的一张图**——不带这两个字段，下游会拿整条 track 的四张
        证据帧去测颜色、拼证据图，而那四张里坐着不同的人，正是切分要消灭
        的错误。
      · 解析不出格位 → 仍进 `needs_resplit`（连同备注原文）。「另两个」
        「其余」「左边那个」这类相对指代**一律不猜**，未指明的格位留空。
      · 切了几个候选、切出几张单人画面、补进几个座位、几条短语没解析出来
        ——**全部打印**（凡有切分必须可见）。静默切分的后果：用户写的
        「另两个是混入人员」被忽略了半句，他永远不会知道。

    Task 53 · **角色与备注的优先级**（用户 2026-08-23：「必须选择创业者，
    备注才会生效吗」）：旧实现把 `role == "创业者"` 当成硬闸门，角色不对时
    备注**根本不会被读到**。但一个 2×2 框里三个人分属不同座位（甚至分属
    相邻项目）时，「这个候选的角色」本身就没有单一答案——用户只能挑一个
    语义不准的值去骗过闸门。现在的判据顺序（顺序即优先级）：

      1. 备注能解析出格位映射 → **按备注切分补位**，`role` 字段此时仅作
         参考，不作为闸门；
      2. 备注解析不出（含没写备注）→ 沿用旧判据：`role == "创业者"` 才补位；
      3. 角色为工作人员/专家/上一组创业者/下一组创业者**且**无可解析备注
         → 一律不补（与 Task 41 改动二完全一致）。

    Task 54 · **格位改用结构化选择题**（用户 2026-08-23：「能放弃解析，转换
    为选择吗……根据选出的空，精准识别，不需要解析了」）：标注里新增 `cells`
    字段（`[{"cell":1,"role":"创业者","seat":2}, {"cell":3,"same_as":1}]`），
    优先级排在备注**之前**——有 cells 就完全按 cells，note 一个字都不生效。
    优先级表与「cells 全空等同于还没填」的理由见 `cell_map_of_label`；
    「同格位X」的引用语义见 `cells_to_cell_map`。备注解析器**保留不删**：
    用户已按备注标完两组，作废等于让他重标。

    Task 56A · **同一个人占多格时，多格的帧全部保留**（用户 2026-08-24 实测
    文化4组 order 2：「标了格位1是创业者2，格位2是同格位1，那么应该将这两帧
    图片都保留下来，而不是登记表只有一张图」）。两格解引用成同一个人 → 仍只
    占**一个**座位（这一点不变），但计划项额外带 `cell_indices` 列出全部格
    位，`pipeline.registry` 据此把每一格的那张都留档。旧实现在两处把多出来
    的格位当成"重复的人"丢掉，且都不报错：同段是 `_merge_same_seat_
    candidates` 只留一份，跨段是第二遍扫描的「该座位已有人，不覆盖」。

    **角色与备注冲突时以备注为准**（例如 `role="工作人员"` 而备注写
    「1是创业者2」）。理由：备注是用户为了描述格位专门写下的、更具体的一句
    话，而角色下拉只能有一个取值，在多人框上本就表达不了「谁是谁」。反过来
    以角色为准，等于把用户明确写出的信息丢掉且不报错——本项目最坏的失败
    模式。判据 2 保证这条不会变成「随便写句话就能放行」：**解析不出的备注
    不给任何豁免**，角色闸门照旧生效。

    Task 53 · **跨段流转**（需求六）：备注里目标为「下一组创业者N」（用户
    真实写法是「下一场创业者N」，`_ROLE_SYNONYMS` 归一）的格位，映射到
    **下一段（order+1）**的座位 N；「上一组创业者N」同理映射到 order-1。
    实测城市2组段 15 有一条这样的标注，而段 16 恰好缺人且整段无标注——这条
    信息此前只被排除、没被利用。

    **必须谨慎**（三条硬规则，都在第二遍扫描里实现）：

      · 跨段流转的候选，其检测帧属于**本段的时间窗口**，不属于目标段。
        补位后一律标记 `cross_segment=1` 与 `src_order=<本段 order>`，
        并落到座位登记表（`pipeline.registry`）与 CSV，让下游能区分。
        不打标记的后果：下游拿到一条「段 16 座位 2」的记录，按段 16 的
        时间范围去核对那几帧，找不到，而流水线照常跑完。
      · 目标段号越界（order+1 超过本次传入的 segments）→ **跳过，不报错**；
        目标座位号超出目标段的 `presenter_num` 同样跳过。跳过了什么一律
        打印（凡有丢弃必须可见）。
      · 目标座位已被占用（就座阶段占着，或目标段自己的标注已经补了它）
        → **不覆盖**，沿用既有规则。

    第二遍扫描放在主循环之后，因为「段 15 要往段 16 塞人」这件事在处理段
    15 时还不知道段 16 最终占了哪些座位（段号大的后处理）。

    返回 `(plan, needs_resplit, needs_seat)`：
        plan: {seg_order: [{"track_id", "kind", "seat", "seat_uncertain"}, ...]}，
              没有可补内容的段不出现在返回值里。**按格位切出来的项**额外带
              `"cell_index"`（代表格位号，1 起）、`"cell_indices"`（同一个
              人占的全部格位，升序，Task 56A）与 `"layout"`（单图构成文案），
              **跨段流转来的项**再额外带 `"cross_segment"`（恒为 1）与
              `"src_order"`（这些帧实际所属的段），见 `_plan_entry`；
              普通项的键与 Task 44 起逐字相同。
        needs_resplit: [{"order", "kind", "track_id", "n_people", "note"}, ...]，
              全部段汇总在一份列表里（跨段用 "order" 字段区分）。Task 51 起
              只收「备注解析不出格位」的那些。
        needs_seat: [{"order", "kind", "track_id"}, ...]，同上汇总方式；
              `BACKFILL_REQUIRE_SEAT=False` 时恒为空列表。
    """
    plan: dict[int, list[dict]] = {}
    needs_resplit: list[dict] = []
    needs_seat: list[dict] = []

    overrides_all = labels.get("seat_overrides") or {}
    unresolved_replacements: list[tuple[int, int, str]] = []
    # Task 53 · 需求六：跨段流转请求，主循环里只**收集**，第二遍才落位
    # （处理段 15 时还不知道段 16 最终占了哪些座位）。
    # 每项 `(目标段号, 本段段号, 切出来的那份候选, 目标座位号)`。
    cross_requests: list[tuple[int, int, dict, int]] = []
    cross_skipped: list[str] = []
    # 指向已有人座位的流转：不换人，画面按补格位并入（见下方打印处）
    cross_merged: list[str] = []
    n_segments_scanned = 0
    # Task 51 · 切分统计（凡有切分必须可见：切出几张、补进几个座位、几条
    # 短语没解析出来、几个格位号超出实际张数，一律打印）。
    n_split_cands = 0
    n_split_pieces = 0
    n_cells_out_of_range = 0
    split_unparsed: list[str] = []

    for segment in segments:
        seg_labels = labels.get(str(segment.order)) or {}
        seg_overrides = _seg_overrides(overrides_all, segment.order)
        if not seg_labels and not seg_overrides:
            continue

        # 配对用**展示 ∪ 折叠**的池子：用户在折叠区里认出并标定的人也要能
        # 补进来（见 candidates_for_labels 的 docstring）。
        candidates = candidates_for_labels(dets, segment)

        # Task 46：候选 ↔ 标注的配对。**匹配即消耗**——`pool` 是本段标注的
        # 副本，配上一条弹出一条，保证一条标注不会被两个候选同时拿走。
        # 三轮，顺序不可颠倒：
        #   1) 稳定标识精确命中（同一份检测、标注未过期时的常态）；
        #   2) 旧格式 `kind:track_id` 精确命中——保留这条只为兼容直接调用
        #      本函数、手写旧格式标注的既有调用方；正式流水线上旧格式已被
        #      `load_entry_labels` 整份作废，走不到这里；
        #   3) 稳定标识模糊匹配（`match_label_to_candidate`），吸收重检测
        #      带来的像素级抖动。
        # 先把两轮精确的全做完再做模糊，否则某个候选可能凭「近」抢走另一个
        # 候选本该精确命中的那条标注。
        pool = dict(seg_labels)
        paired: list[tuple[dict, dict]] = []
        unpaired: list[dict] = []
        for cand in candidates:
            lab = pool.pop(cand["stable_id"], None)
            if lab is None:
                lab = pool.pop(f"{cand['kind']}:{cand['track_id']}", None)
            if lab is None:
                unpaired.append(cand)
            else:
                paired.append((cand, lab))
        for cand in unpaired:
            lab = match_label_to_candidate(pool, cand)
            if lab is not None:
                paired.append((cand, lab))

        # needs_resplit 的检出与「是否还有空座位」无关（哪怕座位已坐满，一个
        # 混了多人的框依然需要被上报去重新切分），所以独立于下面的补位逻辑
        # 扫描全部标注。
        presenters: list[tuple[dict, int | None]] = []
        for cand, lab in paired:
            if not isinstance(lab, dict):
                continue
            # --- Task 53 需求一 + Task 54 需求一：判据顺序即优先级 --------
            # 0) 有结构化的 `cells`（挑选页的格位选择题）→ 按它切分，
            #    **忽略 note**；
            # 1) 无 cells 但备注能解析出格位 → 按备注切分，角色仅作参考；
            # 2) 备注也解析不出（含没写）→ 角色闸门照旧（role == "创业者"）；
            # 3) 其余角色 + 无 cells 无可解析备注 → 一律不补。
            #
            # 先取格位映射**再**看角色，顺序不可颠倒：颠倒回去就是旧实现的
            # 硬闸门，用户为了让备注生效必须先谎报一个角色。
            note = lab.get("note") or ""
            cell_map, report, _cell_src = cell_map_of_label(lab)
            role = lab.get("role")

            if cell_map:
                # 判据 1：以备注为准。备注是用户为了描述格位专门写的，比
                # 只能取单值的角色下拉更具体；冲突时角色让路（`role` 在这
                # 一分支里不参与任何判断，只随 needs_* 的诊断信息带出去）。
                split_unparsed.extend(report["unparsed_phrases"])
                pieces = split_candidate_by_cells(cand, cell_map)
                n_split_cands += 1
                n_split_pieces += len(pieces)
                n_cells_out_of_range += len(cell_map) - len(pieces)
                for piece in pieces:
                    prole = piece.get("role")
                    if prole == "创业者":
                        # 精确等于「创业者」的格位补进**本段**。「混入人员」
                        # 「工作人员」这些格位就此打住——用户已经指明了它们
                        # 不是创业者，这正是切分要保住的信息。
                        presenters.append((piece, piece.get("seat")))
                    elif prole in _NEIGHBOUR_ROLE_DELTA and piece.get("seat"):
                        # Task 53 · 需求六：流转到相邻段，第二遍再落位。
                        cross_requests.append((
                            segment.order + _NEIGHBOUR_ROLE_DELTA[prole],
                            segment.order, piece, int(piece["seat"]),
                        ))
                continue

            # 判据 2/3：没有可解析的备注，角色闸门照旧。**解析不出的备注
            # 不给任何豁免**——否则闸门就退化成「随便写句话就能过」。
            #
            # Task 56E（用户 2026-08-24 实测新能源2组 order 2）：整条候选标成
            # 「上一组/下一组创业者」并给了座位号时，也要流转到相邻段。此前
            # 跨段只认格位备注那条路，这里一个 `continue` 把它连同座位号一起
            # 丢掉、一声不吭——而镜头里只有一个人、他属于隔壁那家企业时，本来
            # 就没什么可切分的，用户就是想整条指过去。
            if role in _NEIGHBOUR_ROLE_DELTA:
                if lab.get("seat"):
                    cross_requests.append((
                        segment.order + _NEIGHBOUR_ROLE_DELTA[role],
                        segment.order, cand, int(lab["seat"]),
                    ))
                else:
                    # 没给座位号就**不猜**。按 median_cx 猜是几何推断，本项目
                    # 已连续失败四次；猜错等于把这个人的衣着编进隔壁企业另一
                    # 个人的行里，而流水线照常跑完、不报错。报出来让用户去补。
                    needs_seat.append({
                        "order": segment.order, "kind": cand["kind"],
                        "track_id": cand["track_id"], "role": role,
                    })
                continue
            if role != "创业者":
                continue
            n_people = lab.get("n_people") or 1
            if n_people > 1:
                needs_resplit.append({
                    "order": segment.order, "kind": cand["kind"],
                    "track_id": cand["track_id"], "n_people": n_people,
                    # 备注是按格位号写的「1 号格是创业者、2 号格是工作
                    # 人员」，重新切分唯一的依据，绝不能在这里丢掉。
                    "note": note,
                })
                split_unparsed.extend(report["unparsed_phrases"])
                continue
            presenters.append((cand, lab.get("seat")))

        # --- 需求二：被否决/被指定替换的座位视为空缺（可被递补）---
        occupied = set(seats_by_order.get(segment.order, {}).values())
        vacated = _vacated_seats(seg_overrides)
        occupied -= vacated
        free_seats = sorted(set(range(1, segment.presenter_num + 1)) - occupied)

        entries: list[dict] = []
        used_ids: set[str] = set()

        # 用户明确指定的替代画面先落位：这是最具体的一条人工指令，比任何
        # 按角色/座位号的推断都更该被采信（也包括位置排除——那是组级的
        # 概括，这是针对这一张画面的判断）。
        for seat in sorted(vacated):
            rec = seg_overrides.get(str(seat)) or {}
            sid = rec.get("replacement_stable_id")
            if not sid:
                continue
            cand = find_candidate_by_stable_id(str(sid), candidates)
            if cand is None:
                unresolved_replacements.append((segment.order, seat, str(sid)))
                continue
            if seat not in free_seats:
                continue
            free_seats.remove(seat)
            used_ids.add(cand["stable_id"])
            entries.append({
                "track_id": cand["track_id"], "kind": cand["kind"],
                "seat": seat, "seat_uncertain": 0,
            })

        # 已被当作替代画面用掉的候选不再参与后面的补位——同一个人不能同时
        # 坐两个座位（Task 45 缺陷一的同类错误）。凡有丢弃必须可见：这种
        # 情形是用户的两条输入自相矛盾（既把这个人标成 X 号位，又把他指定
        # 为 Y 号位的替代画面），静默取其一等于替他做了个他不知道的决定。
        conflicting = [item for item in presenters if item[0]["stable_id"] in used_ids]
        if conflicting:
            print(f"   ⚠ 第 {segment.order} 段：{len(conflicting)} 个候选既被标了"
                  f"座位号、又被指定为另一个座位的替代画面，已按「替代画面」"
                  f"处理（一个人不能同时坐两个座位）："
                  + "、".join(f"{c['stable_id']}（标注座位 {seat}）"
                               for c, seat in conflicting))
        presenters = [item for item in presenters if item[0]["stable_id"] not in used_ids]

        if not presenters:
            if entries:
                plan[segment.order] = entries
            continue

        specified = [item for item in presenters if item[1] is not None]
        # 座位号为 None 的候选不参与「同座位号合并」——它们可能真是不同的
        # 人，合并需要一个共同的座位号作为"同一人"的证据，None 提供不了。
        unspecified = [item for item in presenters if item[1] is None]

        # --- 缺陷一：合并指向同一座位号的多个候选 ---
        by_seat: dict[int, list[dict]] = {}
        for cand, seat in specified:
            by_seat.setdefault(int(seat), []).append(cand)
        # Task 56A：合并成一条计划项的同时，把**胜出候选自己**占的全部格位
        # 号一并带上（`_same_person_cells`）。同一个人占两格时旧实现只留下
        # 一格，另一格那张帧静默消失。
        merged_specified = sorted(
            (_same_person_cells(cands, _merge_same_seat_candidates(cands)) + (seat,)
             for seat, cands in by_seat.items()),
            key=lambda item: item[2],
        )

        for cand, cells, seat in merged_specified:
            if seat in free_seats:
                free_seats.remove(seat)
                entries.append(_plan_entry(cand, seat, cell_indices=cells))

        # --- 缺陷二：座位号未指定时是否参与补位 ---
        if config.BACKFILL_REQUIRE_SEAT:
            for cand, _seat in unspecified:
                needs_seat.append({
                    "order": segment.order, "kind": cand["kind"],
                    "track_id": cand["track_id"],
                })
        else:
            # 旧逻辑（保留，仅在显式关闭前置条件时才走）：按 median_cx
            # 从右到左（降序）依次填入剩余最小的座位号，标记 seat_uncertain=1。
            for cand, _seat in sorted(unspecified, key=lambda item: -item[0]["median_cx"]):
                if not free_seats:
                    break
                seat = free_seats.pop(0)
                entries.append(_plan_entry(cand, seat, seat_uncertain=1))

        if entries:
            plan[segment.order] = entries

    # --- Task 53 · 需求六：跨段流转的第二遍扫描 -----------------------------
    # 放在主循环之后：处理段 15 时还不知道段 16 最终占了哪些座位（段号大的
    # 后处理），先收集再落位，避免「谁先跑到谁占坑」这种取决于遍历顺序的
    # 行为。同一目标座位有多份（备注里 2/3 两格都指同一个人）时以**格位号
    # 最小**的那一份为代表——与同段的 `_same_person_cells` 完全一致。
    #
    # Task 56A：代表之外的那几格**不再当成"重复的人"丢掉**。旧实现靠下面
    # 那句「该座位已有人，不覆盖」把第 3 格挡掉，还打印成一条未落位记录，
    # 而它其实是同一个人的另一张证据（用户实测：登记表因此只剩一张图）。
    # 分组键带上候选标识（`stable_id`+`kind`+`track_id`）：两个**不同**候选
    # 流向同一个座位仍是「两个人抢一个座位」，照旧先到先得、后到的打印跳过。
    seg_by_order = {int(sg.order): sg for sg in segments}
    # 整组有哪些段号。调用方按窗口分批传 `segments` 时（registry、assemble
    # 都是），拿它区分「不在本次窗口里」与「整组真的没有这一段」。不传就退回
    # 按传入的那几段判断，行为一字不变。
    # None＝调用方没说整组有哪些段，那就退回旧行为（照报）。
    # 不要写成 `else set(seg_by_order)`：走到下面那个守卫时，目标段**必然**不在
    # seg_by_order 里，所以那个默认值恒等于空集——看着像一道防线，实际是死代码
    # （变异测试确认为等价变异）。
    known = ({int(o) for o in known_orders} if known_orders is not None
             else None)
    grouped_cross: dict[tuple, dict] = {}
    for target_order, src_order, piece, seat in cross_requests:
        key = (int(target_order), int(seat), int(src_order),
               piece.get("stable_id"), piece.get("kind"), piece.get("track_id"))
        cell = int(piece["cell_index"]) if piece.get("cell_index") else None
        group = grouped_cross.get(key)
        if group is None:
            grouped_cross[key] = {
                "target_order": int(target_order), "src_order": int(src_order),
                "piece": piece, "seat": int(seat),
                "cells": [cell] if cell is not None else [],
            }
            continue
        if cell is None or cell in group["cells"]:
            continue
        group["cells"].append(cell)
        group["cells"].sort()
        if group["cells"][0] == cell:
            # 号最小的那一格作代表：它的 `layout`（写着「第 N 格」）必须与
            # 计划项的 `cell_index` 指同一格，否则送进模型的构成文案说的是
            # 另一格的画面。
            group["piece"] = piece

    n_cross_filled = 0
    for group in grouped_cross.values():
        target_order, src_order = group["target_order"], group["src_order"]
        piece, seat, cross_cells = group["piece"], group["seat"], group["cells"]
        target = seg_by_order.get(int(target_order))
        if target is None:
            # 目标段不在本次传入的 `segments` 里。两种情形，后果完全不同：
            #
            #   · 整组确实没有这一段（最后一段还写「下一场创业者」）→ 这条流转
            #     永远落不了地，必须说出来。
            #   · 只是**不在本次窗口里** → `registry`/`assemble` 逐段跑时传的是
            #     order-1/order/order+1 三段；处理第 4 段时窗口 [3,4,5]，第 5 段
            #     那条指向第 6 段的流转就"看不见"第 6 段。等窗口挪到 [5,6,7] 它
            #     会正常落位——这时报「没有第 6 段」是**假警报**。
            #
            # 用户 2026-08-25 实测新能源2 就撞上后者：日志写着「没有第 6 段」，
            # 而第 6 段存在、那条流转也确实落位了（登记表里 cross_segment=1、
            # src_order=5）。假警报比不报更坏：他会去查一个不存在的问题，久了
            # 就不再信任告警。
            if known is None or int(target_order) not in known:
                cross_skipped.append(
                    f"第 {src_order} 段 → 第 {target_order} 段座位 {seat}"
                    f"（没有第 {target_order} 段）")
            continue
        if not (1 <= seat <= int(target.presenter_num)):
            cross_skipped.append(
                f"第 {src_order} 段 → 第 {target_order} 段座位 {seat}"
                f"（该段只有 {target.presenter_num} 人的编制）")
            continue
        occupied = set((seats_by_order.get(int(target_order)) or {}).values())
        occupied -= _vacated_seats(_seg_overrides(overrides_all, int(target_order)))
        taken = {e["seat"] for e in plan.get(int(target_order), [])}
        if seat in occupied or seat in taken:
            # 沿用既有规则：已经有人的座位一律不覆盖。跨段来的证据比本段
            # 自己的弱（帧不在目标段的时间窗口里），更没有理由顶掉别人。
            #
            # 但这**不是未落位**（用户 2026-09-27「修改措辞」）：计划里不另占
            # 座位，它的画面却会按补格位并进该座位的 2×2（registry/assemble
            # 的 `topup_pairs`，多个来源轮流出格）。原来挂在「⚠ 未落位」下面，
            # 用户据此一直当成遗留问题；实测电子10 那 7 条全部已经并入。
            cross_merged.append(
                f"第 {src_order} 段 → 第 {target_order} 段座位 {seat}")
            continue
        plan.setdefault(int(target_order), []).append(
            _plan_entry(piece, seat, cross_segment=1, src_order=int(src_order),
                        cell_indices=cross_cells))
        n_cross_filled += 1
    if n_cross_filled or cross_skipped:
        print(f"   ⇄ 跨段流转（「上/下一组创业者」的格位备注）：补进 "
              f"{n_cross_filled} 个相邻段的座位；这些画面来自**源段**的时间"
              f"窗口，计划项已标 cross_segment=1 与 src_order")
    if cross_merged:
        print(f"   ⇄ {len(cross_merged)} 条跨段流转指向已有人的座位——不换人，"
              f"画面并入该座位的 2×2（与已有画面轮流出格，重复画面除外）："
              + "、".join(cross_merged))
    if cross_skipped:
        print(f"   ⚠ {len(cross_skipped)} 条跨段流转未落位："
              + "、".join(cross_skipped))

    # Task 51 · 凡有切分必须可见。静默切分的后果：用户在备注里写的
    # 「另两个是混入人员」被忽略了半句，他永远不会知道，也就永远不会去
    # 补写成明确的格位号。
    n_split_filled = sum(1 for items in plan.values() for e in items
                         if e.get("cell_index") and not e.get("cross_segment"))
    if n_split_cands or split_unparsed:
        print(f"   ⇄ 按格位备注切分：{n_split_cands} 个多人候选切出 "
              f"{n_split_pieces} 张单人画面，补进 {n_split_filled} 个座位")
    if split_unparsed:
        print(f"   ⚠ {len(split_unparsed)} 条备注短语未能解析（相对指代如"
              f"「另两个」「其余」「左边那个」一律不猜，对应格位留空）："
              + "、".join(f"「{p}」" for p in split_unparsed))
    if n_cells_out_of_range:
        print(f"   ⚠ {n_cells_out_of_range} 个格位号超出了证据图实际的张数，"
              "已忽略（宁可不补，也不拿相邻那一格另一个人的画面顶上）")

    if unresolved_replacements:
        detail = "、".join(f"第 {order} 段座位 {seat} → {sid}"
                           for order, seat, sid in unresolved_replacements)
        print(f"   ⚠ {len(unresolved_replacements)} 个「改用我指定的候选」在"
              f"当前候选里找不到，该座位仍算空缺但不会拿对不上的候选去补："
              f"{detail}")

    return plan, needs_resplit, needs_seat


def _plan_entry(cand: dict, seat: int, seat_uncertain: int = 0,
                cross_segment: int = 0, src_order: int | None = None,
                cell_indices: list[int] | None = None) -> dict:
    """一条补位计划项。

    普通候选的形状与 Task 44 起逐字相同（`track_id`/`kind`/`seat`/
    `seat_uncertain`）——既有测试对整条字典做相等断言，多一个键就是行为
    变更。**只有按格位切出来的那种**才额外带 `cell_index`/`cell_indices`
    与 `layout`：

      · `cell_index`：下游（`assemble`）据此用 `cell_dets` 只取那一格的
        一张图。不带这个字段，下游会拿整条 track 的四张证据帧去测颜色、
        拼证据图——那四张里坐着不同的人，正是切分要消灭的错误。
      · `cell_indices`（Task 56A）：这个人占的**全部**格位，升序。两格是
        同一个人时（「格位2 = 同格位1」）只占一个座位，但两格的帧都是他本
        人的证据，`pipeline.registry` 据此把两张都留档。硬不变式：
        **`cell_index == cell_indices[0]`**（代表恒为号最小的那一格，见
        `_same_person_cells`）——两处说的必须是同一格，否则 `layout` 里
        写的「第 N 格」与实际取的那张对不上，且不报错。
        `cell_index` 保留而不是直接换成列表：`assemble` 目前每个座位只送
        一张图，读的就是它；把它删掉等于让整条打标路径静默退回「拿四张
        混了多人的证据帧」。
      · `layout`：**代表那一格**的证据图构成文案（`describe_layout` 的第
        4 种构成）。放在计划里而不是让下游再算一遍，是因为「切自第几格」
        这个事实只有这里知道。
    """
    entry = {
        "track_id": cand["track_id"], "kind": cand["kind"],
        "seat": seat, "seat_uncertain": seat_uncertain,
    }
    cells = sorted({int(c) for c in (cell_indices or []) if c})
    if not cells and cand.get("cell_index"):
        # 调用方没给（直接调本函数的既有路径）时退回候选自己那一格。
        cells = [int(cand["cell_index"])]
    if cells:
        entry["cell_index"] = cells[0]
        entry["cell_indices"] = cells
        # Task 56D：构成文案按**这个人占的全部格位**重算，不能沿用
        # `cand["layout"]`——那份是 `split_candidate_by_cells` 给单张切片
        # 写的，只说得出「共 1 张、第 N 格」。占两格的人拿着它去打标，模型
        # 被告知只有一张图，却收到两张，且一处也不会报错。
        #
        # 它同时是 Task 45 增量复用的「构成签名」：格位集合变了 = 文案变了
        # = 上一轮结果不会被误判为可复用。单格时 `describe_layout` 逐字给
        # 出与 Task 51 相同的原文案，因此既有的单格座位不会因本次改动全部
        # 失效重跑。
        from pipeline.assemble import describe_layout
        is_entry = cand.get("kind") == "entry"
        entry["layout"] = describe_layout(
            0,
            n_entry=len(cells) if is_entry else 0,
            n_exit=0 if is_entry else len(cells),
            cell_indices=cells,
        )
    if cross_segment:
        # Task 53 · 需求六。**只有跨段项才有这两个键**（同上，普通项的形状
        # 一字不变）。`src_order` 不是可有可无的注解：这几帧的 `seg_order`
        # 是源段，下游按目标段的时间范围去核对必然对不上，不记等于让人
        # 以为数据错了。
        entry["cross_segment"] = 1
        entry["src_order"] = int(src_order)
    return entry


def _seg_overrides(overrides_all: dict, order: int) -> dict:
    """取某一段的座位否决表，键一律归一成 `str`（JSON 的对象键是字符串，
    但直接调用本函数的调用方可能用 int）。"""
    seg = overrides_all.get(str(order))
    if seg is None:
        seg = overrides_all.get(order)
    if not isinstance(seg, dict):
        return {}
    return {str(k): v for k, v in seg.items() if isinstance(v, dict)}


def _vacated_seats(seg_overrides: dict) -> set[int]:
    """本段被人工「腾空」的座位号：标了 `rejected`，或指定了替代候选
    （指定替代本身就意味着「就座阶段那个结果不再作数」）。"""
    out: set[int] = set()
    for seat_key, rec in seg_overrides.items():
        if not (rec.get("rejected") or rec.get("replacement_stable_id")):
            continue
        try:
            out.add(int(seat_key))
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# 追加改动三（用户 2026-08-17）：导出的标注 JSON 要能落到该组的 work 目录。
#
# 浏览器不允许网页写任意路径，`<a download>` 只能把文件丢进浏览器默认下载
# 目录（entry_review.html 侧新增了 File System Access API 的"另存为"优先
# 路径，但 Safari/Firefox 不支持、用户也可能取消——那条不是可靠保障）。
# 这里是主要保障：流水线主动去下载目录里找、复制过来。
# ---------------------------------------------------------------------------

def default_downloads_dir() -> Path:
    """浏览器导出文件的默认落地目录。

    抽成函数（而不是在 `adopt_downloaded_labels` 里直接写 `Path.home() /
    "Downloads"`）供测试 monkeypatch 掉——测试必须用 tmp_path 造假的下载
    目录，不能真的读用户的 `~/Downloads`（brief 明确要求）。
    """
    return Path.home() / "Downloads"


def _downloaded_label_candidates(downloads: Path, sheet: str) -> list[Path]:
    """`downloads` 目录里与本组同名的 `entry_labels_{sheet}.json`，含浏览器
    给重名文件加的后缀变体。

    实测浏览器会产出 `entry_labels_X (1).json`（带空格）与
    `entry_labels_X(2).json`（不带空格）两种形态——用户导出多次时几乎必然
    出现，这里的正则两种都识别。`downloads` 不存在（目录被删/从未创建）时
    返回空列表，不抛异常。
    """
    downloads = Path(downloads)
    if not downloads.is_dir():
        return []
    safe = config.sanitize_filename_component(sheet)
    stem = f"entry_labels_{safe}"
    pattern = re.compile(rf"^{re.escape(stem)}(?: ?\(\d+\))?\.json$")
    return sorted(p for p in downloads.iterdir() if p.is_file() and pattern.match(p.name))


def adopt_downloaded_labels(work: Path, sheet: str,
                             downloads: Path | None = None) -> Path | None:
    """把用户从浏览器导出的标注文件收进该组的 work 目录。

    浏览器不允许网页写任意路径，导出一律落在下载目录。此函数在
    `downloads`（默认 `default_downloads_dir()`，即 `~/Downloads`）里查找
    与本组同名的 `entry_labels_{sheet}.json`（含浏览器加的重名后缀变体，
    取其中最新修改时间的一个），**且仅当它比 work 目录里那份更新**时，
    复制（不是移动）过来并返回目标路径；否则返回 None。

    复制而非移动：用户的下载文件是他的劳动成果，本流水线无权替他删掉/
    搬走它。仅当更新才覆盖：避免用一份旧的下载覆盖掉用户后来直接放进
    work 目录的新文件（例如手动放的、或上一次已经收取过的）。
    """
    downloads = Path(downloads) if downloads is not None else default_downloads_dir()
    candidates = _downloaded_label_candidates(downloads, sheet)
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)

    safe = config.sanitize_filename_component(sheet)
    target_path = Path(work) / f"entry_labels_{safe}.json"
    if target_path.exists() and newest.stat().st_mtime <= target_path.stat().st_mtime:
        return None   # work 里那份已是最新（或更新），不许覆盖

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(newest, target_path)
    return target_path


def vacated_seats_of(labels: dict | None, seg_order: int) -> set[int]:
    """某段被人工「腾空」的座位号（标了 rejected，或指定了替代候选）。

    供 `assemble.assemble_annotations` 判断哪些已占座位允许被替换。装配那边
    有一道「不许递补覆盖已有就座结果」的防御——那是为了不让**自动**递补
    悄悄改写人工可见的座位表；但用户在挑选页**显式否决**的座位不在此列，
    否则用户说了判错、页面也记进了文件，跑一遍却什么都没变且不报错。

    `labels` 为 None / 无 seat_overrides 时返回空集合，装配行为完全不变。
    """
    if not labels:
        return set()
    seg_overrides = (labels.get("seat_overrides") or {}).get(str(seg_order)) or {}
    return _vacated_seats(seg_overrides)


def seat_reassign_of(labels: dict | None, seg_order: int) -> dict:
    """某段被人工改号的已定座位：`{原座位号: 新座位号}`。

    用户 2026-08-29 实测城市2 order 6：「自动标的 seat1 实际为 seat2，这样
    自动标错的如何调整，否则标错了，无法将该图用于其他座位」。

    进出场候选上的座位号本来就随手指（在候选卡片上写几号就是几号）；**就座
    那一路**的图此前只能整体否决，要改号得绕一圈：否决座位 1 → 去进出场候选
    里找到同一个人 → 指给座位 2。那个人恰好没有好用的进出场候选时，一张本来
    很好的就座图就此浪费。这个通道让他就地改号。

    **键用 `(order, 原座位号)`**，与 `cell_exclusions_of` / `no_candidate_
    seats_of` 同一条理由：座位号是人工确定的、跨重检测稳定；track_id 重跑检测
    即全变（Task 46）。

    脏值一律跳过（非数字、0、负数、None）。**「改成自己」也跳**：那是用户点开
    下拉又选回原值的痕迹，留着会让下游以为这个座位被人工处理过。

    **两个座位改到同一个目标时抛 `ValueError`**，不是留一个。留哪一个取决于
    字典顺序，结果会随标注先后次序变而一处不报错——宁可整段不生效，让用户
    自己去改清楚。
    """
    if not labels:
        return {}
    raw = (labels.get("seat_reassign") or {}).get(str(seg_order)) or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, int] = {}
    for src_key, dst in raw.items():
        try:
            src, tgt = int(src_key), int(dst)
        except (TypeError, ValueError):
            continue
        if src < 1 or tgt < 1 or src == tgt:
            continue
        out[src] = tgt
    dupes = sorted({t for t in out.values()
                    if list(out.values()).count(t) > 1})
    if dupes:
        raise ValueError(
            f"第 {seg_order} 段有多个座位同时改到座位 "
            f"{'、'.join(str(d) for d in dupes)}——一个座位只能有一个人。"
            f"请在挑选页把它改清楚（当前映射 {out}）。")
    return out


def cell_exclusions_of(labels: dict | None, seg_order: int, seat: int) -> set[int]:
    """某座位被人工排除的证据图格位号（1 起）。

    **键用 `(order, seat)` 而不是 track_id**：座位号是人工确定的、跨重检测
    稳定；track_id 由跟踪器分配，重跑检测即全部改变（Task 46 已因此把标注
    标识改成由视频内容导出）。若用 track_id 作键，用户排掉的那一格在下次
    重检测后会静默失效，他得把每组重排一遍——而失效时不报错，最难发现。

    为什么需要这个：2×2 的四格来自同一条 track 的四帧，跟踪串号会让其中
    一帧的框实际框到别人。四格红框标的都是同一个 track，**模型无从分辨哪
    格才是目标**，会把两个人的衣着混起来判断，不报错、数字全错。

    `labels` 为 None / 无该字段时返回空集合，行为完全不变。
    """
    if not labels:
        return set()
    per_seg = (labels.get("cell_exclusions") or {}).get(str(seg_order)) or {}
    raw = per_seg.get(str(seat)) or []
    out = set()
    for v in raw:
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n >= 1:
            out.add(n)
    return out


def parse_timestamp(text) -> float | None:
    """把人工填的时刻解析成秒。解析不出返回 `None`，**绝不猜**。

    接受三种写法（用户看片时读到的就是这几种）：

        "1234.5"    秒
        "20:34"     分:秒
        "1:02:03"   时:分:秒

    猜一个数的后果很具体：按错的时刻抽出来的是**别家企业**的画面，而它会被
    写进本段目录，此后衣着/颜色/性别全部取自隔壁公司，一处不报错（见
    `frames.segment_frame_mismatches` 记的那次真实事故）。所以「20分34秒」
    「大概二十分钟」这类一律返回 None，由调用方如实报出来让人工改写。

    分/秒超出 0-59 也判为写错（"20:99"）：那多半是把秒数当成分秒写了。
    """
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    parts = t.split(":")
    if len(parts) > 3:
        return None
    try:
        nums = [float(x) for x in parts]
    except ValueError:
        return None
    if any(n < 0 for n in nums):
        return None
    if len(parts) == 1:
        return float(nums[0])
    if any(n >= 60 for n in nums[1:]):
        return None
    total = 0.0
    for n in nums:
        total = total * 60 + n
    return float(total)


_TS_SEPARATORS = ",，、;；\t\n "


def _timestamp_pieces(text) -> list[str]:
    if not isinstance(text, str):
        return []
    out, cur = [], ""
    for ch in text:
        if ch in _TS_SEPARATORS:
            if cur.strip():
                out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_timestamps(text) -> list[float]:
    """一串时刻 → 秒数列表。读不出的那几段跳过（由
    `unparsable_timestamps` 如实报出），能读的照用。

    一个座位可以填**多个**时刻：那个人可能在几处露过面。用户 2026-09-09 给
    城市1 order 4 座位 1 填的就是 `1:01:12,1:24:21`（进场那一下和退场那一下）
    ——只认单个值的话整条解析失败，他填的两个一个都用不上。

    分隔符认逗号（中英文）、顿号、分号、空白。
    """
    out = []
    for piece in _timestamp_pieces(text):
        v = parse_timestamp(piece)
        if v is not None:
            out.append(v)
    return out


def unparsable_timestamps(text) -> list[str]:
    """读不出的那几段原文。**不猜**：按错的时刻抽出来的可能是别家企业的画面。"""
    return [p for p in _timestamp_pieces(text) if parse_timestamp(p) is None]


def no_candidate_notes_of(labels: dict | None, seg_order: int) -> dict[int, dict]:
    """某段「无可用画面」座位旁边人工填的重抽线索：
    `{座位号: {"at": 秒数|None, "text": 原文}}`（用户 2026-09-09）。

    用户原话：「标记为『座位无可用画面』时，给出一个 note 窗口，填写该座位
    重抽的时间戳，供重抽参考」。此前只能记下「这个座位找不到人」，加密抽帧
    只好按预设窗口整段再抽一遍；而用户往往**眼睛已经看到那个人在第几分钟出现
    了**，只是那一带没被抽到。

    **与 `no_candidate_seats` 分开一条通道**：那一条的值是数组、三个组的既有
    文件里都是这个形状，改形状会让它们读不进来。这一条是新加的对象通道，旧
    文件里没有就是空字典，行为与此前一字不差。

    时刻解析只在 `parse_timestamp` 一处（页面不另写一份 JS 解析器——两处各写
    一份必然走样，而走样的表现是页面显示的秒数与真正抽帧的位置不是一个地方）。
    解析不出的一律留 `None` 并把原文留在 `text` 里，由调用方如实报出来。
    """
    if not labels:
        return {}
    raw = (labels.get("no_candidate_notes") or {}).get(str(seg_order)) or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, dict] = {}
    for seat_key, rec in raw.items():
        try:
            seat = int(seat_key)
        except (TypeError, ValueError):
            continue
        if seat < 1 or not isinstance(rec, dict):
            continue
        raw = rec.get("at")
        out[seat] = {"ats": parse_timestamps(raw),
                     "unparsed": unparsable_timestamps(raw),
                     "text": str(rec.get("text") or "")}
    return out


def no_candidate_seats_of(labels: dict | None, seg_order: int) -> set[int]:
    """某段被人工标为「本座位在所有候选里都没有合适画面」的座位号（Task 56C）。

    用户 2026-08-24 原话：「如果某个座位的人没有合适的候选图，能标记下来，
    再抽帧二次有更多的备选吗」。此前这种座位只会一直显示为「未确定」，与
    「还没标」混作一谈——用户下次打开挑选页无从分辨哪些是真找不到、哪些
    只是没标完，而目录颜色正是他决定「还要标哪些」的唯一依据。

    **键用 `(order, seat)` 而不是 track_id**，与 `cell_exclusions_of` 同理：
    座位号是人工确定的、跨重检测稳定；track_id 重跑检测即全变（Task 46）。
    这一条在本功能上尤其要紧——标记的唯一用途就是驱动加密抽帧，而加密抽帧
    之后必然重检测。

    脏值（非数字、0、负数）一律跳过：手改过的文件不该让整条流水线崩掉，
    但「0 号座位」这种不存在的东西也绝不能当真。
    """
    if not labels:
        return set()
    raw = (labels.get("no_candidate_seats") or {}).get(str(seg_order)) or []
    out = set()
    for v in raw:
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n >= 1:
            out.add(n)
    return out


def labelled_seat_candidates(dets: list, segment, labels: dict | None,
                             neighbours=None) -> dict:
    """本段里「用户明确标成创业者 + 座位号」的进出场候选，`{座位号: 候选}`。

    与 `entryexit_fill_plan` 的区别只有一条，但很要紧：**不看该座位是否已经
    有人**。fill_plan 里那条「不覆盖已占座位」是对的——自动递补不该顶掉就座
    结果；而补格位是另一回事：座位上的人没变，只是他那张 2×2 被人工勾掉了
    一格，需要**同一个人**的另一张画面顶上（用户 2026-08-25：「如果在自动标
    的图片中选择了排除某个窗格，增加通过进出场的照片对该窗格图片进行补充，
    以最大化填充 2×2」）。

    只收精确等于「创业者」的：工作人员、专家、上/下一组创业者一概不算。拿他
    们的画面去补格位，等于把别人的衣着编进这位创业者的行里，且不报错——本项目
    正是因为进出场格里反复混进场地工作人员，才在 Task 26 一度禁掉进出场帧。
    这次重新放开的**唯一**依据是「这个候选是用户亲手指名给这个座位的」。

    候选层的 role/seat 与格位分配里的那几格都算数：多人框按格位切分之后，
    每一格才是一个人，只看候选层会把这一整类漏掉。

    **同一座位的每一条标注都留下**，值是一串 `[(候选, 格位表), ...]`，按候选
    顺序（即时间顺序）排。此前这里是 `setdefault`，只留最先遇到的那一条——
    用户 2026-08-26 实测文化4组 order 3：座位 2 在 exit@3364 标了两格、又在
    exit@3366 标了一格，登记表里只有前面那两张；座位 3 在 entry@2502 整条标
    了他、又在 exit@3366 标了一格，登记表里只剩一张。「单帧可能只选了一个窗
    格，而在多帧分别选的窗格没有拼在一起，应该根据标注尽可能全的拼成 2×2
    四张图」。丢掉的那几张是**用户亲手标的**，丢了没有任何一处会报错。

    格位表为空表示「整条候选都是他」（候选层直接标的那种）。
    """
    if not labels:
        return {}
    recs = labels.get(str(segment.order))
    if not isinstance(recs, dict):
        return {}
    out: dict[int, list[tuple[dict, list[int]]]] = {}
    for cand in candidates_for_labels(dets, segment):
        lab = recs.get(cand["stable_id"])
        if not isinstance(lab, dict):
            continue
        cell_map, _report, _src = cell_map_of_label(lab)
        if cell_map:
            # **按座位号归拢格位**：一个人可能占多格（「格位3 = 同格位1」，
            # 解引用之后两格同属一个座位）。此前这里只存整条候选，补帧时按
            # `best_dets` 的顺序往上补——而多人框的 best_dets 里混着**别人的
            # 格**，补进去就是把另一个人的衣着编进这位创业者的行里，且一处
            # 不报错。必须精确到格。
            per_seat: dict[int, list[int]] = {}
            for cell, info in cell_map.items():
                if info.get("role") == "创业者" and info.get("seat"):
                    per_seat.setdefault(int(info["seat"]), []).append(int(cell))
            for seat, cells in per_seat.items():
                out.setdefault(seat, []).append((cand, sorted(cells)))
            continue
        if lab.get("role") == "创业者" and lab.get("seat"):
            # 候选层直接标的：整条候选都是他，用空列表表示。
            out.setdefault(int(lab["seat"]), []).append((cand, []))

    # --- 相邻段标过来的（「上/下一组创业者」）也算本段的补帧来源 ----------
    #
    # 用户 2026-09-09 实测城市1：第 16 段 exit track 99 的格 2 标了「下一组
    # 创业者座位 1」，而第 17 段座位 1 的证据图只有 2 张、四格空着两格，那一格
    # 却没补进去——日志里只有一句「该座位已有人，不覆盖」。
    #
    # `entryexit_fill_plan` 里那条「不覆盖已占座位」对**顶掉**而言是对的（自动
    # 递补不该改写人工可见的座位），但补格位是另一回事：座位上的人没变，只是
    # 他那张 2×2 没填满，需要同一个人的另一张画面顶上。本段内的多条标注早就
    # 是这么合并的（Task 56T），跨段那一路一直没接上。
    #
    # **排在本段自己的标注之后**：同一个人优先用本段窗口里的画面，相邻段那张
    # 只在还没填满时才补。重复的帧由 `topup_pairs` 按检测框去重，跨段标注已经
    # 把座位填上时不会再补一遍。
    for nb in (neighbours or []):
        if int(nb.order) == int(segment.order):
            continue
        nb_recs = labels.get(str(nb.order))
        if not isinstance(nb_recs, dict):
            continue
        want_delta = int(segment.order) - int(nb.order)
        for cand in candidates_for_labels(dets, nb):
            lab = nb_recs.get(cand["stable_id"])
            if not isinstance(lab, dict):
                continue
            cell_map, _rep, _src = cell_map_of_label(lab)
            if cell_map:
                per_seat: dict[int, list[int]] = {}
                for cell, info in cell_map.items():
                    role = info.get("role")
                    if (_NEIGHBOUR_ROLE_DELTA.get(role) == want_delta
                            and info.get("seat")):
                        per_seat.setdefault(int(info["seat"]), []).append(int(cell))
                for seat, cells in per_seat.items():
                    out.setdefault(seat, []).append((cand, sorted(cells)))
                continue
            if (_NEIGHBOUR_ROLE_DELTA.get(lab.get("role")) == want_delta
                    and lab.get("seat")):
                out.setdefault(int(lab["seat"]), []).append((cand, []))
    return out


def spec_pairs(spec) -> list:
    """一条补帧候选摊平成 `[(格位号, 帧), ...]`，格位号是它在**自己那张**证据
    图上的编号。

    `spec` 是 `labelled_seat_candidates` 里的 `(候选, 格位表)`；格位表为空表示
    整条候选都是这个人，此时位置 i 就是格位 i（与 `cell_dets` 的口径一致：
    格位 k ↔ `best_dets[k-1]`）。为了兼容早先只传候选本身的调用，裸 dict 也收。

    格位号必须跟着帧一起走：登记表的 `cell_indices` 与 `frames` 是逐项一一对应
    的（Task 56A），补进来的帧若没有格位号，两者就此错位——「第几格」会指到另
    一张帧上，且一处不报错。
    """
    if spec is None:
        return []
    cand, cells = (spec if isinstance(spec, tuple) else (spec, []))
    if cand is None:
        return []
    if cells:
        # 逐格一律走 `cell_dets`，不在这里另算一次 `best_dets[k-1]`——两处各切
        # 一次必然日后走样，而走样的表现是衣着编码给了错的人、不报错。
        return [(k, d) for k in sorted({int(c) for c in cells})
                for d in cell_dets(cand, k)]
    return list(enumerate(cand.get("best_dets") or [], start=1))


def _spec_list(specs) -> list:
    """`None` / 单条 / 一串，一律归一成 list。

    两个调用方历来各传各的（registry 的就座那一路传单条，格位那一路传一串），
    在这里收口一次，好过每处各写一份 isinstance 判断——写两份必然日后只改一份。
    """
    if specs is None:
        return []
    return list(specs) if isinstance(specs, list) else [specs]


def picture_key(d) -> tuple:
    """一帧**画面本身**的身份：视频里的时刻 + 框。不含路径、段号、kind。

    同一时刻的同一个框就是同一张画面，不管它被存进了哪个窗口的目录。用户
    2026-09-26 电子10 第 13 段座位 2：13 段退场窗口与 14 段进场窗口重叠（前者
    exitTime 恰等于后者 entryTime），同一批帧存了两份；他在两个窗口里都标了
    这个人，按 Detection 整体去重把两份当成不同的图，2×2 成了两张图各出现两次。
    """
    return (round(float(d.t), 3),
            round(float(d.x1), 1), round(float(d.y1), 1),
            round(float(d.x2), 1), round(float(d.y2), 1))


def _most_different_next(queue: list, taken: list, have: list):
    """从一个来源里取下一格：第一格按该来源自己的次序（质量最好的那张），
    之后每一格挑**时间上离本来源已选各格最远**的那张。

    同一个来源里按质量排第二的那张往往只隔 0.1 秒——等于同一张图放两遍
    （实测 7 组：单一来源的四帧首尾中位只隔 0.5 秒）。已经被别的来源用掉的
    帧跳过：同一张图白占一格，还会让模型以为「这个角度我看了两遍」。
    """
    left = [(i, pr) for i, pr in enumerate(queue) if picture_key(pr[1]) not in have]
    if not left:
        return None
    if not taken:
        i, pr = left[0]
    else:
        ts = [float(d.t) for _c, d in taken]
        # 距离相同按原次序（质量）取，保证结果确定
        i, pr = max(left, key=lambda ipr: (min(abs(float(ipr[1][1].t) - t) for t in ts),
                                          -ipr[0]))
    queue.pop(i)
    return pr


def topup_pairs(pairs: list, specs, want: int = CELL_COUNT) -> list:
    """把 `[(格位号, 帧), ...]` 与该座位**其余**人工标定的候选合成一张 2×2。

    **只有一个来源时原样返回、一张不动**——没有别的画面可合并，宁可四张相近
    也不留空格（本项目的优先级是「识别准确度 > 样本保留 > 分辨率」）。

    **有多个来源时按来源轮流出格**，`pairs` 这一路排在最前：两个来源 2+2、
    三个 2+1+1、四个及以上各 1。同一个来源出第二格时，挑时间上离它已选那格
    最远的一张（`_most_different_next`）。

    用户 2026-09-26：「如果有多个标注，尽量不同站姿或差异较大的图片各保留一个
    格位；自动识别出的 2×2 格位，可能图片是高度雷同的，看情况只保留 1-2 个，
    尽量与其他不同画面的各位进行合并」。实测 7 组：99 个座位有多个来源，**30 个
    的 2×2 只用上了第一个**——它四帧占满，用户另外标的画面一格都没进去；而单一
    来源的四帧首尾中位只隔 0.5 秒，121 个里 99 个在 2 秒以内。

    此前 `pairs` 这一路是「原样排在前面、一对不动」，理由是「排除格 N」按位置
    数。逐一核过：排除格只在两处生效（`registry` 与 `assemble` 的就座分支），都
    作用在 `select_best_frames(就座帧, 4)` 上、**早于**本函数；核对页上没有任何
    逐格控件。所以让 `pairs` 参与轮流不会让用户勾过的任何一格改指别人。

    格位号始终跟着它自己那一帧走（成对取、成对放），`cell_indices` 与
    `frames` 因此仍逐项对应。补不满不报错——有几张算几张。
    """
    queues = [q for q in (spec_pairs(spec) for spec in _spec_list(specs)) if q]
    if not queues:
        return list(pairs)
    sources = [list(pairs)] + queues
    taken: list[list] = [[] for _ in sources]
    out: list = []
    have: list = []
    while len(out) < want:
        progressed = False
        for i, q in enumerate(sources):
            if len(out) >= want:
                break
            pr = _most_different_next(q, taken[i], have)
            if pr is None:
                continue
            progressed = True
            out.append(pr)
            have.append(picture_key(pr[1]))
            taken[i].append(pr)
        if not progressed:
            break
    # 选完之后**按来源排回一起**：同一来源的几格挨着放，`pairs` 那一路仍在
    # 最前。一行一个来源比交错着看更好认（上排就座、下排退场），而且
    # 「第一个来源排在最前」对它的每一格都成立。模型那边不按位置读构成——
    # `describe_layout` 按最终那几张各自的 kind 数（见 assemble）。
    order = {id(pr): i for i, grp in enumerate(taken) for pr in grp}
    return sorted(out, key=lambda pr: order[id(pr)])


def topup_frames(frames: list, spec, want: int = CELL_COUNT) -> list:
    """把 `frames` 用该座位人工标定的候选补到 `want` 张，原有的不动、排在前面。

    `spec` 可以是单条 `(候选, 格位表)`，也可以是 `labelled_seat_candidates`
    给出的一整串——同一个人常常被标在**好几条**候选上（用户 2026-08-26 实测
    文化4组 order 3），只认单条就会把其余几张静默丢掉。

    `spec` 为 None（该座位没有人工标定的进出场候选）时原样返回：宁可少几张，
    也不拿一个不知道是谁的人去填。切帧与去重全在 `topup_pairs` 里做，本函数
    只是丢掉格位号的那一层——两处各算一遍必然日后走样。
    """
    return [d for _c, d in topup_pairs([(None, f) for f in frames], spec, want=want)]


def apply_cell_exclusions(dets: list, excluded: set[int]) -> list:
    """按格位号剔除对应的检测帧。格位号 k ↔ `dets[k-1]`（见 `cell_dets`）。

    超出实际张数的排除号一律忽略：证据图不足 4 张时本就有灰块占位，
    为不存在的格报错没有意义。
    """
    if not excluded:
        return list(dets)
    return [d for i, d in enumerate(dets, start=1) if i not in excluded]
