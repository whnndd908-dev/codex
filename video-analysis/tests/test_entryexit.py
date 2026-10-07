import json
import re
import pytest
from pathlib import Path

from pipeline import config
from pipeline import paths as _pipeline_paths
from pipeline.detect import Detection
from pipeline.schedule import Segment
from pipeline import entryexit
from pipeline.entryexit import (
    entryexit_candidates, entryexit_candidate_totals,
    load_entry_labels, entryexit_fill_plan,
)

W = 1440
H = W * 0.75  # 1080，与 seats.py 一贯的 4:3 换算口径一致
RIGHT_CX = 0.55 * W  # RIGHT_REGION_X 截断线 = 792.0

STANDING_H = 0.55 * H   # 早期方案里的“站立”高度——现在只当作普通高度用


def mkseg(order=2, presenter_num=2):
    return Segment(order=order, company="测试企业", start=0, end=1000,
                    presenter_num=presenter_num, presenters=())


def det(tid, cx, h, kind="entry", order=2, t=0.0, frame_path=None, bottom=1000.0, width=30.0):
    """构造一个人形检测框，底边固定在 `bottom` 附近。默认宽 30px——本文件
    绝大多数用例只关心高度/位置/合并逻辑，不是长宽比本身，默认宽度选得
    足够窄，使得连本文件里出现的最矮高度（0.05*H=54px）算出的长宽比也
    舒服地高于 config.ENTRYEXIT_MIN_ASPECT=1.30（54/30=1.8），不会被
    Task 41 新增的长宽比过滤误伤——长宽比专属测试会显式传入 width 覆盖
    这个默认值。"""
    if frame_path is None:
        frame_path = Path("nonexistent_frame.jpg")
    half = width / 2.0
    return Detection(order, kind, t, frame_path, tid,
                      cx - half, bottom - h, cx + half, bottom, 0.9)


def track(tid, cx, h, n_frames, kind="entry", order=2, start_t=0.0, bottom=1000.0, width=30.0):
    return [det(tid, cx, h, kind=kind, order=order, t=start_t + i, bottom=bottom, width=width)
            for i in range(n_frames)]


# ---------------------------------------------------------------------------
# 排序不设阈值——两版框高阈值判据都被证伪：已确认的就座创业者（90 人）框高
# 中位 0.45（p10=0.23 p90=0.65 max=0.70），进出场右区 track（756 条）框高
# 中位 0.40（p75=0.61 p90=0.73 max=0.76）——两个分布完全重叠，0.05 的矮个子
# 与 0.70 的高个子在这个机位下都可能是任意角色，没有可信阈值能把两者分开。
# entryexit_candidates 因此只排序、不阈值过滤：全部候选都保留，只按框高
# 降序排，展示上限见下面 ENTRYEXIT_TOP_N 的测试。
# ---------------------------------------------------------------------------

def test_candidates_are_ranked_by_height_not_thresholded():
    seg = mkseg()
    # 5 条互不合并的轨迹（cx 间隔远超合并容差），框高横跨 0.16~0.70——包含
    # 一个远低于两版旧阈值（0.30/0.40）的矮个子，它必须仍然出现在结果里。
    # 下限从 0.05 提到 0.16：Task 52 加了一条**绝对下限**
    # （ENTRYEXIT_MIN_HEIGHT_RATIO=0.15，取值依据见 config.py），0.05 的框
    # 现在会被那条挡下并进折叠区。这里要守的仍是原来那件事——**高度不是
    # 角色判据**：0.16 远低于两版被证伪的阈值（0.30/0.40），它必须照常入选。
    heights = [0.16 * H, 0.20 * H, 0.40 * H, 0.60 * H, 0.70 * H]
    dets = []
    for i, h in enumerate(heights):
        dets += track(100 + i, RIGHT_CX + 100 + i * 300, h, 6, order=seg.order)

    out = entryexit_candidates(dets, seg, frame_w=W)
    assert len(out) == 5, f"5 条互不阈值过滤的候选应全部保留，实得 {len(out)} 个"
    assert [round(c["median_h"], 1) for c in out] == sorted(
        (round(h, 1) for h in heights), reverse=True
    ), "候选必须按框高降序排列"
    assert any(c["track_id"] == 100 for c in out), (
        "框高只有画面 16% 的候选（远低于两版被证伪的旧阈值 0.30/0.40）也必须"
        "出现在结果里——站立/就座的框高分布完全重叠，高度不是角色判据；"
        f"绝对下限只到 {config.ENTRYEXIT_MIN_HEIGHT_RATIO}"
    )


# ---------------------------------------------------------------------------
# 每段每个窗口最多展示 config.ENTRYEXIT_TOP_N 个（展示上限，不是判据），
# 被截断的真实总数由 entryexit_candidate_totals 给出。
# ---------------------------------------------------------------------------

def test_truncation_totals_match_and_top_n_is_enforced(monkeypatch):
    monkeypatch.setattr(config, "ENTRYEXIT_TOP_N", 3)
    seg = mkseg()
    # 5 条，超过 TOP_N=3；最矮的 0.16 仍高于绝对下限
    # ENTRYEXIT_MIN_HEIGHT_RATIO（0.15），本条测的是截断不是过滤。
    heights = [0.16 * H, 0.20 * H, 0.30 * H, 0.40 * H, 0.50 * H]
    dets = []
    for i, h in enumerate(heights):
        dets += track(200 + i, RIGHT_CX + 100 + i * 300, h, 6, order=seg.order)

    out = entryexit_candidates(dets, seg, frame_w=W)
    totals = entryexit_candidate_totals(dets, seg, frame_w=W)

    assert len(out) == 3, "超过 ENTRYEXIT_TOP_N 时只应展示前 N 个"
    assert totals["entry"] == 5, "截断前的真实候选总数必须如实给出，不能只报展示出来的那几个"
    # 展示的应是框高最高的前 3 个（0.50/0.40/0.30），矮的两个（0.10/0.20）被截断。
    shown_ids = {c["track_id"] for c in out}
    assert shown_ids == {204, 203, 202}


# ---------------------------------------------------------------------------
# 右区过滤——必须复用 exclude_judges 的结果，不能另写一份口径不同的判据
# ---------------------------------------------------------------------------

def test_entryexit_candidates_excludes_left_region(monkeypatch):
    seg = mkseg()

    calls = []
    orig = entryexit.exclude_judges

    def spy(dets_in, frame_w):
        calls.append((len(dets_in), frame_w))
        return orig(dets_in, frame_w)

    monkeypatch.setattr(entryexit, "exclude_judges", spy)

    # cx 恰好落在阈值两侧：792.0 是 exclude_judges 的边界（d.cx > cut 严格
    # 大于），792.0 本身应被排除；792.1 应被保留——用这一对样本锁住
    # entryexit_candidates 用的和 exclude_judges 完全一致的边界语义。
    dets = (track(10, RIGHT_CX, STANDING_H, 6, order=seg.order)         # 恰在边界，应被排除
            + track(11, RIGHT_CX + 0.1, STANDING_H, 6, order=seg.order)  # 边界右侧，应保留
            + track(12, 100.0, STANDING_H, 6, order=seg.order))          # 明显左区（评委席）

    out = entryexit_candidates(dets, seg, frame_w=W)
    ids = {c["track_id"] for c in out}

    assert calls, "entryexit_candidates 必须调用 exclude_judges，不能另写一份右区逻辑"
    assert 10 not in ids, "cx 恰在 exclude_judges 边界上的轨迹应被排除（与 exclude_judges 的 > 语义一致）"
    assert 11 in ids, "cx 刚过边界的轨迹应保留"
    assert 12 not in ids, "明显左区（评委席）的轨迹不应入选"


# ---------------------------------------------------------------------------
# 单帧高个子必须保留——没有帧数门槛
# ---------------------------------------------------------------------------

def test_single_frame_tall_person_is_kept():
    seg = mkseg()
    dets = track(30, RIGHT_CX + 200, 0.74 * H, 1, order=seg.order)  # 只 1 帧
    out = entryexit_candidates(dets, seg, frame_w=W)
    ids = {c["track_id"] for c in out}
    assert 30 in ids, "只出现 1 帧的候选不应被任何帧数门槛砍掉"
    cand = next(c for c in out if c["track_id"] == 30)
    assert cand["n_frames"] == 1


# ---------------------------------------------------------------------------
# 走动中被拆成多条单帧 track 的同一个人，必须先合并再计数
# （复用 seats.merge_tracks_by_position，不另写一份合并逻辑）。
# ---------------------------------------------------------------------------

def test_fragments_of_one_walking_person_are_merged():
    seg = mkseg()
    cx = RIGHT_CX + 200
    # 3 条单帧碎片：同一位置（cx/y2 差在 SEAT_MERGE_CX_TOL/Y2_TOL 容差内），
    # 时间互不重叠（t=0/1/2），模拟走动中跟踪器每帧都换一次 track_id。
    dets = [
        det(101, cx, STANDING_H, order=seg.order, t=0.0),
        det(102, cx + 5, STANDING_H, order=seg.order, t=1.0),
        det(103, cx - 5, STANDING_H, order=seg.order, t=2.0),
    ]
    out = entryexit_candidates(dets, seg, frame_w=W)
    assert len(out) == 1, f"3 条碎片应合并成 1 个候选，实得 {len(out)} 个"
    cand = out[0]
    assert cand["n_frames"] == 3, "合并后帧数应是 3 条碎片的并集"
    assert cand["track_id"] == 101, "应以最先出现（发起该人）的原始 track_id 作代表"


# ---------------------------------------------------------------------------
# load_entry_labels 缺文件
# ---------------------------------------------------------------------------

def test_load_entry_labels_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "entry_labels_不存在.json"
    assert not missing.exists()
    assert load_entry_labels(missing) == {}


def test_load_entry_labels_reads_existing_file(tmp_path):
    """Task 46 起文件带 `_meta`，键是稳定标识；未提供当前检测指纹时跳过
    校验、原样返回（并打印一行说明，不静默）。"""
    p = tmp_path / "entry_labels_test.json"
    p.write_text(
        '{"_meta": {"detections_fingerprint": "abc123", "sheet": "x", '
        '"n_candidates": 1}, '
        '"2": {"entry@100#900,1000": {"role": "创业者", "seat": 1}}}',
        encoding="utf-8")
    labels = load_entry_labels(p)
    assert labels["2"]["entry@100#900,1000"]["role"] == "创业者"
    assert "_meta" not in labels


# ---------------------------------------------------------------------------
# fill_plan 只补未占用的座位
# ---------------------------------------------------------------------------

def test_fill_plan_only_fills_unoccupied_seats():
    seg = mkseg(order=3, presenter_num=2)
    # 唯一一个候选，用户指定的座位号 1 已经被就座阶段占用。
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, kind="entry", order=seg.order)
    labels = {"3": {"entry:5": {"role": "创业者", "seat": 1}}}
    seats_by_order = {3: {99: 1}}  # 座位 1 已被就座轨迹 tid=99 占用

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    entries = plan.get(3, [])
    assert not any(e["seat"] == 1 for e in entries), "已被就座阶段占用的座位号不得被覆盖"


# ---------------------------------------------------------------------------
# fill_plan 未指定座位号时的处理（Task 45 追加改动四 · 缺陷二）
#
# 默认 config.BACKFILL_REQUIRE_SEAT=True：未指定座位号的候选不参与补位，
# 改为收进 needs_seat——按 cx 猜座位号是几何推断，本项目已连续失败四次，
# 实测 order 13 爱尚游正是猜错座位号导致性别与人工标注的"女女女"对不上。
# 旧的"按 cx 猜"逻辑仍保留在 BACKFILL_REQUIRE_SEAT=False 分支，用单独的
# 测试锁住，不许被删掉。
# ---------------------------------------------------------------------------

def test_candidate_without_seat_goes_to_needs_seat():
    seg = mkseg(order=4, presenter_num=3)
    dets = (track(7, RIGHT_CX + 400, STANDING_H, 6, kind="entry", order=seg.order)
            + track(8, RIGHT_CX + 100, STANDING_H, 6, kind="entry", order=seg.order))
    labels = {
        "4": {
            "entry:7": {"role": "创业者", "seat": None},
            "entry:8": {"role": "创业者", "seat": None},
        }
    }
    seats_by_order = {4: {}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    assert plan.get(4, []) == [], (
        "未指定座位号的候选默认不得进入补位计划（BACKFILL_REQUIRE_SEAT=True）"
    )
    assert {n["track_id"] for n in needs_seat} == {7, 8}
    for n in needs_seat:
        assert n["order"] == 4 and n["kind"] == "entry"


def test_fill_plan_still_guesses_seat_when_require_seat_disabled(monkeypatch):
    """`BACKFILL_REQUIRE_SEAT=False` 时退回旧逻辑：按 median_cx 从右到左
    依次填入剩余最小座位号，标 seat_uncertain=1——这条代码路径保留，不是
    删掉，仅默认不再是它。"""
    monkeypatch.setattr(config, "BACKFILL_REQUIRE_SEAT", False)
    seg = mkseg(order=4, presenter_num=3)
    dets = (track(7, RIGHT_CX + 400, STANDING_H, 6, kind="entry", order=seg.order)   # 更靠右
            + track(8, RIGHT_CX + 100, STANDING_H, 6, kind="entry", order=seg.order))  # 更靠左
    labels = {
        "4": {
            "entry:7": {"role": "创业者", "seat": None},
            "entry:8": {"role": "创业者", "seat": None},
        }
    }
    seats_by_order = {4: {}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    assert needs_seat == [], "关闭前置条件时不应产出 needs_seat"
    entries = plan[4]
    assert len(entries) == 2
    for e in entries:
        assert e["seat_uncertain"] == 1, "用户没指定座位号时必须标 seat_uncertain=1"

    by_tid = {e["track_id"]: e for e in entries}
    assert by_tid[7]["seat"] == 1, "更靠右（cx更大）的候选应优先拿到剩余最小的座位号"
    assert by_tid[8]["seat"] == 2


# ---------------------------------------------------------------------------
# fill_plan 忽略非创业者标注
# ---------------------------------------------------------------------------

def test_fill_plan_ignores_non_presenter_labels():
    seg = mkseg(order=6, presenter_num=4)
    # Task 41 改动二：扩展覆盖新增的「上一组创业者」「下一组创业者」——
    # 与「工作人员」一样，一律不应参与补位。
    dets = (track(9, RIGHT_CX + 200, STANDING_H, 6, kind="entry", order=seg.order)
            + track(15, RIGHT_CX + 500, STANDING_H, 6, kind="entry", order=seg.order)
            + track(16, RIGHT_CX + 800, STANDING_H, 6, kind="entry", order=seg.order))
    labels = {"6": {
        "entry:9": {"role": "工作人员", "seat": 1},
        "entry:15": {"role": "上一组创业者", "seat": 2},
        "entry:16": {"role": "下一组创业者", "seat": 3},
    }}
    seats_by_order = {6: {}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    assert plan.get(6, []) == [], (
        "标为工作人员/上一组创业者/下一组创业者的候选都不应被补进座位表——"
        "只有精确等于「创业者」才参与补位"
    )
    assert needs_resplit == []


def test_fill_plan_ignores_expert_and_uncertain_labels():
    seg = mkseg(order=7, presenter_num=2)
    dets = (track(1, RIGHT_CX + 200, STANDING_H, 6, kind="entry", order=seg.order)
            + track(2, RIGHT_CX + 500, STANDING_H, 6, kind="entry", order=seg.order))
    labels = {"7": {
        "entry:1": {"role": "专家", "seat": None},
        "entry:2": {"role": "不确定", "seat": None},
    }}
    seats_by_order = {7: {}}
    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    assert plan.get(7, []) == []


# ---------------------------------------------------------------------------
# Task 41 改动一：长宽比过滤局部检测（只拍到一只手/胳膊）——高度阈值不
# 加回来（已被证伪，见模块顶部大段说明），但长宽比 h/w 可以区分：
#     已确认创业者（90 人）        中位 2.31  p10 1.43  min 1.17
#     进出场小框 <150px（98 个）    中位 1.26  p90 1.87  max 3.45
# ---------------------------------------------------------------------------

def test_low_aspect_box_is_not_a_candidate():
    seg = mkseg()
    # 正方形框：h/w = 300/300 = 1.0，明显低于 config.ENTRYEXIT_MIN_ASPECT
    # (1.30)——人是竖长的，手偏方，这种形状应判为局部检测，不进候选池。
    # 高度取 300px（>0.15*1080=162）使它**只**被长宽比挡下——原来的 100px
    # 在 Task 52 之后两条规则都会挡，长宽比这一条改坏了也测不出来。
    dets = track(50, RIGHT_CX + 200, 300.0, 6, order=seg.order, width=300.0)
    out = entryexit_candidates(dets, seg, frame_w=W)
    assert 50 not in {c["track_id"] for c in out}, (
        "长宽比过低（近似方形，疑似只拍到一只手）的框应被挡在候选池外"
    )


def test_tall_narrow_box_survives_even_if_short():
    seg = mkseg()
    # 高只有 200px（远低于历史上任何一版高度阈值 0.30*1080=324/0.40*1080=432，
    # 但高于 Task 52 的绝对下限 0.15*1080=162），h/w = 200/140 ≈ 1.43，稳稳
    # 高于 ENTRYEXIT_MIN_ASPECT=1.30、低于「阈值提高到 2.0」这种变异会挡掉
    # 的区间——矮不等于不是人，角色判据只有长宽比，被证伪的高度判据不能
    # 借着 Task 52 的绝对下限悄悄回来。
    dets = track(51, RIGHT_CX + 200, 200.0, 6, order=seg.order, width=140.0)
    out = entryexit_candidates(dets, seg, frame_w=W)
    assert 51 in {c["track_id"] for c in out}, (
        "长宽比达标（哪怕仅比 1.30 高出一点）时，矮框仍应入选——高度不是判据"
    )


def test_filtered_by_aspect_candidates_are_retrievable():
    """被长宽比挡掉的候选不静默丢弃——entryexit_filtered_candidates 必须
    能原样取到它（页面折叠区块的数据来源，见 test_entryreview.py 里对应
    的 HTML 级用例）。"""
    seg = mkseg()
    dets = (track(60, RIGHT_CX + 200, STANDING_H, 6, order=seg.order)              # 正常候选
            + track(61, RIGHT_CX + 500, 300.0, 6, order=seg.order, width=300.0))   # 方框，应被挡
    # 61 高 300px（>0.15*1080）——只被长宽比挡下，不与 Task 52 的高度下限混淆

    kept = entryexit_candidates(dets, seg, frame_w=W)
    filtered = entryexit.entryexit_filtered_candidates(dets, seg, frame_w=W)

    assert {c["track_id"] for c in kept} == {60}
    assert {c["track_id"] for c in filtered} == {61}, (
        "被长宽比挡掉的候选必须能通过 entryexit_filtered_candidates 原样取到"
    )
    assert filtered[0]["aspect"] < config.ENTRYEXIT_MIN_ASPECT


# ---------------------------------------------------------------------------
# Task 41 改动二：六个角色里新增的「上一组创业者」「下一组创业者」——与
# 工作人员/专家/不确定一样，一律不参与补位。
# ---------------------------------------------------------------------------

def test_new_roles_are_not_filled_in():
    seg = mkseg(order=8, presenter_num=2)
    dets = (track(20, RIGHT_CX + 200, STANDING_H, 6, kind="entry", order=seg.order)
            + track(21, RIGHT_CX + 500, STANDING_H, 6, kind="entry", order=seg.order))
    labels = {"8": {
        "entry:20": {"role": "上一组创业者", "seat": 1},
        "entry:21": {"role": "下一组创业者", "seat": 2},
    }}
    seats_by_order = {8: {}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    assert plan.get(8, []) == [], "标为上一组/下一组创业者的候选不应被补进座位表"
    assert needs_resplit == []


# ---------------------------------------------------------------------------
# Task 41 改动三：一个框里混进多人时（n_people>1），不自动补位，转入
# needs_resplit；未标注 n_people（旧格式/用户没碰过下拉）时按 1 处理，
# 正常补位——两条断言在同一个测试里对照，避免默认值方向反了（变异 e）
# 或多人候选照常补位（变异 f）时不被发现。
# ---------------------------------------------------------------------------

def test_multi_person_candidate_goes_to_needs_resplit():
    seg = mkseg(order=9, presenter_num=3)
    dets = (track(30, RIGHT_CX + 200, STANDING_H, 6, kind="entry", order=seg.order)    # n_people=2
            + track(31, RIGHT_CX + 500, STANDING_H, 6, kind="entry", order=seg.order))  # 未标 n_people
    labels = {"9": {
        "entry:30": {"role": "创业者", "seat": None, "n_people": 2},
        # 座位号显式指定（Task 45 起未指定座位号的候选默认转入
        # needs_seat，不参与补位——这个测试关心的是 n_people 默认值，
        # 不是座位号猜测，须指定座位号才能继续测到 n_people 这条判据）。
        "entry:31": {"role": "创业者", "seat": 1},
    }}
    seats_by_order = {9: {}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    entries = plan.get(9, [])

    assert all(e["track_id"] != 30 for e in entries), (
        "n_people>1 的候选不得出现在补位计划里——绝不能拿混了两个人的框去编码衣着"
    )
    assert any(e["track_id"] == 31 for e in entries), (
        "未标 n_people 的候选必须按默认值 1 处理、正常补位——默认值不能是"
        "「未标注=可疑」，否则会把绝大多数正常候选错误地挡在补位计划之外"
    )

    # Task 49：needs_resplit 多带一个 note 字段（按格位号写的「谁是谁」）
    assert {"order": 9, "kind": "entry", "track_id": 30, "n_people": 2,
            "note": ""} in needs_resplit, (
        "n_people>1 的候选必须出现在 needs_resplit 里，交由后续重新切分"
    )
    assert all(item["track_id"] != 31 for item in needs_resplit), (
        "未标 n_people（默认 1）的候选不应出现在 needs_resplit 里"
    )


# ---------------------------------------------------------------------------
# Task 45 追加改动四（控制器用真实标注实测出的两处缺陷，不是用户标错）：
#   缺陷一：同一人在 entry/exit 两个窗口各标一次同一座位号，须合并为
#           一条，不能当成两个人（复刻 order 9 华韵传媒的实测缺陷：
#           entry:19/exit:10 都标了座位2，未合并时第二个滑到其它空位，
#           顶掉了本该在那的人）。
#   缺陷二：座位号未指定时不得靠猜（见上方 test_candidate_without_seat_
#           goes_to_needs_seat / test_fill_plan_still_guesses_seat_when_
#           require_seat_disabled）。
# ---------------------------------------------------------------------------

def test_same_seat_candidates_are_merged_as_one_person():
    seg = mkseg(order=9, presenter_num=3)
    dets = (track(19, RIGHT_CX + 200, STANDING_H, 4, kind="entry", order=seg.order)
            + track(10, RIGHT_CX + 300, STANDING_H, 8, kind="exit", order=seg.order))
    labels = {"9": {
        "entry:19": {"role": "创业者", "seat": 2},
        "exit:10": {"role": "创业者", "seat": 2},
    }}
    seats_by_order = {9: {}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    entries = plan.get(9, [])
    seat2_entries = [e for e in entries if e["seat"] == 2]
    assert len(seat2_entries) == 1, (
        f"指向同一座位号的 entry/exit 候选必须合并为一条，实得 {seat2_entries}"
    )
    # 护栏：合并后不应意外占用第二个座位号（例如座位3）——那正是缺陷一
    # 的症状：没合并时第二个候选会滑到其它空位，顶掉本该在那的人。
    assert len(entries) == 1, f"两个候选应合并成恰好一条，实得 {entries}"


def test_merge_prefers_exit_over_entry():
    seg = mkseg(order=9, presenter_num=3)
    # exit 候选帧数更少也应胜出——kind 优先级压过证据量判据（用户领域
    # 知识：退场多正面、进场多背影，正面更利于衣着/性别判定）。
    dets = (track(19, RIGHT_CX + 200, STANDING_H, 20, kind="entry", order=seg.order)
            + track(10, RIGHT_CX + 300, STANDING_H, 2, kind="exit", order=seg.order))
    labels = {"9": {
        "entry:19": {"role": "创业者", "seat": 2},
        "exit:10": {"role": "创业者", "seat": 2},
    }}
    seats_by_order = {9: {}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    merged = next(e for e in plan.get(9, []) if e["seat"] == 2)
    assert merged["kind"] == "exit" and merged["track_id"] == 10, (
        f"合并时应优先取 exit（正面更利于判定），实得 {merged}"
    )


def test_unspecified_seat_candidates_are_not_merged(monkeypatch):
    """座位号为 None 的多个候选不应互相合并——它们可能真是不同的人。用
    BACKFILL_REQUIRE_SEAT=False（旧的"按 cx 猜"路径，仍保留）验证：两个
    seat=None 候选应各自独立占用一个座位号，而不是被合并成一条。"""
    monkeypatch.setattr(config, "BACKFILL_REQUIRE_SEAT", False)
    seg = mkseg(order=9, presenter_num=3)
    dets = (track(19, RIGHT_CX + 200, STANDING_H, 4, kind="entry", order=seg.order)
            + track(10, RIGHT_CX + 300, STANDING_H, 8, kind="exit", order=seg.order))
    labels = {"9": {
        "entry:19": {"role": "创业者", "seat": None},
        "exit:10": {"role": "创业者", "seat": None},
    }}
    seats_by_order = {9: {}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    entries = plan.get(9, [])
    assert len(entries) == 2, (
        f"座位号为 None 的两个候选应各自独立补位（可能是不同的人），"
        f"不应被合并成一条，实得 {entries}"
    )
    assert {e["track_id"] for e in entries} == {19, 10}


def test_fill_plan_returns_three_tuple():
    """签名变更后所有调用点仍正确——空输入下三个返回值都应是空容器。"""
    seg = mkseg(order=1, presenter_num=1)
    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], [], {}, {})
    assert plan == {} and needs_resplit == [] and needs_seat == []


# ---------------------------------------------------------------------------
# 追加改动三（用户 2026-08-17）：把浏览器下载目录里的 entry_labels_*.json
# 收进 work 目录——浏览器不允许网页写任意路径，这是主要保障（另一半
# File System Access API 优先路径在 entry_review.html 里，见 run_all.py
# 阶段⑤生成挑选页部分的测试）。测试一律用 tmp_path 造假的下载目录，不
# 真的读用户的 ~/Downloads。
# ---------------------------------------------------------------------------

def _touch(path: Path, text: str, mtime: float) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_adopt_copies_newer_downloaded_file(tmp_path):
    import time
    from pipeline.entryexit import adopt_downloaded_labels

    work = tmp_path / "work"
    work.mkdir()
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    dl = downloads / "entry_labels_测试组.json"
    _touch(dl, '{"2": {}}', time.time())

    result = adopt_downloaded_labels(work, "测试组", downloads=downloads)
    assert result == work / "entry_labels_测试组.json"
    assert result.read_text(encoding="utf-8") == dl.read_text(encoding="utf-8")


def test_adopt_does_not_move_original(tmp_path):
    import time
    from pipeline.entryexit import adopt_downloaded_labels

    work = tmp_path / "work"
    work.mkdir()
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    dl = downloads / "entry_labels_测试组.json"
    _touch(dl, '{"2": {}}', time.time())

    adopt_downloaded_labels(work, "测试组", downloads=downloads)
    assert dl.exists(), "复制而非移动：原文件必须仍在下载目录"


def test_adopt_skips_when_work_copy_is_newer(tmp_path):
    import time
    from pipeline.entryexit import adopt_downloaded_labels

    work = tmp_path / "work"
    work.mkdir()
    downloads = tmp_path / "downloads"
    downloads.mkdir()

    now = time.time()
    dl = downloads / "entry_labels_测试组.json"
    _touch(dl, '{"tag": "old_download"}', now - 100)

    existing = work / "entry_labels_测试组.json"
    _touch(existing, '{"tag": "newer_work_copy"}', now)

    result = adopt_downloaded_labels(work, "测试组", downloads=downloads)
    assert result is None
    assert existing.read_text(encoding="utf-8") == '{"tag": "newer_work_copy"}', (
        "work 里更新的副本不能被下载目录里更旧的文件覆盖"
    )


def test_adopt_picks_newest_among_browser_suffixed_variants(tmp_path):
    import time
    from pipeline.entryexit import adopt_downloaded_labels

    work = tmp_path / "work"
    work.mkdir()
    downloads = tmp_path / "downloads"
    downloads.mkdir()

    now = time.time()
    base = downloads / "entry_labels_测试组.json"
    v1 = downloads / "entry_labels_测试组 (1).json"     # 带空格变体
    v2 = downloads / "entry_labels_测试组(2).json"       # 不带空格变体
    _touch(base, '{"tag": "base"}', now)
    _touch(v1, '{"tag": "v1"}', now + 10)
    _touch(v2, '{"tag": "v2"}', now + 20)

    result = adopt_downloaded_labels(work, "测试组", downloads=downloads)
    assert result is not None
    assert result.read_text(encoding="utf-8") == '{"tag": "v2"}', (
        "应取三个变体里 mtime 最新的那个（这里是 v2），两种加后缀的形态都要识别"
    )


def test_adopt_missing_downloads_dir_returns_none(tmp_path):
    from pipeline.entryexit import adopt_downloaded_labels

    work = tmp_path / "work"
    work.mkdir()
    missing = tmp_path / "does_not_exist"

    result = adopt_downloaded_labels(work, "测试组", downloads=missing)
    assert result is None


# ===========================================================================
# Task 46：标注的稳定标识（不依赖 track_id）+ 失效校验
#
# 事故经过：Book1 文化4组从 13 家变 14 家，新增那家排在第 8 位，其后企业
# 行号后移一位，帧却是按旧编号抽的——seg 8-13 的衣着/颜色/性别全部取自
# 隔壁公司，整条流水线照常跑完、没有一处报错。修复要重抽帧 + 重检测，重
# 检测后 track_id 全部改变（跟踪器状态，不是稳定标识），用户此前标的
# `entry:19`/`exit:4` 大部分指向不存在的 track（递补 5 人→0 人），少数
# 碰巧编号还在、但那已经是另一个人——错配不报错。用户要标 60 多组。
# ===========================================================================

import json as _json

from pipeline.entryexit import (
    detections_fingerprint, match_label_to_candidate, parse_candidate_id,
    stable_candidate_id,
)


def _cand(kind="entry", t_mid=100.0, cx=900.0, y2=1000.0, track_id=1):
    """一个只带匹配所需字段的候选（与 entryexit_candidates 产出的键名一致）。"""
    return {
        "kind": kind, "t_mid": t_mid, "median_cx": cx, "median_y2": y2,
        "track_id": track_id,
        "stable_id": stable_candidate_id(kind, t_mid, cx, y2),
    }


def test_stable_id_survives_position_jitter():
    """重检测只会让框位置有像素级抖动、时间有零点几秒的漂移——这类抖动
    必须仍然匹配得上，否则用户标 60 多组的成果会因为一次重检测全部作废。"""
    cand = _cand(t_mid=100.0, cx=900.0, y2=1000.0)
    old_id = stable_candidate_id("entry", 101.0, 930.0, 1030.0)  # 位置抖 30px、时间抖 1 秒
    labels = {old_id: {"role": "创业者", "seat": 2}}

    got = match_label_to_candidate(labels, cand)
    assert got is not None, "位置抖动 30px、时间抖动 1 秒时必须仍能匹配上"
    assert got["seat"] == 2
    assert labels == {}, "匹配上的标注必须移出候选池，防止被第二个候选再抢一次"


def test_stable_id_rejects_different_person():
    """位置差 300px 是另一个人（本项目座位间距 200px 以上）——绝不能匹配。
    错配会把别人的衣着编码成这个人的，而且不报错。"""
    cand = _cand(t_mid=100.0, cx=900.0, y2=1000.0)
    old_id = stable_candidate_id("entry", 100.0, 1200.0, 1000.0)  # 同一时刻，右移 300px
    labels = {old_id: {"role": "创业者", "seat": 2}}

    assert match_label_to_candidate(labels, cand) is None, (
        "相距 300px 的是另一个人，不得匹配"
    )
    assert old_id in labels, "没匹配上的标注必须原样留在池子里"


def test_stable_id_rejects_far_in_time():
    """位置巧合地相同、但时间差远超容差（进场窗口跨 40 秒，前后两个人
    完全可能站在同一位置）——不得匹配。"""
    cand = _cand(t_mid=100.0, cx=900.0, y2=1000.0)
    old_id = stable_candidate_id("entry", 130.0, 900.0, 1000.0)
    labels = {old_id: {"role": "创业者", "seat": 2}}
    assert match_label_to_candidate(labels, cand) is None


def test_stable_id_rejects_other_kind():
    """进场候选的标注不得匹配到退场候选身上——两个窗口的时间/位置完全
    可能撞上（同一个人往往在两个窗口各出现一次）。"""
    cand = _cand(kind="exit", t_mid=100.0, cx=900.0, y2=1000.0)
    old_id = stable_candidate_id("entry", 100.0, 900.0, 1000.0)
    labels = {old_id: {"role": "创业者", "seat": 2}}
    assert match_label_to_candidate(labels, cand) is None


def test_old_format_key_is_never_fuzzy_matched():
    """旧格式 `entry:19` 不含任何视频内容信息，无从判断是不是同一个人——
    绝不能"尽量匹配"，那正是会错配的情形（少数编号碰巧还在、但已是另一
    个人）。整份作废由 load_entry_labels 负责，这里只保证不匹配。"""
    cand = _cand(track_id=19)
    labels = {"entry:19": {"role": "创业者", "seat": 2}}
    assert match_label_to_candidate(labels, cand) is None
    assert parse_candidate_id("entry:19") is None
    assert parse_candidate_id(stable_candidate_id("entry", 1.0, 2.0, 3.0)) is not None


def test_one_label_matches_at_most_one_candidate():
    """两个相近候选不得抢同一条标注——否则一条「创业者·座位2」会同时补
    给两个人，第二个还会滑到别的空位，顶掉本该在那的人（Task 45 缺陷一
    的同类错误）。"""
    labels = {stable_candidate_id("entry", 100.0, 910.0, 1000.0):
              {"role": "创业者", "seat": 1}}
    c1 = _cand(t_mid=100.0, cx=900.0, y2=1000.0, track_id=1)
    c2 = _cand(t_mid=100.0, cx=920.0, y2=1000.0, track_id=2)

    got1 = match_label_to_candidate(labels, c1)
    got2 = match_label_to_candidate(labels, c2)
    assert (got1 is None) != (got2 is None), (
        f"两个都在容差内的候选里恰好一个能拿到这条标注，实得 {got1!r} / {got2!r}"
    )
    assert labels == {}


def test_nearest_candidate_wins():
    """容差内有多条旧标注时，最近者胜出（不是字典里的第一条）。"""
    near = stable_candidate_id("entry", 100.0, 905.0, 1000.0)
    far = stable_candidate_id("entry", 100.0, 960.0, 1000.0)
    labels = {far: {"role": "工作人员", "seat": None},
              near: {"role": "创业者", "seat": 3}}
    got = match_label_to_candidate(labels, _cand(t_mid=100.0, cx=900.0, y2=1000.0))
    assert got is not None and got["seat"] == 3, "应取容差内最近的那条标注"
    assert far in labels and near not in labels


def test_candidates_carry_stable_ids():
    """entryexit_candidates 产出的每个候选都要带 stable_id 与匹配所需的
    t_mid/median_y2——否则页面导出与回填两侧各算一份必然走样。"""
    seg = mkseg(order=2, presenter_num=2)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, kind="entry", order=seg.order,
                 start_t=100.0)
    cands = entryexit_candidates(dets, seg, frame_w=W)
    assert len(cands) == 1
    c = cands[0]
    assert c["stable_id"] == stable_candidate_id(
        c["kind"], c["t_mid"], c["median_cx"], c["median_y2"])
    assert "@" in c["stable_id"] and "#" in c["stable_id"]
    assert str(c["track_id"]) not in c["stable_id"].split("@")[0], (
        "稳定 id 不得由 track_id 导出——那是跟踪器状态，重检测即全部改变"
    )


# ---------------------------------------------------------------------------
# 检测结果指纹
# ---------------------------------------------------------------------------

def test_fingerprint_is_stable_for_same_detections(tmp_path):
    """同一份检测重复读取指纹必须相同（含 mtime 被改动的情形——用
    (size, mtime) 做指纹会在这里给出假阳性，把好端端的标注错误作废）；
    检测内容一变，指纹必须变。"""
    import os

    p = tmp_path / "detections.json"
    p.write_text('{"dets": [1, 2, 3]}', encoding="utf-8")
    fp1 = detections_fingerprint(p)
    assert fp1 == detections_fingerprint(p), "同一份检测重复读取指纹必须相同"

    os.utime(p, (p.stat().st_atime + 10_000, p.stat().st_mtime + 10_000))
    assert detections_fingerprint(p) == fp1, (
        "仅 mtime 变化（重新落盘同样的内容）不得改变指纹"
    )

    p.write_text('{"dets": [1, 2, 4]}', encoding="utf-8")
    assert detections_fingerprint(p) != fp1, "检测内容一变，指纹必须变"


def test_fingerprint_changes_after_redetection(tmp_path):
    """走真实的 save_detections：重检测（track_id 全变）后指纹必然改变。"""
    from pipeline.detect import save_detections

    p = tmp_path / "detections.json"
    dets_v1 = track(5, RIGHT_CX + 200, STANDING_H, 6, kind="entry", order=2)
    save_detections(dets_v1, p)
    fp1 = detections_fingerprint(p)

    dets_v2 = track(1005, RIGHT_CX + 203, STANDING_H, 6, kind="entry", order=2)
    save_detections(dets_v2, p)
    assert detections_fingerprint(p) != fp1


# ---------------------------------------------------------------------------
# 失效校验：指纹不符时的分级作废
# ---------------------------------------------------------------------------

def _labelled_scene(order=2, n=4, presenter_num=4, start_cx=None, spacing=250.0):
    """造 n 个互不合并的进场候选，返回 (segment, dets, candidates)。"""
    seg = mkseg(order=order, presenter_num=presenter_num)
    if start_cx is None:
        start_cx = RIGHT_CX + 60
    dets = []
    for i in range(n):
        dets += track(10 + i, start_cx + i * spacing, STANDING_H, 6,
                      kind="entry", order=order, start_t=100.0 + i)
    cands = entryexit_candidates(dets, seg, frame_w=W)
    assert len(cands) == n, f"夹具本身要造出 {n} 个候选，实得 {len(cands)}"
    return seg, dets, cands


def _write_labels(path, body, fingerprint, sheet="测试组", n_candidates=None):
    payload = {"_meta": {"detections_fingerprint": fingerprint, "sheet": sheet,
                         "n_candidates": n_candidates if n_candidates is not None
                         else sum(len(v) for v in body.values())}}
    payload.update(body)
    path.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_fingerprint_match_loads_normally(tmp_path, capsys):
    seg, dets, cands = _labelled_scene()
    body = {"2": {c["stable_id"]: {"role": "创业者", "seat": i + 1,
                                   "n_people": 1, "note": "", "track_id": c["track_id"]}
                  for i, c in enumerate(cands)}}
    p = tmp_path / "entry_labels_x.json"
    _write_labels(p, body, "FINGERPRINT-A")

    out = load_entry_labels(p, detections_fp="FINGERPRINT-A",
                            candidates_by_order={2: cands})
    assert set(out["2"]) == {c["stable_id"] for c in cands}
    assert "_meta" not in out, "_meta 是元信息，不得混进标注本体"
    assert "作废" not in capsys.readouterr().out


def test_fingerprint_mismatch_with_old_format_voids_all(tmp_path, capsys):
    """旧格式（无 _meta，键是 `entry:19`）整份作废并提示重标——绝不尝试
    匹配：少数编号碰巧还在、但那已经是另一个人，错配不报错。"""
    seg, dets, cands = _labelled_scene()
    p = tmp_path / "entry_labels_x.json"
    p.write_text(_json.dumps(
        {"2": {f"entry:{c['track_id']}": {"role": "创业者", "seat": 1}
               for c in cands}}, ensure_ascii=False), encoding="utf-8")

    out = load_entry_labels(p, detections_fp="FINGERPRINT-NEW",
                            candidates_by_order={2: cands})
    assert out == {}, "旧格式必须整份作废"
    msg = capsys.readouterr().out
    assert "作废" in msg and "重新标注" in msg, f"作废必须给出醒目提示，实得：\n{msg}"


def test_fingerprint_mismatch_with_stable_ids_attempts_match(tmp_path, capsys):
    """稳定 id 格式时不整份作废：先模糊匹配，如实报告匹配数与丢弃数。"""
    seg, dets, cands = _labelled_scene(n=4)
    # 旧标注的 id 由「重检测前」的位置导出：位置抖 20px、时间抖 0.5 秒。
    body = {"2": {}}
    for i, c in enumerate(cands):
        old_id = stable_candidate_id(c["kind"], c["t_mid"] + 0.5,
                                     c["median_cx"] + 20, c["median_y2"] - 20)
        body["2"][old_id] = {"role": "创业者", "seat": i + 1, "n_people": 1,
                             "note": "", "track_id": 999 + i}
    p = tmp_path / "entry_labels_x.json"
    _write_labels(p, body, "FINGERPRINT-OLD")

    out = load_entry_labels(p, detections_fp="FINGERPRINT-NEW",
                            candidates_by_order={2: cands})
    assert set(out["2"]) == {c["stable_id"] for c in cands}, (
        "匹配上的标注必须改挂到当前候选的稳定 id 下"
    )
    seats = {k: v["seat"] for k, v in out["2"].items()}
    for i, c in enumerate(cands):
        assert seats[c["stable_id"]] == i + 1, "座位号必须跟着人走，不得错位"
    msg = capsys.readouterr().out
    assert "4" in msg and "匹配" in msg, f"必须报告匹配率，实得：\n{msg}"
    assert "作废" not in msg


def test_partial_mismatch_reports_matched_and_dropped(tmp_path, capsys):
    """8 条标注里 6 条能对上、2 条对不上：保留 6 条、丢弃 2 条，并如实
    把这两个数字打出来（匹配率 0.75 ≥ 0.7，不整份作废）。"""
    seg, dets, cands = _labelled_scene(n=8, presenter_num=8)
    body = {"2": {}}
    for i, c in enumerate(cands[:6]):
        old_id = stable_candidate_id(c["kind"], c["t_mid"], c["median_cx"] + 10,
                                     c["median_y2"])
        body["2"][old_id] = {"role": "创业者", "seat": i + 1, "n_people": 1,
                             "note": "", "track_id": 900 + i}
    for j in range(2):
        stray = stable_candidate_id("entry", 100.0 + j, RIGHT_CX - 5000, 1000.0)
        body["2"][stray] = {"role": "创业者", "seat": None, "n_people": 1,
                            "note": "", "track_id": 800 + j}
    p = tmp_path / "entry_labels_x.json"
    _write_labels(p, body, "FINGERPRINT-OLD")

    out = load_entry_labels(p, detections_fp="FINGERPRINT-NEW",
                            candidates_by_order={2: cands})
    assert len(out["2"]) == 6
    msg = capsys.readouterr().out
    assert "6" in msg and "2" in msg and "丢弃" in msg, (
        f"必须如实给出匹配数与丢弃数，实得：\n{msg}"
    )


def test_low_match_rate_voids_all(tmp_path, capsys):
    """匹配率低于 config.LABEL_MATCH_MIN_RATE 时整份作废——宁可全废也不
    要错配（本项目刚因同一类问题损失了一整组数据）。"""
    seg, dets, cands = _labelled_scene(n=4)
    body = {"2": {}}
    # 只有 1/5 能对上，匹配率 0.2 < 0.7。
    old_id = stable_candidate_id(cands[0]["kind"], cands[0]["t_mid"],
                                 cands[0]["median_cx"] + 10, cands[0]["median_y2"])
    body["2"][old_id] = {"role": "创业者", "seat": 1, "n_people": 1, "note": ""}
    for j in range(4):
        stray = stable_candidate_id("entry", 500.0 + j * 10, 900.0, 1000.0)
        body["2"][stray] = {"role": "创业者", "seat": None, "n_people": 1, "note": ""}
    p = tmp_path / "entry_labels_x.json"
    _write_labels(p, body, "FINGERPRINT-OLD")

    out = load_entry_labels(p, detections_fp="FINGERPRINT-NEW",
                            candidates_by_order={2: cands})
    assert out == {}, "匹配率 0.2 低于阈值 0.7，必须整份作废"
    msg = capsys.readouterr().out
    assert "作废" in msg and "重新标注" in msg


def test_void_is_reported_not_silent(tmp_path, capsys):
    """三条作废路径（旧格式 / 指纹不符且无候选可比对 / 匹配率过低）
    一条都不许静默返回空——静默作废会让用户以为"这组本来就没标过"。"""
    seg, dets, cands = _labelled_scene(n=4)
    p = tmp_path / "entry_labels_x.json"

    scenarios = []
    # 1) 旧格式
    p.write_text(_json.dumps({"2": {"entry:10": {"role": "创业者", "seat": 1}}}),
                 encoding="utf-8")
    scenarios.append(("旧格式", dict(detections_fp="NEW", candidates_by_order={2: cands})))
    for name, kwargs in list(scenarios):
        capsys.readouterr()
        assert load_entry_labels(p, **kwargs) == {}
        assert capsys.readouterr().out.strip(), f"{name} 作废时必须打印，不得静默"

    # 2) 指纹不符且没有候选可供比对
    body = {"2": {stable_candidate_id("entry", 100.0, 900.0, 1000.0):
                  {"role": "创业者", "seat": 1}}}
    _write_labels(p, body, "OLDFP")
    capsys.readouterr()
    assert load_entry_labels(p, detections_fp="NEWFP", candidates_by_order=None) == {}
    assert capsys.readouterr().out.strip(), "无候选可比对时作废必须打印"

    # 3) 匹配率过低
    body = {"2": {stable_candidate_id("entry", 900.0 + i, 100.0, 100.0):
                  {"role": "创业者", "seat": 1} for i in range(5)}}
    _write_labels(p, body, "OLDFP")
    capsys.readouterr()
    assert load_entry_labels(p, detections_fp="NEWFP",
                             candidates_by_order={2: cands}) == {}
    assert capsys.readouterr().out.strip(), "匹配率过低时作废必须打印"


def test_matching_labels_survive_redetection(tmp_path, capsys):
    """**本任务的核心价值**：模拟一次真实的重检测——track_id 全部改变、
    框位置微抖、时间微移——用户此前的标注仍能对上，递补计划照旧把正确的
    座位号落到「新」的 track 上。"""
    from pipeline.detect import save_detections

    seg = mkseg(order=2, presenter_num=3)

    def build(track_base, dcx, dt):
        d = []
        for i in range(3):
            d += track(track_base + i, RIGHT_CX + 60 + i * 250 + dcx, STANDING_H, 6,
                        kind="entry", order=seg.order, start_t=100.0 + i + dt)
        return d

    dets_v1 = build(10, 0.0, 0.0)
    cands_v1 = entryexit_candidates(dets_v1, seg, frame_w=W)
    det_path = tmp_path / "detections.json"
    save_detections(dets_v1, det_path)
    fp_v1 = detections_fingerprint(det_path)

    labels_path = tmp_path / "entry_labels_x.json"
    body = {"2": {c["stable_id"]: {"role": "创业者", "seat": i + 1, "n_people": 1,
                                   "note": "", "track_id": c["track_id"]}
                  for i, c in enumerate(cands_v1)}}
    _write_labels(labels_path, body, fp_v1)

    # --- 重检测：track_id 从 10.. 变成 5010..，位置抖 12px，时间移 0.6 秒 ---
    dets_v2 = build(5010, 12.0, 0.6)
    save_detections(dets_v2, det_path)
    fp_v2 = detections_fingerprint(det_path)
    assert fp_v2 != fp_v1, "重检测后指纹必须改变"
    cands_v2 = entryexit_candidates(dets_v2, seg, frame_w=W)
    assert {c["track_id"] for c in cands_v2} & {c["track_id"] for c in cands_v1} == set(), (
        "夹具要真的模拟出「track_id 全部改变」，否则这条测试测不到东西"
    )

    labels = load_entry_labels(labels_path, detections_fp=fp_v2,
                               candidates_by_order={seg.order: cands_v2})
    assert labels, f"重检测后标注不该丢失，实得：{labels!r}\n{capsys.readouterr().out}"

    plan, needs_resplit, needs_seat = entryexit_fill_plan(
        [seg], dets_v2, labels, {seg.order: {}})
    got = {e["seat"]: e["track_id"] for e in plan[seg.order]}
    want = {i + 1: c["track_id"] for i, c in enumerate(cands_v2)}
    assert got == want, (
        f"重检测后每个座位号应落到「新」检测里同一个人身上，期望 {want}，实得 {got}"
    )
    assert needs_seat == []


def test_fill_plan_accepts_stable_id_keys():
    """fill_plan 直接吃稳定 id 键（导出格式），不需要先经过 load_entry_labels。"""
    seg = mkseg(order=3, presenter_num=2)
    dets = track(5, RIGHT_CX + 200, STANDING_H, 6, kind="entry", order=seg.order,
                 start_t=50.0)
    cands = entryexit_candidates(dets, seg, frame_w=W)
    labels = {"3": {cands[0]["stable_id"]: {"role": "创业者", "seat": 2}}}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {3: {}})
    assert plan[3] == [{"track_id": 5, "kind": "entry", "seat": 2, "seat_uncertain": 0}]


def test_fill_plan_does_not_reuse_one_label_for_two_candidates():
    """fill_plan 里「精确命中」那一轮也必须把标注消耗掉：否则候选甲精确
    命中的那条标注还留在池子里，紧挨着的候选乙再靠模糊匹配把它抢一遍，
    一条标注补出两个人（先精确后模糊的两轮结构掩盖了这个缺陷，单测
    match_label_to_candidate 测不到）。"""
    seg = mkseg(order=8, presenter_num=2)
    # 两条轨迹相距 20px、帧集合完全重叠（同一批 t）——不会被
    # merge_tracks_by_position 合并成一个人，却都落在位置容差之内。
    dets = (track(41, RIGHT_CX + 300, STANDING_H, 6, kind="entry", order=seg.order)
            + track(42, RIGHT_CX + 320, STANDING_H, 6, kind="entry", order=seg.order))
    cands = entryexit_candidates(dets, seg, frame_w=W)
    assert len(cands) == 2, f"夹具要造出两个相邻候选，实得 {len(cands)}"

    labels = {"8": {cands[0]["stable_id"]:
                    {"role": "创业者", "seat": None, "n_people": 1}}}
    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, {8: {}})
    assert len(needs_seat) == 1, (
        f"一条标注只能被一个候选拿走，实得 {needs_seat}"
    )


# ===========================================================================
# Task 49：三种新的人工输入 —— 格位备注、座位否决/替代、位置排除
#
# 用户 2026-08-23 三条需求：
#   ① 2×2 证据图给每格编号，用户在备注里按格位号写清谁是谁（本任务只采集，
#      「按备注自动切分」另做）；
#   ② 已定座位可能判错：允许人工否决，或指定一个更清楚的候选替换它；
#   ③ 「这个组里这个位置一直是工作人员/专家」——算法认不出「同一个人」
#      （见 entryexit_fill_plan docstring 里的实测记录），但用户的观察
#      可以作为输入。
# ===========================================================================

def test_needs_resplit_carries_note():
    """n_people>1 的候选转入 needs_resplit 时必须带上备注原文。

    备注是用户按格位号写下的「1 号格是创业者、2 号格是工作人员」——重新
    切分唯一的依据。丢了它，needs_resplit 只剩「这个框有 2 个人」，谁是
    谁无从得知，用户白标一遍。

    Task 51 起，能解析出格位的备注会被**当场切分**、不再进 needs_resplit，
    所以这里改用一条解析不出来的备注（相对指代「左边那个」，解析器一律
    不猜）——needs_resplit 现在专收「切不开的那些」，带上备注原文的价值
    反而更大：人要照着这句话把它改写成明确的格位号。
    """
    seg = mkseg(order=9, presenter_num=3)
    dets = track(30, RIGHT_CX + 200, STANDING_H, 6, kind="entry", order=seg.order)
    note = "左边那个是创业者，右边那个是工作人员"
    labels = {"9": {"entry:30": {"role": "创业者", "seat": None,
                                  "n_people": 2, "note": note}}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, {9: {}})
    assert len(needs_resplit) == 1, f"应恰有一条待切分，实得 {needs_resplit}"
    assert needs_resplit[0]["note"] == note, (
        f"needs_resplit 必须带上备注原文（重新切分的唯一依据），实得 {needs_resplit[0]}"
    )


# --- 需求二：人工否决已定座位 / 指定替代画面 -------------------------------

def test_rejected_seat_becomes_vacant():
    """被标 rejected 的座位视为空缺，可被递补——否则「我人工核实时告诉你
    这个座位判错了」这句话在下游没有任何效果。"""
    seg = mkseg(order=8, presenter_num=2)
    dets = track(41, RIGHT_CX + 300, STANDING_H, 6, kind="exit", order=seg.order)
    cands = entryexit_candidates(dets, seg, frame_w=W)
    seg_labels = {cands[0]["stable_id"]: {"role": "创业者", "seat": 1, "n_people": 1}}
    seats_by_order = {8: {77: 1}}      # 就座阶段把座位 1 判给了 track 77

    # 对照：没有否决时，座位 1 已被占用，fill_plan 一律不覆盖
    plan_before, _nr, _ns = entryexit_fill_plan(
        [seg], dets, {"8": seg_labels}, seats_by_order)
    assert plan_before.get(8, []) == [], (
        f"夹具前提：未否决时座位 1 是占用的、不该被补，实得 {plan_before}"
    )

    labels = {"8": seg_labels,
              "seat_overrides": {"8": {"1": {"rejected": True,
                                              "replacement_stable_id": None,
                                              "note": "这个座位判错了"}}}}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, seats_by_order)
    assert [(e["seat"], e["track_id"]) for e in plan.get(8, [])] == [(1, 41)], (
        f"被否决的座位应视为空缺、可被递补，实得 {plan}"
    )


def test_replacement_stable_id_fills_that_seat():
    """「改用我指定的候选」——直接用指定候选补那个座位，不必再标角色/座位号。"""
    seg = mkseg(order=8, presenter_num=2)
    dets = (track(41, RIGHT_CX + 200, STANDING_H, 6, kind="exit", order=seg.order)
            + track(42, RIGHT_CX + 600, STANDING_H, 6, kind="exit", order=seg.order))
    cands = entryexit_candidates(dets, seg, frame_w=W)
    pick = next(c for c in cands if c["track_id"] == 42)

    labels = {"seat_overrides": {"8": {"1": {
        "rejected": True, "replacement_stable_id": pick["stable_id"],
        "note": "就座那张糊了，用退场这张"}}}}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {8: {77: 1}})
    assert plan.get(8) == [{"track_id": 42, "kind": "exit", "seat": 1,
                            "seat_uncertain": 0}], (
        f"指定的候选必须补进被否决的那个座位，实得 {plan}"
    )


def _redetection_pair(seg, n=4, kind="exit"):
    """返回 build(base_tid, dx, dt) —— 造同一批人的两版检测（重检测后
    track_id 全变、位置抖 dx 像素、时间移 dt 秒）。"""
    def build(base_tid, dx, dt):
        d = []
        for i in range(n):
            d += track(base_tid + i, RIGHT_CX + 100 + i * 150 + dx, STANDING_H, 6,
                       kind=kind, order=seg.order, start_t=10.0 + i * 5 + dt)
        return d
    return build


def _write_labels_with_overrides(path, fingerprint, body, overrides):
    payload = {"_meta": {"detections_fingerprint": fingerprint, "sheet": "测试组",
                          "n_candidates": sum(len(v) for v in body.values())}}
    payload.update(body)
    payload["seat_overrides"] = overrides
    path.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_unmatched_replacement_is_reported_not_dropped(tmp_path, capsys):
    """`replacement_stable_id` 匹配不上时不得静默丢弃：要计入失配数、打印
    出来，并退回「仅否决、不替换」（座位仍算空缺，但不拿一个对不上的候选
    去补）。静默丢弃 = 用户明确指定的画面没被采用且他永远不会知道。"""
    from pipeline.detect import save_detections

    seg = mkseg(order=2, presenter_num=4)
    build = _redetection_pair(seg)
    dets_v1 = build(10, 0.0, 0.0)
    cands_v1 = entryexit_candidates(dets_v1, seg, frame_w=W)
    assert len(cands_v1) == 4

    det_path = tmp_path / "detections.json"
    save_detections(dets_v1, det_path)
    fp_v1 = detections_fingerprint(det_path)

    labels_path = tmp_path / "entry_labels_x.json"
    ghost = "exit@9999#100,100"        # 当前候选里根本没有这个位置/时刻
    _write_labels_with_overrides(
        labels_path, fp_v1,
        {"2": {c["stable_id"]: {"role": "创业者", "seat": i + 1, "n_people": 1,
                                 "note": "", "track_id": c["track_id"]}
               for i, c in enumerate(cands_v1)}},
        {"2": {"1": {"rejected": True, "replacement_stable_id": ghost,
                     "note": "画面不够清楚"}}})

    dets_v2 = build(5010, 12.0, 0.6)   # 重检测：编号全变、位置抖 12px
    save_detections(dets_v2, det_path)
    fp_v2 = detections_fingerprint(det_path)
    cands_v2 = entryexit_candidates(dets_v2, seg, frame_w=W)

    labels = load_entry_labels(labels_path, detections_fp=fp_v2,
                               candidates_by_order={seg.order: cands_v2})
    out = capsys.readouterr().out

    assert labels, f"4/5 的匹配率不该整份作废：{out}"
    assert ghost in out, f"对不上的 replacement_stable_id 必须打印出来：{out}"
    ovr = labels["seat_overrides"]["2"]["1"]
    assert ovr["rejected"] is True, "否决本身要保留（座位仍算空缺）"
    assert ovr["replacement_stable_id"] is None, (
        f"对不上的替换必须清掉，绝不能拿它去补座位，实得 {ovr}"
    )
    assert "5 条标注中 4 条匹配上" in out, (
        f"对不上的替换必须计入失配数（4 条候选标注 + 1 条替换 = 5），实得：{out}"
    )


def test_seat_overrides_survive_fingerprint_mismatch_via_fuzzy_match(tmp_path, capsys):
    """座位否决要与候选标注走**同一套**稳定标识校验：重检测后
    `replacement_stable_id` 按视频内容（时间+位置）重新对齐到当前候选。"""
    from pipeline.detect import save_detections

    seg = mkseg(order=2, presenter_num=4)
    build = _redetection_pair(seg)
    dets_v1 = build(10, 0.0, 0.0)
    cands_v1 = entryexit_candidates(dets_v1, seg, frame_w=W)
    # 座位 1/2/4 由候选标注补，座位 3 由「改用我指定的候选」指向第 3 个
    # 候选（它自己没有候选标注，避免一个人同时被指派到两个座位）
    old_sid = cands_v1[2]["stable_id"]

    det_path = tmp_path / "detections.json"
    save_detections(dets_v1, det_path)
    fp_v1 = detections_fingerprint(det_path)

    labels_path = tmp_path / "entry_labels_x.json"
    _write_labels_with_overrides(
        labels_path, fp_v1,
        {"2": {cands_v1[i]["stable_id"]: {"role": "创业者", "seat": seat,
                                           "n_people": 1, "note": "",
                                           "track_id": cands_v1[i]["track_id"]}
               for i, seat in ((0, 1), (1, 2), (3, 4))}},
        {"2": {"3": {"rejected": True, "replacement_stable_id": old_sid,
                     "note": "用这张"}}})

    dets_v2 = build(5010, 12.0, 0.6)
    save_detections(dets_v2, det_path)
    fp_v2 = detections_fingerprint(det_path)
    cands_v2 = entryexit_candidates(dets_v2, seg, frame_w=W)
    new_sid = cands_v2[2]["stable_id"]
    assert new_sid != old_sid, "夹具要真的让稳定标识发生抖动，否则测不到重新对齐"

    labels = load_entry_labels(labels_path, detections_fp=fp_v2,
                               candidates_by_order={seg.order: cands_v2})
    ovr = labels["seat_overrides"]["2"]["3"]
    assert ovr["replacement_stable_id"] == new_sid, (
        f"重检测后替换标识应重新对齐到当前候选 {new_sid}，实得 {ovr}"
    )

    # 重新对齐之后必须真的能用：拿它去 fill_plan，座位 3 应由那个人补上
    plan, _nr, _ns = entryexit_fill_plan([seg], dets_v2, labels, {2: {77: 3}})
    got = [e for e in plan.get(2, []) if e["seat"] == 3]
    assert len(got) == 1 and got[0]["track_id"] == cands_v2[2]["track_id"], (
        f"重新对齐后的替换应补进座位 3，实得 {plan.get(2)}"
    )


# --- 需求三：按位置排除 —— Task 53 已按用户要求整块删除 ---------------------
#
# 用户 2026-08-23 原话：「这个选项实际没用，删掉吧」。页面控件、
# `position_exclusions` 的读取与应用一并删除，原来的三条测试
# （drops_candidates_within_tolerance / keeps_candidates_outside_tolerance /
# count_is_printed）随功能一起删。**替代它们的是两条「删干净了」的测试**：
#   · `test_legacy_position_exclusions_ignored_not_crash`（旧文件里的残留
#     字段静默忽略并打印，不报错、不影响其余标注）
#   · `test_position_exclusions_are_not_applied_even_if_hand_written`
#     （绕过 load_entry_labels 手写这个键同样不生效）
#   · 页面侧：`tests/test_entryreview.py::test_position_exclusion_控件_removed`
# 只删不补的话，「删掉」这件事本身没有任何测试守着，日后有人把它加回来
# 不会有任何一处变红。


# ===========================================================================
# Task 51 · 要做的一：按格位备注切分
#
# Task 49 只采集了「格位编号 + 备注」，切分明确留给后续。用户真实标注里
# 已经攒了三条按格位写的备注（`work/文化4-4.28/entry_labels_*.json`），
# 全部卡在 needs_resplit 里没人处理：
#     「1、4是创业者3，另两个是混入人员」
#     「1号是创业者2」
#     「1/3是创业者3，2/4是创业者2」
# 只差切分这一步，order 1 就能从「已定 2 人」补齐到 3 人。
#
# **相对指代一律不猜**：「另两个」「其余」「剩下的」需要知道「另」是相对
# 谁而言，猜错就把工作人员当成创业者编码进衣着表，且不报错——本项目一贯
# 的最坏失败模式。未指明的格位留空，未解析的短语如实报告。
# ===========================================================================

def test_parse_slash_form():
    """「1/3是创业者3，2/4是创业者2」——用户真实写法（order 9 华韵传媒）。"""
    got = entryexit.parse_cell_note("1/3是创业者3，2/4是创业者2")
    assert set(got) == {1, 2, 3, 4}, f"四个格位都该解析出来，实得 {got}"
    assert [(got[c]["role"], got[c]["seat"]) for c in (1, 3)] == [
        ("创业者", 3), ("创业者", 3)], f"1/3 应指向创业者 3 号位，实得 {got}"
    assert [(got[c]["role"], got[c]["seat"]) for c in (2, 4)] == [
        ("创业者", 2), ("创业者", 2)], f"2/4 应指向创业者 2 号位，实得 {got}"


def test_parse_dun_and_hao_form():
    """「1、4是创业者3」「1号是创业者2」——用户真实写法（order 1 律持兴观）。"""
    got = entryexit.parse_cell_note("1、4是创业者3")
    assert set(got) == {1, 4}, f"顿号分隔的两个格位都该解析出来，实得 {got}"
    assert got[1] == got[4], "同一条短语里的格位指向同一个人"
    assert (got[1]["role"], got[1]["seat"]) == ("创业者", 3), got

    got2 = entryexit.parse_cell_note("1号是创业者2")
    assert set(got2) == {1}, f"「1号」是一个格位，实得 {got2}"
    assert (got2[1]["role"], got2[1]["seat"]) == ("创业者", 2), got2

    got3 = entryexit.parse_cell_note("第2格是工作人员")
    assert set(got3) == {2}, f"「第N格」写法要支持，实得 {got3}"
    assert got3[2]["role"] == "工作人员", got3
    assert got3[2]["seat"] is None, "工作人员没有座位号"


def test_relative_reference_is_not_guessed():
    """**本组最重要的一条**：「另两个」「其余」「剩下的」这类相对指代一律
    不解析，未指明的格位留空，且必须出现在 unparsed_phrases 里。

    猜错的代价：把工作人员当成创业者编码进衣着表，整条流水线照常跑完、
    没有任何一处报错——本项目一贯的最坏失败模式。
    """
    note = "1、4是创业者3，另两个是混入人员"
    rep = entryexit.parse_cell_note_report(note)
    assert set(rep["parsed"]) == {1, 4}, (
        f"只有 1、4 是明确指明的格位；2、3 必须留空，实得 {rep['parsed']}"
    )
    assert any("另两个" in p for p in rep["unparsed_phrases"]), (
        f"「另两个是混入人员」必须如实列进 unparsed_phrases，实得 {rep}"
    )

    for phrase in ("其余是工作人员", "剩下的是工作人员", "剩下两个是专家",
                   "另外一个是专家", "左边那个是创业者2"):
        r = entryexit.parse_cell_note_report(phrase)
        assert r["parsed"] == {}, (
            f"相对指代「{phrase}」不该解析出任何格位，实得 {r['parsed']}"
        )
        assert r["unparsed_phrases"] == [phrase], (
            f"「{phrase}」必须如实列进 unparsed_phrases，实得 {r}"
        )


def test_parse_returns_empty_on_garbage():
    """解析不出来时返回空 dict、不抛异常——备注是自由文本，用户写什么都
    可能，一个 ValueError 会把整条流水线打断。"""
    for note in ("", "   ", "看不清", "都是工作人员", "不知道", None):
        got = entryexit.parse_cell_note(note)
        assert got == {}, f"{note!r} 不该解析出格位，实得 {got}"

    assert entryexit.parse_cell_note_report("")["unparsed_phrases"] == [], (
        "空备注没有任何短语，不该凭空报告一条未解析"
    )
    assert entryexit.parse_cell_note_report("看不清")["unparsed_phrases"] == ["看不清"]


def _cell_number_of_input_index(tmp_path):
    """实测「`build_evidence_sheet` 输入的第 i 张图，在图上被标成几号」。

    全程不读注释、不读 `_cell_origin` 的返回值：用四张**不同颜色**的纯色
    图渲染一张 `label_cells=True` 的证据图，按象限中心像素认出每张图落在
    哪个象限，再用同一个 `_draw_cell_label` 把 1/2/3/4 各画一遍作参照字形、
    逐像素认出那个象限画的是几号。返回 `{输入下标: 图上编号}`。
    """
    import io as _io

    import numpy as _np
    from PIL import Image as _Image, ImageDraw as _ImageDraw

    from pipeline.annotate import build_evidence_sheet, _draw_cell_label

    colours = [(10, 120, 200), (20, 200, 60), (200, 180, 10), (150, 20, 190)]
    paths = []
    for i, c in enumerate(colours):
        p = tmp_path / f"cellsrc{i}.png"
        _Image.fromarray(_np.full((200, 200, 3), c, dtype=_np.uint8)).save(p)
        paths.append(p)
    png = build_evidence_sheet(paths, [(10, 10, 190, 190)] * 4, label_cells=True)
    arr = _np.asarray(_Image.open(_io.BytesIO(png)).convert("RGB"))

    refs = {}
    for n in range(1, 5):
        im = _Image.new("RGB", (100, 100), (32, 32, 32))
        _draw_cell_label(_ImageDraw.Draw(im), 0, 0, n)
        refs[n] = _np.all(_np.asarray(im) > 240, axis=-1)

    h, w = arr.shape[:2]
    ch, cw = h // 2, w // 2
    out = {}
    for i, colour in enumerate(colours):
        quad = [(col, row) for col in (0, 1) for row in (0, 1)
                if tuple(int(v) for v in arr[row * ch + ch // 2,
                                             col * cw + cw // 2]) == colour]
        assert len(quad) == 1, f"第 {i} 张图应恰好落在一个象限，实得 {quad}"
        col, row = quad[0]
        patch = arr[row * ch: row * ch + 100, col * cw: col * cw + 100]
        mask = _np.all(patch > 240, axis=-1)
        digits = [n for n, ref in refs.items() if _np.array_equal(mask, ref)]
        assert len(digits) == 1, f"第 {i} 张图那一格的编号认不出（匹配到 {digits}）"
        out[i] = digits[0]
    return out


def _four_frame_cand(order=7, tid=61, kind="exit"):
    """一个 best_dets 恰有 4 张、四张互不相同（t 各异）的候选。"""
    dets = track(tid, RIGHT_CX + 200, STANDING_H, 4, kind=kind, order=order)
    seg = mkseg(order=order, presenter_num=3)
    cands = entryexit_candidates(dets, seg, frame_w=W)
    cand = next(c for c in cands if c["track_id"] == tid)
    assert len(cand["best_dets"]) == 4, (
        f"夹具前提：这个候选必须恰有 4 张证据帧，实得 {len(cand['best_dets'])}"
    )
    return seg, dets, cand


def test_split_uses_row_major_cell_order(tmp_path):
    """格位号 k 必须对上 `build_evidence_sheet` 实际把哪一张图标成了 k 号。

    映射由渲染出的证据图**实测**得出（见 `_cell_number_of_input_index`），
    不照抄任何注释。列优先（把 2 号格接到 best_dets[2] 上）会让用户按图上
    写的号码指人、程序却切出另一格的画面——衣着编码给了错的人，不报错。
    """
    number_of_index = _cell_number_of_input_index(tmp_path)
    index_of_number = {n: i for i, n in number_of_index.items()}
    assert len(index_of_number) == 4, f"四个编号应互不相同，实得 {number_of_index}"

    _seg, _dets, cand = _four_frame_cand()
    cell_map = {c: {"role": "创业者", "seat": c, "text": f"创业者{c}"}
                for c in (1, 2, 3, 4)}
    pieces = entryexit.split_candidate_by_cells(cand, cell_map)
    assert len(pieces) == 4, f"四个格位应切出四份，实得 {len(pieces)}"

    by_cell = {p["cell_index"]: p for p in pieces}
    for n in (1, 2, 3, 4):
        assert len(by_cell[n]["best_dets"]) == 1, (
            f"每个格位只对应一张图，实得 {len(by_cell[n]['best_dets'])} 张"
        )
        expect = cand["best_dets"][index_of_number[n]]
        assert by_cell[n]["best_dets"][0] is expect, (
            f"{n} 号格应取图上标着 {n} 的那一张（best_dets[{index_of_number[n]}]），"
            f"实得 t={by_cell[n]['best_dets'][0].t}，应为 t={expect.t}"
        )


def test_split_out_of_range_cells_are_dropped():
    """证据图不足 4 张时，超出的格位号视为无效（宁可不补，不能拿别人的
    画面顶上）。"""
    seg = mkseg(order=7, presenter_num=3)
    dets = track(62, RIGHT_CX + 200, STANDING_H, 2, kind="exit", order=seg.order)
    cand = next(c for c in entryexit_candidates(dets, seg, frame_w=W)
                if c["track_id"] == 62)
    assert len(cand["best_dets"]) == 2, "夹具前提：本候选只有 2 张证据帧"
    pieces = entryexit.split_candidate_by_cells(
        cand, {1: {"role": "创业者", "seat": 1, "text": "创业者1"},
               4: {"role": "创业者", "seat": 2, "text": "创业者2"}})
    assert [p["cell_index"] for p in pieces] == [1], (
        f"只有 1 号格有对应的图，4 号格无效，实得 {[p['cell_index'] for p in pieces]}"
    )


def test_split_fills_seats_from_cells():
    """按格位切分后补位：一个「2 个人」的候选按「1/3是创业者3，2/4是创业者2」
    切开，两个座位都该补上。这正是 order 9 华韵传媒卡住的那一条。"""
    seg, dets, cand = _four_frame_cand(order=9, tid=71)
    labels = {"9": {cand["stable_id"]: {
        "role": "创业者", "seat": 3, "n_people": 2,
        "note": "1/3是创业者3，2/4是创业者2"}}}

    plan, needs_resplit, _ns = entryexit_fill_plan([seg], dets, labels, {9: {}})
    got = sorted((e["seat"], e.get("cell_index")) for e in plan.get(9, []))
    assert got == [(2, 2), (3, 1)], (
        f"座位 3 取 1 号格、座位 2 取 2 号格（同一目标的多个格位取号最小的"
        f"那一格），实得 {plan}"
    )
    assert needs_resplit == [], (
        f"备注已能解析出格位，不该再进 needs_resplit，实得 {needs_resplit}"
    )


def test_unparsable_note_still_goes_to_needs_resplit(capsys):
    """备注解析不出格位时仍进 needs_resplit（且不抛异常）——不能因为
    「切分功能上线了」就把解析不了的那批悄悄丢掉。"""
    seg, dets, cand = _four_frame_cand(order=6, tid=81)
    labels = {"6": {cand["stable_id"]: {
        "role": "创业者", "seat": 2, "n_people": 2, "note": "看不清谁是谁"}}}

    plan, needs_resplit, _ns = entryexit_fill_plan([seg], dets, labels, {6: {}})
    assert plan.get(6, []) == [], f"解析不出格位就不该补位，实得 {plan}"
    assert [(r["order"], r["track_id"], r["note"]) for r in needs_resplit] == [
        (6, 81, "看不清谁是谁")], f"实得 {needs_resplit}"


def test_split_layout_describes_single_image():
    """切分出的那个「人」只有一张图，证据图不再是四格拼图——layout 文案
    必须如实说 1 张，绝不能沿用「四格」的说法。

    沿用旧文案的后果：告诉模型「本组图共 4 张」，模型据此描述并不存在的
    另外三张画面，不报错。
    """
    seg, dets, cand = _four_frame_cand(order=9, tid=91)
    labels = {"9": {cand["stable_id"]: {
        "role": "创业者", "seat": 3, "n_people": 2, "note": "1是创业者3"}}}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {9: {}})
    entries = plan.get(9, [])
    assert len(entries) == 1, f"实得 {plan}"
    layout = entries[0].get("layout") or ""
    assert layout, f"切分出的补位项必须带自己的 layout 文案，实得 {entries[0]}"
    assert "1 张" in layout, f"layout 必须如实说只有 1 张，实得 {layout!r}"
    for bad in ("四格", "四张", "2×2", "2x2", "4 张", "共 2 张", "共 3 张"):
        assert bad not in layout, (
            f"单图 layout 里不该出现「{bad}」（那是多图拼图的说法）：{layout!r}"
        )

    from pipeline.assemble import describe_layout
    assert layout != describe_layout(0, n_entry=0, n_exit=4), (
        "切分出的单图不能沿用四张进出场镜头的构成文案"
    )
    assert layout != describe_layout(4, 0, 0), (
        "切分出的单图更不能沿用「四格皆为就座镜头」的构成文案"
    )


def test_split_counts_are_printed(capsys):
    """凡有切分必须可见：切出几张、补了几个座位、几条短语没解析出来，
    一律打印。静默切分 = 用户永远不知道「另两个」那半句被忽略了。"""
    seg, dets, cand = _four_frame_cand(order=1, tid=16)
    labels = {"1": {cand["stable_id"]: {
        "role": "创业者", "seat": 3, "n_people": 3,
        "note": "1、4是创业者3，另两个是混入人员"}}}

    entryexit_fill_plan([seg], dets, labels, {1: {}})
    out = capsys.readouterr().out
    m = re.search(r"切出\s*(\d+)\s*张单人画面", out)
    assert m and int(m.group(1)) == 2, (
        f"必须打印实际切出的单人画面数（1、4 两格 = 2 张），实得输出：{out!r}"
    )
    m2 = re.search(r"补进\s*(\d+)\s*个座位", out)
    assert m2 and int(m2.group(1)) == 1, (
        f"必须打印切分实际补进的座位数（1、4 同指创业者3，合并后 1 个），"
        f"实得输出：{out!r}"
    )
    m3 = re.search(r"(\d+)\s*条备注短语未能解析", out)
    assert m3 and int(m3.group(1)) == 1, (
        f"必须打印未解析短语数，实得输出：{out!r}"
    )
    assert "另两个是混入人员" in out, (
        f"未解析的短语必须原样打印出来（凡有未解析必须可见），实得输出：{out!r}"
    )


# ---------------------------------------------------------------------------
# 端到端：用户真实标注（只读 work/，一个字节都不写）
# ---------------------------------------------------------------------------

# 标注文件的位置由 `pipeline.paths` 解析（Task 56G 起交付物归
# `work/Output/<组>/`，旧位置还在时也读得到）。这里**不能**自己拼一遍：
# 拼错了这几条端到端会静静地 skip，而 skip 的表现恰恰是「没有验证过任何
# 东西」——本次移交付物时就是这样一口气多出 5 条 skip，差点没注意到。
_REAL_GROUP = Path(__file__).resolve().parents[1] / "work" / "文化4-4.28"
_REAL_LABELS = _pipeline_paths.labels_path(_REAL_GROUP, "文化4-4月28日视频")


def test_real_labels_produce_expected_splits():
    """用户真实标注端到端：order 1 律持兴观「已定 2 人（其中座位 2 被人工
    否决）+ 两条格位备注」，切分后必须补齐到 3 人。

    这条测试只读 `work/`，不写任何文件、不跑 run_all、不调 VLM。
    """
    import pytest

    missing = [str(p) for p in (
        _REAL_GROUP / "segments.json", _REAL_GROUP / "detections.json",
        _REAL_GROUP / "seats.json", _REAL_LABELS) if not p.exists()]
    if missing:
        pytest.skip(
            "跳过原因：用户真实标注端到端验证需要 work/文化4-4.28/ 下的真实"
            f"产物，当前缺少 {missing}——这条测试**没有验证过任何东西**，"
            "不要把它算作通过。要跑它请先备齐这些文件（阶段 2/3/4 的产物 + "
            "挑选页导出的 entry_labels_*.json）。"
        )

    import json as _json

    from pipeline.detect import load_detections
    from pipeline.entryexit import detections_fingerprint
    from pipeline.schedule import load_segments_json

    segs = load_segments_json(_REAL_GROUP / "segments.json")
    dets = load_detections(_REAL_GROUP / "detections.json")
    seats_raw = _json.loads((_REAL_GROUP / "seats.json").read_text(encoding="utf-8"))
    seats_by_order = {int(o): {int(t): s for t, s in m.items()}
                      for o, m in seats_raw.items()}
    det_fp = detections_fingerprint(_REAL_GROUP / "detections.json")
    cands_by_order = {s.order: entryexit_candidates(dets, s) for s in segs}
    labels = load_entry_labels(_REAL_LABELS, detections_fp=det_fp,
                               candidates_by_order=cands_by_order)
    assert labels, "夹具前提：真实标注应通过指纹校验（否则本测试测不到切分）"

    plan, needs_resplit, _ns = entryexit_fill_plan(
        segs, dets, labels, seats_by_order)

    seg1 = next(s for s in segs if s.order == 1)
    assert seg1.presenter_num == 3, f"夹具前提：order 1 应是 3 人项目，实得 {seg1.presenter_num}"
    occupied = set(seats_by_order.get(1, {}).values())
    rejected = entryexit.vacated_seats_of(labels, 1)
    assert rejected == {2}, f"夹具前提：真实标注里 order 1 的座位 2 被人工否决，实得 {rejected}"

    # order 1 律持兴观：两条格位备注（「1、4是创业者2」「1号是创业者3」）
    # 分别指向座位 2 与座位 3——用户 2026-08-23 20:00 那一版把第一条从
    # 「…是创业者3」改成了「…是创业者2」，正是为了补上此前空着的座位 2
    # （Task 51/52 报告里记的「order 1 缺座位 2」那条缺口，是**用户数据的
    # 缺口**，用户自己改标注补上了，解析器一字未动）。
    # 座位 2 曾被人工否决，切分把它换成了备注指定的那一格。
    filled = sorted(e["seat"] for e in plan.get(1, []))
    assert filled == [2, 3], (
        f"order 1 的两条备注分别写「创业者2」「创业者3」，应补上这两个座位，"
        f"实得 {plan.get(1)}"
    )
    assert all(e.get("cell_index") for e in plan[1]), (
        f"这个补位来自格位切分，必须带 cell_index，实得 {plan[1]}"
    )
    total = len((occupied - rejected) | set(filled))
    assert total == 3, (
        f"order 1 需 3 人，两条备注补齐座位 2、3 后应满员，实得 {total}"
    )

    # order 9 华韵传媒：「1/3是创业者3，2/4是创业者2」等备注应产出切分补位。
    #
    # 断言写成**结构性**的，不写死具体座位号列表：本测试读的是用户的实时
    # 标注文件，他每次重标都会改变结果（实测 00:48 那次重标后由 [2,3] 变成
    # [1,2,3]，测试因数据变动而红，代码并无问题）。真正要守的是「切分确实
    # 发生了、补的都是该段的合法座位、且切分项带 cell_index」。
    plan9 = plan.get(9, [])
    seg9 = next(x for x in segs if x.order == 9)
    filled9 = sorted(e["seat"] for e in plan9)
    assert filled9, f"order 9 的备注应产出补位，实得 {plan9}"
    assert len(filled9) == len(set(filled9)), f"同一座位不得补两次：{filled9}"
    assert all(1 <= n <= seg9.presenter_num for n in filled9), (
        f"补的座位号必须在 1..{seg9.presenter_num} 内，实得 {filled9}")
    assert any(e.get("cell_index") for e in plan9), (
        f"order 9 的备注含格位映射，至少一项应带 cell_index，实得 {plan9}")

    # 待切分项应一个不剩：用户 2026-08-23 20:00 那一版把 order 12 那条
    # 「人数 2、备注为空」也补上了备注。全部 10 条非空备注见
    # `test_real_note_forms_all_parse`，逐条可解析。
    assert needs_resplit == [], (
        f"全部格位备注都应可解析、当场切开，不该再有待切分项，实得 {needs_resplit}"
    )


def test_parse_keeps_neighbouring_group_role_distinct():
    """「上一组创业者」「下一组创业者」标的是相邻项目的人，绝不能被解析成
    「创业者」——`entryexit_fill_plan` 只对精确等于「创业者」的目标补位，
    认错就是把隔壁项目的人补进本项目（Task 41 引入这两个角色时的原意）。"""
    got = entryexit.parse_cell_note("1是上一组创业者，2是下一组创业者")
    assert got[1]["role"] == "上一组创业者", got
    assert got[2]["role"] == "下一组创业者", got
    assert got[1]["seat"] is None and got[2]["seat"] is None, got

    got2 = entryexit.parse_cell_note("1是上一组创业者2")
    assert got2[1]["role"] == "上一组创业者" and got2[1]["seat"] == 2, got2


def test_non_presenter_cells_are_not_backfilled():
    """备注里指明「不是创业者」的格位不得参与补位——这恰恰是切分要保住的
    信息。放宽成「凡是解析出来的格位都补」，就把工作人员/混入人员的衣着
    编进了创业者的行里，不报错。"""
    seg, dets, cand = _four_frame_cand(order=4, tid=64)
    labels = {"4": {cand["stable_id"]: {
        "role": "创业者", "seat": 2, "n_people": 2,
        "note": "1是创业者2，2是混入人员，3是工作人员"}}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan([seg], dets, labels, {4: {}})
    assert [(e["seat"], e.get("cell_index")) for e in plan.get(4, [])] == [(2, 1)], (
        f"只有 1 号格（创业者2）该补位，实得 {plan}"
    )
    assert needs_seat == [], (
        f"「混入人员」「工作人员」两格既不补位、也不该混进 needs_seat"
        f"（那是给「标了创业者但没写座位号」的），实得 {needs_seat}"
    )
    assert needs_resplit == [], needs_resplit


# ===========================================================================
# Task 52 · 需求三：候选绝对下限（框高占画面高度的比例）
#
# 用户 2026-08-23：「其实只用了 33/160 个候选，就标记完了，减少候选的图片
# 数，比如特别小的人脸图，就可以不用做候选了」。
#
# 高度阈值在本项目被证伪过两次（0.30 / 0.40，见模块顶部），这一版**不是
# 判据、只是绝对下限**：阈值由用户真实标注实测定出——文化4组 32 个被标为
# 创业者的候选里最矮的一个占比 0.2108（227.7px），把下限放在 0.15
# （162px）落在实测分布的天然断档 0.1368→0.1567 中间，最矮的真人仍比它高
# 41%。被挡掉的候选与长宽比一样进折叠区，不静默丢弃。
# ===========================================================================

def test_tiny_candidates_are_filtered():
    """低于 `config.ENTRYEXIT_MIN_HEIGHT_RATIO` 的候选不进候选池。

    夹具刻意让被挡的那条**长宽比达标**（h/w = 100/30 ≈ 3.3 ≫ 1.30），
    否则它会被既有的长宽比过滤挡掉，这条新规则改不改都绿。
    """
    seg = mkseg()
    tiny_h = 0.10 * H          # 108px，低于 0.15*1080=162
    ok_h = 0.30 * H            # 324px，稳稳高于下限
    dets = (track(70, RIGHT_CX + 200, ok_h, 6, order=seg.order)
            + track(71, RIGHT_CX + 500, tiny_h, 6, order=seg.order))
    assert tiny_h / 30.0 >= config.ENTRYEXIT_MIN_ASPECT, (
        "夹具前提：被挡的那条长宽比必须达标，否则测的是长宽比过滤不是高度下限"
    )
    assert tiny_h / H < config.ENTRYEXIT_MIN_HEIGHT_RATIO <= ok_h / H, (
        f"夹具前提：{tiny_h/H} 应低于下限 {config.ENTRYEXIT_MIN_HEIGHT_RATIO}"
        f"、{ok_h/H} 应高于它"
    )

    out = entryexit_candidates(dets, seg, frame_w=W, frame_h=int(H))
    assert {c["track_id"] for c in out} == {70}, (
        f"框高占比低于 {config.ENTRYEXIT_MIN_HEIGHT_RATIO} 的候选不应进候选池，实得 {out}"
    )


def test_filtered_tiny_are_disclosed_with_reason():
    """被高度下限挡掉的候选进旁路列表，且标明是被**哪条**规则挡的。

    凡有过滤必须可见；而「可见」不止是「能看到这个框」，还得说清它为什么
    被挡——用户要判断的是「这条规则挡错了没有」，看不到理由就判断不了。
    """
    seg = mkseg()
    dets = (track(80, RIGHT_CX + 100, 0.40 * H, 6, order=seg.order)                     # 正常
            + track(81, RIGHT_CX + 400, 0.10 * H, 6, order=seg.order)                    # 只矮
            + track(82, RIGHT_CX + 700, 0.40 * H, 6, order=seg.order, width=400.0)       # 只方
            + track(83, RIGHT_CX + 1000, 0.10 * H, 6, order=seg.order, width=200.0))     # 又矮又方

    kept = entryexit_candidates(dets, seg, frame_w=W, frame_h=int(H))
    filtered = entryexit.entryexit_filtered_candidates(dets, seg, frame_w=W, frame_h=int(H))
    assert {c["track_id"] for c in kept} == {80}
    reasons = {c["track_id"]: c["filter_reasons"] for c in filtered}
    assert set(reasons) == {81, 82, 83}, (
        f"三条被挡的候选都必须出现在旁路列表里（不静默丢弃），实得 {sorted(reasons)}"
    )
    assert reasons[81] == ["height"], f"只矮不方的应只标高度，实得 {reasons[81]}"
    assert reasons[82] == ["aspect"], f"只方不矮的应只标长宽比，实得 {reasons[82]}"
    assert sorted(reasons[83]) == ["aspect", "height"], (
        f"两条都触发的必须两条都标出来，实得 {reasons[83]}"
    )
    for c in filtered:
        assert c["best_dets"], "旁路候选必须原样带着证据帧，否则页面上看不到图"


def test_threshold_keeps_all_user_labelled_presenters():
    """**用用户真实标注**验证他标过的真人一个也没从配对池里消失。

    原先这条钉的是「阈值必须定在实测最小值以下」，理由是：阈值定高一点点，
    已标定的创业者就会被挡出候选池，页面上再也找不到那个人，而流水线照常跑
    完、不报错。

    2026-08-25 起这条前提变了——被挡的候选进折叠区，而折叠区**可标注、也照常
    补位**（`candidates_for_labels` = 展示 ∪ 折叠）。用户实测城市2 order 4：
    那个人就在折叠区里，框高占比 0.096。所以「有人低于阈值」不再是事故，只是
    「他被折叠了」。

    真正的安全护栏因此改成：**用户标过的每一个创业者，都必须还能在配对池里
    找得到**。找不到才是事故——那意味着这条标注配不上任何候选，会被静默忽略。
    低于阈值的那几个另行列出来，供人判断阈值是不是该调，但不再作为失败条件。
    """
    import pytest

    if not _REAL_LABELS.exists() or not (_REAL_GROUP / "detections.json").exists():
        pytest.skip(
            "跳过原因：本条要用 work/文化4-4.28/ 下用户真实标注的 "
            f"{_REAL_LABELS.name} 与 detections.json/segments.json 反查每个"
            "被标为创业者的候选的框高占比，当前文件不齐——这条测试**没有"
            "验证过任何东西**，不要把它算作通过。"
        )

    import json as _json

    from pipeline.detect import load_detections
    from pipeline.schedule import load_segments_json

    segs = load_segments_json(_REAL_GROUP / "segments.json")
    dets = load_detections(_REAL_GROUP / "detections.json")
    labels = _json.loads(_REAL_LABELS.read_text(encoding="utf-8"))

    frame_h = 1080
    ratios: list[tuple[float, int, str]] = []
    lost: list[str] = []
    for seg in segs:
        rec = labels.get(str(seg.order))
        if not isinstance(rec, dict):
            continue
        # 配对池 = 展示 ∪ 折叠：折叠区的候选同样能标、能补位。
        pool = entryexit.candidates_for_labels(dets, seg, frame_h=frame_h)
        by_sid = {c["stable_id"]: c for c in pool}
        for sid, lab in rec.items():
            if not isinstance(lab, dict) or lab.get("role") != "创业者":
                continue
            cand = by_sid.get(sid)
            if cand is None:
                # 精确没中就走**流水线用的那套模糊重对齐**：重抽帧/重检测之后
                # 稳定标识会有像素级、秒级的抖动，`entryexit_fill_plan` 正是靠
                # 这一步吸收掉的。测试若只做精确匹配，就会在每次密抽之后误报
                # 「用户的人丢了」——而流水线其实好好地对上了。
                probe = {sid: lab}
                cand = next((c for c in pool
                             if entryexit.match_label_to_candidate(dict(probe), c)),
                            None)
            if cand is None:
                lost.append(f"order {seg.order} {sid}")
                continue
            ratios.append((cand["median_h"] / frame_h, seg.order, sid))

    assert len(ratios) + len(lost) >= 30, (
        f"夹具前提：真实标注里应有 30 个以上被标为创业者的候选，实得 "
        f"{len(ratios) + len(lost)}——数量对不上说明没真的读到标注"
    )
    assert not lost, (
        f"用户已标为创业者的这几条在配对池里找不到对应候选：{lost}。"
        "配不上就会被静默忽略——他标了、页面绿了、人一个也没补进去。"
        "（注意：被阈值挡进折叠区**不算**丢，折叠区同样在配对池里。）"
    )
    # 低于阈值的那几个只报出来，不作为失败条件：它们进折叠区，用户在那里照样
    # 能标、也照常补位。数字仍有参考价值——低于阈值的人越多，说明这一组的阈值
    # 越可能偏高，值得调。
    below = sorted(r for r in ratios if r[0] < config.ENTRYEXIT_MIN_HEIGHT_RATIO)
    if below:
        print(f"\n   [信息] {len(below)} 个已标定的创业者低于框高下限 "
              f"{config.ENTRYEXIT_MIN_HEIGHT_RATIO}，会进折叠区（仍可标注）："
              + "、".join(f"order{o} 占比{r:.3f}" for r, o, _s in below[:5])
              + ("…" if len(below) > 5 else ""))


# ===========================================================================
# Task 52 · 需求四：格位备注解析器扩展
# ===========================================================================

def test_parse_space_separated_multi_group():
    """「1 2 3是创业者2 4是创业者1」——空格分隔的多格位 + 同一条短语里
    两组映射。用户 order 7 的真实写法，此前整条解析失败、那一段少一个人。
    """
    got = entryexit.parse_cell_note("1 2 3是创业者2 4是创业者1")
    assert set(got) == {1, 2, 3, 4}, f"四个格位都应有归属，实得 {sorted(got)}"
    for k in (1, 2, 3):
        assert got[k]["role"] == "创业者" and got[k]["seat"] == 2, (k, got[k])
    assert got[4]["role"] == "创业者" and got[4]["seat"] == 1, got[4]

    rep = entryexit.parse_cell_note_report("1 2 3是创业者2 4是创业者1")
    assert rep["unparsed_phrases"] == [], rep


def test_ambiguous_trailing_number_is_not_guessed():
    """「2为创业者2 3」结尾孤立的「3」是歧义写法——可能是「格2、格3都是
    创业者2」，也可能是「格2是创业者2的第3人」。**歧义不猜**，整条进
    unparsed_phrases 让用户改写。

    猜错的后果与相对指代同级：把别人的衣着编进这个座位，且不报错。
    """
    rep = entryexit.parse_cell_note_report("2为创业者2 3")
    assert rep["parsed"] == {}, (
        f"歧义写法不得解析出任何格位（含只认前半截的「部分解析」），实得 {rep['parsed']}"
    )
    assert "2为创业者2 3" in rep["unparsed_phrases"], rep

    # 用户真实写法（order 3）：整条里既有能解析的也有歧义的——能解析的照常
    # 生效，歧义那条留空并如实列出。
    rep2 = entryexit.parse_cell_note_report("1为创业者3； 2为创业者2 3 ；4为创业者1")
    assert set(rep2["parsed"]) == {1, 4}, (
        f"歧义的第 2 条不得占用格位 2/3，实得 {sorted(rep2['parsed'])}"
    )
    assert any("2为创业者2 3" in p for p in rep2["unparsed_phrases"]), rep2


# ---------------------------------------------------------------------------
# Task 52 追加：用户真实写过的每一条非空备注，逐条钉住。解析器的验收样本
# 一律取自真实写法——自己编的样本只能证明代码跟自己一致（Task 46 报告里
# 那条教训）。
#
# 表里既有**当前**文化4组文件里的 10 条，也保留了用户此后改写掉的旧写法
# （如「另两个是混入人员」）——那些是回归样本：解析器绝不能哪天忽然开始
# 猜相对指代，哪怕现在的数据里已经没有这种写法。Task 53 又添了城市2组
# 那条跨段写法（「下一场创业者N」）。
# ---------------------------------------------------------------------------

_REAL_NOTE_CASES = [
    # (备注原文, 期望的 {格位: (角色, 座位)}, 期望未解析的短语)
    # ↓ 旧写法，用户已改成「1、4是创业者2」；保留作「相对指代不许猜」的回归
    ("1、4是创业者3，另两个是混入人员",
     {1: ("创业者", 3), 4: ("创业者", 3)}, ["另两个是混入人员"]),
    ("1、4是创业者2", {1: ("创业者", 2), 4: ("创业者", 2)}, []),
    ("1 2 3是创业者2", {1: ("创业者", 2), 2: ("创业者", 2), 3: ("创业者", 2)}, []),
    # 城市2组段 15：跨段流转（Task 53 需求六）。「下一场」归一成「下一组」。
    ("1/4是下一场创业者1；2/3是下一场创业者2",
     {1: ("下一组创业者", 1), 2: ("下一组创业者", 2),
      3: ("下一组创业者", 2), 4: ("下一组创业者", 1)}, []),
    ("1号是创业者3", {1: ("创业者", 3)}, []),
    ("1 3 为创业者2", {1: ("创业者", 2), 3: ("创业者", 2)}, []),
    ("1为创业者3； 2为创业者2 ；3 4为创业者1",
     {1: ("创业者", 3), 2: ("创业者", 2), 3: ("创业者", 1), 4: ("创业者", 1)}, []),
    ("1 2 3是创业者2 4是创业者1",
     {1: ("创业者", 2), 2: ("创业者", 2), 3: ("创业者", 2), 4: ("创业者", 1)}, []),
    ("1/3是创业者3，2/4是创业者2",
     {1: ("创业者", 3), 2: ("创业者", 2), 3: ("创业者", 3), 4: ("创业者", 2)}, []),
    ("3 4 是创业者2", {3: ("创业者", 2), 4: ("创业者", 2)}, []),
    ("1 3是创业者2； 2 4是创业者3",
     {1: ("创业者", 2), 2: ("创业者", 3), 3: ("创业者", 2), 4: ("创业者", 3)}, []),
    ("1 2 是创业者2；3 4 是创业者3",
     {1: ("创业者", 2), 2: ("创业者", 2), 3: ("创业者", 3), 4: ("创业者", 3)}, []),
]


def test_real_note_forms_all_parse():
    """用户真实写过的每一条备注逐条解析，只有相对指代那一条留空。"""
    for note, expect, unparsed in _REAL_NOTE_CASES:
        rep = entryexit.parse_cell_note_report(note)
        got = {k: (v["role"], v["seat"]) for k, v in rep["parsed"].items()}
        assert got == expect, f"备注 {note!r} 应解析为 {expect}，实得 {got}"
        assert rep["unparsed_phrases"] == unparsed, (
            f"备注 {note!r} 的未解析短语应为 {unparsed}，实得 {rep['unparsed_phrases']}"
        )


def test_real_labels_fill_every_labelled_seat():
    """端到端：用户真实标注里**指名道姓标出来的每个座位**都必须补得进去。

    断言从标注文件里**推导**，不写死具体数字。原先这里钉的是「34/34 全齐」
    和 order 1 两条备注的原文——那些是用户某一版标注的样子，而这份文件他
    随时在改：2026-08-24 19:49 他重新导出了一版，order 8 座位 3 这一版还没
    标，测试当场变红，而代码一个字没动。同一个坑本会话已经踩过一次
    （`test_real_labels_produce_expected_splits` 曾钉死 `[2, 3]`）。

    钉住的是**不变式**，与他标到了几成无关：

      · 标了的座位一个都不能丢——标注里明确写了「创业者 + 座位号」的，要么
        就座阶段已经占着、要么出现在补位计划里。悄悄少一个，用户会以为自己
        标漏了而反复重标，代码却一处也不报错。
      · 同一段里一个座位不许被两条计划项同时认领。
      · 不许覆盖就座阶段已经占住的座位。

    **绝不许**为了凑齐座位数去猜：猜错就是把工作人员的衣着编进创业者的行里，
    而整条流水线照常跑完、不报错。覆盖率（还差几个人）是用户的进度，不是
    代码的正确性，该由 CLI 打印出来给他看，不该由测试来判定。
    """
    import pytest

    missing = [str(p) for p in (
        _REAL_GROUP / "segments.json", _REAL_GROUP / "detections.json",
        _REAL_GROUP / "seats.json", _REAL_LABELS) if not p.exists()]
    if missing:
        pytest.skip(
            "跳过原因：本条端到端要读 work/文化4-4.28/ 下的真实产物，当前"
            f"缺少 {missing}——这条测试**没有验证过任何东西**，不要把它算作"
            "通过。"
        )

    import json as _json

    from pipeline.detect import load_detections
    from pipeline.entryexit import detections_fingerprint
    from pipeline.schedule import load_segments_json

    segs = load_segments_json(_REAL_GROUP / "segments.json")
    dets = load_detections(_REAL_GROUP / "detections.json")
    seats_raw = _json.loads((_REAL_GROUP / "seats.json").read_text(encoding="utf-8"))
    seats_by_order = {int(o): {int(t): s for t, s in m.items()}
                      for o, m in seats_raw.items()}
    det_fp = detections_fingerprint(_REAL_GROUP / "detections.json")
    cands_by_order = {s.order: entryexit_candidates(dets, s) for s in segs}
    labels = load_entry_labels(_REAL_LABELS, detections_fp=det_fp,
                               candidates_by_order=cands_by_order)
    assert labels, "夹具前提：真实标注应通过指纹校验"

    plan, needs_resplit, _needs_seat = entryexit_fill_plan(
        segs, dets, labels, seats_by_order)

    # 标注里**明确指名**的座位：候选层写了「创业者 + 座位号」，或格位分配
    # 逐格解引用之后落到某个座位上。这是「用户说了的」，与他还有多少没说
    # 无关——判据必须从文件里推导，不能写死某一版的数字。
    labelled: dict[int, set[int]] = {}
    for order_key, recs in labels.items():
        # 旁路通道（seat_overrides / cell_exclusions / no_candidate_seats …）
        # 与段号平级地挂在顶层，且**不都以下划线开头**——手写一份判断必然
        # 漏掉后加的那个，这里只认 LABEL_RESERVED_KEYS 这一个来源。
        if order_key in entryexit.LABEL_RESERVED_KEYS or not isinstance(recs, dict):
            continue
        order = int(order_key)
        for rec in recs.values():
            if not isinstance(rec, dict):
                continue
            cell_map, _report, _src = entryexit.cell_map_of_label(rec)
            if cell_map:
                for info in cell_map.values():
                    if info.get("role") == "创业者" and info.get("seat"):
                        labelled.setdefault(order, set()).add(int(info["seat"]))
            elif rec.get("role") == "创业者" and rec.get("seat"):
                labelled.setdefault(order, set()).add(int(rec["seat"]))

    assert labelled, (
        "夹具前提：真实标注里应至少有一个指名到座位号的创业者；一个都没有"
        "说明标注文件读空了，下面的断言会全部空转"
    )

    dropped: dict[int, list[int]] = {}
    covered = 0
    for seg in segs:
        occupied = set(seats_by_order.get(seg.order, {}).values())
        rejected = entryexit.vacated_seats_of(labels, seg.order)
        planned = {e["seat"] for e in plan.get(seg.order, []) if e.get("seat")}
        have = ((occupied - rejected) | planned)
        covered += len({s for s in have if s})
        miss = sorted(s for s in labelled.get(seg.order, set()) if s not in have)
        if miss:
            dropped[seg.order] = miss

    assert dropped == {}, (
        f"这些座位用户明确标了创业者+座位号，却没能补进去——标了不生效且"
        f"不报错，他会以为自己标漏了而反复重标：{dropped}"
    )

    # 一个座位不许被两条计划项同时认领：两条都会去调 VLM，后写的覆盖先写的，
    # 而覆盖掉的可能正是对的那一条，且不报错。
    for order, entries in plan.items():
        seats = [e["seat"] for e in entries if e.get("seat")]
        assert len(seats) == len(set(seats)), (
            f"第 {order} 段有座位被多条计划项同时认领：{sorted(seats)}")

    # 不许覆盖就座阶段已经占住的座位（entryexit_fill_plan 自身的不变式，
    # 这里在真实数据上再验一次）。
    for order, entries in plan.items():
        occupied = set(seats_by_order.get(order, {}).values())
        rejected = entryexit.vacated_seats_of(labels, order)
        clash = sorted({e["seat"] for e in entries if e.get("seat")}
                       & (occupied - rejected))
        assert not clash, f"第 {order} 段递补覆盖了已占座位 {clash}"

    # 覆盖率只打印，不判定：还差几个人是用户的标注进度，不是代码的正确性。
    total = sum(s.presenter_num for s in segs)
    print(f"\n   [信息] 文化4组当前 {covered}/{total} 个座位已定，"
          f"待切分项 {len(needs_resplit)} 个")


# ===========================================================================
# Task 53 · 需求一：多人框不再强制「先选创业者」
#
# 用户 2026-08-23 原话：「必须选择创业者，备注才会生效吗」。
#
# 一个 2×2 框里三个人分属不同座位（甚至分属相邻项目）时，「这个候选的角色」
# 本身就没有单一答案——旧实现却把 `role == "创业者"` 当成硬闸门，角色不对
# 时**备注根本不会被读到**，用户只能挑一个语义不准的值去骗过闸门。
#
# 新判据（三条，顺序即优先级，写在 `entryexit_fill_plan` 的 docstring 里）：
#   1) 备注能解析出格位映射 → 按备注切分，角色字段仅作参考；
#   2) 备注解析不出 → 沿用旧判据，`role == "创业者"` 才补位；
#   3) 工作人员/专家/上下一组创业者 **且**无可解析备注 → 一律不补。
# ===========================================================================

def test_note_only_candidate_is_processed_without_role():
    """没选角色（role 缺失）但写了可解析的格位备注 → 照常切分补位。

    旧实现在这里直接 `continue`，用户写的备注一个字都不会被读到，而且
    没有任何提示——最坏的那类失败：他以为标完了，那两个座位一直空着。
    """
    seg, dets, cand = _four_frame_cand(order=21, tid=121)
    labels = {"21": {cand["stable_id"]: {
        # role 这一项**根本不存在**（不是空串），模拟用户只写备注没点角色
        "n_people": 3, "note": "1是创业者1，2是创业者2"}}}

    plan, needs_resplit, needs_seat = entryexit_fill_plan(
        [seg], dets, labels, {21: {}})
    got = sorted((e["seat"], e.get("cell_index")) for e in plan.get(21, []))
    assert got == [(1, 1), (2, 2)], (
        f"备注写清了格位就该按备注补位，与有没有选角色无关，实得 {plan}")
    assert needs_resplit == [] and needs_seat == [], (
        f"实得 needs_resplit={needs_resplit} needs_seat={needs_seat}")


def test_note_overrides_conflicting_role():
    """角色写着「工作人员」、备注写着「1是创业者2」→ **以备注为准**。

    理由（同样写进 docstring）：备注是用户为了描述格位专门写的，比一个
    只能有单一取值的角色下拉更具体；一个框里三个人分属不同角色时，角色
    字段本就没有正确答案。
    """
    seg, dets, cand = _four_frame_cand(order=22, tid=122)
    labels = {"22": {cand["stable_id"]: {
        "role": "工作人员", "seat": None, "n_people": 2,
        "note": "1是创业者2，2是工作人员"}}}

    plan, needs_resplit, _ns = entryexit_fill_plan([seg], dets, labels, {22: {}})
    assert [(e["seat"], e.get("cell_index")) for e in plan.get(22, [])] == [(2, 1)], (
        f"备注说 1 号格是创业者2，就该补座位 2；角色「工作人员」只作参考，"
        f"实得 {plan}")
    assert needs_resplit == [], needs_resplit


def test_no_note_still_requires_presenter_role():
    """**没有**可解析备注时判据一字未变：`role == "创业者"` 才补位。

    这条守住「放行一切」那种过度放宽——工作人员/专家/相邻组的人若因为
    没写备注就一律补进座位，衣着编码会给错的人，且不报错。

    夹具三条候选都指定了座位号、座位号都是空的（`seats_by_order` 为空），
    所以「补不进去」只可能是角色判据挡下的，不是没座位可补。
    """
    seg = mkseg(order=23, presenter_num=3)
    dets = (track(131, RIGHT_CX + 100, STANDING_H, 4, kind="exit", order=23)
            + track(132, RIGHT_CX + 400, STANDING_H, 4, kind="exit", order=23)
            + track(133, RIGHT_CX + 700, STANDING_H, 4, kind="exit", order=23))
    cands = {c["track_id"]: c for c in entryexit_candidates(dets, seg, frame_w=W)}
    assert len(cands) == 3, f"夹具要造出 3 个互不合并的候选，实得 {len(cands)}"
    labels = {"23": {
        cands[131]["stable_id"]: {"role": "工作人员", "seat": 1, "n_people": 1,
                                   "note": ""},
        cands[132]["stable_id"]: {"role": "下一组创业者", "seat": 2, "n_people": 1,
                                   "note": ""},
        cands[133]["stable_id"]: {"role": "创业者", "seat": 3, "n_people": 1,
                                   "note": ""},
    }}

    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {23: {}})
    assert [(e["track_id"], e["seat"]) for e in plan.get(23, [])] == [(133, 3)], (
        f"只有角色精确等于「创业者」的那条该补位，实得 {plan}")


def test_unparsable_note_does_not_bypass_the_role_gate():
    """备注写了但**解析不出来**时，角色闸门照旧生效。

    这是判据 2 与判据 3 的交界：不能因为「写了备注」就放行——那等于把
    闸门换成一个更容易绕过的（随便写一句话就能进）。
    """
    seg, dets, cand = _four_frame_cand(order=24, tid=124)
    labels = {"24": {cand["stable_id"]: {
        "role": "工作人员", "seat": 1, "n_people": 2,
        "note": "左边那个是创业者1"}}}   # 相对指代，解析器一律不猜

    plan, needs_resplit, _ns = entryexit_fill_plan([seg], dets, labels, {24: {}})
    assert plan.get(24, []) == [], (
        f"解析不出格位、角色又不是创业者 → 一律不补，实得 {plan}")
    assert needs_resplit == [], (
        f"needs_resplit 只收「标为创业者的多人框」，工作人员切开也还是工作"
        f"人员，实得 {needs_resplit}")


# ===========================================================================
# Task 53 · 需求三：删掉「本组该位置恒为工作人员/专家」
#
# 用户 2026-08-23 原话：「这个选项实际没用，删掉吧」。页面控件、
# `position_exclusions` 的读取与应用一并删除，**但旧文件里残留的该字段不得
# 让载入报错**——用户手上已经有导出过的文件，一个 KeyError/TypeError 会让
# 整份标注读不进来。
# ===========================================================================

def test_legacy_position_exclusions_ignored_not_crash(tmp_path, capsys):
    """旧标注文件里残留 `position_exclusions` → 静默忽略并打印一行说明。

    三件事都要成立：① 不抛异常；② 其余标注**照常可用**（不是整份作废）；
    ③ 返回值里不再挂这个键（挂着会让下游以为它还生效）。
    """
    import json as _json2

    from pipeline.detect import save_detections
    from pipeline.entryexit import detections_fingerprint

    seg = mkseg(order=3, presenter_num=2)
    dets = track(41, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=3)
    cands = entryexit_candidates(dets, seg, frame_w=W)
    sid = cands[0]["stable_id"]

    det_path = tmp_path / "detections.json"
    save_detections(dets, det_path)
    fp = detections_fingerprint(det_path)

    labels_path = tmp_path / "entry_labels_x.json"
    labels_path.write_text(_json2.dumps({
        "_meta": {"detections_fingerprint": fp, "sheet": "x", "n_candidates": 1},
        "3": {sid: {"role": "创业者", "seat": 1, "n_people": 1, "note": ""}},
        # 用户上一版页面导出的残留字段
        "position_exclusions": [{"cx": 1204.0, "y2": 1079.0,
                                  "role": "工作人员", "note": "旧字段"}],
    }, ensure_ascii=False), encoding="utf-8")

    labels = load_entry_labels(labels_path, detections_fp=fp)
    assert "position_exclusions" not in labels, (
        f"已废弃的字段不得再挂在返回值上，实得 {sorted(labels)}")
    assert labels["3"][sid]["seat"] == 1, (
        f"其余标注必须照常可用（不是整份作废），实得 {labels}")

    out = capsys.readouterr().out
    assert "position_exclusions" in out and "废弃" in out, (
        f"静默忽略也要打印一行说明该字段已废弃，实得输出：{out!r}")

    # 而且它真的不再影响补位：这条排除正对着候选的位置也拦不住它
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {3: {}})
    assert [e["seat"] for e in plan.get(3, [])] == [1], (
        f"已废弃的位置排除不得再挡下任何候选，实得 {plan}")


def test_position_exclusions_are_not_applied_even_if_hand_written():
    """直接把 `position_exclusions` 塞进 labels 调 `entryexit_fill_plan`
    （绕过 `load_entry_labels`）时同样不生效——功能是删掉了，不是换个入口
    藏起来了。"""
    seg = mkseg(order=4, presenter_num=1)
    dets = track(51, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=4)
    cand = entryexit_candidates(dets, seg, frame_w=W)[0]
    labels = {
        "4": {cand["stable_id"]: {"role": "创业者", "seat": 1, "n_people": 1}},
        "position_exclusions": [{"cx": cand["median_cx"], "y2": cand["median_y2"],
                                  "role": "工作人员", "note": ""}],
    }
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {4: {}})
    assert [e["seat"] for e in plan.get(4, [])] == [1], (
        f"位置排除已删除，正对着候选也不该挡下它，实得 {plan}")


# ===========================================================================
# Task 53 · 需求六：「下一组创业者」自动流转到下一段
#
# 实测城市2组段 15 有一条 `role='下一组创业者' n=2
# note='1/4是下一场创业者1；2/3是下一场创业者2'`，而段 16 恰好缺人、整段
# 一条标注都没有。这条信息此前只被排除、没被利用。
#
# 跨段流转**必须谨慎**：这些候选的检测帧属于**本段**的时间窗口，不属于目标
# 段。补位后一律打 `cross_segment=1` + `src_order=<本段>`，让下游能区分。
# ===========================================================================

def _cross_scene(src_order=15, tgt_order=16, tid=135,
                 note="1/4是下一场创业者1；2/3是下一场创业者2",
                 role="下一组创业者", tgt_presenter_num=2):
    """源段一个 4 帧的多人退场候选 + 目标段（无任何标注）。"""
    src = Segment(order=src_order, company="源企业", start=0, end=1000,
                   presenter_num=1, presenters=())
    tgt = Segment(order=tgt_order, company="目标企业", start=1000, end=2000,
                   presenter_num=tgt_presenter_num, presenters=())
    dets = track(tid, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=src_order)
    cand = next(c for c in entryexit_candidates(dets, src, frame_w=W)
                if c["track_id"] == tid)
    assert len(cand["best_dets"]) == 4, (
        f"夹具前提：候选要恰有 4 张证据帧，实得 {len(cand['best_dets'])}")
    labels = {str(src_order): {cand["stable_id"]: {
        "role": role, "seat": None, "n_people": 2, "note": note}}}
    return src, tgt, dets, cand, labels


def test_next_group_presenter_flows_to_next_segment():
    """段 15 标成「下一组创业者」的格位备注，要变成段 16 的补位候选。

    夹具照城市2组实测那条造：段 16 需 2 人、就座只定了座位 1，座位 2 空着
    ——所以「补进去了」这件事只可能来自跨段流转（段 16 自己一条标注都没有）。
    """
    src, tgt, dets, _cand, labels = _cross_scene()
    plan, needs_resplit, needs_seat = entryexit_fill_plan(
        [src, tgt], dets, labels, {16: {99: 1}})

    assert plan.get(15, []) == [], (
        f"跨段候选不该补进**本段**（它标的是下一场的人），实得 {plan.get(15)}")
    got = [(e["seat"], e.get("cell_index")) for e in plan.get(16, [])]
    assert got == [(2, 2)], (
        f"「2/3是下一场创业者2」应补进段 16 的座位 2（取号最小的第 2 格）；"
        f"座位 1 已被就座阶段占着，不覆盖。实得 {plan.get(16)}")
    assert needs_resplit == [] and needs_seat == [], (
        f"实得 needs_resplit={needs_resplit} needs_seat={needs_seat}")


def test_cross_segment_fill_is_flagged():
    """跨段流转的补位项必须带 `cross_segment=1` 与 `src_order`。

    不打标记的后果：下游拿到一条「段 16 座位 2」的计划项，无从知道它的
    检测帧其实来自段 15 的时间窗口——查证时按段 16 的时间范围去找，找不到，
    而流水线照常跑完。
    """
    src, tgt, dets, _cand, labels = _cross_scene()
    plan, _nr, _ns = entryexit_fill_plan([src, tgt], dets, labels, {16: {99: 1}})
    item = plan[16][0]
    assert item.get("cross_segment") == 1, (
        f"跨段补位必须标记 cross_segment=1，实得 {item}")
    assert item.get("src_order") == 15, (
        f"必须记下这些帧实际来自哪一段（15），实得 {item}")

    # 同段的普通补位项**不许**多出这两个键（既有测试对整条字典做相等断言）
    seg = mkseg(order=30, presenter_num=1)
    d2 = track(60, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=30)
    c2 = entryexit_candidates(d2, seg, frame_w=W)[0]
    plan2, _nr, _ns = entryexit_fill_plan(
        [seg], d2, {"30": {c2["stable_id"]: {"role": "创业者", "seat": 1}}}, {30: {}})
    assert plan2[30] == [{"track_id": 60, "kind": "exit", "seat": 1,
                          "seat_uncertain": 0}], (
        f"同段补位项的形状一字不变，实得 {plan2[30]}")


def test_cross_segment_out_of_range_is_skipped(capsys):
    """目标段号越界（最后一段还写「下一场创业者」）→ 跳过，不报错。

    夹具只给一段（order=16，本组最后一段），它的备注指向不存在的 order 17。
    """
    src, _tgt, dets, _cand, labels = _cross_scene(src_order=16, tgt_order=17)
    plan, needs_resplit, needs_seat = entryexit_fill_plan(
        [src], dets, labels, {16: {}})     # 只喂源段，目标段不存在
    assert plan == {}, f"目标段不存在时不该凭空造出计划项，实得 {plan}"
    assert needs_resplit == [] and needs_seat == []
    out = capsys.readouterr().out
    assert "17" in out, f"跳过了什么要说出来（凡有丢弃必须可见），实得 {out!r}"


def test_cross_segment_does_not_overwrite_occupied_seat():
    """目标段的座位已被占用 → 不覆盖（沿用既有规则）。

    夹具让备注指向的**两个**座位（1 和 2）都被就座阶段占着——不这么造的话
    「不许覆盖」这条路径压根走不到，改与不改都绿。
    """
    src, tgt, dets, _cand, labels = _cross_scene()
    plan, _nr, _ns = entryexit_fill_plan(
        [src, tgt], dets, labels, {16: {99: 1, 98: 2}})
    assert plan.get(16, []) == [], (
        f"座位 1、2 都被就座阶段占着，跨段候选一个也不许补，实得 {plan.get(16)}")

    # 对照：把座位 2 腾出来，同一条备注立刻补得进去——证明上面的空结果
    # 是「被占用挡下的」，不是夹具压根没走到这条路径。
    plan2, _nr, _ns = entryexit_fill_plan(
        [src, tgt], dets, labels, {16: {99: 1}})
    assert [e["seat"] for e in plan2.get(16, [])] == [2], (
        f"对照组：座位 2 空着时必须补得进去，实得 {plan2.get(16)}")


def test_cross_segment_seat_beyond_presenter_num_is_skipped():
    """备注里的座位号超出目标段的 presenter_num → 跳过，不硬塞。

    「下一场创业者3」而下一段只有 2 个人：这个 3 只能是用户写错或数错了，
    硬塞会造出一条根本不存在的第 3 人记录。
    """
    src, tgt, dets, _cand, labels = _cross_scene(
        note="1/4是下一场创业者3", tgt_presenter_num=2)
    plan, _nr, _ns = entryexit_fill_plan([src, tgt], dets, labels, {16: {}})
    assert plan.get(16, []) == [], (
        f"座位 3 超出目标段的 2 人编制，不许补，实得 {plan.get(16)}")


def test_previous_group_presenter_flows_backwards():
    """「上一组创业者」同理映射到 order-1。"""
    src, tgt, dets, _cand, labels = _cross_scene(
        src_order=8, tgt_order=7, role="上一组创业者",
        note="1/4是上一场创业者2", tgt_presenter_num=2)
    plan, _nr, _ns = entryexit_fill_plan([tgt, src], dets, labels, {7: {}})
    got = [(e["seat"], e.get("src_order"), e.get("cross_segment"))
           for e in plan.get(7, [])]
    assert got == [(2, 8, 1)], (
        f"「上一场创业者2」应补进第 7 段座位 2 并打跨段标记，实得 {plan}")


def test_cross_segment_target_role_words_are_recognised():
    """用户写的是「下一场创业者N」（场，不是组）——解析器必须认得，并归一
    成页面上那个角色名「下一组创业者」。

    不认的后果：整条目标被当成一个未知角色（`role='下一场创业者1'`、
    `seat=None`），既不补位也不流转，而且不报错。
    """
    parsed = entryexit.parse_cell_note("1是下一场创业者2，2是上一场创业者1")
    assert parsed[1]["role"] == "下一组创业者" and parsed[1]["seat"] == 2, parsed
    assert parsed[2]["role"] == "上一组创业者" and parsed[2]["seat"] == 1, parsed


# ---------------------------------------------------------------------------
# 端到端（只读用户真实标注）：城市2组段 15 的「下一组创业者」要为段 16 产出
# 候选。段 16 需 2 人、就座只定了座位 1、**整段一条标注都没有**——补上的那
# 一个只可能来自段 15 的跨段流转。
# ---------------------------------------------------------------------------
_CITY_GROUP = Path(__file__).resolve().parents[1] / "work" / "城市2-4.26"
_CITY_LABELS = _pipeline_paths.labels_path(_CITY_GROUP, "城市2-4.26")


def test_real_cross_segment_labels_all_land_or_are_reported():
    """用户真实标注端到端：**每一条**跨段流转要么落位，要么有明确的跳过理由。

    原先这条钉的是城市2 段 15 → 段 16 那一例的全部细节（段 16 一条标注都没有、
    track_id == 35、cell_index == 2…）。那是他某一版标注的样子——2026-08-25 他
    给段 16 也标了人，夹具前提当场不成立，测试变红而代码一字未动（同类坑本会话
    第四次）。

    改为扫真实标注**自己找**所有跨段意图，逐条核对下场。合法的下场只有三种：
      · 落进目标段的计划里（带 cross_segment=1 与 src_order）；
      · 目标段不存在（第一段标「上一组」、最后一段标「下一组」）；
      · 目标座位已有人——跨段来的证据比本段自己的弱（帧不在目标段的时间窗口
        里），照规矩不覆盖。
    除此之外就是静默消失，那正是这条要防的。

    只读 `work/`，不写任何文件、不跑 run_all、不调 VLM。
    """
    import pytest

    missing = [str(p) for p in (
        _CITY_GROUP / "segments.json", _CITY_GROUP / "detections.json",
        _CITY_GROUP / "seats.json", _CITY_LABELS) if not p.exists()]
    if missing:
        pytest.skip(
            "跳过原因：跨段流转的端到端验证需要 work/城市2-4.26/ 下的真实产物，"
            f"当前缺少 {missing}——这条测试**没有验证过任何东西**，不要把它"
            "算作通过。"
        )

    import json as _json3

    from pipeline.detect import load_detections
    from pipeline.entryexit import detections_fingerprint
    from pipeline.schedule import load_segments_json

    segs = load_segments_json(_CITY_GROUP / "segments.json")
    dets = load_detections(_CITY_GROUP / "detections.json")
    seats_raw = _json3.loads((_CITY_GROUP / "seats.json").read_text(encoding="utf-8"))
    seats_by_order = {int(o): {int(t): s for t, s in m.items()}
                      for o, m in seats_raw.items()}
    det_fp = detections_fingerprint(_CITY_GROUP / "detections.json")
    cands = {s.order: entryexit.candidates_for_labels(dets, s) for s in segs}
    labels = load_entry_labels(_CITY_LABELS, detections_fp=det_fp,
                               candidates_by_order=cands)
    assert labels, "真实标注应能通过指纹校验载入（否则下面断言的是空数据）"

    # 扫出全部跨段意图：候选层的角色，以及格位分配里那几格。
    wants: list[tuple[int, int, int]] = []      # (源段, 目标段, 座位号)
    for key, recs in labels.items():
        if key in entryexit.LABEL_RESERVED_KEYS or not isinstance(recs, dict):
            continue
        src = int(key)
        for rec in recs.values():
            if not isinstance(rec, dict):
                continue
            cell_map, _rep, _s = entryexit.cell_map_of_label(rec)
            targets = (list(cell_map.values()) if cell_map else [rec])
            for info in targets:
                role = info.get("role")
                d = entryexit._NEIGHBOUR_ROLE_DELTA.get(role)
                if d is None or not info.get("seat"):
                    continue
                wants.append((src, src + d, int(info["seat"])))

    if not wants:
        pytest.skip(
            "当前这一版真实标注里没有任何「上/下一组创业者 + 座位号」——这条"
            "端到端无从验证。不是失败，但也**没有验证过任何东西**。")

    plan, _nr, _ns = entryexit_fill_plan(segs, dets, labels, seats_by_order)
    by_order = {int(s.order): s for s in segs}

    for src, tgt, seat in wants:
        landed = [e for e in plan.get(tgt, [])
                  if e.get("cross_segment") and e["seat"] == seat]
        if landed:
            assert landed[0]["src_order"] == src, (
                f"帧来自第 {src} 段，必须记下，实得 {landed[0]}")
            continue
        # 没落位，那必须是两条合法理由之一
        tgt_seg = by_order.get(tgt)
        if tgt_seg is None:
            continue                                    # 目标段不存在
        occupied = set(seats_by_order.get(tgt, {}).values())
        occupied -= entryexit.vacated_seats_of(labels, tgt)
        taken = {e["seat"] for e in plan.get(tgt, [])}
        assert seat in occupied or seat in taken or seat > tgt_seg.presenter_num, (
            f"第 {src} 段 → 第 {tgt} 段座位 {seat} 既没落位，也不是「目标段"
            f"不存在 / 该座位已有人 / 超出编制」——它是静默消失的，"
            f"目标段计划：{plan.get(tgt)}")


# ===========================================================================
# Task 54 · 需求一：格位备注 → 结构化选择题（`cells`）
#
# 用户 2026-08-23 原话：「能放弃解析，转换为选择吗，这样就能精准实时匹配到
# 座位人是否识别，比如选人数为 2 时，就给出这两人身份和座位号的选择题，且
# 可留空，根据选出的空，精准识别，不需要解析了」。
#
# 解析器已经因写法问题反复失败三轮（「2为创业者2 3」歧义、「1 2 3是创业者2
# 4是创业者1」缺分隔符、「另两个」相对指代）。更要命的是解析发生在 Python
# 侧，页面无法实时反馈：用户在城市2组 order1 明明已把三个人都写进备注，
# 目录仍是红的——要等下一次跑流水线才知道解析结果。
#
# 读取优先级（顺序即优先级，`cell_map_of_label`）：
#   1. 有 `cells` 且能解出内容 → 用 cells，**忽略 note**
#   2. 无 cells（或 cells 全空）→ 沿用解析器（`parse_cell_note`，保留不删）
#   3. 都没有 → 既有的角色闸门
# ===========================================================================

def _cells_label(cand, **extra):
    rec = {"role": "工作人员", "seat": None, "n_people": 2}
    rec.update(extra)
    return {cand["stable_id"]: rec}


def test_cells_structure_replaces_note_parsing():
    """有 `cells` 时按它补位，note 一个字都不许生效。

    夹具刻意让两者**指向不同的座位**：note 说全部四格都是「创业者1」（只会
    补座位 1），cells 说 1 号格是创业者2、2 号格是创业者3。三个座位都空着，
    所以「补进了哪些座位」只可能由优先级决定，不是没座位可补。
    """
    seg, dets, cand = _four_frame_cand(order=31, tid=141)
    labels = {"31": _cells_label(
        cand,
        note="1/2/3/4是创业者1",
        cells=[{"cell": 1, "role": "创业者", "seat": 2},
               {"cell": 2, "role": "创业者", "seat": 3}],
    )}
    # 夹具前提：这条 note 单独拿去解析确实会补座位 1（否则本条测不到差异）
    assert entryexit.parse_cell_note("1/2/3/4是创业者1") == {
        c: {"role": "创业者", "seat": 1, "text": "创业者1"} for c in (1, 2, 3, 4)
    }

    plan, needs_resplit, needs_seat = entryexit_fill_plan(
        [seg], dets, labels, {31: {}})
    got = sorted((e["seat"], e.get("cell_index")) for e in plan.get(31, []))
    assert got == [(2, 1), (3, 2)], (
        f"应完全按 cells 补位（座位2←1号格、座位3←2号格），实得 {plan}")
    assert 1 not in [e["seat"] for e in plan.get(31, [])], (
        f"note 指的座位 1 不许出现——有 cells 时 note 一律忽略，实得 {plan}")
    assert needs_resplit == [] and needs_seat == [], (
        f"实得 needs_resplit={needs_resplit} needs_seat={needs_seat}")


def test_cells_blank_role_or_seat_not_filled():
    """留空的格位（身份留空、座位号留空、整行留空）一律不参与补位。

    「留空 = 未判定」是本项目一贯的约定（Task 32：预填等于把「没判定」
    伪装成「已判定」）。三个座位全空，所以「没补上」只可能是留空判据挡下
    的；若留空也参与补位，座位 3 会被那条只有座位号、没有身份的行补上。
    """
    seg, dets, cand = _four_frame_cand(order=32, tid=142)
    labels = {"32": _cells_label(
        cand,
        n_people=3,
        cells=[{"cell": 1, "role": "创业者", "seat": 2},
               {"cell": 2, "role": "创业者"},          # 有身份、无座位号
               {"cell": 3, "seat": 3},                  # 有座位号、无身份
               {"cell": 4}],                            # 整行留空
    )}
    plan, _nr, needs_seat = entryexit_fill_plan([seg], dets, labels, {32: {}})
    got = sorted((e["seat"], e.get("cell_index")) for e in plan.get(32, []))
    assert got == [(2, 1)], (
        f"只有 1 号格身份+座位号都填了，只该补这一个，实得 {plan}")
    assert 3 not in [e["seat"] for e in plan.get(32, [])], (
        f"只填了座位号、身份留空的那行不许补位（未判定），实得 {plan}")
    # 「标了创业者却没座位号」照旧进 needs_seat（可见，但补不进去）
    assert [n["order"] for n in needs_seat] == [32], (
        f"2 号格标了创业者但没选座位号，应如实上报 needs_seat，实得 {needs_seat}")


def test_legacy_note_still_works_when_no_cells():
    """没有 `cells` 时沿用解析器——用户已按备注标完两组，绝不能作废。

    另核对「cells 是空列表」这一路：那等同于「还没填」，同样退回 note，
    不能让用户把人数拨到 2、控件刚展开还没选，此前写的备注就静默失效。
    """
    for cells in (None, []):
        seg, dets, cand = _four_frame_cand(order=33, tid=143)
        rec = {"role": "创业者", "seat": 3, "n_people": 2,
               "note": "1/3是创业者3，2/4是创业者2"}
        if cells is not None:
            rec["cells"] = cells
        plan, needs_resplit, _ns = entryexit_fill_plan(
            [seg], dets, {"33": {cand["stable_id"]: rec}}, {33: {}})
        got = sorted((e["seat"], e.get("cell_index")) for e in plan.get(33, []))
        assert got == [(2, 2), (3, 1)], (
            f"cells={cells!r} 时应沿用备注解析（座位3←1号格、座位2←2号格），"
            f"实得 {plan}")
        assert needs_resplit == [], needs_resplit


# --- 「同格位X」（用户追加需求）------------------------------------------
#
# 2×2 里同一个人常占两格（真实写法 `1/3是创业者3，2/4是创业者2`）。改成
# 选择题后每格都重填身份+座位号反而比写备注更累，因此格位 2 及以后可以选
# 「同格位X」，**只能引用比自己小的格位号**——引用只指向更小的号，按格位号
# 升序解引用必然终止，结构上不可能成环，不必事后检测环。

def test_same_as_inherits_role_and_seat():
    """「同格位1」继承格位 1 的身份与座位号（存的是引用，不是副本）。"""
    seg, dets, cand = _four_frame_cand(order=34, tid=144)
    labels = {"34": _cells_label(
        cand,
        cells=[{"cell": 1, "role": "创业者", "seat": 3},
               {"cell": 3, "same_as": 1}],
    )}
    cell_map = entryexit.cells_to_cell_map(labels["34"][cand["stable_id"]]["cells"])
    assert cell_map[3]["role"] == "创业者" and cell_map[3]["seat"] == 3, (
        f"3 号格应继承 1 号格的身份与座位号，实得 {cell_map}")

    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {34: {}})
    assert [(e["seat"], e.get("cell_index")) for e in plan.get(34, [])] == [(3, 1)], (
        f"两格是同一个人 → 只占一个座位，取号最小的那一格，实得 {plan}")


def test_same_as_chain_resolves():
    """链式引用（格4 → 格3 → 格1）要能一路解到底。"""
    cell_map = entryexit.cells_to_cell_map(
        [{"cell": 1, "role": "创业者", "seat": 2},
         {"cell": 3, "same_as": 1},
         {"cell": 4, "same_as": 3}])
    assert {c: (v["role"], v["seat"]) for c, v in cell_map.items()} == {
        1: ("创业者", 2), 3: ("创业者", 2), 4: ("创业者", 2)}, cell_map


def test_same_as_blank_source_stays_blank():
    """被引用的格位自身是空的 → 引用它的格位也视为空，不参与补位。

    若「引用空格」也能补位，用户还没判定的那一行会凭空占一个座位——衣着
    编码给了一个他根本没确认过的人，且不报错。
    """
    seg, dets, cand = _four_frame_cand(order=35, tid=145)
    labels = {"35": _cells_label(
        cand,
        cells=[{"cell": 1},                       # 源格全空
               {"cell": 2, "same_as": 1},
               {"cell": 3, "role": "创业者", "seat": 2}],
    )}
    cell_map = entryexit.cells_to_cell_map(labels["35"][cand["stable_id"]]["cells"])
    assert set(cell_map) == {3}, f"1、2 号格都该视为空，实得 {cell_map}"
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {35: {}})
    assert [(e["seat"], e.get("cell_index")) for e in plan.get(35, [])] == [(2, 3)], (
        f"只有 3 号格判定了，实得 {plan}")


def test_same_as_forward_reference_is_ignored(capsys):
    """只能引用**更小**的格位号：页面根本不给「同格位3」这个选项（见
    tests/test_entryreview.py::test_same_as_only_references_earlier_cells），
    手写文件里若出现前向引用（可能成环），一律视为空并打印，不静默。
    """
    cell_map = entryexit.cells_to_cell_map(
        [{"cell": 1, "same_as": 2}, {"cell": 2, "role": "创业者", "seat": 1}])
    assert set(cell_map) == {2}, f"前向引用应视为空，实得 {cell_map}"
    assert "同格位" in capsys.readouterr().out, "丢弃了东西必须打印出来"


def test_same_as_dangling_reference_is_ignored(capsys):
    """引用了一个根本不存在的格位 → 视为空并打印。"""
    cell_map = entryexit.cells_to_cell_map(
        [{"cell": 2, "same_as": 1}, {"cell": 3, "role": "创业者", "seat": 1}])
    assert set(cell_map) == {3}, f"悬空引用应视为空，实得 {cell_map}"
    assert "同格位" in capsys.readouterr().out


def test_cells_cross_segment_role_still_flows():
    """cells 里选「下一组创业者」照样跨段流转——切分之后的下游代码与备注
    那条路完全共用一套，不许另开一条路径。"""
    seg15 = mkseg(order=15, presenter_num=1)
    seg16 = mkseg(order=16, presenter_num=2)
    dets = track(151, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=15)
    cand = next(c for c in entryexit_candidates(dets, seg15, frame_w=W)
                if c["track_id"] == 151)
    labels = {"15": {cand["stable_id"]: {
        "role": "下一组创业者", "n_people": 2,
        "cells": [{"cell": 1, "role": "下一组创业者", "seat": 2}]}}}
    plan, _nr, _ns = entryexit_fill_plan([seg15, seg16], dets, labels,
                                         {15: {}, 16: {}})
    cross = [e for e in plan.get(16, []) if e.get("cross_segment")]
    assert len(cross) == 1 and cross[0]["seat"] == 2 and cross[0]["src_order"] == 15, (
        f"cells 里的「下一组创业者2」应流转到段 16 座位 2，实得 {plan}")


# --- 逐格排除（用户 2026-08-24：2×2 里个别格不是目标人）---

def test_cell_exclusions_read_from_labels():
    """从标注文件读出某座位要排除哪几格。"""
    from pipeline.entryexit import cell_exclusions_of
    labels = {"_meta": {}, "cell_exclusions": {"8": {"2": [2, 4]}}}
    assert cell_exclusions_of(labels, 8, 2) == {2, 4}
    assert cell_exclusions_of(labels, 8, 1) == set()
    assert cell_exclusions_of(labels, 9, 2) == set()
    assert cell_exclusions_of(None, 8, 2) == set()


def test_cell_exclusions_keyed_by_seat_not_track():
    """键是 (order, seat)，不是 track_id。

    座位号是人工确定的、跨重检测稳定；track_id 重检测即全变（Task 46 已
    因此把标注标识改成由视频内容导出）。若用 track_id 作键，用户排掉的那
    一格在下次重检测后会失效，且不报错——他得把每组重排一遍。
    """
    from pipeline.entryexit import cell_exclusions_of
    labels = {"_meta": {}, "cell_exclusions": {"8": {"2": [3]}}}
    # 同一 (order, seat) 无论 track_id 如何都取到同一份排除
    assert cell_exclusions_of(labels, 8, 2) == {3}


def test_apply_cell_exclusions_drops_those_frames():
    """构建证据图前按格位号剔除对应的帧。"""
    from pipeline.entryexit import apply_cell_exclusions
    dets = ["d1", "d2", "d3", "d4"]
    assert apply_cell_exclusions(dets, {2, 4}) == ["d1", "d3"]
    assert apply_cell_exclusions(dets, set()) == dets
    assert apply_cell_exclusions(dets, {1, 2, 3, 4}) == []


def test_apply_cell_exclusions_out_of_range_is_ignored():
    """排除号超出实际张数时忽略，不报错——证据图不足 4 张时会有灰块。"""
    from pipeline.entryexit import apply_cell_exclusions
    assert apply_cell_exclusions(["d1", "d2"], {4}) == ["d1", "d2"]


def test_cell_exclusions_survive_load(tmp_path):
    """`cell_exclusions` 必须能从文件读回来，且不被当成段号。

    它与 `seat_overrides` 一样是顶层旁路字段，必须同时做两件事：进
    LABEL_RESERVED_KEYS（否则会被当成一个「段」展开成垃圾条目），
    以及被 `_with_side_channels` 挂回结果（否则读不到、排除静默失效）。
    只做前者不做后者最难发现——文件里明明有，跑下来却毫无效果且不报错。
    """
    import json
    from pipeline.entryexit import load_entry_labels, cell_exclusions_of
    p = tmp_path / "entry_labels_x.json"
    p.write_text(json.dumps({
        "_meta": {"detections_fingerprint": "fp1", "sheet": "x", "n_candidates": 1},
        "8": {"exit@100#1200,900": {"role": "创业者", "seat": 1, "n_people": 1,
                                     "note": "", "track_id": 5}},
        "cell_exclusions": {"8": {"1": [2, 4]}},
    }, ensure_ascii=False), encoding="utf-8")

    labels = load_entry_labels(p, "fp1", {8: []})
    assert cell_exclusions_of(labels, 8, 1) == {2, 4}
    # 不得把 cell_exclusions 当成段号
    assert "cell_exclusions" not in [k for k in labels if k.isdigit()]


# ===========================================================================
# Task 56A · 同一个人占多格时，补位计划要留下**每一格**的格位号
#
# 用户 2026-08-24 实测（文化4组 order 2）：「标了格位1是创业者2，格位2是同
# 格位1，那么应该将这两帧图片都保留下来，而不是登记表只有一张图」。
#
# 根因就在本函数：两格解引用成同一个人后，`_merge_same_seat_candidates`
# 只留下证据最充分的那一份（切出来的每份都只有 1 帧，于是取先出现的那格），
# 另一格连同它那张帧一起消失，且没有任何一处报错——下游（登记表、核对页）
# 只能看到一半的证据，看上去却完全正常。
#
# 计划项因此改为同时带 `cell_index`（**号最小**的那一格，下游只送单图时用
# 它，形状与 Task 51 起逐字不变）与 `cell_indices`（这个人占的**全部**格位，
# 升序）。两者的关系是硬不变式：`cell_index == cell_indices[0]`。
# ===========================================================================

def test_fill_plan_keeps_every_cell_of_one_person():
    """「格位2 = 同格位1」→ 仍只占一个座位，但两格的格位号都要留下。"""
    seg, dets, cand = _four_frame_cand(order=36, tid=146)
    labels = {"36": _cells_label(
        cand,
        cells=[{"cell": 1, "role": "创业者", "seat": 2},
               {"cell": 2, "same_as": 1}],
    )}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {36: {}})
    got = [(e["seat"], e.get("cell_index"), e.get("cell_indices"))
           for e in plan.get(36, [])]
    assert got == [(2, 1, [1, 2])], (
        f"两格是同一个人 → 一个座位；但 1、2 两格的帧都是他本人的证据，"
        f"丢掉任何一格都是静默丢证据。实得 {plan.get(36)}")


def test_fill_plan_cell_indices_covers_note_written_pairs():
    """备注写法 `1/3是创业者3，2/4是创业者2` 同样要保住两格。

    走的是 `parse_cell_note` 那条路（没有 `same_as` 可依），「同一个人」的
    唯一依据是**同一个目标（role+seat）**——与下游 `by_seat` 合并的口径
    完全一致，不另立一套。
    """
    seg, dets, cand = _four_frame_cand(order=37, tid=147)
    labels = {"37": _cells_label(cand, note="1/3是创业者3，2/4是创业者2")}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {37: {}})
    got = sorted((e["seat"], e.get("cell_index"), tuple(e.get("cell_indices") or ()))
                 for e in plan.get(37, []))
    assert got == [(2, 2, (2, 4)), (3, 1, (1, 3))], (
        f"座位 3 占 1、3 两格，座位 2 占 2、4 两格，实得 {plan.get(37)}")


def test_fill_plan_does_not_merge_cells_of_different_candidates():
    """两个**不同候选**被标到同一个座位（Task 45 缺陷一：同一人在进/退场
    各留一个候选）时，仍旧只取其一，绝不把两条 track 的帧混成一个人的证据。

    `cell_indices` 只收**胜出那条候选自己**的格位号：另一条候选的格位号在
    它自己的证据图上编号，混进来就是拿另一张图的坐标去取帧。
    """
    seg = mkseg(order=38, presenter_num=3)
    dets = (track(151, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=38)
            + track(152, RIGHT_CX + 600, STANDING_H, 4, kind="entry", order=38))
    cands = {c["track_id"]: c for c in entryexit_candidates(dets, seg, frame_w=W)}
    labels = {"38": {
        cands[151]["stable_id"]: {"role": "工作人员", "seat": None, "n_people": 2,
                                  "cells": [{"cell": 1, "role": "创业者", "seat": 2},
                                            {"cell": 2, "same_as": 1}]},
        cands[152]["stable_id"]: {"role": "工作人员", "seat": None, "n_people": 2,
                                  "cells": [{"cell": 3, "role": "创业者", "seat": 2},
                                            {"cell": 4, "same_as": 3}]},
    }}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {38: {}})
    items = plan.get(38, [])
    assert len(items) == 1, f"同一座位只该有一条计划项，实得 {items}"
    item = items[0]
    assert item["kind"] == "exit" and item["track_id"] == 151, (
        f"退场候选优先（正面镜头，Task 45 缺陷一），实得 {item}")
    assert item.get("cell_indices") == [1, 2], (
        f"只收胜出那条候选自己的格位号；另一条的 3、4 号格是**另一张证据图**"
        f"上的编号，混进来等于拿错图的坐标取帧。实得 {item}")


def test_cross_segment_keeps_every_cell_of_one_person():
    """跨段流转同样要保住多格：「2/3是下一场创业者2」占 2、3 两格。

    旧实现在第二遍扫描里靠「该座位已有人，不覆盖」把第 3 格挡掉，于是那一
    格的帧被当成"重复的人"丢弃——它其实是同一个人的另一张证据。
    """
    src, tgt, dets, _cand, labels = _cross_scene()
    plan, _nr, _ns = entryexit_fill_plan([src, tgt], dets, labels, {16: {99: 1}})
    got = [(e["seat"], e.get("cell_index"), e.get("cell_indices"))
           for e in plan.get(16, [])]
    assert got == [(2, 2, [2, 3])], (
        f"座位 2 由 2、3 两格解引用而来，两格的帧都要留下，实得 {plan.get(16)}")


def test_cells_dets_returns_one_frame_per_cell():
    """`cells_dets` 按格位号升序各取一张，越界的格位跳过（不拿邻格顶上）。"""
    from pipeline.entryexit import cell_dets, cells_dets
    _seg, _dets, cand = _four_frame_cand(order=39, tid=148)
    assert cells_dets(cand, [3, 1]) == cell_dets(cand, 1) + cell_dets(cand, 3), (
        "必须按格位号升序、每格一张，且与 cell_dets 逐张相同（两处各切一次"
        "必然日后走样）")
    assert cells_dets(cand, [2, 9]) == cell_dets(cand, 2), "越界格位跳过"
    assert cells_dets(cand, []) == [], "没有格位就没有帧"


def test_plan_layout_text_matches_every_cell_it_carries():
    """计划项的 `layout`（送进模型的构成文案）必须与 `cell_indices` 逐格自洽：
    张数对得上、每一格的格位号都点到。

    Task 56A 时 `assemble` 每座位只送代表那一格，这条测试因此只要求文案点到
    `cell_index`。Task 56D 起多格全送，口径跟着改：文案漏掉任何一格，模型都
    会收到一张它没被告知的画面；张数写错，模型会去找一张不存在的图。两者都
    不报错、结果照常生成。

    这条测试真正防的错没变——`layout` 与格位信息出自不同的地方（前者按格位
    集合算，后者是合并多格之后的结果），对不上就是「第 3 格那张的说明配上
    第 2 格的画面」。同段与跨段两条路径都要钉。
    """
    def _check(item, where):
        cells = item["cell_indices"]
        assert item["cell_index"] == cells[0], (
            f"{where}：代表恒为号最小的那一格，实得 {item}")
        assert f"共 {len(cells)} 张" in item["layout"], (
            f"{where}：文案的张数必须等于格位数 {len(cells)}，"
            f"实得 {item['layout']!r}")
        for c in cells:
            assert str(c) in item["layout"], (
                f"{where}：第 {c} 格没在文案里点到，模型会收到一张它不知道的"
                f"画面，实得 {item['layout']!r}")

    seg, dets, cand = _four_frame_cand(order=40, tid=149)
    labels = {"40": _cells_label(
        cand,
        cells=[{"cell": 2, "role": "创业者", "seat": 2},
               {"cell": 3, "same_as": 2}],
    )}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {40: {}})
    item = plan[40][0]
    assert item["cell_indices"] == [2, 3], item
    _check(item, "同段")
    # 进场、退场两路的张数各走一个参数（n_entry / n_exit），只测一路等于另一
    # 路无人盯：变异测试实测「张数写死 1 而不跟格位数走」在只测退场时存活。
    seg_e, dets_e, cand_e = _four_frame_cand(order=42, tid=151, kind="entry")
    labels_e = {"42": _cells_label(
        cand_e,
        cells=[{"cell": 1, "role": "创业者", "seat": 2},
               {"cell": 4, "same_as": 1}],
    )}
    plan_e, _nr, _ns = entryexit_fill_plan([seg_e], dets_e, labels_e, {42: {}})
    item_e = plan_e[42][0]
    assert item_e["cell_indices"] == [1, 4], item_e
    assert "进场" in item_e["layout"], f"这是进场候选，实得 {item_e['layout']!r}"
    _check(item_e, "同段·进场")

    src, tgt, dets2, _c, labels2 = _cross_scene()
    plan2, _nr, _ns = entryexit_fill_plan([src, tgt], dets2, labels2, {16: {99: 1}})
    item2 = plan2[16][0]
    assert item2["cell_indices"] == [2, 3], item2
    _check(item2, "跨段")


def test_guessed_seat_split_piece_still_carries_its_cell(monkeypatch):
    """`BACKFILL_REQUIRE_SEAT=False`（旧的「按 cx 猜座位号」路径）下，按格位
    切出来的那一份仍要带上它自己那一格。

    这条路径不经过同座位号合并（座位号是猜的，谈不上"同一个人占两格"），
    格位号只能从候选自身取。丢了它，下游会拿整条 track 的四张证据帧去测
    颜色、拼证据图——那四张里坐着不同的人，正是切分要消灭的错误，且不报错。
    """
    monkeypatch.setattr(config, "BACKFILL_REQUIRE_SEAT", False)
    seg, dets, cand = _four_frame_cand(order=41, tid=150)
    labels = {"41": _cells_label(
        cand,
        cells=[{"cell": 2, "role": "创业者"}],      # 身份填了、座位号留空
    )}
    plan, _nr, needs_seat = entryexit_fill_plan([seg], dets, labels, {41: {}})
    assert needs_seat == [], "关掉前置条件后不该再收进 needs_seat"
    got = [(e["seat"], e.get("seat_uncertain"), e.get("cell_index"),
            e.get("cell_indices")) for e in plan.get(41, [])]
    assert got == [(1, 1, 2, [2])], (
        f"座位号是猜的（seat_uncertain=1），但格位号是用户点的第 2 格，"
        f"必须原样带下去，实得 {plan.get(41)}")


def test_cross_segment_does_not_merge_cells_of_different_candidates(capsys):
    """两个**不同候选**都把同一个目标座位标成「下一场创业者N」时，仍旧只补
    一个，绝不把两条 track 的格位并成同一个人的证据。

    与同段的 `_same_person_cells` 是同一条底线：另一条候选的「第 1 格」是在
    **它自己那张证据图**上编号的，混进来等于拿另一张图的坐标去取帧，衣着
    编码会给错的人且不报错。后到的那条照旧按「该座位已有人，不覆盖」跳过，
    并打印出来（凡有丢弃必须可见）。
    """
    src = Segment(order=15, company="源企业", start=0, end=1000,
                  presenter_num=1, presenters=())
    tgt = Segment(order=16, company="目标企业", start=1000, end=2000,
                  presenter_num=2, presenters=())
    dets = (track(135, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=15)
            + track(136, RIGHT_CX + 600, STANDING_H, 4, kind="entry", order=15))
    cands = {c["track_id"]: c for c in entryexit_candidates(dets, src, frame_w=W)}
    labels = {"15": {
        cands[135]["stable_id"]: {"role": "工作人员", "seat": None, "n_people": 2,
                                  "cells": [{"cell": 2, "role": "下一组创业者",
                                             "seat": 2}]},
        cands[136]["stable_id"]: {"role": "工作人员", "seat": None, "n_people": 2,
                                  "cells": [{"cell": 1, "role": "下一组创业者",
                                             "seat": 2}]},
    }}
    plan, _nr, _ns = entryexit_fill_plan([src, tgt], dets, labels, {16: {}})
    items = plan.get(16, [])
    assert len(items) == 1, f"同一个目标座位只该补一个人，实得 {items}"
    assert items[0]["cell_indices"] in ([1], [2]), (
        f"格位号只能来自**补上去的那一条候选自己**，两条候选的格位绝不合并，"
        f"实得 {items[0]}")
    assert len(items[0]["cell_indices"]) == 1, (
        f"每条候选各只标了一格，实得 {items[0]['cell_indices']}")
    out = capsys.readouterr().out
    # 2026-09-27 改口：后到的那条**没有被丢掉**——计划里不另占座位（不换人），
    # 但它的画面会按补格位并进目标座位的 2×2（`topup_pairs` 按来源轮流出格）。
    # 原来那句「未落位（该座位已有人，不覆盖）」读起来像标注没生效，用户据此
    # 一直当成遗留问题。
    assert "不换人" in out and "并入" in out, f"要说明画面并入了该座位：{out}"
    assert "未落位" not in out, f"它不是未落位，不许挂在「未落位」下面：{out}"
# Task 56C：座位「无合适候选」可标记
#
# 用户 2026-08-24：「如果某个座位的人没有合适的候选图，能标记下来，再抽帧
# 二次有更多的备选吗」。此前某座位在所有候选里都找不到合适画面时，用户没有
# 任何办法把这件事记下来——它只会一直显示为「未确定」，与「还没标」混作
# 一谈，下次打开页面无从分辨哪些是真找不到、哪些只是没标完。
#
# 通道与 `cell_exclusions` 同源：顶层键、`(order, seat)` 作键（座位号跨重
# 检测稳定，track_id 不是），进 LABEL_RESERVED_KEYS 并由 `_with_side_channels`
# 挂回来。
# ===========================================================================

def test_no_candidate_seats_read_from_labels():
    """`no_candidate_seats_of` 按 (order, seat) 取出人工标记的座位号。"""
    from pipeline.entryexit import no_candidate_seats_of
    labels = {"_meta": {}, "no_candidate_seats": {"8": [2, 3]}}
    assert no_candidate_seats_of(labels, 8) == {2, 3}
    assert no_candidate_seats_of(labels, 9) == set()
    assert no_candidate_seats_of(None, 8) == set()


def test_no_candidate_seats_ignore_garbage_values():
    """脏值（非数字、0、负数）一律跳过，不抛异常——手改过的文件不该让
    整条流水线崩掉，但也绝不能把 0 号座位这种不存在的东西当真。"""
    from pipeline.entryexit import no_candidate_seats_of
    labels = {"no_candidate_seats": {"8": [1, "2", None, "abc", 0, -3]}}
    assert no_candidate_seats_of(labels, 8) == {1, 2}


def test_no_candidate_seats_survive_load(tmp_path):
    """必须能从文件读回来，且不被当成段号。

    两处都要改到才算数：进 `LABEL_RESERVED_KEYS`（否则会被当成一个「段」
    展开成垃圾条目），以及被 `_with_side_channels` 挂回结果（否则读不到、
    标记静默失效——本项目在 `cell_exclusions` 上踩过同一个坑）。
    """
    import json as _json
    from pipeline.entryexit import load_entry_labels, no_candidate_seats_of

    p = tmp_path / "entry_labels_x.json"
    p.write_text(_json.dumps({
        "_meta": {"detections_fingerprint": "FP", "sheet": "x", "n_candidates": 0},
        "8": {"entry@1#2,3": {"role": "创业者", "seat": 1}},
        "no_candidate_seats": {"8": [2]},
    }, ensure_ascii=False), encoding="utf-8")

    labels = load_entry_labels(p, detections_fp="FP")
    assert no_candidate_seats_of(labels, 8) == {2}
    assert "no_candidate_seats" not in [k for k in labels if k.isdigit()]


def test_no_candidate_seats_survive_fingerprint_remap(tmp_path, capsys):
    """指纹不符、走模糊重对齐那条路时，`(order, seat)` 那几个旁路通道
    **必须一并保留**。

    座位号是人工确定的、跨重检测稳定，凭什么因为 track 编号变了就把它扔掉？
    加密抽帧（Task 56C 第三部分）之后必然重检测、指纹必然改变，若这里丢了，
    用户刚标的「这个座位没画面」在重检测后静默消失——而这正是他标它的原因。
    """
    import json as _json
    from pipeline.entryexit import (
        cell_exclusions_of, entryexit_candidates, load_entry_labels,
        no_candidate_seats_of,
    )
    from pipeline.detect import Detection
    from pipeline.schedule import Segment

    seg = Segment(order=8, company="甲", start=0, end=1000,
                  presenter_num=3, presenters=())
    dets = [Detection(8, "entry", float(i), "f.jpg", 7,
                      800.0, 400.0, 900.0, 1000.0, 0.9) for i in range(6)]
    cands = entryexit_candidates(dets, seg)
    assert cands, "夹具前提：这一段要有候选，否则匹配率为 0、整份作废"
    sid = cands[0]["stable_id"]

    p = tmp_path / "entry_labels_y.json"
    p.write_text(_json.dumps({
        "_meta": {"detections_fingerprint": "OLD-FP", "sheet": "y",
                  "n_candidates": 1},
        "8": {sid: {"role": "创业者", "seat": 1, "n_people": 1, "note": ""}},
        "cell_exclusions": {"8": {"1": [2]}},
        "no_candidate_seats": {"8": [3]},
    }, ensure_ascii=False), encoding="utf-8")

    labels = load_entry_labels(p, detections_fp="NEW-FP",
                               candidates_by_order={8: cands})
    assert labels, f"匹配率应达标、不该整份作废：{capsys.readouterr().out}"
    assert no_candidate_seats_of(labels, 8) == {3}, (
        "重检测后「本座位无可用画面」必须还在——它的键是 (order, seat)，"
        "与 track 编号无关")
    assert cell_exclusions_of(labels, 8, 1) == {2}, (
        "逐格排除同理：键是 (order, seat)，重对齐不该把它一起丢掉")


# --- 候选层直接标「下一组创业者 + 座位号」也要流转（用户 2026-08-24 实测）---

def test_whole_candidate_marked_next_group_flows_with_its_seat():
    """整个候选标成「下一组创业者」并给了座位号，要流转到下一段。

    用户实测（新能源2组 order 2）：「选择下一组创业者后，目前还是无法指定
    是下一组创业者的几号位」。页面上没有座位号控件是一半原因，另一半在这里
    ——跨段流转此前**只认格位备注那条路**：候选层的角色闸门写着
    `if role != "创业者": continue`，标成「下一组/上一组创业者」的整条候选
    连同它的座位号一起被丢掉，一声不吭。

    这条路径的场景与格位备注那条不同：镜头里只有一个人，他属于下一家企业，
    没什么可切分的，用户就是想整条指过去。
    """
    src = Segment(order=15, company="源企业", start=0, end=1000,
                  presenter_num=1, presenters=())
    tgt = Segment(order=16, company="目标企业", start=1000, end=2000,
                  presenter_num=2, presenters=())
    dets = track(136, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=15)
    cand = next(c for c in entryexit_candidates(dets, src, frame_w=W)
                if c["track_id"] == 136)
    labels = {"15": {cand["stable_id"]: {
        "role": "下一组创业者", "seat": 2, "n_people": 1, "note": ""}}}

    plan, _nr, _ns = entryexit_fill_plan([src, tgt], dets, labels, {16: {99: 1}})

    assert plan.get(15, []) == [], (
        f"标的是下一场的人，不该补进本段，实得 {plan.get(15)}")
    got = plan.get(16, [])
    assert len(got) == 1, f"应流转成段 16 的一条计划项，实得 {got}"
    e = got[0]
    assert e["seat"] == 2 and e.get("cross_segment") == 1 and e.get("src_order") == 15, e
    assert not e.get("cell_index"), (
        f"整条候选流转，没有格位可言——凭空带上格位号会让下游只取四格里的"
        f"一格，另外三张证据凭空消失，实得 {e}")


def test_previous_group_presenter_flows_backwards_too():
    """「上一组创业者」同理，流向 order-1。方向写死成 +1 是最容易犯的错。"""
    prv = Segment(order=15, company="上一家", start=0, end=1000,
                  presenter_num=2, presenters=())
    cur = Segment(order=16, company="本家", start=1000, end=2000,
                  presenter_num=1, presenters=())
    dets = track(137, RIGHT_CX + 200, STANDING_H, 4, kind="entry", order=16)
    cand = next(c for c in entryexit_candidates(dets, cur, frame_w=W)
                if c["track_id"] == 137)
    labels = {"16": {cand["stable_id"]: {
        "role": "上一组创业者", "seat": 2, "n_people": 1, "note": ""}}}

    plan, _nr, _ns = entryexit_fill_plan([prv, cur], dets, labels, {15: {99: 1}})
    assert [e["seat"] for e in plan.get(15, [])] == [2], (
        f"「上一组创业者」应流向 order-1，实得 {plan}")
    assert plan.get(16, []) == []


def test_next_group_candidate_without_seat_is_reported_not_guessed():
    """标了「下一组创业者」却没给座位号：**不猜**，进 needs_seat 报出来。

    按 median_cx 猜座位号是几何推断，本项目已连续失败四次。猜错的后果是把
    这个人的衣着编进下一家企业另一个人的行里，而流水线照常跑完、不报错。
    """
    src = Segment(order=15, company="源企业", start=0, end=1000,
                  presenter_num=1, presenters=())
    tgt = Segment(order=16, company="目标企业", start=1000, end=2000,
                  presenter_num=2, presenters=())
    dets = track(138, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=15)
    cand = next(c for c in entryexit_candidates(dets, src, frame_w=W)
                if c["track_id"] == 138)
    labels = {"15": {cand["stable_id"]: {
        "role": "下一组创业者", "seat": None, "n_people": 1, "note": ""}}}

    plan, _nr, needs_seat = entryexit_fill_plan(
        [src, tgt], dets, labels, {16: {99: 1}})
    assert plan.get(16, []) == [], f"没有座位号就不许落位，实得 {plan.get(16)}"
    assert any(x["track_id"] == 138 for x in needs_seat), (
        f"要报进 needs_seat 让用户去补座位号，实得 {needs_seat}")


# --- 就座图被勾掉格位后，用人工标定的进出场候选补满（用户 2026-08-25）---

def test_labelled_seat_candidates_ignores_occupancy():
    """返回「用户明确标成创业者+座位号」的进出场候选，**不管该座位是否已有人**。

    `entryexit_fill_plan` 里那条「不覆盖已占座位」是对的——自动递补不该顶掉
    就座结果。但补格位是另一回事：座位上的人没变，只是他那张 2×2 被勾掉了
    一格，需要同一个人的另一张画面顶上。所以这里必须绕开占用判断。
    """
    seg = mkseg(order=8, presenter_num=2)
    dets = (track(90, RIGHT_CX + 300, STANDING_H, 6, kind="seated", order=8)
            + track(91, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=8))
    cand = next(c for c in entryexit_candidates(dets, seg, frame_w=W)
                if c["track_id"] == 91)
    labels = {"8": {cand["stable_id"]: {"role": "创业者", "seat": 1}}}

    got = entryexit.labelled_seat_candidates(dets, seg, labels)
    assert set(got) == {1}, f"座位 1 有人工标定的进出场候选，实得 {sorted(got)}"
    assert len(got[1]) == 1, f"只标了一条，实得 {len(got[1])} 条"
    cand_got, cells_got = got[1][0]
    assert cand_got["track_id"] == 91, cand_got
    assert cells_got == [], f"候选层直接标的用空列表表示整条候选，实得 {cells_got}"


def test_labelled_seat_candidates_reads_cells_too():
    """格位分配里标成「创业者 + 座位号」的那几格同样算数。

    多人框按格位切分之后，每一格才是一个人；只看候选层的 role/seat 会把这
    整类漏掉，而它恰恰是本项目最常见的标法。
    """
    seg, dets, cand = _four_frame_cand(order=9, tid=92)
    labels = {"9": _cells_label(
        cand, cells=[{"cell": 2, "role": "创业者", "seat": 2}])}
    got = entryexit.labelled_seat_candidates(dets, seg, labels)
    assert set(got) == {2}, f"实得 {sorted(got)}"


def test_labelled_seat_candidates_skips_non_presenter_roles():
    """工作人员/专家/上下一组创业者一概不收——拿他们的画面去补格位，
    等于把别人的衣着编进这位创业者的行里，且不报错。"""
    seg = mkseg(order=8, presenter_num=2)
    dets = (track(90, RIGHT_CX + 300, STANDING_H, 6, kind="seated", order=8)
            + track(91, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=8))
    cand = next(c for c in entryexit_candidates(dets, seg, frame_w=W)
                if c["track_id"] == 91)
    for role in ("工作人员", "专家", "不确定", "下一组创业者"):
        labels = {"8": {cand["stable_id"]: {"role": role, "seat": 1}}}
        assert entryexit.labelled_seat_candidates(dets, seg, labels) == {}, role


def test_topup_frames_fills_the_sheet_back_to_four():
    """勾掉两格之后，用那个人的进出场帧补回四张。"""
    seg = mkseg(order=8, presenter_num=2)
    dets = (track(90, RIGHT_CX + 300, STANDING_H, 6, kind="seated", order=8)
            + track(91, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=8))
    cand = next(c for c in entryexit_candidates(dets, seg, frame_w=W)
                if c["track_id"] == 91)
    seated = [d for d in dets if d.kind == "seated"][:2]     # 勾剩两张
    got = entryexit.topup_frames(seated, cand, want=4)
    assert len(got) == 4, f"两张就座 + 两张进出场 = 四张，实得 {len(got)}"
    assert got[:2] == seated, "就座那几张必须排在前面且原样保留"
    assert all(d.kind == "exit" for d in got[2:]), got[2:]


def test_topup_frames_never_exceeds_the_cap_and_is_a_noop_when_full():
    """已经满四张时一张不加；补不满也不报错，有几张算几张。"""
    seg = mkseg(order=8, presenter_num=2)
    dets = (track(90, RIGHT_CX + 300, STANDING_H, 6, kind="seated", order=8)
            + track(91, RIGHT_CX + 200, STANDING_H, 1, kind="exit", order=8))
    cand = next(c for c in entryexit_candidates(dets, seg, frame_w=W)
                if c["track_id"] == 91)
    seated4 = [d for d in dets if d.kind == "seated"][:4]
    # 2026-09-26 起「满了就别动」不再成立：有别的人工来源时，就座那四张相近
    # 的要让出格子（用户：自动识别出的 2×2 可能高度雷同，看情况只保留 1-2 个，
    # 与其他不同画面合并）。这一条来源只有 1 张，所以是 3 + 1。
    full = entryexit.topup_frames(seated4, cand, want=4)
    assert len(full) == 4, f"不超过 want，实得 {len(full)}"
    assert sum(d.kind == "exit" for d in full) == 1, (
        f"满了也要让出一格给另一个来源，实得 {[d.kind for d in full]}")
    assert full[0] == seated4[0], "第一个来源仍排在最前"
    one = seated4[:1]
    got = entryexit.topup_frames(one, cand, want=4)
    assert len(got) == 2, f"该候选只有 1 张可补，补不满不报错，实得 {len(got)}"
    assert entryexit.topup_frames(one, None, want=4) == one, "没有候选就原样返回"


# --- 折叠区（被过滤）的候选也能标（用户 2026-08-25：order4 那个人就在里面）---

def test_candidates_for_labels_includes_the_filtered_ones():
    """标注配对要认得**被过滤掉的**候选。

    用户实测城市2 order 4：密抽之后那个人确实在候选里，只是框高占比 0.096~
    0.12（阈值 0.25）被挡进了折叠区。折叠区当初的设定是「只读，看一眼确认没
    误挡」——前提是被挡的都是手或太小认不出。现在的场景恰恰相反：真人就在
    里面，而用户没有任何办法把他标上。

    光让页面能标是不够的：`entryexit_fill_plan` 只在**展示候选**里找标注，
    标在折叠区的那条配不上任何候选，会被静默忽略——用户标了、页面绿了、
    人一个也没补进去，且不报错。
    """
    seg = mkseg(order=12, presenter_num=2)
    dets = (track(60, RIGHT_CX + 200, STANDING_H, 6, kind="exit", order=12)
            # 远处的小人：框高占比远低于阈值，必被挡进折叠区
            + track(61, RIGHT_CX + 500, STANDING_H * 0.12, 8, kind="exit",
                    order=12, width=12.0))
    kept = entryexit_candidates(dets, seg, frame_w=W)
    filt = entryexit.entryexit_filtered_candidates(dets, seg, frame_w=W)
    assert kept and filt, f"夹具前提：要一留一挡，实得 留{len(kept)} 挡{len(filt)}"

    both = entryexit.candidates_for_labels(dets, seg, frame_w=W)
    sids = {c["stable_id"] for c in both}
    assert sids == ({c["stable_id"] for c in kept} | {c["stable_id"] for c in filt}), (
        f"应是展示候选 ∪ 折叠区候选，实得 {len(both)} 个")
    assert len(both) == len(sids), "同一个候选不许出现两次"


def test_fill_plan_honours_a_label_on_a_filtered_candidate():
    """标在折叠区候选上的「创业者 + 座位号」照样补位。"""
    seg = mkseg(order=12, presenter_num=2)
    dets = (track(60, RIGHT_CX + 200, STANDING_H, 6, kind="exit", order=12)
            + track(61, RIGHT_CX + 500, STANDING_H * 0.12, 8, kind="exit",
                    order=12, width=12.0))
    filt = entryexit.entryexit_filtered_candidates(dets, seg, frame_w=W)
    labels = {"12": {filt[0]["stable_id"]: {"role": "创业者", "seat": 2}}}

    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {12: {}})
    got = [(e["seat"], e["track_id"]) for e in plan.get(12, [])]
    assert (2, filt[0]["track_id"]) in got, (
        f"用户在折叠区里认出并标定的人必须补得进去，实得 {got}")


def test_unlabelled_filtered_candidates_still_do_not_backfill():
    """没标过的折叠区候选**不会**自己混进补位——把它们纳入配对只是为了
    「用户标了就认」，不是把过滤规则架空。"""
    seg = mkseg(order=12, presenter_num=2)
    dets = (track(60, RIGHT_CX + 200, STANDING_H, 6, kind="exit", order=12)
            + track(61, RIGHT_CX + 500, STANDING_H * 0.12, 8, kind="exit",
                    order=12, width=12.0))
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, {}, {12: {}})
    assert plan.get(12, []) == [], f"一条标注都没有就一个也不补，实得 {plan}"


def test_candidates_for_labels_is_not_capped_by_the_display_limit():
    """配对池不套展示上限 `ENTRYEXIT_TOP_N`。

    那个上限是「页面一次看得过来多少个」，不是判据。拿它筛配对池，第 9 个
    之后的候选就既不在展示里、也不在折叠区——重检测之后某条标注一旦滑出前
    8，配不上任何候选、静默消失，用户只会发现「我标过的人不见了」。
    """
    seg = mkseg(order=13, presenter_num=2)
    dets = []
    for i in range(config.ENTRYEXIT_TOP_N + 4):
        dets += track(200 + i, RIGHT_CX + 60 * i, STANDING_H * (1 - 0.01 * i),
                      4, kind="exit", order=13)
    shown = [c for c in entryexit_candidates(dets, seg, frame_w=W)
             if c["kind"] == "exit"]
    pool = [c for c in entryexit.candidates_for_labels(dets, seg, frame_w=W)
            if c["kind"] == "exit"]
    assert len(shown) == config.ENTRYEXIT_TOP_N, (
        f"夹具前提：展示要真的被截断到 {config.ENTRYEXIT_TOP_N}，实得 {len(shown)}")
    assert len(pool) > len(shown), (
        f"配对池必须含被截断掉的那几个，实得 池{len(pool)} 展示{len(shown)}")
    assert len({c["stable_id"] for c in pool}) == len(pool), "不许有重复"


def test_labelled_seat_candidates_returns_the_cells_of_that_seat():
    """补帧候选表要带上**这个座位占了哪几格**，不是整条候选。

    此前存的是整条候选，`topup_frames` 于是按 `best_dets` 的顺序往上补——
    而多人框的 best_dets 里混着**别人的格**。补进去就是把另一个人的衣着编进
    这位创业者的行里，且一处不报错，正是本项目最该避免的那类错。

    用户 2026-08-25：「凡是标注的图是不是都可以利用，如果标注写了同格位X 的，
    和自动标注的图合并到一起，用于 VLM，最大化利用标注的图」——「同格位X」正是
    同一个人占两格，这两格都该并进来，而别人的格一格都不能进。
    """
    seg, dets, cand = _four_frame_cand(order=14, tid=93)
    labels = {"14": _cells_label(cand, cells=[
        {"cell": 1, "role": "创业者", "seat": 2},
        {"cell": 3, "same_as": 1},                    # 同一个人的第二格
        {"cell": 2, "role": "创业者", "seat": 3},      # 另一个人
    ])}
    got = entryexit.labelled_seat_candidates(dets, seg, labels)
    assert set(got) == {2, 3}, f"两个座位各有标注，实得 {sorted(got)}"
    (c2, cells2), = got[2]
    assert cells2 == [1, 3], f"座位 2 占第 1、3 格（含「同格位1」），实得 {cells2}"
    (c3, cells3), = got[3]
    assert cells3 == [2], f"座位 3 只占第 2 格，实得 {cells3}"
    assert c2["stable_id"] == c3["stable_id"] == cand["stable_id"]


def test_labelled_seat_candidates_uses_empty_cells_for_whole_candidate():
    """候选层直接标的（没有格位分配）用空列表表示「整条候选都是他」。"""
    seg = mkseg(order=15, presenter_num=2)
    dets = (track(94, RIGHT_CX + 300, STANDING_H, 6, kind="seated", order=15)
            + track(95, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=15))
    cand = next(c for c in entryexit_candidates(dets, seg, frame_w=W)
                if c["track_id"] == 95)
    labels = {"15": {cand["stable_id"]: {"role": "创业者", "seat": 1}}}
    got = entryexit.labelled_seat_candidates(dets, seg, labels)
    assert got[1][0][1] == [], f"整条候选用空列表表示，实得 {got[1][0][1]}"


def test_topup_uses_only_the_cells_of_that_seat():
    """补帧只取**这个座位自己那几格**，绝不把别人的格补进来。"""
    seg, dets, cand = _four_frame_cand(order=16, tid=96)
    labels = {"16": _cells_label(cand, cells=[
        {"cell": 1, "role": "创业者", "seat": 2},
        {"cell": 3, "same_as": 1},
        {"cell": 2, "role": "创业者", "seat": 3},
    ])}
    spec = entryexit.labelled_seat_candidates(dets, seg, labels)[2]
    seated = [entryexit.cell_dets(cand, 1)[0]]      # 假装就座只剩一张（＝第1格）
    got = entryexit.topup_frames(seated, spec, want=4)
    assert got == entryexit.cells_dets(cand, [1, 3]), (
        f"结果该是第 1、3 格那两张：原有的第 1 格 + 补上的第 3 格。"
        f"第 1 格已在手上，去重后不重复补，实得 {got}")
    assert entryexit.cell_dets(cand, 2)[0] not in got, (
        "第 2 格是另一个人，一格都不许补进来")


def test_cross_skip_is_silent_when_the_target_is_merely_outside_the_window():
    """目标段只是**不在本次传入的窗口里**时，不许报「没有第 N 段」。

    用户 2026-08-25 实测新能源2：日志里一条「第 5 段 → 第 6 段座位 3（没有第
    6 段）」，而第 6 段明明存在、那条流转也**成功落位了**（登记表里座位 3 的
    cross_segment=1、src_order=5）。

    噪音的来源：`registry` 逐段跑时传的是 order-1/order/order+1 三段窗口，
    处理第 4 段时窗口是 [3,4,5]，第 5 段的标注产生一条指向第 6 段的流转，而
    第 6 段不在这个窗口里——于是报了一句「没有第 6 段」。等窗口挪到 [5,6,7]
    时它落位了，但那一句已经印出去了。

    假警报比不报更坏：用户会去查一个根本不存在的问题，久了就不再信任告警。
    调用方传 `known_orders`（整组的段号）之后，「不在窗口里」与「真的没有这
    一段」就分得开了。
    """
    src = Segment(order=5, company="源企业", start=0, end=1000,
                  presenter_num=1, presenters=())
    dets = track(150, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=5)
    cand = next(c for c in entryexit_candidates(dets, src, frame_w=W)
                if c["track_id"] == 150)
    labels = {"5": {cand["stable_id"]: {
        "role": "下一组创业者", "seat": 2, "n_people": 1, "note": ""}}}

    import io
    from contextlib import redirect_stdout

    # 只传第 5 段（模拟窗口没覆盖到目标段），但告诉它整组有 1..16 段
    buf = io.StringIO()
    with redirect_stdout(buf):
        entryexit_fill_plan([src], dets, labels, {}, known_orders=range(1, 17))
    assert "没有第 6 段" not in buf.getvalue(), (
        f"第 6 段存在、只是不在本次窗口里，不该报没有它：{buf.getvalue()!r}")

    # 真的没有第 6 段时照旧要报——这条告警本身是有用的
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        entryexit_fill_plan([src], dets, labels, {}, known_orders=[5])
    assert "没有第 6 段" in buf2.getvalue(), (
        f"整组确实只有第 5 段时必须报出来：{buf2.getvalue()!r}")


def test_cross_skip_defaults_to_the_passed_segments():
    """不传 `known_orders` 时行为一字不变（按传入的那几段判断）。"""
    src = Segment(order=5, company="源企业", start=0, end=1000,
                  presenter_num=1, presenters=())
    dets = track(151, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=5)
    cand = next(c for c in entryexit_candidates(dets, src, frame_w=W)
                if c["track_id"] == 151)
    labels = {"5": {cand["stable_id"]: {
        "role": "下一组创业者", "seat": 2, "n_people": 1, "note": ""}}}
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        entryexit_fill_plan([src], dets, labels, {})
    assert "没有第 6 段" in buf.getvalue(), buf.getvalue()


# --- Task 56T：同一座位的**多条**人工标注要全部并进来 -----------------------
# 用户 2026-08-26 实测文化4组 order 3：「单帧可能只选了一个窗格，而在多帧分别
# 选的窗格没有拼在一起，应该根据标注尽可能全的拼成 2×2 四张图」。该段座位 2
# 在 exit@3364 标了两格、又在 exit@3366 标了一格，登记表却只留下前者那两张；
# 座位 3 在 entry@2502 整条标了他、又在 exit@3366 标了一格，登记表只剩一张。

def _two_labelled_cands(order, tid_a, tid_b):
    """同一段里两条互不相同的候选（一 exit 一 entry），各有 4 张证据帧。"""
    dets = (track(tid_a, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=order)
            + track(tid_b, RIGHT_CX + 320, STANDING_H, 4, kind="entry", order=order))
    seg = mkseg(order=order, presenter_num=3)
    cands = entryexit_candidates(dets, seg, frame_w=W)
    a = next(c for c in cands if c["track_id"] == tid_a and c["kind"] == "exit")
    b = next(c for c in cands if c["track_id"] == tid_b and c["kind"] == "entry")
    assert len(a["best_dets"]) == len(b["best_dets"]) == 4, "夹具前提：两条各 4 张"
    assert not (set(a["best_dets"]) & set(b["best_dets"])), "夹具前提：两条不共帧"
    return seg, dets, a, b


def test_labelled_seat_candidates_keeps_every_candidate_of_a_seat():
    """一个座位在**多条**候选上都被标过时，每一条都要留下。

    此前是 `setdefault`——只留最先遇到的那一条，其余连同用户亲手标的格位一起
    静默消失。用户看得见自己标过，交付包里却没有那几张，且一处不报错。
    """
    seg, dets, a, b = _two_labelled_cands(order=40, tid_a=200, tid_b=201)
    labels = {"40": {
        a["stable_id"]: {"role": "工作人员", "seat": None, "n_people": 2,
                         "cells": [{"cell": 1, "role": "创业者", "seat": 2}]},
        b["stable_id"]: {"role": "创业者", "seat": 2, "n_people": 1},
    }}
    got = entryexit.labelled_seat_candidates(dets, seg, labels)
    assert set(got) == {2}, f"只有座位 2 被标过，实得 {sorted(got)}"
    specs = got[2]
    assert isinstance(specs, list), f"每个座位存一**串**候选，实得 {type(specs)}"
    assert [(s[0]["stable_id"], s[1]) for s in specs] == [
        (b["stable_id"], []), (a["stable_id"], [1])], (
        f"两条都要在，且按 `candidates_for_labels` 的顺序（entry 在前）排，实得 "
        f"{[(s[0]['stable_id'], s[1]) for s in specs]}")


def test_topup_draws_from_every_labelled_candidate_of_that_seat():
    """补帧要把该座位**全部**标注候选的画面都拿来，直到补满 want 张。

    **逐条轮流取**，不是把第一条抽干再看第二条：只有 4 格，而 b 一条就有 4 张
    ——按顺序取会让 a 那一格（用户亲手标的另一个时刻的画面）整条落空，换来的
    是 b 里第 4 张几乎重复的画面。轮流取则每一条标注都至少露一面。
    """
    seg, dets, a, b = _two_labelled_cands(order=41, tid_a=202, tid_b=203)
    labels = {"41": {
        a["stable_id"]: {"role": "工作人员", "seat": None, "n_people": 2,
                         "cells": [{"cell": 1, "role": "创业者", "seat": 2}]},
        b["stable_id"]: {"role": "创业者", "seat": 2, "n_people": 1},
    }}
    specs = entryexit.labelled_seat_candidates(dets, seg, labels)[2]
    got = entryexit.topup_frames([], specs, want=4)
    assert entryexit.cell_dets(a, 1)[0] in got, (
        "a 只有一格，被 b 的四张挤掉就等于用户白标了——轮流取才不会")
    # b 出三格（自己第一张 + 时间上离得最远的两张）、a 出一格；同一来源挨着放
    # （2026-09-26：同源第二格起挑最不一样的那张，选完按来源排回一起）。
    assert len(got) == 4, got
    assert got[0] == b["best_dets"][0], "按标注顺序第一条的第一张排最前"
    assert got[-1] == entryexit.cell_dets(a, 1)[0], "a 那一格挨着排在它自己那组"
    assert set(got[:3]) <= set(b["best_dets"]) and len(set(got[:3])) == 3, got


def test_topup_still_takes_a_single_spec():
    """单条候选（非列表）照旧能用——两个调用方各传各的，不许只支持一种。"""
    seg, dets, cand = _four_frame_cand(order=42, tid=204)
    labels = {"42": _cells_label(cand, cells=[
        {"cell": 1, "role": "创业者", "seat": 2},
        {"cell": 3, "same_as": 1},
    ])}
    spec = entryexit.labelled_seat_candidates(dets, seg, labels)[2][0]
    assert entryexit.topup_frames([], spec, want=4) == entryexit.cells_dets(cand, [1, 3])


def test_topup_pairs_carries_the_cell_number_of_each_frame():
    """补帧要连同**它在自己那张证据图上是第几格**一起带出来。

    登记表的 `cell_indices` 与 `frames` 是逐项一一对应的（Task 56A）。补进来
    的帧若没有格位号，两者就此错位——「第几格」会指到另一张帧上，且不报错。
    """
    seg, dets, a, b = _two_labelled_cands(order=43, tid_a=205, tid_b=206)
    labels = {"43": {
        a["stable_id"]: {"role": "工作人员", "seat": None, "n_people": 2,
                         "cells": [{"cell": 2, "role": "创业者", "seat": 1}]},
        b["stable_id"]: {"role": "创业者", "seat": 1, "n_people": 1},
    }}
    specs = entryexit.labelled_seat_candidates(dets, seg, labels)[1]
    pairs = entryexit.topup_pairs([], specs, want=3)
    # 这条测试要守的是「格位号跟着自己那一帧走」，不是具体挑了哪几张：
    # 每一对的格位号必须等于那一帧在它**自己那张**证据图上的位置。
    for c, d in pairs:
        if d in b["best_dets"]:
            assert b["best_dets"].index(d) + 1 == c, (c, d.t)
        else:
            assert d == entryexit.cell_dets(a, 2)[0] and c == 2, (c, d.t)
    assert entryexit.cell_dets(a, 2)[0] in [d for _c, d in pairs], "a 那一格要在"


def test_topup_pairs_never_repeats_a_frame_already_in_hand():
    """已经在手上的那张不再补一遍——重复的图白占一格。"""
    seg, dets, a, b = _two_labelled_cands(order=44, tid_a=207, tid_b=208)
    labels = {"44": {a["stable_id"]: {"role": "创业者", "seat": 1, "n_people": 1}}}
    specs = entryexit.labelled_seat_candidates(dets, seg, labels)[1]
    have = [(1, a["best_dets"][0])]
    pairs = entryexit.topup_pairs(have, specs, want=4)
    got = [d for _c, d in pairs]
    assert len(pairs) == 4, f"四张互不相同，实得 {len(pairs)}"
    assert len(set(got)) == 4, "同一张不许出现两次"
    assert set(got) == set(a["best_dets"][:4]) and got[0] == a["best_dets"][0], got


def test_two_whole_candidate_labels_on_one_seat_both_survive():
    """两条都在**候选层**直接标给同一个座位时，两条都要留下。

    与上面那条的区别是执行次序：`candidates_for_labels` 先 entry 后 exit，上面
    的夹具里候选层那一支恰好排在第一位，先建表、后面走格位分支的照样能 append
    ——两条都在，看不出这一支自己有没有丢东西。这里让它排**第二**，它必须往
    已有的那一串上追加，而不是另起一串把先来的那条盖掉。
    """
    seg, dets, a, b = _two_labelled_cands(order=45, tid_a=209, tid_b=210)
    labels = {"45": {
        b["stable_id"]: {"role": "创业者", "seat": 1, "n_people": 1},   # entry，先
        a["stable_id"]: {"role": "创业者", "seat": 1, "n_people": 1},   # exit，后
    }}
    got = entryexit.labelled_seat_candidates(dets, seg, labels)
    assert [s[0]["stable_id"] for s in got[1]] == [b["stable_id"], a["stable_id"]], (
        f"两条候选层标注都要在，实得 {[s[0]['stable_id'] for s in got[1]]}")


# --- Task 56V：已定座位改号（用户 2026-08-29 实测城市2 order 6）-------------
# 「自动标的 seat1 实际为 seat2，这样自动标错的如何调整，否则标错了，无法将
# 该图用于其他座位」。进出场候选上的座位号本来就随手指；**就座那一路**的图
# 此前只能整体否决，要改号得绕一圈去候选里重新指认同一个人——那个人恰好没有
# 好用的进出场候选时，一张本来很好的就座图就此浪费。

def test_seat_reassign_of_reads_the_mapping():
    """`seat_reassign` 是 `{段: {原座位号: 新座位号}}`，键与 cell_exclusions 同源。"""
    labels = {"seat_reassign": {"6": {"1": 2, "3": "1"}}}
    assert entryexit.seat_reassign_of(labels, 6) == {1: 2, 3: 1}, (
        "字符串型的座位号也要收（JSON 的对象键只能是字符串）")
    assert entryexit.seat_reassign_of(labels, 7) == {}
    assert entryexit.seat_reassign_of(None, 6) == {}


def test_seat_reassign_of_skips_dirty_and_self_mappings():
    """脏值与「改成自己」一律跳过，不让手改过的文件把整条流水线带偏。

    「改成自己」尤其要跳：它是用户点开下拉又选回原值的痕迹，留着会让下游
    以为这个座位被人工处理过（与 `no_candidate_seats` 跳 0/负数同一条理由）。
    """
    labels = {"seat_reassign": {"6": {
        "1": 1,          # 改成自己
        "2": 0,          # 不存在的座位号
        "3": -1,
        "4": None,
        "5": "甲",
        "x": 2,          # 原座位号本身就不是数字
        "7": 3,          # 唯一有效的一条
    }}}
    assert entryexit.seat_reassign_of(labels, 6) == {7: 3}


def test_seat_reassign_of_drops_collisions_onto_the_same_target():
    """两个座位同时改到同一个目标：整段作废并报错，绝不静默留一个。

    留一个的话「留哪一个」取决于字典顺序——用户看到的结果会随他标注的先后
    次序变，且一处不报错。宁可整段不生效，让他自己去改清楚。
    """
    labels = {"seat_reassign": {"6": {"1": 3, "2": 3}}}
    with pytest.raises(ValueError, match="3"):
        entryexit.seat_reassign_of(labels, 6)


def test_load_entry_labels_carries_seat_reassign_through(tmp_path):
    """`seat_reassign` 必须原样带过载入这一关——三条路径都要。

    用户 2026-08-29 实测城市2 order 6：他在页面上把两张就座图改了号，导出的
    JSON 里 `seat_reassign` 明明写着，跑下来却毫无变化。根因是
    `_with_side_channels` 加了参数、**三个调用点一个都没传**，载入时静默丢掉。

    这类「只进 LABEL_RESERVED_KEYS、不挂回来」的漏法是本项目最难发现的一种：
    文件里明明有、页面上明明标了、跑下来毫无效果且一处不报错。
    """
    import json

    def _write(fp):
        p = tmp_path / f"entry_labels_{fp}.json"
        p.write_text(json.dumps({
            "_meta": {"detections_fingerprint": "FP-A", "sheet": "组",
                      "n_candidates": 0},
            "seat_reassign": {"6": {"1": 2}},
            "cell_exclusions": {"6": {"1": [2]}},
        }, ensure_ascii=False), encoding="utf-8")
        return p

    # ① 调用方没给指纹
    got = entryexit.load_entry_labels(_write("a"), detections_fp=None)
    assert got.get("seat_reassign") == {"6": {"1": 2}}, (
        f"跳过校验那条路径丢了改号，实得 {got.get('seat_reassign')}")
    # ② 指纹一致
    got = entryexit.load_entry_labels(_write("b"), detections_fp="FP-A")
    assert got.get("seat_reassign") == {"6": {"1": 2}}, (
        f"指纹一致那条路径丢了改号，实得 {got.get('seat_reassign')}")
    # 顺带钉住同一层的另外几个通道没被这次改动带歪
    assert got.get("cell_exclusions") == {"6": {"1": [2]}}


def test_realigned_labels_still_carry_seat_reassign(tmp_path):
    """**重对齐**那条路也要带过 `seat_reassign`（第三条载入路径）。

    加密抽帧（Task 56C）之后必然重检测、指纹必然改变，走的正是这条路。改号
    的键是 `(段, 原座位号)`——座位号人工确定、跨重检测稳定，凭什么因为 track
    编号变了就一起扔掉？丢在这里的表现是「二次抽帧之后我改的号全没了」，而
    用户根本不会想到去这里找。

    上一版只测了前两条路径，把这一条的还原变异漏了过去（变异测试实测存活）。
    """
    import json

    seg = mkseg(order=6, presenter_num=2)
    dets = track(77, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=6)
    cand = next(c for c in entryexit_candidates(dets, seg, frame_w=W)
                if c["track_id"] == 77)
    p = tmp_path / "entry_labels_组.json"
    p.write_text(json.dumps({
        "_meta": {"detections_fingerprint": "FP-旧", "sheet": "组",
                  "n_candidates": 1},
        "6": {cand["stable_id"]: {"role": "创业者", "seat": 1}},
        "seat_reassign": {"6": {"1": 2}},
    }, ensure_ascii=False), encoding="utf-8")

    got = entryexit.load_entry_labels(
        p, detections_fp="FP-新", candidates_by_order={6: [cand]})
    assert got.get("6"), "夹具前提：这条标注要能重新对齐上，否则测的是整份作废"
    assert got.get("seat_reassign") == {"6": {"1": 2}}, (
        f"重对齐那条路丢了改号，实得 {got.get('seat_reassign')}")


# ---------------------------------------------------------------------------
# 「本座位无可用画面」旁边的重抽时间戳（用户 2026-09-09：「标记为『座位无可用
# 画面』时，给出一个 note 窗口，填写该座位重抽的时间戳，供重抽参考」）。
#
# 之前只能记下「这个座位找不到人」，加密抽帧只好按预设窗口整段再抽一遍；而
# 用户往往**眼睛已经看到那个人在第几分钟出现了**，只是那一带没被抽到。把那个
# 时刻记下来，加密就能直接对着它抽（`windows.spot_window` 早就为这件事写好了，
# 一直只能靠命令行 `--at` 手输）。
# ---------------------------------------------------------------------------

def test_parse_timestamp_accepts_seconds_and_clock_forms():
    from pipeline.entryexit import parse_timestamp
    assert parse_timestamp("1234.5") == 1234.5
    assert parse_timestamp("1234") == 1234.0
    assert parse_timestamp("20:34") == 20 * 60 + 34
    assert parse_timestamp("1:02:03") == 3600 + 2 * 60 + 3
    assert parse_timestamp(" 3:07 ") == 187.0
    assert parse_timestamp("20:34.5") == 20 * 60 + 34.5


def test_parse_timestamp_refuses_junk_instead_of_guessing():
    """猜不出来就返回 None，绝不猜一个数。

    按错的时刻抽出来的是**别家企业**的画面，而它会被写进本段目录，此后衣着/
    颜色/性别全部取自隔壁公司，一处不报错。
    """
    from pipeline.entryexit import parse_timestamp
    for junk in ("", "   ", "大概二十分钟", "20分34秒", "abc", None,
                 "1:2:3:4", "-5", "20:99"):
        assert parse_timestamp(junk) is None, junk


def test_no_candidate_notes_are_read_per_seat():
    from pipeline.entryexit import no_candidate_notes_of
    labels = {"no_candidate_notes": {"13": {
        "2": {"at": "20:34", "text": "他在这一带从右边走过"},
        "3": {"at": "", "text": "只写了备注"},
        "4": {"at": "看不清"},
    }}}
    got = no_candidate_notes_of(labels, 13)
    assert got[2]["ats"] == [20 * 60 + 34]
    assert got[2]["text"] == "他在这一带从右边走过"
    assert got[3]["ats"] == [] and got[3]["text"] == "只写了备注"
    assert got[4]["ats"] == [], "解析不出的时刻必须留空，不许猜"
    assert got[4]["unparsed"] == ["看不清"], "读不出的原文要原样报出来"
    assert no_candidate_notes_of(labels, 99) == {}
    assert no_candidate_notes_of(None, 13) == {}


def test_no_candidate_notes_is_a_reserved_top_level_key():
    """与另外几条通道同级；不登记进 LABEL_RESERVED_KEYS 会被当成段号展开，
    得到一堆对不上任何候选的预填项，且不报错。"""
    from pipeline.entryexit import LABEL_RESERVED_KEYS
    assert "no_candidate_notes" in LABEL_RESERVED_KEYS


def test_cells_are_ignored_when_the_user_says_one_person():
    """人数明确改回 1 时，逐格判定不作数——一个框里一个人，逐格分配没有意义。

    用户 2026-09-09 实测城市1 order 16 exit track 72：「选的是人数为 1，却还是
    出现了 4 个格位标注」。页面与这里必须同一条口径，否则页面藏起来了、流水线
    照旧按 cells 切分，人还是被切成好几个，而一处不报错。

    **cells 本身不删**：用户可能只是拨错了又拨回去，删掉等于把他的工作悄悄
    毁掉（`setPeople` 的注释记着这条教训）。这里只是不作数。
    """
    from pipeline.entryexit import cell_map_of_label
    cells = [{"cell": 1, "role": "创业者", "seat": 2},
             {"cell": 2, "same_as": 1}]
    got, _rep, src = cell_map_of_label({"n_people": 2, "cells": cells})
    assert src == "cells" and got, "人数 2 时照旧按 cells 走"
    got1, _r1, src1 = cell_map_of_label({"n_people": 1, "cells": cells})
    assert src1 == "none" and not got1, f"人数 1 时 cells 不作数，实得 {src1} {got1}"


def test_cells_still_count_when_n_people_is_absent():
    """`n_people` 整个不在时照旧按 cells 走。

    缺失不等于「用户说了只有一个人」——手改过的文件、更早的格式都可能没有
    这个键，按 1 处理会把已有的逐格工作静默作废。四个组的既有文件实测：
    带 cells 的记录**全部**带着 n_people≥2，所以这条改动对它们零影响。
    """
    from pipeline.entryexit import cell_map_of_label
    cells = [{"cell": 1, "role": "创业者", "seat": 2}]
    got, _rep, src = cell_map_of_label({"cells": cells})
    assert src == "cells" and got


def test_parse_timestamps_accepts_several_moments():
    """一个座位可以填**多个**时刻——那个人可能在几处露过面。

    用户 2026-09-09 给城市1 order 4 座位 1 填的就是 `1:01:12,1:24:21`
    （进场那一下和退场那一下）。只认单个值的话整条解析失败，densify 打一句
    「看不懂」，他填的两个时刻一个都没用上。
    """
    from pipeline.entryexit import parse_timestamps
    assert parse_timestamps("1:01:12,1:24:21") == [3672.0, 5061.0]
    assert parse_timestamps("20:34") == [1234.0]
    assert parse_timestamps("20:34、21:00") == [1234.0, 1260.0]
    assert parse_timestamps("20:34，21:00 1234.5") == [1234.0, 1260.0, 1234.5]
    assert parse_timestamps("") == []
    assert parse_timestamps(None) == []


def test_parse_timestamps_reports_the_pieces_it_cannot_read():
    """能读的照用，读不出的原样报出来——不因为一个错值把整条丢掉，
    也不猜那个错值是什么意思。"""
    from pipeline.entryexit import parse_timestamps, unparsable_timestamps
    assert parse_timestamps("20:34,大概一小时") == [1234.0]
    assert unparsable_timestamps("20:34,大概一小时") == ["大概一小时"]
    assert unparsable_timestamps("20:34,21:00") == []


def test_no_candidate_notes_returns_all_moments():
    from pipeline.entryexit import no_candidate_notes_of
    labels = {"no_candidate_notes": {"4": {
        "1": {"at": "1:01:12,1:24:21", "text": ""},
        "2": {"at": "20:34,看不清", "text": "备注"},
    }}}
    got = no_candidate_notes_of(labels, 4)
    assert got[1]["ats"] == [3672.0, 5061.0]
    assert got[1]["unparsed"] == []
    assert got[2]["ats"] == [1234.0] and got[2]["unparsed"] == ["看不清"]
    assert got[2]["text"] == "备注"


def test_the_judges_channel_survives_label_realignment(tmp_path, capsys):
    """重抽之后重新对齐时，顶层的 `judges` 通道必须原样留下（用户 2026-09-18）。

    `judges` 与段号平级。不列进 `LABEL_RESERVED_KEYS` 的后果与 `cell_exclusions`
    那几个逐字相同：被当成段号去和候选比对，一条也对不上，整份被算进「失配丢弃」
    并从结果里消失——**而一处不报错**。

    实测城市1：对 order 5/6/7/8/10 重抽之后，五位评审人的标注（标掉的格、补选的
    格、改判给别人的格）全部消失，登记表里五位都成了「没人标过」，而命令行只说
    「57 条标注中 50 条匹配上」——那 7 条里有 5 条就是评审席，看不出来。
    """
    import json
    from pipeline.entryexit import load_entry_labels
    p = tmp_path / "entry_labels_x.json"
    judges = {"judge#110,919": {"judge": None, "role": "",
                                "cells": {"2": "x"}, "picks": [7, 15]}}
    p.write_text(json.dumps({
        "_meta": {"detections_fingerprint": "OLD"},
        "judges": judges,
    }, ensure_ascii=False), encoding="utf-8")
    got = load_entry_labels(p, detections_fp="NEW", candidates_by_order={1: []})
    assert got.get("judges") == judges, got


def test_the_judges_channel_is_not_counted_as_a_failed_match(tmp_path, capsys):
    """评审席不参与候选匹配，也就不该算进匹配率。

    算进去会把匹配率压下来，甚至压到 70% 的作废线以下——一份好端端的标注
    因为评审席的存在被整份判废。
    """
    import json
    from pipeline.entryexit import load_entry_labels
    p = tmp_path / "entry_labels_y.json"
    p.write_text(json.dumps({
        "_meta": {"detections_fingerprint": "OLD"},
        "judges": {f"judge#{i},{i}": {"picks": [1]} for i in range(1, 6)},
    }, ensure_ascii=False), encoding="utf-8")
    load_entry_labels(p, detections_fp="NEW", candidates_by_order={1: []})
    out = capsys.readouterr().out
    assert "5 条失配" not in out, out


def _labels_with_meta(tmp_path, **meta) -> Path:
    base = {"detections_fingerprint": "abc123", "sheet": "x", "n_candidates": 1}
    base.update(meta)
    p = tmp_path / "entry_labels_test.json"
    p.write_text(json.dumps(
        {"_meta": base, "2": {"entry@100#900,1000": {"role": "创业者", "seat": 1}}},
        ensure_ascii=False), encoding="utf-8")
    return p


def test_load_warns_when_labels_came_from_an_older_page(tmp_path, capsys):
    """标注文件出自**旧版挑选页**时要说出来。

    用户 2026-09-19 实测电子27：我修好导出、重出页面、请他重新导出，收回来
    那五条专家的 seat 还是 null。页面本身是对的（命令行上跑它导出的是
    `seat: 5`），但读文件这一侧一声不吭，我只能猜是不是旧标签页导的——再请
    他导一次，还是猜。带上来源指纹就不必猜了。

    检测指纹在这个情形下**一定相符**（重出页面没有重跑检测），所以这条
    提醒必须独立于它，否则永远不会出现。
    """
    p = _labels_with_meta(tmp_path, page_code_fp="0000000000000000")
    load_entry_labels(p, detections_fp="abc123")
    out = capsys.readouterr().out
    assert "旧版" in out or "不是当前" in out, f"应提示页面版本对不上，实得：{out!r}"
    assert "重新导出" in out, f"要说清楚怎么办（刷新页面重新导出），实得：{out!r}"


def test_load_is_quiet_when_the_page_version_matches(tmp_path, capsys):
    """指纹一致时不要出声——每次都喊的提醒等于没有提醒。"""
    from pipeline.entryreview import page_code_fp
    p = _labels_with_meta(tmp_path, page_code_fp=page_code_fp())
    load_entry_labels(p, detections_fp="abc123")
    out = capsys.readouterr().out
    assert "旧版" not in out and "不是当前" not in out, f"不该报警：{out!r}"


def test_load_warns_when_the_page_version_is_absent(tmp_path, capsys):
    """连这个字段都没有 = 出自比这项改动更早的页面，同样要提醒。

    电子27 那份正是这种：文件里只有 detections_fingerprint。默认「没写就
    当它是对的」会让这条提醒恰好在最需要它的那一次缺席。
    """
    p = _labels_with_meta(tmp_path)
    load_entry_labels(p, detections_fp="abc123")
    out = capsys.readouterr().out
    assert "旧版" in out or "不是当前" in out, f"应提示页面版本未知/过旧：{out!r}"


def test_the_page_version_warning_does_not_void_the_labels(tmp_path):
    """只是提醒，标注照常可用——作废会让用户手上整批导出读不进来。"""
    p = _labels_with_meta(tmp_path, page_code_fp="0000000000000000")
    labels = load_entry_labels(p, detections_fp="abc123")
    assert labels["2"]["entry@100#900,1000"]["role"] == "创业者"


# ---------------------------------------------------------------------------
# exit 优先的质量门槛（用户 2026-09-20 实测电子28 order 6 座位 3）
# ---------------------------------------------------------------------------

def _filtered_and_clean_pair(order=6):
    """一条**被过滤**的 exit（又扁又矮：长宽比与框高两条都不过）+ 一条干净
    的 entry（帧多、框高）。两条都标同一个座位。"""
    seg = mkseg(order=order, presenter_num=3)
    dets = (
        # 被过滤：框高 0.06*H 远低于 ENTRYEXIT_MIN_HEIGHT_RATIO=0.15，
        # 宽 90 使长宽比 65/90≈0.72 低于 ENTRYEXIT_MIN_ASPECT=1.30
        track(34, RIGHT_CX + 300, 0.06 * H, 2, kind="exit", order=order, width=90.0)
        # 干净：高大、帧多
        + track(99, RIGHT_CX + 100, STANDING_H, 20, kind="entry", order=order))
    return seg, dets


def test_a_filtered_exit_does_not_outrank_a_clean_entry():
    """被过滤规则挡掉的 exit，不得仅凭「是 exit」压过没被过滤的候选。

    用户 2026-09-20 实测电子28 order 6 座位 3：那一格的图来自
    `exit@8033#1383,717`——**9 帧、框高 122 像素、四帧的框右边全部顶到
    x=1440（人已走出画面右缘）**，页面自己早把它挡进「被过滤」折叠区
    （aspect + height 两条都踩中）。同座位另有一条 37 帧、407 像素高、
    完整在画面内的 entry 候选，却因为 `_merge_same_seat_candidates` 里
    exit 是**无条件**优先而落选。

    两条规则互相打架，输的是更严格的那条：过滤规则说「这张认不出是谁」，
    合并规则说「exit 就是比 entry 好」。exit 优先的前提是那张图确实能看清
    人，前提不成立时它不该生效。
    """
    seg, dets = _filtered_and_clean_pair()
    cands = {c["track_id"]: c
             for c in entryexit.candidates_for_labels(dets, seg, frame_w=W, frame_h=H)}
    assert cands[34]["filter_reasons"], "装置前提：那条 exit 必须是被过滤的"
    assert not cands[99]["filter_reasons"], "装置前提：那条 entry 必须是干净的"
    labels = {"6": {
        cands[34]["stable_id"]: {"role": "创业者", "seat": 3},
        cands[99]["stable_id"]: {"role": "创业者", "seat": 3},
    }}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {6: {}})
    items = [e for e in plan.get(6, []) if e["seat"] == 3]
    assert len(items) == 1, f"同一座位只该有一条计划项，实得 {items}"
    assert items[0]["track_id"] == 99, (
        f"该由没被过滤的那条胜出，实得 track {items[0]['track_id']}")


def test_exit_still_wins_when_both_are_clean():
    """护栏：两条都没被过滤时，exit 依旧无条件优先——**帧数少也照样赢**。

    只加门槛不留这一条，很容易顺手改成「按帧数比」，把用户那条领域知识
    （退场多正面、进场多背影）悄悄废掉。
    """
    seg = mkseg(order=7, presenter_num=3)
    dets = (track(19, RIGHT_CX + 200, STANDING_H, 20, kind="entry", order=7)
            + track(10, RIGHT_CX + 300, STANDING_H, 2, kind="exit", order=7))
    cands = {c["track_id"]: c
             for c in entryexit.candidates_for_labels(dets, seg, frame_w=W, frame_h=H)}
    assert not any(cands[t]["filter_reasons"] for t in (19, 10)), "装置前提：两条都干净"
    labels = {"7": {cands[19]["stable_id"]: {"role": "创业者", "seat": 2},
                    cands[10]["stable_id"]: {"role": "创业者", "seat": 2}}}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {7: {}})
    items = [e for e in plan.get(7, []) if e["seat"] == 2]
    assert len(items) == 1 and items[0]["kind"] == "exit", (
        f"两条都干净时 exit 仍应优先（帧数 2 对 20 也一样），实得 {items}")


def test_all_filtered_still_follows_exit_priority():
    """全都被过滤时，门槛不改变任何结果——否则这个座位会因为「没有干净
    候选」而落到 entry 上，那是另一种无声的改判。
    """
    seg = mkseg(order=8, presenter_num=3)
    dets = (track(34, RIGHT_CX + 300, 0.06 * H, 9, kind="exit", order=8, width=90.0)
            + track(99, RIGHT_CX + 100, 0.06 * H, 20, kind="entry", order=8, width=90.0))
    cands = {c["track_id"]: c
             for c in entryexit.candidates_for_labels(dets, seg, frame_w=W, frame_h=H)}
    assert all(cands[t]["filter_reasons"] for t in (34, 99)), "装置前提：两条都被过滤"
    labels = {"8": {cands[34]["stable_id"]: {"role": "创业者", "seat": 1},
                    cands[99]["stable_id"]: {"role": "创业者", "seat": 1}}}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {8: {}})
    items = [e for e in plan.get(8, []) if e["seat"] == 1]
    assert len(items) == 1 and items[0]["track_id"] == 34, (
        f"全被过滤时仍按 exit 优先，实得 {items}")


def _frame_with_two_people(tmp_path, dark_box, bright_box, name="f.jpg"):
    """造一张真图：`dark_box` 区域几乎全黑，`bright_box` 区域明亮。"""
    import cv2, numpy as np
    img = np.full((int(H), W, 3), 30, dtype=np.uint8)
    x1, y1, x2, y2 = (int(v) for v in dark_box)
    img[y1:y2, x1:x2] = 4                      # 亮度 4，远低于阈值 10
    x1, y1, x2, y2 = (int(v) for v in bright_box)
    img[y1:y2, x1:x2] = 150
    p = tmp_path / name
    assert cv2.imwrite(str(p), img)
    return p


def test_a_pitch_dark_exit_does_not_outrank_a_visible_entry(tmp_path):
    """**特别暗**的 exit 不得压过看得清的 entry（用户 2026-09-20 指定的口径）。

    一张几乎全黑的图，衣着颜色根本无从判读，而颜色正是要编码的东西之一。
    这条与「特别小」那条互补：实测电子28 那一组里，好几条**没被任何过滤
    规则挡下**的退场候选亮度只有 5.8/8.5/8.8（全组中位数 25.8），它们个头
    不小，只靠尺寸门槛拦不住。
    """
    seg = mkseg(order=11, presenter_num=3)
    dark = (1100, 400, 1250, 1000)             # 高 600，尺寸完全合格
    lit = (860, 400, 1010, 1000)
    frame = _frame_with_two_people(tmp_path, dark, lit)
    # `track()` 不接帧路径，这里直接用 `det()` 拼——要的就是真图上的真坐标
    dets = ([det(34, 1175, 600, kind="exit", order=11, t=float(i),
                 frame_path=frame, bottom=1000.0, width=150.0) for i in range(4)]
            + [det(99, 935, 600, kind="entry", order=11, t=float(i),
                   frame_path=frame, bottom=1000.0, width=150.0) for i in range(4)])
    cands = {c["track_id"]: c
             for c in entryexit.candidates_for_labels(dets, seg, frame_w=W, frame_h=H)}
    b_dark = entryexit.candidate_brightness(cands[34])
    b_lit = entryexit.candidate_brightness(cands[99])
    assert b_dark is not None and b_dark < config.EXIT_PRIORITY_MIN_BRIGHTNESS, (
        f"装置前提：exit 那条要够暗，实测 {b_dark}")
    assert b_lit is not None and b_lit > config.EXIT_PRIORITY_MIN_BRIGHTNESS, (
        f"装置前提：entry 那条要够亮，实测 {b_lit}")
    assert not cands[34]["filter_reasons"], (
        "装置前提：这条 exit **没有**被任何过滤规则挡下——否则测的就是尺寸门槛了")

    labels = {"11": {cands[34]["stable_id"]: {"role": "创业者", "seat": 2},
                     cands[99]["stable_id"]: {"role": "创业者", "seat": 2}}}
    plan, _nr, _ns = entryexit_fill_plan([seg], dets, labels, {11: {}})
    items = [e for e in plan.get(11, []) if e["seat"] == 2]
    assert len(items) == 1 and items[0]["track_id"] == 99, (
        f"几乎全黑的退场画面不该胜出，实得 {items}")


def test_unreadable_frames_are_not_treated_as_dark():
    """读不到图 = 「不知道」，绝不能当成「很暗」。

    否则一处读图失败（路径变了、解码坏了）就会悄悄把一条好候选判出局，
    而表现只是「这个座位换了张图」，没有任何报错。本文件绝大多数装置用的
    正是不存在的帧路径，真按暗处理的话它们会集体改判。
    """
    seg = mkseg(order=12, presenter_num=2)
    dets = track(7, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=12)
    cand = entryexit.candidates_for_labels(dets, seg, frame_w=W, frame_h=H)[0]
    assert entryexit.candidate_brightness(cand) is None, "读不到图应返回 None"
    assert not entryexit.candidate_is_too_poor_to_prefer(cand), (
        "尺寸合格、亮度未知的候选不该被判为「太差」")


# ---------------------------------------------------------------------------
# 多个来源时，2×2 按来源轮流出格（用户 2026-09-26）
# ---------------------------------------------------------------------------

def _pairs_at(times, kind="exit", tid=1, cell_start=1):
    """一串 (格位号, 帧)，帧的时刻依次取 `times`。"""
    return [(cell_start + i, det(tid, RIGHT_CX + 200, STANDING_H, kind=kind,
                                 order=3, t=float(t)))
            for i, t in enumerate(times)]


class _Spec(dict):
    """`topup_pairs` 的一条补帧来源：一条候选 + 空格位表（= 整条都是他）。"""


def _spec(pairs):
    cand = {"stable_id": f"src{id(pairs)}", "best_dets": [d for _c, d in pairs],
            "kind": pairs[0][1].kind, "track_id": pairs[0][1].track_id}
    return (cand, [])


def test_several_sources_share_the_sheet_instead_of_the_first_filling_it():
    """有第二个来源时，第一个来源不许独占四格。

    用户 2026-09-26：「如果有多个标注，尽量不同站姿或差异较大的图片各保留一个
    格位；自动识别出的 2×2 格位，可能图片是高度雷同的，看情况只保留 1-2 个，
    尽量与其他不同画面的各位进行合并」。

    实测 7 组：99 个座位有多个来源，其中 **30 个的 2×2 只用上了一个**——第一
    个来源四帧占满，用户另外标的画面一格都没进去。而单一来源的四帧**首尾只隔
    0.5 秒（中位数）**，121 个里 99 个在 2 秒以内：同一个姿势拍了四次。
    """
    first = _pairs_at([0.0, 0.1, 0.2, 0.3], tid=1)
    other = _pairs_at([100.0, 100.1, 100.2, 100.3], kind="entry", tid=2)
    got = entryexit.topup_pairs(first, [_spec(other)], want=4)
    tids = [d.track_id for _c, d in got]
    assert len(got) == 4
    assert tids.count(1) == 2 and tids.count(2) == 2, f"两个来源应各占两格，实得 {tids}"
    assert tids[0] == 1, "第一个来源仍排在最前"


def test_within_one_source_the_second_cell_is_the_most_different_in_time():
    """同一个来源出第二格时，挑时间上离第一格最远的那张。

    按质量排第二的那张往往只隔 0.1 秒——等于同一张图放两遍。
    """
    first = _pairs_at([0.0, 0.1, 0.2, 5.0], tid=1)
    other = _pairs_at([100.0, 100.1], kind="entry", tid=2)
    got = entryexit.topup_pairs(first, [_spec(other)], want=4)
    mine = [d.t for _c, d in got if d.track_id == 1]
    assert mine == [0.0, 5.0], f"第一个来源应留 0.0 与离它最远的 5.0，实得 {mine}"


def test_three_sources_give_two_one_one():
    """三个来源：第一个 2 格、其余各 1 格——「看情况只保留 1-2 个」。"""
    a = _pairs_at([0.0, 0.1, 0.2, 0.3], tid=1)
    b = _pairs_at([50.0, 50.1], kind="entry", tid=2)
    c = _pairs_at([90.0, 90.1], kind="exit", tid=3)
    got = entryexit.topup_pairs(a, [_spec(b), _spec(c)], want=4)
    tids = [d.track_id for _c, d in got]
    assert tids == [1, 1, 2, 3], (
        f"第一个来源 2 格、其余各 1 格，同一来源的格挨着放，实得 {tids}")


def test_a_single_source_is_left_exactly_as_it_was():
    """只有一个来源时一张不动——没有别的画面可合并，宁可四张相近也不留空格。

    （本项目的优先级是「识别准确度 > 样本保留 > 分辨率」。）
    """
    only = _pairs_at([0.0, 0.1, 0.2, 0.3], tid=1)
    assert entryexit.topup_pairs(only, [], want=4) == only
    assert entryexit.topup_pairs(only, None, want=4) == only


def test_cell_numbers_travel_with_their_frames_when_sources_interleave():
    """轮流取之后，每一帧仍带着它在**自己那张**证据图上的格位号。

    格位号与帧拆开了，「第几格」就会指到另一张帧上，而一处不报错。
    """
    first = _pairs_at([0.0, 0.1, 0.2, 9.0], tid=1, cell_start=1)
    # 整条标的来源，格位号就是它在自己证据图上的位置 1..n（`spec_pairs`）
    other = _pairs_at([100.0, 100.5], kind="entry", tid=2)
    got = entryexit.topup_pairs(first, [_spec(other)], want=4)
    assert {d.track_id for _c, d in got} == {1, 2}, "装置前提：两个来源都得出格"
    for cell, d in got:
        if d.track_id == 1:
            assert {0.0: 1, 0.1: 2, 0.2: 3, 9.0: 4}[d.t] == cell, (cell, d.t)
        else:
            assert {100.0: 1, 100.5: 2}[d.t] == cell, (cell, d.t)
    # 第一个来源出的是 0.0 与 9.0：格位号必须是 1 与 4，不是它们在结果里的位置
    assert [c for c, d in got if d.track_id == 1] == [1, 4], got


def test_a_short_first_source_is_still_filled_up_by_the_others():
    """第一个来源只有 1 张时，其余格照旧由别的来源补满。"""
    first = _pairs_at([0.0], tid=1)
    other = _pairs_at([100.0, 101.0, 102.0, 103.0], kind="entry", tid=2)
    got = entryexit.topup_pairs(first, [_spec(other)], want=4)
    tids = [d.track_id for _c, d in got]
    assert tids.count(1) == 1 and tids.count(2) == 3, tids


def test_the_same_picture_saved_in_two_windows_is_not_used_twice():
    """同一时刻、同一个框的画面，存在两个窗口目录里，也只能占一格。

    用户 2026-09-26 电子10 第 13 段座位 2：多个来源轮流出格之后，那张 2×2 是
    **两张图各出现两次**。第 13 段的退场窗口与第 14 段的进场窗口重叠（13 段
    exitTime 30102.4 恰好等于 14 段 entryTime），同一批视频帧被分别存进
    `frames/13/exit/` 与 `frames/14/entry/`；他在两个窗口里都标了这个人，两条
    来源各出两格，而去重按 Detection 整体比较——路径、段号、kind 都不同，
    于是一模一样的画面当成了不同的图。

    旧规则下第一个来源占满四格，这个重复从来没机会露出来。去重要比**画面是
    什么**（时刻 + 框），不是它存在哪个目录。
    """
    from pipeline.detect import Detection
    from pathlib import Path

    def at(seg, kind, t, x1):
        return Detection(seg, kind, t, Path(f"frames/{seg:02d}/{kind}/{t:010.3f}.jpg"),
                         39 if kind == "exit" else 13, x1, 100.0, x1 + 300.0, 900.0, 0.9)

    first = [(1, at(13, "exit", 30105.1, 800.0)), (2, at(13, "exit", 30104.8, 1050.0))]
    # 第 14 段进场窗口里存的是**同一时刻、同一个框**
    twin = [(1, at(14, "entry", 30105.1, 800.0)), (2, at(14, "entry", 30104.8, 1050.0))]
    got = entryexit.topup_pairs(first, [_spec(twin)], want=4)
    pics = [(round(d.t, 3), round(d.x1, 1)) for _c, d in got]
    assert len(pics) == len(set(pics)), f"同一张画面出现了两次：{pics}"
    assert len(got) == 2, f"只有两张不同的画面，就只该有两格，实得 {len(got)}"



def test_genuinely_unplaced_flows_keep_their_warning(capsys):
    """真正落不了地的流转（目标段不存在、座位号超出编制）仍是 ⚠ 未落位。

    改口只针对「座位已有人」那一种；把真正的失败也说成「并入」，就是另一句
    假话。
    """
    src = Segment(order=15, company="源企业", start=0, end=1000,
                  presenter_num=1, presenters=())
    tgt = Segment(order=16, company="目标企业", start=1000, end=2000,
                  presenter_num=1, presenters=())
    dets = track(135, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=15)
    cand = entryexit_candidates(dets, src, frame_w=W)[0]
    labels = {"15": {cand["stable_id"]: {"role": "下一组创业者", "seat": 3,
                                         "n_people": 1}}}
    entryexit_fill_plan([src, tgt], dets, labels, {16: {}})
    out = capsys.readouterr().out
    assert "未落位" in out and "编制" in out, f"座位号超出编制必须仍报未落位：{out}"
    assert "并入" not in out, f"落不了地的不许说成并入：{out}"
