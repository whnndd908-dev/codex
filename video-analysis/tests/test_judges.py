"""评审席（画面左侧的专家）识别。

用户 2026-09-17：「重新读取视频左侧的专家的衣服图片，以对专家着装的正式度
进行评价……专家的位置参考[会场平面图]」，并明确「每组恒定为 5 个专家，只是
有的可能不在其中，将交由我人工核实」。

三条与创业者那一路不同的性质，本文件的断言都围绕它们：

1. **评审人是组级的，不是段级的**——同一批人坐一整天，衣服不变。所以跨全部
   段汇集同一个位置的检测，产出每组 5 条，而不是每家企业重复一遍。
2. **席位恒为 5**，检出不足 5 人时空着的那几号必须是**可见的缺口**，不是
   「这组只有 4 个」——由人工判断是谁没入座、或是谁被漏检了。
3. **编号方向沿用创业者那条已验证的约定**：y2 越大离镜头越近，最近的那位是
   专家 1（`seats.py`：「y2 越小离镜头越远；最里侧那位即 y2 中位数最小者」）。
   平面图里评审人1 与创业者1 同在靠工作人员计时那一侧，与此一致。
"""
from pathlib import Path

import pytest

from pipeline import config
from pipeline.detect import Detection

W, H = 1440, 1080
LEFT_CX = 0.30 * W          # 评审席恒在画面左侧（x < RIGHT_REGION_X）
SEATED_H = 0.35 * H


def det(tid, cx, y2, kind="seated", order=2, t=0.0, h=SEATED_H, width=60.0,
        frame="f.jpg"):
    return Detection(order, kind, t, Path(frame), tid,
                     cx - width / 2, y2 - h, cx + width / 2, y2, 0.9)


def judge_track(tid, cx, y2, n, order=2, t0=0.0, frame="f.jpg"):
    return [det(tid, cx, y2, order=order, t=t0 + i, frame=frame) for i in range(n)]


def _scene(n_seg=3):
    """三段，左侧四位评审人坐在一条进深线上（离镜头由近及远）。"""
    dets = []
    line = [(200.0, 900.0), (300.0, 800.0), (420.0, 700.0), (540.0, 600.0)]
    for order in range(1, n_seg + 1):
        for i, (cx, y2) in enumerate(line):
            dets += judge_track(10 + i, cx, y2, 8, order=order,
                                t0=order * 1000, frame=f"seg{order}.jpg")
    return dets


def test_judges_are_pooled_across_all_segments():
    """同一位评审人在每一段都有一条自己的 track，必须归并成**一个人**。

    不跨段归并的话，一位评审人会变成 N 个候选（N = 段数），人工得把同一张脸
    认 N 遍，而衣服在整场里是同一套。
    """
    from pipeline.judges import judge_candidates
    got = judge_candidates(_scene(n_seg=3), frame_w=W, frame_h=H, min_frames=10)
    assert len(got) == 4, [f"{c['median_cx']:.0f},{c['median_y2']:.0f}" for c in got]
    for c in got:
        assert c["n_segments"] == 3, c
        assert c["n_frames"] == 24, c


def test_judges_are_numbered_from_the_camera_inwards():
    """离镜头最近（y2 最大）的是专家 1，与创业者座位号同一条约定。"""
    from pipeline.judges import judge_candidates
    got = judge_candidates(_scene(), frame_w=W, frame_h=H, min_frames=10)
    ys = [c["median_y2"] for c in got]
    assert ys == sorted(ys, reverse=True), ys
    assert got[0]["median_y2"] == pytest.approx(900.0)
    assert got[-1]["median_y2"] == pytest.approx(600.0)


def test_only_the_left_region_counts():
    """右半区是创业者席，一个都不许混进评审人里——混进去就是把创业者的衣着
    当成专家的记进表里，而一处不报错。"""
    from pipeline.judges import judge_candidates
    dets = _scene() + judge_track(99, 0.80 * W, 900.0, 30, order=1)
    got = judge_candidates(dets, frame_w=W, frame_h=H, min_frames=10)
    assert len(got) == 4
    assert all(c["median_cx"] < config.RIGHT_REGION_X * W for c in got)


def test_only_seated_frames_count():
    """只看就座窗口：进出场窗口里左侧走动的是工作人员与正在离场的上一组人，
    把他们当评审人会让同一张脸在几个组里反复出现。"""
    from pipeline.judges import judge_candidates
    walking = [det(98, 250.0, 950.0, kind="entry", order=1, t=float(i))
               for i in range(30)]
    got = judge_candidates(_scene() + walking, frame_w=W, frame_h=H, min_frames=10)
    assert len(got) == 4, [f"{c['median_cx']:.0f}" for c in got]


def test_fragments_below_the_frame_floor_are_dropped():
    """只闪现几帧的碎片不作候选——评审人坐一整场，几帧的东西是误检。"""
    from pipeline.judges import judge_candidates
    dets = _scene() + judge_track(97, 350.0, 500.0, 2, order=1)
    got = judge_candidates(dets, frame_w=W, frame_h=H, min_frames=10)
    assert len(got) == 4


def test_stable_id_is_derived_from_position_not_track_id():
    """标识由**位置**导出，重跑检测后仍然成立。

    评审人整场不动，位置就是他的身份；track_id 重跑即全变（Task 46 的教训）。
    也不能用时间——他在每一段都出现，没有一个「时刻」可言。
    """
    from pipeline.judges import judge_candidates
    a = judge_candidates(_scene(), frame_w=W, frame_h=H, min_frames=10)
    shifted = []
    for d in _scene():
        shifted.append(Detection(d.seg_order, d.kind, d.t, d.frame_path,
                                 d.track_id + 500, d.x1, d.y1, d.x2, d.y2, d.conf))
    b = judge_candidates(shifted, frame_w=W, frame_h=H, min_frames=10)
    assert [c["stable_id"] for c in a] == [c["stable_id"] for c in b]
    assert all(c["stable_id"].startswith("judge#") for c in a)


def test_five_slots_even_when_fewer_are_detected():
    """席位恒为 5：只检出 4 位时，第 5 号必须是**可见的缺口**。

    用户 2026-09-17：「每组恒定为 5 个专家，只是有的可能不在其中，将交由我
    人工核实」。把它记成「这组只有 4 个」，人工就无从分辨「确实只来了 4 位」
    与「第 3 位被漏检了」。
    """
    from pipeline.judges import judge_slots
    got = judge_slots(judge_candidates_of(_scene()), labels=None)
    assert sorted(got) == [1, 2, 3, 4, 5]
    assert got[5] is None, "第 5 号该是空缺"
    assert got[1]["median_y2"] > got[4]["median_y2"]


def judge_candidates_of(dets):
    from pipeline.judges import judge_candidates
    return judge_candidates(dets, frame_w=W, frame_h=H, min_frames=10)


def test_human_numbering_overrides_the_geometric_order():
    """人工改号压过几何排序——几何只是基线。

    只检出 4 位时基线会把他们排成 1..4，而缺的那位可能在中间；人工说「这位
    其实是专家 3」，剩下的位置就得跟着让开，不能替他猜一个整体平移。
    """
    from pipeline.judges import judge_slots
    cands = judge_candidates_of(_scene())
    labels = {"judges": {cands[1]["stable_id"]: {"judge": 3}}}
    got = judge_slots(cands, labels)
    assert got[3] is not None and got[3]["stable_id"] == cands[1]["stable_id"]
    assert got[2] is None or got[2]["stable_id"] != cands[1]["stable_id"]


def test_a_candidate_marked_not_a_judge_is_dropped():
    """人工标「不是评审人」（工作人员、路过的人）就不占席位。"""
    from pipeline.judges import judge_slots
    cands = judge_candidates_of(_scene())
    labels = {"judges": {cands[0]["stable_id"]: {"judge": None, "role": "不是评审人"}}}
    got = judge_slots(cands, labels)
    assert all(v is None or v["stable_id"] != cands[0]["stable_id"]
               for v in got.values())


def test_two_candidates_on_one_slot_raises():
    """两个人被指到同一号——拒绝整组，不挑一个。

    挑一个的结果取决于字典顺序，会随标注先后变，且一处不报错（与
    `entryexit.seat_reassign_of` 同一条规矩）。
    """
    from pipeline.judges import judge_slots
    cands = judge_candidates_of(_scene())
    labels = {"judges": {cands[0]["stable_id"]: {"judge": 2},
                         cands[1]["stable_id"]: {"judge": 2}}}
    with pytest.raises(ValueError, match="专家 2"):
        judge_slots(cands, labels)


# --- 逐格核实（用户 2026-09-18）-------------------------------------------
#
# 「专家的 2×2 需要每个格子进行人工核实，因为有的格子混进了其他专家」。
# 评审席比创业者座位更容易混：跨段归并按位置容差（cx±90、y2±45）并人，
# 相邻两位评审人本来就坐得近，一段里跟丢重识别之后很容易把隔壁那位的框
# 并进同一个人。四格红框标的都是同一个候选，**模型无从分辨哪格才是本人**，
# 会把两个人的衣着混起来判断——不报错，数字全错（与 `cell_exclusions_of`
# 记下的是同一类事故）。


def test_judge_cell_exclusions_read_from_labels():
    """人工勾掉的格位号从 `judges[stable_id].cell_exclusions` 读出来。"""
    from pipeline.judges import judge_cell_exclusions
    labels = {"judges": {"judge#200,900": {"judge": 1, "cell_exclusions": [2, 4]}}}
    assert judge_cell_exclusions(labels, "judge#200,900") == {2, 4}


def test_judge_cell_exclusions_default_to_empty():
    """没标过的候选、没有该字段、labels 为 None——一律空集合，行为不变。"""
    from pipeline.judges import judge_cell_exclusions
    assert judge_cell_exclusions(None, "judge#1,2") == set()
    assert judge_cell_exclusions({}, "judge#1,2") == set()
    assert judge_cell_exclusions({"judges": {"judge#1,2": {"judge": 3}}},
                                 "judge#1,2") == set()


def test_judge_cell_exclusions_ignore_unreadable_numbers():
    """读不出的格位号跳过，不猜也不炸——旁路通道里混进脏值是常态。"""
    from pipeline.judges import judge_cell_exclusions
    labels = {"judges": {"j": {"cell_exclusions": [1, "3", None, "x", 0, -2]}}}
    assert judge_cell_exclusions(labels, "j") == {1, 3}


def test_judge_cell_exclusions_are_keyed_by_stable_id_not_slot():
    """键是 stable_id，不是席位号。

    席位号是人工随时会改的：把某位从专家 3 改成专家 2，勾掉的格位必须跟着
    **这个人**走。若按席位号存，改号之后排除就落到了另一位身上——勾掉的那
    一格重新混进模型，而一处不报错。
    """
    from pipeline.judges import judge_cell_exclusions
    labels = {"judges": {"judge#540,600": {"judge": 3, "cell_exclusions": [1]}}}
    assert judge_cell_exclusions(labels, "judge#540,600") == {1}
    assert judge_cell_exclusions(labels, "judge#200,900") == set()


# --- 逐格改判：那一格其实是谁（用户 2026-09-18）-----------------------------
#
# 「专家的每个格有可能是其他人，给出这个修改选项」。只能勾掉是不够的：城市1
# 实测四位专家共勾掉 7 格，专家1 与专家3 各只剩 1 格——模型拿一格判整场衣着，
# 比拿四格混着两个人还糟。那些格并没有坏，只是**归错了人**；说清它是谁，帧就
# 换到那一位名下，而不是扔掉。与创业者那一路的「格位切分」同一条思路。


def test_cell_assignment_says_which_judge_a_cell_really_is():
    """`cells` 里 {格位号: 专家号}——那一格其实是那一位。"""
    from pipeline.judges import judge_cell_assignments
    labels = {"judges": {"j": {"cells": {"2": 3, "4": 1}}}}
    assert judge_cell_assignments(labels, "j") == {2: 3, 4: 1}


def test_a_cell_that_is_nobody_maps_to_none():
    """`"x"` = 其他人（工作人员、路过的）：丢掉，不归给任何一位。"""
    from pipeline.judges import judge_cell_assignments
    labels = {"judges": {"j": {"cells": {"3": "x"}}}}
    assert judge_cell_assignments(labels, "j") == {3: None}


def test_cells_not_mentioned_are_the_person_themselves():
    """没说的格位就是本人——默认不动，人只需要点出**错的**那几格。"""
    from pipeline.judges import judge_cell_assignments
    labels = {"judges": {"j": {"cells": {"2": "x"}}}}
    got = judge_cell_assignments(labels, "j")
    assert 1 not in got and 3 not in got and 4 not in got, got


def test_the_old_checkbox_export_still_reads():
    """旧的 `cell_exclusions: [2,4]` 照旧读得出——用户 2026-09-18 11:30 已经
    用那一版勾完城市1 导出了，换个字段名就把他那份工作静默作废了。"""
    from pipeline.judges import judge_cell_assignments, judge_cell_exclusions
    labels = {"judges": {"j": {"cell_exclusions": [2, 4]}}}
    assert judge_cell_assignments(labels, "j") == {2: None, 4: None}
    assert judge_cell_exclusions(labels, "j") == {2, 4}


def test_exclusions_cover_every_cell_that_is_not_the_person():
    """归给别人的格，对**这一位**来说同样是要剔掉的——不然他的 2×2 里还留着
    一张别人的脸，而那张脸同时又出现在另一位名下。"""
    from pipeline.judges import judge_cell_exclusions
    labels = {"judges": {"j": {"cells": {"2": 3, "4": "x"}}}}
    assert judge_cell_exclusions(labels, "j") == {2, 4}


def test_unreadable_cell_targets_are_dropped_not_guessed():
    """读不出的目标号按「其他人」处理：宁可丢一格，不可把它记到某一位名下。"""
    from pipeline.judges import judge_cell_assignments
    labels = {"judges": {"j": {"cells": {"1": "zzz", "2": 99, "3": 0, "4": None}}}}
    assert judge_cell_assignments(labels, "j") == {1: None, 2: None, 3: None, 4: None}


# --- 备选池：2×2 不足时拿什么来补（用户 2026-09-18）------------------------
#
# 「针对 2×2 不足的专家，重新给出新的挑选图，以实现 2×2 四张图」。城市1 标完
# 之后五位专家分别只剩 1/3/3/2/3 格，共缺 8 格——而每位在盘上有 846 到 2173
# 帧，材料多得是，缺的只是**把它们拿出来给人挑**。
#
# 两条硬约束：
# 1. 池的**前四张必须与现在的 best_dets 完全一致**。格位号是人工标注的键，
#    前四张一变，用户已经标好的「格2 是其他人」就指到了另一张图上——静默
#    张冠李戴，比没有备选糟得多。
# 2. 备选要**跨段取**。同一位被误并进来的邻座往往集中在某一段里，按分数往下
#    顺的话第 5 张多半还是那一段的同一个人；换一段才真正换到人。


def _pool_scene(n_seg=6, per_seg=10):
    """n_seg 段，一位评审人每段 per_seg 帧。"""
    dets = []
    for order in range(1, n_seg + 1):
        for k in range(per_seg):
            dets += judge_track(10, 300.0, 800.0, 1, order=order,
                                t0=order * 100 + k, frame=f"s{order}_{k}.jpg")
    return dets


def test_the_pool_starts_with_exactly_the_current_best_four():
    """池的前四张 = 原来的 best_dets，一张不差、次序不变。

    格位号是标注的键：前四张一变，用户标好的「格2 是其他人」就指到了另一张
    图上，而一处不报错。
    """
    from pipeline.annotate import select_best_frames
    from pipeline.judges import judge_candidates
    dets = _pool_scene()
    got = judge_candidates(dets, frame_w=W, frame_h=H, min_frames=5)[0]
    want = select_best_frames(dets, n=config.JUDGE_CELL_COUNT)
    assert [d.t for d in got["pool_dets"][:config.JUDGE_CELL_COUNT]] == \
           [d.t for d in want]
    assert [d.t for d in got["best_dets"]] == [d.t for d in want]


def test_the_pool_reaches_the_configured_size():
    """池按 `JUDGE_CELL_POOL` + 位置分散追加的 `JUDGE_CELL_SPREAD` 取满。"""
    from pipeline.judges import judge_candidates
    got = judge_candidates(_pool_scene(), frame_w=W, frame_h=H, min_frames=5)[0]
    want = config.JUDGE_CELL_POOL + config.JUDGE_CELL_SPREAD
    assert len(got["pool_dets"]) == want, len(got["pool_dets"])


def test_the_spares_spread_across_segments():
    """备选优先**换一段**取，不按分数一路往下顺。

    误并进来的邻座常集中在某一段；顺着分数取第 5 张多半还是那一段的同一个
    人，换了也白换——而用户会以为自己已经换过一张了。
    """
    from pipeline.judges import judge_candidates
    got = judge_candidates(_pool_scene(n_seg=6), frame_w=W, frame_h=H,
                           min_frames=5)[0]
    spares = got["pool_dets"][config.JUDGE_CELL_COUNT:]
    head = {d.seg_order for d in got["pool_dets"][:config.JUDGE_CELL_COUNT]}
    assert [d for d in spares if d.seg_order not in head], (
        f"备选一段都没换：{[d.seg_order for d in spares]}")
    # 六段一段都不许漏：拿了某一段的第三张、却整段没碰过另一段，正是这条要
    # 防的事——漏掉的那一段里才可能有真正不同的一个人。
    assert {d.seg_order for d in got["pool_dets"]} == set(range(1, 7)), \
        sorted({d.seg_order for d in got["pool_dets"]})
    # 先给没用过的段，用完了才回头取同段的次好
    n_new = 6 - len(head)
    firsts = [d.seg_order for d in spares[:n_new]]
    assert len(set(firsts)) == n_new and not (set(firsts) & head), (firsts, head)


def test_the_pool_falls_back_to_more_of_the_same_segment():
    """段数不够时用同段的次好补齐——宁可近似，也不给人一个空格。"""
    from pipeline.judges import judge_candidates
    got = judge_candidates(_pool_scene(n_seg=2, per_seg=20), frame_w=W, frame_h=H,
                           min_frames=5)[0]
    want = config.JUDGE_CELL_POOL + config.JUDGE_CELL_SPREAD
    assert len(got["pool_dets"]) == want
    assert len({id(d) for d in got["pool_dets"]}) == want, "有帧重复了"


def test_the_pool_never_exceeds_what_is_on_disk():
    """帧本来就少于池大小时，有多少给多少，不重复凑数。"""
    from pipeline.judges import judge_candidates
    dets = judge_track(10, 300.0, 800.0, 7, order=1, frame="a.jpg")
    got = judge_candidates(dets, frame_w=W, frame_h=H, min_frames=5)[0]
    assert len(got["pool_dets"]) == 7, len(got["pool_dets"])


# --- 人工挑选：哪四格组成 2×2（用户 2026-09-18）---------------------------
#
# 「不是按后面的备选自动顶上，而是我可以挑选 4 个最清晰的组成 2×2 图」。
# 自动顶上只按分数（面积×清晰度）往下顺，而「清晰」这件事机器排得并不准——
# 真正看得清衣服的那张要人眼定。自动顺序留作**基线**：没人挑过的那几位照旧
# 有一张四格图，不会因为没人动过就空着。


def test_picks_are_read_from_the_labels():
    """`picks` 就是人工挑中的池格位号，按他挑的次序。"""
    from pipeline.judges import judge_picks
    labels = {"judges": {"j": {"picks": [1, 6, 9, 12]}}}
    assert judge_picks(labels, "j") == [1, 6, 9, 12]


def test_picks_default_to_empty():
    """没挑过就是空——由调用方退回自动基线，而不是在这里替他猜。"""
    from pipeline.judges import judge_picks
    assert judge_picks(None, "j") == []
    assert judge_picks({"judges": {"j": {}}}, "j") == []


def test_picks_are_capped_and_deduplicated():
    """最多四格，重复的只算一次——2×2 只有四格，多的画不下。"""
    from pipeline.judges import judge_picks
    labels = {"judges": {"j": {"picks": [3, 3, 1, 7, 9, 11]}}}
    assert judge_picks(labels, "j") == [3, 1, 7, 9]


def test_unreadable_picks_are_skipped():
    """读不出的格位号跳过，不猜也不炸。"""
    from pipeline.judges import judge_picks
    labels = {"judges": {"j": {"picks": [1, "5", None, "x", 0, -2]}}}
    assert judge_picks(labels, "j") == [1, 5]


# --- 重抽之后标注要对得回来（用户 2026-09-18）------------------------------
#
# 评审人的标识由**位置中位数**导出（`judge#cx,y2`，整数像素）。只要重抽了任何
# 一段，那一段的检测就会重新参与中位数，位置挪动零点几个像素、标识即变——而
# 查找是精确字符串匹配，对不上就静默当作「这一位没人标过」。
#
# 实测城市1：只把 order 5/6/7/8/10 加密（占全组 seated 检测的 28%），专家2 的
# cx 中位数从 245.48 挪到 244.45，标识 `judge#245,801` → `judge#244,801`，他身上
# 那一整条标注（标掉格4、补选格5）就没了，页面上看着像是从没标过。
#
# 创业者那一路早就有按内容重新对齐这一层（Task 46），评审人这一路漏了。


def _cand(cx, y2, sid=None):
    from pipeline.judges import judge_stable_id
    return {"stable_id": sid or judge_stable_id(cx, y2),
            "median_cx": cx, "median_y2": y2}


def test_exactly_matching_labels_are_kept():
    """标识没变的照旧对上。"""
    from pipeline.judges import realign_judge_labels
    cands = [_cand(245.4, 801.1), _cand(110.0, 919.1)]
    labels = {"judges": {"judge#245,801": {"picks": [5]}}}
    out, stats = realign_judge_labels(cands, labels)
    assert out["judges"]["judge#245,801"] == {"picks": [5]}
    assert stats["matched"] == 1 and stats["total"] == 1


def test_a_label_follows_a_candidate_that_moved_a_pixel():
    """中位数挪了一个像素，标注跟着挪到新标识上。

    这正是城市1 专家2 的情形：只重抽了五段，他的 cx 从 245.48 挪到 244.45。
    """
    from pipeline.judges import realign_judge_labels
    cands = [_cand(244.4, 801.3)]          # 新一轮：judge#244,801
    labels = {"judges": {"judge#245,801": {"cells": {"4": "x"}, "picks": [5]}}}
    out, stats = realign_judge_labels(cands, labels)
    assert "judge#244,801" in out["judges"], out["judges"]
    assert out["judges"]["judge#244,801"]["picks"] == [5]
    assert "judge#245,801" not in out["judges"]
    assert stats["matched"] == 1 and stats["moved"] == 1


def test_a_label_too_far_from_every_candidate_is_dropped_and_reported():
    """离所有候选都太远的标注不硬塞给谁——但必须报出来，否则工作没了没人知道。"""
    from pipeline.judges import realign_judge_labels
    cands = [_cand(110.0, 919.1)]
    labels = {"judges": {"judge#530,601": {"picks": [5]}}}
    out, stats = realign_judge_labels(cands, labels)
    assert out["judges"] == {}, out["judges"]
    assert stats["matched"] == 0 and stats["dropped"] == ["judge#530,601"]


def test_an_ambiguous_label_is_refused_not_guessed():
    """两个候选都在容差内就不认——挑一个的结果取决于字典顺序，且认错了
    就是把这一位的衣着判定安到另一个人头上，一处不报错。"""
    from pipeline.judges import realign_judge_labels
    cands = [_cand(400.0, 655.0), _cand(404.0, 655.0)]
    labels = {"judges": {"judge#402,655": {"picks": [5]}}}
    out, stats = realign_judge_labels(cands, labels)
    assert out["judges"] == {}, out["judges"]
    assert stats["ambiguous"] == ["judge#402,655"]


def test_an_exact_hit_wins_over_a_nearer_neighbour():
    """标识完全相同的先配对，配过的候选不再参与就近匹配。

    否则一个挪了位的标注可能抢走另一位的候选，两条标注互相串位。
    """
    from pipeline.judges import realign_judge_labels
    a, b = _cand(400.0, 655.0), _cand(401.0, 655.0)
    labels = {"judges": {"judge#401,655": {"picks": [1]},
                         "judge#400,655": {"picks": [2]}}}
    out, _ = realign_judge_labels([a, b], labels)
    assert out["judges"]["judge#400,655"]["picks"] == [2]
    assert out["judges"]["judge#401,655"]["picks"] == [1]


def test_realigning_leaves_the_other_channels_untouched():
    """只动 `judges` 这一段，别的旁路通道一个字都不碰。"""
    from pipeline.judges import realign_judge_labels
    labels = {"_meta": {"x": 1}, "seat_overrides": {"2": {"1": 3}},
              "judges": {"judge#110,919": {"picks": [1]}}}
    out, _ = realign_judge_labels([_cand(110.0, 919.1)], labels)
    assert out["_meta"] == {"x": 1}
    assert out["seat_overrides"] == {"2": {"1": 3}}
    assert labels["judges"] == {"judge#110,919": {"picks": [1]}}, "原字典被改了"


def test_no_judges_channel_is_a_no_op():
    """没有 judges 这一段时原样返回，行为完全不变。"""
    from pipeline.judges import realign_judge_labels
    out, stats = realign_judge_labels([_cand(110.0, 919.1)], {"_meta": {}})
    assert out == {"_meta": {}}
    assert stats["total"] == 0
    out2, _ = realign_judge_labels([], None)
    assert out2 is None


# --- 坐在右侧的评审人（用户 2026-09-19）-----------------------------------
#
# 「文化4 的专家5 在右侧最里面靠近投影仪的位置」。`judge_candidates` 只看画面
# **左区**（`cx < RIGHT_REGION_X`），坐在右侧的那一位一个候选都出不来——文化4
# 实测评审席 4/5，第 5 号空缺且「未入席的候选」里也是空的，人工无从指认。
#
# 他并没有丢：右区的人都进了**创业者候选**。所以出口是逐格身份那一栏——把某个
# 创业者候选的某一格标成「专家 N」，那一帧就归到第 N 位评审人名下。这与左区
# 评审人之间互相改判（`judge_cell_assignments`）是同一条通路，只是来源不同。


def test_a_candidate_cell_marked_as_a_judge_is_collected():
    """创业者候选里标成「专家 N」的格位，收进第 N 位评审人。"""
    from pipeline.judges import judge_donations_from_candidates
    cand = {"stable_id": "exit@100#900,700",
            "best_dets": [det(1, 900.0, 700.0), det(1, 901.0, 701.0)]}
    labels = {"2": {cand["stable_id"]: {"n_people": 2, "cells": [
        {"cell": 1, "role": "创业者", "seat": 3},
        {"cell": 2, "role": "专家", "seat": 5},
    ]}}}
    got = judge_donations_from_candidates(labels, {2: [cand]})
    assert list(got) == [5], got
    assert [(s, k) for s, k, _d in got[5]] == [(cand["stable_id"], 2)]
    assert got[5][0][2] is cand["best_dets"][1]


def test_a_judge_cell_without_a_number_is_not_collected():
    """标了「专家」却没选号的不收——不知道是哪一位，硬塞就是张冠李戴。"""
    from pipeline.judges import judge_donations_from_candidates
    cand = {"stable_id": "exit@100#900,700", "best_dets": [det(1, 900.0, 700.0)]}
    labels = {"2": {cand["stable_id"]: {"n_people": 2, "cells": [
        {"cell": 1, "role": "专家", "seat": None},
    ]}}}
    assert judge_donations_from_candidates(labels, {2: [cand]}) == {}


def test_other_roles_are_not_collected():
    """只收精确等于「专家」的：创业者、工作人员、不确定一概不算。"""
    from pipeline.judges import judge_donations_from_candidates
    cand = {"stable_id": "exit@100#900,700",
            "best_dets": [det(1, 900.0, 700.0)] * 3}
    for role in ("创业者", "工作人员", "不确定"):
        labels = {"2": {cand["stable_id"]: {"n_people": 2, "cells": [
            {"cell": 1, "role": role, "seat": 5}]}}}
        assert judge_donations_from_candidates(labels, {2: [cand]}) == {}, role


def test_a_judge_number_out_of_range_is_dropped():
    """号超出 1..JUDGE_COUNT 的不收，也不钳到边界——钳过去就是记到别人头上。"""
    from pipeline.judges import judge_donations_from_candidates
    cand = {"stable_id": "exit@100#900,700", "best_dets": [det(1, 900.0, 700.0)]}
    for bad in (0, 99, -1, "x"):
        labels = {"2": {cand["stable_id"]: {"n_people": 2, "cells": [
            {"cell": 1, "role": "专家", "seat": bad}]}}}
        assert judge_donations_from_candidates(labels, {2: [cand]}) == {}, bad


def test_donations_from_candidates_are_ordered_deterministically():
    """多格指向同一位时按 (候选标识, 格位号) 排序——不能靠字典顺序。

    靠字典顺序的话，2×2 装不下时被丢掉的是哪一格会随运行而变。
    """
    from pipeline.judges import judge_donations_from_candidates
    a = {"stable_id": "exit@100#900,700", "best_dets": [det(1, 900.0, 700.0)] * 4}
    b = {"stable_id": "exit@050#880,690", "best_dets": [det(1, 880.0, 690.0)] * 4}
    cells = [{"cell": 3, "role": "专家", "seat": 5},
             {"cell": 1, "role": "专家", "seat": 5}]
    labels = {"2": {a["stable_id"]: {"n_people": 2, "cells": list(cells)},
                    b["stable_id"]: {"n_people": 2, "cells": list(cells)}}}
    got = judge_donations_from_candidates(labels, {2: [a, b]})
    assert [(s, k) for s, k, _d in got[5]] == [
        (b["stable_id"], 1), (b["stable_id"], 3),
        (a["stable_id"], 1), (a["stable_id"], 3)]


def test_a_whole_candidate_marked_as_a_judge_is_collected():
    """人数为 1 的候选整张标成「专家 N」时，它的帧全部收给第 N 位。

    实测文化4 entry/exit track 2：右侧那位评审人的候选只有一个人，不展开格位
    分配，人工只能在卡片顶上选身份+号。只认逐格标注的话，他标了等于没标。
    """
    from pipeline.judges import judge_donations_from_candidates
    cand = {"stable_id": "exit@100#900,700",
            "best_dets": [det(1, 900.0, 700.0), det(1, 901.0, 701.0)]}
    labels = {"2": {cand["stable_id"]: {"role": "专家", "seat": 5}}}
    got = judge_donations_from_candidates(labels, {2: [cand]})
    assert [(s, k) for s, k, _d in got[5]] == [(cand["stable_id"], 1),
                                               (cand["stable_id"], 2)]


def test_per_cell_marks_win_over_the_whole_candidate_role():
    """逐格标过就以逐格为准——整张的角色是粗判，逐格是细judgement。

    两者都收的话，同一帧会被收两遍（整张一次、逐格一次），2×2 里出现重复画面。
    """
    from pipeline.judges import judge_donations_from_candidates
    cand = {"stable_id": "exit@100#900,700",
            "best_dets": [det(1, 900.0, 700.0), det(1, 901.0, 701.0)]}
    labels = {"2": {cand["stable_id"]: {
        "role": "专家", "seat": 5, "n_people": 2,
        "cells": [{"cell": 2, "role": "专家", "seat": 4}]}}}
    got = judge_donations_from_candidates(labels, {2: [cand]})
    assert list(got) == [4], got
    assert [(s, k) for s, k, _d in got[4]] == [(cand["stable_id"], 2)]


def test_a_whole_candidate_judge_without_a_number_is_not_collected():
    """整张标了「专家」却没选号的不收——不知道是哪一位。"""
    from pipeline.judges import judge_donations_from_candidates
    cand = {"stable_id": "exit@100#900,700", "best_dets": [det(1, 900.0, 700.0)]}
    for bad in (None, "", 0, 99):
        labels = {"2": {cand["stable_id"]: {"role": "专家", "seat": bad}}}
        assert judge_donations_from_candidates(labels, {2: [cand]}) == {}, bad


# --- 位置分散的备选（用户 2026-09-19）-------------------------------------
#
# 新能源2 左侧确实坐着 5 位，但第 4、5 位相距 cx 80px、y2 40px，**正好落在归并
# 容差之内**（90/45），合成了一个 1701 帧的候选——其余三位各约 900 帧。
#
# 扫过容差与单链聚类都分不开：相邻评审人的位置本来就重叠（同一人的片段 cx 跨度
# 达 70-205px，而人与人的间距最小只有 80px），任何全局阈值要么并两个人、要么把
# 一个人劈成两半。这类阈值该由人判断，不该由参数决定。
#
# 真正可修的是**备选池的取法**。实测那个合并候选：底层 896 帧在专家4 那一侧、
# 819 帧在专家5 那一侧，而 16 格池子里专家5 占 15 格、专家4 只有 1 格——池子按
# 「面积×清晰度」全局排名取，谁的画面分数高谁就通吃。材料一直都在，只是没浮上来。
#
# 前 16 格**一格都不能动**（用户已经标了其中几格，动了就指到别的图上），所以
# 位置分散的那些**追加在后面**。


def _two_cluster_dets(n=30):
    """两簇：cx=560 的框小（分数低）、cx=640 的框大（分数高），各 n 帧。"""
    out = []
    for i in range(n):
        out += judge_track(10, 560.0, 640.0, 1, order=1 + i % 6,
                           t0=i, frame=f"a{i}.jpg")
    for i in range(n):
        out += [det(11, 640.0, 600.0, order=1 + i % 6, t=100 + i,
                    h=0.45 * H, width=110.0, frame=f"b{i}.jpg")]
    return out


def test_the_spare_tail_covers_the_under_represented_position():
    """分数低的那一侧也必须在池里出现——它才是被并进来的另一个人。"""
    from pipeline.judges import judge_candidates
    got = judge_candidates(_two_cluster_dets(), frame_w=W, frame_h=H,
                           min_frames=10)
    assert len(got) == 1, [c["stable_id"] for c in got]
    pool = got[0]["pool_dets"]
    lo = [d for d in pool if d.cx < 600]
    assert len(lo) >= 4, (
        f"池里 {len(pool)} 格，分数低的那一侧只有 {len(lo)} 格——"
        f"另一个人几乎浮不上来")


def test_the_first_sixteen_cells_are_untouched_by_the_tail(monkeypatch):
    """追加的那一批不许动前 `JUDGE_CELL_POOL` 格——人已经标过它们了。

    判据是「把追加关掉，前 16 格一帧不差」，不能拿 `select_best_frames(dets, 4)`
    去比：`judge_candidates` 喂给排序的是**按片段重组过**的次序，分数相同的帧
    靠稳定排序保留输入次序，两边输入次序不同，结果自然不同——那样比出来的红
    是夹具的错，不是代码的错。
    """
    from pipeline.judges import judge_candidates
    dets = _two_cluster_dets()
    got = judge_candidates(dets, frame_w=W, frame_h=H, min_frames=10)[0]
    monkeypatch.setattr(config, "JUDGE_CELL_SPREAD", 0)
    plain = judge_candidates(dets, frame_w=W, frame_h=H, min_frames=10)[0]
    assert [d.t for d in got["pool_dets"][:config.JUDGE_CELL_POOL]] == \
           [d.t for d in plain["pool_dets"]]
    assert [d.t for d in got["best_dets"]] == \
           [d.t for d in got["pool_dets"][:config.JUDGE_CELL_COUNT]]


def test_a_single_position_candidate_still_gets_a_full_pool():
    """只有一个人（位置单一）时照旧取满，不因为分不出两侧就少给。"""
    from pipeline.judges import judge_candidates
    dets = []
    for i in range(40):
        dets += judge_track(10, 300.0, 800.0, 1, order=1 + i % 6,
                            t0=i, frame=f"s{i}.jpg")
    got = judge_candidates(dets, frame_w=W, frame_h=H, min_frames=10)[0]
    assert len(got["pool_dets"]) == config.JUDGE_CELL_POOL + config.JUDGE_CELL_SPREAD
    assert len({id(d) for d in got["pool_dets"]}) == len(got["pool_dets"])


# --- 冲突不能挡住修复的工具（用户 2026-09-19，电子27）---------------------
#
# 两位候选被指到同一号时 `judge_slots` 抛 ValueError——挑一个取决于字典顺序，
# 这条拒绝是对的。但它同时**挡住了挑选页的渲染**，而挑选页正是改这个号的唯一
# 地方：页面出不来 → 改不了 → 页面永远出不来。实测电子27 就卡在这里。
#
# 分工：登记表那一侧继续硬拒（产出一份猜出来的座次就是静默出错）；挑选页把
# 冲突的那几位降级为「未指号」，其余人的号照旧，并在页面上把冲突写出来。


def test_conflicts_are_reported_without_raising():
    """`judge_number_conflicts` 只报告，不抛——调用方据此决定拒还是降级。"""
    from pipeline.judges import judge_candidates, judge_number_conflicts
    cands = judge_candidates(_scene(), frame_w=W, frame_h=H, min_frames=10)
    labels = {"judges": {cands[0]["stable_id"]: {"judge": 4},
                         cands[1]["stable_id"]: {"judge": 4},
                         cands[2]["stable_id"]: {"judge": 2}}}
    got = judge_number_conflicts(cands, labels)
    assert list(got) == [4], got
    assert sorted(got[4]) == sorted([cands[0]["stable_id"], cands[1]["stable_id"]])


def test_no_conflict_reports_nothing():
    from pipeline.judges import judge_candidates, judge_number_conflicts
    cands = judge_candidates(_scene(), frame_w=W, frame_h=H, min_frames=10)
    labels = {"judges": {cands[0]["stable_id"]: {"judge": 1},
                         cands[1]["stable_id"]: {"judge": 2}}}
    assert judge_number_conflicts(cands, labels) == {}


def test_dropping_conflicts_keeps_everyone_elses_number():
    """降级只动冲突的那几位——别人的号是他们各自的工作，不能一起清掉。"""
    from pipeline.judges import (judge_candidates, judge_slots,
                                 without_conflicting_numbers)
    cands = judge_candidates(_scene(), frame_w=W, frame_h=H, min_frames=10)
    labels = {"judges": {cands[0]["stable_id"]: {"judge": 4},
                         cands[1]["stable_id"]: {"judge": 4},
                         cands[2]["stable_id"]: {"judge": 2}}}
    clean = without_conflicting_numbers(cands, labels)
    # 冲突的两位不再带号，第三位的 2 号保留
    assert clean["judges"][cands[2]["stable_id"]]["judge"] == 2
    assert clean["judges"][cands[0]["stable_id"]].get("judge") is None
    assert clean["judges"][cands[1]["stable_id"]].get("judge") is None
    # 清理之后不再抛
    slots = judge_slots(cands, clean)
    assert slots[2] is not None and slots[2]["stable_id"] == cands[2]["stable_id"]


def test_dropping_conflicts_keeps_their_other_marks():
    """只清号，逐格标注与补选一概保留——那是另一件工作。"""
    from pipeline.judges import judge_candidates, without_conflicting_numbers
    cands = judge_candidates(_scene(), frame_w=W, frame_h=H, min_frames=10)
    labels = {"judges": {
        cands[0]["stable_id"]: {"judge": 4, "cells": {"2": "x"}, "picks": [7]},
        cands[1]["stable_id"]: {"judge": 4}}}
    clean = without_conflicting_numbers(cands, labels)
    rec = clean["judges"][cands[0]["stable_id"]]
    assert rec["cells"] == {"2": "x"} and rec["picks"] == [7]
    assert labels["judges"][cands[0]["stable_id"]]["judge"] == 4, "原字典被改了"


def test_the_registry_still_refuses_a_conflict(monkeypatch):
    """登记表那一侧不降级：产出一份猜出来的座次就是静默出错。"""
    import pytest
    from pipeline.judges import judge_candidates, judge_slots
    cands = judge_candidates(_scene(), frame_w=W, frame_h=H, min_frames=10)
    labels = {"judges": {cands[0]["stable_id"]: {"judge": 4},
                         cands[1]["stable_id"]: {"judge": 4}}}
    with pytest.raises(ValueError, match="专家 4"):
        judge_slots(cands, labels)
