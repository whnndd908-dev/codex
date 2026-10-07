"""Task 51 · 要做的二：座位登记表（`pipeline/registry.py`）。

用户原话：「最核心的工作是通过人工复核，将每个项目的每个座位的人确定下来，
并存储下来，接下来针对抽好的画面可以反复用不同的 VLM 进行识别着装颜色和
款式」。

登记表是这个项目的核心资产：人工复核一次，之后换 VLM、改提示词、加字段都
只需重跑打标，不必重跑检测与人工复核。因此它必须自己带上**检测指纹**——
它记的是帧路径与检测框，换一次检测这些全都失效（Task 46 的血教训：企业行序
变动后重检测，旧标注里少数编号碰巧还在、但那已经是另一个人，错配不报错）。
"""
import base64
import json
import re
from pathlib import Path

import pytest

from pipeline import config, registry
from pipeline.detect import Detection
from pipeline.schedule import Segment

W = 1440
H = W * 0.75
RIGHT_CX = 0.55 * W
STANDING_H = 0.55 * H


def mkseg(order=2, presenter_num=2, company="测试企业"):
    return Segment(order=order, company=company, start=0, end=1000,
                   presenter_num=presenter_num, presenters=())


def det(tid, cx, h, kind="entry", order=2, t=0.0, bottom=1000.0, width=30.0):
    half = width / 2.0
    return Detection(order, kind, t, "nonexistent_frame.jpg", tid,
                     cx - half, bottom - h, cx + half, bottom, 0.9)


def track(tid, cx, h, n_frames, kind="entry", order=2, start_t=0.0,
          bottom=1000.0, width=30.0):
    return [det(tid, cx, h, kind=kind, order=order, t=start_t + i,
                bottom=bottom, width=width) for i in range(n_frames)]


def _scene():
    """一段、3 人项目：座位 1 由就座阶段判定，另有两个进出场候选待人工标注。"""
    seg = mkseg(order=5, presenter_num=3)
    dets = (
        track(1, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=5)
        + track(41, RIGHT_CX + 100, STANDING_H, 4, kind="exit", order=5)
        + track(42, RIGHT_CX + 700, STANDING_H, 4, kind="exit", order=5)
    )
    seats_by_order = {5: {1: 1}}          # track 1 → 座位 1
    return seg, dets, seats_by_order


def test_registry_records_source_per_seat():
    """每个 (order, seat) 都要写明这个人是**怎么定下来的**：就座阶段判定
    （seated）、进出场人工递补（entryexit）、还是按格位切分（cell_split）。

    不记来源，换 VLM 重跑时无从判断哪些座位是人工确认过的、哪些只是算法
    的几何排序结果——而后者恰恰是本项目反复出错的地方。
    """
    from pipeline.entryexit import entryexit_candidates

    seg, dets, seats_by_order = _scene()
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 42)
    labels = {"5": {cand["stable_id"]: {"role": "创业者", "seat": 2,
                                        "n_people": 1, "note": ""}}}

    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, labels, detections_fp="FP1", sheet="测试组")

    seats = reg["segments"]["5"]["seats"]
    assert set(seats) == {"1", "2"}, (
        f"座位 1（就座）与座位 2（人工递补）都该登记，座位 3 无人可定、"
        f"不该凭空造一条，实得 {sorted(seats)}"
    )
    assert seats["1"]["source"] == "seated", seats["1"]
    assert seats["2"]["source"] == "entryexit", seats["2"]
    assert seats["1"]["confirmed"] is False, (
        "就座阶段的座位号来自几何排序，没有经过人工确认——不许标成已确认"
    )
    assert seats["2"]["confirmed"] is True, (
        "进出场递补的座位是用户在挑选页一个个点出来的，就是人工确认"
    )
    assert reg["segments"]["5"]["unresolved_seats"] == [3], (
        f"定不下来的座位必须如实列出，实得 {reg['segments']['5']}"
    )
    assert seats["2"]["stable_id"] == cand["stable_id"], (
        "进出场来源必须记下 Task 46 的稳定标识，重检测后才有对齐的可能"
    )
    for rec in seats.values():
        assert rec["frames"], f"每个座位都要记下用到的帧，实得 {rec}"
        for fr in rec["frames"]:
            assert set(fr) >= {"frame_path", "t", "box"}, fr
            assert len(fr["box"]) == 4, fr


def test_registry_roundtrip(tmp_path):
    """写盘 → 读回，内容逐字相同（指纹一致时不作废、不改写）。"""
    seg, dets, seats_by_order = _scene()
    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, {}, detections_fp="FP1", sheet="测试组")

    path = registry.registry_path(tmp_path, "测试组")
    registry.write_seat_registry(reg, path)
    assert path.exists(), f"登记表应写到 {path}"
    assert path.name == "seat_registry_测试组.json", path.name

    back = registry.load_seat_registry(path, detections_fp="FP1")
    assert back == reg, "指纹一致时读回的内容必须与写出的逐字相同"
    assert back["_meta"]["detections_fingerprint"] == "FP1"
    assert json.loads(path.read_text(encoding="utf-8")) == reg


def test_registry_fingerprint_mismatch_is_detected(tmp_path, capsys):
    """换了检测之后登记表必须被判定失效并**打印**——它记的是帧路径与检测
    框，重检测后这些指向的是另一批框，照常使用等于把别人的画面当成这个人的
    （Task 46 就是被这个坑过一次）。"""
    seg, dets, seats_by_order = _scene()
    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, {}, detections_fp="FP1", sheet="测试组")
    path = registry.registry_path(tmp_path, "测试组")
    registry.write_seat_registry(reg, path)

    got = registry.load_seat_registry(path, detections_fp="FP2")
    out = capsys.readouterr().out
    assert got == {}, f"指纹不符必须整份作废，实得 {got}"
    assert "FP1" in out and "FP2" in out, (
        f"作废必须打印出两个指纹，让用户看得出是哪一份对不上：{out!r}"
    )

    # 对照：不提供指纹时不作废，但要明说跳过了校验（静默使用同样是缺陷）
    same = registry.load_seat_registry(path)
    out2 = capsys.readouterr().out
    assert same == reg, "未提供指纹时应原样返回（调用方自己负责）"
    assert "跳过" in out2 or "未提供" in out2, (
        f"跳过校验必须打印，实得 {out2!r}"
    )


def test_registry_covers_cell_split_seats():
    """按格位切分定下来的座位同样要进登记表，并记下**是哪几格**、只用那几张图。

    这是登记表存在的意义之一：切分是一次性的人工判断，之后换 VLM 重跑时
    必须还能找回「用第 1、3 格那两张」这条信息。
    """
    from pipeline.entryexit import entryexit_candidates

    seg = mkseg(order=9, presenter_num=3)
    dets = (track(1, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=9)
            + track(12, RIGHT_CX + 100, STANDING_H, 4, kind="exit", order=9))
    seats_by_order = {9: {1: 1}}
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 12)
    assert len(cand["best_dets"]) == 4, "夹具前提：该候选恰有 4 张证据帧"

    labels = {"9": {cand["stable_id"]: {
        "role": "创业者", "seat": 3, "n_people": 2,
        "note": "1/3是创业者3，2/4是创业者2"}}}

    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, labels, detections_fp="FP1", sheet="测试组")
    seats = reg["segments"]["9"]["seats"]
    assert set(seats) == {"1", "2", "3"}, (
        f"座位 1 就座、2 与 3 由格位切分定下，实得 {sorted(seats)}"
    )
    for s in ("2", "3"):
        assert seats[s]["source"] == "cell_split", seats[s]
        assert all(c in (1, 2, 3, 4) for c in seats[s]["cell_indices"]), seats[s]
        # Task 56A：`1/3` 是同一个人占两格 → 两格的帧都要留档。旧实现只留
        # 一张（用户 2026-08-24 实测发现），这里钉死张数与格位数一致。
        assert len(seats[s]["frames"]) == len(seats[s]["cell_indices"]) == 2, (
            f"座位 {s} 由两格解引用而来，两格的帧都要留下，实得 "
            f"{len(seats[s]['frames'])} 张 / 格位 {seats[s]['cell_indices']}"
        )
        assert seats[s]["confirmed"] is True
        assert "1/3是创业者3" in seats[s]["note"], (
            f"备注原文要留档（切分的唯一依据），实得 {seats[s]}"
        )
    assert seats["3"]["cell_indices"] == [1, 3] and seats["2"]["cell_indices"] == [2, 4], (
        f"座位 3 占 1、3 两格，座位 2 占 2、4 两格，实得 "
        f"{[(s, seats[s]['cell_indices']) for s in ('2', '3')]}"
    )
    assert reg["_meta"]["counts_by_source"] == {
        "seated": 1, "entryexit": 0, "cell_split": 2}, reg["_meta"]


def test_registry_cli_writes_file(tmp_path, monkeypatch, capsys):
    """CLI（`python -m pipeline.registry`）能从一组的 work 目录产出登记表。

    全程用 tmp_path 造的合成产物，绝不碰真实 work/。
    """
    from pipeline.detect import save_detections
    from pipeline.schedule import save_segments

    seg, dets, seats_by_order = _scene()
    work = tmp_path / "组"
    work.mkdir()
    save_segments([seg], work / "segments.json")
    save_detections(dets, work / "detections.json")
    (work / "seats.json").write_text(
        json.dumps({"5": {"1": 1}}), encoding="utf-8")

    rc = registry.main(["--work", str(work), "--sheet", "测试组"])
    assert rc == 0, f"CLI 应正常退出，实得 {rc}"
    out_path = registry.registry_path(work, "测试组")
    assert out_path.exists(), f"CLI 应写出 {out_path}"

    from pipeline.entryexit import detections_fingerprint
    fp = detections_fingerprint(work / "detections.json")
    written = json.loads(out_path.read_text(encoding="utf-8"))
    # 断言指纹**确实等于当前检测的哈希**，而不是只断言「读回来是真值」——
    # 写 None 时 load_seat_registry 会走「登记表自身没记指纹，跳过校验」
    # 那条路照常返回内容，只看真值恒绿（自审变异 x13 当场漏网）。
    assert written["_meta"]["detections_fingerprint"] == fp, (
        f"CLI 必须把当前 detections.json 的指纹写进登记表，实得 "
        f"{written['_meta']['detections_fingerprint']!r}，应为 {fp!r}"
    )
    reg = registry.load_seat_registry(out_path, detections_fp=fp)
    assert reg, "指纹一致时应正常读回"
    assert registry.load_seat_registry(out_path, detections_fp="另一份检测") == {}, (
        "换一份检测后 CLI 写出的登记表必须能被判定失效"
    )
    assert reg["segments"]["5"]["seats"]["1"]["source"] == "seated"
    printed = capsys.readouterr().out
    assert "5" in printed and str(out_path) in printed, (
        f"CLI 要打印写到哪里、定下了多少个座位，实得 {printed!r}"
    )


def test_registry_respects_rejected_seat():
    """人工否决过的座位（`seat_overrides` 的 rejected）绝不能被就座阶段的
    结果原样写回登记表——用户已经明说「这个座位判错了」，写回等于当没听见，
    而登记表正是下一步打标唯一的输入。

    两种情形都测：换上了人（来源必须是 entryexit，不是 seated），以及换不
    上人（那就该缺着、进 unresolved_seats，不许用被否决的那个人填坑）。
    """
    from pipeline.entryexit import entryexit_candidates

    seg, dets, seats_by_order = _scene()          # 就座阶段把座位 1 判给 track 1
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 41)

    replaced = registry.build_seat_registry(
        [seg], dets, seats_by_order,
        {"5": {cand["stable_id"]: {"role": "创业者", "seat": 1, "n_people": 1}},
         "seat_overrides": {"5": {"1": {"rejected": True, "note": "判错了"}}}},
        detections_fp="FP1", sheet="测试组")
    rec = replaced["segments"]["5"]["seats"]["1"]
    assert rec["source"] == "entryexit", (
        f"座位 1 已被人工否决并改判给进出场候选，登记表不得记成就座结果，实得 {rec}"
    )
    assert rec["track_id"] == 41 and rec["confirmed"] is True, rec
    assert replaced["segments"]["5"]["rejected_seats"] == [1], replaced["segments"]["5"]

    only_rejected = registry.build_seat_registry(
        [seg], dets, seats_by_order,
        {"seat_overrides": {"5": {"1": {"rejected": True, "note": "判错了"}}}},
        detections_fp="FP1", sheet="测试组")
    seg_rec = only_rejected["segments"]["5"]
    assert "1" not in seg_rec["seats"], (
        f"没人补上时座位 1 就该缺着，不许把被否决的就座结果写回来，实得 {seg_rec['seats']}"
    )
    assert seg_rec["unresolved_seats"] == [1, 2, 3], seg_rec


# ===========================================================================
# Task 53 · 需求六：跨段流转要落进登记表并打上标记
#
# 实测城市2组段 15 有一条 `role='下一组创业者'
# note='1/4是下一场创业者1；2/3是下一场创业者2'`，而段 16 恰好缺人。这些
# **画面属于段 15 的时间窗口**——登记表若不记下这一点，日后拿 `frames` 里
# 的时间戳去核对段 16 的时间范围必然对不上，而没有任何一处会报错。
# ===========================================================================

def _cross_scene():
    """段 15（一条跨段备注）+ 段 16（需 2 人、就座只定了座位 1、无标注）。"""
    from pipeline.entryexit import entryexit_candidates

    src = mkseg(order=15, presenter_num=1, company="源企业")
    tgt = mkseg(order=16, presenter_num=2, company="目标企业")
    dets = (
        track(90, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=16)
        + track(35, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=15)
    )
    cand = next(c for c in entryexit_candidates(dets, src) if c["track_id"] == 35)
    assert len(cand["best_dets"]) == 4, (
        f"夹具前提：源段候选要有 4 张证据帧才切得出 4 格，实得 "
        f"{len(cand['best_dets'])}")
    labels = {"15": {cand["stable_id"]: {
        "role": "下一组创业者", "seat": None, "n_people": 2,
        "note": "1/4是下一场创业者1；2/3是下一场创业者2"}}}
    seats_by_order = {16: {90: 1}}        # 座位 1 已由就座阶段占着
    return src, tgt, dets, labels, seats_by_order


def test_registry_records_cross_segment_fill():
    """跨段流转的座位要真的进登记表，并带 `cross_segment=1` 与 `src_order`。

    夹具让段 16 **一条标注都没有**——它的座位 2 若被填上，只可能来自段 15
    那条备注。
    """
    src, tgt, dets, labels, seats_by_order = _cross_scene()
    reg = registry.build_seat_registry(
        [src, tgt], dets, seats_by_order, labels,
        detections_fp="FPX", sheet="城市2测试")

    seats16 = reg["segments"]["16"]["seats"]
    assert sorted(seats16) == ["1", "2"], (
        f"段 16 的两个座位都该定下来（1 来自就座、2 来自跨段流转），实得 "
        f"{sorted(seats16)}")
    rec = seats16["2"]
    assert rec["cross_segment"] == 1, f"跨段流转必须打标记，实得 {rec}"
    assert rec["src_order"] == 15, (
        f"这几帧来自段 15，必须记下来源段号，实得 {rec}")
    assert rec["source"] == "cell_split" and rec["cell_indices"] == [2, 3], (
        f"它确实是按格位切出来的，且 2/3 两格是同一个人——两格的帧都要留下"
        f"（Task 56A），实得 {rec}")
    assert rec["note"] == "1/4是下一场创业者1；2/3是下一场创业者2", (
        f"备注原文取自**源段**的标注，不是目标段（目标段一条标注都没有），"
        f"实得 {rec['note']!r}")
    assert len(rec["frames"]) == 2, (
        f"2/3 两格是同一个人，两格各一张画面，实得 {len(rec['frames'])} 张")

    # 本段自己的记录不许被跨段标记污染
    assert reg["segments"]["16"]["seats"]["1"]["cross_segment"] == 0, (
        f"就座阶段定下的座位不是跨段来的，实得 {reg['segments']['16']['seats']['1']}")
    # 源段自己不该多出一个人（那条备注说的是下一场的人）
    assert reg["segments"]["15"]["unresolved_seats"] == [1], (
        f"段 15 自己仍缺座位 1（跨段候选不占它的名额），实得 "
        f"{reg['segments']['15']}")


def test_registry_cross_segment_does_not_overwrite_occupied_seat():
    """目标段的座位已被就座阶段占着 → 跨段流转不许顶掉它。

    夹具让备注指向的**两个**座位都被占着，否则这条路径根本走不到。
    """
    src, tgt, dets, labels, _seats = _cross_scene()
    dets = dets + track(91, RIGHT_CX + 600, STANDING_H, 5,
                        kind="seated", order=16)
    reg = registry.build_seat_registry(
        [src, tgt], dets, {16: {90: 1, 91: 2}}, labels,
        detections_fp="FPX", sheet="城市2测试")
    seats16 = reg["segments"]["16"]["seats"]
    assert [s["source"] for s in seats16.values()] == ["seated", "seated"], (
        f"两个座位都由就座阶段占着，跨段候选一个也不许顶掉，实得 {seats16}")
    assert all(s["cross_segment"] == 0 for s in seats16.values()), seats16


def test_cross_segment_reaches_the_csv_row():
    """跨段标记要一路走到 CSV 那一行（`gallery.annotation_to_row`）。

    登记表记住了没用——最终交付的是 CSV，下游（Stata 那一侧）只看得到它。
    普通行也必须有这两列，否则列集合会因行而异。
    """
    from pipeline.annotate import PersonAnnotation
    from pipeline.gallery import annotation_to_row

    a = PersonAnnotation(person_id="p", seg_order=16, company="目标企业",
                         seat_index=2, presenter_num=2, gender_female=0,
                         speaks_pre=0, speaks_qa=0, seg_start=0, seg_end=1)
    row = annotation_to_row(a)
    assert row["cross_segment"] == 0 and row["src_seg_order"] == 0, (
        f"普通行也要有这两列（值为 0），否则列集合因行而异，实得 "
        f"{ {k: row.get(k) for k in ('cross_segment', 'src_seg_order')} }")

    a.cross_segment = 1
    a.src_seg_order = 15
    row2 = annotation_to_row(a)
    assert row2["cross_segment"] == 1 and row2["src_seg_order"] == 15, (
        f"跨段行必须把来源段号带进 CSV，实得 {row2['cross_segment']}/"
        f"{row2['src_seg_order']}")


# ===========================================================================
# Task 54 · 需求二：核对页改用 VLM 实际看到的图
#
# 用户 2026-08-23 原话：「不需要给出座位核对.html，给我看 VLM 看的图让我核对
# 更准」。
#
# 此前控制器拿登记表里的框自己裁了一份 `座位核对.html`——那不是 VLM 看到的
# 东西。核对一张模型没看过的图，得出的结论对模型不成立，而且不会有任何一处
# 报错。现在一律由 `annotate.build_evidence_sheet` 生成（与 `assemble` 送进
# 模型前的最后一步同一个函数、同一批帧、同一组默认参数）。
# ===========================================================================

def _real_frames(tmp_path, n=4):
    """n 张 1440×1080 的真图——build_evidence_sheet 会按检测框坐标裁剪，
    图太小会越界。"""
    import numpy as np
    from PIL import Image

    out = []
    for i in range(n):
        p = tmp_path / f"f{i}.jpg"
        Image.fromarray(np.full((1080, 1440, 3), 60 + i * 20, dtype=np.uint8)).save(p)
        out.append(p)
    return out


def _scene_with_cell_split(tmp_path):
    """一段 3 人：座位 1 由就座阶段判定（2×2 四帧），座位 2、3 由一个
    「框里挤了 3 个人」的候选按格位切出来（各 1 帧）。"""
    from pipeline.entryexit import entryexit_candidates

    frames = _real_frames(tmp_path)
    seg = mkseg(order=5, presenter_num=3)
    dets = []
    for i in range(4):
        dets.append(Detection(5, "seated", float(i), frames[i], 1,
                              RIGHT_CX + 380, 1000 - STANDING_H,
                              RIGHT_CX + 420, 1000, 0.9))
        dets.append(Detection(5, "exit", float(i), frames[i], 41,
                              RIGHT_CX + 80, 1000 - STANDING_H,
                              RIGHT_CX + 120, 1000, 0.9))
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 41)
    assert len(cand["best_dets"]) == 4, "夹具前提：该候选要有 4 张证据帧"
    labels = {"5": {cand["stable_id"]: {
        "role": "创业者", "seat": None, "n_people": 3,
        "cells": [{"cell": 1, "role": "创业者", "seat": 2},
                  {"cell": 3, "role": "创业者", "seat": 3}]}}}
    reg = registry.build_seat_registry([seg], dets, {5: {1: 1}}, labels,
                                       detections_fp="FP1", sheet="测试组")
    seats = reg["segments"]["5"]["seats"]
    assert set(seats) == {"1", "2", "3"}, f"夹具前提，实得 {sorted(seats)}"
    assert seats["1"]["source"] == "seated" and len(seats["1"]["frames"]) == 4, seats["1"]
    assert seats["2"]["source"] == "cell_split" and seats["2"]["cell_indices"] == [1], seats["2"]
    assert len(seats["2"]["frames"]) == 1, (
        f"夹具前提：这两个座位各只占一格（cells 里 1、3 号格分属不同座位），"
        f"因此各只有一帧——否则测不到「单张 vs 2×2」的差异。实得 "
        f"{len(seats['2']['frames'])}")
    return reg, cand


def test_preview_uses_same_evidence_builder(tmp_path, monkeypatch):
    """核对页的图必须由 `annotate.build_evidence_sheet` 生成——不许另拼
    一份。

    另拼一份的后果：裁法、留边、红框哪怕差一点，人核对的就是一张模型没
    看过的图，结论对模型不成立且不报错。这里把 build_evidence_sheet 换成
    记录器，断言每个座位都真的走了它，且喂进去的正是登记表留档的那几帧。
    """
    reg, _cand = _scene_with_cell_split(tmp_path)
    calls = []

    def _fake(paths, boxes, **kw):
        calls.append({"paths": [str(p) for p in paths], "boxes": list(boxes),
                      "kw": dict(kw)})
        return b"PNG-STUB"

    monkeypatch.setattr(registry, "build_evidence_sheet", _fake)
    out = registry.render_vlm_preview(reg, tmp_path / "vlm.html")
    html = out.read_text(encoding="utf-8")

    assert len(calls) == 3, (
        f"三个已定座位都该走 build_evidence_sheet，实得 {len(calls)} 次")
    seats = reg["segments"]["5"]["seats"]
    for i, seat in enumerate(("1", "2", "3")):
        want = [fr["frame_path"] for fr in seats[seat]["frames"]]
        assert calls[i]["paths"] == want, (
            f"座位 {seat} 应原样喂登记表留档的帧 {want}，实得 {calls[i]['paths']}")
    assert all(not c["kw"].get("label_cells") for c in calls), (
        "VLM 那份证据图不带格位编号（多余的数字会干扰模型判读），核对页必须"
        "与它一致，否则又不是「模型看到的那张」了")
    assert base64.b64encode(b"PNG-STUB").decode() in html, (
        "页面里放的必须是 build_evidence_sheet 的产物本身")


def test_preview_cell_split_uses_single_image(tmp_path, monkeypatch):
    """格位切分的座位只送**那一格的单张**图，不是父候选的 2×2 四格。

    沿用 2×2 会让人核对四格里另外三个人——而模型看到的只有一格；这正是
    Task 51 切分要消灭的那个错误在核对页上的重演。
    """
    reg, cand = _scene_with_cell_split(tmp_path)
    calls = []
    monkeypatch.setattr(registry, "build_evidence_sheet",
                        lambda paths, boxes, **kw: calls.append(list(paths)) or b"P")
    registry.render_vlm_preview(reg, tmp_path / "vlm.html")

    assert [len(c) for c in calls] == [4, 1, 1], (
        f"座位1（就座）四帧拼 2×2；座位2、3（格位切分）各只有一张，实得 "
        f"{[len(c) for c in calls]}")
    from pipeline.entryexit import cell_dets
    assert str(calls[1][0]) == str(cell_dets(cand, 1)[0].frame_path), (
        "格位 1 那一张必须与 entryexit.cell_dets 给出的完全相同——切分与核对"
        "两处各切一次必然日后走样")
    assert str(calls[2][0]) == str(cell_dets(cand, 3)[0].frame_path), (
        "格位 3 同理")


def test_preview_marks_what_vlm_sees(tmp_path):
    """每张卡片都要写明：座位号、来源、cell_indices（若有），以及**这张图是不是
    就是 VLM 会看到的那张**。

    帧文件已经不在磁盘上时那张卡片就**不是**——此时说「这就是模型看的图」
    等于骗人，必须逐卡如实写，不能在页首笼统说一句。
    """
    reg, _cand = _scene_with_cell_split(tmp_path)
    # 座位 3 的帧指向一个不存在的文件：核对页必须如实说这张重建不出来
    reg["segments"]["5"]["seats"]["3"]["frames"][0]["frame_path"] = str(
        tmp_path / "已被删掉.jpg")

    out = registry.render_vlm_preview(reg, tmp_path / "vlm.html")
    html = out.read_text(encoding="utf-8")
    cards = re.findall(r'<div class="card"([^>]*)>(.*?)(?=<div class="card"|$)',
                       html, re.DOTALL)
    assert len(cards) == 3, f"三个已定座位三张卡片，实得 {len(cards)}"

    by_seat = {re.search(r'data-seat="(\d+)"', a).group(1): (a, b) for a, b in cards}
    a1, t1 = by_seat["1"]
    assert 'data-source="seated"' in a1 and "就座" in t1, t1[:400]
    assert "座位 1" in t1 and "就是模型会看到的那张" in t1, t1[:400]

    a2, t2 = by_seat["2"]
    assert 'data-cell="1"' in a2, a2
    assert "格位 1" in t2, f"格位切分的卡片要写明是第几格：{t2[:400]}"
    assert "单张" in t2, f"要写明模型只看这一张，不是四格拼图：{t2[:400]}"

    a3, t3 = by_seat["3"]
    assert 'data-vlm="0"' in a3, a3
    assert "这里没有图" in t3 and "已被删掉.jpg" in t3, (
        f"帧文件不在了要如实说，不能声称这就是模型看到的图：{t3[:400]}")
    assert "✔" not in t3, f"没有图的卡片不许打勾：{t3[:400]}"


def test_preview_renders_a_real_evidence_sheet(tmp_path):
    """不打桩、真跑一遍：页面里放的是真的 PNG（base64），且尺寸与
    `build_evidence_sheet` 直接生成的那张**逐字节相同**。"""
    from pipeline.annotate import build_evidence_sheet

    reg, _cand = _scene_with_cell_split(tmp_path)
    out = registry.render_vlm_preview(reg, tmp_path / "vlm.html")
    html = out.read_text(encoding="utf-8")

    seat1 = reg["segments"]["5"]["seats"]["1"]
    expect = build_evidence_sheet([Path(f["frame_path"]) for f in seat1["frames"]],
                                  [tuple(f["box"]) for f in seat1["frames"]])
    assert base64.b64encode(expect).decode() in html, (
        "核对页上的那张图必须与 build_evidence_sheet 直接生成的逐字节相同")


def test_preview_cli_writes_the_page(tmp_path, monkeypatch, capsys):
    """`python -m pipeline.registry --preview` 产出核对页；**不跑任何
    run_all 阶段、不调 VLM**。"""
    frames = _real_frames(tmp_path)
    seg = mkseg(order=5, presenter_num=2)
    dets = [Detection(5, "seated", float(i), frames[i], 1,
                      RIGHT_CX + 380, 1000 - STANDING_H, RIGHT_CX + 420, 1000, 0.9)
            for i in range(4)]
    work = tmp_path / "work"
    work.mkdir()
    (work / "segments.json").write_text(json.dumps({
        "segments": [{"order": 5, "company": "测试企业", "start": 0, "end": 1000,
                      "presenter_num": 2, "presenters": []}]}, ensure_ascii=False),
        encoding="utf-8")
    (work / "detections.json").write_text(json.dumps(
        [{"seg_order": d.seg_order, "kind": d.kind, "t": d.t,
          "frame_path": str(d.frame_path), "track_id": d.track_id,
          "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2, "conf": d.conf}
         for d in dets], ensure_ascii=False), encoding="utf-8")
    (work / "seats.json").write_text(json.dumps({"5": {"1": 1}}), encoding="utf-8")

    rc = registry.main(["--work", str(work), "--sheet", "测试组", "--preview"])
    assert rc == 0, capsys.readouterr()
    page = registry.preview_path(work, "测试组")
    assert page.exists(), f"--preview 应写出 {page}"
    assert "VLM" in page.read_text(encoding="utf-8")
    assert "vlm_preview_测试组.html" in capsys.readouterr().out, "写到哪里必须说出来"


# ---------------------------------------------------------------------------
# Task 56B · 登记表落后于人工标注时必须报错，不许静默
#
# 真实事故（2026-08-23）：用户 14:03 在 entry_review.html 上重新标注并导出
# `entry_labels_<组>.json`，而 `seat_registry_<组>.json` 与
# `vlm_preview_<组>.html` 还停在 13:53——产出登记表那一步是手敲
# `python -m pipeline.registry --preview`、**不自动**，两边都不报错，用户
# 对着 10 分钟前的旧图做了一轮人工核对，只能靠肉眼比对文件时间戳才发现。
# 这条流水线此前已被「产物没重新生成、代码却改了」这类静默陈旧坑过五次
# 以上。因此登记表必须**自证它由哪一份标注产出**：读回时对不上就报错。
# ---------------------------------------------------------------------------

def _labels_file(work, sheet, body):
    """在 work 里落一份 `entry_labels_<SHEET>.json`（内容随便，只看摘要）。"""
    p = registry.labels_path(work, sheet)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return p


def _built(labels_path=None, *, sheet="测试组", labels=None, fp="FP1"):
    seg, dets, seats_by_order = _scene()
    return registry.build_seat_registry(
        [seg], dets, seats_by_order, labels or {},
        detections_fp=fp, sheet=sheet, labels_path=labels_path)


def test_registry_records_labels_digest(tmp_path):
    """登记表要在 `_meta` 里记下**它产自哪一份标注**：内容摘要 + 文件名。

    用内容摘要而不是 mtime：用户可能原样重存一次文件（mtime 变了内容没变，
    不该报警），也可能改了内容而 mtime 因为拷贝反而更早——mtime 变新变旧
    都骗人。
    """
    work = tmp_path / "组"
    work.mkdir()
    lp = _labels_file(work, "测试组", {"5": {"a": {"seat": 1}}})

    meta = _built(lp)["_meta"]
    assert meta["labels_digest"] == registry.labels_digest(lp), (
        f"_meta 必须记下标注的内容摘要，实得 {meta.get('labels_digest')!r}")
    assert meta["labels_digest"], "有标注文件时摘要不该是空的"
    assert meta["labels_file"] == lp.name, (
        f"还要记下是哪一份标注文件，实得 {meta.get('labels_file')!r}")


def test_labels_digest_follows_content_not_mtime(tmp_path):
    """摘要只跟内容走：原样重存（mtime 变新）摘要不变；改了内容即便把
    mtime 改回更早，摘要照样变。"""
    import os

    work = tmp_path / "组"
    work.mkdir()
    lp = _labels_file(work, "测试组", {"5": {"a": {"seat": 1}}})
    d0 = registry.labels_digest(lp)

    lp.write_text(lp.read_text(encoding="utf-8"), encoding="utf-8")
    os.utime(lp, (lp.stat().st_atime + 9999, lp.stat().st_mtime + 9999))
    assert registry.labels_digest(lp) == d0, (
        "原样重存只改 mtime，摘要必须不变，否则一份好好的登记表会被误判陈旧")

    lp.write_text(json.dumps({"5": {"a": {"seat": 2}}}), encoding="utf-8")
    os.utime(lp, (0, 0))
    assert registry.labels_digest(lp) != d0, (
        "内容改了就必须换摘要——哪怕 mtime 被拷贝弄成更早的")
    assert registry.labels_digest(work / "根本不存在.json") == "", (
        "标注文件不存在是合法状态（该组还没人工标过），摘要记为空值")


def test_registry_staleness_flags_relabeled_export(tmp_path):
    """本任务的正题：登记表产出后用户又重新标注了一次 → 判定为陈旧，并把
    两个摘要都给出来。判定函数是**唯一的判定来源**（CLI 与 load 都调它）。"""
    work = tmp_path / "组"
    work.mkdir()
    lp = _labels_file(work, "测试组", {"5": {"a": {"seat": 1}}})
    old_digest = registry.labels_digest(lp)
    registry.write_seat_registry(_built(lp), registry.registry_path(work, "测试组"))

    fresh = registry.registry_staleness(work, "测试组")
    assert fresh.stale is False, f"刚生成就该是新鲜的，实得 {fresh}"
    assert fresh.reason == "ok", fresh
    assert fresh.registry_exists is True and fresh.labels_exists is True, fresh

    # 用户 14:03 在 entry_review.html 上重标一次并导出
    lp.write_text(json.dumps({"5": {"a": {"seat": 3}}}), encoding="utf-8")
    v = registry.registry_staleness(work, "测试组")
    assert v.stale is True, f"标注变了、登记表没重跑 → 必须判陈旧，实得 {v}"
    assert v.reason == "digest_mismatch", v
    assert v.stored_digest == old_digest, v
    assert v.current_digest == registry.labels_digest(lp) != old_digest, v
    assert "registry" in v.message and "--preview" in v.message, (
        f"结论里要直说该重跑哪条命令，实得 {v.message!r}")


def test_registry_staleness_missing_registry_and_missing_labels(tmp_path):
    """两种边界各有明确处置，不许含糊：
      · 还没有登记表 → 陈旧（reason=no_registry），CLI 据此让用户先跑一遍；
      · 该组还没人工标过（标注文件不存在）→ 这是**合法状态**，摘要空值，
        判为一致。
    """
    work = tmp_path / "组"
    work.mkdir()
    none = registry.registry_staleness(work, "测试组")
    assert none.stale is True and none.reason == "no_registry", none
    assert none.registry_exists is False and none.labels_exists is False, none

    path = registry.registry_path(work, "测试组")
    registry.write_seat_registry(_built(None), path)
    v = registry.registry_staleness(work, "测试组")
    assert v.stale is False, (
        f"没有标注文件是合法状态（登记表只含就座阶段的结果），"
        f"不该报陈旧，实得 {v}")
    assert v.current_digest == "" and v.stored_digest == "", v
    assert registry.load_seat_registry(path, detections_fp="FP1"), (
        "没有标注文件时读回必须照常成功")


def test_load_seat_registry_raises_when_labels_changed(tmp_path):
    """读回登记表时摘要对不上 → **抛异常**，不是打印警告。

    打印早就试过了：这条流水线的每一步都在打印，用户 14:03 那次谁也没看见。
    """
    work = tmp_path / "组"
    work.mkdir()
    lp = _labels_file(work, "测试组", {"5": {"a": {"seat": 1}}})
    old_digest = registry.labels_digest(lp)
    path = registry.registry_path(work, "测试组")
    registry.write_seat_registry(_built(lp), path)
    assert registry.load_seat_registry(path, detections_fp="FP1"), "前提：先能读回"

    lp.write_text(json.dumps({"5": {"a": {"seat": 3}}}), encoding="utf-8")
    with pytest.raises(registry.RegistryStaleError) as ei:
        registry.load_seat_registry(path, detections_fp="FP1")
    msg = str(ei.value)
    assert old_digest in msg and registry.labels_digest(lp) in msg, (
        f"异常里要说清「登记表由哪个摘要产出、当前标注是哪个」，实得 {msg!r}")
    assert "python -m pipeline.registry --preview" in msg, (
        f"还要说清怎么修，实得 {msg!r}")

    # 不带检测指纹调用时同样要报：检测指纹那条路上有两条「跳过校验」的分支
    # （调用方没给、登记表自己没记），标注新鲜度绝不能顺着它们一起被跳过。
    with pytest.raises(registry.RegistryStaleError):
        registry.load_seat_registry(path)


def test_load_seat_registry_allow_stale_is_opt_in(tmp_path, capsys):
    """逃生口 `allow_stale=True` 只给诊断用（想看看旧登记表里到底存了
    什么），**默认必须是严格的**，且用了就要大声说出来。"""
    work = tmp_path / "组"
    work.mkdir()
    lp = _labels_file(work, "测试组", {"5": {"a": {"seat": 1}}})
    path = registry.registry_path(work, "测试组")
    reg = _built(lp)
    registry.write_seat_registry(reg, path)
    lp.write_text(json.dumps({"5": {"a": {"seat": 3}}}), encoding="utf-8")

    got = registry.load_seat_registry(path, detections_fp="FP1", allow_stale=True)
    assert got == reg, "逃生口应原样返回内容"
    out = capsys.readouterr().out
    assert "陈旧" in out or "stale" in out.lower(), (
        f"明知陈旧还照用，必须打印，实得 {out!r}")

    import inspect
    sig = inspect.signature(registry.load_seat_registry)
    assert sig.parameters["allow_stale"].default is False, (
        "默认必须严格：逃生口一旦默认打开就等于没做这个任务")


def test_registry_without_labels_digest_is_stale(tmp_path):
    """本任务之前产出的旧登记表 `_meta` 里根本没有 `labels_digest` →
    当作陈旧并说明「旧版登记表不带来源摘要，无法确认是否最新」。

    这条最容易被写成「没有就跳过校验」（沿用检测指纹那条路的口径），而那
    正是静默失效本身：旧文件恰恰是最可能已经落后于标注的那一批。
    """
    work = tmp_path / "组"
    work.mkdir()
    lp = _labels_file(work, "测试组", {"5": {"a": {"seat": 1}}})
    path = registry.registry_path(work, "测试组")
    reg = _built(lp)
    reg["_meta"].pop("labels_digest", None)      # 旧版登记表就是这个样子
    registry.write_seat_registry(reg, path)

    v = registry.registry_staleness(work, "测试组")
    assert v.stale is True and v.reason == "no_digest", v
    assert v.stored_digest is None, (
        f"「压根没记」与「记的是空值（没有标注文件）」是两回事，实得 {v}")
    with pytest.raises(registry.RegistryStaleError) as ei:
        registry.load_seat_registry(path, detections_fp="FP1")
    assert "旧版" in str(ei.value) or "未记录" in str(ei.value), str(ei.value)


def test_load_seat_registry_has_one_judgement_source(tmp_path, monkeypatch):
    """判定只能有一处：`load_seat_registry` 必须调 `registry_staleness`，
    不许自己再写一份比较逻辑（两份迟早各自漂移，而漂移的表现是其中一份
    悄悄不报了）。"""
    work = tmp_path / "组"
    work.mkdir()
    lp = _labels_file(work, "测试组", {"5": {"a": {"seat": 1}}})
    path = registry.registry_path(work, "测试组")
    registry.write_seat_registry(_built(lp), path)
    lp.write_text(json.dumps({"5": {"a": {"seat": 3}}}), encoding="utf-8")

    calls = []

    def fake(work_, sheet_, **kw):
        calls.append((Path(work_), sheet_))
        return registry.RegistryStaleness(
            stale=False, reason="ok", registry_exists=True, labels_exists=True,
            stored_digest="x", current_digest="x", message="（打桩）",
            registry_mtime=None, labels_mtime=None)

    monkeypatch.setattr(registry, "registry_staleness", fake)
    registry.load_seat_registry(path, detections_fp="FP1")
    assert calls, "load_seat_registry 必须走 registry_staleness 这唯一一处判定"
    assert calls[0][1] == "测试组", f"要把组名传对，实得 {calls}"


def test_preview_stamps_which_labels_it_came_from(tmp_path):
    """核对页自己要能自证新鲜度：**显眼地**印上它由哪一份标注、什么时刻
    产出（摘要前 8 位 + 时间）。用户就是对着这张页面做核对的。"""
    reg, _cand = _scene_with_cell_split(tmp_path)
    reg["_meta"]["labels_digest"] = "0123456789abcdef"
    reg["_meta"]["labels_file"] = "entry_labels_测试组.json"
    reg["_meta"]["built_at"] = "2026-08-23T14:03:00+00:00"
    reg["_meta"]["labels_mtime"] = "2026-08-23T22:03:00"

    html = registry.render_vlm_preview(reg, tmp_path / "vlm.html").read_text(
        encoding="utf-8")
    assert "01234567" in html, f"摘要前 8 位要印在页面上：{html[:1200]!r}"
    assert "entry_labels_测试组.json" in html, "由哪一份标注产出要写明"
    # 时间按**本地时区**印：登记表里 built_at 存的是 UTC，而用户是拿这个时间
    # 去和 Finder 里标注文件的修改时间（本地）比对的——那次事故就是
    # 「14:03 导出、页面停在 13:53」，中间再隔一次时区换算必然看不出来。
    from datetime import datetime
    local = datetime.fromisoformat("2026-08-23T14:03:00+00:00").astimezone()
    assert local.strftime("%Y-%m-%d %H:%M:%S") in html, (
        f"生成时刻要按本地时区印出来：{html[:1400]!r}")
    assert "2026-08-23T22:03:00" in html, (
        "还要印上那份标注当时的修改时间——用户一眼就能对上「我 14:03 导出的"
        "那份进去了没有」")
    # 断言的是**位置关系**而不是字节数：钢印必须排在第一张卡片之前。
    # 原来写的是「前 2500 字符内」——那只是这条性质的一个代理，页首每加一段
    # 合法标记（Task 56U 的左侧目录、多一段 CSS）就会假报一次，而它本要防的
    # 「埋进某张卡片里」根本没发生。
    i_stamp = html.index("01234567")
    i_card = html.index('<div class="card"')
    assert i_stamp < i_card, (
        f"钢印要排在第一张卡片之前，不能埋在卡片里（钢印 @{i_stamp}、"
        f"首卡 @{i_card}）")
    assert 'class="stamp"' in html, "钢印要用醒目的那套样式，不是普通正文"

    reg["_meta"].pop("labels_digest")
    old = registry.render_vlm_preview(reg, tmp_path / "vlm_old.html").read_text(
        encoding="utf-8")
    assert "无法确认" in old or "未记录" in old, (
        "旧版登记表渲染出来的核对页要明说「无从确认是不是最新」")


def test_registry_cli_check_is_read_only(tmp_path, capsys):
    """`python -m pipeline.registry --check`：只判定新鲜度、给退出码、
    **不写盘**，方便用户在打标前先确认一句。"""
    work = tmp_path / "组"
    work.mkdir()
    lp = _labels_file(work, "测试组", {"5": {"a": {"seat": 1}}})
    path = registry.registry_path(work, "测试组")
    registry.write_seat_registry(_built(lp), path)
    before = path.read_bytes()

    rc = registry.main(["--check", "--work", str(work), "--sheet", "测试组"])
    assert rc == 0, f"新鲜时退出码应为 0，实得 {rc}；{capsys.readouterr().out}"

    lp.write_text(json.dumps({"5": {"a": {"seat": 3}}}), encoding="utf-8")
    rc = registry.main(["--check", "--work", str(work), "--sheet", "测试组"])
    out = capsys.readouterr().out
    assert rc == 1, f"陈旧时退出码必须非 0（用户会 && 串命令），实得 {rc}"
    assert registry.labels_digest(lp)[:8] in out, f"要打印出两个摘要：{out!r}"
    assert path.read_bytes() == before, "--check 是只读的，不许写盘"
    assert not registry.preview_path(work, "测试组").exists(), (
        "--check 更不该顺手产出核对页")
# ===========================================================================
# Task 56A · 同一座位占多格 → 登记表要留下**每一格**的那张帧
#
# 用户 2026-08-24 实测（文化4组 order 2）原话：「标了格位1是创业者2，格位2
# 是同格位1，那么应该将这两帧图片都保留下来，而不是登记表只有一张图」。
#
# 根因在 `entryexit.entryexit_fill_plan`（不只在本模块）：两格解引用成同一
# 个人后，同座位号合并只留下一份计划项，另一格连同它那张帧一起消失，没有
# 任何一处报错。登记表是本项目的核心资产（人工复核一次、反复换 VLM 打标），
# 少一半证据看不出来，却会让下一轮打标只看到一张图。
#
# 字段形状因此变了：`cell_index`（标量）→ `cell_indices`（升序列表，非格位
# 切分的记录是空列表）。**旧文件不能被当作能读**，见
# `test_old_registry_file_is_rejected_by_schema_version`。
# ===========================================================================

def _multi_cell_scene(order=2, tid=41):
    """一段 3 人：座位 1 由就座阶段判定；一个退场候选的 1、2 两格被标成
    同一个人（创业者 2）——正是用户实测的那种标法。"""
    from pipeline.entryexit import entryexit_candidates

    seg = mkseg(order=order, presenter_num=3)
    dets = (track(1, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=order)
            + track(tid, RIGHT_CX + 100, STANDING_H, 4, kind="exit", order=order))
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == tid)
    assert len(cand["best_dets"]) == 4, (
        f"夹具前提：候选要恰有 4 张证据帧，实得 {len(cand['best_dets'])}")
    labels = {str(order): {cand["stable_id"]: {
        "role": "创业者", "seat": None, "n_people": 2,
        "cells": [{"cell": 1, "role": "创业者", "seat": 2},
                  {"cell": 2, "same_as": 1}]}}}
    return seg, dets, cand, labels


def test_registry_keeps_every_cell_of_one_person():
    """格位 1 与「同格位1」的格位 2 是同一个人 → 两格的帧都要进登记表。"""
    from pipeline.entryexit import cell_dets

    seg, dets, cand, labels = _multi_cell_scene()
    reg = registry.build_seat_registry([seg], dets, {2: {1: 1}}, labels,
                                       detections_fp="FP1", sheet="文化4测试")
    rec = reg["segments"]["2"]["seats"]["2"]
    assert rec["source"] == "cell_split", rec
    assert rec["cell_indices"] == [1, 2], (
        f"要记下这个座位是由哪几格解引用而来的（下次打标才知道用哪几张），"
        f"实得 {rec.get('cell_indices')}")
    got = [(fr["frame_path"], fr["t"]) for fr in rec["frames"]]
    want = [(str(cell_dets(cand, k)[0].frame_path), cell_dets(cand, k)[0].t)
            for k in (1, 2)]
    assert got == want, (
        f"两格各一张、按格位号升序，且与 entryexit.cell_dets 给出的逐张相同"
        f"（两处各切一次必然日后走样）。实得 {got}，应为 {want}")


def test_registry_multi_cell_counts_seats_not_cells():
    """`counts_by_source` 数的是**座位**数，不是格位数——多格合并后这个
    统计不许虚高（它是「多少人是算法排的、多少是人工点的」的唯一口径）。"""
    seg, dets, _cand, labels = _multi_cell_scene()
    reg = registry.build_seat_registry([seg], dets, {2: {1: 1}}, labels,
                                       detections_fp="FP1", sheet="文化4测试")
    assert reg["_meta"]["counts_by_source"] == {
        "seated": 1, "entryexit": 0, "cell_split": 1}, (
        f"两格是同一个人、只占一个座位，cell_split 必须是 1 而不是 2，实得 "
        f"{reg['_meta']['counts_by_source']}")
    assert reg["_meta"]["n_seats_resolved"] == 2, reg["_meta"]


def test_cell_exclusions_apply_to_the_seated_card_not_the_cell_split_record():
    """逐格排除**只作用于就座那张 2×2**，同座位号的格位切分记录不受影响。

    这条曾经写反过（Task 55 起，格位切分那一路也拿同一个 `(段, 座位)` 键查了
    一遍），用户 2026-08-29 实测城市2 order 6 撞上：他在「就座座位1」那张图上
    勾了第 2、4 格，被套到 `entry@5001` 切给座位 1 的那两格上（按位置数剔掉第
    2 个），**两格变一格**，且一处不报错。

    页面上那排「排除格 N」复选框只长在 `filled_seats` 上，`n_cells` 取自
    `sheet_dets`（全部来自 seated 检测）——格位切分来的座位根本没有那排框，
    用户无从表达「排除切分记录的第 N 格」这件事。两张完全不同的图共用一个键，
    剔掉的必然不是他勾的那一格。
    """
    seg, dets, cand, labels = _multi_cell_scene()
    from pipeline.entryexit import cell_dets

    partial = dict(labels, cell_exclusions={"2": {"2": [2]}})
    reg = registry.build_seat_registry([seg], dets, {2: {1: 1}}, partial,
                                       detections_fp="FP1", sheet="文化4测试")
    rec = reg["segments"]["2"]["seats"]["2"]
    assert rec["source"] == "cell_split", f"夹具前提，实得 {rec['source']}"
    assert [fr["t"] for fr in rec["frames"]] == [cell_dets(cand, 1)[0].t,
                                                 cell_dets(cand, 2)[0].t], (
        f"两格都该留着——那排复选框说的是就座那张图，与这里无关。"
        f"实得 {[fr['t'] for fr in rec['frames']]}")
    assert rec["cell_indices"] == [1, 2], (
        f"格位号跟着实际留下的帧走，实得 {rec['cell_indices']}")


def test_seated_seat_with_every_cell_excluded_falls_into_unresolved():
    """就座那张图的格全被勾掉 = 这个座位没有可用证据，落进 unresolved。

    绝不许留一条 `frames` 为空的记录假装它定下来了——核对页上是一张没有图的
    卡片、交付包里少一张 PNG，而「还缺几人」的数字一点没变（凡有缺口必须可见）。
    """
    seg = mkseg(order=3, presenter_num=2)
    dets = track(1, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=3)
    labels = {"cell_exclusions": {"3": {"1": [1, 2, 3, 4]}}}
    reg = registry.build_seat_registry([seg], dets, {3: {1: 1}}, labels,
                                       detections_fp="FP1", sheet="文化4测试")
    seg_out = reg["segments"]["3"]
    assert "1" not in seg_out["seats"], (
        f"四格全被勾掉，不该留记录，实得 {seg_out['seats'].get('1')}")
    assert 1 in seg_out["unresolved_seats"], (
        f"它该如实列进 unresolved_seats，实得 {seg_out['unresolved_seats']}")

def test_old_registry_file_is_rejected_by_schema_version(tmp_path, capsys):
    """字段形状变了（`cell_index` → `cell_indices`），旧登记表**不许被静默
    读回**：`cell_indices` 缺失会被当成「这个座位不是格位切分来的」，而
    `cell_index` 那个标量没有任何一处会再去读它——错得不报错。

    `REGISTRY_SCHEMA` 常量存在的全部意义就是挡住这一类：形状一变就升版，
    读到旧版本一律整份作废并说清怎么重来。
    """
    assert registry.REGISTRY_SCHEMA != "seat_registry_v1", (
        "字段形状变了就必须升版本号，否则旧文件与新文件长得一样、读回来"
        "却是另一个意思")

    seg, dets, seats_by_order = _scene()
    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, {}, detections_fp="FP1", sheet="测试组")
    path = registry.registry_path(tmp_path, "测试组")
    registry.write_seat_registry(reg, path)

    body = json.loads(path.read_text(encoding="utf-8"))
    body["_meta"]["schema"] = "seat_registry_v1"          # 伪造成旧版
    for rec in body["segments"]["5"]["seats"].values():
        rec.pop("cell_indices", None)
        rec["cell_index"] = None
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")

    got = registry.load_seat_registry(path, detections_fp="FP1")
    out = capsys.readouterr().out
    assert got == {}, f"旧版本必须整份作废，实得 {got}"
    assert "seat_registry_v1" in out and registry.REGISTRY_SCHEMA in out, (
        f"要说清读到的是哪一版、期望的是哪一版，实得 {out!r}")
    assert "python -m pipeline.registry --preview" in out, (
        f"要说清怎么重来（重跑那条命令），实得 {out!r}")

    # 对照：本版本的文件照常读回，不许连自己都拒了
    registry.write_seat_registry(reg, path)
    assert registry.load_seat_registry(path, detections_fp="FP1") == reg


def _scene_with_multi_cell(tmp_path):
    """与 `_scene_with_cell_split` 同形，但 1、2 两格是同一个人（座位 2）
    ——核对页要把两张都拼进证据图。"""
    from pipeline.entryexit import entryexit_candidates

    frames = _real_frames(tmp_path)
    seg = mkseg(order=2, presenter_num=3)
    dets = []
    for i in range(4):
        dets.append(Detection(2, "seated", float(i), frames[i], 1,
                              RIGHT_CX + 380, 1000 - STANDING_H,
                              RIGHT_CX + 420, 1000, 0.9))
        dets.append(Detection(2, "exit", float(i), frames[i], 41,
                              RIGHT_CX + 80, 1000 - STANDING_H,
                              RIGHT_CX + 120, 1000, 0.9))
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 41)
    assert len(cand["best_dets"]) == 4, "夹具前提：该候选要有 4 张证据帧"
    labels = {"2": {cand["stable_id"]: {
        "role": "创业者", "seat": None, "n_people": 2,
        "cells": [{"cell": 1, "role": "创业者", "seat": 2},
                  {"cell": 2, "same_as": 1}]}}}
    reg = registry.build_seat_registry([seg], dets, {2: {1: 1}}, labels,
                                       detections_fp="FP1", sheet="文化4测试")
    assert set(reg["segments"]["2"]["seats"]) == {"1", "2"}, "夹具前提"
    return reg, cand


def test_preview_shows_all_cells_of_a_multi_cell_seat(tmp_path, monkeypatch):
    """核对页要把这个座位占的**每一格**都拼进图，并写明是「格位 1+2」。

    只显示一格等于让用户拿一半的证据去核对，而登记表里明明有两张——他看不
    出少了什么，也就不会发现另一格框错了人。
    """
    reg, _cand = _scene_with_multi_cell(tmp_path)
    calls = []
    monkeypatch.setattr(registry, "build_evidence_sheet",
                        lambda paths, boxes, **kw: calls.append(list(paths)) or b"P")
    out = registry.render_vlm_preview(reg, tmp_path / "vlm.html")
    html = out.read_text(encoding="utf-8")

    assert [len(c) for c in calls] == [4, 2], (
        f"座位 1（就座）四帧拼 2×2；座位 2 占 1、2 两格 → 两张，实得 "
        f"{[len(c) for c in calls]}")
    assert "格位 1+2" in html, (
        f"卡片上要写明这个座位是由哪几格来的，实得 {html[html.find('data-seat=\"2\"'):][:400]!r}")
    assert 'data-cell="1+2"' in html, "机器可读的那份属性同样要能表达多格"
# ===========================================================================
# Task 56C · 二：`unresolved_seats` 要分清「人工看过、确实没画面」与
# 「还没人看过」
#
# 用户 2026-08-24：「如果某个座位的人没有合适的候选图，能标记下来，再抽帧
# 二次有更多的备选吗」。两者混在一个数字里，既没法告诉他还剩多少要看，
# 也没法据此挑出该重抽帧的那几段。
# ===========================================================================

def test_unresolved_reason_marks_no_candidate():
    """人工在挑选页标了「本座位无可用画面」的座位，理由记 `no_candidate`。"""
    seg, dets, seats_by_order = _scene()          # 需 3 人，就座只定了座位 1
    labels = {"no_candidate_seats": {"5": [2]}}

    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, labels, detections_fp="FP1", sheet="测试组")
    rec = reg["segments"]["5"]
    assert rec["unresolved_seats"] == [2, 3], rec
    assert rec["unresolved_reason"] == {"2": "no_candidate", "3": "not_reviewed"}, (
        f"座位 2 是人工判定确实没画面，座位 3 只是还没人看过——两者必须分开，"
        f"实得 {rec.get('unresolved_reason')!r}")


def test_unresolved_reason_defaults_to_not_reviewed():
    """没有任何人工标记时，所有未定座位一律记 `not_reviewed`。

    默认绝不能是 `no_candidate`：那等于替用户断言「这里确实没画面」，
    而他一眼都还没看过——第三部分会据此去重抽帧，白跑一遍。
    """
    seg, dets, seats_by_order = _scene()
    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, {}, detections_fp="FP1", sheet="测试组")
    assert reg["segments"]["5"]["unresolved_reason"] == {
        "2": "not_reviewed", "3": "not_reviewed"}, reg["segments"]["5"]


def test_unresolved_reason_only_covers_unresolved_seats():
    """已定下来的座位不进 `unresolved_reason`，哪怕文件里残留着标记。

    用户先标「没画面」、后来又找到人补上，那条陈旧标记必须自动作废，
    否则登记表会同时说「座位 2 已定」和「座位 2 没画面」。
    """
    seg, dets, seats_by_order = _scene()
    labels = {"no_candidate_seats": {"5": [1, 2]}}   # 座位 1 其实已由就座定下
    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, labels, detections_fp="FP1", sheet="测试组")
    rec = reg["segments"]["5"]
    assert "1" not in rec["unresolved_reason"], rec
    assert set(rec["unresolved_reason"]) == {"2", "3"}, rec


def test_unresolved_reason_counts_split_the_two_kinds():
    """`unresolved_reason_counts` 把两类分开数——CLI 与 `_meta` 都用它，
    两处各数一份必然日后走样。"""
    seg, dets, seats_by_order = _scene()
    labels = {"no_candidate_seats": {"5": [2]}}
    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, labels, detections_fp="FP1", sheet="测试组")
    assert registry.unresolved_reason_counts(reg) == {
        "no_candidate": 1, "not_reviewed": 1}, reg["segments"]["5"]


def test_registry_cli_reports_the_two_kinds_separately(tmp_path, capsys):
    """CLI 打印时把两者分开报，别混在一个数字里。"""
    from pipeline.detect import save_detections
    from pipeline.schedule import save_segments

    seg, dets, seats_by_order = _scene()
    work = tmp_path / "组"
    work.mkdir()
    save_segments([seg], work / "segments.json")
    save_detections(dets, work / "detections.json")
    (work / "seats.json").write_text(json.dumps({"5": {"1": 1}}), encoding="utf-8")

    from pipeline.entryexit import detections_fingerprint
    fp = detections_fingerprint(work / "detections.json")
    (work / "entry_labels_测试组.json").write_text(json.dumps({
        "_meta": {"detections_fingerprint": fp, "sheet": "测试组",
                  "n_candidates": 0},
        "no_candidate_seats": {"5": [2]},
    }, ensure_ascii=False), encoding="utf-8")

    rc = registry.main(["--work", str(work), "--sheet", "测试组", "--dry-run"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "无可用画面" in printed, (
        f"CLI 要单独报「人工判定确实没有画面」这一类，实得 {printed!r}")
    assert "还没人看过" in printed or "未复核" in printed, (
        f"CLI 要单独报「还没人看过」这一类，实得 {printed!r}")
    # 逐段那一行也必须把两类分开写，不能只报一个「座位 [2, 3] 仍未定」的合计
    # ——那正是本改动之前的样子，用户从中看不出哪几个是他已经看过的。
    line = [ln for ln in printed.splitlines() if "第 5 段" in ln]
    assert line, f"应有第 5 段的明细行，实得 {printed!r}"
    assert "座位 [3] 还没人看过" in line[0], (
        f"逐段行要写清哪几个还没人看过，实得 {line[0]!r}")
    assert "座位 [2] 人工判定无可用画面" in line[0], (
        f"逐段行要写清哪几个是人工判定没画面，实得 {line[0]!r}")


# --- 就座那一路也要应用逐格排除（用户 2026-08-25 实测城市2 order2 座位3）---

def test_seated_seat_honours_cell_exclusions():
    """就座定下来的座位，勾掉的格位同样不许留档。

    用户实测：城市2组 order 2 座位 3 勾了「排除格1」「排除格3」，登记表里
    却still是四帧全在，他以为「没生效」。

    实际上**打标那一路早就生效了**（`assemble` 的就座分支走
    `apply_cell_exclusions`），只有登记表这一路漏了。两边就此分叉：
    核对页/交付包里的证据图是四格，模型真正看到的是两格——正是本项目最忌讳
    的「人核对了一张模型没看过的图」，而一处也不报错。

    口径与 assemble、与挑选页上那几个复选框逐字相同：三处都用
    `select_best_frames(该座位的就座帧, n=CELL_COUNT)`，因此**位置 i 就是
    格位 i**。
    """
    seg = mkseg(order=5, presenter_num=2)
    # track_id 刻意与座位号错开：`{1: 1, 2: 2}` 那种夹具里两者相等，把排除号
    # 拿去按 track_id 查也照样过（变异测试实测存活）。
    dets = (track(71, RIGHT_CX + 400, STANDING_H, 6, kind="seated", order=5)
            + track(72, RIGHT_CX + 100, STANDING_H, 6, kind="seated", order=5))
    labels = {"cell_exclusions": {"5": {"1": [2, 4]}}}

    reg = registry.build_seat_registry([seg], dets, {5: {71: 1, 72: 2}}, labels,
                                       detections_fp="FP", sheet="S")
    seats = reg["segments"]["5"]["seats"]
    s1 = seats["1"]
    assert s1["source"] == "seated", s1
    assert len(s1["frames"]) == 2, (
        f"勾掉第 2、4 格之后只该留两帧，实得 {len(s1['frames'])} 帧——"
        f"登记表与模型看到的画面就此对不上")
    # 没被勾的那个座位一帧不少
    assert len(seats["2"]["frames"]) == 4, seats["2"]


def test_seated_seat_with_every_cell_excluded_is_unresolved():
    """四格全被勾掉的就座座位落进 unresolved_seats，不留一条空记录。

    留一条 frames 为空的记录等于假装这个人定下来了：核对页上是一张没有图的
    卡片，交付包里少一张 PNG，而「还缺几人」的数字一点没变。
    """
    seg = mkseg(order=5, presenter_num=2)
    dets = (track(71, RIGHT_CX + 400, STANDING_H, 6, kind="seated", order=5)
            + track(72, RIGHT_CX + 100, STANDING_H, 6, kind="seated", order=5))
    labels = {"cell_exclusions": {"5": {"1": [1, 2, 3, 4]}}}

    reg = registry.build_seat_registry([seg], dets, {5: {71: 1, 72: 2}}, labels,
                                       detections_fp="FP", sheet="S")
    seg_rec = reg["segments"]["5"]
    assert "1" not in seg_rec["seats"], (
        f"四格全勾掉就定不下来，不许留空记录，实得 {seg_rec['seats'].get('1')}")
    assert 1 in seg_rec["unresolved_seats"], seg_rec["unresolved_seats"]


def test_seated_seat_tops_up_from_the_labelled_transition_candidate():
    """就座图勾掉格位后，用该座位人工标定的进出场候选补回四张。

    用户 2026-08-25：「如果在自动标的图片中选择了排除某个窗格，增加通过进出场
    的照片对该窗格图片进行补充，以最大化填充 2×2」。勾掉一格之后只剩三张，
    模型能看的证据凭空少了四分之一，而那一格本来只是框错了人、不是这个人没有
    别的画面。
    """
    seg = mkseg(order=8, presenter_num=2)
    dets = (track(90, RIGHT_CX + 300, STANDING_H, 6, kind="seated", order=8)
            + track(91, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=8))
    from pipeline.entryexit import entryexit_candidates
    cand = next(c for c in entryexit_candidates(dets, seg)
                if c["track_id"] == 91)
    labels = {"8": {cand["stable_id"]: {"role": "创业者", "seat": 1}},
              "cell_exclusions": {"8": {"1": [2, 4]}}}

    reg = registry.build_seat_registry([seg], dets, {8: {90: 1}}, labels,
                                       detections_fp="FP", sheet="S")
    s1 = reg["segments"]["8"]["seats"]["1"]
    kinds = [f["kind"] for f in s1["frames"]]
    assert len(s1["frames"]) == 4, (
        f"勾掉两格后应由进出场补回四张，实得 {len(s1['frames'])} 张 {kinds}")
    assert kinds[:2] == ["seated", "seated"], f"就座那两张排前面：{kinds}"
    assert kinds[2:] == ["exit", "exit"], f"补上的是进出场帧：{kinds}"


def test_seated_seat_is_not_topped_up_without_a_labelled_candidate():
    """没有人工标定的进出场候选就**不补**——宁可少两张。

    随便抓进出场帧来填，填进去的很可能是场地工作人员（本项目正因为这个在
    Task 26 一度整体禁掉进出场帧）。重新放开的唯一依据是「这个候选是用户亲手
    指名给这个座位的」。
    """
    seg = mkseg(order=8, presenter_num=2)
    dets = (track(90, RIGHT_CX + 300, STANDING_H, 6, kind="seated", order=8)
            + track(91, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=8))
    labels = {"cell_exclusions": {"8": {"1": [2, 4]}}}      # 没有任何候选标注
    reg = registry.build_seat_registry([seg], dets, {8: {90: 1}}, labels,
                                       detections_fp="FP", sheet="S")
    s1 = reg["segments"]["8"]["seats"]["1"]
    assert len(s1["frames"]) == 2, (
        f"没人指名就不补，实得 {len(s1['frames'])} 张")


def test_seated_sheet_merges_the_same_cell_labelled_images():
    """标了「同格位X」的那几格，与自动选的就座图**合并**进同一张 2×2。

    用户 2026-08-25：「凡是标注的图是不是都可以利用，如果标注写了同格位X 的，
    和自动标注的图合并到一起，用于 VLM，最大化利用标注的图」。

    「同格位X」＝同一个人占多格，解引用之后两格同属一个座位。这两格都是他本人
    的证据，都该并进来；而同一张多人框里**别人的格**一格都不许进——补进去就是
    把另一个人的衣着编进这位创业者的行里，且一处不报错。
    """
    from pipeline.entryexit import entryexit_candidates

    seg = mkseg(order=9, presenter_num=2)
    dets = (track(80, RIGHT_CX + 300, STANDING_H, 6, kind="seated", order=9)
            + track(81, RIGHT_CX + 200, STANDING_H, 4, kind="exit", order=9))
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 81)
    labels = {
        "9": {cand["stable_id"]: {
            "role": "", "seat": None, "n_people": 2,
            "cells": [{"cell": 1, "role": "创业者", "seat": 1},
                      {"cell": 3, "same_as": 1},          # 同一个人的第二格
                      {"cell": 2, "role": "创业者", "seat": 2}]}},
        # 就座那张 2×2 勾掉两格 → 只剩两张，留出两格给标注的图
        "cell_exclusions": {"9": {"1": [3, 4]}},
    }

    reg = registry.build_seat_registry([seg], dets, {9: {80: 1}}, labels,
                                       detections_fp="FP", sheet="S")
    s1 = reg["segments"]["9"]["seats"]["1"]
    kinds = [f["kind"] for f in s1["frames"]]
    assert len(s1["frames"]) == 4, (
        f"两张就座 + 标注的第 1、3 格 = 四张，实得 {len(s1['frames'])} {kinds}")
    assert kinds == ["seated", "seated", "exit", "exit"], kinds

    # 别人的那一格（第 2 格 → 座位 2）绝不能出现在座位 1 的证据里。
    # 按**帧时间**判别：本夹具里同一条 track 的四帧框完全相同，只有 t 不同，
    # 按框比对什么也测不出来（上一版就是这么写的，测了个寂寞）。
    from pipeline.entryexit import cell_dets
    other_t = cell_dets(cand, 2)[0].t
    mine_t = {d.t for d in cell_dets(cand, 1) + cell_dets(cand, 3)}
    got_t = {f["t"] for f in s1["frames"] if f["kind"] == "exit"}
    assert got_t == mine_t, (
        f"补进来的必须正好是第 1、3 格那两帧（t={sorted(mine_t)}），"
        f"实得 t={sorted(got_t)}")
    assert other_t not in got_t, (
        f"第 2 格（t={other_t}）是座位 2 的人，不许混进座位 1")


# --- Task 56T：一个座位在多条候选上都被标过时，登记表要把它们并起来 --------

def _seat_with_two_labelled_cands(order=21, seat=2):
    """一段里同一个座位被标了两次：exit 上切了一格，entry 上整条标成他。"""
    from pipeline.entryexit import entryexit_candidates

    seg = mkseg(order=order, presenter_num=3)
    dets = (track(1, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=order)
            + track(12, RIGHT_CX + 100, STANDING_H, 4, kind="exit", order=order)
            + track(13, RIGHT_CX + 250, STANDING_H, 4, kind="entry", order=order))
    cands = entryexit_candidates(dets, seg)
    ex = next(c for c in cands if c["track_id"] == 12 and c["kind"] == "exit")
    en = next(c for c in cands if c["track_id"] == 13 and c["kind"] == "entry")
    assert len(ex["best_dets"]) == len(en["best_dets"]) == 4, "夹具前提：两条各 4 张"
    labels = {str(order): {
        ex["stable_id"]: {"role": "工作人员", "seat": None, "n_people": 2,
                          "note": "1 是创业者2，2 是创业者3",
                          "cells": [{"cell": 1, "role": "创业者", "seat": seat},
                                    {"cell": 2, "role": "创业者", "seat": 3}]},
        en["stable_id"]: {"role": "创业者", "seat": seat, "n_people": 1},
    }}
    return seg, dets, {order: {1: 1}}, labels, ex, en


def test_cell_split_seat_is_topped_up_from_its_other_labelled_candidates():
    """按格位切分定下的座位，也要用该座位**其余**人工标注把 2×2 补满。

    用户 2026-08-26 实测文化4组 order 3：「单帧可能只选了一个窗格，而在多帧
    分别选的窗格没有拼在一起，应该根据标注尽可能全的拼成 2×2 四张图」。

    根因是补帧只长在**就座**那一路上：由人工标注的进出场候选定下来的座位
    （`cell_split` / `entryexit`）压根不走 `topup_frames`，拿到一格就收工。
    该段三个座位全是这一类，于是全都只有一两张——而用户明明标了更多。
    """
    seg, dets, seats_by_order, labels, ex, en = _seat_with_two_labelled_cands()
    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, labels, detections_fp="FP1", sheet="测试组")
    rec = reg["segments"]["21"]["seats"]["2"]
    assert len(rec["frames"]) == 4, (
        f"exit 切出的那一格 + entry 那条整标的四张，够拼满 2×2，实得 "
        f"{len(rec['frames'])} 张")
    paths = {(f["kind"], f["t"]) for f in rec["frames"]}
    assert ("exit", ex["best_dets"][0].t) in paths, (
        "先落位的那一格必须留着——它是用户对这个座位最具体的一次判断")
    assert any(k == "entry" for k, _t in paths), (
        f"另一条标注一张都没并进来，实得 {sorted(paths)}")


def test_topped_up_cell_indices_stay_in_lockstep_with_frames():
    """补进来的帧要连同**它在自己那张证据图上是第几格**一起留档。

    `cell_indices` 与 `frames` 逐项一一对应是 Task 56A 钉下的不变量：补帧只
    加 `frames` 不加 `cell_indices`，「第几格」就会错位到另一张帧上，下一轮换
    VLM 打标时按错的格位去截图，且一处不报错。
    """
    seg, dets, seats_by_order, labels, _ex, _en = _seat_with_two_labelled_cands()
    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, labels, detections_fp="FP1", sheet="测试组")
    for s, rec in reg["segments"]["21"]["seats"].items():
        if rec["source"] != "cell_split":
            continue
        assert len(rec["cell_indices"]) == len(rec["frames"]), (
            f"座位 {s}：格位 {rec['cell_indices']} 与 {len(rec['frames'])} 张帧"
            f"必须逐项对应")


def test_topup_does_not_repeat_the_candidate_used_for_placement():
    """补帧不许把落位用的那条候选**再补一遍**（同一张图占两格）。

    原先这条测的是「被逐格排除剔掉的那一格不许被补回来」。Task 56W 之后格位
    切分这一路根本不查逐格排除（那排复选框说的是就座那张图），前提不再成立；
    但「不重复」这条规矩仍然要钉——重复的图白占一格，还会让模型以为「这个角度
    我看了两遍」。
    """
    seg, dets, seats_by_order, labels, ex, en = _seat_with_two_labelled_cands()
    reg = registry.build_seat_registry(
        [seg], dets, seats_by_order, labels, detections_fp="FP1", sheet="测试组")
    rec = reg["segments"]["21"]["seats"]["2"]
    keys = [(f["kind"], f["t"]) for f in rec["frames"]]
    assert len(keys) == len(set(keys)), f"有重复的帧：{keys}"
    assert ("exit", ex["best_dets"][0].t) in keys, (
        "落位用的那一格必须在——它是用户对这个座位最具体的一次判断")
    assert any(k == "entry" for k, _t in keys), (
        f"另一条标注也要并进来（Task 56T），实得 {keys}")



# --- Task 56U：核对页左侧目录（用户 2026-08-29「VLM html 左侧也做个目录」）---

def _two_segment_registry(tmp_path):
    """两段的登记表：第 5 段三个座位全定，第 6 段两人只定了一个。"""
    reg, _cand = _scene_with_cell_split(tmp_path)
    frames = _real_frames(tmp_path)
    seg6 = mkseg(order=6, presenter_num=2, company="第六家企业")
    dets6 = [Detection(6, "seated", float(i), frames[i], 1,
                       RIGHT_CX + 380, 1000 - STANDING_H,
                       RIGHT_CX + 420, 1000, 0.9) for i in range(4)]
    reg6 = registry.build_seat_registry([seg6], dets6, {6: {1: 1}}, {},
                                        detections_fp="FP1", sheet="测试组")
    reg["segments"]["6"] = reg6["segments"]["6"]
    assert reg["segments"]["6"]["unresolved_seats"] == [2], (
        f"夹具前提：第 6 段要恰好缺座位 2，实得 "
        f"{reg['segments']['6']['unresolved_seats']}")
    return reg


def test_preview_has_a_left_toc_listing_every_segment(tmp_path, monkeypatch):
    """核对页左侧要有目录，每段一行、点了能跳到那一段。

    用户 2026-08-29：「VLM html 左侧也做个目录」。这页要一个座位一个座位地
    核对，几十段滚下来找不到位置——挑选页早就有这个目录（Task 47），核对页
    没有，两页来回对照时只有一边能定位。
    """
    monkeypatch.setattr(registry, "build_evidence_sheet", lambda *a, **k: b"P")
    reg = _two_segment_registry(tmp_path)
    html = registry.render_vlm_preview(reg, tmp_path / "vlm.html").read_text(
        encoding="utf-8")

    assert 'class="toc"' in html, "左侧目录整个不在"
    for order in ("5", "6"):
        assert f'id="seg-{order}"' in html, f"第 {order} 段缺锚点，目录点了跳不过去"
        assert f'href="#seg-{order}"' in html, f"目录里没有指向第 {order} 段的那一行"
    assert "第六家企业" in html, "目录/正文要写企业名，只有段号认不出是哪一家"


def test_preview_toc_marks_segments_that_still_miss_seats(tmp_path, monkeypatch):
    """目录要标出**哪几段还缺人**，且不能只靠颜色。

    这是目录的主要用途：一眼看出还有哪几段没定完。只用背景色标的话，色觉
    障碍者用不了，截图/打印成灰度后也全部失效——挑选页那份目录逐字写着这
    条理由（⛔/⚠/✓ 加文字），核对页必须同样办。
    """
    monkeypatch.setattr(registry, "build_evidence_sheet", lambda *a, **k: b"P")
    reg = _two_segment_registry(tmp_path)
    html = registry.render_vlm_preview(reg, tmp_path / "vlm.html").read_text(
        encoding="utf-8")

    import re
    rows = re.findall(r'<a class="toc-row[^"]*" href="#seg-(\d+)"[^>]*>(.*?)</a>',
                      html, re.S)
    by_order = {o: body for o, body in rows}
    assert set(by_order) == {"5", "6"}, f"两段各一行，实得 {sorted(by_order)}"
    assert "缺 1 人" in by_order["6"], (
        f"第 6 段缺座位 2，目录必须写出来（不能只标颜色），实得 {by_order['6']}")
    assert "缺" not in by_order["5"], (
        f"第 5 段三个座位全定，不该报缺人，实得 {by_order['5']}")
    assert "3/3" in by_order["5"] and "1/2" in by_order["6"], (
        "每行要写「已定/应有」，这是判断还剩多少活的唯一数字")


# --- Task 56V：已定座位改号 ------------------------------------------------

def _two_seated_scene(order=31):
    """一段 3 人，就座阶段定下座位 1、2（track 7、8），座位 3 空着。

    `seats_by_order` 里**故意把座位 2 排在座位 1 前面**：登记表按这个字典的
    顺序逐条处理，被改号顶掉的那位若恰好排在后面，「它已经被占了」这条兜底
    会替真正的让位逻辑背锅——变异掉让位那一句也不会有测试变红（实测存活过
    一次）。排在前面才测得到。
    """
    seg = mkseg(order=order, presenter_num=3)
    dets = (track(7, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=order)
            + track(8, RIGHT_CX + 600, STANDING_H, 5, kind="seated", order=order))
    return seg, dets, {order: {8: 2, 7: 1}}


def test_seated_seat_can_be_reassigned_to_another_seat():
    """就座那一路的图可以就地改号：座位 1 的那张图改挂到座位 3。

    用户 2026-08-29 实测城市2 order 6：「自动标的 seat1 实际为 seat2……否则
    标错了，无法将该图用于其他座位」。
    """
    seg, dets, sbo = _two_seated_scene()
    labels = {"seat_reassign": {"31": {"1": 3}}}
    reg = registry.build_seat_registry([seg], dets, sbo, labels,
                                       detections_fp="FP1", sheet="测试组")
    seats = reg["segments"]["31"]["seats"]
    assert "3" in seats, f"改号后座位 3 要有人，实得 {sorted(seats)}"
    assert seats["3"]["track_id"] == 7, (
        f"座位 3 该拿到原座位 1 那条 track 的画面，实得 {seats['3']['track_id']}")
    assert "1" not in seats, (
        f"原座位 1 改走之后就空了，不能两边都留一份，实得 {sorted(seats)}")
    assert 1 in reg["segments"]["31"]["unresolved_seats"], (
        "改走之后座位 1 是缺口，必须可见（凡有缺口必须可见）")


def test_reassign_overwrites_the_target_and_frees_its_previous_occupant():
    """目标座位已有人时**覆盖**，被顶掉的那个变回未定。

    用户 2026-08-29 选定的规则。相邻两人排反是最常见的错法，但不保证一定是
    对调——所以只搬用户明说的那一个，被顶掉的落回「待标」由他另行指认，
    而不是替他猜一个对调。
    """
    seg, dets, sbo = _two_seated_scene()
    labels = {"seat_reassign": {"31": {"1": 2}}}      # 座位 2 本来是 track 8
    reg = registry.build_seat_registry([seg], dets, sbo, labels,
                                       detections_fp="FP1", sheet="测试组")
    seg_out = reg["segments"]["31"]
    seats = seg_out["seats"]
    assert seats["2"]["track_id"] == 7, (
        f"座位 2 该被改号来的 track 7 顶掉，实得 {seats['2']['track_id']}")
    assert "1" not in seats, f"原座位 1 空了，实得 {sorted(seats)}"
    assert set(seg_out["unresolved_seats"]) == {1, 3}, (
        f"座位 1（被改走）与座位 3（本来就没人）都是缺口；被顶掉的 track 8 "
        f"不再占任何座位。实得 {seg_out['unresolved_seats']}")
    assert all(r["track_id"] != 8 for r in seats.values()), (
        "被顶掉的那条 track 绝不能自己找个别的座位坐下——那是替用户猜")


def test_reassign_does_not_touch_seats_the_user_named_on_a_candidate():
    """人工在进出场候选上指名的座位**更具体**，改号不许覆盖它。

    两者都是人工输入，但候选上那一条是「我看着这张图说这是谁」，改号只是
    「这张就座图挂错了号」。让后者顶掉前者会把更确切的判断静默丢掉。
    """
    from pipeline.entryexit import entryexit_candidates

    seg = mkseg(order=32, presenter_num=3)
    dets = (track(7, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=32)
            + track(41, RIGHT_CX + 100, STANDING_H, 4, kind="exit", order=32))
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 41)
    labels = {"32": {cand["stable_id"]: {"role": "创业者", "seat": 3,
                                         "n_people": 1}},
              "seat_reassign": {"32": {"1": 3}}}
    reg = registry.build_seat_registry([seg], dets, {32: {7: 1}}, labels,
                                       detections_fp="FP1", sheet="测试组")
    seats = reg["segments"]["32"]["seats"]
    assert seats["3"]["track_id"] == 41 and seats["3"]["kind"] == "exit", (
        f"座位 3 该留给候选上指名的那条，实得 {seats['3']}")
    assert "1" not in seats, (
        "改号没落地也不能把原座位留着假装无事发生——那样用户看不出他的改号"
        f"失效了。实得 {sorted(seats)}")


def test_cell_exclusions_follow_the_card_not_the_new_seat_number():
    """逐格排除跟着**那张图**走，不跟新座位号走。

    用户是在「座位 1」那张 2×2 上勾的「排除格 2」。这张图改挂到座位 3 之后，
    勾掉的仍是同一张图的第 2 格——按新号去查会查到另一个座位的排除表上，
    表现是勾了的那一格又回来了、没勾的反而没了，且一处不报错。
    """
    seg, dets, sbo = _two_seated_scene(order=33)
    labels = {"seat_reassign": {"33": {"1": 3}},
              "cell_exclusions": {"33": {"1": [2]}}}
    reg = registry.build_seat_registry([seg], dets, sbo, labels,
                                       detections_fp="FP1", sheet="测试组")
    rec = reg["segments"]["33"]["seats"]["3"]
    assert len(rec["frames"]) == 3, (
        f"四帧勾掉一格该剩三帧，实得 {len(rec['frames'])}——按新座位号查排除表"
        f"会得到四帧（那张表上没有座位 3 的条目）")


# --- Task 56W：改号与人工标注合并（用户 2026-08-29 实测城市2 order 6）------
# 「自动标注的座位1改为座位2且剔除了其中的两格，应与后边人工标注的座位2进行
# 合并；自动标注的座位2改为座位3，也应与后面人工标注的座位3合并」。

def _reassign_merge_scene(order=41):
    """城市2 order 6 的真实形状：座位 2 被人工否决腾空，再由一个多人候选按格位
    切给座位 2；座位 1、2 各有一张自动排的就座图。

    **必须带那条否决**：`entryexit_fill_plan` 不填已被就座阶段占住的座位，
    不腾空的话候选根本补不进来，测试测的就成了另一件事（上一版夹具正是如此，
    跑出来 source 是 seated）。
    """
    from pipeline.entryexit import entryexit_candidates

    seg = mkseg(order=order, presenter_num=3)
    dets = (track(7, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=order)
            + track(8, RIGHT_CX + 600, STANDING_H, 5, kind="seated", order=order)
            + track(41, RIGHT_CX + 100, STANDING_H, 4, kind="exit", order=order))
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 41)
    assert len(cand["best_dets"]) == 4, "夹具前提：候选要有 4 张证据帧"
    labels = {
        str(order): {cand["stable_id"]: {
            "role": "", "seat": None, "n_people": 2,
            "cells": [{"cell": 1, "role": "创业者", "seat": 2},
                      {"cell": 2, "same_as": 1}]}},
        "seat_overrides": {str(order): {
            "2": {"rejected": True, "replacement_stable_id": None, "note": ""}}},
    }
    return seg, dets, {order: {7: 1, 8: 2}}, labels, cand


def test_cell_exclusions_do_not_touch_the_cell_split_path():
    """逐格排除**只针对就座证据图**，不许套到格位切分出来的座位上。

    用户 2026-08-29 实测城市2 order 6：`entry@5001` 上标给座位 1 的两格
    （第 2 格 + 「第 3 格＝同第 2 格」）在登记表里只剩一格。

    根因：那排「排除格 N」复选框只长在**就座**证据图上（`entryreview` 的
    `filled_seats`，`n_cells` 取自 `sheet_dets`，全部来自 seated 检测），而
    格位切分这一路拿同一个 `(段, 座位)` 键也去查了一遍——他在「就座座位1」
    那张图上勾的格，被套到了另一张完全不同的图上，剔掉的是别的东西，且一处
    不报错。
    """
    seg, dets, sbo, labels, cand = _reassign_merge_scene(order=42)
    # 用户在「就座座位 2」那张图上勾了两格；座位 2 同时又有格位切分的记录。
    labels["cell_exclusions"] = {"42": {"2": [2]}}
    reg = registry.build_seat_registry([seg], dets, sbo, labels,
                                       detections_fp="FP1", sheet="测试组")
    rec = reg["segments"]["42"]["seats"]["2"]
    assert rec["source"] == "cell_split", f"夹具前提，实得 {rec['source']}"
    assert rec["cell_indices"] == [1, 2] and len(rec["frames"]) == 2, (
        f"候选上标的两格都该留着——那排复选框说的是就座那张图，与这里无关。"
        f"实得 格位 {rec['cell_indices']}、{len(rec['frames'])} 张")


def test_reassigned_seated_card_merges_into_the_labelled_seat():
    """改号到一个**已有人工标注**的座位时：两边的图合并，不是丢掉改号。

    用户 2026-08-29 原话：「自动标注的座位1改为座位2……应与后边人工标注的
    座位2进行合并」。此前的规则是「候选上指名的更具体，改号不顶掉它」，于是
    改号那张图整个丢掉——他在页面上改了号，跑下来什么都没发生。
    """
    seg, dets, sbo, labels, cand = _reassign_merge_scene(order=43)
    labels["seat_reassign"] = {"43": {"1": 2}}
    reg = registry.build_seat_registry([seg], dets, sbo, labels,
                                       detections_fp="FP1", sheet="测试组")
    rec = reg["segments"]["43"]["seats"]["2"]
    assert len(rec["frames"]) == 4, (
        f"候选切出的两格 + 改号进来的就座帧，拼满 2×2。实得 "
        f"{len(rec['frames'])} 张")
    kinds = {f["kind"] for f in rec["frames"]}
    assert kinds == {"exit", "seated"}, (
        f"两边的画面都要在，实得 {kinds}")
    assert len(rec["cell_indices"]) == len(rec["frames"]), (
        "格位与帧必须逐项对应（Task 56A）")
    assert "1" not in reg["segments"]["43"]["seats"], (
        "改走之后原座位 1 空了")


def test_reassignment_supersedes_a_stale_rejection():
    """改号压过早先的「是否替换」勾选。

    用户 2026-08-29 实测城市2 order 6：座位 1、2、3 都带着他上一轮用「否决 +
    去候选里重新指认」时留下的 `rejected`，新做的两次改号因此一次都没生效——
    他在页面上改了号，跑下来毫无变化且不报错。

    改号本身就是「这个自动结果不对，它其实是座位 N」的更具体说法，压过那句
    笼统的「这个座位判错了」。
    """
    seg, dets, sbo, labels, cand = _reassign_merge_scene(order=44)
    labels["seat_reassign"] = {"44": {"1": 2}}
    # 座位 1 也带上上一轮的否决痕迹：两处 rejected 都不该挡住这次改号。
    labels["seat_overrides"]["44"]["1"] = {
        "rejected": True, "replacement_stable_id": None, "note": ""}
    reg = registry.build_seat_registry([seg], dets, sbo, labels,
                                       detections_fp="FP1", sheet="测试组")
    rec = reg["segments"]["44"]["seats"]["2"]
    assert any(f["kind"] == "seated" for f in rec["frames"]), (
        f"改号进来的就座帧必须在——两处 rejected 都是上一轮的痕迹，不该挡住它。"
        f"实得 {[f['kind'] for f in rec['frames']]}")


def test_exclusions_on_a_reassigned_card_still_follow_that_card():
    """改号的那张就座图，它自己的逐格排除照旧生效（按**原**座位号查）。"""
    seg, dets, sbo, labels, cand = _reassign_merge_scene(order=45)
    labels["seat_reassign"] = {"45": {"1": 2}}
    labels["cell_exclusions"] = {"45": {"1": [1, 2]}}   # 就座座位1 勾掉两格
    reg = registry.build_seat_registry([seg], dets, sbo, labels,
                                       detections_fp="FP1", sheet="测试组")
    rec = reg["segments"]["45"]["seats"]["2"]
    seated = [f for f in rec["frames"] if f["kind"] == "seated"]
    assert len(seated) == 2, (
        f"四帧勾掉两格，只该并进来两帧就座画面，实得 {len(seated)}")


def test_merge_does_not_repeat_a_frame_already_in_the_record():
    """合并时同一张帧不留两份。

    用户可能把同一条 track 既标在候选上、又让它的就座卡改号并进同一个座位；
    此时两边指的是同一批帧。重复的图白占一格（本就只有 4 格），还会让模型
    以为「这个角度我看了两遍」。

    夹具让**候选与就座共用同一条 track 的帧**——上一版夹具里两边帧完全不同，
    去重那一句删掉也不会有测试变红（变异测试实测存活）。
    """
    seg = mkseg(order=46, presenter_num=3)
    # 同一批帧同时以 seated 与 exit 两种 kind 存在：exit 那份进候选，seated
    # 那份走就座那一路，两者的 frame_path/t 逐一相同。
    seated = track(7, RIGHT_CX + 100, STANDING_H, 4, kind="seated", order=46)
    exits = [Detection(46, "exit", d.t, d.frame_path, 41,
                       d.x1, d.y1, d.x2, d.y2, d.conf) for d in seated]
    dets = seated + track(8, RIGHT_CX + 600, STANDING_H, 5,
                          kind="seated", order=46) + exits
    from pipeline.entryexit import entryexit_candidates
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 41)
    labels = {
        "46": {cand["stable_id"]: {
            "role": "", "seat": None, "n_people": 2,
            "cells": [{"cell": 1, "role": "创业者", "seat": 2},
                      {"cell": 2, "same_as": 1}]}},
        "seat_overrides": {"46": {
            "2": {"rejected": True, "replacement_stable_id": None, "note": ""}}},
        "seat_reassign": {"46": {"1": 2}},      # 就座卡 1（同一批帧）并进座位 2
    }
    reg = registry.build_seat_registry([seg], dets, {46: {7: 1, 8: 2}}, labels,
                                       detections_fp="FP1", sheet="测试组")
    rec = reg["segments"]["46"]["seats"]["2"]
    keys = [(f["frame_path"], f["t"]) for f in rec["frames"]]
    assert len(keys) == len(set(keys)), (
        f"同一张帧留了两份：{[(k[0].split('/')[-1], k[1]) for k in keys]}")
    assert len(rec["cell_indices"]) == len(rec["frames"]), (
        "格位与帧必须逐项对应")


def test_a_freed_seat_can_be_filled_by_a_labelled_candidate():
    """改号腾出来的原座位，要能被人工在候选上指名的那条补上。

    用户 2026-09-08 实测城市1 order 4：他先勾「这张图其实是座位 2」把就座那张
    挪走，再在退场候选上标「创业者 · 座位 1」。结果座位 1 仍是「还没人看过」，
    而且日志里另有一条「第 5 段 → 第 4 段座位 1（该座位已有人，不覆盖）」——
    两条都指向同一个根因：喂给 `entryexit_fill_plan` 的占用表是**阶段 4 的
    原始基线**，既没扣掉人工腾空（rejected）的，也没跟着改号走。于是座位 1
    被当成「还有人占着」，两条人工标注一条都没落地，一处不报错。

    `assemble` 那边喂的是 `by_final_seat`（已含否决与改号），本来就是对的；
    走样的只有登记表这一路。
    """
    from pipeline.entryexit import entryexit_candidates

    seg = mkseg(order=61, presenter_num=2)
    dets = (track(7, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=61)
            + track(41, RIGHT_CX + 100, STANDING_H, 4, kind="exit", order=61))
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 41)
    labels = {"61": {cand["stable_id"]: {"role": "创业者", "seat": 1,
                                         "n_people": 1}},
              "seat_reassign": {"61": {"1": 2}}}
    reg = registry.build_seat_registry([seg], dets, {61: {7: 1}}, labels,
                                       detections_fp="FP1", sheet="测试组")
    seats = reg["segments"]["61"]["seats"]
    assert seats["2"]["track_id"] == 7 and seats["2"]["kind"] == "seated", (
        f"就座那张按改号落到座位 2，实得 {seats.get('2')}")
    assert "1" in seats, (
        f"改号腾出来的座位 1 应由候选上标的那条补上，实得 {sorted(seats)}")
    assert seats["1"]["track_id"] == 41 and seats["1"]["kind"] == "exit", seats["1"]


def test_a_rejected_seat_can_be_filled_by_a_labelled_candidate():
    """人工勾了「是否替换」的座位同理——勾了就等于空缺，候选上标的那条要补上。"""
    from pipeline.entryexit import entryexit_candidates

    seg = mkseg(order=62, presenter_num=2)
    dets = (track(7, RIGHT_CX + 400, STANDING_H, 5, kind="seated", order=62)
            + track(41, RIGHT_CX + 100, STANDING_H, 4, kind="exit", order=62))
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 41)
    labels = {"62": {cand["stable_id"]: {"role": "创业者", "seat": 1,
                                         "n_people": 1}},
              "seat_overrides": {"62": {"1": {"rejected": True}}}}
    reg = registry.build_seat_registry([seg], dets, {62: {7: 1}}, labels,
                                       detections_fp="FP1", sheet="测试组")
    seats = reg["segments"]["62"]["seats"]
    assert seats.get("1", {}).get("track_id") == 41, (
        f"被否决的座位 1 应由候选上标的那条补上，实得 {seats.get('1')}")


def test_a_cross_segment_cell_tops_up_an_underfilled_target_seat():
    """相邻段标过来的那一格，要能**补进**目标座位没填满的 2×2。

    用户 2026-09-09 实测城市1：第 16 段 exit track 99 的格 2 标了「下一组创业者
    座位 1」，而第 17 段座位 1 的证据图只有 2 张、四格里空着两格，那一格却没被
    补进去——日志里只有一句「该座位已有人，不覆盖」。

    「不覆盖已占座位」对**顶掉**而言是对的（自动递补不该改写人工可见的座位），
    但补格位是另一回事：座位上的人没变，只是他那张 2×2 没填满，需要同一个人的
    另一张画面顶上。本段内的多条标注早就是这么合并的（Task 56T），跨段那一路
    一直没接上。
    """
    from pipeline.entryexit import entryexit_candidates

    src = mkseg(order=71, presenter_num=1)
    tgt = mkseg(order=72, presenter_num=1)
    dets = (track(41, RIGHT_CX + 100, STANDING_H, 2, kind="exit", order=71)
            + track(52, RIGHT_CX + 300, STANDING_H, 2, kind="exit", order=72))
    c_src = next(c for c in entryexit_candidates(dets, src) if c["track_id"] == 41)
    c_tgt = next(c for c in entryexit_candidates(dets, tgt) if c["track_id"] == 52)
    labels = {
        # 目标段自己标的那条：座位 1 有人了，但只有 2 帧、没填满
        "72": {c_tgt["stable_id"]: {"role": "创业者", "seat": 1, "n_people": 1}},
        # 源段标过来的一格：同一个人的另一张画面
        "71": {c_src["stable_id"]: {"role": "", "seat": None, "n_people": 2,
                                    "cells": [{"cell": 1, "role": "下一组创业者",
                                               "seat": 1}]}},
    }
    reg = registry.build_seat_registry([src, tgt], dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    seat = reg["segments"]["72"]["seats"]["1"]
    assert len(seat["frames"]) > 2, (
        f"目标座位的 2×2 没填满，跨段标过来的那一格该补进去，实得 "
        f"{len(seat['frames'])} 帧")
    assert seat.get("n_cross_frames", 0) >= 1, (
        f"补进来的帧来自相邻段，必须如实记下来：{seat}")


def test_a_cross_segment_label_is_not_counted_twice():
    """目标座位本来就空、由这条跨段标注补上时，不能再把同一批帧补一遍。"""
    from pipeline.entryexit import entryexit_candidates

    src = mkseg(order=73, presenter_num=1)
    tgt = mkseg(order=74, presenter_num=1)
    dets = track(41, RIGHT_CX + 100, STANDING_H, 6, kind="exit", order=73)
    c_src = next(c for c in entryexit_candidates(dets, src) if c["track_id"] == 41)
    labels = {"73": {c_src["stable_id"]: {
        "role": "", "seat": None, "n_people": 2,
        "cells": [{"cell": 1, "role": "下一组创业者", "seat": 1}]}}}
    reg = registry.build_seat_registry([src, tgt], dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    seat = reg["segments"]["74"]["seats"]["1"]
    paths = [(f["frame_path"], f["t"]) for f in seat["frames"]]
    assert len(paths) == len(set(paths)), f"同一帧被补了两次：{paths}"


# --- 评审席：组级的一块，与 42 个座位互不干扰（用户 2026-09-17）-----------

def _judge_scene(n_seg=2):
    """两段，左侧三位评审人坐在一条进深线上。"""
    dets = []
    for order in range(1, n_seg + 1):
        for i, (cx, y2) in enumerate([(200.0, 900.0), (320.0, 780.0), (450.0, 660.0)]):
            for k in range(6):
                dets.append(det(20 + i, cx, 0.35 * H, kind="seated", order=order,
                                t=order * 100 + k, bottom=y2, width=60.0))
    segs = [mkseg(order=o, presenter_num=1) for o in range(1, n_seg + 1)]
    return segs, dets


def test_registry_carries_a_group_level_judges_block(monkeypatch):
    """登记表里评审席是**组级**的一块，恒有 5 个号，空缺留 None。

    用户 2026-09-17：「每组恒定为 5 个专家，只是有的可能不在其中，将交由我
    人工核实」。挂在某一段底下会变成每家企业重复一遍（17×5 条），而那 5 个人
    整场没换过衣服。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene()
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    assert "judges" in reg, sorted(reg)
    j = reg["judges"]
    assert sorted(j["slots"]) == ["1", "2", "3", "4", "5"]
    assert j["slots"]["1"] is not None and j["slots"]["5"] is None
    assert j["n_expected"] == 5 and j["n_resolved"] == 3
    assert j["unresolved"] == [4, 5], j["unresolved"]
    # 离镜头由近及远
    assert j["slots"]["1"]["median_y2"] > j["slots"]["3"]["median_y2"]
    assert j["slots"]["1"]["frames"], "每位评审人要带上证据帧"


def test_judges_block_does_not_touch_the_seat_counts(monkeypatch):
    """评审人不算进 42 个创业者座位的任何一个计数里。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene()
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    assert reg["_meta"]["n_seats_expected"] == len(segs)
    assert reg["_meta"]["n_seats_resolved"] == 0, "左侧的人一个都不许占创业者座位"
    assert reg["_meta"]["n_judges_resolved"] == 3


def test_judges_block_honours_the_human_numbering(monkeypatch):
    """人工改号压过几何基线。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene()
    cands = judge_candidates(dets)
    labels = {"judges": {cands[0]["stable_id"]: {"judge": 4}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    j = reg["judges"]["slots"]
    assert j["4"]["stable_id"] == cands[0]["stable_id"]
    assert j["1"] is None or j["1"]["stable_id"] != cands[0]["stable_id"]


def test_preview_page_shows_the_judges(tmp_path, monkeypatch):
    """核对页也要列出专家 1-5 的 2×2（用户 2026-09-17：entry 与 vlm 两个产出
    里都要有）。图必须与登记表里那几帧同一个 `build_evidence_sheet` 拼出来
    ——另裁一份就是「人核对了一张模型没看过的图」。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene()
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    out = tmp_path / "preview.html"
    registry.render_vlm_preview(reg, out)
    html = out.read_text(encoding="utf-8")
    assert "本组评审人" in html
    assert "专家 1" in html and "专家 5" in html
    assert "尚无候选" in html, "空缺的号要作为缺口显示出来"


def test_preview_judge_cards_match_the_seat_card_size(tmp_path, monkeypatch):
    """核对页的评审人卡片也是一位一行、与座位卡片同一条尺寸规则。

    用户 2026-09-18：五张挤一行太小看不清衣服。两个页面都要改——只改一个，
    人在挑选页看清了、到核对页又看不清，而核对页才是「模型看到的那张」。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene()
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    out = tmp_path / "preview.html"
    registry.render_vlm_preview(reg, out)
    html = out.read_text(encoding="utf-8")
    assert 'class="card jcard' in html, "评审人卡片要复用座位卡片的类"
    import re as _re
    css = _re.search(r"\.jcard\{([^}]*)\}", _re.sub(r"/\*.*?\*/", "", html, flags=_re.S))
    assert css and "width:" not in css.group(1), css and css.group(1)
    row = _re.search(r"\.jrow\{([^}]*)\}", html)
    assert row and "flex-wrap" not in row.group(1), row and row.group(1)


def test_judge_frames_drop_the_cells_the_human_crossed_out(monkeypatch):
    """人工勾掉的格位，其对应的那一帧不再进登记表（用户 2026-09-18）。

    「专家的 2×2 需要每个格子进行人工核实，因为有的格子混进了其他专家」。
    登记表的 `frames` 是**唯一**的下游来源——核对页、交付包的证据图、送模型
    的那张都由它拼出来。在这里滤一次，三处同时干净；分三处各滤一遍必然有一
    处漏掉，而漏掉的表现是模型照旧看见了别人的衣服，不报错。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene()
    cands = judge_candidates(dets)
    sid = cands[0]["stable_id"]
    base = registry.build_seat_registry(segs, dets, {}, None,
                                        detections_fp="FP1", sheet="测试组")
    kept_all = base["judges"]["slots"]["1"]["frames"]
    assert len(kept_all) == 4, kept_all
    pool = [d.t for d in cands[0]["pool_dets"]]

    labels = {"judges": {sid: {"judge": 1, "cell_exclusions": [2]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    got = [f["t"] for f in reg["judges"]["slots"]["1"]["frames"]]
    assert kept_all[1]["t"] not in got, f"勾掉的格还在：{got}"
    # 空出来的位置**不自动顶上**（用户 2026-09-18：「不需要总是勾选 4 个，
    # 默认选的被标记为其他人后，再从后面标的选中进 2×2 递补」）——等人补选。
    assert got == [pool[0], pool[2], pool[3]], (got, pool[:6])


def test_judge_record_reports_which_cells_were_crossed_out(monkeypatch):
    """登记表里要留下「勾掉了哪几格」，人工才核得回去。

    只看剩下三帧是分不出「第 2 格被勾掉了」与「这人本来只拍到三帧」的——
    而这两件事的处置完全不同（前者已核实，后者还欠一次核实）。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene()
    sid = judge_candidates(dets)[0]["stable_id"]
    labels = {"judges": {sid: {"judge": 1, "cell_exclusions": [2, 4]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    assert reg["judges"]["slots"]["1"]["excluded_cells"] == [2, 4]
    assert reg["judges"]["slots"]["2"]["excluded_cells"] == []


def test_judge_cell_exclusions_follow_the_person_across_a_renumber(monkeypatch):
    """改号之后，勾掉的格位跟着**这个人**走，不留在原来的号上。

    按席位号存排除的话，把某位从专家 1 改成专家 4 之后，勾掉的那一格就落到
    了接替 1 号的另一位身上：他好端端的一格被扔掉，而混进来的那一格重新进了
    模型。两头都错，且一处不报错。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene()
    cands = judge_candidates(dets)
    labels = {"judges": {cands[0]["stable_id"]: {"judge": 4, "cell_exclusions": [3]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    slots = reg["judges"]["slots"]
    assert slots["4"]["stable_id"] == cands[0]["stable_id"]
    assert slots["4"]["excluded_cells"] == [3]
    assert cands[0]["pool_dets"][2].t not in [f["t"] for f in slots["4"]["frames"]]
    assert slots["1"]["excluded_cells"] == [] and len(slots["1"]["frames"]) == 4


def test_judge_frames_survive_an_exclusion_beyond_the_real_cell_count(monkeypatch):
    """排除号超出实际张数时照旧——不足 4 帧时余下的格本就是灰块。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene()
    sid = judge_candidates(dets)[0]["stable_id"]
    labels = {"judges": {sid: {"cell_exclusions": [9]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    assert len(reg["judges"]["slots"]["1"]["frames"]) == 4


def test_preview_shows_which_judge_cells_were_crossed_out(tmp_path, monkeypatch):
    """核对页要写明「这一位勾掉了哪几格」（用户 2026-09-18）。

    核对页是「模型看到的那张」的复核面。图上少一格时，人必须能分清那是
    **他自己勾掉的**还是这一位本来就只拍到三帧——前者已核实，后者还欠一次。
    不写出来，两件事在页面上长得一模一样。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene()
    sid = judge_candidates(dets)[0]["stable_id"]
    labels = {"judges": {sid: {"judge": 1, "cell_exclusions": [2]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    out = tmp_path / "preview.html"
    registry.render_vlm_preview(reg, out)
    html = out.read_text(encoding="utf-8")
    assert "人工勾掉" in html, "核对页没说这一位勾掉过格"
    assert "格 2" in html, html[html.find("本组评审人"):][:900]


def _judge_scene_distinct(n_seg=2):
    """同 `_judge_scene`，但**每位评审人的帧时刻各不相同**。

    共用时刻的夹具测不出「帧换到了另一位名下」——目标那一位本来就有一个同样
    时刻的帧，断言恒真。第一版就是这么绿的（2026-09-18）。
    """
    dets = []
    for order in range(1, n_seg + 1):
        for i, (cx, y2) in enumerate([(200.0, 900.0), (320.0, 780.0), (450.0, 660.0)]):
            for k in range(6):
                dets.append(det(20 + i, cx, 0.35 * H, kind="seated", order=order,
                                t=order * 100 + i * 10 + k, bottom=y2, width=60.0))
    segs = [mkseg(order=o, presenter_num=1) for o in range(1, n_seg + 1)]
    return segs, dets


def test_the_distinct_scene_really_has_distinct_times():
    """夹具自检：三位评审人的时刻不许有交集，否则上面几条断言恒真。"""
    segs, dets = _judge_scene_distinct()
    by_cx = {}
    for d in dets:
        by_cx.setdefault(round(d.cx), set()).add(d.t)
    sets = list(by_cx.values())
    assert len(sets) == 3, sorted(by_cx)
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (sets[i] & sets[j]), (i, j, sets[i] & sets[j])


def test_a_cell_reassigned_to_another_judge_moves_to_that_judge(monkeypatch):
    """归给别人的那一格，帧**换到那一位名下**，不是扔掉（用户 2026-09-18）。

    城市1 实测：四位专家共勾掉 7 格，专家1 与专家3 各只剩 1 格。那些格并没有
    坏，只是归错了人——扔掉等于让模型拿一格判整场衣着，比拿四格混着两个人
    还糟。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct()
    cands = judge_candidates(dets)
    base = registry.build_seat_registry(segs, dets, {}, None,
                                        detections_fp="FP1", sheet="测试组")
    moved_t = base["judges"]["slots"]["1"]["frames"][1]["t"]

    # 专家3 自己也勾掉了两格——真实情形正是这样（城市1 专家3 勾掉了 3 格），
    # 他因此腾得出位置来收别人还给他的那一格。
    labels = {"judges": {cands[0]["stable_id"]: {"cells": {"2": 3}},
                         cands[2]["stable_id"]: {"cells": {"3": "x", "4": "x"}}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    one = [f["t"] for f in reg["judges"]["slots"]["1"]["frames"]]
    three = [f["t"] for f in reg["judges"]["slots"]["3"]["frames"]]
    assert moved_t not in one, f"格2 还留在专家1 名下：{one}"
    assert moved_t in three, f"格2 没有换到专家3 名下：{three}"


def test_a_cell_marked_nobody_is_dropped_not_moved(monkeypatch):
    """标「其他人」的格谁都不归——工作人员、路过的，进了谁的 2×2 都是错的。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct()
    cands = judge_candidates(dets)
    base = registry.build_seat_registry(segs, dets, {}, None,
                                        detections_fp="FP1", sheet="测试组")
    dropped_t = base["judges"]["slots"]["1"]["frames"][1]["t"]
    labels = {"judges": {cands[0]["stable_id"]: {"cells": {"2": "x"}}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    for n in ("1", "2", "3", "4", "5"):
        rec = reg["judges"]["slots"][n]
        if rec:
            assert dropped_t not in [f["t"] for f in rec["frames"]], n


def test_donated_cells_never_push_a_judge_past_four(monkeypatch):
    """收进来的格不许把 2×2 撑破——本人的格先占，多出来的不要。

    超过四格的部分 `build_evidence_sheet` 本来就画不下，悄悄多带几帧只会让
    登记表里的 frames 与图上看到的对不上，而一处不报错。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct()
    cands = judge_candidates(dets)
    # 专家3 自己的默认四格全标掉，五格指给他——2×2 只有四格
    labels = {"judges": {cands[0]["stable_id"]: {"cells": {"1": 3, "2": 3, "3": 3}},
                         cands[1]["stable_id"]: {"cells": {"1": 3, "2": 3}},
                         cands[2]["stable_id"]: {"cells": {"1": "x", "2": "x",
                                                           "3": "x", "4": "x"}}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    three = reg["judges"]["slots"]["3"]["frames"]
    assert len(three) == _c.JUDGE_CELL_COUNT, len(three)
    # 而且**装不下的必须说出来**：默默丢掉的话，人指了一格过去、图上没有它、
    # 页面上一个字都没有，他只会以为自己点错了。
    left = reg["judges"]["slots"]["3"]["unplaced_cells"]
    assert len(left) == 1, left
    assert {"from", "cell"} <= set(left[0]), left


def test_a_judge_who_lost_cells_gets_them_back_from_others(monkeypatch):
    """互换：专家1 的格2 其实是专家3、专家3 的格1 其实是专家1——两边都补回来。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct()
    cands = judge_candidates(dets)
    base = registry.build_seat_registry(segs, dets, {}, None,
                                        detections_fp="FP1", sheet="测试组")
    t1 = base["judges"]["slots"]["1"]["frames"][1]["t"]
    t3 = base["judges"]["slots"]["3"]["frames"][0]["t"]
    # 两边各腾一格出来收对方的
    labels = {"judges": {cands[0]["stable_id"]: {"cells": {"2": 3}},
                         cands[2]["stable_id"]: {"cells": {"1": 1}}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    one = [f["t"] for f in reg["judges"]["slots"]["1"]["frames"]]
    three = [f["t"] for f in reg["judges"]["slots"]["3"]["frames"]]
    assert t1 in three and t1 not in one, (t1, one, three)
    assert t3 in one and t3 not in three, (t3, one, three)


def test_registry_records_where_each_donated_cell_came_from(monkeypatch):
    """登记表要写明「这一位收了谁的哪一格」——人工才核得回去。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct()
    cands = judge_candidates(dets)
    labels = {"judges": {cands[0]["stable_id"]: {"cells": {"2": 3}},
                         cands[2]["stable_id"]: {"cells": {"4": "x"}}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    got = reg["judges"]["slots"]["3"]["donated_cells"]
    assert got == [{"from": cands[0]["stable_id"], "cell": 2}], got
    assert reg["judges"]["slots"]["1"]["donated_cells"] == []


def test_preview_shows_donated_and_unplaced_cells(tmp_path, monkeypatch):
    """核对页要写明这一位「收了谁的哪一格」「还有哪一格没装下」。

    核对页是「模型看到的那张」的复核面。收进来的那一格在图上与本人的格长得
    一样，不写出来人就无从复核自己指得对不对；装不下的更要说——默默丢掉的话
    他只会以为自己点错了。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct()
    cands = judge_candidates(dets)
    # 专家3 默认四格全标掉，五格指给他，装不下一格
    labels = {"judges": {cands[0]["stable_id"]: {"cells": {"1": 3, "2": 3, "3": 3}},
                         cands[1]["stable_id"]: {"cells": {"1": 3, "2": 3}},
                         cands[2]["stable_id"]: {"cells": {"1": "x", "2": "x",
                                                           "3": "x", "4": "x"}}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    out = tmp_path / "preview.html"
    registry.render_vlm_preview(reg, out)
    html = out.read_text(encoding="utf-8")
    assert "收进" in html, "核对页没说这一位收了别人的格"
    assert "没装下" in html, "核对页没说有格没装下"


def test_a_short_judge_is_refilled_from_the_cells_he_picked(monkeypatch):
    """标掉几格之后，由**人从备选里补选**把 2×2 填回四格（用户 2026-09-18）。

    城市1 标完五位专家只剩 1/3/3/2/3 格——而每位在盘上有 846 到 2173 帧，材料
    不缺。补选的那几张接在默认还剩的那几格后面，不自动顶上。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    cands = judge_candidates(dets)
    sid = cands[0]["stable_id"]
    labels = {"judges": {sid: {"cells": {"2": "x", "3": "x", "4": "x"},
                               "picks": [5, 6, 7]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    rec = reg["judges"]["slots"]["1"]
    assert len(rec["frames"]) == _c.JUDGE_CELL_COUNT, (
        f"补不满：{len(rec['frames'])} 格")
    pool = cands[0]["pool_dets"]
    want = [pool[0].t] + [d.t for d in pool[4:7]]
    assert [f["t"] for f in rec["frames"]] == want, [f["t"] for f in rec["frames"]]


def test_cells_beyond_the_first_four_can_be_marked_too(monkeypatch):
    """备选那几格同样要能标——顶上来的可能还是同一个错的人。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    cands = judge_candidates(dets)
    sid = cands[0]["stable_id"]
    pool = cands[0]["pool_dets"]
    labels = {"judges": {sid: {"cells": {"2": "x", "5": "x"},
                               "picks": [5, 6]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    got = [f["t"] for f in reg["judges"]["slots"]["1"]["frames"]]
    assert pool[4].t not in got, f"第 5 格标了「其他人」还补了进来：{got}"
    assert pool[5].t in got, f"补选的第 6 格没进来：{got}"


def test_a_judge_never_gets_more_than_four_from_the_pool(monkeypatch):
    """池有十几张，送进 2×2 的永远只有四张——多的画不下。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    for n, rec in reg["judges"]["slots"].items():
        if rec:
            assert len(rec["frames"]) == _c.JUDGE_CELL_COUNT, (n, len(rec["frames"]))


def test_a_spare_cell_can_be_handed_to_another_judge(monkeypatch):
    """备选格也能指给别人——顶上来的那张可能正是另一位。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    cands = judge_candidates(dets)
    pool = cands[0]["pool_dets"]
    labels = {"judges": {
        cands[0]["stable_id"]: {"cells": {"6": 3}},
        cands[2]["stable_id"]: {"cells": {"1": "x", "2": "x", "3": "x",
                                          "4": "x", "5": "x", "6": "x",
                                          "7": "x", "8": "x", "9": "x",
                                          "10": "x", "11": "x", "12": "x",
                                          "13": "x", "14": "x", "15": "x",
                                          "16": "x"}}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    three = [f["t"] for f in reg["judges"]["slots"]["3"]["frames"]]
    assert three == [pool[5].t], three
    assert reg["judges"]["slots"]["3"]["donated_cells"] == [
        {"from": cands[0]["stable_id"], "cell": 6}]


def test_the_human_picks_which_cells_make_up_the_two_by_two(monkeypatch):
    """人工挑中的那几格组成 2×2，不再按池顺序自动顶上（用户 2026-09-18）。

    自动顶上只按面积×清晰度往下顺，而「哪张看得清衣服」机器排得并不准。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    cands = judge_candidates(dets)
    sid = cands[0]["stable_id"]
    pool = cands[0]["pool_dets"]
    labels = {"judges": {sid: {"cells": {"1": "x", "2": "x", "3": "x", "4": "x"},
                               "picks": [9, 2, 11, 5]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    got = [f["t"] for f in reg["judges"]["slots"]["1"]["frames"]]
    # 格2 已标成「其他人」，挑了也不作数
    assert got == [pool[8].t, pool[10].t, pool[4].t], got


def test_the_automatic_order_is_only_the_baseline(monkeypatch):
    """一格都没挑的那几位照旧有四格图——没人动过不等于该空着。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    pool = judge_candidates(dets)[0]["pool_dets"]
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    got = [f["t"] for f in reg["judges"]["slots"]["1"]["frames"]]
    assert got == [d.t for d in pool[:4]], got


def test_a_picked_cell_that_is_not_the_person_is_ignored(monkeypatch):
    """挑中的格若已标成「其他人」，不作数——两处相反的指示里，说「这不是他」
    的那条更具体，硬要送进去就是把别人的脸交给模型。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    cands = judge_candidates(dets)
    sid = cands[0]["stable_id"]
    pool = cands[0]["pool_dets"]
    labels = {"judges": {sid: {"picks": [2, 6],
                               "cells": {"1": "x", "2": "x", "3": "x", "4": "x"}}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    got = [f["t"] for f in reg["judges"]["slots"]["1"]["frames"]]
    assert pool[1].t not in got, got
    assert got == [pool[5].t], got


def test_the_registry_records_what_was_picked(monkeypatch):
    """登记表要写明这四格是人挑的还是自动排的——两者的可信度完全不同。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    sid = judge_candidates(dets)[0]["stable_id"]
    labels = {"judges": {sid: {"picks": [9, 2]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    assert reg["judges"]["slots"]["1"]["picked_cells"] == [9, 2]
    assert reg["judges"]["slots"]["2"]["picked_cells"] == []


def test_picks_come_before_cells_handed_over_by_others(monkeypatch):
    """人挑满四格时，别人还回来的格排在后面、装不下就报出来。

    挑选是「这一张进 2×2」的直接指令，还回来的格说的是「这一帧是那个人」
    ——后者是身份判断，不是排版判断。装不下的照旧写进 `unplaced_cells`，
    人看得见就能回去取消一格。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    cands = judge_candidates(dets)
    pool0 = cands[0]["pool_dets"]
    pool2 = cands[2]["pool_dets"]
    labels = {"judges": {cands[0]["stable_id"]: {"cells": {"6": 3}},
                         cands[2]["stable_id"]: {}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    rec = reg["judges"]["slots"]["3"]
    assert [f["t"] for f in rec["frames"]] == [d.t for d in pool2[:4]]
    assert rec["unplaced_cells"] == [{"from": cands[0]["stable_id"], "cell": 6}]
    assert pool0[5].t not in [f["t"] for f in rec["frames"]]


def test_preview_says_whether_the_four_cells_were_picked_by_hand(tmp_path, monkeypatch):
    """核对页要分清这四格是人挑的还是按清晰度自动排的。

    两者可信度完全不同：自动排的那几位还欠一次核实，而页面上长得一模一样。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    sid = judge_candidates(dets)[0]["stable_id"]
    labels = {"judges": {sid: {"picks": [9, 2, 11, 5]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    out = tmp_path / "preview.html"
    registry.render_vlm_preview(reg, out)
    html = out.read_text(encoding="utf-8")
    assert "人工挑选" in html, "核对页没说这四格是人挑的"
    assert "格 9、2、11、5" in html, html[html.find("本组评审人"):][:1200]


def test_picks_top_up_the_default_four_rather_than_replacing_them(monkeypatch):
    """挑选是**递补**，不是整组替换（用户 2026-09-18）。

    「不需要总是勾选 4 个，默认选的被标记为其他人后，再从后面标的选中进 2×2
    递补」。城市1 专家5 只把格1 标成其他人、补选了格6，期望的是 2、3、4、6
    ——整组替换的话他只剩格6 一格，另外两格好端端的画面凭空没了。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    cands = judge_candidates(dets)
    sid = cands[0]["stable_id"]
    pool = cands[0]["pool_dets"]
    labels = {"judges": {sid: {"cells": {"1": "x"}, "picks": [6]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    got = [f["t"] for f in reg["judges"]["slots"]["1"]["frames"]]
    assert got == [pool[1].t, pool[2].t, pool[3].t, pool[5].t], got


def test_nothing_is_auto_promoted_into_a_gap(monkeypatch):
    """默认那四格空出来的位置**不自动顶上**，等人来补。

    自动顶上只按面积×清晰度往下顺，人却会以为那一格是他挑的。宁可 2×2 少
    一格（页面上写着还差几格），也不悄悄塞一张没人看过的脸进去。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    cands = judge_candidates(dets)
    sid = cands[0]["stable_id"]
    pool = cands[0]["pool_dets"]
    labels = {"judges": {sid: {"cells": {"2": "x"}}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    got = [f["t"] for f in reg["judges"]["slots"]["1"]["frames"]]
    assert got == [pool[0].t, pool[2].t, pool[3].t], got
    assert pool[4].t not in got, "第 5 格自动顶上来了"


def test_an_untouched_judge_still_gets_the_default_four(monkeypatch):
    """一处没动过的那几位照旧是默认的前四格。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    pool = judge_candidates(dets)[0]["pool_dets"]
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    got = [f["t"] for f in reg["judges"]["slots"]["1"]["frames"]]
    assert got == [d.t for d in pool[:4]], got


def test_extra_picks_beyond_the_gaps_are_reported(monkeypatch):
    """补选多于空缺时，多出来的要说出来——不然人勾了一格、图上没有它。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    sid = judge_candidates(dets)[0]["stable_id"]
    labels = {"judges": {sid: {"cells": {"1": "x"}, "picks": [6, 9]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    rec = reg["judges"]["slots"]["1"]
    assert rec["picked_cells"] == [6, 9]
    assert rec["unplaced_picks"] == [9], rec["unplaced_picks"]


def test_preview_reports_picks_that_did_not_fit(tmp_path, monkeypatch):
    """补选多于空缺时，核对页要说出来——不然人勾了一格、图上没有它。"""
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    sid = judge_candidates(dets)[0]["stable_id"]
    labels = {"judges": {sid: {"cells": {"1": "x"}, "picks": [6, 9]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    out = tmp_path / "preview.html"
    registry.render_vlm_preview(reg, out)
    html = out.read_text(encoding="utf-8")
    assert "补选没装下" in html, html[html.find("本组评审人"):][:1400]
    assert "格 9" in html


def test_registry_realigns_judge_labels_after_a_re_extraction(monkeypatch, capsys):
    """重抽之后，挪了位的评审人标注要对回来，并把对上多少条**打印出来**。

    标识由位置中位数导出，重抽任何一段都会让它挪零点几个像素。不对齐就静默
    当作没人标过——而这是工作有没有丢的唯一可见迹象。
    """
    from pipeline import config as _c
    from pipeline.judges import judge_candidates
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    cands = judge_candidates(dets)
    sid = cands[0]["stable_id"]
    cx, y2 = sid.split("#")[1].split(",")
    stale = f"judge#{int(cx) - 1},{int(y2) + 1}"        # 挪了一个像素的旧标识
    labels = {"judges": {stale: {"cells": {"1": "x"}, "picks": [6]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    rec = reg["judges"]["slots"]["1"]
    assert rec["excluded_cells"] == [1], rec["excluded_cells"]
    assert rec["picked_cells"] == [6], rec["picked_cells"]
    out = capsys.readouterr().out
    assert "评审席标注" in out, out
    assert "1/1" in out, out


def test_the_preview_toc_lists_the_judges_first(tmp_path, monkeypatch):
    """核对页左侧目录也要有评审席一栏，且排在分段目录**之前**（用户 2026-09-18）。

    两个页面都要有：只在挑选页给，人到核对页复核时又找不着评审席在哪一屏。
    位置（cx,y2）同样要写——席位号只按进深排，一个贴着画面边缘的人插进序列
    就会把后面的整体顶掉一位（城市2 实测 `judge#35,668`，cx=35）。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    out = tmp_path / "preview.html"
    registry.render_vlm_preview(reg, out)
    html = out.read_text(encoding="utf-8")
    # 只看 <nav> 里的次序：「分段目录」这几个字在页首说明里也出现，
    # 拿整页去 index 会找到那一处，断言就成了对着说明文字比大小。
    nav = html[html.index('<nav class="toc"'):]
    nav = nav[:nav.index("</nav>")]
    i_j = nav.index("评审席")
    i_c = nav.index("分段目录")
    assert i_j < i_c, f"评审席那一栏排在了分段目录后面：{nav[:200]}"
    # 每位的位置都要写进目录
    for n, rec in reg["judges"]["slots"].items():
        if rec:
            assert str(int(rec["median_cx"])) in nav[i_j:i_c], (n, nav[i_j:i_c])


def test_the_preview_toc_judge_rows_link_to_their_cards(tmp_path, monkeypatch):
    """目录里每位评审人都要能点回他自己那张卡片。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    out = tmp_path / "preview.html"
    registry.render_vlm_preview(reg, out)
    html = out.read_text(encoding="utf-8")
    for n, rec in reg["judges"]["slots"].items():
        if rec:
            assert f'id="judge-{n}"' in html, n
            assert f'href="#judge-{n}"' in html, n


def test_the_preview_toc_hides_extras_once_every_seat_is_filled(tmp_path, monkeypatch):
    """五个席位都坐满时，目录里不再列「未入席 N 位」（用户 2026-09-18）。

    那一行是给**还有空号**时用的——要从候选里提人补位才需要一个入口。坐满之后
    它只是噪声，而目录越短越好用。

    只去掉目录那一行：评审席区块里的候选卡片照旧列出来，否则哪天发现坐错了号
    （城市2 的 `judge#35,668` 就是贴着画面边缘挤进 3 号位的），想把人提回席位
    却找不着他。
    """
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    # 席位数压到 2，而场上有 3 位候选——坐满且**确实有人没入席**。
    # 三位候选配三个席位的话 extra 本来就是空的，断言恒真（第一版正是这样）。
    monkeypatch.setattr(_c, "JUDGE_COUNT", 2)
    segs, dets = _judge_scene_distinct(n_seg=8)
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    assert reg["judges"]["n_resolved"] == 2, reg["judges"]["n_resolved"]
    assert reg["judges"]["extra_candidates"], "夹具前提：要有没入席的候选"
    out = tmp_path / "p.html"
    registry.render_vlm_preview(reg, out)
    html = out.read_text(encoding="utf-8")
    nav = html[html.index('<nav class="toc"'):]
    nav = nav[:nav.index("</nav>")]
    assert "未入席" not in nav, nav[:600]


def test_the_preview_toc_still_lists_extras_when_a_seat_is_empty(tmp_path, monkeypatch):
    """还有空号时那一行必须在——它正是「从这里把人提上来」的入口。"""
    from pipeline import config as _c
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    reg = registry.build_seat_registry(segs, dets, {}, None,
                                       detections_fp="FP1", sheet="测试组")
    assert reg["judges"]["unresolved"], "夹具前提：要有空号"
    out = tmp_path / "p2.html"
    registry.render_vlm_preview(reg, out)
    html = out.read_text(encoding="utf-8")
    nav = html[html.index('<nav class="toc"'):]
    nav = nav[:nav.index("</nav>")]
    assert "尚无候选" in nav, nav[:600]


def test_a_right_side_person_marked_as_a_judge_fills_the_empty_seat(monkeypatch):
    """右侧候选里标成「专家 5」的格位，把空着的第 5 号补上（用户 2026-09-19）。

    文化4 的专家5 坐在右侧靠投影仪的位置，`judge_candidates` 只看左区，他一个
    候选都出不来：评审席 4/5、第 5 号空缺，「未入席的候选」也是空的。出口是
    逐格身份那一栏——标成「专家 5」，那一帧就归到第 5 位名下。
    """
    from pipeline import config as _c
    from pipeline.entryexit import candidates_for_labels
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    # 右区加一个人：进出场窗口里站在 cx=1100
    for o in (1, 2):
        for k in range(8):
            dets.append(det(77, 1100.0, 0.5 * H, kind="exit", order=o,
                            t=o * 100 + 50 + k, bottom=900.0, width=70.0))
    cands = candidates_for_labels(dets, segs[0], 1440, 1080)
    assert cands, "夹具前提：右区要有候选"
    sid = cands[0]["stable_id"]
    labels = {"1": {sid: {"n_people": 2, "cells": [
        {"cell": 1, "role": "专家", "seat": 5}]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    five = reg["judges"]["slots"]["5"]
    assert five is not None, "第 5 号仍然空着"
    assert len(five["frames"]) == 1, five["frames"]
    assert five["donated_cells"] == [{"from": sid, "cell": 1}], five["donated_cells"]


def test_an_empty_slot_filled_only_by_donation_has_no_stable_id(monkeypatch):
    """全靠别人还回来的那个号没有自己的候选——标识留空，不许编一个。

    编一个就等于凭空造出一位「检测到的评审人」，下游分不清他是检出来的还是
    人工拼出来的。
    """
    from pipeline import config as _c
    from pipeline.entryexit import candidates_for_labels
    monkeypatch.setattr(_c, "JUDGE_MIN_FRAMES", 5)
    segs, dets = _judge_scene_distinct(n_seg=8)
    for o in (1, 2):
        for k in range(8):
            dets.append(det(77, 1100.0, 0.5 * H, kind="exit", order=o,
                            t=o * 100 + 50 + k, bottom=900.0, width=70.0))
    sid = candidates_for_labels(dets, segs[0], 1440, 1080)[0]["stable_id"]
    labels = {"1": {sid: {"n_people": 2, "cells": [
        {"cell": 1, "role": "专家", "seat": 5}]}}}
    reg = registry.build_seat_registry(segs, dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    five = reg["judges"]["slots"]["5"]
    assert five["stable_id"] is None, five["stable_id"]
    assert five["n_frames"] == 1
    assert reg["judges"]["n_resolved"] == 4, "凑出来的号不算「已排定」"


def test_a_cross_segment_flow_into_a_full_seat_still_gets_a_cell():
    """相邻段标过来的画面，目标座位**已满四格**时也要分到格子。

    这是 fill_plan 那句提示的依据（用户 2026-09-27「修改措辞」）：日志原来写
    「跨段流转未落位（该座位已有人，不覆盖）」，读起来像是标注没生效；而多个
    来源轮流出格之后，这种流转的画面总会并进目标座位的 2×2。实测电子10 那 7 条
    「未落位」全部已经并入（各 1–3 格）。提示要改口说「并入」，这里先钉住它
    确实会并入——不然改掉的就是一句真话。
    """
    from pipeline.entryexit import entryexit_candidates

    src = mkseg(order=73, presenter_num=1)
    tgt = mkseg(order=74, presenter_num=1)
    dets = (track(43, RIGHT_CX + 100, STANDING_H, 4, kind="exit", order=73)
            + track(54, RIGHT_CX + 300, STANDING_H, 4, kind="exit", order=74))
    c_src = next(c for c in entryexit_candidates(dets, src) if c["track_id"] == 43)
    c_tgt = next(c for c in entryexit_candidates(dets, tgt) if c["track_id"] == 54)
    assert len(c_tgt["best_dets"]) == 4, "装置前提：目标座位自己就有四格"
    labels = {
        "74": {c_tgt["stable_id"]: {"role": "创业者", "seat": 1, "n_people": 1}},
        "73": {c_src["stable_id"]: {"role": "下一组创业者", "seat": 1, "n_people": 1}},
    }
    reg = registry.build_seat_registry([src, tgt], dets, {}, labels,
                                       detections_fp="FP1", sheet="测试组")
    seat = reg["segments"]["74"]["seats"]["1"]
    assert len(seat["frames"]) == 4, seat
    assert seat.get("n_cross_frames", 0) >= 1, (
        f"目标座位满四格时，相邻段标过来的画面也要分到格子：{seat}")
