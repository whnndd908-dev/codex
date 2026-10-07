"""评审席的 VLM 打标（用户 2026-09-19：「阶段 7 的 VLM 打标接评审人」）。"""
import json
from pathlib import Path

import pytest


def _reg(n=2, frames=2):
    slots = {}
    for i in range(1, 6):
        if i > n:
            slots[str(i)] = None
            continue
        slots[str(i)] = {
            "judge": i, "stable_id": f"judge#{i}00,{i}00",
            "median_cx": i * 100.0, "median_y2": i * 100.0,
            "n_frames": 500, "n_segments": 10,
            "frames": [{"frame_path": f"f{i}_{k}.jpg", "t": float(k),
                        "box": [1.0, 2.0, 3.0, 4.0], "conf": 0.9}
                       for k in range(frames)],
        }
    return {"_meta": {"sheet": "测试组"},
            "judges": {"n_expected": 5, "n_resolved": n, "slots": slots,
                       "unresolved": list(range(n + 1, 6)),
                       "extra_candidates": []}}


def test_one_record_per_filled_slot(monkeypatch):
    """每个**有图**的席位一条记录，空号不产出——组级 5 条，不按企业复制。"""
    from pipeline import judgevlm
    monkeypatch.setattr(judgevlm, "build_evidence_sheet", lambda *a, **k: b"PNG")
    monkeypatch.setattr(judgevlm, "ask_vlm",
                        lambda png, ctx, prompt=None: {"formality_1_5": 3})
    got = judgevlm.assemble_judges(_reg(n=2), sheet="测试组")
    assert [r["judge"] for r in got] == [1, 2], got
    assert all(r["sheet"] == "测试组" for r in got)


def test_the_judge_prompt_is_the_one_sent(monkeypatch):
    """必须送评审人那一版 prompt——送创业者版就是拿企业信息去校验一个没有
    企业的人，而两版量表一旦分叉，跨组比较就是在比两把尺子。"""
    from pipeline import annotate, judgevlm
    seen = []
    monkeypatch.setattr(judgevlm, "build_evidence_sheet", lambda *a, **k: b"PNG")
    monkeypatch.setattr(judgevlm, "ask_vlm",
                        lambda png, ctx, prompt=None: seen.append(prompt) or {})
    judgevlm.assemble_judges(_reg(n=1), sheet="测试组")
    assert seen == [annotate.JUDGE_PROMPT]


def test_no_company_or_seat_reaches_the_context(monkeypatch):
    """上下文里不许出现企业、座位号、人数——评审人没有这些，塞进去就是
    给模型一个无关公司的先验（且违反「绝不把人工结论喂给模型」）。"""
    from pipeline import judgevlm
    seen = []
    monkeypatch.setattr(judgevlm, "build_evidence_sheet", lambda *a, **k: b"PNG")
    monkeypatch.setattr(judgevlm, "ask_vlm",
                        lambda png, ctx, prompt=None: seen.append(ctx) or {})
    judgevlm.assemble_judges(_reg(n=1), sheet="测试组")
    assert "company" not in seen[0] and "presenter_num" not in seen[0]
    assert "seat_index" not in seen[0]
    assert "layout" in seen[0], "必须如实告知图里拍到了什么"


def test_the_sheet_is_built_from_the_registry_frames(monkeypatch):
    """图必须由登记表里那几帧拼出来——与人工核对过的、交付包里的那张同源。"""
    from pipeline import judgevlm
    calls = []
    monkeypatch.setattr(judgevlm, "build_evidence_sheet",
                        lambda paths, boxes, **k: calls.append((paths, boxes)) or b"P")
    monkeypatch.setattr(judgevlm, "ask_vlm", lambda *a, **k: {})
    judgevlm.assemble_judges(_reg(n=1, frames=3), sheet="测试组")
    paths, boxes = calls[0]
    assert [p.name for p in paths] == ["f1_0.jpg", "f1_1.jpg", "f1_2.jpg"]
    assert boxes == [(1.0, 2.0, 3.0, 4.0)] * 3


def test_a_failed_call_is_recorded_not_swallowed(monkeypatch):
    """调用失败要如实记进 `vlm_error`，不能当成「没检出」——两者处置不同。"""
    from pipeline import judgevlm
    def boom(*a, **k): raise RuntimeError("503")
    monkeypatch.setattr(judgevlm, "build_evidence_sheet", lambda *a, **k: b"P")
    monkeypatch.setattr(judgevlm, "ask_vlm", boom)
    got = judgevlm.assemble_judges(_reg(n=1), sheet="测试组")
    assert got[0]["vlm"] == {}
    assert "503" in got[0]["vlm_error"]


def test_a_slot_without_frames_is_skipped(monkeypatch):
    """没有图的席位不调模型——空号是缺口，不是一条待打标的记录。"""
    from pipeline import judgevlm
    reg = _reg(n=2)
    reg["judges"]["slots"]["2"]["frames"] = []
    monkeypatch.setattr(judgevlm, "build_evidence_sheet", lambda *a, **k: b"P")
    n = []
    monkeypatch.setattr(judgevlm, "ask_vlm", lambda *a, **k: n.append(1) or {})
    got = judgevlm.assemble_judges(reg, sheet="测试组")
    assert [r["judge"] for r in got] == [1]
    assert len(n) == 1


def test_previous_results_are_reused(monkeypatch):
    """已经打过标的不重复烧钱——与创业者那一路同一条增量口径。

    复用的判据是**这一位的帧集合没变**：帧变了图就变了，旧结论是对着另一张
    图给的。
    """
    from pipeline import judgevlm
    monkeypatch.setattr(judgevlm, "build_evidence_sheet", lambda *a, **k: b"P")
    calls = []
    monkeypatch.setattr(judgevlm, "ask_vlm",
                        lambda *a, **k: calls.append(1) or {"formality_1_5": 4})
    first = judgevlm.assemble_judges(_reg(n=2), sheet="测试组")
    assert len(calls) == 2
    again = judgevlm.assemble_judges(_reg(n=2), sheet="测试组", prior=first)
    assert len(calls) == 2, "又调了一遍"
    assert [r["vlm"] for r in again] == [r["vlm"] for r in first]


def test_changed_frames_force_a_refresh(monkeypatch):
    """帧集合变了就必须重打——旧结论是对着另一张图给的。"""
    from pipeline import judgevlm
    monkeypatch.setattr(judgevlm, "build_evidence_sheet", lambda *a, **k: b"P")
    calls = []
    monkeypatch.setattr(judgevlm, "ask_vlm", lambda *a, **k: calls.append(1) or {})
    first = judgevlm.assemble_judges(_reg(n=1, frames=2), sheet="测试组")
    judgevlm.assemble_judges(_reg(n=1, frames=3), sheet="测试组", prior=first)
    assert len(calls) == 2, "帧变了却复用了旧结论"


def test_use_vlm_false_skips_the_call(monkeypatch):
    """`use_vlm=False` 只出记录不花钱，供离线重算与测试。"""
    from pipeline import judgevlm
    monkeypatch.setattr(judgevlm, "build_evidence_sheet", lambda *a, **k: b"P")
    def boom(*a, **k): raise AssertionError("不该调模型")
    monkeypatch.setattr(judgevlm, "ask_vlm", boom)
    got = judgevlm.assemble_judges(_reg(n=2), sheet="测试组", use_vlm=False)
    assert len(got) == 2 and all(r["vlm"] == {} for r in got)
