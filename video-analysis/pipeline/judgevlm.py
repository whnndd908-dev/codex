"""阶段 ⑦ 之评审席：把每位评审人的 2×2 交给 VLM 打着装正式度。

用户 2026-09-19：「阶段 7 的 VLM 打标接评审人」。

与创业者那一路的三条区别，每一条都决定了它为什么不能直接塞进
`assemble.assemble_annotations`：

1. **组级，不是段级。** 同一批评审人坐一整天，衣服不变。产出每组
   `JUDGE_COUNT` 条，而不是每家企业重复一遍（17 段 × 5 位 = 85 条纯冗余，
   后期按企业行 join 时还会把同一件衣服计上十几次）。`PersonAnnotation`
   的每一个字段——seg_order、company、seat_index、presenter_num、
   speaks_pre——对评审人都没有意义，硬塞就得给每个下游消费者加一条
   `if seg_order == 0` 的特例。

2. **提示词是评审人版**（`annotate.JUDGE_PROMPT`）：没有企业、没有本组人数。
   量表锚点与 JSON 结构由创业者版切出来，逐字相同——两套量表一旦分叉，跨组
   比较就是在比两把尺子，而数字看上去一样合理。

3. **图来自登记表的 `frames`**，与交付包里 `evidence/judgeN.png`、核对页上
   那张同源。人核对过的图与模型看到的图必须是同一张，这是本项目的第一条
   非协商项。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pipeline import config
from pipeline.annotate import JUDGE_PROMPT, ask_vlm, build_evidence_sheet
from pipeline.assemble import describe_layout


def _frames_of(rec: dict) -> tuple[list[Path], list[tuple]]:
    """登记表一条评审人记录里的帧 → `build_evidence_sheet` 的两个入参。"""
    paths, boxes = [], []
    for fr in rec.get("frames") or []:
        box = fr.get("box") or []
        if not fr.get("frame_path") or len(box) != 4:
            continue
        paths.append(Path(fr["frame_path"]))
        boxes.append(tuple(float(v) for v in box))
    return paths, boxes


def frames_digest(rec: dict) -> str:
    """这一位当前这几帧的摘要，用来判断「还是不是上次打标的那张图」。

    判据是**帧集合**，不是席位号也不是标识：席位号会被人工改来改去，而改号
    不改图时旧结论照样有效；反过来帧换了图就换了，旧结论是对着另一张图给的，
    再用就是把 A 的衣着记到 B 头上（与阶段 3 的增量检测同一条思路：摘要变了
    才重跑）。
    """
    parts = [f"{fr.get('frame_path')}|{fr.get('t')}|{fr.get('box')}"
             for fr in (rec.get("frames") or [])]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def assemble_judges(registry: dict, sheet: str, use_vlm: bool = True,
                    prior: list[dict] | None = None) -> list[dict]:
    """每位**有图**的评审人一条记录。

    空号不产出：那是一个缺口，不是一条待打标的记录——产出一条空记录会让下游
    分不清「这组只有四位」与「第五位还没人核实」。

    `prior` 是上一轮的产出。帧摘要没变的直接复用，不重复调用模型（与创业者
    那一路的增量打标同一条口径）。
    """
    by_judge = {}
    for row in prior or []:
        try:
            by_judge[int(row.get("judge"))] = row
        except (TypeError, ValueError):
            continue

    block = (registry or {}).get("judges") or {}
    slots = block.get("slots") or {}
    out: list[dict] = []
    for n in range(1, int(block.get("n_expected", config.JUDGE_COUNT)) + 1):
        rec = slots.get(str(n))
        if not rec:
            continue
        paths, boxes = _frames_of(rec)
        if not paths:
            continue
        digest = frames_digest(rec)
        old = by_judge.get(n)
        if old is not None and old.get("frames_digest") == digest:
            out.append(dict(old))
            continue

        row = {
            "sheet": sheet,
            "judge": n,
            "stable_id": rec.get("stable_id"),
            "median_cx": rec.get("median_cx"),
            "median_y2": rec.get("median_y2"),
            "n_frames": rec.get("n_frames"),
            "n_segments": rec.get("n_segments"),
            "n_cells": len(paths),
            "frames_digest": digest,
            "confirmed": int(bool(rec.get("confirmed"))),
            "vlm": {},
            "vlm_error": "",
        }
        sheet_png = build_evidence_sheet(paths, boxes)
        if use_vlm:
            # 构成如实描述：评审人恒为就座镜头，下装与站姿不可见。谎报构成
            # 是系统性偏差里最可能活到发表的那一种。
            ctx = {"layout": describe_layout(len(paths), 0, 0),
                   "person_id": f"judge{n}"}
            try:
                row["vlm"] = ask_vlm(sheet_png, ctx, prompt=JUDGE_PROMPT)
            except Exception as exc:                      # noqa: BLE001
                # 调用失败与「没检出人」是两种缺失，必须能从产出里分开。
                row["vlm_error"] = f"{type(exc).__name__}: {exc}"
        out.append(row)
    return out
