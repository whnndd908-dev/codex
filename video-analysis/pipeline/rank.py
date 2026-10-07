"""阶段 ⑦：项目排名，为后续实验挑选素材。

两套分数并列输出：纯测量置信度，与加权实验可用性。
经用户确认，推荐项目取实验可用性第一名，但测量置信度排名一并保留供对照。
"""
from __future__ import annotations

from statistics import pstdev

from pipeline.annotate import PersonAnnotation
from pipeline.validate import RULE_COUNT, RULE_GENDER, Violation

_QUALITY = {"good": 1.0, "fair": 0.6, "poor": 0.2}
_SEAT_PRIOR = {1: 0.5, 2: 0.85, 3: 1.0, 4: 1.0}

# 各分量权重之和为 1
_W = {"quality": 0.25, "agree": 0.2, "border": 0.1, "stability": 0.15,
      "vlm_conf": 0.15, "seat": 0.15}

# suspect_staff=1 意味着「可能不是创业者本人」，严重性与人数/性别闸门
# 同级；但安全网只标记不删除、人工可在画廊恢复，故用折扣而非归零。
_STAFF_PENALTY = 0.3

# 个体测量置信度低于此阈值时，top_reason 显式给出警示——防止把
# 「靠猜出来的分数造成的队内差异」包装成「有对比空间」的卖点。
_LOW_CONF = 0.5


def person_confidence(a: PersonAnnotation) -> float:
    q = _QUALITY.get(a.vlm.get("evidence_quality", "poor"), 0.2)
    agree = 1.0 if a.pixel.get("hue_class_agree", 1) else 0.4
    border = 0.6 if a.pixel.get("hue_class_borderline", 0) else 1.0
    stab = a.pixel.get("outer_color_stability", 0.0) or 0.0
    stability = max(0.0, 1.0 - stab / 90.0)
    vlm_conf = float(a.vlm.get("confidence", 0.5))
    seat = _SEAT_PRIOR.get(a.seat_index, 0.5)

    score = (_W["quality"] * q + _W["agree"] * agree + _W["border"] * border
             + _W["stability"] * stability + _W["vlm_conf"] * vlm_conf
             + _W["seat"] * seat)
    if a.suspect_staff:
        score *= _STAFF_PENALTY
    return max(0.0, min(1.0, score))


def project_confidence(
    anns: list[PersonAnnotation], count_ok: bool, gender_ok: bool
) -> float:
    """人数或性别校验未过则归零——编码可能整体错位，不应进入候选。"""
    if not anns or not count_ok or not gender_ok:
        return 0.0
    return sum(person_confidence(a) for a in anns) / len(anns)


def project_experiment_score(
    conf: float, n: int, sharpness_norm: float, diversity: float
) -> float:
    """在测量置信度上加权实验可用性。

    三个权重为经验设定，仅影响排序、不影响任何测量值；
    输出时三个分量分开列示，以便看出名次由哪一项决定。
    """
    size_factor = 1.0 if n >= 2 else 0.7
    # 清晰度只做有界调节（0.75–1.0，最大 1.33 倍），不做完整乘性因子。
    # 原来的 max(0.1, sharpness_norm) 摆幅达 10 倍，是四个杠杆里最大的，
    # 而用户把「分辨率」明确排在优先级最末（准确度 > 样本保留 > 分辨率）。
    # 改后各杠杆最大摆幅：conf 0→1（基数）> size 1.43x > sharpness 1.33x
    # > diversity 1.20x，与该优先级一致。
    #
    # sharpness_norm 名义上由 pipeline.assemble.normalize_sharpness 产出且已
    # clip 到 [0,1]，但那个模块尚不存在——目前只是 task-15-brief 里的一份
    # 文档承诺。清晰度是被用户排在优先级最末的因素，它的摆幅上界不该依赖
    # 另一个尚未写出的模块是否记得 clip：入参 5.0 会让本式给出 1.6，是意图
    # 上界 0.8 的两倍，等于把刚收紧的杠杆重新放开。故在此无条件钳制。
    sharpness_norm = min(1.0, max(0.0, sharpness_norm))
    #
    # diversity 只做接近时的加权，最多 +20%：用户明确的优先级是
    # 识别准确度 > 样本保留 > 分辨率，diversity 不在其中。原来的
    # (1.0 + diversity) 最多能让分数翻倍，足以让 conf=0.5 的项目
    # 反超 conf=0.9 的项目（审查给出定量反例），与该优先级直接冲突。
    return conf * size_factor * (0.75 + 0.25 * sharpness_norm) * (1.0 + 0.2 * diversity)


def _diversity(anns: list[PersonAnnotation]) -> float:
    """队内正式度差异，归一到 0–1。用于判断能否构造对比条件。"""
    vals = [float(a.vlm.get("formality_1_5", 0) or 0) for a in anns]
    if len(vals) < 2:
        return 0.0
    return min(1.0, pstdev(vals) / 2.0)


def rank_projects(
    annotations: list[PersonAnnotation], violations: list[Violation]
) -> list[dict]:
    by_order: dict[int, list[PersonAnnotation]] = {}
    for a in annotations:
        by_order.setdefault(a.seg_order, []).append(a)

    failed_count = {v.seg_order for v in violations if v.rule == RULE_COUNT}
    failed_gender = {v.seg_order for v in violations if v.rule == RULE_GENDER}

    rows = []
    for order, anns in by_order.items():
        conf = project_confidence(
            anns, count_ok=order not in failed_count, gender_ok=order not in failed_gender
        )
        sharp = sum(float(a.pixel.get("sharpness_norm", 0.5)) for a in anns) / len(anns)
        div = _diversity(anns)
        exp = project_experiment_score(conf, len(anns), sharp, div)
        person_confs = [person_confidence(a) for a in anns]
        rows.append({
            "order": order,
            "company": anns[0].company,
            "n_presenters": len(anns),
            "confidence_score": round(conf, 4),
            "experiment_score": round(exp, 4),
            "sharpness_norm": round(sharp, 4),
            "diversity": round(div, 4),
            "_min_person_confidence": min(person_confs) if person_confs else 0.0,
        })

    # 二级排序键 order：同分时名次不能取决于 annotations 入参的原始顺序。
    for key, rank_name in (("confidence_score", "confidence_rank"),
                           ("experiment_score", "experiment_rank")):
        ordered = sorted(rows, key=lambda x: (-x[key], x["order"]))
        for i, r in enumerate(ordered, start=1):
            r[rank_name] = i

    # 主排序键为置信度（用户 2026-08-16 定版）。
    #
    # 此前主排序键是 experiment_score = 置信度 × 人数 × 清晰度 × 队内差异，
    # 四项权重全是实现者自定的、无依据。三组重跑时它真的与置信度排名冲突了
    # ——推荐了 A 而置信度第一名是 B，只好在理由里并列两个公司名让用户自己
    # 取舍（原 Ruling R44 的补救）。用户要的是「系统最有信心的项目」，掺进
    # 人数与清晰度只会让结论无法解释，于是把排序键直接换成置信度，R44 那段
    # 并列说明也随之失去存在前提（排第一的就是置信度第一，不可能再分歧）。
    # experiment_score / diversity 仍照常计算并作为列输出，想用的人可自行
    # 排序，只是不再决定名次。
    rows.sort(key=lambda r: (r["confidence_rank"], r["order"]))
    if rows:
        t = rows[0]
        reason = (
            f"测量置信度 {t['confidence_score']:.3f}；{t['n_presenters']} 位创业者；"
            f"截图清晰度 {t['sharpness_norm']:.2f}；队内正式度差异 {t['diversity']:.2f}"
        )
        if len(rows) > 1:
            runner_up = rows[1]
            if runner_up["confidence_score"] > 0:
                pct = ((t["confidence_score"] - runner_up["confidence_score"])
                       / runner_up["confidence_score"] * 100)
            else:
                pct = 0.0
            reason += (f"，比第二名「{runner_up['company']}」"
                       f"（{runner_up['confidence_score']:.3f}）高 {pct:.0f}%")
        if t["_min_person_confidence"] < _LOW_CONF:
            reason += (
                f"\n⚠ 队内有成员测量置信度仅 {t['_min_person_confidence']:.2f}，"
                "做实验前请先在画廊核对该人的截图。"
            )
        rows[0]["top_reason"] = reason
    for r in rows[1:]:
        r["top_reason"] = ""
    for r in rows:
        del r["_min_person_confidence"]
    return rows
