"""校验规则。全部为报警，不静默通过，也不自动删除数据。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pipeline.annotate import PersonAnnotation
from pipeline.schedule import Segment


@dataclass(frozen=True)
class Violation:
    rule: str
    seg_order: int
    person_id: str
    detail: str


# 规则名提成常量：rank.py 要按名字筛违规，两处各写一份字面量的话，
# 任何一侧改了拼写都会让筛选静默失效而测试照样全绿（审查已变异实测复现）。
RULE_COUNT = "人数闸门"
RULE_SEAT_UNIQUE = "座位唯一"
RULE_GENDER = "性别闸门"
RULE_STAFF_SAFETY_NET = "工作人员安全网"
RULE_COLOR_STABILITY = "颜色稳定性"
RULE_HUE_DUAL_TRACK = "色相分类双轨"
RULE_HUE_BORDERLINE = "色相边界样本"
RULE_ROW_COUNT = "行数闸门"
RULE_VIEW_CONFLICT = "视角外层不一致"
RULE_VLM_FAILED = "VLM 调用失败"


def check_all(
    annotations: list[PersonAnnotation], segments: list[Segment]
) -> list[Violation]:
    out: list[Violation] = []
    by_order: dict[int, list[PersonAnnotation]] = {}
    for a in annotations:
        by_order.setdefault(a.seg_order, []).append(a)

    for s in segments:
        got = by_order.get(s.order, [])

        if len(got) != s.presenter_num:
            out.append(Violation(
                RULE_COUNT, s.order, "",
                f"标注 {s.presenter_num} 人，检出 {len(got)} 人",
            ))

        seats = Counter(a.seat_index for a in got)
        for seat, n in seats.items():
            if n > 1:
                out.append(Violation(
                    RULE_SEAT_UNIQUE, s.order, "",
                    f"座位 {seat} 被分配了 {n} 次",
                ))

        for a in got:
            judged = a.vlm.get("gender_judged", "不确定")
            if judged in ("男", "女"):
                expected = "女" if a.gender_female == 1 else "男"
                if judged != expected:
                    out.append(Violation(
                        RULE_GENDER, s.order, a.person_id,
                        f"标注为{expected}，模型判为{judged}",
                    ))
            if a.suspect_staff:
                out.append(Violation(
                    RULE_STAFF_SAFETY_NET, s.order, a.person_id,
                    "跨片段外观聚类命中，请人工确认",
                ))
            stab = a.pixel.get("outer_color_stability")
            if stab is not None and stab > 45.0:
                out.append(Violation(
                    RULE_COLOR_STABILITY, s.order, a.person_id,
                    f"跨帧色相标准差 {stab:.1f}°，超阈值",
                ))
            if a.pixel.get("hue_class_agree") == 0:
                out.append(Violation(
                    RULE_HUE_DUAL_TRACK, s.order, a.person_id,
                    f"像素判 {a.pixel.get('hue_class_pixel')}，VLM 判 {a.vlm.get('outer_hue_class')}",
                ))
            if a.pixel.get("hue_class_borderline") == 1:
                out.append(Violation(
                    RULE_HUE_BORDERLINE, s.order, a.person_id,
                    f"距箱边界 {a.pixel.get('hue_class_margin', 0):.1f}°",
                ))
            if a.vlm.get("view_conflict") == 1:
                out.append(Violation(
                    RULE_VIEW_CONFLICT, s.order, a.person_id,
                    f"外层={a.vlm.get('outer_garment_type')}，"
                    f"notes={a.vlm.get('notes', '')}",
                ))
            vlm_error = a.vlm.get("vlm_error")
            if vlm_error:
                out.append(Violation(
                    RULE_VLM_FAILED, s.order, a.person_id,
                    f"VLM 调用失败：{vlm_error}",
                ))

    expected_rows = sum(s.presenter_num for s in segments)
    if len(annotations) != expected_rows:
        out.append(Violation(
            RULE_ROW_COUNT, 0, "",
            f"共 {len(annotations)} 行，传入的 {len(segments)} 个项目要求 {expected_rows} 行",
        ))

    return out
