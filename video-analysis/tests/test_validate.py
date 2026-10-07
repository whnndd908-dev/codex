import pytest
from pipeline.schedule import Segment, Presenter
from pipeline.annotate import PersonAnnotation
from pipeline.validate import check_all, Violation
from pipeline import validate


def seg(order=2, n=3):
    ps = tuple(Presenter(k, 0, 1 if k == 1 else 0, 0) for k in range(1, n + 1))
    return Segment(order, "测试公司", 960, 1836, n, ps)


def ann(order=2, seat=1, gender_female=0, gender_judged="男", n=3):
    return PersonAnnotation(
        person_id=f"seg{order:02d}_seat{seat}", seg_order=order, company="测试公司",
        seat_index=seat, presenter_num=n, gender_female=gender_female,
        speaks_pre=1, speaks_qa=0, seg_start=960, seg_end=1836,
        vlm={"gender_judged": gender_judged}, pixel={},
    )


def test_no_violations_when_everything_matches():
    s = seg(n=2)
    anns = [ann(seat=1, n=2), ann(seat=2, n=2)]
    assert check_all(anns, [s]) == []


def test_count_mismatch_is_reported():
    s = seg(n=3)
    anns = [ann(seat=1, n=3), ann(seat=2, n=3)]      # 少一人
    v = check_all(anns, [s])
    assert any(x.rule == "人数闸门" for x in v)


def test_gender_mismatch_is_reported():
    s = seg(n=1)
    anns = [ann(seat=1, gender_female=1, gender_judged="男", n=1)]
    v = check_all(anns, [s])
    assert any(x.rule == "性别闸门" for x in v)


def test_uncertain_gender_is_not_a_violation():
    s = seg(n=1)
    anns = [ann(seat=1, gender_female=1, gender_judged="不确定", n=1)]
    assert not any(x.rule == "性别闸门" for x in check_all(anns, [s]))


def test_duplicate_seat_is_reported():
    s = seg(n=2)
    anns = [ann(seat=1, n=2), ann(seat=1, n=2)]
    v = check_all(anns, [s])
    assert any(x.rule == "座位唯一" for x in v)


def test_row_count_gate_fires_on_orphan_or_missing_rows():
    # 段要求 2 人但只给了 1 行 → 行数闸门必须报警
    s = seg(n=2)
    anns = [ann(seat=1, n=2)]
    v = check_all(anns, [s])
    assert any(x.rule == "行数闸门" for x in v), f"应报行数闸门，实得 {[x.rule for x in v]}"


def test_row_count_gate_silent_when_rows_match_segments():
    # 行数与传入项目的人数之和一致 → 该闸门不应报警
    s = seg(n=2)
    anns = [ann(seat=1, n=2), ann(seat=2, n=2)]
    assert not any(x.rule == "行数闸门" for x in check_all(anns, [s]))


def test_suspect_staff_is_reported():
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.suspect_staff = 1
    assert any(x.rule == "工作人员安全网" for x in check_all([a], [s]))


def test_unstable_colour_is_reported():
    # 跨帧色相标准差超阈值（45.0）→ 该行颜色不可信，须标红
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.pixel = {"outer_color_stability": 60.0}
    v = check_all([a], [s])
    assert any(x.rule == "颜色稳定性" for x in v), f"实得 {[x.rule for x in v]}"


def test_stable_colour_is_not_reported():
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.pixel = {"outer_color_stability": 5.0}
    assert not any(x.rule == "颜色稳定性" for x in check_all([a], [s]))


def test_dual_track_disagreement_is_reported():
    # 像素算法与 VLM 对色相分类不一致 → 交人工裁决
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.pixel = {"hue_class_agree": 0, "hue_class_pixel": "蓝"}
    a.vlm = {"gender_judged": "男", "outer_hue_class": "紫"}
    v = check_all([a], [s])
    assert any(x.rule == "色相分类双轨" for x in v), f"实得 {[x.rule for x in v]}"


def test_dual_track_agreement_is_not_reported():
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.pixel = {"hue_class_agree": 1, "hue_class_pixel": "蓝"}
    assert not any(x.rule == "色相分类双轨" for x in check_all([a], [s]))


def test_borderline_hue_is_reported():
    # 色相距分箱边界过近（< 15°）→ 分类可能因光照噪声翻箱，须标红
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.pixel = {"hue_class_borderline": 1, "hue_class_margin": 5.0}
    v = check_all([a], [s])
    assert any(x.rule == "色相边界样本" for x in v), f"实得 {[x.rule for x in v]}"


# ---------------------------------------------------------------------------
# Task 21 · Ruling R50 —— 进出场画面的外套若与就座画面不一致，须标红交人工裁决。
# ---------------------------------------------------------------------------

def test_view_conflict_raises_a_violation():
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.vlm = {"gender_judged": "男", "view_conflict": 1,
             "outer_garment_type": "西装外套", "notes": "进场穿外套，就座时已脱"}
    v = check_all([a], [s])
    assert any(x.rule == validate.RULE_VIEW_CONFLICT for x in v), f"实得 {[x.rule for x in v]}"


def test_no_view_conflict_no_violation():
    s = seg(n=2)
    a0 = ann(seat=1, n=2)
    a0.vlm = {"gender_judged": "男", "view_conflict": 0}
    a1 = ann(seat=2, n=2)
    a1.vlm = {"gender_judged": "男"}  # 缺失字段（--no-vlm 像素轨）也不能报警
    v = check_all([a0, a1], [s])
    assert not any(x.rule == validate.RULE_VIEW_CONFLICT for x in v), f"实得 {[x.rule for x in v]}"


def test_view_conflict_rule_name_is_a_constant():
    import inspect

    assert hasattr(validate, "RULE_VIEW_CONFLICT")
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.vlm = {"gender_judged": "男", "view_conflict": 1}
    v = check_all([a], [s])
    hit = [x for x in v if x.rule == validate.RULE_VIEW_CONFLICT]
    assert len(hit) == 1

    # 光靠值相等无法区分"引用了常量"和"写死了同文字面量"（CPython 会把
    # 形如标识符的中文字符串自动驻留，is 比较也测不出来）。改用源码检查：
    # 规则值这段文字若在源文件里出现不止一次，说明产出违规处没有引用
    # RULE_VIEW_CONFLICT，而是自己又写了一遍字面量。
    src = inspect.getsource(validate)
    literal = validate.RULE_VIEW_CONFLICT
    assert src.count(literal) == 1, (
        "RULE_VIEW_CONFLICT 的值只应在常量定义处出现一次；"
        "产出违规的代码必须引用常量本身，不能重复写字面量"
    )
    assert src.count("RULE_VIEW_CONFLICT") >= 2, "常量至少要被定义一次、引用一次"


# ---------------------------------------------------------------------------
# 修复轮次 1 · Ruling R50 续 —— view_conflict 改三态：控制器补跑的阳性对照里，
# 进出场画面是逆光剪影、根本无法与就座画面比较，模型被二值 schema 逼着把
# 「无法比较」记成 0（=确认一致），==1 才报警的规则于是把这种情况静默当成
# "已核实一致" 放过。新增 2 = 无法比较，且该值绝不能触发报警（否则又会把
# "拍到剪影" 错报成"外层不一致"），也不能被静默丢弃（要能摊平进 CSV 供人工筛）。
# ---------------------------------------------------------------------------

def test_view_conflict_two_means_incomparable_and_does_not_warn():
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.vlm = {"gender_judged": "男", "view_conflict": 2,
             "notes": "进出场为逆光剪影，无法与就座画面比较外层"}
    v = check_all([a], [s])
    assert not any(x.rule == validate.RULE_VIEW_CONFLICT for x in v), (
        f"view_conflict=2（无法比较）不应报警，实得 {[x.rule for x in v]}"
    )


def test_view_conflict_one_still_warns():
    # 改三态时最容易犯的错是把条件从 ==1 改宽（例如 >=1），顺带把 2 也报了；
    # 或者改反（==2 才报）。这条测试单独锁住 1 这个取值本身仍然报警。
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.vlm = {"gender_judged": "男", "view_conflict": 1}
    v = check_all([a], [s])
    assert any(x.rule == validate.RULE_VIEW_CONFLICT for x in v), (
        f"view_conflict=1（明显不同）仍应报警，实得 {[x.rule for x in v]}"
    )


# ---------------------------------------------------------------------------
# Task 22 · Ruling R54 —— 单人 VLM 调用失败必须可见，画廊要能把该人标红。
# ---------------------------------------------------------------------------

def test_vlm_failure_raises_a_violation():
    s = seg(n=1)
    a = ann(seat=1, n=1)
    a.vlm = {"gender_judged": "男", "evidence_quality": "poor", "confidence": 0.0,
             "vlm_error": "RuntimeError: 503 No available accounts"}
    v = check_all([a], [s])
    assert any(x.rule == validate.RULE_VLM_FAILED for x in v), (
        f"应报 VLM 调用失败，实得 {[x.rule for x in v]}"
    )


def test_no_violation_when_vlm_error_absent():
    # --no-vlm 跑像素轨时 vlm 字典里根本没有 vlm_error 这个键；
    # 也要覆盖"键存在但是空字符串"的情况，二者都不应报警。
    s = seg(n=2)
    a0 = ann(seat=1, n=2)
    a0.vlm = {"gender_judged": "男"}                       # 键缺失
    a1 = ann(seat=2, n=2)
    a1.vlm = {"gender_judged": "男", "vlm_error": ""}        # 键存在但为空
    v = check_all([a0, a1], [s])
    assert not any(x.rule == validate.RULE_VLM_FAILED for x in v), (
        f"vlm_error 缺失或为空不应报警，实得 {[x.rule for x in v]}"
    )
