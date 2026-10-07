import pytest
from pipeline.annotate import PersonAnnotation
from pipeline.validate import Violation, RULE_COUNT, RULE_GENDER, RULE_COLOR_STABILITY
from pipeline.rank import (
    person_confidence, project_confidence, project_experiment_score, rank_projects,
    _diversity,
)


def mk(order=2, seat=3, quality="good", agree=1, borderline=0, stability=5.0, conf=0.9,
       formality=4, suspect_staff=0, sharpness=0.8):
    return PersonAnnotation(
        person_id=f"seg{order:02d}_seat{seat}", seg_order=order, company=f"公司{order}",
        seat_index=seat, presenter_num=2, gender_female=0, speaks_pre=1, speaks_qa=0,
        seg_start=0, seg_end=900,
        vlm={"evidence_quality": quality, "confidence": conf, "formality_1_5": formality,
             "outer_hue_class": "蓝"},
        pixel={"hue_class_agree": agree, "hue_class_borderline": borderline,
               "outer_color_stability": stability, "sharpness_norm": sharpness},
        suspect_staff=suspect_staff,
    )


def test_seat_one_scores_lower_than_seat_three():
    # 座位 1 紧贴镜头、裁切最严重，可见性先验应打折
    assert person_confidence(mk(seat=1)) < person_confidence(mk(seat=3))


def test_poor_evidence_scores_lower():
    assert person_confidence(mk(quality="poor")) < person_confidence(mk(quality="good"))


def test_disagreement_lowers_confidence():
    assert person_confidence(mk(agree=0)) < person_confidence(mk(agree=1))


def test_confidence_is_bounded():
    assert 0.0 <= person_confidence(mk()) <= 1.0
    assert 0.0 <= person_confidence(mk(seat=1, quality="poor", agree=0, borderline=1,
                                       stability=90.0, conf=0.1)) <= 1.0


def test_failed_count_check_zeroes_project():
    assert project_confidence([mk()], count_ok=False, gender_ok=True) == 0.0


def test_failed_gender_check_zeroes_project():
    assert project_confidence([mk()], count_ok=True, gender_ok=False) == 0.0


def test_single_person_project_is_penalised_for_experiment_use():
    solo = project_experiment_score(conf=0.9, n=1, sharpness_norm=0.8, diversity=0.0)
    duo = project_experiment_score(conf=0.9, n=2, sharpness_norm=0.8, diversity=0.0)
    assert solo < duo


def test_internal_diversity_raises_experiment_score():
    flat = project_experiment_score(conf=0.9, n=2, sharpness_norm=0.8, diversity=0.0)
    varied = project_experiment_score(conf=0.9, n=2, sharpness_norm=0.8, diversity=0.5)
    assert varied > flat


def test_rank_projects_returns_all_projects_sorted():
    anns = [mk(order=1, seat=1, quality="poor"), mk(order=1, seat=2, quality="poor"),
            mk(order=2, seat=2, quality="good"), mk(order=2, seat=3, quality="good")]
    out = rank_projects(anns, [])
    assert len(out) == 2
    assert out[0]["order"] == 2, "证据质量好的项目应排前"
    assert out[0]["experiment_rank"] == 1
    assert "top_reason" in out[0]


def test_rank_projects_reports_both_scores():
    anns = [mk(order=1, seat=2), mk(order=1, seat=3)]
    out = rank_projects(anns, [])
    assert "confidence_score" in out[0] and "experiment_score" in out[0]
    assert "confidence_rank" in out[0] and "experiment_rank" in out[0]


# ---------------------------------------------------------------------------
# 修复轮次 1：Critical 1 —— 闸门接线覆盖空洞
# ---------------------------------------------------------------------------

def test_rank_projects_zeroes_project_with_real_count_violation():
    anns = [mk(order=1, seat=2), mk(order=1, seat=3)]
    violations = [Violation(RULE_COUNT, 1, "", "标注 2 人，检出 1 人")]
    out = rank_projects(anns, violations)
    row = next(r for r in out if r["order"] == 1)
    assert row["confidence_score"] == 0.0
    assert row["experiment_score"] == 0.0


def test_rank_projects_zeroes_project_with_real_gender_violation():
    anns = [mk(order=1, seat=2), mk(order=1, seat=3)]
    violations = [Violation(RULE_GENDER, 1, "seg01_seat2", "标注为男，模型判为女")]
    out = rank_projects(anns, violations)
    row = next(r for r in out if r["order"] == 1)
    assert row["confidence_score"] == 0.0
    assert row["experiment_score"] == 0.0


def test_rank_projects_unaffected_by_unrelated_violation():
    anns = [mk(order=1, seat=2), mk(order=1, seat=3)]
    violations = [Violation(RULE_COLOR_STABILITY, 1, "seg01_seat2", "跨帧色相标准差超阈值")]
    out = rank_projects(anns, violations)
    row = next(r for r in out if r["order"] == 1)
    assert row["confidence_score"] > 0.0
    assert row["experiment_score"] > 0.0


# ---------------------------------------------------------------------------
# 修复轮次 1：Critical 2 —— _diversity() 本体零覆盖
# ---------------------------------------------------------------------------

def test_diversity_is_zero_for_identical_formality():
    anns = [mk(order=1, seat=2, formality=4), mk(order=1, seat=3, formality=4)]
    assert _diversity(anns) == 0.0


def test_diversity_grows_with_formality_spread():
    wide = _diversity([mk(order=1, seat=2, formality=1), mk(order=1, seat=3, formality=5)])
    narrow = _diversity([mk(order=1, seat=2, formality=3), mk(order=1, seat=3, formality=4)])
    assert wide > narrow


def test_diversity_is_zero_for_single_person():
    assert _diversity([mk(order=1, seat=2, formality=5)]) == 0.0


def test_diversity_is_capped_at_one():
    div = _diversity([mk(order=1, seat=2, formality=1), mk(order=1, seat=3, formality=5)])
    assert div <= 1.0


def test_diversity_magnitude_matches_known_pstdev():
    # formality 1 vs 5 → pstdev = 2.0 → 2.0 / 2.0 == 1.0（恰好触顶）。
    # 只检查相对顺序（wide > narrow）不足以锁定除数的具体大小——
    # 除数从 2.0 换成 200.0 时相对顺序仍然成立，但绝对量级会被压缩成
    # 近似 0，因此这里额外锁一个具体量级下限，防止除数被悄悄改大。
    div = _diversity([mk(order=1, seat=2, formality=1), mk(order=1, seat=3, formality=5)])
    assert div == pytest.approx(1.0)


def test_rank_projects_uses_real_formality_spread():
    flat_anns = [mk(order=1, seat=2, formality=4), mk(order=1, seat=3, formality=4)]
    varied_anns = [mk(order=2, seat=2, formality=1), mk(order=2, seat=3, formality=5)]
    out = rank_projects(flat_anns + varied_anns, [])
    flat_row = next(r for r in out if r["order"] == 1)
    varied_row = next(r for r in out if r["order"] == 2)
    assert varied_row["diversity"] > flat_row["diversity"]


# ---------------------------------------------------------------------------
# 修复轮次 1：Critical 3 —— diversity 乘性项不可压过测量置信度
# ---------------------------------------------------------------------------

def test_confidence_dominates_diversity():
    # A: 高置信度、零 diversity；B: 中低置信度、满 diversity。
    # 用户优先级：识别准确度 > 样本保留 > 分辨率；diversity 不在其中，
    # 不应能让 conf=0.5 的项目反超 conf=0.9 的项目。
    score_a = project_experiment_score(conf=0.9, n=2, sharpness_norm=0.9, diversity=0.0)
    score_b = project_experiment_score(conf=0.5, n=2, sharpness_norm=0.9, diversity=1.0)
    assert score_a > score_b


# ---------------------------------------------------------------------------
# 修复轮次 1：Important 4 —— suspect_staff 未纳入折扣
# ---------------------------------------------------------------------------

def test_suspect_staff_is_heavily_discounted():
    clean = person_confidence(mk(suspect_staff=0))
    flagged = person_confidence(mk(suspect_staff=1))
    assert flagged > 0.0
    assert flagged / clean <= 0.35


# ---------------------------------------------------------------------------
# 修复轮次 1：top_reason 增强
# ---------------------------------------------------------------------------

def test_top_reason_warns_about_low_confidence_member():
    # order 1 队内一人证据极差（poor/disagree/borderline/低 conf）、一人证据
    # 充分，平均置信度仍高于 order 2（两人证据均中等偏差），故 order 1 拿到
    # experiment_rank == 1；但它队内那个低分成员本身测量置信度低于 _LOW_CONF，
    # top_reason 必须显式警示，而不是把这种「猜出来的差异」包装成卖点。
    weak = mk(order=1, seat=2, quality="poor", agree=0, borderline=1, stability=40.0, conf=0.1)
    strong = mk(order=1, seat=3, quality="good", conf=0.95)
    other = mk(order=2, seat=2, quality="fair", agree=0, borderline=1, stability=40.0, conf=0.3)
    other2 = mk(order=2, seat=3, quality="fair", agree=0, borderline=1, stability=40.0, conf=0.3)
    out = rank_projects([weak, strong, other, other2], [])
    row1 = next(r for r in out if r["order"] == 1)
    assert row1["experiment_rank"] == 1
    assert "⚠" in row1["top_reason"] or "警" in row1["top_reason"]


def test_top_reason_omits_comparison_when_single_project():
    anns = [mk(order=1, seat=2), mk(order=1, seat=3)]
    out = rank_projects(anns, [])
    assert "第二名" not in out[0]["top_reason"]


# ---------------------------------------------------------------------------
# 修复轮次 1：同分 tie-breaker
# ---------------------------------------------------------------------------

def test_equal_scores_are_ordered_by_order_ascending():
    anns = (
        [mk(order=5, seat=2), mk(order=5, seat=3)]
        + [mk(order=1, seat=2), mk(order=1, seat=3)]
    )
    out = rank_projects(anns, [])
    assert out[0]["experiment_score"] == out[1]["experiment_score"]
    assert out[0]["order"] == 1
    assert out[1]["order"] == 5


# ---------------------------------------------------------------------------
# 修复轮次 1：person_confidence 越界输入钳制
# ---------------------------------------------------------------------------

def test_person_confidence_clamped_for_out_of_range_vlm_conf():
    # ask_vlm 只做 json.loads，不校验数值范围；VLM 现实中可能返回越界值。
    out_of_range = mk(conf=1.7)
    assert 0.0 <= person_confidence(out_of_range) <= 1.0


# ---------------------------------------------------------------------------
# 修复轮次 2 · Ruling R43 —— 清晰度杠杆摆幅需与用户优先级（准确度 > 样本
# 保留 > 分辨率）一致，不能是四个杠杆里最大的那个。
# ---------------------------------------------------------------------------

def test_sharpness_lever_is_bounded():
    # sharpness_norm 从 0 拉到 1（其余不变）时，实验分之比不应超过 1.35——
    # 用户把「分辨率」排在优先级最末，清晰度不该是摆幅最大的杠杆。
    # 现行 max(0.1, sharpness_norm) 的摆幅是 10 倍，本条断言下必红。
    lo = project_experiment_score(conf=0.8, n=2, sharpness_norm=0.0, diversity=0.0)
    hi = project_experiment_score(conf=0.8, n=2, sharpness_norm=1.0, diversity=0.0)
    assert hi / lo <= 1.35


def test_sharpness_lever_is_bounded_for_out_of_range_input():
    # 上游 normalize_sharpness 会 clip 到 [0,1]，但那个函数目前还不存在——
    # 只是 task-15-brief 里的一份文档承诺。清晰度是被用户排在最末的因素，
    # 它的摆幅上界不该依赖另一个尚未写出的模块是否记得 clip。
    lo = project_experiment_score(conf=0.8, n=2, sharpness_norm=0.0, diversity=0.0)
    hi = project_experiment_score(conf=0.8, n=2, sharpness_norm=1.0, diversity=0.0)
    assert project_experiment_score(
        conf=0.8, n=2, sharpness_norm=5.0, diversity=0.0) == pytest.approx(hi)
    assert project_experiment_score(
        conf=0.8, n=2, sharpness_norm=-3.0, diversity=0.0) == pytest.approx(lo)


def test_high_confidence_blurry_project_is_not_buried():
    # A：置信度 0.9、清晰度 0.85、无队内差异；D：置信度 0.95（四组里最高）、
    # 清晰度仅 0.40、无队内差异——D 测得更准，只是照片糊。
    # 现行公式 D/A ≈ 0.497，测得最准的项目被清晰度惩罚掉过半分数、掉到第三；
    # 断言 D 至少达到 A 的 0.9 倍，现行实现下必红。
    a = project_experiment_score(conf=0.9, n=2, sharpness_norm=0.85, diversity=0.0)
    d = project_experiment_score(conf=0.95, n=2, sharpness_norm=0.40, diversity=0.0)
    assert d >= 0.9 * a


def test_lever_ordering_matches_user_priority():
    # 核心不变量：把 size / sharpness / diversity 各自从最低拉到最高（其余
    # 固定），三者引起的分数比值必须满足 size 比值 > sharpness 比值 >
    # diversity 比值——直接对应用户优先级「样本保留 > 分辨率」（size 属于
    # 样本保留一侧）以及「分辨率排在 diversity 之前虽未明说，但清晰度不该
    # 压过队内差异这个专门为实验设计服务的信号」。
    # 现行 sharpness 摆幅（10 倍）远超 size（1.43 倍），本条必红。
    size_lo = project_experiment_score(conf=0.5, n=1, sharpness_norm=0.5, diversity=0.5)
    size_hi = project_experiment_score(conf=0.5, n=2, sharpness_norm=0.5, diversity=0.5)
    size_ratio = size_hi / size_lo

    sharp_lo = project_experiment_score(conf=0.5, n=2, sharpness_norm=0.0, diversity=0.5)
    sharp_hi = project_experiment_score(conf=0.5, n=2, sharpness_norm=1.0, diversity=0.5)
    sharp_ratio = sharp_hi / sharp_lo

    div_lo = project_experiment_score(conf=0.5, n=2, sharpness_norm=0.5, diversity=0.0)
    div_hi = project_experiment_score(conf=0.5, n=2, sharpness_norm=0.5, diversity=1.0)
    div_ratio = div_hi / div_lo

    assert size_ratio > sharp_ratio > div_ratio


# ---------------------------------------------------------------------------
# 修复轮次 2 · Ruling R44 —— confidence_rank 与 experiment_rank 分歧时，
# top_reason 必须如实说明，不能只报实验分第一名。
# ---------------------------------------------------------------------------


def test_top_reason_omits_disclosure_when_ranks_agree():
    anns = [mk(order=1, seat=2, quality="poor", conf=0.3),
            mk(order=1, seat=3, quality="poor", conf=0.3),
            mk(order=2, seat=2, quality="good", conf=0.95),
            mk(order=2, seat=3, quality="good", conf=0.95)]
    out = rank_projects(anns, [])
    conf_winner = next(r for r in out if r["confidence_rank"] == 1)
    exp_winner = next(r for r in out if r["experiment_rank"] == 1)
    assert conf_winner["order"] == exp_winner["order"], "本用例需要两个口径一致才有意义"
    assert "测量最准的是" not in exp_winner["top_reason"]
    assert "ℹ" not in exp_winner["top_reason"]



def test_ranking_is_ordered_by_confidence_alone():
    """最终排序必须只看 confidence_score。

    此前主排序键是 experiment_score = 置信度 × 人数 × 清晰度 × 队内差异 的
    加权，权重是拍脑袋定的、无依据，且会与置信度排名冲突——三组重跑时真的
    冲突了（推荐 A 而置信度第一是 B），日志只好并列两个公司名，等于把一个
    没法解释的取舍甩给用户。用户定版：直接按置信度排。
    """
    from pipeline.rank import rank_projects

    # order 1：单人、队内无差异，但证据好、置信高
    # order 2：四人、队内正式度分散、清晰度高——旧公式下实验分会反超
    anns = ([mk(order=1, seat=1, conf=0.95, quality="good", formality=4)]
            + [mk(order=2, seat=i, conf=0.30, quality="fair", formality=f, sharpness=1.0)
               for i, f in enumerate([1, 2, 4, 5], start=1)])
    rows = rank_projects(anns, [])

    assert rows[0]["order"] == 1, "置信度高的项目必须排第一，不该被人数/差异反超"
    scores = [r["confidence_score"] for r in rows]
    assert scores == sorted(scores, reverse=True), "行序必须按置信度单调递减"


def test_top_reason_has_no_self_contradiction():
    """理由里不得再出现「最准的是 A，但推荐 B」这种自相矛盾的并列。

    排序键统一为置信度之后，推荐的就是置信度第一名，那段并列说明失去了
    存在前提；若它还在，说明排序没真正统一。
    """
    from pipeline.rank import rank_projects

    anns = ([mk(order=1, seat=1, conf=0.95)]
            + [mk(order=2, seat=i, conf=0.30, sharpness=1.0, formality=f)
               for i, f in enumerate([1, 2, 4, 5], start=1)])
    rows = rank_projects(anns, [])
    reason = rows[0]["top_reason"]
    assert "但本推荐" not in reason
    assert "综合实验可用性" not in reason
    assert "测量最准的是" not in reason
    # 第二名的名字与置信度仍应给出，供人判断领先幅度
    assert rows[1]["company"] in reason


def test_confidence_winner_is_always_the_recommended_one():
    """置信度第一名与被推荐项目必须恒为同一个——这是排序统一后的不变量。

    取代原先两条锁 Ruling R44「分歧时并列说明」的测试：R44 是为
    experiment_score 主排序打的补丁，排序键换成置信度后分歧在结构上
    不可能发生，那两条测试的前提已消失。这条守住新的不变量，若将来
    有人把主排序键改回加权分，它会立刻变红。
    """
    from pipeline.rank import rank_projects

    anns = ([mk(order=1, seat=1, conf=0.95)]
            + [mk(order=2, seat=i, conf=0.30, sharpness=1.0, formality=f)
               for i, f in enumerate([1, 2, 4, 5], start=1)]
            + [mk(order=3, seat=i, conf=0.70, sharpness=0.9, formality=f)
               for i, f in enumerate([2, 5], start=1)])
    rows = rank_projects(anns, [])
    assert rows[0]["confidence_rank"] == 1, "排第一的必须就是置信度第一名"
    assert rows[0]["top_reason"], "第一名必须带理由"
    assert all(not r["top_reason"] for r in rows[1:]), "其余行不带理由"
