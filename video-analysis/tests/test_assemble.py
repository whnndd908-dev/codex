import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pipeline import config
from pipeline.detect import Detection
from pipeline.schedule import Segment, Presenter
from pipeline.assemble import (
    normalize_sharpness, measure_person_color, assemble_annotations,
    write_result_csv, write_ranking_csv,
)


# --- sharpness 归一化（Ruling R5 的核心） ---

def test_normalize_sharpness_maps_95th_percentile_to_one():
    vals = list(range(1, 101))          # 1..100，95 分位数是 95.05
    out = normalize_sharpness(vals)
    assert out[-1] == pytest.approx(1.0)     # 100 超过 95 分位，钳到 1.0
    assert 0.0 <= min(out) <= 1.0


def test_normalize_sharpness_is_bounded_everywhere():
    out = normalize_sharpness([0.0, 5.0, 1e6])
    assert all(0.0 <= v <= 1.0 for v in out)


def test_normalize_sharpness_handles_all_zeros():
    # 全零不能除零，应退回中性值
    out = normalize_sharpness([0.0, 0.0, 0.0])
    assert all(0.0 <= v <= 1.0 for v in out)


def test_normalize_sharpness_preserves_order():
    out = normalize_sharpness([3.0, 1.0, 2.0])
    assert out[0] > out[2] > out[1]


def test_normalize_sharpness_empty_list():
    assert normalize_sharpness([]) == []


def test_normalize_sharpness_never_exceeds_one_without_clip_regression():
    """专门锁住钳制逻辑本身（控制器变异清单第4条：去掉 min/max 钳制）。

    构造一个值故意超过 95 分位数很多倍的情况：若钳制被去掉，
    该值会大于 1.0。
    """
    out = normalize_sharpness([1.0] * 19 + [1000.0])
    assert max(out) == pytest.approx(1.0)
    assert all(v <= 1.0 for v in out)


# --- 颜色测量：合成图，答案已知 ---

def _write_solid(tmp_path: Path, name: str, rgb, size=(400, 300)) -> Path:
    """写一张纯色图，并在下半部放一块近中性高亮区作为白平衡参照物。"""
    from PIL import Image
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = rgb
    img[h // 2 + 20:h // 2 + 120, 20:220] = (205, 203, 200)   # 参照物
    p = tmp_path / name
    Image.fromarray(img).save(p)
    return p


@pytest.fixture
def pixel_track_on(monkeypatch):
    """显式开启像素轨。

    像素轨默认关闭（config.PIXEL_COLOR_ENABLED=False，Task 34）——三组视频
    实测双轨一致率仅 33%/41%/43%，新能源2组 30 人全判黑、L* 中位 1.3 已在
    传感器噪底。但**代码是保留的**，用户明确要求「以备之后想执行」，所以
    这些测试必须继续跑，否则保留下来的是一坨没人验证过的死代码——等哪天
    换到光照更好的素材想重新开启，才发现它早就坏了。
    """
    monkeypatch.setattr(config, "PIXEL_COLOR_ENABLED", True)


def test_measure_person_color_returns_all_required_fields(pixel_track_on, tmp_path):
    p = _write_solid(tmp_path, "a.jpg", (20, 30, 90))
    dets = [Detection(2, "seated", 0.0, p, 5, 240.0, 10.0, 390.0, 140.0, 0.9)]
    out = measure_person_color(dets)

    required = {
        "outer_hue_raw", "outer_sat_raw", "outer_val_raw",
        "outer_hue_corr", "outer_sat_corr", "outer_val_corr",
        "outer_lab_L", "outer_lab_a", "outer_lab_b",
        "outer_hue_valid", "outer_color_stability", "n_frames_color",
        "hue_class_pixel", "hue_class_11", "hue_class_6",
        "warm_cool", "lightness_class", "chroma_class",
        "hue_class_margin", "hue_class_borderline",
        "inner_visible", "inner_pixel_count", "inner_quality",
        "sharpness_raw", "wb_ok_ratio",
    }
    missing = required - set(out)
    assert not missing, f"缺字段：{sorted(missing)}"


def _force_full_crop_as_upper_clothes(monkeypatch):
    """让分割模型把整个裁剪都判为上衣。

    真实的 SegFormer 服装分割模型是按人体特征训练的，喂一张无人形特征的
    纯色合成图会判不出任何上衣像素（`test_parsing.py` 里 Task 9 的作者已
    验证过这一点：其真实模型测试标了 `@pytest.mark.slow` 且专用一张来自
    pilot 抽帧的真实截图，合成图测试全部走 monkeypatch）。这里的测试目的
    是验证 `measure_person_color` 的 HSV/Lab 聚合与色相分类链路，不是验证
    分割模型本身的准确率——分割准确率已有 test_parsing.py 专门覆盖，此处
    比照其先例 monkeypatch 掉 `_upper_mask`。
    """
    import numpy as np
    import pipeline.parsing as P
    monkeypatch.setattr(P, "_upper_mask", lambda crop: np.ones(crop.shape[:2], dtype=bool))


def test_measure_person_color_identifies_a_navy_garment(pixel_track_on, tmp_path, monkeypatch):
    _force_full_crop_as_upper_clothes(monkeypatch)
    p = _write_solid(tmp_path, "navy.jpg", (20, 30, 90))
    dets = [Detection(2, "seated", 0.0, p, 5, 240.0, 10.0, 390.0, 140.0, 0.9)]
    out = measure_person_color(dets)
    assert out["hue_class_pixel"] in ("蓝", "紫", "黑"), out["hue_class_pixel"]
    assert out["n_frames_color"] == 1


def test_measure_person_color_flags_dark_garment_hue_as_invalid(pixel_track_on, tmp_path, monkeypatch):
    _force_full_crop_as_upper_clothes(monkeypatch)
    p = _write_solid(tmp_path, "black.jpg", (12, 12, 14))
    dets = [Detection(2, "seated", 0.0, p, 5, 240.0, 10.0, 390.0, 140.0, 0.9)]
    out = measure_person_color(dets)
    assert out["outer_hue_valid"] == 0
    assert out["hue_class_pixel"] == "黑"


def test_measure_person_color_handles_zero_frames(pixel_track_on):
    out = measure_person_color([])
    assert out["n_frames_color"] == 0
    assert out["hue_class_pixel"] == ""


def test_measure_person_color_caps_frame_count(pixel_track_on, tmp_path):
    """需要显式开启像素轨（Task 35 修复）：n_frames_color 现属
    PIXEL_ONLY_FIELDS，像素轨关闭时会被 _finalize_pixel_dict 过滤掉——
    不加 pixel_track_on 这条断言会直接 KeyError，而不是像此前那样悄悄
    通过（关闭状态下该字段固定为 _EMPTY_COLOR 里的 0，从未真正测过"是否
    真的封顶在 6 张"这件事）。
    """
    p = _write_solid(tmp_path, "many.jpg", (20, 30, 90))
    dets = [Detection(2, "seated", float(i), p, 5, 240.0, 10.0, 390.0, 140.0, 0.9)
            for i in range(40)]
    out = measure_person_color(dets, max_frames=6)
    assert out["n_frames_color"] <= 6, "不应对全部帧做分割，太慢"


# --- 装配 ---

def _seg(order=2, n=2):
    ps = tuple(Presenter(k, 0, 1 if k == 1 else 0, 0) for k in range(1, n + 1))
    return Segment(order, f"公司{order}", 960, 1836, n, ps)


def _dets(tmp_path, order=2):
    p = _write_solid(tmp_path, f"f{order}.jpg", (20, 30, 90))
    out = []
    for t in range(3):
        out.append(Detection(order, "seated", float(t), p, 7, 240, 10, 390, 140, 0.9))
        out.append(Detection(order, "seated", float(t), p, 8, 250, 150, 400, 280, 0.9))
    return out


def test_assemble_produces_one_row_per_presenter(tmp_path):
    seg = _seg(n=2)
    dets = _dets(tmp_path)
    seats = {8: 1, 7: 2}
    anns = assemble_annotations([seg], dets, {2: seats}, use_vlm=False)
    assert len(anns) == 2
    assert sorted(a.seat_index for a in anns) == [1, 2]


def test_assemble_carries_ground_truth_from_segment(tmp_path):
    seg = _seg(n=2)
    anns = assemble_annotations([seg], _dets(tmp_path), {2: {8: 1, 7: 2}}, use_vlm=False)
    a1 = next(a for a in anns if a.seat_index == 1)
    assert a1.company == "公司2"
    assert a1.presenter_num == 2
    assert a1.speaks_pre == 1, "座位 1 是主讲人，preprst1 恒为 1"


def test_assemble_fills_sharpness_norm_for_every_person(tmp_path):
    anns = assemble_annotations([_seg(n=2)], _dets(tmp_path), {2: {8: 1, 7: 2}},
                                use_vlm=False)
    for a in anns:
        assert "sharpness_norm" in a.pixel
        assert 0.0 <= a.pixel["sharpness_norm"] <= 1.0


def test_assemble_emits_person_for_missing_seat(tmp_path):
    # 标注 2 人但只分配到 1 个座位：仍须输出 2 行，缺的那行标 poor
    seg = _seg(n=2)
    anns = assemble_annotations([seg], _dets(tmp_path), {2: {8: 1}}, use_vlm=False)
    assert len(anns) == 2, "样本保留优先于分辨率，缺人也要占行"
    missing = next(a for a in anns if a.seat_index == 2)
    assert missing.vlm.get("evidence_quality") == "poor"


def test_segment_gain_returns_neutral_when_no_reference_found():
    from pipeline.assemble import segment_gain
    assert segment_gain([]) == (1.0, 1.0, 1.0)


def test_segment_gain_is_one_value_for_the_whole_segment(tmp_path):
    # 同一段的两帧参照物略有差异，应得到单一中位增益而非逐帧各异
    from pipeline.assemble import segment_gain
    p1 = _write_solid(tmp_path, "g1.jpg", (20, 30, 90))
    p2 = _write_solid(tmp_path, "g2.jpg", (20, 30, 90))
    dets = [Detection(2, "seated", 0.0, p1, 5, 240, 10, 390, 140, 0.9),
            Detection(2, "seated", 1.0, p2, 5, 240, 10, 390, 140, 0.9)]
    g = segment_gain(dets)
    assert len(g) == 3 and all(gg > 0 for gg in g)


# ---------------------------------------------------------------------------
# Task 26 · 需求一 —— 证据图与复核图改为「4 张就座裁剪」，去掉进出场帧。
#
# 用户目视核对交付物后发现：证据图上排的进出场格里出现的是来回走动的场地
# 工作人员，不是创业者（order 2/6/12/13/15 座位1 五个不同项目都拍到了同一个
# 穿深色大衣、背对镜头的人）。控制器排查过全部可用判别信号（末段速度、累计
# 路径长度、轨迹是否持续到窗口结束），没有一个能干净切分——裁定直接不再把
# 进出场帧送入证据图，describe_layout 也相应改为只描述就座帧数，不再接受
# n_trans 参数（旧签名 describe_layout(n_trans, n_seated) 已废弃）。
# ---------------------------------------------------------------------------

def test_describe_layout_says_seated_only_regardless_of_count():
    """无论就座帧多少张，文案都必须说明「不含进出场画面」、下装不可见。"""
    from pipeline.assemble import describe_layout
    s = describe_layout(4)
    assert "进出场" in s          # 要提到，但必须是"不含"
    assert any(kw in s for kw in ("不含", "没有", "无"))
    assert "下装" in s
    assert "上排" not in s and "下排" not in s   # 不许沿用写死的行位描述


@pytest.mark.parametrize("n_seated", [1, 2, 3, 4])
def test_describe_layout_reports_the_actual_seated_count(n_seated):
    """描述里的张数必须与实际组图一致，不许写死成固定数字。"""
    from pipeline.assemble import describe_layout
    s = describe_layout(n_seated)
    assert str(n_seated) in s


def test_describe_layout_never_claims_upper_row_regardless_of_seated_count():
    """控制器变异清单第3条的翻版：即便 n_seated 恰好等于 2，也不许写死上排/下排。"""
    from pipeline.assemble import describe_layout
    s = describe_layout(2)
    assert "上排" not in s and "下排" not in s


def test_describe_layout_states_seated_only():
    """新增测试 3：layout 文本必须明确说清楚——全部为就座镜头、没有进出场
    画面、下装不可见、不要推断下装。这是本次改法的直接验收标准。"""
    from pipeline.assemble import describe_layout
    s = describe_layout(4)
    assert "就座镜头" in s
    assert "进出场" in s and any(kw in s for kw in ("不含", "没有", "无"))
    assert "下装" in s and "不可见" in s
    assert any(kw in s for kw in ("不要推断", "不要臆测", "不要据此推断"))


# ---------------------------------------------------------------------------
# Task 42 —— 项目即将把进出场的站立全身图混入证据图（此前只有就座图）。
# describe_layout 从单参数（只描述就座帧数）扩展为三种构成：纯就座
# （向后兼容，n_entry=n_exit=0）、混合（就座+进出场）、纯进出场
# （n_seated=0，就座画面本次不可用时的兜底）。旧签名/旧调用点
# describe_layout(n_seated) 完全保留（上面几条测试原样通过），这里只补
# 新增的两个分支。
# ---------------------------------------------------------------------------

def test_seated_only_layout_says_no_transition_frames():
    """显式传 n_entry=n_exit=0 与旧的单参数调用等价：无进出场画面，下装
    不可见。"""
    from pipeline.assemble import describe_layout
    s = describe_layout(3, 0, 0)
    assert "进出场" in s and any(kw in s for kw in ("不含", "没有", "无"))
    assert "下装" in s and "不可见" in s


def test_mixed_layout_permits_lower_body_judgement():
    """混合构成必须明确**允许**据站立的进出场镜头判定下装/站姿——这是本次
    改动要解锁的核心行为。旧的一刀切禁止语（"不要据此判定或臆测下装"）
    针对下装本身不得再整句出现，"一律不可见"这种无条件断言也不得出现
    （它只在纯就座分支里成立）。"""
    from pipeline.assemble import describe_layout
    s = describe_layout(2, n_entry=1, n_exit=1)
    assert "下装" in s and "可据此判定" in s
    assert "不要据此判定或臆测下装" not in s
    assert "一律不可见" not in s


def test_mixed_layout_does_not_assert_fixed_camera_angle():
    """混合构成不得断言"摄像机位于该人左后侧方"这种适用于全组图的固定
    机位——进场镜头多背向、退场镜头多正向，必须明确提示机位随镜头变化，
    不要假设固定方位。"""
    from pipeline.assemble import describe_layout
    s = describe_layout(2, n_entry=1, n_exit=1)
    assert "摄像机位于该人左后侧方" not in s
    assert any(kw in s for kw in ("不要假设", "随镜头变化"))
    assert "背向" in s and "正向" in s


def test_transition_only_layout_flags_seated_unavailable():
    """就座画面本次不可用（n_seated=0）时，必须明确提示这一点，而不是
    悄悄留空让模型去猜/等一张并不存在的就座证据；同时下装/站姿因站立
    镜头而可判定。"""
    from pipeline.assemble import describe_layout
    s = describe_layout(0, n_entry=2, n_exit=2)
    assert any(kw in s for kw in ("不可用", "无法使用", "没有可用"))
    assert "就座" in s
    assert "下装" in s and "可据此判定" in s


def test_assemble_person_ids_are_unique(tmp_path):
    anns = assemble_annotations([_seg(n=2)], _dets(tmp_path), {2: {8: 1, 7: 2}},
                                use_vlm=False)
    ids = [a.person_id for a in anns]
    assert len(set(ids)) == len(ids)


def test_person_id_format_is_closed(tmp_path):
    """person_id 必须只由数字组成，绝不能混入企业名等自由文本。

    这不是洁癖，是安全不变量（Ruling R37）：画廊模板里
    `onclick="save('${r.person_id}',...)"` 把该值拼进 JS 字符串字面量，
    而 esc() 只做 HTML 实体编码——浏览器在把属性值当 JS 解析**之前**
    会先做实体解码，编码被撤销。含引号的 person_id 即可逃逸出字符串执行任意脚本。
    企业名来自 Excel，是本项目唯一的不可信自由文本来源；只要它永远不进
    person_id，这条注入路径就打不开。本测试就是钉死这个不变量的那颗钉子。
    """
    import re
    from dataclasses import replace

    # Segment 是 frozen dataclass，只能用 replace 造变体
    seg = replace(_seg(n=2), company='A & B <script>"x"</script>')
    anns = assemble_annotations([seg], _dets(tmp_path), {2: {8: 1, 7: 2}},
                                use_vlm=False)
    for a in anns:
        assert re.fullmatch(r"seg\d+_seat\d+", a.person_id), a.person_id
    # 企业名本身要照常保留在它自己的字段里，不能被这条约束顺手弄丢
    assert anns[0].company == seg.company


# --- 进出场帧的来源（Ruling R18，必须锁死） ---

def test_entry_frame_is_empty_without_pairing(tmp_path):
    """没有配对结果时宁可不给进出场帧，也不许按 track_id 猜。

    entry 窗口里恰好存在一个与就座 track_id 同号的轨迹——那是**另一个人**
    （跟踪器每个 (seg_order, kind) 从 1 重新编号）。若实现退回按 tid 筛，
    这条会拿到非空路径而失败。
    """
    dets = _dets(tmp_path)
    p = _write_solid(tmp_path, "entry_other.jpg", (200, 30, 30))
    dets.append(Detection(2, "entry", -5.0, p, 8, 240, 10, 390, 140, 0.9))
    anns = assemble_annotations([_seg(n=1)], dets, {2: {8: 1}}, use_vlm=False)
    assert anns[0].best_frame_entry == ""


def test_entry_frame_comes_from_pairing_result(tmp_path):
    """配对结果指向 entry track 3，就必须取 3 号的帧，不是同号的 8 号。"""
    dets = _dets(tmp_path)
    wrong = _write_solid(tmp_path, "entry_wrong.jpg", (200, 30, 30))
    right = _write_solid(tmp_path, "entry_right.jpg", (30, 200, 30))
    dets.append(Detection(2, "entry", -5.0, wrong, 8, 240, 10, 390, 140, 0.9))
    dets.append(Detection(2, "entry", -5.0, right, 3, 240, 10, 390, 140, 0.9))
    pairs = {2: {1: {"entry_track": 3, "exit_track": None}}}
    anns = assemble_annotations([_seg(n=1)], dets, {2: {8: 1}},
                                use_vlm=False, pairs_by_order=pairs)
    assert anns[0].best_frame_entry == str(right)


def test_entry_frame_without_pairing_ignores_same_numbered_other_person(tmp_path):
    """控制器变异清单第1条的翻版：就座 tid 与 entry 里另一个人的 tid 同号时，
    即便 pairs_by_order 提供了结果，也必须严格按配对指定的 track 取帧，
    绝不能因为号码相同就顺手拿就座的 tid 去筛 entry。
    """
    dets = _dets(tmp_path)   # 就座座位 1 的 tid=8
    same_num_other_person = _write_solid(tmp_path, "entry_same_tid.jpg", (200, 30, 30))
    actual_entry_person = _write_solid(tmp_path, "entry_actual.jpg", (30, 200, 30))
    # entry 窗口里 tid=8 是另一个人（entry 与 seated 的编号空间互相独立）
    dets.append(Detection(2, "entry", -5.0, same_num_other_person, 8, 240, 10, 390, 140, 0.9))
    # 真正配对到座位 1 的 entry track 是 3 号
    dets.append(Detection(2, "entry", -5.0, actual_entry_person, 3, 240, 10, 390, 140, 0.9))
    pairs = {2: {1: {"entry_track": 3, "exit_track": None}}}
    anns = assemble_annotations([_seg(n=1)], dets, {2: {8: 1}},
                                use_vlm=False, pairs_by_order=pairs)
    assert anns[0].best_frame_entry == str(actual_entry_person)
    assert anns[0].best_frame_entry != str(same_num_other_person)


def test_pairs_by_order_none_leaves_transitions_empty_not_guessed(tmp_path):
    """控制器变异清单第2条：pairs_by_order=None 时必须留空，不许退回按 tid 猜。

    构造一个 entry 轨迹的 track_id 与就座 tid 完全相同、且颜色差异极大
    （若按 tid 猜会一猜就中），验证在 pairs_by_order=None 时依旧不取用它。
    """
    dets = _dets(tmp_path)
    decoy = _write_solid(tmp_path, "decoy_same_tid.jpg", (250, 5, 5))
    dets.append(Detection(2, "entry", -5.0, decoy, 8, 240, 10, 390, 140, 0.9))
    anns = assemble_annotations([_seg(n=1)], dets, {2: {8: 1}},
                                use_vlm=False, pairs_by_order=None)
    assert anns[0].best_frame_entry == ""


def test_staff_ids_set_flag_but_keep_the_row(tmp_path):
    """标记不等于删除：被标的人仍要出现在结果里（画廊可一键恢复）。"""
    anns = assemble_annotations([_seg(n=2)], _dets(tmp_path), {2: {8: 1, 7: 2}},
                                use_vlm=False, staff_ids={"seg02_seat2"})
    assert len(anns) == 2
    flags = {a.person_id: a.suspect_staff for a in anns}
    assert flags["seg02_seat2"] == 1 and flags["seg02_seat1"] == 0


# --- 产出 ---

def test_write_result_csv_has_header_and_one_row_per_person(tmp_path):
    anns = assemble_annotations([_seg(n=2)], _dets(tmp_path), {2: {8: 1, 7: 2}},
                                use_vlm=False)
    out = tmp_path / "result.csv"
    write_result_csv(anns, out)
    lines = out.read_text(encoding="utf-8-sig").strip().splitlines()
    assert len(lines) == 3            # 表头 + 2 行
    assert "company" in lines[0]
    assert "hue_class_best" in lines[0]


def test_write_ranking_csv_roundtrip(tmp_path):
    rows = [{"order": 2, "company": "公司2", "n_presenters": 2,
             "confidence_score": 0.8, "experiment_score": 0.6,
             "sharpness_norm": 0.7, "diversity": 0.1,
             "confidence_rank": 1, "experiment_rank": 1, "top_reason": "测试"}]
    out = tmp_path / "ranking.csv"
    write_ranking_csv(rows, out)
    text = out.read_text(encoding="utf-8-sig")
    assert "experiment_rank" in text and "公司2" in text


# --- person_id 里不能混入企业名（控制器变异清单第5条的直接锁定） ---

def test_person_id_never_contains_company_substring(tmp_path):
    seg = _seg(order=2, n=1)
    anns = assemble_annotations([seg], _dets(tmp_path), {2: {8: 1}}, use_vlm=False)
    assert "公司" not in anns[0].person_id


# --- 缺座位样本保留（控制器变异清单第7条的直接锁定） ---

def test_missing_seat_row_still_has_correct_ground_truth(tmp_path):
    """缺座位的行不能只是"占了一行"，company/presenter_num/gender 等
    ground truth 字段也必须正确携带，供人工复核定位。
    """
    seg = _seg(n=2)
    anns = assemble_annotations([seg], _dets(tmp_path), {2: {8: 1}}, use_vlm=False)
    missing = next(a for a in anns if a.seat_index == 2)
    assert missing.company == "公司2"
    assert missing.presenter_num == 2
    assert missing.person_id == "seg02_seat2"


# ---------------------------------------------------------------------------
# Task 44：进出场递补——剔除最里侧专家后从进出场候选补回一位真人。
# entry_labels=None 时行为完全不变是硬要求（brief 变异清单 (a)）。
# ---------------------------------------------------------------------------

def _entry_dets(tmp_path, order=2, tid=50, kind="entry", cx=900.0, n=6,
                 h=500.0, w=180.0, bottom=1200.0, name=None):
    """构造一条进出场候选轨迹：cx=900 落在 RIGHT_REGION_X(0.55*1440=792)
    右侧，h/w=500/180≈2.78 远高于 config.ENTRYEXIT_MIN_ASPECT=1.30，确保
    这条轨迹能穿过 entryexit.entryexit_candidates 的右区/长宽比过滤，成为
    真正可递补的候选（不是摆设的夹具，见 brief 反复强调的「夹具是否真的
    走到了被测路径」）。图开到 1300x1300，保证这个 bbox 落在图像范围内，
    measure_person_color/build_evidence_sheet 才能正常裁出非空图块。
    """
    from PIL import Image
    if name is None:
        name = f"entry_{order}_{tid}.jpg"
    p = tmp_path / name
    if not p.exists():
        img = np.zeros((1300, 1300, 3), dtype=np.uint8)
        img[:, :] = (20, 30, 90)
        img[600:700, 20:220] = (205, 203, 200)   # 白平衡参照物
        Image.fromarray(img).save(p)
    half = w / 2.0
    return [
        Detection(order, kind, float(t), p, tid, cx - half, bottom - h,
                  cx + half, bottom, 0.9)
        for t in range(n)
    ]


def test_assemble_unchanged_when_no_entry_labels(tmp_path):
    """entry_labels 缺省 / 显式传 None 两种写法结果必须逐字段相同，且新增
    的三个标志字段一律停在中性默认值——这是最重要的护栏（brief 明确要求）。
    """
    seg = _seg(n=2)
    dets = _dets(tmp_path)
    seats = {2: {8: 1, 7: 2}}

    default = assemble_annotations([seg], dets, seats, use_vlm=False)
    explicit_none = assemble_annotations([seg], dets, seats, use_vlm=False,
                                         entry_labels=None)

    assert [dataclasses.asdict(a) for a in default] == \
           [dataclasses.asdict(a) for a in explicit_none]
    for a in default:
        assert a.source == "seated"
        assert a.seat_uncertain == 0
        assert a.expert_removed == 0


def test_backfilled_row_carries_source_flag(tmp_path):
    seg = _seg(n=2)
    dets = _dets(tmp_path) + _entry_dets(tmp_path, order=2, tid=50)
    labels = {"2": {"entry:50": {"role": "创业者", "seat": 2}}}

    # 只喂座位1（tid=8）——座位2 在就座阶段本就是空的，才有得递补。
    anns = assemble_annotations([seg], dets, {2: {8: 1}}, use_vlm=False,
                                entry_labels=labels)
    by_seat = {a.seat_index: a for a in anns}
    assert by_seat[1].source == "seated", "就座阶段来的行必须记 source='seated'"
    assert by_seat[2].source == "entryexit", "递补行必须记 source='entryexit'"


def test_backfilled_row_reaches_csv_with_all_three_flags(tmp_path):
    from pipeline.assemble import write_result_csv
    import csv as csv_mod

    seg = _seg(n=2)
    dets = _dets(tmp_path) + _entry_dets(tmp_path, order=2, tid=51)
    # Task 45 追加改动四起，未指定座位号的候选默认不参与递补（改进
    # needs_seat），本测试关心的是三个标志字段能否落到 CSV，须显式指定
    # 座位号才能走到补位路径——座位号猜测那条已被证伪并默认关闭。
    labels = {"2": {"entry:51": {"role": "创业者", "seat": 2}}}

    anns = assemble_annotations([seg], dets, {2: {8: 1}}, use_vlm=False,
                                entry_labels=labels)
    out = tmp_path / "result.csv"
    write_result_csv(anns, out)
    with out.open(encoding="utf-8-sig") as fh:
        rows = {int(r["seat_index"]): r for r in csv_mod.DictReader(fh)}

    for col in ("source", "seat_uncertain", "expert_removed"):
        assert col in rows[2], f"{col} 未落到 CSV"
    assert rows[2]["source"] == "entryexit"
    assert rows[2]["seat_uncertain"] == "0", (
        "用户在复核页明确指定了座位号（seat=2），fill_plan 不应标 seat_uncertain"
    )


def test_backfill_uses_transition_layout_not_seated(tmp_path):
    """递补行的证据图构成必须走 describe_layout 的『纯进出场』分支——
    该分支明确宣称下装/站姿可见；绝不能沿用第 1 步就座候选人用的『纯就座』
    分支，那条分支会告诉模型下装不可见，而递补候选恰恰是下装可见的站立
    镜头（brief 变异清单 (d)，Task 42 报告详述过这个缺陷）。
    """
    seg = _seg(n=2)
    dets = _dets(tmp_path) + _entry_dets(tmp_path, order=2, tid=52)
    labels = {"2": {"entry:52": {"role": "创业者", "seat": 2}}}

    captured = []

    def fake(sheet_png, context):
        captured.append(context)
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}

    anns = _run_with_fake_ask_vlm(fake, [seg], dets, {2: {8: 1}}, entry_labels=labels)

    backfill_ctx = next(c for c in captured if "_entryexit" in c.get("person_id", ""))
    layout = backfill_ctx["layout"]
    assert "可见" in layout, f"递补 layout 应声明下装可见，实得：{layout}"
    assert "不可见" not in layout, (
        f"递补 layout 绝不能出现『不可见』——那是就座分支专属的措辞，实得：{layout}"
    )

    seated_ctx = next(c for c in captured if "_entryexit" not in c.get("person_id", ""))
    assert "不可见" in seated_ctx["layout"], "对照组：就座候选人的 layout 仍应是原有的纯就座措辞"


def test_backfill_skips_needs_resplit_candidates(tmp_path):
    """候选标了 n_people>1（一个框混了不止一个人）时不得参与递补——
    Ruling R14 的教训：混了两人的框拿去编码衣着会安静地给出两人混合的
    结果，不报错、数字全错。"""
    seg = _seg(n=2)
    dets = _dets(tmp_path) + _entry_dets(tmp_path, order=2, tid=53)
    labels = {"2": {"entry:53": {"role": "创业者", "seat": 2, "n_people": 2}}}

    anns = assemble_annotations([seg], dets, {2: {8: 1}}, use_vlm=False,
                                entry_labels=labels)
    by_seat = {a.seat_index: a for a in anns}
    assert by_seat[2].source == "seated", "混了多人的候选不应被用来递补"
    assert by_seat[2].vlm.get("evidence_quality") == "poor", (
        "座位2仍应是缺座位占位行，因为唯一的候选被 needs_resplit 挡在外面"
    )


def test_backfill_does_not_overwrite_seated_seats(tmp_path):
    """已有真实就座结果的座位号绝不能被递补候选顶替（brief 变异清单 (f)）。"""
    seg = _seg(n=2)
    dets = _dets(tmp_path) + _entry_dets(tmp_path, order=2, tid=54)
    # 用户（误）把座位号填成已被就座阶段占用的 1 号
    labels = {"2": {"entry:54": {"role": "创业者", "seat": 1}}}

    anns = assemble_annotations([seg], dets, {2: {8: 1, 7: 2}}, use_vlm=False,
                                entry_labels=labels)
    by_seat = {a.seat_index: a for a in anns}
    assert by_seat[1].source == "seated"
    assert by_seat[1].pixel.get("track_id") == 8, (
        "座位1本就有真实就座结果（tid=8），不该被递补候选（tid=54）顶替"
    )


# --- 按企业/座位导出关键帧（Ruling R13） ---

def test_export_keyframes_creates_per_company_per_seat_dirs(tmp_path):
    anns = assemble_annotations([_seg(n=2)], _dets(tmp_path), {2: {8: 1, 7: 2}},
                                use_vlm=False)
    from pipeline.assemble import export_keyframes
    out = tmp_path / "keyframes"
    export_keyframes(anns, _dets(tmp_path), out)
    dirs = sorted(p.relative_to(out).as_posix() for p in out.glob("*/*"))
    assert dirs == ["02_公司2/seat1", "02_公司2/seat2"]


def test_export_keyframes_writes_context_frame_with_box(tmp_path):
    anns = assemble_annotations([_seg(n=1)], _dets(tmp_path), {2: {8: 1}},
                                use_vlm=False)
    from pipeline.assemble import export_keyframes
    out = tmp_path / "keyframes"
    export_keyframes(anns, _dets(tmp_path), out)
    ctx = list(out.glob("*/seat1/context_full_frame.jpg"))
    assert len(ctx) == 1
    from PIL import Image
    with Image.open(ctx[0]) as im:
        assert im.size[0] > 0 and im.size[1] > 0


def test_export_keyframes_sanitises_company_names(tmp_path):
    seg = Segment(3, "某公司/含:非法*字符", 960, 1836, 1,
                  (Presenter(1, 0, 1, 0),))
    dets = [Detection(3, "seated", float(t), _write_solid(tmp_path, "s.jpg", (20, 30, 90)),
                      9, 240, 10, 390, 140, 0.9) for t in range(3)]
    anns = assemble_annotations([seg], dets, {3: {9: 1}}, use_vlm=False)
    from pipeline.assemble import export_keyframes
    out = tmp_path / "keyframes"
    export_keyframes(anns, dets, out)

    # 必须恰好一层企业目录：out/<企业目录>/seat1/...。若净化被去掉，公司名里的
    # "/" 会被 pathlib 当成路径分隔符自行拆出多层子目录（out.iterdir() 只看
    # 得到最外层，那一层天然不含 "/"，看不出破绽），glob("*/seat1") 要求
    # seat1 恰好在一层通配符目录之下，能揭穿这种意外的多层嵌套。
    seat_dirs = list(out.glob("*/seat1"))
    assert len(seat_dirs) == 1, (
        f"应恰好一层企业目录，实得 {[str(p.relative_to(out)) for p in out.rglob('seat1')]}"
    )
    company_dir_name = seat_dirs[0].parent.name
    assert "/" not in company_dir_name
    assert ":" not in company_dir_name
    assert "*" not in company_dir_name


def test_evidence_png_bytes_never_reach_csv(tmp_path):
    anns = assemble_annotations([_seg(n=1)], _dets(tmp_path), {2: {8: 1}},
                                use_vlm=False)
    out = tmp_path / "result.csv"
    write_result_csv(anns, out)
    header = out.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    # 逐列名精确比对，而非整份文本子串搜索：pytest 的 tmp_path 目录名本身
    # 是从测试函数名派生的（这里恰好含有 "_evidence_png" 子串，因为函数名是
    # test_evidence_png_bytes_never_reach_csv），若某字段存了含 tmp_path 的
    # 完整文件路径（如 best_frame_seated），子串搜索会把这个巧合误判成真的
    # 字段泄漏。逐列名比对不受这个巧合影响，仍然精确锁住"该键不得成为一列"。
    assert "_evidence_png" not in header, "二进制证据图不得进入 CSV"


def test_pixel_dict_does_not_retain_evidence_png_key_after_export(tmp_path):
    """控制器变异清单第6条：写盘后必须从 a.pixel 中删除 _evidence_png。

    直接跑一遍 use_vlm=True 路径（用 monkeypatch 假装 VLM 调用），
    确认写出关键帧之后 pixel 里不再残留该键。
    """
    import pipeline.assemble as assemble_mod

    def _fake_ask_vlm(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9,
                "outer_hue_class": "蓝", "formality_1_5": 4}

    orig = assemble_mod.ask_vlm
    assemble_mod.ask_vlm = _fake_ask_vlm
    try:
        anns = assemble_mod.assemble_annotations(
            [_seg(n=1)], _dets(tmp_path), {2: {8: 1}}, use_vlm=True,
        )
    finally:
        assemble_mod.ask_vlm = orig

    assert "_evidence_png" in anns[0].pixel, "写盘前应暂存证据图字节供 export_keyframes 使用"
    out = tmp_path / "keyframes"
    from pipeline.assemble import export_keyframes
    export_keyframes(anns, _dets(tmp_path), out)
    assert "_evidence_png" not in anns[0].pixel, "写盘后必须从 pixel 中删除该键"


# ---------------------------------------------------------------------------
# Task 22 · Ruling R54 —— 单人 VLM 调用失败不得拖垮整批 34 人的连续调用。
# 中转服务此前实测连续 3 次 503 No available accounts（Ruling R28）；
# 阶段 5 要等 assemble_annotations 全部返回后才落盘 annotations.json，
# 任一次调用抛异常都会让前面已成功的调用全部丢失，必须从头再跑。
# ---------------------------------------------------------------------------

def _dets3(tmp_path, order=2):
    """3 人一段，座位对应 track_id 7/8/9。"""
    p = _write_solid(tmp_path, f"f3_{order}.jpg", (20, 30, 90))
    out = []
    for t in range(3):
        for tid in (7, 8, 9):
            out.append(Detection(order, "seated", float(t), p, tid, 240, 10, 390, 140, 0.9))
    return out


def _make_failing_ask_vlm(fail_on_call_index: int, exc: Exception | None = None):
    """第 fail_on_call_index 次调用抛异常，其余正常返回。"""
    calls = {"n": 0}

    def fake(sheet_png, context):
        calls["n"] += 1
        if calls["n"] == fail_on_call_index:
            raise exc if exc is not None else RuntimeError("503 No available accounts")
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}

    return fake


def _run_with_fake_ask_vlm(fake, segs, dets, seats, **kwargs):
    import pipeline.assemble as assemble_mod
    orig = assemble_mod.ask_vlm
    assemble_mod.ask_vlm = fake
    try:
        return assemble_mod.assemble_annotations(segs, dets, seats, use_vlm=True, **kwargs)
    finally:
        assemble_mod.ask_vlm = orig


def test_vlm_failure_does_not_abort_the_batch(tmp_path):
    seg = _seg(order=2, n=3)
    dets = _dets3(tmp_path)
    seats = {7: 1, 8: 2, 9: 3}
    fake = _make_failing_ask_vlm(fail_on_call_index=2)   # 第 2 个人（座位2）失败

    anns = _run_with_fake_ask_vlm(fake, [seg], dets, {2: seats})

    assert len(anns) == 3, "任一人失败都不能让其余人丢失整批"
    by_seat = {a.seat_index: a for a in anns}
    assert "vlm_error" not in by_seat[1].vlm
    assert by_seat[1].vlm.get("outer_hue_class") == "蓝"
    assert "vlm_error" not in by_seat[3].vlm
    assert by_seat[3].vlm.get("outer_hue_class") == "蓝"
    assert by_seat[2].vlm.get("vlm_error"), "座位2应记到失败态"


def test_vlm_failure_is_recorded_visibly(tmp_path):
    seg = _seg(order=2, n=3)
    dets = _dets3(tmp_path)
    seats = {7: 1, 8: 2, 9: 3}
    fake = _make_failing_ask_vlm(fail_on_call_index=2)

    anns = _run_with_fake_ask_vlm(fake, [seg], dets, {2: seats})
    failed = next(a for a in anns if a.seat_index == 2)

    assert failed.vlm.get("vlm_error"), "失败态必须是显式的非空字符串，不是空字典"
    assert failed.vlm.get("evidence_quality") == "poor"
    assert failed.vlm.get("confidence") == 0.0


def test_vlm_failure_marks_dual_track_as_disagreeing(tmp_path, monkeypatch):
    # 像素轨测试：hue_class_agree 依赖像素轨（Task 34 默认关闭），显式打开
    # 才能验证这段仍保留、可用的逻辑。
    monkeypatch.setattr(config, "PIXEL_COLOR_ENABLED", True)
    seg = _seg(order=2, n=3)
    dets = _dets3(tmp_path)
    seats = {7: 1, 8: 2, 9: 3}
    fake = _make_failing_ask_vlm(fail_on_call_index=2)

    anns = _run_with_fake_ask_vlm(fake, [seg], dets, {2: seats})
    failed = next(a for a in anns if a.seat_index == 2)

    # use_vlm=False 是刻意不调用，记 1 合理；这里是调用失败、没有第二条轨
    # 可对，记 0 才诚实——绝不能沿用 use_vlm=False 分支的 1。
    assert failed.pixel["hue_class_agree"] == 0
    assert failed.pixel["hue_class_final"] == failed.pixel.get("hue_class_pixel", "")


def test_vlm_error_is_truncated_and_has_no_traceback(tmp_path):
    seg = _seg(order=2, n=1)
    dets = _dets(tmp_path)
    long_msg = "connection reset by relay host " * 20   # 远超 200 字符
    full_untruncated = f"RuntimeError: {long_msg}"

    fake = _make_failing_ask_vlm(fail_on_call_index=1, exc=RuntimeError(long_msg))
    anns = _run_with_fake_ask_vlm(fake, [seg], dets, {2: {8: 1}})

    err = anns[0].vlm["vlm_error"]
    assert len(err) < len(full_untruncated), "超长异常信息必须被截断"
    assert "Traceback" not in err, "绝不能把 traceback 写进去"
    assert "http" not in err, "绝不能把请求 URL/响应体原样糊进去"


# ---------------------------------------------------------------------------
# Task 23 · 缺陷二 —— 调用失败时证据图不能丢。恰恰是失败的人最需要留下
# evidence_sent_to_model.png 供诊断；控制器复现失败案例时发现图不存在，
# 只能自己重建。_evidence_png 必须在 ask_vlm(...) 调用之前、且在
# try/except 之外暂存，失败分支绝不能把它清掉，export_keyframes 也不能
# 因为某人失败就跳过写图。
# ---------------------------------------------------------------------------

def test_evidence_png_is_staged_before_vlm_call(tmp_path):
    seg = _seg(order=2, n=1)
    dets = _dets(tmp_path)
    fake = _make_failing_ask_vlm(fail_on_call_index=1)

    anns = _run_with_fake_ask_vlm(fake, [seg], dets, {2: {8: 1}})

    assert anns[0].vlm.get("vlm_error"), "前提：这次调用确实失败了"
    assert anns[0].pixel.get("_evidence_png"), (
        "调用失败时也必须留存证据图字节，供人工诊断当时送去的究竟是什么"
    )


def test_evidence_png_survives_vlm_failure_into_keyframes(tmp_path):
    seg = _seg(order=2, n=1)
    dets = _dets(tmp_path)
    fake = _make_failing_ask_vlm(fail_on_call_index=1)

    anns = _run_with_fake_ask_vlm(fake, [seg], dets, {2: {8: 1}})
    assert anns[0].vlm.get("vlm_error"), "前提：这次调用确实失败了"

    out = tmp_path / "keyframes"
    from pipeline.assemble import export_keyframes
    export_keyframes(anns, dets, out)

    matches = list(out.glob("*/seat1/evidence_sent_to_model.png"))
    assert len(matches) == 1, "失败者的目录里也必须导出证据图"
    assert matches[0].stat().st_size > 0, "证据图不能是空文件"


def test_vlm_failure_summary_is_printed(tmp_path, capsys):
    """人只顾单人失败态不够：批跑完必须打印一行汇总，人工才知道有多少人
    要靠 --stages 5 补跑（控制器变异清单第7条：去掉这行打印必须变红）。
    """
    seg = _seg(order=2, n=3)
    dets = _dets3(tmp_path)
    seats = {7: 1, 8: 2, 9: 3}
    fake = _make_failing_ask_vlm(fail_on_call_index=2)

    _run_with_fake_ask_vlm(fake, [seg], dets, {2: seats})
    out = capsys.readouterr().out
    assert "VLM 调用失败 1/3 人" in out, f"未打印汇总行，实得 stdout:\n{out}"
    assert "vlm_error" in out, "汇总行应指向 result.csv 的 vlm_error 列"


def test_vlm_error_reaches_csv(tmp_path):
    seg = _seg(order=2, n=3)
    dets = _dets3(tmp_path)
    seats = {7: 1, 8: 2, 9: 3}
    fake = _make_failing_ask_vlm(fail_on_call_index=2)

    anns = _run_with_fake_ask_vlm(fake, [seg], dets, {2: seats})
    out = tmp_path / "result.csv"
    write_result_csv(anns, out)

    import csv as csv_mod
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv_mod.DictReader(fh))
    assert "vlm_error" in rows[0].keys(), "CSV 必须有 vlm_error 列，人工复核才看得见"
    failed_row = next(r for r in rows if r["seat_index"] == "2")
    assert failed_row["vlm_error"], "失败者那一行 vlm_error 不能是空的"


# ---------------------------------------------------------------------------
# Task 25 · 缺陷一 —— 画廊缩略图几乎不含创业者：best_frame_seated 存的是
# 1440x1080 整帧，人只占最右一小条。证据图（build_evidence_sheet 生成的
# 2x2 红框图）才是给人工复核用的正确素材。裁定：无论 use_vlm 与否都要构建
# 证据图并存入 a.pixel["_evidence_png"]（写盘前必须弹出，不能进 JSON/CSV）。
# ---------------------------------------------------------------------------

def test_evidence_sheet_is_built_even_without_vlm(tmp_path):
    """现行实现下必红：旧代码只在 use_vlm=True 分支才 build_evidence_sheet。"""
    anns = assemble_annotations([_seg(n=2)], _dets(tmp_path), {2: {8: 1, 7: 2}},
                                use_vlm=False)
    seated_people = [a for a in anns if a.best_frame_seated]
    assert seated_people, "前提：至少有人分到了座位帧"
    for a in seated_people:
        sheet = a.pixel.get("_evidence_png")
        assert sheet, f"{a.person_id} 缺少证据图（use_vlm=False 时也应构建）"
        assert sheet[:8] == b"\x89PNG\r\n\x1a\n", "必须是合法 PNG 字节"


def test_evidence_png_still_never_reaches_csv(tmp_path):
    """护栏：为修「画廊缩略图用证据图」而误把二进制字节留进 CSV/JSON
    是控制器变异清单第 (d) 条明确点名的事故。use_vlm=False 也要测，因为
    这条路径此前根本不产生该键，护栏此前是"trivially green"。"""
    anns = assemble_annotations([_seg(n=1)], _dets(tmp_path), {2: {8: 1}},
                                use_vlm=False)
    assert anns[0].pixel.get("_evidence_png"), "前提：这一行确实带着证据图字节"

    out = tmp_path / "result.csv"
    write_result_csv(anns, out)
    header = out.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert "_evidence_png" not in header, "二进制证据图不得进入 CSV"

    import json
    from pipeline.gallery import annotation_to_row
    row = annotation_to_row(anns[0])
    assert "_evidence_png" not in row, "也不得进入展平后的行（JSON 序列化前必弹出）"
    json.dumps(row, ensure_ascii=False)  # 不应抛 TypeError


# ---------------------------------------------------------------------------
# Task 26 · 需求一（新增测试 1、2）—— 证据图只用就座帧，不再混入进出场帧。
#
# 用户点名了五个座位（order 2/6/12/13/15 的 seat1），控制器目视确认同一个
# 穿深色大衣、背对镜头的场地工作人员出现在这五个不同项目的进场格里。控制器
# 排查过全部可用判别信号（末段速度、累计路径长度、轨迹是否持续到窗口结
# 束），没有一个能干净切分——裁定不再调阈值，直接把进出场帧从证据图中移除。
# ---------------------------------------------------------------------------

def test_evidence_sheet_uses_only_seated_frames(tmp_path):
    """即便有进出场配对结果，证据图也只能用就座帧。

    现行实现下必红（改前）：旧代码 `ev = best_trans + best_seated`，会把
    entry_p 也送进 build_evidence_sheet。
    """
    import pipeline.assemble as assemble_mod

    seated_p = _write_solid(tmp_path, "seated_only.jpg", (20, 30, 90))
    entry_p = _write_solid(tmp_path, "entry_only.jpg", (200, 30, 30))
    seated_dets = [Detection(2, "seated", float(t), seated_p, 8, 240, 10, 390, 140, 0.9)
                   for t in range(3)]
    entry_dets = [Detection(2, "entry", -5.0 + t, entry_p, 3, 240, 10, 390, 140, 0.9)
                  for t in range(2)]
    dets = seated_dets + entry_dets
    pairs = {2: {1: {"entry_track": 3, "exit_track": None}}}

    captured = {}
    orig = assemble_mod.build_evidence_sheet

    def _spy(paths, boxes, **kw):
        captured["paths"] = list(paths)
        return orig(paths, boxes, **kw)

    assemble_mod.build_evidence_sheet = _spy
    try:
        anns = assemble_mod.assemble_annotations(
            [_seg(n=1)], dets, {2: {8: 1}}, use_vlm=False, pairs_by_order=pairs,
        )
    finally:
        assemble_mod.build_evidence_sheet = orig

    assert anns[0].best_frame_entry == str(entry_p), "前提：进出场确实配对成功"
    assert captured.get("paths"), "前提：证据图确实被构建"
    assert entry_p not in captured["paths"], "证据图不得包含进出场帧（工作人员误入的风险来源）"
    assert all(p == seated_p for p in captured["paths"]), "证据图只能用就座帧"


def test_evidence_sheet_handles_fewer_than_four_seated(tmp_path):
    """就座帧不足 4 张（此处只有 2 张）时不崩、不拿进出场帧补位。

    进出场帧唾手可得（entry_p 就在 dets 里且配对成功），若实现图省事拿它
    补足 4 格，这条测试会抓到。
    """
    import pipeline.assemble as assemble_mod

    seated_p = _write_solid(tmp_path, "seated_few.jpg", (20, 30, 90))
    entry_p = _write_solid(tmp_path, "entry_few.jpg", (200, 30, 30))
    seated_dets = [Detection(2, "seated", float(t), seated_p, 8, 240, 10, 390, 140, 0.9)
                   for t in range(2)]   # 只有 2 张就座帧
    entry_dets = [Detection(2, "entry", -5.0, entry_p, 3, 240, 10, 390, 140, 0.9)]
    dets = seated_dets + entry_dets
    pairs = {2: {1: {"entry_track": 3, "exit_track": None}}}

    captured = {}
    orig = assemble_mod.build_evidence_sheet

    def _spy(paths, boxes, **kw):
        captured["paths"] = list(paths)
        return orig(paths, boxes, **kw)

    assemble_mod.build_evidence_sheet = _spy
    try:
        anns = assemble_mod.assemble_annotations(
            [_seg(n=1)], dets, {2: {8: 1}}, use_vlm=False, pairs_by_order=pairs,
        )
    finally:
        assemble_mod.build_evidence_sheet = orig

    assert captured.get("paths") is not None, "不应因帧数不足而跳过证据图构建"
    assert 0 < len(captured["paths"]) <= 2, "不足 4 张就按实际张数组图，不许补位到 4"
    assert entry_p not in captured["paths"], "不得用进出场帧补位"
    assert anns[0].pixel.get("_evidence_png"), "不应因帧数不足而崩溃或跳过证据图"


# ---------------------------------------------------------------------------
# Task 25 · 缺陷二（改稿）—— has_jacket 废弃，改为让 formality_1_5 的每一档
# 含义在 result.csv 里直接可见，用户不必回去翻文档。新增 formality_label
# 列，文案从 pipeline.annotate.FORMALITY_LABELS 派生；缺座位/VLM 失败时
# formality_1_5 本身为空，该列也必须留空，不能编造档位。
# ---------------------------------------------------------------------------

def test_result_csv_has_formality_label(tmp_path):
    from pipeline.annotate import formality_label
    seg = _seg(n=2)
    dets = _dets(tmp_path)
    seats = {8: 1}  # 座位2 无 track_id 分配 -> 走 _poor_annotation，无正式度

    import pipeline.assemble as assemble_mod

    def _fake_ask_vlm(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9,
                "outer_hue_class": "蓝", "formality_1_5": 4}

    orig = assemble_mod.ask_vlm
    assemble_mod.ask_vlm = _fake_ask_vlm
    try:
        anns = assemble_mod.assemble_annotations([seg], dets, {2: seats}, use_vlm=True)
    finally:
        assemble_mod.ask_vlm = orig

    out = tmp_path / "result.csv"
    write_result_csv(anns, out)

    import csv as csv_mod
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv_mod.DictReader(fh))
    assert "formality_label" in rows[0].keys(), "CSV 必须有 formality_label 列"

    from pipeline.annotate import FORMALITY_LABELS

    seat1 = next(r for r in rows if r["seat_index"] == "1")
    assert seat1["formality_label"] == formality_label(4)
    assert seat1["formality_label"] == f"4 = {FORMALITY_LABELS[4]}"

    seat2 = next(r for r in rows if r["seat_index"] == "2")
    assert seat2["formality_1_5"] == "", "前提：缺座位行确实没有正式度"
    assert seat2["formality_label"] == "", "没有正式度就不能编造标签"


# ---------------------------------------------------------------------------
# Task 26 · 需求二 —— 一企业一行的宽表（result_wide.csv）。
#
# 用户原话：「每个企业为一行，依次输出座位1、座位2…的相关值到不同列，
# 如果相应座位没有，取缺失值即可」。build_wide_rows 是纯函数，直接消费
# annotation_to_row 同源的长表行（list[dict]），不重算任何度量——这里用
# 手写的长表字典做单元测试，比端到端跑 assemble_annotations 更直接、更快，
# 也更容易控制"某人缺测量"这种边界情况。
# ---------------------------------------------------------------------------

def _long_row(order, company, seat, presenter_num, **extra):
    row = {
        "order": order, "company": company, "seat_index": seat,
        "presenter_num": presenter_num, "seg_start": 100, "seg_end": 200,
        "gender_female": 0, "speaks_pre": 1 if seat == 1 else 0, "speaks_qa": 0,
        "n_frames_color": 5, "sharpness_norm": 0.5,
    }
    row.update(extra)
    return row


def test_wide_csv_one_row_per_company():
    """16 家企业 -> 16 行。"""
    from pipeline.assemble import build_wide_rows
    rows = [
        _long_row(order, f"公司{order}", 1, 1, formality_1_5=3,
                  hue_class_final="蓝", confidence=0.8)
        for order in range(1, 17)
    ]
    wide = build_wide_rows(rows)
    assert len(wide) == 16
    assert sorted(r["order"] for r in wide) == list(range(1, 17))


def test_wide_csv_missing_seat_is_blank():
    """某企业只有 1 人时，seat2_*/seat3_*/seat4_* 全为空字符串，不能填 0。"""
    from pipeline.assemble import build_wide_rows, WIDE_SEAT_FIELDS
    rows = [_long_row(1, "公司1", 1, 1, formality_1_5=4, hue_class_final="黑",
                       confidence=0.9, neutral_ref_found=0)]
    wide = build_wide_rows(rows)
    r = wide[0]
    for seat in (2, 3, 4):
        for field in WIDE_SEAT_FIELDS:
            assert r[f"seat{seat}_{field}"] == "", (
                f"seat{seat}_{field} 应留空，实得 {r[f'seat{seat}_{field}']!r}"
            )
    # 座位1 的真实值不能被上面这条"缺失留空"逻辑误伤。
    # 原先这里断言 seat1_gender_female == 0，专挑一个值为 0 的字段来守
    # 「0 是合法值、不许当成缺失抹掉」。gender_female 已从宽表移除（Task 34
    # 改动三），改用了 view_conflict（0=外层一致）；view_conflict 本身已于
    # Task 35 · 改动一整体删除（实测三组 103 行零信息量：56 行为 2、47 行
    # 为空，无一行为 0 或 1），改用 neutral_ref_found——它同样有合法的 0
    # 值（0=未在画面中找到中性参照物），守的是同一件事。
    assert r["seat1_formality_1_5"] == 4
    assert r["seat1_neutral_ref_found"] == 0, "0 是合法值，不能被当作缺失抹成空串"


# ---------------------------------------------------------------------------
# Task 35 · 改动四：以下 4 条测试原是团队层面聚合列（team_formality_mean/
# team_n_measured/team_hue_mean 护栏/team_formality_lead/team_hue_modal/
# team_any_light/team_all_dark）的专属测试，随 9 个 team_* 列一并删除——
# 那些列本身已不存在，测试无对象可测。
#
# 但删之前逐条核实过是否顺带守着别的不变量：
#   - test_wide_csv_lead_equals_seat1 顺带守着一条重要的、与 team_* 无关的
#     不变量——`by_seat_idx = {int(r["seat_index"]): r for r in seat_rows}`
#     必须按 seat_index 精确取行，不能偷懒取 `seat_rows[0]`（长表里遇到的
#     第一条记录）。这条不变量关系到**全部** seatN_* 字段，不只是已删除的
#     team_formality_lead，因此迁移到下面的
#     `test_wide_csv_seat_fields_keyed_by_seat_index_not_row_order`，
#     继续用「座位2 的行排在输入列表前面」这个构造守住它。
#   - 其余 3 条（team_mean 忽略缺失、hue 均值护栏、hue 众数并列取小/
#     any_light/all_dark）纯粹是团队聚合口径本身的测试，没有可迁移的
#     旁路不变量，随功能一起删除。
# ---------------------------------------------------------------------------

def test_wide_csv_seat_fields_keyed_by_seat_index_not_row_order():
    """宽表按 seat_index 精确取行，不是按长表里出现的先后顺序。

    座位 2 的行故意排在输入列表的第一位：若实现偷懒取 `seat_rows[0]`
    （长表里遇到的第一条记录）当座位 1 用，本测试能抓到——这正是删除
    team_formality_lead 之前它所守的同一条不变量（原
    test_wide_csv_lead_equals_seat1，见上方说明），现直接测 seatN_* 字段
    本身，不再依赖已删除的 team_* 列。
    """
    from pipeline.assemble import build_wide_rows
    rows = [
        _long_row(1, "公司1", 2, 2, formality_1_5=5),  # 座位2 的行排在前面
        _long_row(1, "公司1", 1, 2, formality_1_5=2),  # 座位1 的行排在后面
    ]
    wide = build_wide_rows(rows)
    r = wide[0]
    assert r["seat1_formality_1_5"] == 2, "座位1应取seat_index==1的行，不是列表第一条"
    assert r["seat2_formality_1_5"] == 5


def test_write_wide_csv_end_to_end(tmp_path):
    """端到端：write_wide_csv 走真实 assemble_annotations 产出，写盘并回读，
    确认列存在、行数正确、result.csv 不受影响。"""
    from pipeline.assemble import write_wide_csv
    seg = _seg(n=2)
    anns = assemble_annotations([seg], _dets(tmp_path), {2: {8: 1, 7: 2}}, use_vlm=False)

    out = tmp_path / "result_wide.csv"
    write_wide_csv(anns, out)

    import csv as csv_mod
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv_mod.DictReader(fh))
    assert len(rows) == 1, "一个企业只应有一行"
    assert rows[0]["company"] == "公司2"
    assert "seat1_formality_1_5" in rows[0]
    assert "seat4_formality_1_5" in rows[0], "座位数上限取4，即便实测不足4人，列也要在"
    assert rows[0]["seat3_formality_1_5"] == "", "只有2人时座位3应留空"

    # 不得影响 result.csv
    result_out = tmp_path / "result.csv"
    write_result_csv(anns, result_out)
    with result_out.open(encoding="utf-8-sig") as fh:
        result_rows = list(csv_mod.DictReader(fh))
    assert len(result_rows) == 2, "长表仍是每人一行"


# ---------------------------------------------------------------------------
# 需求二·新增测试 9 —— 长表格式不变（硬性约束 7：现有 result.csv 的列与
# 行为不许改变，多条既有测试已锁着它；这里额外补一条明确断言宽表列绝不能
# 渗进长表，作为 build_wide_rows/write_wide_csv 引入后的直接护栏）。
# ---------------------------------------------------------------------------

def test_result_csv_long_format_unchanged(tmp_path):
    anns = assemble_annotations([_seg(n=2)], _dets(tmp_path), {2: {8: 1, 7: 2}},
                                use_vlm=False)
    out = tmp_path / "result.csv"
    write_result_csv(anns, out)

    import csv as csv_mod
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv_mod.DictReader(fh))
    assert len(rows) == 2, "长表仍是每人一行"
    header = list(rows[0].keys())
    assert not any(c.startswith("seat1_") or c.startswith("team_") for c in header), (
        "宽表列不得混进长表"
    )
    assert "seat_index" in header and "person_id" in header and "company" in header


# --- 宽表：区分「未识别」与「无此座位」（用户 2026-08-16 要求）---

def _wide_row(order, company, presenter_num, seats):
    """seats: {seat_index: formality} —— 只造宽表需要的最小长表行。"""
    return [{
        "order": order, "company": company, "presenter_num": presenter_num,
        "seg_start": 0, "seg_end": 100, "seat_index": s,
        "formality_1_5": f, "gender_female": 0, "speaks_pre": 1 if s == 1 else 0,
        "speaks_qa": 0, "hue_class_final": "黑" if f else "",
    } for s, f in seats.items()]


def test_wide_csv_marks_undetected_apart_from_nonexistent_seat():
    """标注 3 人只测到 1 人 vs 标注 1 人本来就没有座位 2——两者在宽表里
    必须能区分开。此前两种情况的 seat2_* 列都是空字符串，长得一模一样，
    用户无法判断「这个人没识别出来」还是「这个企业本来就没有第二个人」。
    """
    from pipeline.assemble import build_wide_rows

    rows = (_wide_row(1, "只有一人", 1, {1: "3"})
            + _wide_row(2, "三人只测到一人", 3, {1: "4"}))
    wide = {r["order"]: r for r in build_wide_rows(rows)}

    assert wide[1]["seat2_status"] == "无此座位", "标注 1 人时座位 2 不存在"
    assert wide[2]["seat2_status"] == "未识别", "标注 3 人时座位 2 是没测到，不是不存在"
    assert wide[1]["seat1_status"] == "已识别"
    assert wide[2]["seat3_status"] == "未识别"


def test_wide_csv_status_counts_are_consistent_with_presenter_num():
    """每企业「已识别 + 未识别」的座位数必须恰等于 presenter_num，
    其余一律「无此座位」——防止状态列自身与人数标注对不上。"""
    from pipeline.assemble import build_wide_rows

    rows = _wide_row(5, "两人测到一人", 2, {1: "3"})
    r = build_wide_rows(rows)[0]
    statuses = [r[f"seat{i}_status"] for i in range(1, 5)]
    assert statuses.count("无此座位") == 2
    assert statuses.count("已识别") + statuses.count("未识别") == 2


# --- Task 34 收尾：宽表去冗余列、新量表落库、像素轨可复启 ---

def test_wide_csv_drops_redundant_annotation_columns():
    """宽表不再输出 gender_female / speaks_pre / speaks_qa。

    这三个是用户自己 dta 里 prst{N}Female / preprst{N} / qaprst{N} 的原样
    搬运，用户要求宽表去掉。长表 result.csv 保留（每行自足有其价值）。
    """
    from pipeline.assemble import build_wide_rows, WIDE_SEAT_FIELDS

    for f in ("gender_female", "speaks_pre", "speaks_qa"):
        assert f not in WIDE_SEAT_FIELDS, f"宽表不该再有 {f}"

    rows = _wide_row(1, "某企业", 2, {1: "3", 2: "4"})
    wide = build_wide_rows(rows)[0]
    for f in ("gender_female", "speaks_pre", "speaks_qa"):
        assert f"seat1_{f}" not in wide


def test_long_csv_keeps_annotation_columns():
    """长表仍保留那三列——用户只要求宽表去掉，别顺手把长表也削了。"""
    from pipeline.gallery import annotation_to_row
    from pipeline.annotate import PersonAnnotation

    a = PersonAnnotation(person_id="p", seg_order=1, company="c", seat_index=1,
                         presenter_num=1, gender_female=1, speaks_pre=1,
                         speaks_qa=0, seg_start=0, seg_end=10)
    row = annotation_to_row(a)
    assert row["gender_female"] == 1
    assert row["speaks_pre"] == 1
    assert "speaks_qa" in row


def test_wide_csv_has_no_pixel_only_columns():
    """宽表固定不含像素轨专属列，且不随 PIXEL_COLOR_ENABLED 回退。

    用户定版：宽表是给人看的最终产物，列集合稳定比可回溯更重要。像素轨
    重新开启时长表仍会带上那些值，但宽表保持这一套列不变。
    """
    from pipeline import config
    from pipeline.assemble import WIDE_SEAT_FIELDS

    pixel_only = ("hue_class_pixel", "hue_class_agree", "outer_hue_corr",
                  "outer_sat_corr", "outer_val_corr", "outer_lab_L",
                  "outer_lab_a", "outer_lab_b")
    for f in pixel_only:
        assert f not in WIDE_SEAT_FIELDS, f"宽表不该有 {f}"

    # 即便重新开启像素轨，宽表列也不变——这是与长表刻意不同的取舍
    import importlib
    from pipeline import assemble as A
    orig = config.PIXEL_COLOR_ENABLED
    try:
        config.PIXEL_COLOR_ENABLED = True
        importlib.reload(A)
        for f in pixel_only:
            assert f not in A.WIDE_SEAT_FIELDS, f"开启像素轨后宽表仍不该有 {f}"
    finally:
        config.PIXEL_COLOR_ENABLED = orig
        importlib.reload(A)


def test_new_vlm_scale_fields_reach_csv():
    """VLM 新增的两个 5 级量表与参照物标志必须真正进入 CSV。

    PROMPT 里要了字段但下游没落库，是本项目反复防的「要了不落地」缺陷：
    模型花 token 算了，结果没人收。
    """
    from pipeline.gallery import annotation_to_row
    from pipeline.annotate import PersonAnnotation

    a = PersonAnnotation(person_id="p", seg_order=1, company="c", seat_index=1,
                         presenter_num=1, gender_female=0, speaks_pre=1,
                         speaks_qa=0, seg_start=0, seg_end=10,
                         vlm={"outer_saturation_1_5": 4, "outer_brightness_1_5": 2,
                              "neutral_ref_found": 1})
    row = annotation_to_row(a)
    assert row["outer_saturation_1_5"] == 4
    assert row["outer_brightness_1_5"] == 2
    assert row["neutral_ref_found"] == 1


def test_pixel_only_fields_absent_from_csv_when_disabled(tmp_path):
    """像素轨关闭时，那 6 个连续色彩量与 3 个派生字段一律不得进 CSV。

    补于变异测试：把 `_finalize_pixel_dict` 的过滤短路掉之后，全套 434 条
    测试竟然全绿——说明「关闭时不输出」这条只有实现、没有任何测试守着。
    此前只测了「开启时字段齐全」（pixel_track_on 那几条），漏了对侧。
    """
    from pipeline.assemble import measure_person_color, PIXEL_ONLY_FIELDS
    from pipeline.gallery import annotation_to_row
    from pipeline.annotate import PersonAnnotation

    assert config.PIXEL_COLOR_ENABLED is False

    p = _write_solid(tmp_path, "a.jpg", (20, 30, 90))
    dets = [Detection(2, "seated", 0.0, p, 5, 240.0, 10.0, 390.0, 140.0, 0.9)]
    pixel = measure_person_color(dets)
    for f in PIXEL_ONLY_FIELDS:
        assert f not in pixel, f"像素轨已关，{f} 不该出现在测量结果里"

    a = PersonAnnotation(person_id="p", seg_order=1, company="c", seat_index=1,
                         presenter_num=1, gender_female=0, speaks_pre=1,
                         speaks_qa=0, seg_start=0, seg_end=10, pixel=pixel)
    row = annotation_to_row(a)
    for f in PIXEL_ONLY_FIELDS:
        assert f not in row, f"像素轨已关，{f} 不该出现在 CSV 行里"


def test_pixel_only_fields_absent_even_for_empty_measurement():
    """零帧占位路径同样要过滤——不能有的路径删了、有的路径漏了。"""
    from pipeline.assemble import measure_person_color, PIXEL_ONLY_FIELDS

    pixel = measure_person_color([])
    for f in PIXEL_ONLY_FIELDS:
        assert f not in pixel, f"零帧占位路径漏删了 {f}"


def test_pixel_color_code_is_retained_and_still_works(pixel_track_on, tmp_path,
                                                      monkeypatch):
    """用户要求「代码保留，只是不执行」——重新开启后必须真的还能跑。

    这条守的是「保留下来的不是一坨死代码」：关掉一个功能最容易的失败方式
    是它悄悄腐烂，等哪天想开回来才发现早就坏了。
    """
    from pipeline.assemble import measure_person_color
    from pipeline.colormath import rgb_to_lab, classify_hue_11
    from pipeline.whitebalance import von_kries_gains

    _force_full_crop_as_upper_clothes(monkeypatch)
    p = _write_solid(tmp_path, "navy.jpg", (20, 30, 90))
    dets = [Detection(2, "seated", 0.0, p, 5, 240.0, 10.0, 390.0, 140.0, 0.9)]
    pixel = measure_person_color(dets)
    assert pixel["outer_lab_L"] is not None, "重新开启后应真正产出测量值"
    assert callable(rgb_to_lab) and callable(classify_hue_11)
    assert callable(von_kries_gains)



def test_evidence_sheet_does_not_depend_on_pixel_track(tmp_path, monkeypatch):
    """送给 VLM 的 2×2 证据图必须与像素轨开关无关（用户 2026-08-16 追问）。

    证据图走的是原始像素：Image.open → 裁剪 → 画红框 → 拼图，不经白平衡
    增益、不经服装分割。选帧用的 Laplacian 清晰度也独立于分割与白平衡。
    这条锁住该不变量——将来若有人把白平衡校正塞进证据图，关掉像素轨就会
    悄悄改变送进模型的图像，而 VLM 的判读会跟着变，且没有任何字段能反映。
    """
    from pipeline.annotate import build_evidence_sheet, select_best_frames

    p = _write_solid(tmp_path, "ev.jpg", (20, 30, 90))
    dets = [Detection(1, "seated", float(i), p, 5, 40.0, 20.0, 200.0, 260.0, 0.9)
            for i in range(4)]
    boxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]

    out = {}
    for state in (True, False):
        monkeypatch.setattr(config, "PIXEL_COLOR_ENABLED", state)
        chosen = select_best_frames(dets, n=4)
        out[state] = (build_evidence_sheet([d.frame_path for d in chosen], boxes),
                      [d.t for d in chosen])

    assert out[True][0] == out[False][0], "证据图字节必须完全一致"
    assert out[True][1] == out[False][1], "选中的帧必须完全一致"


# ---------------------------------------------------------------------------
# Task 35 · 控制器指定的四条删列测试。
# ---------------------------------------------------------------------------

def test_view_conflict_gone_from_prompt_and_wide():
    """改动一：view_conflict 必须从 PROMPT 与 WIDE_SEAT_FIELDS 中彻底消失。

    实测三组 103 行：56 行为 2（无法比较）、47 行为空，无一行为 0 或 1，
    零信息量——PROMPT 里连三态说明那一整段（约 5 行）都要删干净，不能只
    删 schema 行、留半句孤立的指引文字。
    """
    from pipeline.annotate import PROMPT
    from pipeline.assemble import WIDE_SEAT_FIELDS

    assert "view_conflict" not in PROMPT, "PROMPT 里不应再提及 view_conflict"
    assert "view_conflict" not in WIDE_SEAT_FIELDS


def test_n_frames_color_not_in_output_when_pixel_disabled(tmp_path):
    """改动二：像素轨关闭（默认状态）时，n_frames_color 既不进 WIDE_SEAT_
    FIELDS，也不出现在 measure_person_color 的返回字典与长表行里。

    实测三组 103 行该字段全为 0——像素轨关闭后恒为 0，是死列。
    """
    from pipeline.assemble import (
        WIDE_SEAT_FIELDS, PIXEL_ONLY_FIELDS, measure_person_color,
    )
    from pipeline.gallery import annotation_to_row
    from pipeline.annotate import PersonAnnotation

    assert config.PIXEL_COLOR_ENABLED is False
    assert "n_frames_color" not in WIDE_SEAT_FIELDS
    assert "n_frames_color" in PIXEL_ONLY_FIELDS, (
        "n_frames_color 应走 PIXEL_ONLY_FIELDS 既有的过滤机制，不另造一套"
    )

    p = _write_solid(tmp_path, "a.jpg", (20, 30, 90))
    dets = [Detection(2, "seated", 0.0, p, 5, 240.0, 10.0, 390.0, 140.0, 0.9)]
    pixel = measure_person_color(dets)
    assert "n_frames_color" not in pixel, "像素轨已关，字段不该出现在测量结果里"

    a = PersonAnnotation(person_id="p", seg_order=1, company="c", seat_index=1,
                         presenter_num=1, gender_female=0, speaks_pre=1,
                         speaks_qa=0, seg_start=0, seg_end=10, pixel=pixel)
    row = annotation_to_row(a)
    assert "n_frames_color" not in row, "像素轨已关，字段不该出现在 CSV 行里"


def test_no_team_level_columns_in_wide():
    """改动四：9 个 team_* 列一个都不该出现在宽表里。"""
    from pipeline.assemble import build_wide_rows

    team_cols = {
        "team_n_measured", "team_formality_mean", "team_formality_lead",
        "team_formality_sd", "team_hue_lead", "team_hue_modal",
        "team_any_light", "team_all_dark", "team_confidence_mean",
    }
    rows = [
        _long_row(1, "公司1", 1, 2, formality_1_5=3, confidence=0.8,
                  hue_class_final="黑"),
        _long_row(1, "公司1", 2, 2, formality_1_5=5, confidence=0.6,
                  hue_class_final="白"),
    ]
    wide = build_wide_rows(rows)
    r = wide[0]
    present = team_cols & set(r)
    assert not present, f"宽表不该再有 team_* 列，实得 {present}"
    assert not any(k.startswith("team_") for k in r), (
        f"宽表列里仍有以 team_ 开头的键：{[k for k in r if k.startswith('team_')]}"
    )


def test_seat_status_column_survives():
    """护栏：删 team_* 时不能把 seatN_status 三态列一起误删——用户明确
    要求保留这一列（区分「已识别」/「未识别」/「无此座位」）。"""
    from pipeline.assemble import build_wide_rows

    rows = (_wide_row(1, "只有一人", 1, {1: "3"})
            + _wide_row(2, "三人只测到一人", 3, {1: "4"}))
    wide = {r["order"]: r for r in build_wide_rows(rows)}

    for order in (1, 2):
        for seat in range(1, 5):
            assert f"seat{seat}_status" in wide[order]
    assert wide[1]["seat2_status"] == "无此座位"
    assert wide[2]["seat2_status"] == "未识别"
    assert wide[1]["seat1_status"] == "已识别"


# ---------------------------------------------------------------------------
# Task 45 · 改动一：VLM 增量打标——递补之后只跑新增/失败的人，不整组重跑。
#
# 两个理由：(1) 浪费——文化4组34人只有5人是新递补的，其余29次调用是重复
# 劳动；(2) 更要紧的是可验证性——VLM 存在运行间波动，整组重跑会把「递补
# 带来的变化」与「VLM 自身抖动」搅在一起分不开。只有让未变的人结果原样
# 冻结，才能把两者分离开看。
# ---------------------------------------------------------------------------

def test_reuses_valid_previous_annotation(tmp_path):
    """上一轮有效标注（vlm 非空、无 vlm_error、layout 未变）被复用，本轮
    不再真正调用 VLM——这是增量打标存在的意义。"""
    seg = _seg(n=2)
    dets = _dets(tmp_path)
    seats = {2: {8: 1, 7: 2}}

    calls_round1 = []

    def fake1(sheet_png, context):
        calls_round1.append(context["person_id"])
        return {"evidence_quality": "good", "confidence": 0.9,
                "outer_hue_class": "蓝", "formality_1_5": 4, "gender_judged": "男"}

    anns1 = _run_with_fake_ask_vlm(fake1, [seg], dets, seats)
    assert len(calls_round1) == 2, "首轮两人都应真正调用 VLM"

    prior = [dataclasses.asdict(a) for a in anns1]

    def fake2(sheet_png, context):
        raise AssertionError(
            f"不应调用 VLM：{context.get('person_id')} 本应命中增量缓存被复用"
        )

    anns2 = _run_with_fake_ask_vlm(fake2, [seg], dets, seats, prior_annotations=prior)

    assert len(anns2) == 2
    for a1, a2 in zip(sorted(anns1, key=lambda a: a.seat_index),
                       sorted(anns2, key=lambda a: a.seat_index)):
        assert a2.vlm == a1.vlm, "复用轮的 vlm 字典必须与上一轮逐字节相同"


def test_reruns_previously_failed_person(tmp_path):
    """上一轮 vlm_error 非空的人必须重跑——这正是要重试的对象（brief 变异
    清单 (a)：增量判据不得忽略 vlm_error）。"""
    seg = _seg(n=2)
    dets = _dets(tmp_path)
    seats = {2: {8: 1, 7: 2}}

    def fake_fail_seat1(sheet_png, context):
        if context["person_id"].endswith("track8"):   # 座位1 对应 tid=8
            raise RuntimeError("503 no channel")
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}

    anns1 = _run_with_fake_ask_vlm(fake_fail_seat1, [seg], dets, seats)
    by_seat1 = {a.seat_index: a for a in anns1}
    assert by_seat1[1].vlm.get("vlm_error"), "夹具前提：座位1首轮应失败"
    prior = [dataclasses.asdict(a) for a in anns1]

    called = []

    def fake2(sheet_png, context):
        called.append(context["person_id"])
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "红"}

    _run_with_fake_ask_vlm(fake2, [seg], dets, seats, prior_annotations=prior)
    assert any("track8" in c for c in called), f"上次失败的人必须重跑，实际调用：{called}"


def test_reruns_when_layout_changed(tmp_path):
    """上一轮 layout 与本轮不同时必须重跑——防「拿错误前提下的判断充数」
    （brief 变异清单 (b)）。"""
    seg = _seg(n=1)
    dets = _dets(tmp_path)
    seats = {2: {8: 1}}

    def fake1(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}

    anns1 = _run_with_fake_ask_vlm(fake1, [seg], dets, seats)
    prior = [dataclasses.asdict(a) for a in anns1]
    assert prior[0]["layout"], "夹具前提：首轮应写下非空 layout"
    # 人为篡改上一轮的 layout，模拟"证据图构成变了"（例如从纯就座变成含
    # 进出场）——即便 vlm 本身有效、无 vlm_error，构成一变就必须重跑。
    prior[0]["layout"] = "【证据图构成说明】完全不同的构成文案"

    called = []

    def fake2(sheet_png, context):
        called.append(context["person_id"])
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "红"}

    _run_with_fake_ask_vlm(fake2, [seg], dets, seats, prior_annotations=prior)
    assert called, "layout 变化后必须重新调用 VLM，不能沿用旧判断"


def test_reruns_when_previous_file_lacks_layout(tmp_path):
    """旧文件没有 layout 字段（Task 45 之前产出的 annotations.json）时，
    保守起见视为"构成未知"并重跑（brief 变异清单 (b) 的另一半）。"""
    seg = _seg(n=1)
    dets = _dets(tmp_path)
    seats = {2: {8: 1}}

    def fake1(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}

    anns1 = _run_with_fake_ask_vlm(fake1, [seg], dets, seats)
    prior = [dataclasses.asdict(a) for a in anns1]
    del prior[0]["layout"]

    called = []

    def fake2(sheet_png, context):
        called.append(context["person_id"])
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "红"}

    _run_with_fake_ask_vlm(fake2, [seg], dets, seats, prior_annotations=prior)
    assert called, "旧文件没有 layout 字段时必须保守重跑"


def test_new_backfilled_person_is_annotated(tmp_path):
    """已有一份"稳定"上一轮结果（座位1 就座、有效），本轮新增递补出座位2
    ——座位1 应命中复用，座位2（新递补）必须真正调用 VLM，不能因为『存在
    prior_annotations』就被当成"曾经处理过"（brief：source=="entryexit"
    且上一轮没有该 person_id）。"""
    seg = _seg(n=2)
    dets = _dets(tmp_path) + _entry_dets(tmp_path, order=2, tid=60)
    labels = {"2": {"entry:60": {"role": "创业者", "seat": 2}}}

    def fake1(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}

    # 首轮只有座位1（就座），座位2 就座阶段本就是空的、也没有 entry_labels。
    anns1 = _run_with_fake_ask_vlm(fake1, [seg], dets, {2: {8: 1}})
    prior = [dataclasses.asdict(a) for a in anns1]

    called = []

    def fake2(sheet_png, context):
        called.append(context["person_id"])
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "红"}

    anns2 = _run_with_fake_ask_vlm(fake2, [seg], dets, {2: {8: 1}},
                                   prior_annotations=prior, entry_labels=labels)

    by_seat = {a.seat_index: a for a in anns2}
    assert by_seat[2].source == "entryexit"
    assert any("_entryexit" in c for c in called), (
        f"新递补的人必须真正调用 VLM，实际调用：{called}"
    )
    assert not any(c.endswith("track8") for c in called), (
        "座位1（未变）应命中复用，不应被重新调用"
    )


def test_force_vlm_ignores_cache(tmp_path):
    """--force-vlm 时即便缓存完全有效也必须忽略，全部重新调用
    （brief 变异清单 (d)）。"""
    seg = _seg(n=1)
    dets = _dets(tmp_path)
    seats = {2: {8: 1}}

    def fake1(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}

    anns1 = _run_with_fake_ask_vlm(fake1, [seg], dets, seats)
    prior = [dataclasses.asdict(a) for a in anns1]   # 完全有效、可复用的上一轮结果

    called = []

    def fake2(sheet_png, context):
        called.append(context["person_id"])
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "红"}

    _run_with_fake_ask_vlm(fake2, [seg], dets, seats,
                           prior_annotations=prior, force_vlm=True)
    assert called, "--force-vlm 时即便缓存有效也必须忽略，全部重新调用"


def test_stage5_reports_reuse_counts(tmp_path, capsys):
    """复用/实调人数必须打印（凡有复用必须可见，brief 变异清单 (c)）。"""
    seg = _seg(n=2)
    dets = _dets(tmp_path)
    seats = {2: {8: 1, 7: 2}}

    def fake1(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}

    anns1 = _run_with_fake_ask_vlm(fake1, [seg], dets, seats)
    capsys.readouterr()   # 清掉首轮的输出，只看第二轮

    prior = [dataclasses.asdict(a) for a in anns1]
    # 只让座位1失效（改动其 layout），座位2 保持可复用。
    prior_seat1 = next(r for r in prior if r["seat_index"] == 1)
    prior_seat1["layout"] = "【证据图构成说明】不同的构成"

    def fake2(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "红"}

    _run_with_fake_ask_vlm(fake2, [seg], dets, seats, prior_annotations=prior)
    out = capsys.readouterr().out
    assert "复用 1 人" in out and "调用 VLM 1 人" in out, f"实得：\n{out}"


def test_stage5_reports_full_rerun_when_incremental_disabled(tmp_path, monkeypatch, capsys):
    """VLM_INCREMENTAL=False 时打印"全量重跑"，不套用"复用N/调用M"措辞。"""
    from pipeline import config as cfg
    seg = _seg(n=1)
    dets = _dets(tmp_path)
    seats = {2: {8: 1}}

    def fake1(sheet_png, context):
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}

    anns1 = _run_with_fake_ask_vlm(fake1, [seg], dets, seats)
    prior = [dataclasses.asdict(a) for a in anns1]
    capsys.readouterr()

    monkeypatch.setattr(cfg, "VLM_INCREMENTAL", False)
    _run_with_fake_ask_vlm(fake1, [seg], dets, seats, prior_annotations=prior)
    out = capsys.readouterr().out
    assert "全量重跑" in out


def test_no_backfill_when_labels_none_regression(tmp_path):
    """护栏：entry_labels=None 时行为必须与增量打标/递补前置条件改动之前
    完全相同——不能因为新增的 needs_seat/合并/复用逻辑而意外触碰这条
    向后兼容路径。"""
    seg = _seg(n=2)
    dets = _dets(tmp_path)
    seats = {2: {8: 1, 7: 2}}
    anns = assemble_annotations([seg], dets, seats, use_vlm=False, entry_labels=None)
    for a in anns:
        assert a.source == "seated"
        assert a.seat_uncertain == 0
        assert a.expert_removed == 0


# --- 人工否决的座位必须真能被替换（用户 2026-08-23）---

def test_rejected_seat_is_replaced_not_blocked(tmp_path, monkeypatch):
    """用户标「这个座位判错了」并指定替代候选后，装配阶段必须真的换掉它。

    `entryexit_fill_plan` 已把被否决的座位当作空缺吐进补位计划，但
    `assemble_annotations` 里有一道防御 `if seat in by_final_seat: continue`
    ——它挡住了递补覆盖就座结果（那是对的，防止自动递补悄悄改写人工可见
    的座位表），却连**用户明确否决**的座位也一并挡住。后果：用户说了判错、
    页面也照做记进了标注文件，跑一遍却什么都没变，且不报错。

    正确边界：自动递补不许覆盖；用户显式否决的必须覆盖。两者的区别在于
    标注文件里有没有那条 override，不在于座位是否已被占用。
    """
    from pipeline.assemble import assemble_annotations

    seg = _seg(order=2, n=2)
    dets = _dets(tmp_path, order=2)
    seats = {8: 1, 7: 2}
    labels = {
        "_meta": {"detections_fingerprint": "x"},
        "seat_overrides": {"2": {"1": {"rejected": True,
                                       "replacement_stable_id": None,
                                       "note": "看错人了"}}},
    }
    anns = assemble_annotations([seg], dets, {2: seats}, use_vlm=False,
                                entry_labels=labels)
    a1 = next(a for a in anns if a.seat_index == 1)
    assert getattr(a1, "seat_rejected", 0) == 1, (
        "被人工否决的座位必须带标志，否则下游无从区分「没测到」与「测了但人工否掉」")


def test_seat_rejected_reaches_csv():
    """`seat_rejected` 必须真的落到 CSV。

    本项目已多次出现「加了字段却不落库」：Task 34 的两个 VLM 量表差点如此。
    加字段的目的就是让下游能区分「没测到」与「测了但人工否掉」，不进 CSV
    等于没加。
    """
    from pipeline.gallery import annotation_to_row
    from pipeline.annotate import PersonAnnotation

    a = PersonAnnotation(person_id="p", seg_order=1, company="c", seat_index=1,
                         presenter_num=1, gender_female=0, speaks_pre=1,
                         speaks_qa=0, seg_start=0, seg_end=10, seat_rejected=1)
    row = annotation_to_row(a)
    assert row.get("seat_rejected") == 1


def test_rejected_seat_can_actually_be_replaced(tmp_path):
    """指定了替代候选时，被否决的座位必须真被换掉，而不是被那道防御挡住。

    补于变异测试：把 `and seat not in rejected` 去掉后全套仍绿——「否决后
    可替换」这条只有实现、没有测试守着。上一条测试只验了标志位，验不到
    替换本身。
    """
    from pipeline.assemble import assemble_annotations
    from pipeline import entryexit as EE

    seg = _seg(order=2, n=2)
    dets = list(_dets(tmp_path, order=2))
    # `_dets` 只造 seated，走不到递补路径——必须另造 exit 窗口的检测，
    # 且要落在右区、竖长、足够高，才会被 entryexit_candidates 收为候选。
    p = _write_solid(tmp_path, "exit2.jpg", (30, 30, 30), size=(1080, 1440))
    for i in range(8):
        dets.append(Detection(2, "exit", 900.0 + i, p, 55,
                              1150.0, 300.0, 1330.0, 900.0, 0.9))
    cands = EE.entryexit_candidates(dets, seg)
    assert cands, "夹具应造出至少一个进出场候选，否则测不到替换路径"
    sid = cands[0]["stable_id"]
    labels = {
        "_meta": {"detections_fingerprint": "x"},
        "seat_overrides": {"2": {"1": {"rejected": True,
                                       "replacement_stable_id": sid,
                                       "note": ""}}},
    }
    anns = assemble_annotations([seg], dets, {2: {8: 1, 7: 2}}, use_vlm=False,
                                entry_labels=labels)
    a1 = next(a for a in anns if a.seat_index == 1)
    assert a1.source == "entryexit", (
        f"座位1 应被指定的进出场候选替换，实为 source={a1.source!r}")


# ---------------------------------------------------------------------------
# Task 51：按格位切分出来的递补——**只有那一格的一张图**。
#
# 用户在备注里按证据图的格位编号写清「1/3是创业者3，2/4是创业者2」，
# entryexit 据此把一个混了多人的候选切成若干个单人候选。切完之后：
#   · 送去测颜色/拼证据图的必须只有那一格的一张，拿整条 track 的四张
#     （四张里坐着不同的人）等于把多个人混着编码——Ruling R14 那一类，
#     不报错、数字全错；
#   · 构成文案必须如实说「共 1 张」，沿用四张拼图的说法会让模型去描述
#     并不存在的另外三张画面。
# ---------------------------------------------------------------------------

def test_cell_split_backfill_sends_every_cell_of_that_person(tmp_path):
    """一个人占几格，就把那几格的图**全都**送给 VLM。

    用户 2026-08-24：「登记表存多张就都给 VLM，增加信息量」。此前 assemble
    每个座位只送 `cell_index` 那一张代表格（Task 56A 只改到登记表为止），
    于是「登记表存 2 张 / 模型看 1 张」——留下来的另一张证据谁也用不上。

    夹具的备注是「1/3是创业者3，2/4是创业者2」：两个人各占两格，因此两次
    打标都该拿到 2 张图、构成文案都该说「共 2 张」并点明是第几、第几格。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.entryexit import entryexit_candidates

    seg = _seg(n=3)
    trans = _entry_dets(tmp_path, order=2, tid=60, kind="exit")
    dets = _dets(tmp_path) + trans
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 60)
    assert len(cand["best_dets"]) == 4, (
        f"夹具前提：该候选必须恰有 4 张证据帧（否则测不到「只取一张」），"
        f"实得 {len(cand['best_dets'])}"
    )
    labels = {"2": {cand["stable_id"]: {
        "role": "创业者", "seat": 3, "n_people": 2,
        "note": "1/3是创业者3，2/4是创业者2"}}}

    seen: list[list] = []
    seen_boxes: list[list] = []
    orig_sheet = assemble_mod.build_evidence_sheet

    def spy(paths, boxes, *a, **kw):
        seen.append(list(paths))
        seen_boxes.append(list(boxes))
        return orig_sheet(paths, boxes, *a, **kw)

    captured = []

    def fake(sheet_png, context):
        captured.append(context)
        return {"evidence_quality": "good", "confidence": 0.9, "outer_hue_class": "蓝"}

    assemble_mod.build_evidence_sheet = spy
    try:
        anns = _run_with_fake_ask_vlm(fake, [seg], dets, {2: {8: 1}},
                                      entry_labels=labels)
    finally:
        assemble_mod.build_evidence_sheet = orig_sheet

    by_seat = {a.seat_index: a for a in anns}
    assert by_seat[2].source == "entryexit" and by_seat[3].source == "entryexit", (
        f"座位 2、3 都该由格位切分补上，实得 "
        f"{[(s, a.source) for s, a in sorted(by_seat.items())]}"
    )

    split_ctxs = [c for c in captured if "_entryexit" in c.get("person_id", "")]
    assert len(split_ctxs) == 2, f"应有两次切分递补的打标，实得 {len(split_ctxs)}"
    for ctx in split_ctxs:
        layout = ctx["layout"]
        assert "共 2 张" in layout, (
            f"这个人占两格，构成文案必须如实说共 2 张，实得：{layout}"
        )
        assert "共 4 张" not in layout, (
            f"不得沿用四张拼图的构成文案（那三张里有别人），实得：{layout}"
        )
        assert "可见" in layout and "不可见" not in layout, (
            f"站立镜头下装可见，不能沿用就座措辞，实得：{layout}"
        )

    split_sheets = [p for p in seen if len(p) == 2]
    assert len(split_sheets) == 2, (
        f"两次切分递补的证据图都该用上那个人的两格，实得各次张数 "
        f"{[len(p) for p in seen]}"
    )
    # 「这两张确实是第 1、3 格那两帧」由 `cells_dets` 的单元测试
    # （test_entryexit.py::test_cells_dets_returns_one_frame_per_cell）钉住。
    # 这里只能验张数：本夹具的四张证据帧共用同一个 jpg、连框都相同，
    # 按路径或按框去重都区分不出格位，写了也证明不了任何事。
    assert all(len(b) == 2 for b in seen_boxes if len(b) == 2)


def test_describe_layout_names_every_cell_of_one_person(tmp_path):
    """`describe_layout` 要能表达「这个人占了第 1、2 格」。

    张数与格位数必须自洽：说了两格却只给一张图，等于让模型去描述并不存在
    的画面（与 Task 42「告诉模型下装不可见、而站立镜头里明明可见」同类，
    且下游一处也不会报错）。参数自相矛盾时宁可崩掉。
    """
    from pipeline.assemble import describe_layout

    txt = describe_layout(0, n_entry=2, cell_indices=[1, 2])
    assert "共 2 张" in txt, f"两格两张，实得：{txt}"
    # 断言必须写死「第 1、2 格」这个整串。只查 "1"、"2" 各自在不在是没用的
    # ——「共 2 张」里就有个 2，文案退化成只点第一格时照样能过（变异测试实测
    # 存活）。
    assert "第 1、2 格" in txt, f"每一格都要点到，实得：{txt}"

    single = describe_layout(0, n_entry=1, cell_indices=[3])
    assert single == describe_layout(0, n_entry=1, cell_index=3), (
        "单格时新旧两种传法必须给出逐字相同的文案"
    )
    # 单格文案是 Task 45 增量复用的**构成签名**，措辞动一个字，上一轮所有
    # 单格座位的结果都会被判为不可复用、全部重跑一遍 VLM。因此这里钉的是
    # 字面量本身，不是「两种传法互等」——那种自指的断言两边一起改也照样过
    # （变异测试实测存活）。
    assert single == (
        "【证据图构成说明】本组图共 1 张站立的进场镜头，是从一张挤了多个人的"
        "合图里按编号第 3 格切出来的单张画面，图中只有该创业者本人；"
        "该创业者本次没有其它可用画面，不要描述或推测本图之外的内容。"
        "站立镜头中下装（西裤/短裤等）与整体站姿清晰可见，可据此判定。"
        "进场镜头多为背向拍摄，但不要假设固定的摄像机方位。"
    ), f"单格文案与 Task 51 的原文逐字不符，会让既有标注全部失效重跑：{single}"

    with pytest.raises(ValueError):
        describe_layout(0, n_entry=1, cell_indices=[1, 2])   # 说两格却只有一张
    with pytest.raises(ValueError):
        describe_layout(2, n_entry=2, cell_indices=[1, 2])   # 混进了就座帧


def test_cell_exclusions_never_reach_the_entryexit_path(tmp_path):
    """逐格排除**不作用于进出场递补那一路**——那排框说的是就座那张 2×2。

    这条原先是两个测试，钉的是「按位置剔除」与「按格位号剔除」两种口径
    （assemble 与 registry 当时各用一种，本身就已分叉）。用户 2026-08-29 实测
    城市2 order 6 证明整条前提就不成立：页面上那排「排除格 N」复选框只长在
    `filled_seats`（`n_cells` 取自 `sheet_dets`，全部来自 seated 检测）上，
    进出场递补来的座位根本没有那排框，用户无从表达「排除递补记录的第 N 格」。

    两张完全不同的图共用一个 `(段, 座位)` 键，剔掉的必然不是他勾的那一格——
    实测就是把 `entry@5001` 标给座位 1 的两格砍成了一格，且一处不报错。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.entryexit import entryexit_candidates

    seg = _seg(n=3)
    trans = _entry_dets(tmp_path, order=2, tid=60, kind="exit")
    dets = _dets(tmp_path) + trans
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 60)
    assert len(cand["best_dets"]) == 4, "夹具前提：候选恰有 4 张证据帧"
    labels = {
        "2": {cand["stable_id"]: {"role": "创业者", "seat": 3}},
        # 用户在「就座座位 3」那张图上勾的——与下面这条递补记录无关。
        "cell_exclusions": {"2": {"3": [2]}},
    }

    seen: list[list] = []
    orig_sheet = assemble_mod.build_evidence_sheet

    def spy(paths, boxes, *a, **kw):
        seen.append(list(paths))
        return orig_sheet(paths, boxes, *a, **kw)

    assemble_mod.build_evidence_sheet = spy
    try:
        _run_with_fake_ask_vlm(
            lambda png, ctx: {"evidence_quality": "good", "confidence": 0.9,
                              "outer_hue_class": "蓝"},
            [seg], dets, {2: {8: 1}}, entry_labels=labels)
    finally:
        assemble_mod.build_evidence_sheet = orig_sheet

    # **按路径认出递补那一张**，不靠张数：就座那张本来就只有 3 帧
    # （`_dets` 每条 track 只造 3 帧），拿张数去判会把它误当成"被剔了一格"。
    trans_sheets = [p for p in seen
                    if all("entry_2_60.jpg" in str(x) for x in p)]
    assert trans_sheets, (
        f"没找到递补那张证据图，实得各次送图 {[[str(x)[-14:] for x in p] for p in seen]}")
    assert all(len(p) == 4 for p in trans_sheets), (
        f"递补那张四格拼图要原样送四张——那排框说的是就座那张图，与它无关。"
        f"实得 {[len(p) for p in trans_sheets]}")



def test_cross_segment_presenter_reaches_the_vlm(tmp_path):
    """标成「下一组创业者」的人，打标那一步也要真的给他调 VLM。

    `assemble` 里早有 `cross_segment` / `src_order` 的处理（Task 53 需求六），
    但它调 `entryexit_fill_plan(**[seg]**, ...)`——一次只传**一段**。跨段项
    落在 `plan[目标段号]`，而本轮只读 `plan[seg.order]`；轮到目标段时又只传
    目标段自己，源段的标注根本不会被扫到。于是那段处理代码在真实流水线里
    **一次也执行不到**，跨段的人永远没有着装编码。

    表现有多隐蔽：登记表里有他（registry 用的是 order-1/order/order+1 三段的
    窗口，能看到），核对页里有他的图，最终 CSV 里却没有他这一行——两边对不上，
    一处也不报错。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.entryexit import entryexit_candidates

    src = _seg(order=2, n=1)
    tgt = _seg(order=3, n=2)
    trans = _entry_dets(tmp_path, order=2, tid=70, kind="exit")
    dets = _dets(tmp_path) + _dets(tmp_path, order=3) + trans
    cand = next(c for c in entryexit_candidates(dets, src) if c["track_id"] == 70)
    labels = {"2": {cand["stable_id"]: {
        "role": "下一组创业者", "seat": 2, "n_people": 1, "note": ""}}}

    captured = []

    def fake(sheet_png, context):
        captured.append(context)
        return {"evidence_quality": "good", "confidence": 0.9,
                "outer_hue_class": "蓝"}

    anns = _run_with_fake_ask_vlm(fake, [src, tgt], dets,
                                  {2: {8: 1}, 3: {8: 1}},
                                  entry_labels=labels)
    cross = [a for a in anns if a.seg_order == 3 and a.seat_index == 2]
    assert cross, (
        f"第 3 段座位 2 应由跨段流转补上并打标，实得 "
        f"{[(a.seg_order, a.seat_index, a.source) for a in anns]}")
    assert cross[0].cross_segment == 1 and cross[0].src_seg_order == 2, cross[0]


def test_previous_group_presenter_also_reaches_the_vlm(tmp_path):
    """「上一组创业者」方向同样要打到标——源段在目标段**之后**。

    只测「下一组」那一路是不够的：那时源段恒为 `order-1`，窗口里少了
    `order+1` 也照样过（变异测试实测存活）。而「上一组」的源段是
    `order+1`，窗口砍掉后半截就直接失效——表现依旧是登记表有他、CSV 没他。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.entryexit import entryexit_candidates

    tgt = _seg(order=2, n=2)
    src = _seg(order=3, n=1)
    trans = _entry_dets(tmp_path, order=3, tid=71, kind="entry")
    dets = _dets(tmp_path) + _dets(tmp_path, order=3) + trans
    cand = next(c for c in entryexit_candidates(dets, src) if c["track_id"] == 71)
    labels = {"3": {cand["stable_id"]: {
        "role": "上一组创业者", "seat": 2, "n_people": 1, "note": ""}}}

    anns = _run_with_fake_ask_vlm(
        lambda png, ctx: {"evidence_quality": "good", "confidence": 0.9,
                          "outer_hue_class": "蓝"},
        [tgt, src], dets, {2: {8: 1}, 3: {8: 1}}, entry_labels=labels)
    cross = [a for a in anns if a.seg_order == 2 and a.seat_index == 2]
    assert cross and cross[0].cross_segment == 1, (
        f"第 2 段座位 2 应由第 3 段流转补上并打标，实得 "
        f"{[(a.seg_order, a.seat_index, a.source, a.cross_segment) for a in anns]}")
    assert cross[0].src_seg_order == 3, cross[0]


def test_seated_sheet_tops_up_from_the_labelled_transition_candidate(tmp_path):
    """就座图勾掉格位后，用该座位人工标定的进出场候选补回四张，**且构成文案
    如实说明混了进出场帧**。

    用户 2026-08-25：「如果在自动标的图片中选择了排除某个窗格，增加通过进出场
    的照片对该窗格图片进行补充，以最大化填充 2×2」。

    文案必须跟着改：Task 26 那条纯就座文案写着「下装与整体站姿一律不可见」，
    而补进来的正是站立全身镜头——沿用它等于让模型无视最有用的信息（Task 42
    栽过一次的同一件事）。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.entryexit import entryexit_candidates

    seg = _seg(n=2)
    trans = _entry_dets(tmp_path, order=2, tid=75, kind="exit")
    dets = _dets(tmp_path) + trans
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 75)
    labels = {
        "2": {cand["stable_id"]: {"role": "创业者", "seat": 1}},
        "cell_exclusions": {"2": {"1": [2, 4]}},
    }

    seen: list[list] = []
    orig_sheet = assemble_mod.build_evidence_sheet

    def spy(paths, boxes, *a, **kw):
        seen.append(list(paths))
        return orig_sheet(paths, boxes, *a, **kw)

    captured = []

    def fake(png, ctx):
        captured.append(ctx)
        return {"evidence_quality": "good", "confidence": 0.9,
                "outer_hue_class": "蓝"}

    assemble_mod.build_evidence_sheet = spy
    try:
        _run_with_fake_ask_vlm(fake, [seg], dets, {2: {7: 1, 8: 2}},
                               entry_labels=labels)
    finally:
        assemble_mod.build_evidence_sheet = orig_sheet

    assert 4 in [len(p) for p in seen], (
        f"勾掉两格后应补回四张，实得各次张数 {[len(p) for p in seen]}")
    # 按「另有 N 张…站立镜头」认那条混合文案。不能用「进出场」当关键词——
    # 纯就座那句里写着「不含任何**进出场**画面」，照样命中（实测踩到）。
    mixed = [c for c in captured if "另有" in c["layout"]]
    assert len(mixed) == 1, (
        f"补进了站立镜头的那一位应有且只有一条混合文案，实得 "
        f"{[c['layout'][:40] for c in captured]}")
    m = mixed[0]["layout"]
    assert "共 4 张" in m and "退场站立镜头" in m, m
    assert "一律不可见" not in m, (
        f"不得沿用纯就座那句「下装与站姿一律不可见」——补进来的正是站立全身"
        f"镜头，下装恰恰可见：{m}")


def test_entryexit_seat_sheet_merges_every_labelled_candidate(tmp_path):
    """一个座位在**多条**候选上都被标过时，VLM 要看到全部这些画面。

    用户 2026-08-26 实测文化4组 order 3：「单帧可能只选了一个窗格，而在多帧
    分别选的窗格没有拼在一起，应该根据标注尽可能全的拼成 2×2 四张图」。

    这一路（人工标定的进出场候选定下来的座位）此前压根不补帧——补帧只长在
    就座那一路上。落位用了哪条候选就只送那条的画面，用户在另一帧上亲手标的
    那一格连同它的图一起静默消失，模型看到的证据凭空少一截。

    两条候选各用**不同的 jpg**，这样"合进来了没有"能按路径验，不是只数张数。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.entryexit import entryexit_candidates

    seg = _seg(n=3)
    a = _entry_dets(tmp_path, order=2, tid=60, kind="exit", name="A.jpg")
    b = _entry_dets(tmp_path, order=2, tid=61, kind="entry", cx=1000.0,
                    name="B.jpg")
    dets = _dets(tmp_path) + a + b
    cands = entryexit_candidates(dets, seg)
    ca = next(c for c in cands if c["track_id"] == 60 and c["kind"] == "exit")
    cb = next(c for c in cands if c["track_id"] == 61 and c["kind"] == "entry")
    labels = {"2": {
        ca["stable_id"]: {"role": "工作人员", "seat": None, "n_people": 2,
                          "note": "1 是创业者3",
                          "cells": [{"cell": 1, "role": "创业者", "seat": 3}]},
        cb["stable_id"]: {"role": "创业者", "seat": 3, "n_people": 1},
    }}

    seen: list[list] = []
    orig_sheet = assemble_mod.build_evidence_sheet

    def spy(paths, boxes, *ar, **kw):
        seen.append([str(p) for p in paths])
        return orig_sheet(paths, boxes, *ar, **kw)

    captured = []

    def fake(sheet_png, context):
        captured.append(context)
        return {"evidence_quality": "good", "confidence": 0.9,
                "outer_hue_class": "蓝"}

    assemble_mod.build_evidence_sheet = spy
    try:
        anns = _run_with_fake_ask_vlm(fake, [seg], dets, {2: {8: 1}},
                                      entry_labels=labels)
    finally:
        assemble_mod.build_evidence_sheet = orig_sheet

    by_seat = {an.seat_index: an for an in anns}
    assert by_seat[3].source == "entryexit", (
        f"座位 3 该由人工标定的进出场候选定下，实得 {by_seat[3].source}")

    merged = [p for p in seen
              if any("A.jpg" in x for x in p) and any("B.jpg" in x for x in p)]
    assert merged, (
        f"两条标注的画面必须并进同一张证据图，实得各次送图的来源 "
        f"{[sorted({'A' if 'A.jpg' in x else 'B' for x in p}) for p in seen]}")
    assert len(merged[0]) == 4, (
        f"一格 + 另一条的三张，正好拼满 2×2，实得 {len(merged[0])} 张")

    ctx = next(c for c in captured
               if "站立的进出场镜头" in (c.get("layout") or ""))
    assert "共 4 张" in ctx["layout"], (
        f"构成文案要如实说共 4 张——说少了模型会去找不存在的图，说多了同理，"
        f"实得：{ctx['layout']}")


def test_reassigned_seated_seat_reaches_vlm_under_the_new_number(tmp_path):
    """已定座位改号之后，送进 VLM 的那一行也要挂在**新**座位号下。

    用户 2026-08-29 实测城市2 order 6。登记表改了号而打标这一路没改，表现是
    「核对页显示座位 3、模型产出的却记在座位 1 名下」——衣着编码给了错的人，
    且一处不报错（本项目最忌讳的那一类）。

    这一路自己按性别匹配定最终座位号，与登记表不是同一段代码；两处各判一次
    必然日后走样，所以这条测试钉的是**结果**：那条 track 的行必须是新号。
    """
    seg = _seg(n=3)
    dets = _dets(tmp_path)          # track 7、8 两条就座轨迹
    labels = {"seat_reassign": {"2": {"1": 3}}}
    anns = assemble_annotations([seg], dets, {2: {8: 1, 7: 2}},
                                use_vlm=False, entry_labels=labels)
    by_seat = {a.seat_index: a for a in anns}
    assert 3 in by_seat, f"改号后要有座位 3 这一行，实得 {sorted(by_seat)}"
    assert by_seat[3].pixel.get("track_id") == 8, (
        f"座位 3 该是原座位 1 那条 track（8），实得 "
        f"{by_seat[3].pixel.get('track_id')}")
    assert all(a.seat_index != 1 or a.pixel.get("track_id") != 8
               for a in anns), (
        "同一条 track 不能既留在原座位 1、又出现在座位 3——那是把一个人编码"
        "成两个人")


def test_reassign_onto_an_occupied_seat_drops_the_previous_occupant(tmp_path):
    """改号的目标座位已经有人时：改号那条胜出，原占据者**哪个座位都不去**。

    用户 2026-08-29 选定的规则。相邻两人排反是最常见的错法，但不保证一定是
    对调——只搬他明说的那一个，被顶掉的落回「待标」由他另行指认。

    夹具刻意让目标座位**有人**：上一版只测了搬到空座位，于是「让位」那一句
    删掉也不会有测试变红（实测存活过一次）。
    """
    seg = _seg(n=3)
    dets = _dets(tmp_path)                     # track 7、8 两条就座轨迹
    labels = {"seat_reassign": {"2": {"1": 2}}}   # 座位 1 → 座位 2，而 2 已有人
    anns = assemble_annotations([seg], dets, {2: {7: 1, 8: 2}},
                                use_vlm=False, entry_labels=labels)
    by_seat = {a.seat_index: a for a in anns}
    assert by_seat[2].pixel.get("track_id") == 7, (
        f"座位 2 该被改号来的 track 7 顶掉，实得 "
        f"{by_seat[2].pixel.get('track_id')}")
    assert all(a.pixel.get("track_id") != 8 for a in anns), (
        f"被顶掉的 track 8 不许自己找个别的座位坐下，实得 "
        f"{[(a.seat_index, a.pixel.get('track_id')) for a in anns]}")
    assert by_seat[1].pixel.get("track_id") is None, (
        "原座位 1 改走之后是缺口，必须可见")


def test_reassigned_seat_merges_the_labelled_candidate_into_the_vlm_sheet(tmp_path):
    """改号到一个已有人工标注的座位时，送进 VLM 的证据图要**合并**两边的画面。

    用户 2026-08-29 实测城市2 order 6：「自动标注的座位1改为座位2……应与后边
    人工标注的座位2进行合并」。登记表合并了而这一路没合并，表现是「核对页四张、
    模型看到两张」——本项目最忌讳的「人核对了一张模型没看过的图」，且不报错。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.entryexit import entryexit_candidates

    seg = _seg(n=3)
    trans = _entry_dets(tmp_path, order=2, tid=60, kind="exit", name="T.jpg")
    dets = _dets(tmp_path) + trans          # 就座 track 7、8 + 退场 track 60
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 60)
    labels = {"2": {cand["stable_id"]: {"role": "创业者", "seat": 3,
                                        "n_people": 1}},
              "seat_reassign": {"2": {"1": 3}}}   # 就座座位 1 其实是座位 3

    seen: list[list] = []
    orig = assemble_mod.build_evidence_sheet

    def spy(paths, boxes, *a, **kw):
        seen.append([str(x) for x in paths])
        return orig(paths, boxes, *a, **kw)

    assemble_mod.build_evidence_sheet = spy
    try:
        anns = _run_with_fake_ask_vlm(
            lambda png, ctx: {"evidence_quality": "good", "confidence": 0.9,
                              "outer_hue_class": "蓝"},
            [seg], dets, {2: {8: 1, 7: 2}}, entry_labels=labels)
    finally:
        assemble_mod.build_evidence_sheet = orig

    by_seat = {a.seat_index: a for a in anns}
    assert by_seat[3].pixel.get("track_id") == 8, (
        f"座位 3 该是改号来的 track 8，实得 {by_seat[3].pixel.get('track_id')}")
    merged = [p for p in seen
              if any("T.jpg" in x for x in p) and any("f2.jpg" in x for x in p)]
    assert merged, (
        f"人工标注的退场画面与改号来的就座画面必须并进同一张证据图，"
        f"实得各次送图 {[sorted({'T' if 'T.jpg' in x else 'seated' for x in p}) for p in seen]}")


def test_a_full_seated_sheet_that_yields_cells_is_described_as_mixed(tmp_path):
    """就座四格齐全、另有人工标定的退场画面时：2×2 让出格子，**文案要如实说混合**。

    用户 2026-09-26：多个来源时每个来源轮流出格，就座那四张相近的只留 1-2 张。
    这一改之后，四格就座 + 一条退场标注合成的仍是 4 张（2 就座 + 2 退场）——
    张数**没有变多**。此前 `assemble` 是用「证据图比就座那几张多了几张」来判断
    补没补进站立镜头的（`_n_topup = len(_evidence) - len(_kept_seated)`），多出
    0 张就走纯就座那支，告诉模型「共 4 张就座镜头，不含任何进出场画面，下装与
    站姿一律不可见」——而其中两张恰恰是能看见下装的站立镜头。

    构成必须**按最终那几张各自的 kind 去数**，不能按长度差或位置推断。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.entryexit import entryexit_candidates

    seg = _seg(n=2)
    trans = _entry_dets(tmp_path, order=2, tid=75, kind="exit")
    dets = _dets(tmp_path) + trans
    cand = next(c for c in entryexit_candidates(dets, seg) if c["track_id"] == 75)
    # 注意：**没有**逐格排除——就座那一路四格齐全
    labels = {"2": {cand["stable_id"]: {"role": "创业者", "seat": 1}}}

    seen: list[list] = []
    orig_sheet = assemble_mod.build_evidence_sheet

    def spy(paths, boxes, *a, **kw):
        seen.append(list(paths))
        return orig_sheet(paths, boxes, *a, **kw)

    captured = []

    def fake(png, ctx):
        captured.append(ctx)
        return {"evidence_quality": "good", "confidence": 0.9,
                "outer_hue_class": "蓝"}

    assemble_mod.build_evidence_sheet = spy
    try:
        _run_with_fake_ask_vlm(fake, [seg], dets, {2: {7: 1, 8: 2}},
                               entry_labels=labels)
    finally:
        assemble_mod.build_evidence_sheet = orig_sheet

    mixed = [c for c in captured if "另有" in c["layout"]]
    assert len(mixed) == 1, (
        f"那一位的证据图混了退场镜头，文案必须走混合那一支，实得 "
        f"{[c['layout'][:50] for c in captured]}")
    m = mixed[0]["layout"]
    assert "共 4 张" in m, m
    assert "2 张为该创业者本人的就座镜头" in m and "2 张退场站立镜头" in m, (
        f"应是 2 张就座 + 2 张退场（按最终那几张各自的 kind 数）：{m}")
    assert "一律不可见" not in m, f"不得沿用纯就座文案：{m}"


def test_a_cell_split_seat_sharing_its_sheet_is_not_described_as_one_source(tmp_path):
    """格位切分的座位与别条候选轮流出格后，文案不许再说「本次没有其它可用画面」。

    本条候选四格都是这个人（格 1 = 创业者 3，格 2-4 同格位 1），另有一条进场
    候选也被标成座位 3。多个来源轮流出格之后是 2 + 2，**张数与本条原先的 4 格
    相同**。此前 `assemble` 用「合成之后比本条多了几张」判断并没并别条——多出
    0 张，就沿用本条自己的文案：「是从一张挤了多个人的合图里按编号第 1、2、3、4
    格切出来的……该创业者本次没有其它可用画面」。来源说错了一半。
    """
    import pipeline.assemble as assemble_mod
    from pipeline.entryexit import entryexit_candidates

    seg = _seg(n=3)
    a = _entry_dets(tmp_path, order=2, tid=60, kind="exit", name="A.jpg")
    b = _entry_dets(tmp_path, order=2, tid=61, kind="entry", cx=1000.0,
                    name="B.jpg")
    dets = _dets(tmp_path) + a + b
    cands = entryexit_candidates(dets, seg)
    ca = next(c for c in cands if c["track_id"] == 60 and c["kind"] == "exit")
    cb = next(c for c in cands if c["track_id"] == 61 and c["kind"] == "entry")
    assert len(ca["best_dets"]) == 4, "装置前提：本条要有四格"
    labels = {"2": {
        ca["stable_id"]: {"role": "工作人员", "seat": None, "n_people": 2,
                          "cells": [{"cell": 1, "role": "创业者", "seat": 3},
                                    {"cell": 2, "same_as": 1},
                                    {"cell": 3, "same_as": 1},
                                    {"cell": 4, "same_as": 1}]},
        cb["stable_id"]: {"role": "创业者", "seat": 3, "n_people": 1},
    }}

    seen: list[list] = []
    orig_sheet = assemble_mod.build_evidence_sheet

    def spy(paths, boxes, *ar, **kw):
        seen.append([str(p) for p in paths])
        return orig_sheet(paths, boxes, *ar, **kw)

    captured = []

    def fake(sheet_png, context):
        captured.append(context)
        return {"evidence_quality": "good", "confidence": 0.9,
                "outer_hue_class": "蓝"}

    assemble_mod.build_evidence_sheet = spy
    try:
        _run_with_fake_ask_vlm(fake, [seg], dets, {2: {8: 1}}, entry_labels=labels)
    finally:
        assemble_mod.build_evidence_sheet = orig_sheet

    both = [p for p in seen if any("A.jpg" in x for x in p) and any("B.jpg" in x for x in p)]
    assert both, f"装置前提：座位 3 的证据图应同时含两条候选的画面，实得 {seen}"
    assert len(both[0]) == 4, f"应是 2 + 2 共 4 张，实得 {both[0]}"
    seat3 = [c["layout"] for c in captured if "进场" in c["layout"] and "退场" in c["layout"]]
    assert seat3, f"座位 3 的文案应同时点出进场与退场，实得 {[c['layout'][:60] for c in captured]}"
    t = seat3[0]
    assert "没有其它可用画面" not in t and "切出来" not in t, (
        f"并了别条候选的画面之后，「按编号第 N 格切出来、没有其它画面」就不成立：{t}")
    assert "进场 2 张、退场 2 张" in t, t
