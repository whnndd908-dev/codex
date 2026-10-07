import pytest
from pipeline import config
from pipeline.colormath import (
    classify_hue_11, classify_hue_6, classify_warm_cool, classify_lightness,
    classify_chroma, hue_margin_deg, is_hue_valid, choose_granularity,
)


# --- 无彩色必须先分流，优先于任何色相箱 ---

def test_very_dark_is_black_regardless_of_hue():
    # 深色西装：色相是噪声，必须归黑而非某个随机色相箱
    assert classify_hue_11(h=137.0, s=60.0, v=10.0) == "黑"
    assert classify_hue_11(h=300.0, s=90.0, v=5.0) == "黑"


def test_bright_desaturated_is_white():
    assert classify_hue_11(h=40.0, s=8.0, v=90.0) == "白"


def test_mid_desaturated_is_grey():
    assert classify_hue_11(h=200.0, s=5.0, v=50.0) == "灰"


# --- 组合类：棕是低明度的橙，粉是高明度低饱和的红 ---

def test_dark_orange_is_brown_not_orange():
    assert classify_hue_11(h=25.0, s=50.0, v=40.0) == "棕"


def test_bright_orange_stays_orange():
    assert classify_hue_11(h=25.0, s=80.0, v=90.0) == "橙"


def test_pale_red_is_pink_not_red():
    assert classify_hue_11(h=355.0, s=30.0, v=85.0) == "粉"


def test_saturated_red_stays_red():
    assert classify_hue_11(h=355.0, s=85.0, v=80.0) == "红"


# --- 无彩色优先级 vs 棕/粉组合类：判定顺序的变异防护 ---
# 这三条专门覆盖「色相落在棕/粉范围内、但同时满足无彩色条件」的输入。
# 若把棕/粉判定提到无彩色分流之前，三条全部会失败。

def test_dark_garment_with_brown_range_hue_is_black_not_brown():
    # 深色西装的典型形态：色相在棕范围(10-45°)内，但明度极低。
    # 深色物体的色相本身是噪声，必须先被无彩色规则截住。
    assert classify_hue_11(h=25.0, s=60.0, v=10.0) == "黑"


def test_pale_garment_with_pink_range_hue_is_white_not_pink():
    # 色相在粉范围(345-10°)内，但饱和度低到已属无彩色。
    assert classify_hue_11(h=350.0, s=10.0, v=90.0) == "白"


def test_mid_grey_with_brown_range_hue_is_grey_not_orange():
    # 色相在棕范围内、饱和度低于无彩色阈值、明度居中。
    # 若无彩色分流被后置，会先落入橙箱。
    assert classify_hue_11(h=25.0, s=10.0, v=40.0) == "灰"


# --- 色相箱 ---

@pytest.mark.parametrize("h,expected", [
    (0.0, "红"), (355.0, "红"),
    (30.0, "橙"), (60.0, "黄"),
    (120.0, "绿"), (200.0, "蓝"),
    (240.0, "蓝"), (300.0, "紫"),
])
def test_hue_bins(h, expected):
    assert classify_hue_11(h=h, s=80.0, v=80.0) == expected


@pytest.mark.parametrize("h,expected", [
    (10.0, "橙"), (45.0, "黄"), (70.0, "绿"),
    (165.0, "蓝"), (260.0, "紫"), (345.0, "红"),
])
def test_hue_bin_boundaries_are_half_open_lower_inclusive(h, expected):
    # 六个箱边界恰好取值时应归入上界所在的那个箱（半开区间 [lo, hi)）。
    # 用高饱和高明度避开无彩色与棕/粉规则，确保只考验色相箱本身。
    assert classify_hue_11(h=h, s=80.0, v=80.0) == expected


def test_all_eleven_classes_are_reachable():
    got = {
        classify_hue_11(0, 90, 80), classify_hue_11(30, 90, 90),
        classify_hue_11(60, 90, 90), classify_hue_11(120, 90, 80),
        classify_hue_11(200, 90, 80), classify_hue_11(300, 90, 80),
        classify_hue_11(355, 30, 85), classify_hue_11(25, 50, 40),
        classify_hue_11(0, 0, 5), classify_hue_11(0, 0, 90), classify_hue_11(0, 0, 50),
    }
    assert got == {"红","橙","黄","绿","蓝","紫","粉","棕","黑","白","灰"}


# --- 粗化 ---

@pytest.mark.parametrize("c11,c6", [
    ("红", "红"), ("粉", "红"), ("橙", "黄"), ("黄", "黄"), ("棕", "黄"),
    ("绿", "绿"), ("蓝", "蓝"), ("紫", "紫"),
    ("黑", "无彩色"), ("白", "无彩色"), ("灰", "无彩色"),
])
def test_six_class_mapping(c11, c6):
    assert classify_hue_6(c11) == c6


@pytest.mark.parametrize("c11,wc", [
    ("红", "暖"), ("橙", "暖"), ("黄", "暖"), ("棕", "暖"), ("粉", "暖"),
    ("绿", "冷"), ("蓝", "冷"), ("紫", "冷"),
    ("黑", "中性"), ("白", "中性"), ("灰", "中性"),
])
def test_warm_cool_mapping(c11, wc):
    assert classify_warm_cool(c11) == wc


def test_lightness_classes():
    assert classify_lightness(10.0) == "深"
    assert classify_lightness(50.0) == "中"
    assert classify_lightness(90.0) == "浅"


def test_chroma_classes():
    assert classify_chroma(5.0) == "无彩色"
    assert classify_chroma(30.0) == "低饱和"
    assert classify_chroma(80.0) == "高饱和"


# --- 边界距离 ---

def test_margin_is_zero_at_bin_boundary():
    assert hue_margin_deg(45.0) == pytest.approx(0.0, abs=0.01)


def test_margin_is_large_at_bin_centre():
    # 绿箱 70–165，中心 117.5，到边界 47.5 度
    assert hue_margin_deg(117.5) == pytest.approx(47.5, abs=0.5)


def test_margin_handles_wraparound_red_bin():
    # 红箱 345–10 跨 0 度，中心约 357.5
    assert hue_margin_deg(357.5) == pytest.approx(12.5, abs=0.5)


# --- hue_valid ---

def test_hue_invalid_for_low_saturation():
    assert is_hue_valid(s=10.0, v=50.0) is False


def test_hue_invalid_for_near_black_and_near_white():
    assert is_hue_valid(s=80.0, v=5.0) is False
    assert is_hue_valid(s=80.0, v=98.0) is False


def test_hue_valid_for_normal_garment():
    assert is_hue_valid(s=60.0, v=45.0) is True


# --- 粒度阶梯：不确定时降粒度，不丢样本 ---

def test_confident_sample_keeps_finest_granularity():
    g, val = choose_granularity("蓝", margin=40.0, agree=True, stability=5.0)
    assert (g, val) == ("11类", "蓝")


def test_borderline_sample_falls_back_to_six_class():
    # 落在蓝紫边界：11 类不可靠，但 6 类的「蓝」仍确定
    g, val = choose_granularity("蓝", margin=5.0, agree=True, stability=5.0)
    assert (g, val) == ("6类", "蓝")


def test_disagreement_falls_back_to_binary():
    g, val = choose_granularity("蓝", margin=5.0, agree=False, stability=5.0)
    assert g == "二分"
    assert val in ("暖", "冷", "中性")


def test_unstable_colour_falls_back_to_binary():
    g, val = choose_granularity("蓝", margin=40.0, agree=True, stability=60.0)
    assert g == "二分"


def test_achromatic_is_always_confident_at_finest_granularity():
    # 黑白灰不经色相箱，边界距离与色相稳定性对它们没有意义，应始终保持 11 类
    for c in ("黑", "白", "灰"):
        g, val = choose_granularity(c, margin=0.0, agree=True, stability=80.0)
        assert (g, val) == ("11类", c)


def test_granularity_never_returns_missing_for_valid_class():
    # 「准确度 > 样本保留 > 分辨率」：只要有分类结果就不该丢样本
    g, val = choose_granularity("红", margin=0.0, agree=False, stability=90.0)
    assert g != "无"
    assert val
