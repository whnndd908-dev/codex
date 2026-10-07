import numpy as np
import pytest
from pipeline.colormath import (
    dominant_color, rgb_to_hsv_deg, rgb_to_lab, mean_hue_deg, hue_spread_deg,
)


def test_dominant_color_picks_largest_cluster_not_mean():
    # 深蓝西装 600 像素 + 白衬衫 400 像素。
    # 均值会得到 (108,114,144) 这个衣服上不存在的灰蓝色；
    # 比例不能太悬殊（900:100 时均值仍接近深蓝，测不出区别）。
    navy = np.tile(np.array([[20, 30, 80]], dtype=np.uint8), (600, 1))
    white = np.tile(np.array([[240, 240, 240]], dtype=np.uint8), (400, 1))
    px = np.vstack([navy, white])
    r, g, b = dominant_color(px, k=3)
    assert b > r and b > g, "应返回深蓝而非灰"
    assert r < 60 and g < 70
    # 直接锁住结果就是最大簇的质心，均值实现无法通过
    assert (r, g, b) == pytest.approx((20.0, 30.0, 80.0), abs=5.0)


def test_dominant_color_handles_single_colour_input():
    px = np.tile(np.array([[10, 10, 10]], dtype=np.uint8), (50, 1))
    r, g, b = dominant_color(px, k=3)
    assert (round(r), round(g), round(b)) == (10, 10, 10)


def test_rgb_to_hsv_pure_red():
    h, s, v = rgb_to_hsv_deg((255, 0, 0))
    assert h == pytest.approx(0.0)
    assert s == pytest.approx(100.0)
    assert v == pytest.approx(100.0)


def test_rgb_to_hsv_pure_blue():
    h, s, v = rgb_to_hsv_deg((0, 0, 255))
    assert h == pytest.approx(240.0)


def test_rgb_to_hsv_black_has_zero_value():
    h, s, v = rgb_to_hsv_deg((0, 0, 0))
    assert v == pytest.approx(0.0)


def test_rgb_to_lab_white():
    L, a, b = rgb_to_lab((255, 255, 255))
    assert L == pytest.approx(100.0, abs=0.5)
    assert abs(a) < 1.0 and abs(b) < 1.0


def test_rgb_to_lab_black():
    L, a, b = rgb_to_lab((0, 0, 0))
    assert L == pytest.approx(0.0, abs=0.5)


def test_mean_hue_wraps_around_zero():
    # 350 度与 10 度的圆周均值是 0 度，不是 180 度
    assert mean_hue_deg([350.0, 10.0]) == pytest.approx(0.0, abs=0.5)


def test_hue_spread_small_for_tight_cluster():
    assert hue_spread_deg([200.0, 202.0, 198.0]) < 3.0


def test_hue_spread_large_for_scattered_hues():
    assert hue_spread_deg([10.0, 130.0, 250.0]) > 50.0
