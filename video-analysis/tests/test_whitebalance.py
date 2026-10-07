import numpy as np
import pytest
from pipeline.whitebalance import (
    find_reference_patch, von_kries_gains, apply_gains, correct_image,
)


def make_img(h=1080, w=1440, bg=(60, 50, 45)):
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    return img


def test_finds_bright_neutral_patch_in_lower_half():
    img = make_img()
    img[700:900, 600:900] = (200, 198, 195)      # 近中性高亮块
    ref = find_reference_patch(img)
    assert ref is not None
    r, g, b = ref
    assert 190 < r < 210 and 190 < g < 210 and 185 < b < 205


def test_ignores_saturated_regions():
    img = make_img()
    img[700:900, 600:900] = (240, 20, 20)        # 高饱和红，不能当参照
    assert find_reference_patch(img) is None


def test_ignores_patches_in_upper_half():
    img = make_img()
    img[100:300, 600:900] = (200, 198, 195)      # 高亮但在上半区（投影幕布）
    assert find_reference_patch(img) is None


def test_gains_neutralise_warm_cast():
    # 暖偏色参照物：R 高 B 低
    gains = von_kries_gains((200.0, 150.0, 120.0))
    out = apply_gains(np.full((1, 1, 3), (200, 150, 120), dtype=np.uint8), gains)
    r, g, b = out[0, 0]
    assert abs(int(r) - int(g)) <= 1 and abs(int(g) - int(b)) <= 1


def test_gains_preserve_luminance_scale():
    # 校正后参照物的明度应与原来相当，不应整体变暗或过曝
    gains = von_kries_gains((200.0, 150.0, 120.0))
    out = apply_gains(np.full((1, 1, 3), (200, 150, 120), dtype=np.uint8), gains)
    assert 140 <= int(out[0, 0].mean()) <= 200


def test_correct_image_reports_failure_when_no_reference():
    img = make_img()
    out, ok = correct_image(img)
    assert ok is False
    assert np.array_equal(out, img), "找不到参照物时应原样返回"


def test_correct_image_succeeds_with_reference():
    img = make_img()
    img[700:900, 600:900] = (210, 190, 175)
    out, ok = correct_image(img)
    assert ok is True
    patch = out[700:900, 600:900].reshape(-1, 3).mean(axis=0)
    assert abs(patch[0] - patch[2]) < 6, "校正后参照物应接近中性"
