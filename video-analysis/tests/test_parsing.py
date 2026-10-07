import numpy as np
import pytest
from pipeline import config
from pipeline.parsing import GarmentMasks, split_by_kmeans, inner_quality


def test_inner_quality_thresholds():
    assert inner_quality(600) == "good"
    assert inner_quality(300) == "fair"
    assert inner_quality(100) == "poor"
    assert inner_quality(config.INNER_QUALITY_GOOD) == "good"
    assert inner_quality(config.INNER_QUALITY_FAIR) == "fair"


def test_split_by_kmeans_assigns_larger_area_to_outer():
    # 构造：深蓝占 80%，白色细条占 20% 且在上方（模拟领口）
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (20, 30, 80)
    crop[0:20, 40:60] = (240, 240, 240)
    mask = np.ones((100, 100), dtype=bool)

    g = split_by_kmeans(mask, crop)
    assert g.outer_px > g.inner_px
    outer_mean = crop[g.outer].reshape(-1, 3).mean(axis=0)
    assert outer_mean[2] > outer_mean[0], "外层应是深蓝"


def test_split_by_kmeans_returns_none_inner_for_single_colour():
    crop = np.full((50, 50, 3), (30, 30, 30), dtype=np.uint8)
    mask = np.ones((50, 50), dtype=bool)
    g = split_by_kmeans(mask, crop)
    assert g.inner is None or g.inner_px == 0


def test_split_by_kmeans_handles_empty_mask():
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    g = split_by_kmeans(mask, crop)
    assert g.outer_px == 0 and g.inner_px == 0


def test_parse_garments_marks_fallback_when_model_fails(monkeypatch):
    # 模型不可用时必须退回整图，但要在 source 上留下痕迹供下游识别
    import pipeline.parsing as P
    def boom(_):
        raise RuntimeError("模型加载失败")
    monkeypatch.setattr(P, "_upper_mask", boom)
    crop = np.full((60, 60, 3), (20, 30, 90), dtype=np.uint8)
    g = P.parse_garments(crop)
    assert g.source == "kmeans_fallback"
    assert g.model_error and "RuntimeError" in g.model_error
    assert g.outer_px > 0        # 仍有兜底结果


def test_parse_garments_keeps_small_mask_instead_of_whole_crop(monkeypatch):
    # 模型成功但只分出很小一块（真实场景：创业者大半在画外）：
    # 必须保留这一小块，不得退回整图——整图会把人脸椅背算进衣服颜色
    import pipeline.parsing as P
    small = np.zeros((60, 60), dtype=bool)
    small[0:6, 0:6] = True                       # 36 像素，远小于旧阈值 200
    monkeypatch.setattr(P, "_upper_mask", lambda _: small)
    crop = np.full((60, 60, 3), (20, 30, 90), dtype=np.uint8)
    g = P.parse_garments(crop)
    assert g.source == "model"
    assert g.outer_px <= 36, f"应保留小 mask 而非退回整图，实得 {g.outer_px}"


def test_parse_garments_reports_model_source_on_empty_mask(monkeypatch):
    # 模型跑了但一个上衣像素都没找到：是事实不是错误
    import pipeline.parsing as P
    monkeypatch.setattr(P, "_upper_mask", lambda _: np.zeros((60, 60), dtype=bool))
    crop = np.full((60, 60, 3), (20, 30, 90), dtype=np.uint8)
    g = P.parse_garments(crop)
    assert g.source == "model" and g.outer_px == 0 and g.model_error is None


@pytest.mark.slow
@pytest.mark.skipif(
    not (config.WORK / "frames_index_pilot.json").exists(),
    reason="该断言依赖某一组特有的 pilot 抽帧产物；换视频组后不存在，跳过而非硬失败",
)
def test_parse_garments_on_real_crop():
    from PIL import Image
    from pipeline.frames import load_index
    from pipeline.parsing import parse_garments
    f = [x for x in load_index(config.WORK / "frames_index_pilot.json")
         if x.seg_order == 2 and x.kind == "seated"][0]
    img = np.asarray(Image.open(f.path).convert("RGB"))
    crop = img[200:700, 1150:1440]          # 右侧创业者区域
    g = parse_garments(crop)
    assert g.outer_px > 0, "未能在真实截图上分割出上衣"
