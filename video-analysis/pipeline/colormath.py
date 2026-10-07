"""颜色计算的全部纯函数：主色提取、色彩空间转换、圆周统计。"""
from __future__ import annotations

import colorsys
import math

import numpy as np
from sklearn.cluster import KMeans

from pipeline import config


def dominant_color(pixels_rgb: np.ndarray, k: int = config.KMEANS_K) -> tuple[float, float, float]:
    """取最大簇的质心作为主色。

    不用均值：深蓝西装配白衬衫取均值会得到一个衣服上不存在的灰色。
    """
    px = np.asarray(pixels_rgb, dtype=np.float64).reshape(-1, 3)
    if len(px) == 0:
        raise ValueError("像素为空")
    k_eff = int(min(k, len(np.unique(px, axis=0))))
    if k_eff <= 1:
        return tuple(px.mean(axis=0).tolist())
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit(px)
    counts = np.bincount(km.labels_, minlength=k_eff)
    return tuple(km.cluster_centers_[int(np.argmax(counts))].tolist())


def rgb_to_hsv_deg(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """返回 (色相 0–360 度, 饱和度 0–100, 明度 0–100)。"""
    r, g, b = [max(0.0, min(255.0, float(c))) / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360.0, s * 100.0, v * 100.0)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """sRGB → CIELAB，D65 白点。"""
    r, g, b = [_srgb_to_linear(max(0.0, min(255.0, float(c))) / 255.0) for c in rgb]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    xn, yn, zn = 0.95047, 1.00000, 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def mean_hue_deg(hues: list[float]) -> float:
    """色相的圆周均值。350 度与 10 度的均值是 0 度，不是 180 度。"""
    if not hues:
        raise ValueError("色相列表为空")
    rad = [math.radians(h) for h in hues]
    x = sum(math.cos(r) for r in rad) / len(rad)
    y = sum(math.sin(r) for r in rad) / len(rad)
    result = math.degrees(math.atan2(y, x)) % 360.0
    # IEEE-754 浮点偏差守卫：atan2 在接近 0 的浮点数上会返回很小的负数，
    # 导致 % 360 返回接近 360.0 的值。这不是业务规则模糊匹配，而是数值稳定性修复。
    return 0.0 if result >= 360.0 - 1e-10 else result


def hue_spread_deg(hues: list[float]) -> float:
    """色相的圆周标准差，用作 color_stability。"""
    if len(hues) < 2:
        return 0.0
    rad = [math.radians(h) for h in hues]
    x = sum(math.cos(r) for r in rad) / len(rad)
    y = sum(math.sin(r) for r in rad) / len(rad)
    R = math.hypot(x, y)
    R = min(max(R, 1e-12), 1.0)
    return math.degrees(math.sqrt(-2.0 * math.log(R)))


# --- 以下追加到 pipeline/colormath.py 末尾 ---

_SIX = {
    "红": "红", "粉": "红",
    "橙": "黄", "黄": "黄", "棕": "黄",
    "绿": "绿", "蓝": "蓝", "紫": "紫",
    "黑": "无彩色", "白": "无彩色", "灰": "无彩色",
}
_WARM_COOL = {
    "红": "暖", "橙": "暖", "黄": "暖", "棕": "暖", "粉": "暖",
    "绿": "冷", "蓝": "冷", "紫": "冷",
    "黑": "中性", "白": "中性", "灰": "中性",
}
ACHROMATIC = ("黑", "白", "灰")


def _in_arc(h: float, lo: float, hi: float) -> bool:
    """角度是否落在 [lo, hi) 弧内，支持跨 0 度。"""
    h %= 360.0
    lo %= 360.0
    hi %= 360.0
    return lo <= h < hi if lo < hi else (h >= lo or h < hi)


def classify_hue_11(h: float, s: float, v: float) -> str:
    """基本颜色词 11 类。

    判定顺序不可调换：无彩色先分流，否则深色西装会被塞进随机色相箱；
    棕与粉是色相加明度的组合类，须在纯色相箱之前判定。
    """
    if v < config.ACHROMATIC_BLACK_VAL:
        return "黑"
    if s < config.ACHROMATIC_SAT:
        return "白" if v > config.ACHROMATIC_WHITE_VAL else "灰"

    if (_in_arc(h, *config.BROWN_HUE_RANGE)
            and s >= config.BROWN_MIN_SAT and v < config.BROWN_MAX_VAL):
        return "棕"

    if (_in_arc(h, *config.PINK_HUE_RANGE)
            and s < config.PINK_MAX_SAT and v > config.PINK_MIN_VAL):
        return "粉"

    for name, lo, hi in config.HUE_BINS:
        if _in_arc(h, lo, hi):
            return name
    return "灰"   # 理论上不可达；色相箱已覆盖整圈


def classify_hue_6(cls11: str) -> str:
    return _SIX[cls11]


def classify_warm_cool(cls11: str) -> str:
    return _WARM_COOL[cls11]


def classify_lightness(v: float) -> str:
    if v < config.ACHROMATIC_BLACK_VAL:
        return "深"
    return "浅" if v > config.ACHROMATIC_WHITE_VAL else "中"


def classify_chroma(s: float) -> str:
    if s < config.ACHROMATIC_SAT:
        return "无彩色"
    return "高饱和" if s >= 50.0 else "低饱和"


def hue_margin_deg(h: float) -> float:
    """到最近色相箱边界的角度。越小越可能因光照噪声翻箱。"""
    h %= 360.0
    edges = {lo % 360.0 for _n, lo, _hi in config.HUE_BINS}
    edges |= {hi % 360.0 for _n, _lo, hi in config.HUE_BINS}
    return min(min(abs(h - e), 360.0 - abs(h - e)) for e in edges)


def is_hue_valid(s: float, v: float) -> bool:
    """连续色相是否可解释。深色与低饱和衣物的色相是噪声。"""
    return (s >= config.HUE_VALID_MIN_SAT
            and config.HUE_VALID_VAL_MIN <= v <= config.HUE_VALID_VAL_MAX)


def choose_granularity(
    cls11: str, margin: float, agree: bool, stability: float
) -> tuple[str, str]:
    """粒度阶梯：不确定时退到更粗粒度，而非标缺失。

    用户优先级为「准确度 > 样本保留 > 分辨率」，故本函数对有分类结果的样本
    永不返回「无」。
    """
    if cls11 in ACHROMATIC:
        # 无彩色不经色相箱，边界距离与色相稳定性对它没有意义
        return ("11类", cls11)

    unstable = stability > 45.0
    borderline = margin < config.HUE_BORDERLINE_MARGIN

    if agree and not borderline and not unstable:
        return ("11类", cls11)
    if agree and not unstable:
        return ("6类", classify_hue_6(cls11))
    return ("二分", classify_warm_cool(cls11))
