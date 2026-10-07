from pathlib import Path
from pipeline import config


def test_input_files_exist_and_are_readonly_targets():
    assert config.VIDEO.exists(), "视频文件不在预期位置"
    assert config.XLSX.exists()
    assert config.DTA.exists()


def test_expected_ground_truth_constants():
    """闸门常量必须自洽，但不写死某一组的具体数值。

    原先这里断言 16 段 / 34 人——那是「城市2-4.26」组的值，一切到别的
    视频组就必红。ground truth 本来就是逐组不同的，把它写死等于每换一组
    就要改测试，而改测试正是最容易把真闸门一起改绿的地方。
    改为断言组无关的不变量：段数必须为正；人数为 0 表示「待回填」
    （load_segments 会跳过总数断言并打印实际值），非 0 时必须至少等于段数
    ——每个项目至少 1 位创业者。
    """
    assert config.EXPECTED_SEGMENT_COUNT > 0
    assert config.EXPECTED_PRESENTER_TOTAL >= 0
    if config.EXPECTED_PRESENTER_TOTAL:
        assert config.EXPECTED_PRESENTER_TOTAL >= config.EXPECTED_SEGMENT_COUNT


def test_hue_bins_cover_full_circle_without_overlap():
    # 六个色相箱首尾相接覆盖 360 度，红箱跨 0 度
    spans = 0.0
    for _name, lo, hi in config.HUE_BINS:
        spans += (hi - lo) % 360
    assert abs(spans - 360.0) < 1e-6


def test_hue_bin_names_are_the_six_chromatic_classes():
    names = [n for n, _, _ in config.HUE_BINS]
    assert names == ["红", "橙", "黄", "绿", "蓝", "紫"]


def test_staff_net_is_disabled_by_default():
    """安全网默认关闭，实测精确率 0/5（Ruling R51）。

    CLIP 整体裁剪 embedding 在本场地不携带身份信息：同一段内挨着坐的
    两位不同创业者距离仅 0.0236，而跨段不同人最大也只有 0.1827，两个
    分布完全交叠——不存在可用阈值。开着它会把每一位真实创业者都标红，
    训练复核者忽略标记，比没有标记更糟。
    """
    assert config.STAFF_NET_ENABLED is False


# ---------------------------------------------------------------------------
# Task 40：清掉死参数——landing.py 三常量与 SEAT_DEPTH_GAP_RATIO 已证伪/
# 恒不触发，删除；PAIR_*/STAFF_*/色彩常数仍在生效，不能被清理时手滑带走。
# ---------------------------------------------------------------------------

def test_removed_constants_are_gone():
    """四个已证伪/恒不触发的常量必须从 config 里彻底删除，不能只是改成
    不生效的值——那种"改成不触发的魔数"正是本次要清掉的坏味道本身。"""
    removed = [
        "LANDING_MAX_RATIO",
        "LANDING_MIN_FRAMES",
        "LANDING_DEDUP_RATIO",
        "SEAT_DEPTH_GAP_RATIO",
    ]
    for name in removed:
        assert not hasattr(config, name), f"{name} 应已删除，但仍存在于 config"


def test_pair_and_staff_and_color_constants_survive():
    """护栏：清理死参数时最容易手滑带走的是仍在生效的常量——
    `pairing.py` 每段仍调用 PAIR_*（run_all.py 阶段⑤），STAFF_NET_ENABLED
    是用户明确要求保留的开关，色彩常数是用户明确要求保留的能力（像素轨
    默认关闭不等于删掉）。这些一个都不能少。"""
    pair_names = [
        "PAIR_MIN_MOVE_PX", "PAIR_MIN_TRACK_FRAMES", "PAIR_MIN_H_RATIO",
        "PAIR_COLOR_WEIGHT", "PAIR_POS_WEIGHT", "PAIR_POSTURE_WEIGHT",
        "PAIR_STANDING_H_RATIO", "PAIR_DELTA_E_SCALE", "PAIR_MAX_COST",
        "PAIR_MAX_POS_DIST",
    ]
    staff_names = ["STAFF_CLUSTER_DIST", "STAFF_MIN_SEGMENTS", "STAFF_NET_ENABLED"]
    color_names = [
        "HUE_VALID_MIN_SAT", "HUE_VALID_VAL_MIN", "HUE_VALID_VAL_MAX",
        "ACHROMATIC_BLACK_VAL", "ACHROMATIC_SAT", "ACHROMATIC_WHITE_VAL",
        "BROWN_HUE_RANGE", "BROWN_MIN_SAT", "BROWN_MAX_VAL",
        "PINK_HUE_RANGE", "PINK_MAX_SAT", "PINK_MIN_VAL",
        "WB_REF_MAX_SAT", "WB_REF_MIN_VAL", "KMEANS_K",
    ]
    for name in pair_names + staff_names + color_names:
        assert hasattr(config, name), f"{name} 不该被清理掉——它仍在生效"
    assert len(pair_names) == 10, "PAIR_* 应有 10 个，brief 明确要求全部保留"
