import pytest

from pipeline.audit import threshold_health, audit_report


# ---------------------------------------------------------------------------
# 1. 阈值落在真空里
# ---------------------------------------------------------------------------

def test_threshold_in_void_is_reported_as_void():
    """两簇样本（0-4 附近与 100-104 附近），阈值 50 落在中间的大空档里。
    空档远大于簇内相邻间隙（1），两侧样本数都 >=3，应判定为真空。"""
    low = [0.0, 1.0, 2.0, 3.0, 4.0]
    high = [100.0, 101.0, 102.0, 103.0, 104.0]
    result = threshold_health(low + high, threshold=50.0, name="TEST_VOID")
    assert result["verdict"] == "真空"
    assert result["n_below"] == 5
    assert result["n_above"] == 5
    assert result["nearest_below"] == 4.0
    assert result["nearest_above"] == 100.0
    assert result["gap_rank"] >= 0.95


# ---------------------------------------------------------------------------
# 2. 阈值落在密集连续区
# ---------------------------------------------------------------------------

def test_threshold_in_dense_region_is_reported_as_dense():
    """均匀连续样本（步长恒为 1），任何阈值切开的那道"缝"都不比其它缝大——
    gap_rank 必须是 0（不是被"小于等于"误判成 1），verdict 必须是密集。"""
    values = [float(i) for i in range(20)]   # 0..19
    result = threshold_health(values, threshold=10.3, name="TEST_DENSE")
    assert result["verdict"] == "密集"
    assert result["gap_rank"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. gap_rank 手算锁定值——防止分位算法看似合理实则算错
# ---------------------------------------------------------------------------

def test_gap_rank_matches_hand_computed_value():
    """values 的相邻间隙依次为 [1,2,3,4,5,6,7,8,20]（10 个值，9 个间隙）：
        0, 1, 3, 6, 10, 15, 21, 28, 36, 56
    阈值取 12，恰好落在 10 与 15 之间，被评估的间隙是下标 4 的那个（值=5）。

    排除自身（下标 4，值 5）后剩下的间隙总体是 [1,2,3,4,6,7,8,20]（8 个）。
    严格小于 5 的有 {1,2,3,4} 共 4 个，gap_rank = 4/8 = 0.5。

    若实现错误地把"自身"那个间隙也算进总体（自指偏差，population 变成 9
    个而不是 8 个），算出的会是 5/9 ≈ 0.5556，与手算值不同——这条测试专门
    锁住"排除自身"这一步没有被漏掉。
    """
    values = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 56.0]
    # 相邻间隙核对：1,2,3,4,5,6,7,8,20
    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    assert gaps == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 20.0]

    result = threshold_health(values, threshold=12.0, name="TEST_HAND")
    assert result["n"] == 10
    assert result["n_below"] == 5
    assert result["n_above"] == 5
    assert result["nearest_below"] == 10.0
    assert result["nearest_above"] == 15.0
    assert result["gap"] == pytest.approx(5.0)
    assert result["gap_rank"] == pytest.approx(0.5)
    assert result["verdict"] == "密集"   # 0.5 < 0.80


# ---------------------------------------------------------------------------
# 4. 样本不足时不给自信结论
# ---------------------------------------------------------------------------

def test_small_sample_is_reported_as_insufficient():
    """即便凑巧出现一道大间隙，样本 < 10 时也不能宣称真空/稀疏/密集。"""
    values = [0.0, 1.0, 2.0, 100.0, 101.0]   # 5 个样本，中间有个大间隙
    result = threshold_health(values, threshold=50.0, name="TEST_SMALL")
    assert result["verdict"] == "样本不足"
    assert result["n"] == 5


# ---------------------------------------------------------------------------
# 5. 阈值在全部样本之外
# ---------------------------------------------------------------------------

def test_all_samples_on_one_side():
    """10 个样本全部小于阈值：不能崩，且必须如实报告上方样本数为 0。"""
    values = [float(i) for i in range(10)]   # 0..9
    result = threshold_health(values, threshold=100.0, name="TEST_ONE_SIDE")
    assert result["n_above"] == 0
    assert result["n_below"] == 10
    assert result["nearest_above"] is None
    assert result["nearest_below"] == 9.0
    assert result["verdict"] != "真空", "一侧样本为 0 时不该判定为真空（两侧都要有足够样本）"
    # 反向：全部大于阈值
    result2 = threshold_health(values, threshold=-10.0, name="TEST_ONE_SIDE_2")
    assert result2["n_below"] == 0
    assert result2["n_above"] == 10
    assert result2["nearest_below"] is None


# ---------------------------------------------------------------------------
# 6. audit_report 密集的排最前
# ---------------------------------------------------------------------------

def test_audit_report_sorts_dense_first():
    checks = [
        {"name": "A_VOID", "verdict": "真空", "gap_rank": 0.99, "detail": "A 真空"},
        {"name": "B_DENSE", "verdict": "密集", "gap_rank": 0.1, "detail": "B 密集"},
        {"name": "C_SPARSE", "verdict": "稀疏", "gap_rank": 0.85, "detail": "C 稀疏"},
        {"name": "D_INSUFFICIENT", "verdict": "样本不足", "gap_rank": 0.0, "detail": "D 样本不足"},
    ]
    report = audit_report(checks)
    lines = report.splitlines()
    assert len(lines) == 4
    names_in_order = [line for line in lines]
    idx_dense = next(i for i, l in enumerate(names_in_order) if "B_DENSE" in l)
    idx_sparse = next(i for i, l in enumerate(names_in_order) if "C_SPARSE" in l)
    idx_void = next(i for i, l in enumerate(names_in_order) if "A_VOID" in l)
    idx_insufficient = next(i for i, l in enumerate(names_in_order) if "D_INSUFFICIENT" in l)
    assert idx_dense < idx_sparse < idx_void < idx_insufficient, report
