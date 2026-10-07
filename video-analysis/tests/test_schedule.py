from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from pipeline import config
from pipeline.schedule import (
    load_segments, save_segments, load_segments_json, ScheduleError, Segment, Presenter,
)


def _write_xlsx(path: Path, sheet: str, rows: list[tuple[str, int, int]]) -> None:
    """造一张最小时间表 xlsx：表头占第 1 行，数据从第 2 行起，列为 company/start/end。"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(["company", "start", "end"])
    for company, start, end in rows:
        ws.append([company, start, end])
    wb.save(path)


def _write_dta(path: Path, panel: str, rows: list[dict]) -> None:
    """造一份最小人工标注 dta。

    每个 row dict 至少含 firmName/order/presenterNum；可选 female（长度可故意与
    presenterNum 不符，用来构造"非空 prstNFemale 个数不符"场景），列表之外的
    prstNFemale/preprstN/qaprstN 一律留 NaN，与真实 dta 的稀疏模式一致。
    """
    records = []
    for r in rows:
        presenter_num = r["presenterNum"]
        female = r.get("female", [0] * presenter_num)
        rec = {
            "firmName": r["firmName"],
            "panelName": panel,
            "order": float(r["order"]),
            "presenterNum": float(presenter_num),
        }
        for k in range(1, 5):
            rec[f"prst{k}Female"] = float(female[k - 1]) if k <= len(female) else float("nan")
            rec[f"preprst{k}"] = 1.0 if k <= presenter_num else float("nan")
            rec[f"qaprst{k}"] = 1.0 if k <= presenter_num else float("nan")
        # 人工补录的实际进/退场时刻。只有传了才写这一列——「这两列根本不存在」
        # 是真实存在的一种输入（除城市1组外各组都还没补），必须能造得出来。
        for key in ("entryTime", "exitTime"):
            if key in r:
                rec[key] = float("nan") if r[key] is None else float(r[key])
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_stata(path, write_index=False, version=118)


@pytest.fixture(scope="module")
def segments():
    return load_segments()


def test_segment_count(segments):
    assert len(segments) == config.EXPECTED_SEGMENT_COUNT


def test_total_presenters(segments):
    """总人数须与 config 的期望值一致；期望值为 0 时视为「待回填」而跳过。

    原先写死 31（第二组的值），一切视频组即红。ground truth 本就逐组不同，
    写死等于每换一组都要改测试，而改测试正是最容易把真闸门一起改绿的地方。
    """
    total = sum(s.presenter_num for s in segments)
    assert total > 0
    if config.EXPECTED_PRESENTER_TOTAL:
        assert total == config.EXPECTED_PRESENTER_TOTAL


def test_orders_are_1_to_n_in_sequence(segments):
    """order 必须是 1..N 连续无缺口（N = 本组企业数），不写死某一组的 N。"""
    assert [s.order for s in segments] == list(range(1, len(segments) + 1))
    assert len(segments) == config.EXPECTED_SEGMENT_COUNT


def test_timestamps_strictly_increasing_and_non_overlapping(segments):
    for s in segments:
        assert s.start < s.end, f"order {s.order} 起止时间反了"
    for a, b in zip(segments, segments[1:]):
        assert a.end <= b.start, f"order {a.order} 与 {b.order} 时间重叠"


def test_first_and_last_rows_are_well_formed(segments):
    """首尾行的结构性约束，不写死某一组的企业名与秒数。

    真正要守住的是：企业名非空、首行不早于 0 秒、末行结束时间是全组最大、
    且不超过视频时长（后者由 test_frames 的时长断言把关）。
    """
    assert segments[0].company.strip()
    assert segments[-1].company.strip()
    assert segments[0].start >= 0
    assert segments[-1].end == max(s.end for s in segments)


def test_presenter_list_length_matches_presenter_num(segments):
    for s in segments:
        assert len(s.presenters) == s.presenter_num


def test_seat_indices_are_1_based_and_contiguous(segments):
    for s in segments:
        assert [p.seat_index for p in s.presenters] == list(range(1, s.presenter_num + 1))


def test_presentation_speaker_is_identified_by_the_flag_not_by_seat_one(segments):
    """主讲人由 `speaks_pre` 标识，**不是**座位 1。

    原先这里断言「座位 1 恒为主讲」（spec 3.1，从最初一组的 16 行归纳而来）。
    全库实测它不成立：1101 条记录里 **76 条**（6.9%）的 `preprst1 != 1`，
    涉及电子信息29组、装备制造4组等多个组；用户 2026-08-29 换到城市精细化
    管理1组时撞上其中一条——共友时代（北京）科技股份有限公司
    `preprst = [0, 1, 0]`，**座位 2 才是主讲**。

    更弱的「每段至少一人路演发言」同样不成立（1101 条里 27 条为 0，其中数条
    连 `presenterNum` 都是缺失）。所以这里不对「谁发言、几人发言」作任何断
    言——那是数据事实，不是本流水线能保证的性质。

    真正要钉住的是**读法**：逐座位读 `speaks_pre`，不许拿座位号去推主讲人。
    因此断言结构：每人一个标志、取值 0/1、条数与 presenter_num 一致。按座位
    号推的写法会在这里露馅——它根本不看这几个标志。
    """
    for s in segments:
        pre = [p.speaks_pre for p in s.presenters]
        qa = [p.speaks_qa for p in s.presenters]
        assert len(pre) == len(qa) == s.presenter_num, (
            f"order {s.order} 的发言标志条数与人数不符：{pre} / {qa}")
        assert set(pre) <= {0, 1} and set(qa) <= {0, 1}, (
            f"order {s.order} 的发言标志含非 0/1 值：{pre} / {qa}")
        for k, p in enumerate(s.presenters, start=1):
            assert p.seat_index == k, (
                f"order {s.order} 的第 {k} 位 seat_index={p.seat_index}——"
                f"标志与座位的对应关系错位，按序号读到的就是别人的发言情况")


def test_gender_flags_are_binary_and_sized(segments):
    """性别标志必须是 0/1 且每人一个——原先写死 order 7 是某家公司、
    性别为 [0,1]，换组即红。改为断言对任何组都成立的结构性质。"""
    for s in segments:
        flags = [p.gender_female for p in s.presenters]
        assert len(flags) == s.presenter_num
        assert set(flags) <= {0, 1}, f"order {s.order} 的性别标志含非 0/1 值：{flags}"
    assert any(p.gender_female == 1 for s in segments for p in s.presenters), \
        "全组无一位女性创业者，与既有标注不符，疑似 join 错位"


def test_roundtrip_json(segments, tmp_path):
    p = tmp_path / "segments.json"
    save_segments(segments, p)
    back = load_segments_json(p)
    assert back == segments


def test_missing_panel_raises():
    with pytest.raises(ScheduleError):
        load_segments(panel="不存在的分组名")


def test_row_count_mismatch_raises(tmp_path):
    # Book1 3 行、dta 只有 2 行，且 dta 缺的正是 Book1 第 3 行的企业——
    # 按企业名 join（Task 27）后，行数不等本身不是错误，但 Book1 有、dta 找
    # 不到的企业必须硬失败。
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 0, 100), ("乙公司", 200, 300), ("丙公司", 400, 500)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1},
        {"firmName": "乙公司", "order": 2, "presenterNum": 1},
    ])
    with pytest.raises(ScheduleError, match="丙公司") as exc:
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    assert "第 3 行" in str(exc.value)


def test_company_name_mismatch_raises(tmp_path):
    # 行数相同，但 Book1 第 2 行的企业名在 dta 中根本查不到（dta 里同位置的
    # 记录属于另一家企业「公司丙」）——按企业名 join 后应报「找不到」，而不
    # 再是按行号比对出的「企业名不一致」。
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("公司甲", 0, 100), ("公司乙", 200, 300)])
    _write_dta(dta, "p", [
        {"firmName": "公司甲", "order": 1, "presenterNum": 1},
        {"firmName": "公司丙", "order": 2, "presenterNum": 1},
    ])
    with pytest.raises(ScheduleError, match="找不到人工标注") as exc:
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    msg = str(exc.value)
    assert "第 2 行" in msg and "公司乙" in msg


def test_start_not_before_end_raises(tmp_path):
    # 第 2 行起止时间反转（300 >= 200）
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("公司甲", 0, 100), ("公司乙", 300, 200)])
    _write_dta(dta, "p", [
        {"firmName": "公司甲", "order": 1, "presenterNum": 1},
        {"firmName": "公司乙", "order": 2, "presenterNum": 1},
    ])
    with pytest.raises(ScheduleError, match="起止时间非法") as exc:
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    assert "第 2 行" in str(exc.value)


def test_presenter_num_mismatches_non_null_female_count_raises(tmp_path):
    # 第 2 行 presenterNum=2，但只填了 prst1Female，prst2Female 缺失
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("公司甲", 0, 100), ("公司乙", 200, 300)])
    _write_dta(dta, "p", [
        {"firmName": "公司甲", "order": 1, "presenterNum": 1},
        {"firmName": "公司乙", "order": 2, "presenterNum": 2, "female": [0]},
    ])
    with pytest.raises(ScheduleError, match="非空 prstNFemale") as exc:
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    msg = str(exc.value)
    assert "第 2 行" in msg and "presenterNum=2" in msg


def test_adjacent_segments_overlap_raises(tmp_path):
    # 两行各自起止合法、企业名与人数标注都对得上，但 order1 结束(100) 晚于 order2 开始(50)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("公司甲", 0, 100), ("公司乙", 50, 300)])
    _write_dta(dta, "p", [
        {"firmName": "公司甲", "order": 1, "presenterNum": 1},
        {"firmName": "公司乙", "order": 2, "presenterNum": 1},
    ])
    with pytest.raises(ScheduleError, match="时间重叠") as exc:
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    msg = str(exc.value)
    assert "1" in msg and "2" in msg


# --- Task 27：按企业名 join，而非按行号 zip ---


def test_join_matches_by_firm_name_not_row_order(tmp_path, monkeypatch):
    # Book1 文件顺序：甲、乙、丙；dta 按 order 排序后的顺序打乱为：丙、甲、乙。
    # 现行实现按行号 zip：xl[0]=甲 会配到 dt_sorted[0]=丙，企业名不一致，直接
    # 抛异常——本测试要求 join 改为按企业名匹配后能正确跑通并配对无误，故在
    # 现行实现下必红。
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 0, 100), ("乙公司", 200, 300), ("丙公司", 400, 500)])
    _write_dta(dta, "p", [
        {"firmName": "丙公司", "order": 1, "presenterNum": 3},
        {"firmName": "甲公司", "order": 2, "presenterNum": 1},
        {"firmName": "乙公司", "order": 3, "presenterNum": 2},
    ])
    segments = load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    by_company = {s.company: s for s in segments}
    assert by_company["甲公司"].presenter_num == 1
    assert by_company["乙公司"].presenter_num == 2
    assert by_company["丙公司"].presenter_num == 3


def test_extra_firm_in_dta_is_excluded_with_warning(tmp_path, monkeypatch, capsys):
    # dta 比 Book1 多出一家企业「丙公司」——不是错误（用户会之后补时间戳），
    # 但绝不能静默丢弃，必须在 stdout 打印警告。
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 0, 100), ("乙公司", 200, 300)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1},
        {"firmName": "乙公司", "order": 2, "presenterNum": 2},
        {"firmName": "丙公司", "order": 3, "presenterNum": 3},
    ])
    segments = load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    assert len(segments) == 2
    out = capsys.readouterr().out
    assert "丙公司" in out
    assert "3" in out  # 标注人数


def test_firm_missing_from_dta_raises(tmp_path):
    # Book1 有「乙公司」，dta 里完全找不到——无法定位人工标注，必须硬失败。
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 0, 100), ("乙公司", 200, 300)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1},
    ])
    with pytest.raises(ScheduleError, match="乙公司"):
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)


def test_duplicate_firm_in_dta_raises(tmp_path):
    # dta 中「甲公司」出现两条记录——无法确定该用哪条，必须硬失败。
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 0, 100)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1},
        {"firmName": "甲公司", "order": 2, "presenterNum": 2},
    ])
    with pytest.raises(ScheduleError, match="甲公司"):
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)


def test_order_field_follows_book1_row_order(tmp_path, monkeypatch):
    # segments 的 order 必须继续用 Book1 的行序（1..N），dta_order 单独记录
    # dta 里的原始 order，供人工回溯核对。
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 0, 100), ("乙公司", 200, 300), ("丙公司", 400, 500)])
    _write_dta(dta, "p", [
        {"firmName": "丙公司", "order": 1, "presenterNum": 1},
        {"firmName": "甲公司", "order": 99, "presenterNum": 1},
        {"firmName": "乙公司", "order": 50, "presenterNum": 1},
    ])
    segments = load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    assert [s.order for s in segments] == [1, 2, 3]
    dta_order_by_company = {s.company: s.dta_order for s in segments}
    assert dta_order_by_company == {"甲公司": 99, "乙公司": 50, "丙公司": 1}


def test_presenter_total_assertion_skipped_when_zero(tmp_path, monkeypatch):
    # EXPECTED_PRESENTER_TOTAL == 0 时跳过总人数断言，不应抛异常。
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 0, 100)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1},
    ])
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    segments = load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    assert len(segments) == 1

    # 反向对照：非零且不符时，断言仍应生效——证明「跳过」只在 0 时发生，
    # 而不是断言本身被弱化了。
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 5)
    with pytest.raises(ScheduleError, match="创业者总数"):
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)


def test_firm_name_matching_is_exact_not_fuzzy(tmp_path):
    # Book1「北京爱其（科技）有限公司」vs dta「北京爱其科技有限公司」——只差一对
    # 全角括号。规范化只能是 str(x).strip()，不能去括号/模糊化：括号本身可能是
    # 企业名的一部分，去掉就可能把两家不同企业错配在一起。这里必须精确匹配
    # 失败并硬失败，而不是「聪明地」找到一个近似匹配。
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("北京爱其（科技）有限公司", 0, 100)])
    _write_dta(dta, "p", [
        {"firmName": "北京爱其科技有限公司", "order": 1, "presenterNum": 1},
    ])
    with pytest.raises(ScheduleError, match="找不到人工标注"):
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)


# --- Innermost_expert：用户 2026-08-17 在 dta 末尾补充的分组变量 ---

def test_innermost_expert_is_loaded_from_dta(segments):
    """每段都要带上 innermost_expert，取自 dta 的 Innermost_expert 列。

    该变量按「组」取值（同组内各企业相同），标记会场最里侧是否坐着一位
    可见的专家。用户实测：城市2组=0、新能源2组=0、文化4组=1——与他此前
    逐企业核对时列出的一串「误把专家当创业者」完全对应（那些全是文化4组）。
    """
    for s in segments:
        assert s.innermost_expert in (0, 1), \
            f"order {s.order} 的 innermost_expert={s.innermost_expert!r}，应为 0/1"


def test_innermost_expert_is_uniform_within_group(segments):
    """同组内必须取值一致——它是组级变量，逐企业不同说明 join 错位了。"""
    vals = {s.innermost_expert for s in segments}
    assert len(vals) == 1, f"同一组内 innermost_expert 出现多个取值：{vals}"


def test_innermost_expert_survives_json_roundtrip(tmp_path):
    """写盘再读回不得丢失——阶段 4 读的是 segments.json，不是 dta。"""
    from pipeline.schedule import Segment, Presenter, save_segments, load_segments_json
    seg = Segment(order=1, company="某公司", start=0, end=100, presenter_num=1,
                  presenters=(Presenter(1, 0, 1, 0),), dta_order=1,
                  innermost_expert=1)
    p = tmp_path / "segments.json"
    save_segments([seg], p)
    assert load_segments_json(p)[0].innermost_expert == 1


def test_innermost_expert_defaults_to_zero_when_absent(tmp_path):
    """旧的 segments.json 没有这个键时默认 0（不剔除），不得抛异常。

    默认取 0 而非 1 是刻意的：0 表示「不额外剔除」，是保守行为；默认 1
    会在信息缺失时悄悄砍掉一个真人。
    """
    import json
    from pipeline.schedule import load_segments_json
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"segments": [{
        "order": 1, "company": "旧数据", "start": 0, "end": 100,
        "presenter_num": 1,
        "presenters": [{"seat_index": 1, "gender_female": 0,
                        "speaks_pre": 1, "speaks_qa": 0}],
    }]}, ensure_ascii=False), encoding="utf-8")
    assert load_segments_json(p)[0].innermost_expert == 0


# ---------------------------------------------------------------------------
# 人工补录的实际进/退场时刻（dta 的 entryTime / exitTime 两列）
#
# 用户 2026-09-07 为城市精细化管理1组逐段补了这两个时刻。它们是**这条流水线
# 唯一的进出场 ground truth**：在此之前进出场窗口只能锚在段首/段尾按固定偏移
# 猜（`config.ENTRY_WINDOW` / `EXIT_WINDOW`），而实测该组 order 11 的人是在
# 段首前 130 秒进场的——猜出来的窗口 [start-25, start+15] 与真正的进场整整
# 差了 105 秒，一帧都没覆盖到。
#
# 缺失（NaN / 整列不存在）必须退回旧行为：除城市1组外各组都还没补这两列。
# ---------------------------------------------------------------------------

def test_entry_and_exit_times_are_read_from_dta(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 1000, 2000), ("乙公司", 2500, 3500)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1,
         "entryTime": 970.0, "exitTime": 2010.0},
        {"firmName": "乙公司", "order": 2, "presenterNum": 1,
         "entryTime": 2400.0, "exitTime": 3520.0},
    ])
    segs = load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    assert [s.entry_time for s in segs] == [970.0, 2400.0]
    assert [s.exit_time for s in segs] == [2010.0, 3520.0]


def test_entry_and_exit_times_are_none_when_the_columns_are_absent(tmp_path, monkeypatch):
    """整列不存在 = 这一组还没人补 → None，退回按段首/段尾猜窗口，不抛异常。"""
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 1000, 2000)])
    _write_dta(dta, "p", [{"firmName": "甲公司", "order": 1, "presenterNum": 1}])
    seg = load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)[0]
    assert seg.entry_time is None and seg.exit_time is None


def test_a_single_missing_mark_does_not_void_the_other(tmp_path, monkeypatch):
    """城市1组 order 1 与 order 8 的 entryTime 是空的，exitTime 有值——
    一个缺失不得连累另一个（真实数据就长这样）。"""
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 0, 1110)])
    _write_dta(dta, "p", [{"firmName": "甲公司", "order": 1, "presenterNum": 1,
                           "entryTime": None, "exitTime": 1120.0}])
    seg = load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)[0]
    assert seg.entry_time is None
    assert seg.exit_time == 1120.0


def test_entry_time_after_the_segment_starts_raises(tmp_path, monkeypatch):
    """进场时刻不可能晚于演讲开始。这条拦的是单位/数量级笔误——把 970 敲成
    9700 之后，抽帧窗口会落到别家企业的时间段里，而整条流水线照常跑完。"""
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 1000, 2000), ("乙公司", 2500, 3500)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1, "entryTime": 1500.0},
        {"firmName": "乙公司", "order": 2, "presenterNum": 1},
    ])
    with pytest.raises(ScheduleError, match="进场"):
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)


def test_entry_time_before_the_previous_segment_starts_raises(tmp_path, monkeypatch):
    """允许进场时刻落在上一段的尾巴里（人可以在门口等），但不许落到再上一家
    企业那边去——那必是笔误，而它的后果是把别人的画面抽进本段目录。"""
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 1000, 2000), ("乙公司", 2500, 3500)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1},
        {"firmName": "乙公司", "order": 2, "presenterNum": 1, "entryTime": 900.0},
    ])
    with pytest.raises(ScheduleError, match="进场"):
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)


def test_entry_time_inside_the_previous_segment_is_accepted(tmp_path, monkeypatch):
    """上一段还没讲完就到门口候场——真实会发生，不得当成错误拦掉。"""
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 1000, 2000), ("乙公司", 2500, 3500)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1},
        {"firmName": "乙公司", "order": 2, "presenterNum": 1, "entryTime": 1900.0},
    ])
    segs = load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)
    assert segs[1].entry_time == 1900.0


def test_exit_time_before_the_segment_ends_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 1000, 2000), ("乙公司", 2500, 3500)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1, "exitTime": 1500.0},
        {"firmName": "乙公司", "order": 2, "presenterNum": 1},
    ])
    with pytest.raises(ScheduleError, match="退场"):
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)


def test_exit_time_beyond_the_next_segment_end_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    xlsx = tmp_path / "t.xlsx"
    dta = tmp_path / "t.dta"
    _write_xlsx(xlsx, "s", [("甲公司", 1000, 2000), ("乙公司", 2500, 3500)])
    _write_dta(dta, "p", [
        {"firmName": "甲公司", "order": 1, "presenterNum": 1, "exitTime": 3600.0},
        {"firmName": "乙公司", "order": 2, "presenterNum": 1},
    ])
    with pytest.raises(ScheduleError, match="退场"):
        load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="p", from_dta=False)


def test_entry_exit_times_survive_json_roundtrip(tmp_path):
    """阶段 2 读的是 segments.json，不是 dta——写盘丢了这两个值，
    加密抽帧就会退回按段首猜窗口，而命令看起来一切正常。"""
    seg = Segment(order=1, company="某公司", start=1000, end=2000, presenter_num=1,
                  presenters=(Presenter(1, 0, 1, 0),), dta_order=1,
                  entry_time=970.0, exit_time=2010.0)
    p = tmp_path / "segments.json"
    save_segments([seg], p)
    back = load_segments_json(p)[0]
    assert back.entry_time == 970.0 and back.exit_time == 2010.0
    assert back == seg


def test_entry_exit_times_default_to_none_in_old_segments_json(tmp_path):
    """已跑完的三组的 segments.json 里没有这两个键，读回来必须是 None。"""
    import json
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"segments": [{
        "order": 1, "company": "旧数据", "start": 0, "end": 100,
        "presenter_num": 1,
        "presenters": [{"seat_index": 1, "gender_female": 0,
                        "speaks_pre": 1, "speaks_qa": 0}],
    }]}, ensure_ascii=False), encoding="utf-8")
    seg = load_segments_json(p)[0]
    assert seg.entry_time is None and seg.exit_time is None


# --- 退场时刻可以略早于本段结束（用户 2026-09-19，电子27）-----------------
#
# 原规则要求 exitTime >= endTime，理由写的是「与进场对称」。但它不对称：进场
# 的下界放到了**上一段开始**（候场是真实的），退场的下界却卡死在本段结束。
#
# 全库实测：110 条 exitTime 里 7 条早于本段结束，其中 **6 条只早了 0.4–5.5 秒**
# ——那是「这一段算到哪一秒结束」的判断差异，人起身时主持人还在收尾。剩下 1 条
# 早了 **881 秒**（电子信息2组 order 2：段 8562-9451，exit 填了 8570），那才是
# 这条规则真正要拦的笔误。两拨相差 160 倍，中间是空的。


def _seg(order, start, end, exit_time=None, entry_time=None):
    from pipeline.schedule import Presenter, Segment
    return Segment(order=order, company=f"公司{order}", start=start, end=end,
                   presenter_num=1,
                   presenters=(Presenter(1, 0, 0, 0),),
                   dta_order=order, innermost_expert=0,
                   entry_time=entry_time, exit_time=exit_time)


def test_an_exit_a_few_seconds_early_is_accepted():
    """早几秒是真实的：人起身时主持人还在收尾。实测最多早 5.5 秒。"""
    from pipeline.schedule import _check_marks
    for early in (0.4, 0.7, 2.0, 5.5, 30.0):
        _check_marks([_seg(1, 100.0, 1000.0, exit_time=1000.0 - early),
                      _seg(2, 1100.0, 2000.0)])


def test_an_exit_wildly_before_the_end_is_still_refused():
    """早到讲演刚开始就「退场」的，仍要拦下——那是笔误，按它抽帧抽的是
    这一段中间的画面，人还在台上。"""
    import pytest
    from pipeline.schedule import ScheduleError, _check_marks
    # 电子信息2组 order 2 的真实数值
    with pytest.raises(ScheduleError, match="退场"):
        _check_marks([_seg(2, 8562.0, 9451.0, exit_time=8570.0),
                      _seg(3, 9500.0, 10000.0)])


def test_an_exit_after_the_next_segment_ends_is_still_refused():
    """晚过下一段结束仍然拦——那就抽到下下家的画面了。"""
    import pytest
    from pipeline.schedule import ScheduleError, _check_marks
    with pytest.raises(ScheduleError, match="退场"):
        _check_marks([_seg(1, 100.0, 1000.0, exit_time=2500.0),
                      _seg(2, 1100.0, 2000.0)])


def test_the_entry_rule_is_untouched():
    """进场那一侧一个字不改：可以早到上一段开始，但不许晚于本段开始。"""
    import pytest
    from pipeline.schedule import ScheduleError, _check_marks
    _check_marks([_seg(1, 100.0, 1000.0),
                  _seg(2, 1100.0, 2000.0, entry_time=150.0)])   # 早到上一段内
    with pytest.raises(ScheduleError, match="进场"):
        _check_marks([_seg(1, 100.0, 1000.0),
                      _seg(2, 1100.0, 2000.0, entry_time=1200.0)])  # 晚于本段开始


def test_a_mark_the_human_confirmed_passes_with_a_notice(capsys):
    """人工确认过的「看似越界」的时刻放行，并打印一行说明。

    用户 2026-09-27：「电子2 的时间标注是正确的，请开始处理」。电子信息2组
    order 2 的退场 8570（比本段结束早 881 秒）与 order 16 的进场 23124.5（晚于
    本段开始 23097.5）都被这条校验拦下；数据是用户核对过的。

    校验不能整个关掉——它拦的是 970 敲成 9700 这类笔误，别的组照样需要。所以
    只放行**确认清单里逐条点名**的那几个值。
    """
    from pipeline.schedule import _check_marks
    confirmed = {("电子信息2组", 2, "exit", 8570.0)}
    _check_marks([_seg(2, 8562.0, 9451.0, exit_time=8570.0),
                  _seg(3, 9500.0, 10000.0)],
                 panel="电子信息2组", confirmed=confirmed)
    out = capsys.readouterr().out
    assert "人工确认" in out and "8570" in out, f"放行要说出来：{out}"


def test_a_confirmation_does_not_cover_a_different_value():
    """确认的是**这个数**。dta 里那一格日后被改成别的值，确认随之失效——
    不然一条旧确认会替一个新笔误开绿灯。"""
    import pytest
    from pipeline.schedule import ScheduleError, _check_marks
    confirmed = {("电子信息2组", 2, "exit", 8570.0)}
    with pytest.raises(ScheduleError, match="退场"):
        _check_marks([_seg(2, 8562.0, 9451.0, exit_time=8571.0),
                      _seg(3, 9500.0, 10000.0)],
                     panel="电子信息2组", confirmed=confirmed)


def test_a_confirmation_does_not_leak_to_another_group_or_order():
    """别的组、别的段、另一侧（进场/退场）一律不受影响。"""
    import pytest
    from pipeline.schedule import ScheduleError, _check_marks
    confirmed = {("电子信息2组", 2, "exit", 8570.0)}
    bad = [_seg(2, 8562.0, 9451.0, exit_time=8570.0), _seg(3, 9500.0, 10000.0)]
    with pytest.raises(ScheduleError):
        _check_marks(bad, panel="电子信息27组", confirmed=confirmed)
    with pytest.raises(ScheduleError):
        _check_marks([_seg(4, 8562.0, 9451.0, exit_time=8570.0),
                      _seg(5, 9500.0, 10000.0)],
                     panel="电子信息2组", confirmed=confirmed)
    with pytest.raises(ScheduleError, match="进场"):
        _check_marks([_seg(1, 100.0, 1000.0),
                      _seg(2, 1100.0, 2000.0, entry_time=8570.0)],
                     panel="电子信息2组", confirmed=confirmed)


def test_the_real_confirmations_are_registered():
    """用户确认过的两处写进了 config——换台机器重跑也不会再被拦。"""
    assert ("电子信息2组", 2, "exit", 8570.0) in config.MARKS_CONFIRMED
    assert ("电子信息2组", 16, "entry", 23124.5) in config.MARKS_CONFIRMED


# ---------------------------------------------------------------------------
# Book1 里没有这一组时，用 dta 单独排期（用户 2026-09-20：电子10组）
# ---------------------------------------------------------------------------

def _dta_only_rows():
    return [
        {"firmName": "甲企业", "order": 1, "presenterNum": 2,
         "startTime": 151.2, "endTime": 1074.2, "entryTime": 110.2, "exitTime": 1090.2},
        {"firmName": "乙企业", "order": 2, "presenterNum": 1,
         "startTime": 1699.2, "endTime": 2697.7, "entryTime": 1664.2, "exitTime": 2717.7},
    ]


def _write_dta_with_times(path: Path, panel: str, rows: list[dict]) -> None:
    plain = []
    for r in rows:
        rec = {k: v for k, v in r.items() if k not in ("startTime", "endTime")}
        plain.append(rec)
    _write_dta(path, panel, plain)
    # _write_dta 不写 startTime/endTime，这里补上（dta 单独排期要用它们）
    df = pd.read_stata(path)
    df["startTime"] = [float(r["startTime"]) for r in rows]
    df["endTime"] = [float(r["endTime"]) for r in rows]
    df.to_stata(path, write_index=False, version=118)


def test_schedule_can_come_from_the_dta_alone(tmp_path, monkeypatch):
    """Book1 里没有这一组时，用 dta 的 startTime/endTime 排期。

    用户 2026-09-20 要处理电子10组：Book1 有 60 多组的 sheet，唯独没有它
    （电子9、电子11 都在），而 dta 里这一组 17 行的 startTime/endTime 一个
    不缺。为一组视频手工往只读的输入文件里补一张表，既动了「输入文件只读」
    这条底线，也没有必要。

    段号取 dta 的 `order`——Book1 那一路取的是行序，两者在这一组里同义
    （dta 已按 order 排序，且 1..N 连续，见下一条测试）。
    """
    monkeypatch.setattr(config, "EXPECTED_SEGMENT_COUNT", 2)
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    dta = tmp_path / "x.dta"
    _write_dta_with_times(dta, "电子信息10组", _dta_only_rows())
    # Book1 一次都不许读：路径给个不存在的文件，读了就炸
    segs = load_segments(xlsx=tmp_path / "根本不存在.xlsx", sheet="无此表",
                         dta=dta, panel="电子信息10组", from_dta=True)
    assert [s.order for s in segs] == [1, 2]
    assert [s.company for s in segs] == ["甲企业", "乙企业"]
    assert [(s.start, s.end) for s in segs] == [(151.2, 1074.2), (1699.2, 2697.7)]
    assert [s.presenter_num for s in segs] == [2, 1]
    assert [len(s.presenters) for s in segs] == [2, 1]
    assert segs[0].entry_time == 110.2 and segs[0].exit_time == 1090.2


def test_dta_only_still_rejects_overlapping_segments(tmp_path):
    """dta 单独排期不得绕过任何既有校验——重叠照样硬失败。

    这条路是为一组数据新开的，最容易变成「少走几道检查」的旁路；而时间重叠
    正是会把别家企业的画面抽进这一段的那类错误。
    """
    rows = _dta_only_rows()
    rows[1]["startTime"] = 1000.0        # 早于第 1 段的 endTime=1074.2
    rows[1]["endTime"] = 2697.7
    dta = tmp_path / "x.dta"
    _write_dta_with_times(dta, "电子信息10组", rows)
    with pytest.raises(ScheduleError, match="重叠"):
        load_segments(xlsx=tmp_path / "无.xlsx", sheet="无", dta=dta,
                      panel="电子信息10组", from_dta=True)


def test_dta_only_rejects_non_contiguous_orders(tmp_path):
    """dta 的 order 必须是 1..N 连续——它在这条路上直接当段号用。

    段号是 `frames/{order:02d}`、`seg{order:02d}_seat{k}` 等一长串路径的来源，
    跳号会让证据图与帧目录对不上，而表现只是「某一段的图找不到」。
    """
    rows = _dta_only_rows()
    rows[1]["order"] = 5
    dta = tmp_path / "x.dta"
    _write_dta_with_times(dta, "电子信息10组", rows)
    with pytest.raises(ScheduleError, match="order"):
        load_segments(xlsx=tmp_path / "无.xlsx", sheet="无", dta=dta,
                      panel="电子信息10组", from_dta=True)


def test_book1_path_is_unchanged_and_missing_sheet_mentions_the_dta_option(tmp_path):
    """默认仍走 Book1；缺 sheet 时的报错要指出还有 dta 这条路。

    不加提示的话，下一次遇到 Book1 里没有的组，只会看到「没有 sheet」，
    然后有人会去手改只读的输入文件。
    """
    dta = tmp_path / "x.dta"
    _write_dta_with_times(dta, "电子信息10组", _dta_only_rows())
    xlsx = tmp_path / "b.xlsx"
    _write_xlsx(xlsx, "别的组", [("甲企业", 151, 1074)])
    with pytest.raises(ScheduleError) as e:
        load_segments(xlsx=xlsx, sheet="电子10组4月24日视频", dta=dta,
                      panel="电子信息10组", from_dta=False)
    assert "没有 sheet" in str(e.value)
    assert "from_dta" in str(e.value) or "dta" in str(e.value), (
        f"报错要提到可以改用 dta 排期，实得：{e.value}")


def test_the_default_source_is_the_dta(tmp_path, monkeypatch):
    """默认排期读 dta，不碰 Book1（用户 2026-09-21）。

    切换前逐组比对过：64 个 Book1 sheet 与 dta **起止时刻 0 处不同**，已处理
    的 6 组行数/企业顺序/时刻三项全同，因此换源不动它们的段号。

    这里把 Book1 的路径指向一个不存在的文件：默认路径若还去读它，就会炸。
    """
    monkeypatch.setattr(config, "EXPECTED_SEGMENT_COUNT", 2)
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    dta = tmp_path / "x.dta"
    _write_dta_with_times(dta, "某组", _dta_only_rows())
    segs = load_segments(xlsx=tmp_path / "根本不存在.xlsx", sheet="无此表",
                         dta=dta, panel="某组")
    assert [s.company for s in segs] == ["甲企业", "乙企业"]
    assert [(s.start, s.end) for s in segs] == [(151.2, 1074.2), (1699.2, 2697.7)]
    assert segs[0].entry_time == 110.2 and segs[0].exit_time == 1090.2


def test_book1_remains_reachable_on_request(tmp_path, monkeypatch):
    """`from_dta=False` 仍走 Book1——切换不是删路，是换默认。"""
    monkeypatch.setattr(config, "EXPECTED_SEGMENT_COUNT", 1)
    monkeypatch.setattr(config, "EXPECTED_PRESENTER_TOTAL", 0)
    dta = tmp_path / "x.dta"
    _write_dta_with_times(dta, "某组", _dta_only_rows()[:1])
    xlsx = tmp_path / "b.xlsx"
    # 故意给一个与 dta 不同的窗口，好看出用的是哪一边
    _write_xlsx(xlsx, "s", [("甲企业", 500, 900)])
    segs = load_segments(xlsx=xlsx, sheet="s", dta=dta, panel="某组", from_dta=False)
    assert [(s.start, s.end) for s in segs] == [(500, 900)], (
        f"显式要 Book1 时就得用 Book1 的时刻，实得 {[(s.start, s.end) for s in segs]}")
