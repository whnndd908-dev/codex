"""跨组汇总表：把三组视频各自的产物拼成一份总表。

此前这是个临时脚本，第一次运行就出错——`result_*` 的通配把
`result_wide_*` 也匹配了进去，长表拿到的其实是宽表，而 103 行变 45 行
这种错在列名相似时肉眼很难发现。做成正式步骤并补测试，就是为了不再
靠"这次记得排除"来避免它。
"""
import csv

import pytest

from pipeline import combine


def _write_csv(path, rows, cols=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _fake_group(root, key, n_long=3, n_wide=2):
    """造一组产物：长表 / 宽表 / 排名，外加两个必须被排除的干扰文件。"""
    d = root / key
    _write_csv(d / f"result_{key}.csv",
               [{"person_id": f"p{i}", "company": f"公司{i}", "formality_1_5": 3}
                for i in range(n_long)])
    _write_csv(d / f"result_wide_{key}.csv",
               [{"order": i, "company": f"公司{i}", "seat1_formality_1_5": 3}
                for i in range(n_wide)])
    _write_csv(d / f"project_ranking_{key}.csv",
               [{"order": i, "company": f"公司{i}", "confidence_score": 0.5}
                for i in range(n_wide)])
    # 干扰项一：人工校对版，绝不能混进 VLM 版汇总
    _write_csv(d / f"result_人工校对_{key}.csv",
               [{"person_id": "手改的", "company": "手改的", "formality_1_5": 5}])
    return d


# --- 文件挑选：这是踩过坑的地方 ---

def test_result_glob_does_not_match_wide_or_human_edited(tmp_path):
    """`result_` 必须只匹配长表，不得吃进 `result_wide_` 与 `result_人工校对_`。

    临时脚本版就是栽在这里：sorted() 后第一个匹配恰好是 result_wide_，
    长表悄悄变成了宽表，行数从 103 掉到 45，而两张表都有 company 列、
    肉眼扫一眼看不出来。
    """
    d = _fake_group(tmp_path, "甲组")
    got = combine.find_group_csv(d, "result")
    assert got is not None
    assert got.name == "result_甲组.csv"
    assert "wide" not in got.name and "人工" not in got.name


def test_wide_glob_matches_only_wide(tmp_path):
    d = _fake_group(tmp_path, "甲组")
    assert combine.find_group_csv(d, "result_wide").name == "result_wide_甲组.csv"


def test_ranking_glob_matches_only_ranking(tmp_path):
    d = _fake_group(tmp_path, "甲组")
    assert combine.find_group_csv(d, "project_ranking").name == "project_ranking_甲组.csv"


def test_missing_file_returns_none_not_raises(tmp_path):
    """某组还没跑完时返回 None，由调用方决定是跳过还是报错——
    不能在这里抛异常把整份汇总带崩。"""
    (tmp_path / "空组").mkdir()
    assert combine.find_group_csv(tmp_path / "空组", "result") is None


# --- 合并 ---

def test_combine_adds_panel_and_group_key_as_first_columns(tmp_path):
    """三组堆在一起后必须能分辨每行来自哪一组，且这两列排在最前。"""
    _fake_group(tmp_path, "甲组")
    _fake_group(tmp_path, "乙组")
    specs = [combine.GroupSpec("甲组", "甲面板", tmp_path / "甲组"),
             combine.GroupSpec("乙组", "乙面板", tmp_path / "乙组")]

    rows, cols = combine.combine_rows(specs, "result")
    assert cols[:2] == ["panel", "group_key"]
    assert {r["group_key"] for r in rows} == {"甲组", "乙组"}
    assert {r["panel"] for r in rows} == {"甲面板", "乙面板"}


def test_combine_keeps_every_row_from_every_group(tmp_path):
    _fake_group(tmp_path, "甲组", n_long=3)
    _fake_group(tmp_path, "乙组", n_long=5)
    specs = [combine.GroupSpec("甲组", "甲面板", tmp_path / "甲组"),
             combine.GroupSpec("乙组", "乙面板", tmp_path / "乙组")]

    rows, _ = combine.combine_rows(specs, "result")
    assert len(rows) == 8, "8 行必须一行不少——汇总丢行是最难发现的错"


def test_combine_unions_columns_across_groups(tmp_path):
    """各组列不完全一致时（例如某组跑于旧版代码）取并集，不得丢列。"""
    _write_csv(tmp_path / "甲组" / "result_甲组.csv", [{"a": 1, "b": 2}])
    _write_csv(tmp_path / "乙组" / "result_乙组.csv", [{"a": 3, "c": 4}])
    specs = [combine.GroupSpec("甲组", "甲面板", tmp_path / "甲组"),
             combine.GroupSpec("乙组", "乙面板", tmp_path / "乙组")]

    rows, cols = combine.combine_rows(specs, "result")
    for c in ("a", "b", "c"):
        assert c in cols
    assert rows[1].get("b", "") == "", "缺的列留空，不能填 0 或沿用上一行的值"


def test_combine_skips_missing_group_and_reports_it(tmp_path, capsys):
    """某组产物还没生成时跳过并打印，而不是静默少一组。

    静默跳过会让汇总表看着完整、其实少一整组数据——这正是本项目一路在防
    的"失败伪装成成功"。
    """
    _fake_group(tmp_path, "甲组", n_long=3)
    (tmp_path / "乙组").mkdir()
    specs = [combine.GroupSpec("甲组", "甲面板", tmp_path / "甲组"),
             combine.GroupSpec("乙组", "乙面板", tmp_path / "乙组")]

    rows, _ = combine.combine_rows(specs, "result")
    assert len(rows) == 3
    out = capsys.readouterr().out
    assert "乙组" in out and ("缺" in out or "跳过" in out)


def test_write_combined_writes_all_three_tables(tmp_path):
    _fake_group(tmp_path, "甲组", n_long=3, n_wide=2)
    specs = [combine.GroupSpec("甲组", "甲面板", tmp_path / "甲组")]

    written = combine.write_combined(specs, tmp_path / "out")
    names = {p.name for p in written}
    assert names == {"all_groups_long.csv", "all_groups_wide.csv",
                     "all_groups_ranking.csv"}
    for p in written:
        assert p.exists() and p.stat().st_size > 0

    with (tmp_path / "out" / "all_groups_long.csv").open(encoding="utf-8-sig") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 3
    assert got[0]["panel"] == "甲面板"


def test_group_registry_covers_the_three_real_groups():
    """config 里的组注册表必须列全三组，且 work 目录名互不相同。

    汇总步骤靠这份注册表知道要合并哪些组；漏一组就会静默少一份数据。
    """
    keys = [g.key for g in combine.GROUPS]
    assert len(keys) == 3, f"应有三组，实得 {keys}"
    assert len(set(keys)) == 3, "组标识不得重复"
    assert len({g.work.name for g in combine.GROUPS}) == 3, "work 目录不得重名"


def test_source_column_cannot_override_panel_or_group_key(tmp_path):
    """源 CSV 自带 panel/group_key 列时，不得覆盖掉汇总注入的组标识。

    补于变异测试：把 `cols` 的显式初始化去掉后 10 条测试仍全绿——因为
    `{"panel": ..., "group_key": ..., **r}` 的 dict 顺序恰好也让这两列排在
    最前，断言分不清保证来自显式初始化还是来自巧合。顺着这条查下去发现了
    真缺陷：`**r` 在后，源文件里若有同名列会**反过来覆盖组标识**，于是
    "这行来自哪一组"被文件内容改写，而汇总表看上去完全正常。
    """
    _write_csv(tmp_path / "甲组" / "result_甲组.csv",
               [{"panel": "冒充的面板", "group_key": "冒充的组", "a": 1}])
    specs = [combine.GroupSpec("甲组", "甲面板", tmp_path / "甲组")]

    rows, cols = combine.combine_rows(specs, "result")
    assert cols[:2] == ["panel", "group_key"]
    assert rows[0]["panel"] == "甲面板", "组标识必须由注册表决定，不能被文件内容改写"
    assert rows[0]["group_key"] == "甲组"


def test_legacy_unsuffixed_files_do_not_hijack_the_pick(tmp_path):
    """改名前的旧产物（result.csv / result_wide.csv，无组名后缀）不得被选中。

    真实事故：Task 33 给产物加组名后缀后，目录里同时存在
        result.csv              ← 旧
        result_wide.csv         ← 旧
        result_城市2-4.26.csv     ← 新
        result_wide_城市2-4.26.csv ← 新
    kind="result" 时 `rest = "wide.csv"` 不以 `wide_` 开头，于是排除条件失效，
    sorted() 又把 result_wide.csv 排在 result_城市X.csv 前面——长表再次拿到
    宽表，45 行冒充 103 行。与上一次是同一类错误的不同表现，说明按字符串
    前缀手写排除本身就不牢靠，应改为「排除所有更长的兄弟 kind」。
    """
    d = tmp_path / "甲组"
    _write_csv(d / "result.csv", [{"x": "旧长表"}])
    _write_csv(d / "result_wide.csv", [{"x": "旧宽表"}])
    _write_csv(d / "result_甲组.csv", [{"x": "新长表"}])
    _write_csv(d / "result_wide_甲组.csv", [{"x": "新宽表"}])

    assert combine.find_group_csv(d, "result").name == "result_甲组.csv"
    assert combine.find_group_csv(d, "result_wide").name == "result_wide_甲组.csv"


def test_all_kinds_are_mutually_exclusive(tmp_path):
    """任意两种 kind 挑出的文件必须互不相同——这是汇总正确性的底线。

    不逐条枚举排除规则，而是直接断言最终结果两两不等：无论将来加多少种
    产物、前缀怎么互相包含，这条都能兜住。
    """
    d = _fake_group(tmp_path, "甲组")
    _write_csv(d / "result.csv", [{"x": "旧"}])
    _write_csv(d / "result_wide.csv", [{"x": "旧"}])
    _write_csv(d / "project_ranking.csv", [{"x": "旧"}])

    picked = {k: combine.find_group_csv(d, k) for k in combine.KINDS}
    names = [p.name for p in picked.values() if p]
    assert len(names) == len(set(names)), f"不同 kind 选到了同一个文件：{names}"
    for k, p in picked.items():
        assert p.name.startswith(k + "_"), f"{k} 选到了 {p.name}"
