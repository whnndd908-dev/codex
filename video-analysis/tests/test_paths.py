"""Task 56G · 交付物只存一份：原组目录放原始与过程文件，Output 放输出文件。

用户 2026-08-24：「Output 已经有的文件在原组文件夹下就不用有了，即原来的
位置放原始文件和过程文件，Output 放输出文件，不用重复存储」。

同一份东西存两处是这个项目反复吃亏的根源（Ruling R39 那一类）：两处迟早
不一致，而不一致时哪一份作数没人说得清，且不报错。
"""
from pathlib import Path

import pytest

from pipeline import paths


def test_output_dir_sits_beside_the_group_not_inside_it(tmp_path):
    """`work/Output/<组名>/`——与各组目录平级，不是塞进组目录里。

    塞进组目录（`work/<组>/Output/`）等于把交付物混回过程文件堆里，用户要
    发给别人时得先在几万张帧中间找它。
    """
    work = tmp_path / "work" / "城市2-4.26"
    assert paths.output_dir(work) == tmp_path / "work" / "Output" / "城市2-4.26"


def test_deliverables_default_to_output(tmp_path):
    """四份交付物的默认位置都在 Output 下，不在组目录里。"""
    work = tmp_path / "work" / "测试组"
    work.mkdir(parents=True)
    out = paths.output_dir(work)
    assert paths.registry_path(work, "S") == out / "seat_registry_S.json"
    assert paths.preview_path(work, "S") == out / "vlm_preview_S.html"
    assert paths.labels_path(work, "S") == out / "entry_labels_S.json"
    assert paths.review_page_path(work) == out / "entry_review.html"


def test_legacy_copies_in_the_group_dir_are_still_found(tmp_path):
    """组目录里还留着旧文件时照样读得到——**不能因为改了规矩就当它不存在**。

    用户手上已经有好几组标注躺在旧位置；读不到的表现是「一条标注都没有」，
    而流水线照常跑完、结果全是就座基线，不报错。
    """
    work = tmp_path / "work" / "测试组"
    work.mkdir(parents=True)
    legacy = work / "entry_labels_S.json"
    legacy.write_text("{}", encoding="utf-8")
    assert paths.labels_path(work, "S") == legacy, "旧位置有就用旧位置"


def test_output_wins_when_both_exist(tmp_path):
    """两处都有时以 Output 为准，并且这件事要能被上层查出来（不是静默二选一）。"""
    work = tmp_path / "work" / "测试组"
    work.mkdir(parents=True)
    (work / "entry_labels_S.json").write_text("{}", encoding="utf-8")
    out = paths.output_dir(work)
    out.mkdir(parents=True)
    (out / "entry_labels_S.json").write_text("{}", encoding="utf-8")
    assert paths.labels_path(work, "S") == out / "entry_labels_S.json"
    dups = paths.duplicated(work, "S")
    assert "entry_labels_S.json" in dups, (
        f"两处都有要报得出来，用户才知道该清哪一份，实得 {dups}")


def test_no_duplicates_reported_when_only_one_place_has_it(tmp_path):
    work = tmp_path / "work" / "测试组"
    work.mkdir(parents=True)
    (work / "entry_labels_S.json").write_text("{}", encoding="utf-8")
    assert paths.duplicated(work, "S") == [], "只有一处不算重复"


def test_migrate_moves_and_does_not_copy(tmp_path):
    """迁移是**移动**不是复制——复制就正好造出用户要消灭的那份重复。"""
    work = tmp_path / "work" / "测试组"
    work.mkdir(parents=True)
    (work / "entry_labels_S.json").write_text("A", encoding="utf-8")
    (work / "entry_review.html").write_text("B", encoding="utf-8")
    moved = paths.migrate(work, "S")
    out = paths.output_dir(work)
    assert sorted(moved) == ["entry_labels_S.json", "entry_review.html"], moved
    assert (out / "entry_labels_S.json").read_text(encoding="utf-8") == "A"
    assert not (work / "entry_labels_S.json").exists(), "旧位置必须清掉"
    assert not (work / "entry_review.html").exists()


def test_migrate_never_overwrites_a_newer_output_copy(tmp_path):
    """Output 里已经有同名文件时**不覆盖**，报出来让人自己定。

    覆盖掉的可能正是对的那一份（用户刚在 Output 里标完），而覆盖不报错。
    """
    work = tmp_path / "work" / "测试组"
    work.mkdir(parents=True)
    (work / "entry_labels_S.json").write_text("旧", encoding="utf-8")
    out = paths.output_dir(work)
    out.mkdir(parents=True)
    (out / "entry_labels_S.json").write_text("新", encoding="utf-8")
    moved = paths.migrate(work, "S")
    assert moved == [], f"两处都有时不动，实得 {moved}"
    assert (out / "entry_labels_S.json").read_text(encoding="utf-8") == "新"
    assert (work / "entry_labels_S.json").exists(), "也不许悄悄删掉旧的那份"


def test_migrate_is_a_noop_when_nothing_to_move(tmp_path):
    work = tmp_path / "work" / "测试组"
    work.mkdir(parents=True)
    assert paths.migrate(work, "S") == []


def test_the_judge_annotations_have_a_path_of_their_own():
    """评审席的打标结果有自己的文件名，且走 `_resolve`（Output 优先）。

    与 `annotations.json` 分开：那一份按 (段, 座位) 组织、每一行都带企业与
    座位号，而评审人是**组级**的，没有段也没有企业。混进去就得给每个下游
    消费者加一条「这一行 seg_order=0，特殊对待」。
    """
    from pipeline import paths
    # 文件名净化用的必须是**同一个** `_safe`，不另写一套规则——两套规则的
    # 表现是写端与读端各算一个名字，文件明明在盘上却「找不到」。
    sheet = "甲 组/名"
    p = paths.judges_path(Path("/tmp/w"), sheet)
    assert p.name == f"judges_{paths._safe(sheet)}.json", p.name
    assert "/" not in p.name
    # 与别的交付物放同一个目录（Output 优先、回落组目录）
    assert p.parent == paths.registry_path(Path("/tmp/w"), sheet).parent
