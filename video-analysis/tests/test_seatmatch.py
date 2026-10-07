from pipeline.seatmatch import match_seats_by_gender, gender_match_report


def person(tid, y2, gender_judged):
    return {"track_id": tid, "y2": y2, "gender_judged": gender_judged}


def test_gender_swap_is_corrected():
    """复刻 order 14 的真实情形：标注 [男,女,男]，VLM 按 y2 降序检出
    [男,男,女]——座位 2、3 的性别颠倒了。重排后座位 2 应是女、座位 3 应是男。
    """
    people = [
        person(1, 900.0, "男"),   # y2 最大，最靠前 -> 基线座位 1
        person(2, 700.0, "男"),   # 基线座位 2
        person(3, 500.0, "女"),   # 基线座位 3
    ]
    annotated = [0, 1, 0]   # 座位1=男 座位2=女 座位3=男
    seats = match_seats_by_gender(people, annotated)
    assert seats[1] == 1, "座位1本就对得上，不应变动"
    assert seats[3] == 2, "VLM判女的人(tid=3)应排进座位2"
    assert seats[2] == 3, "VLM判男的人(tid=2)应排进座位3"


def test_baseline_order_kept_when_genders_already_match():
    """标注与 VLM 检出（按 y2 降序）已经一致时，不做任何重排，结果与纯
    y2 排序完全相同。"""
    people = [
        person(10, 900.0, "男"),
        person(11, 700.0, "女"),
        person(12, 500.0, "男"),
    ]
    annotated = [0, 1, 0]
    seats = match_seats_by_gender(people, annotated)
    assert seats == {10: 1, 11: 2, 12: 3}


def test_minimal_rearrangement_when_ambiguous():
    """标注 [男,男,女]，检出（y2序）也是 [男,男,女]——两个男的位置理论上
    可以互换而不违反约束，但代价函数应选择「不换」这个零代价方案，同性别
    内部保持原有 y2 次序。"""
    people = [
        person(21, 900.0, "男"),
        person(22, 700.0, "男"),
        person(23, 500.0, "女"),
    ]
    annotated = [0, 0, 1]
    seats = match_seats_by_gender(people, annotated)
    assert seats == {21: 1, 22: 2, 23: 3}, (
        "两个男的可互换但不应无谓交换，应保持 y2 原有次序"
    )


def test_uncertain_gender_is_wildcard():
    """VLM 判「不确定」的人不构成约束（通配符），但仍占一个座位——
    标注 [男,女]，检出 [不确定,女]：座位1本可以是「不确定」那位而无冲突，
    不因为它判不出性别就被排除在外。"""
    people = [
        person(31, 900.0, "不确定"),
        person(32, 700.0, "女"),
    ]
    annotated = [0, 1]
    seats = match_seats_by_gender(people, annotated)
    assert seats == {31: 1, 32: 2}
    assert set(seats) == {31, 32}, "「不确定」的人仍必须占一个座位，不能被丢弃"


def test_uncertain_gender_does_not_block_rearrangement():
    """「不确定」不参与约束这一点必须是真的可测行为，不只是「仍占座位」：
    标注 [女,男]，VLM 按 y2 序检出 [男,不确定]——若把「不确定」错误当成
    硬约束「男」，会判定无解退回基线；但它本应被当通配符，允许把 tid=42
    换到座位1（标注女），tid=41（VLM判男）换到座位2（标注男，满足）。"""
    people = [
        person(41, 900.0, "男"),
        person(42, 700.0, "不确定"),
    ]
    annotated = [1, 0]   # 座位1=女 座位2=男
    seats = match_seats_by_gender(people, annotated)
    assert seats == {42: 1, 41: 2}, (
        "不确定应作通配符参与重排，而不是被当成硬约束导致退回基线"
    )


def test_minimal_cost_beats_lexicographically_first_feasible():
    """护栏（对应控制器变异清单 (b)：重排取任意可行解而非最小改动）：构造
    一个 4 人场景，其中 itertools.permutations 枚举到的**第一个**可行解
    （代价 4）并非全局最小代价的可行解（代价 2）——若实现退化成「随便取
    一个可行解」而不是真正比较代价，会静默选中前者，只有真正做代价比较
    才会选中后者。"""
    people = [
        person(100, 900.0, "不确定"),
        person(101, 700.0, "男"),
        person(102, 500.0, "男"),
        person(103, 300.0, "不确定"),
    ]
    annotated = [0, 1, 0, 0]   # 座位1=男 座位2=女 座位3=男 座位4=男
    seats = match_seats_by_gender(people, annotated)
    assert seats == {101: 1, 100: 2, 102: 3, 103: 4}, (
        f"应取代价最小的可行解，实得 {seats}"
    )


def test_unmatchable_falls_back_with_conflict_flag():
    """性别多重集对不上（标注 2 男 1 女，VLM 检出 3 男）时，match_seats_by_gender
    退回基线指派，gender_match_report 必须把这段标记为冲突。"""
    people = [
        person(51, 900.0, "男"),
        person(52, 700.0, "男"),
        person(53, 500.0, "男"),
    ]
    annotated = [0, 1, 0]   # 2 男 1 女，与 VLM 检出的 3 男矛盾
    seats = match_seats_by_gender(people, annotated)
    assert seats == {51: 1, 52: 2, 53: 3}, "对不上时必须退回纯 y2 基线"

    report = gender_match_report([
        {"seg_order": 9, "people": people, "annotated_genders": annotated}
    ])
    assert len(report) == 1
    assert report[0]["gender_conflict"] is True
    assert report[0]["rearranged"] is False


def test_gender_match_report_marks_rearranged_segment():
    people = [
        person(1, 900.0, "男"),
        person(2, 700.0, "男"),
        person(3, 500.0, "女"),
    ]
    annotated = [0, 1, 0]
    report = gender_match_report([
        {"seg_order": 14, "people": people, "annotated_genders": annotated}
    ])
    row = report[0]
    assert row["seg_order"] == 14
    assert row["annotated"] == "男女男"
    assert row["vlm"] == "男男女"
    assert row["rearranged"] is True
    assert row["gender_conflict"] is False


def test_person_count_mismatch_is_reported_as_conflict():
    """检出人数与标注人数不等时（候选丢了一个人），性别多重集根本无法
    比较，必须退回基线并标记冲突，而不是尝试比较不等长的序列。"""
    people = [person(1, 900.0, "男"), person(2, 700.0, "女")]
    annotated = [0, 1, 0]   # 标注 3 人，只检出 2 人
    seats = match_seats_by_gender(people, annotated)
    assert seats == {1: 1, 2: 2}

    report = gender_match_report([
        {"seg_order": 5, "people": people, "annotated_genders": annotated}
    ])
    assert report[0]["gender_conflict"] is True
