"""阶段 ④ 辅助：用人工标注的性别序列校正座位号（Task 33）。

背景：几何排座位已连续三次证伪——深度分不开专家与后排创业者、覆盖率过滤
顺序错把碎片砍光、这次又发现站立者的脚落地导致框底与前排齐平、被排成
靠前的座位（order 14 座位 2 框高 849px，全组中位 528px，y1=214，明显是
站姿）。而用户的人工标注性别准确度很高，且 Task 29 已把 PROMPT 里的性别
泄漏修掉——`gender_judged`（VLM 独立判定）不再是抄标注答案的循环论证。
实测第三组 30 人里 27 人（90%）性别一致，3 处不一致里有 2 处（order 14）
只是座位顺序颠倒——段内重排即可对上。

`match_seats_by_gender` 因此把「按 y2 降序」这个几何基线，改成「用性别
标注约束下、离基线最近的可行重排」：几何仍是主排序依据（重排只在性别
约束逼着换的地方才换），性别是校正信号，不是从零重新猜。
"""
from __future__ import annotations

from itertools import permutations


def _to_bit(gender_judged) -> int | None:
    """VLM 判定文本 -> 0/1 位；「不确定」或缺失记 None（通配符，规则 5）。"""
    if gender_judged == "女":
        return 1
    if gender_judged == "男":
        return 0
    return None


def _gender_label(bit: int | None) -> str:
    if bit == 1:
        return "女"
    if bit == 0:
        return "男"
    return "不确定"


def _best_assignment(
    ordered_people: list[dict], annotated_genders: list[int]
) -> tuple[tuple[int, ...], bool]:
    """在满足性别约束的全部座位指派里，找「与基线（y2 降序原始次序）位置差
    之和」最小的那个。

    `ordered_people` 必须已按 y2 降序排好、长度与 `annotated_genders` 相等。
    人数 ≤4，`itertools.permutations` 穷举 n! 完全够用，不需要匈牙利算法。

    返回 (perm, conflict)：`perm[s]` = 座位 s（0 记）取 `ordered_people` 里
    第几个人；`conflict=True` 表示没有任何指派能同时满足全部性别约束
    （多重集对不上），此时 `perm` 就是基线恒等映射 `(0,1,...,n-1)`。

    代价函数天然覆盖规则 2/3：
      - 基线本身（恒等排列）若可行，代价恒为 0，严格小于任何非恒等排列
        （交换任意两人代价至少 +2），故「已经匹配上」时必然原样返回，不
        会无谓打乱（规则 2）。
      - 多个可行排列代价打平时，`itertools.permutations` 按字典序枚举，
        取第一个见到的最小代价者——对同性别成员而言，字典序最小等价于
        尽量保持原有相对次序，只在被迫处才交换（规则 3）。
    """
    n = len(annotated_genders)
    judged_bits = [_to_bit(p.get("gender_judged")) for p in ordered_people]
    identity = tuple(range(n))

    best: tuple[int, ...] | None = None
    best_cost: int | None = None
    for perm in permutations(range(n)):
        feasible = all(
            judged_bits[perm[s]] is None or judged_bits[perm[s]] == annotated_genders[s]
            for s in range(n)
        )
        if not feasible:
            continue
        cost = sum(abs(perm[s] - s) for s in range(n))
        if best_cost is None or cost < best_cost:
            best, best_cost = perm, cost

    if best is None:
        return identity, True
    return best, False


def match_seats_by_gender(
    people: list[dict],
    annotated_genders: list[int],
) -> dict[int, int]:
    """用人工标注性别序列校正按 y2 排出的座位号。

    people: 每人一个 dict，至少含 `track_id`、`y2`（越大越靠镜头/越靠前）、
        `gender_judged`（VLM 独立判定，"男"/"女"/"不确定"，缺省视为不确定）。
    annotated_genders: 人工标注，按座位号 1..N 顺序，1=女 0=男
        （对应 `Segment.presenters` 里 `gender_female`，presenters 本就按
        seat_index 升序排列，直接取 `[p.gender_female for p in presenters]`
        即可）。

    返回 {track_id: seat_index}（1 记）。

    规则：
      1. 先按 y2 降序给出基线指派——无信息（人数不等、全员「不确定」、
         `annotated_genders` 为空）时的默认，与既有纯几何排序完全一致。
      2. 若 VLM 性别与标注序列能通过段内重排完全对上，采用该重排。
      3./4. 多种重排都可行时取「与基线位置差之和」最小者，同性别内部尽量
         不打乱（细节见 `_best_assignment`）。
      5. `gender_judged` 为「不确定」或缺失的人不参与约束，仍占座位——
         `_to_bit` 把它们记成 `None`，`_best_assignment` 里通配符判据
         `judged_bits[...] is None` 恒真。
      完全对不上（人数不等，或性别多重集在任何重排下都无法满足约束）时，
      退回基线指派——`gender_conflict` 标志由 `gender_match_report` 给出，
      本函数只返回座位号，不改变调用方既有的 `{track_id: seat}` 契约。
    """
    n = len(annotated_genders)
    ordered = sorted(people, key=lambda p: -p["y2"])

    if len(ordered) != n or n == 0:
        # 人数不等：性别多重集根本无法比较，直接退回基线（按 y2 截取前 n 个，
        # 不足则照单全收，与既有 assign_segment_seats「不硬凑」的约定一致）。
        chosen = ordered[:n]
        return {p["track_id"]: i + 1 for i, p in enumerate(chosen)}

    perm, _conflict = _best_assignment(ordered, annotated_genders)
    return {ordered[perm[s]]["track_id"]: s + 1 for s in range(n)}


def gender_match_report(segments_data: list[dict]) -> list[dict]:
    """每段一行的性别匹配标定报告，供 `run_all.py` 阶段 4 打印用。

    `segments_data`：每段一个 dict {"seg_order", "people", "annotated_genders"}
    ——`people`/`annotated_genders` 与 `match_seats_by_gender` 同义。

    返回每段一个 dict：
        seg_order, annotated（标注序列，如"男女男"）, vlm（VLM 序列，按 y2
        降序排列，如"男男女"，含"不确定"）, rearranged（是否发生了重排）,
        gender_conflict（是否退回基线：人数不等，或性别多重集无可行解）。
    """
    rows: list[dict] = []
    for seg in segments_data:
        seg_order = seg["seg_order"]
        people = seg["people"]
        annotated_genders = seg["annotated_genders"]
        n = len(annotated_genders)
        ordered = sorted(people, key=lambda p: -p["y2"])
        annotated_label = "".join(_gender_label(g) for g in annotated_genders)
        vlm_label = "".join(_gender_label(_to_bit(p.get("gender_judged"))) for p in ordered)

        if len(ordered) != n or n == 0:
            rows.append({
                "seg_order": seg_order, "annotated": annotated_label, "vlm": vlm_label,
                "rearranged": False, "gender_conflict": True,
            })
            continue

        perm, conflict = _best_assignment(ordered, annotated_genders)
        rearranged = (not conflict) and perm != tuple(range(n))
        rows.append({
            "seg_order": seg_order, "annotated": annotated_label, "vlm": vlm_label,
            "rearranged": rearranged, "gender_conflict": conflict,
        })
    return rows
