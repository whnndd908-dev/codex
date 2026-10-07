"""阈值体检：把「这个阈值切在数据的什么位置」变成可自动检查的东西。

背景（详见 task-40-report.md）：本项目在阈值上栽过三次——
  1. 按各组标定 SEAT_MIN_H_RATIO/SEAT_MIN_FRAME_COVERAGE，三组差 2-5 倍，
     其实只是把阈值调到让候选数正好等于应有人数，不是物理上真实的分界。
  2. RIGHT_REGION_X 切在 630-1170px 这段 540px 宽的真空里（零个 track）——
     这是可信阈值该有的样子。
  3. 进出场高度阈值：坐姿(0.23-0.70)与站姿(0.05-0.76)分布完全重叠，任何
     阈值都切在密集区，这类阈值根本不该存在（已删）。

区分「可信」与「拍脑袋」的不是阈值本身的数值，而是**它切在样本分布的什么
位置**：真空（两侧稀疏、间隙远大于周围间隙）可信；密集区（换一批数据，
两侧的计数、甚至哪边计数更多都可能反转）不可信。`threshold_health` 把这个
判断自动化，`audit_report` 把多个判断结果排成一张表，密集的（最需要人核对
的）排最前。
"""
from __future__ import annotations


def threshold_health(values: list[float], threshold: float, name: str = "") -> dict:
    """判断一个阈值切在样本分布的什么位置。

    判据：
      - `gap`：阈值两侧最近样本的距离（`nearest_above - nearest_below`）。
        若阈值落在样本范围之外（一侧没有样本），退化为阈值到唯一那侧最近
        样本的距离——不崩，如实报告另一侧为 0。
      - `gap_rank`：这个 `gap` 在**全部相邻样本间隙**里的分位（0-1，越大
        越像真空）。之所以用分位而不用 `gap` 的绝对值：绝对间隙依赖量纲
        （像素 vs 比例 vs 角度），换一个量纲/换一组视频，同样的物理间隙
        数值上可以差几个数量级；分位不依赖量纲，只问「这个间隙比这批数据
        里其他间隙的间隙大多少」。
        计算时**排除阈值自身所在的那个相邻间隙**再参与排序——阈值恰好落在
        `vals[k]` 与 `vals[k+1]` 之间时，被评估的 `gap` 就是相邻间隙数组里
        下标为 `k` 的那一个；如果把它自己也放进用来打分的"其他间隙"总体里，
        它必然会因为“不小于自己”而至少不吃亏，把分位系统性地推高——尤其是
        小样本时这个偏差占比更大。所以总体只取**其它**下标的间隙。
        排名用"严格小于"计数（不是"小于等于"）：`gap_rank = 0` 表示这个
        间隙不比任何其他间隙大——均匀连续分布下，所有相邻间隙相等，
        任何阈值算出的 `gap_rank` 都恰好是 0，这正是"密集"判据要的效果
        （若改用"小于等于"，均匀分布下所有间隙互相打平，会被错判成 100%
        分位=真空，这是需要用测试锁住的一个容易犯的错）。
      - `verdict`：
          `样本不足`  n < 10 时，样本太少不足以谈"分布"，不给自信结论，
                     无论 gap_rank 多高都不算数（防止小样本凑巧出现一次
                     大间隙就被误判为真空）。
          `真空`      gap_rank ≥ 0.95 且阈值两侧样本数各 ≥ 3——分位高只说明
                     "这个间隙比其他间隙大"，还得两侧都有足够样本，否则
                     "两侧" 的说法本身就不成立（比如一侧只有 1 个样本，
                     那不是真空，是采样不足偏向了一侧）。
          `稀疏`      gap_rank ≥ 0.80 但不满足真空的两侧样本数要求，或分位
                     没到 0.95。
          `密集`      以上都不满足——阈值切在样本扎堆的地方，换一批数据，
                     两侧的计数、甚至哪边更密都可能整个反过来。
    """
    vals = sorted(values)
    n = len(vals)
    n_below = sum(1 for v in vals if v <= threshold)
    n_above = n - n_below
    nearest_below = vals[n_below - 1] if n_below > 0 else None
    nearest_above = vals[n_below] if n_above > 0 else None

    gaps = [vals[i + 1] - vals[i] for i in range(n - 1)] if n >= 2 else []

    self_idx: int | None = None
    if nearest_below is not None and nearest_above is not None:
        self_idx = n_below - 1
        gap = nearest_above - nearest_below
    elif nearest_above is not None:
        gap = nearest_above - threshold
    elif nearest_below is not None:
        gap = threshold - nearest_below
    else:
        gap = None

    if gap is None:
        gap_rank = 0.0
    else:
        population = (gaps[:self_idx] + gaps[self_idx + 1:]) if self_idx is not None else gaps
        if population:
            gap_rank = sum(1 for g in population if g < gap) / len(population)
        else:
            # 没有其它间隙可比——通常发生在 n<=2 这种连"样本不足"都算宽松
            # 的场景，此时无法给出有意义的相对判断，记 0（不主张真空）。
            gap_rank = 0.0

    if n < 10:
        verdict = "样本不足"
    elif gap_rank >= 0.95 and n_below >= 3 and n_above >= 3:
        verdict = "真空"
    elif gap_rank >= 0.80:
        verdict = "稀疏"
    else:
        verdict = "密集"

    if verdict == "样本不足":
        detail = f"阈值 {threshold} 的判定基于样本不足（n={n}<10），暂不给出结论"
    else:
        gap_str = f"{gap:.3f}" if gap is not None else "无"
        detail = (
            f"阈值 {threshold} 落在{verdict}区：两侧最近样本距 {gap_str}"
            f"（间隙分位 {gap_rank:.2f}），下方 {n_below} 个、上方 {n_above} 个"
        )
        if verdict == "密集":
            detail += " —— 换一组数据结论可能翻转"

    return {
        "name": name,
        "threshold": threshold,
        "n": n,
        "n_below": n_below,
        "n_above": n_above,
        "nearest_below": nearest_below,
        "nearest_above": nearest_above,
        "gap": gap,
        "gap_rank": gap_rank,
        "verdict": verdict,
        "detail": detail,
    }


_VERDICT_PRIORITY = {"密集": 0, "稀疏": 1, "真空": 2, "样本不足": 3}


def audit_report(checks: list[dict]) -> str:
    """把多个 `threshold_health` 结果排成一张表，密集的排最前（最需要关注）。

    排序键：verdict 优先级（密集 < 稀疏 < 真空 < 样本不足，数值越小越靠前）
    为主键；同一 verdict 内按 `gap_rank` 升序（更密的更靠前）为次键；
    `name` 为末位键，保证结果在同优先级、同分位时仍然确定可复现。
    """
    ordered = sorted(
        checks,
        key=lambda c: (
            _VERDICT_PRIORITY.get(c.get("verdict"), 99),
            c.get("gap_rank") if c.get("gap_rank") is not None else 0.0,
            c.get("name") or "",
        ),
    )
    lines = [f"[{c.get('verdict')}] {c.get('name')}：{c.get('detail')}" for c in ordered]
    return "\n".join(lines)
