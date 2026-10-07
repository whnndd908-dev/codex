"""评审席（画面左侧的专家）识别。

用户 2026-09-17：「重新读取视频左侧的专家的衣服图片，以对专家着装的正式度
进行评价」，并明确「每组恒定为 5 个专家，只是有的可能不在其中，将交由我人工
核实」。会场平面图：大屏幕在最里侧，评审人 1-5 沿左侧一列由近及远排开，
创业者 1-3 在右侧，工作人员计时在最靠镜头那一端。

与创业者那一路的三条关键区别：

1. **组级，不是段级。** 同一批评审人坐一整天，衣服不变。跨全部段汇集同一个
   位置的检测，每组产出 5 条记录，而不是每家企业重复一遍（复制 5 条成 85 条
   是纯冗余，要按企业行合并后期 join 即可）。

2. **席位恒为 5。** 检出不足 5 位时空着的号是**可见的缺口**，不是「这组只有
   4 个」——人工才分得清「确实只来了 4 位」与「第 3 位被漏检了」。

3. **标识由位置导出。** 评审人整场不动，位置就是身份；track_id 重跑检测即
   全变（Task 46），而「时刻」对一个全程在座的人没有意义——所以 stable_id
   里**没有时间**，只有 `#cx,y2`，与进出场候选的 `kind@t#cx,y2` 区分开。

数据来源上这一路很便宜：左侧的人本来就已经被检测到了（`seats.exclude_judges`
一直按 x < RIGHT_REGION_X 把他们滤掉不用），四个组盘上现成的 seated 帧里
左区检测框有 2594-8513 个，**不需要重新抽帧**。
"""
from __future__ import annotations

import statistics as st

from pipeline import config
from pipeline.detect import Detection


def judge_stable_id(cx: float, y2: float) -> str:
    """评审人的稳定标识 `judge#cx,y2`（整数像素）。

    不带时间：他从头到尾坐在那里，没有一个「时刻」可言；带上任何时间都只是
    把某一次抽帧的偶然固化进标识里。与进出场候选的 `kind@t_mid#cx,y2` 前缀
    不同，两类标识因此不会在同一份标注文件里撞上。
    """
    return f"judge#{cx:.0f},{y2:.0f}"


def judge_candidates(dets: list[Detection], frame_w: int = 1440,
                     frame_h: int = 1080,
                     min_frames: int | None = None) -> list[dict]:
    """全组的评审人候选，按**离镜头由近及远**排序（y2 降序）。

    两级归并：先按 (段, track) 聚成「这一段里的这个人」，再按位置跨段归并。
    跨段直接按 track 归并是不行的——跟踪器在每个 (段, 窗口) 都重新编号，同一位
    评审人在 17 段里有 17 个不同的 track_id。

    只看 `seated` 窗口：进出场窗口里左侧走动的是工作人员和正在离场的上一组人，
    把他们收进来会让同一张脸在几个组的评审席里反复出现。

    `min_frames` 缺省取 `config.JUDGE_MIN_FRAMES`；显式传是给测试与探查用的
    （真实数据里一位评审人有几百到两千帧，夹具不必也堆那么多）。

    返回列表，每项：
        {"stable_id", "median_cx", "median_y2", "median_h", "height_ratio",
         "n_frames", "n_segments", "segments": [段号...], "best_dets": [...]}
    """
    from pipeline.annotate import select_best_frames

    cut = config.RIGHT_REGION_X * frame_w
    left = [d for d in dets if d.kind == "seated" and d.cx < cut]

    per_track: dict[tuple[int, int], list[Detection]] = {}
    for d in left:
        per_track.setdefault((int(d.seg_order), int(d.track_id)), []).append(d)

    # 段内碎片（跟踪器丢失重识别后拆出来的几段）先不管：跨段归并的容差本来
    # 就能把它们并到同一个位置上，多一层反而多一套容差要调。
    people: list[dict] = []
    frags = []
    for (order, _tid), ds in per_track.items():
        if len(ds) < config.JUDGE_MIN_TRACK_FRAMES:
            continue
        frags.append({
            "cx": st.median([d.cx for d in ds]),
            "y2": st.median([d.bottom_y for d in ds]),
            "h": st.median([d.height for d in ds]),
            "order": order, "dets": ds,
        })
    frags.sort(key=lambda f: -f["y2"])

    for f in frags:
        for p in people:
            if (abs(p["median_cx"] - f["cx"]) < config.JUDGE_MERGE_CX_TOL
                    and abs(p["median_y2"] - f["y2"]) < config.JUDGE_MERGE_Y2_TOL):
                p["_cx"].append(f["cx"]); p["_y2"].append(f["y2"]); p["_h"].append(f["h"])
                p["median_cx"] = st.median(p["_cx"])
                p["median_y2"] = st.median(p["_y2"])
                p["median_h"] = st.median(p["_h"])
                p["dets"].extend(f["dets"])
                p["segments"].add(f["order"])
                break
        else:
            people.append({"median_cx": f["cx"], "median_y2": f["y2"],
                           "median_h": f["h"], "_cx": [f["cx"]], "_y2": [f["y2"]],
                           "_h": [f["h"]], "dets": list(f["dets"]),
                           "segments": {f["order"]}})

    out = []
    for p in people:
        if len(p["dets"]) < (config.JUDGE_MIN_FRAMES if min_frames is None
                             else min_frames):
            continue
        out.append({
            "stable_id": judge_stable_id(p["median_cx"], p["median_y2"]),
            "median_cx": p["median_cx"],
            "median_y2": p["median_y2"],
            "median_h": p["median_h"],
            "height_ratio": p["median_h"] / float(frame_h),
            "n_frames": len(p["dets"]),
            "n_segments": len(p["segments"]),
            "segments": sorted(p["segments"]),
            **_pool_fields(p["dets"]),
        })
    # 离镜头由近及远 = y2 降序 = 专家 1..5。与创业者座位号同一条约定
    # （`seats.py`：「y2 越小离镜头越远；最里侧那位即 y2 中位数最小者」），
    # 平面图里评审人1 与创业者1 也同在靠工作人员计时那一端。
    out.sort(key=lambda c: -c["median_y2"])
    return out


def _pool_fields(dets: list[Detection]) -> dict:
    """备选池：`pool_dets`（给人挑的若干张）与 `best_dets`（其中前四张）。

    用户 2026-09-18：「针对 2×2 不足的专家，重新给出新的挑选图，以实现 2×2
    四张图」。城市1 标完之后五位专家只剩 1/3/3/2/3 格，共缺 8 格；材料不缺
    （每位 846-2173 帧），缺的是把它们拿出来给人挑。

    两条硬约束：

    1. **前四张必须与原来的 `best_dets` 一字不差。** 格位号是人工标注的键，
       前四张一变，用户标好的「格2 是其他人」就指到了另一张图上——静默张冠
       李戴，比没有备选糟得多。所以池就是同一个排序的前缀。

    2. **备选跨段取。** 被误并进来的邻座往往集中在某一段里（跟踪在那一段跟丢
       了重识别到隔壁）；顺着分数往下取第 5 张多半还是那一段的同一个人，换了
       也白换，而用户会以为自己已经换过一张。段数不够时才回头取同段的次好
       ——宁可近似，也不给人一个空格。

    排序只做一次：`select_best_frames` 会把每一帧都读图算清晰度，这是这一路
    最贵的一步，取前 4 与取全量的代价完全相同（它本来就是先全量打分再切片）。
    """
    from pipeline.annotate import select_best_frames

    ranked = select_best_frames(dets, n=len(dets))
    head = ranked[:config.JUDGE_CELL_COUNT]
    want = config.JUDGE_CELL_POOL
    pool = list(head)
    seen_seg = {d.seg_order for d in head}
    taken = {id(d) for d in head}
    for d in ranked[config.JUDGE_CELL_COUNT:]:
        if len(pool) >= want:
            break
        if d.seg_order in seen_seg:
            continue
        pool.append(d)
        seen_seg.add(d.seg_order)
        taken.add(id(d))
    for d in ranked[config.JUDGE_CELL_COUNT:]:
        if len(pool) >= want:
            break
        if id(d) not in taken:
            pool.append(d)
            taken.add(id(d))
    pool.extend(_spread_by_position(ranked, taken, config.JUDGE_CELL_SPREAD))
    return {"pool_dets": pool, "best_dets": head}


def _spread_by_position(ranked: list[Detection], taken: set[int],
                        want: int) -> list[Detection]:
    """按 cx 分箱轮流取，补一批**位置上分散**的备选。

    为什么需要：`select_best_frames` 是全局排名，谁的画面分数高谁就通吃。两位
    坐得近的评审人被并成一个候选时（新能源2 第 4、5 位，相距 cx 80px），实测
    16 格池子里一侧占 15 格、另一侧只有 1 格——而底层是 896 对 819，材料一直
    都在，只是浮不上来。人看不到另一位的画面，就没法把他拆出去。

    分箱按这个候选自己的 cx 跨度均分，轮流从每个箱里取当前分数最高的一张：
    位置上的每一处都会有画面，而箱内仍按分数挑。跨度小（只有一个人）时各箱
    取到的是同一位的不同角度，不浪费。
    """
    rest = [d for d in ranked if id(d) not in taken]
    if not rest or want <= 0:
        return []
    lo = min(d.cx for d in rest)
    hi = max(d.cx for d in rest)
    n_bins = max(1, min(want, 4))
    step = (hi - lo) / n_bins if hi > lo else 1.0
    bins: list[list[Detection]] = [[] for _ in range(n_bins)]
    for d in rest:
        i = min(n_bins - 1, int((d.cx - lo) / step)) if hi > lo else 0
        bins[i].append(d)      # rest 已按分数降序，箱内自然也是
    out: list[Detection] = []
    while len(out) < want and any(bins):
        for b in bins:
            if len(out) >= want:
                break
            if b:
                out.append(b.pop(0))
    return out


def _parse_stable_id(sid: str) -> tuple[float, float] | None:
    """`judge#cx,y2` → (cx, y2)；解析不出返回 None（绝不猜）。"""
    try:
        head, rest = str(sid).split("#", 1)
        cx, y2 = rest.split(",", 1)
        if head != "judge":
            return None
        return float(cx), float(y2)
    except (ValueError, AttributeError):
        return None


def realign_judge_labels(candidates: list[dict], labels: dict | None
                         ) -> tuple[dict | None, dict]:
    """把旧标注的键对回**当前这一轮**候选的标识。

    用户 2026-09-18 要对城市1 的 order 5/6/7/8/10 按进出场时刻重抽。评审人的
    标识由位置中位数导出（`judge#cx,y2`，整数像素），重抽哪怕一段，那一段的
    检测就重新参与中位数、位置挪零点几个像素，标识即变——而查找是精确字符串
    匹配，对不上就静默当作「这一位没人标过」。实测那五段占全组 seated 检测的
    28%，专家2 的 cx 从 245.48 挪到 244.45，`judge#245,801` → `judge#244,801`，
    他身上那一整条标注（标掉格4、补选格5）就没了。创业者那一路早有按内容重新
    对齐这一层（Task 46），评审人这一路漏了。

    先配**标识完全相同**的，配过的候选不再参与就近匹配——否则一个挪了位的标注
    可能抢走另一位的候选，两条标注互相串位。其余的按 `JUDGE_LABEL_MATCH_TOL`
    就近配；两个候选同时落在容差内一律**拒配**，挑一个的结果取决于字典顺序，
    而认错了就是把这一位的衣着判定安到另一个人头上。

    返回 `(新的 labels 副本, 统计)`。统计里 `matched` / `moved` / `dropped` /
    `ambiguous` 必须打印出来——**这是工作有没有丢的唯一可见迹象**。
    原字典不改（调用方可能还要拿它比对）。
    """
    stats = {"total": 0, "matched": 0, "moved": 0, "dropped": [], "ambiguous": []}
    if not labels or not isinstance(labels.get("judges"), dict):
        return labels, stats
    old = labels["judges"]
    stats["total"] = len(old)
    by_sid = {c["stable_id"]: c for c in candidates}
    out: dict[str, dict] = {}
    used: set[str] = set()

    rest = []
    for sid, rec in old.items():
        if sid in by_sid:
            out[sid] = rec
            used.add(sid)
            stats["matched"] += 1
        else:
            rest.append((sid, rec))

    tol = config.JUDGE_LABEL_MATCH_TOL
    for sid, rec in rest:
        pos = _parse_stable_id(sid)
        if pos is None:
            stats["dropped"].append(sid)
            continue
        near = [c for c in candidates
                if c["stable_id"] not in used
                and abs(c["median_cx"] - pos[0]) <= tol
                and abs(c["median_y2"] - pos[1]) <= tol]
        if len(near) > 1:
            stats["ambiguous"].append(sid)
            continue
        if not near:
            stats["dropped"].append(sid)
            continue
        out[near[0]["stable_id"]] = rec
        used.add(near[0]["stable_id"])
        stats["matched"] += 1
        stats["moved"] += 1

    new = dict(labels)
    new["judges"] = out
    return new, stats


def judge_labels_of(labels: dict | None) -> dict:
    """标注文件里 `judges` 那一段：`{stable_id: {"judge": int|None, "role": str}}`。"""
    if not labels:
        return {}
    raw = labels.get("judges") or {}
    return raw if isinstance(raw, dict) else {}


def judge_slots(candidates: list[dict], labels: dict | None) -> dict:
    """把候选排进**恒为 5 个**的席位：`{1..5: 候选 | None}`。

    基线是几何排序（离镜头由近及远），人工改号压过它。空缺留 `None`——那是
    一个必须看得见的缺口，不是「这组只有 4 个」。

    两人被指到同一号时抛 `ValueError`：挑一个的结果取决于字典顺序，会随标注
    先后次序变，且一处不报错（与 `entryexit.seat_reassign_of` 同一条规矩）。
    """
    marks = judge_labels_of(labels)
    slots: dict[int, dict | None] = {k: None for k in range(1, config.JUDGE_COUNT + 1)}

    named: dict[int, dict] = {}
    dropped: set[str] = set()
    for cand in candidates:
        rec = marks.get(cand["stable_id"])
        if not isinstance(rec, dict):
            continue
        if str(rec.get("role") or "") == config.JUDGE_ROLE_NOT_A_JUDGE:
            dropped.add(cand["stable_id"])
            continue
        n = rec.get("judge")
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue
        if not (1 <= n <= config.JUDGE_COUNT):
            continue
        if n in named:
            raise ValueError(
                f"有两个候选同时被指为专家 {n}——一个席位只能有一个人。"
                f"请在挑选页改清楚（{named[n]['stable_id']} 与 {cand['stable_id']}）。")
        named[n] = cand
    slots.update(named)

    taken = {c["stable_id"] for c in named.values()} | dropped
    rest = [c for c in candidates if c["stable_id"] not in taken]
    for n in range(1, config.JUDGE_COUNT + 1):
        if slots[n] is not None or not rest:
            continue
        slots[n] = rest.pop(0)
    return slots


JUDGE_CELL_NOBODY = "x"


def judge_picks(labels: dict | None, stable_id: str) -> list[int]:
    """人工挑中组成 2×2 的那几格（池格位号，按他挑的次序）。

    用户 2026-09-18：「不是按后面的备选自动顶上，而是我可以挑选 4 个最清晰的
    组成 2×2 图」。自动顶上只按分数（面积×清晰度）往下顺，而「清晰」这件事
    机器排得并不准——真正看得清衣服的那张要人眼定。

    **空列表 = 没挑过**，由调用方退回自动基线：没人动过的那几位照旧有一张
    四格图，不会因为没人挑就空着（与本项目其余各处「几何给基线、人工压过它」
    同一条规矩）。

    最多 `JUDGE_CELL_COUNT` 格、去重、读不出的跳过：2×2 只有四格，多的画不下，
    而悄悄多带几格会让登记表里的 frames 与图上看到的对不上。
    """
    rec = judge_labels_of(labels).get(str(stable_id))
    if not isinstance(rec, dict):
        return []
    out: list[int] = []
    for v in rec.get("picks") or []:
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n >= 1 and n not in out:
            out.append(n)
        if len(out) >= config.JUDGE_CELL_COUNT:
            break
    return out


def judge_cell_assignments(labels: dict | None, stable_id: str) -> dict[int, int | None]:
    """这一位的 2×2 里，哪几格**其实是别人**：`{格位号: 专家号 | None}`。

    用户 2026-09-18：「专家的每个格有可能是其他人，给出这个修改选项」。只能
    勾掉是不够的——城市1 实测四位专家共勾掉 7 格，专家1 与专家3 各只剩 1 格，
    而模型拿一格判整场衣着比拿四格混着两个人还糟。那些格并没有坏，只是**归错
    了人**：说清它是谁，帧就换到那一位名下（与创业者那一路的「格位切分」同一
    条思路），而不是扔掉。

    值为专家号 = 那一格其实是那一位；`None` = 其他人（工作人员、路过的），
    丢掉。**没提到的格位不在返回值里**，那就是本人——人只需要点出错的那几格。

    读不出的目标号一律按「其他人」处理：宁可丢一格，也不把一张不知道是谁的
    脸记到某一位名下（记错的那一位整场衣着就判错了，而一处不报错）。

    同时兼容旧的 `cell_exclusions: [2, 4]`：用户 2026-09-18 11:30 已经用那一版
    勾完城市1 导出了，换个字段名就把他那份工作静默作废了。
    """
    rec = judge_labels_of(labels).get(str(stable_id))
    if not isinstance(rec, dict):
        return {}
    out: dict[int, int | None] = {}
    for v in rec.get("cell_exclusions") or []:
        try:
            k = int(v)
        except (TypeError, ValueError):
            continue
        if k >= 1:
            out[k] = None
    cells = rec.get("cells")
    if isinstance(cells, dict):
        for raw_k, raw_v in cells.items():
            try:
                k = int(raw_k)
            except (TypeError, ValueError):
                continue
            if k < 1:
                continue
            try:
                n = int(raw_v)
            except (TypeError, ValueError):
                n = None
            out[k] = n if (n is not None and 1 <= n <= config.JUDGE_COUNT) else None
    return out


def judge_cell_exclusions(labels: dict | None, stable_id: str) -> set[int]:
    """某位评审人被人工勾掉的证据图格位号（1 起）。

    用户 2026-09-18：「专家的 2×2 需要每个格子进行人工核实，因为有的格子
    混进了其他专家」。评审席比创业者座位更容易混——跨段归并按位置容差并人
    （`JUDGE_MERGE_CX_TOL` / `JUDGE_MERGE_Y2_TOL`），相邻两位本来就坐得近，
    某一段里跟丢重识别之后隔壁那位很容易被并进同一个候选。四格红框标的都是
    同一个候选，**模型无从分辨哪格才是本人**，两个人的衣着混起来判断，不
    报错而数字全错（与 `entryexit.cell_exclusions_of` 记下的是同一类事故）。

    **键用 stable_id，不用席位号**：席位号是人工随时会改的，按号存的话把某位
    从专家 3 改成专家 2 之后，勾掉的格位就落到了另一位身上——被勾掉的那一格
    重新混进模型，同样不报错。stable_id 由位置导出，跨重检测稳定。

    读不出的格位号跳过：旁路通道里混进脏值是常态，为它炸掉整组没有意义。
    """
    # 归给别人的格，对**这一位**来说同样要剔掉——不然他的 2×2 里还留着一张
    # 别人的脸，而那张脸同时又出现在另一位名下。
    return set(judge_cell_assignments(labels, stable_id))


def _judge_seat_of(rec: dict) -> int | None:
    """一条标注里「专家第几位」的号；不是专家、号读不出或越界一律 None。

    号留空就是「不知道是哪一位」，钳到边界等于把这一帧记到另一个人头上，
    而一处不报错（与 `entryexit_fill_plan` 只认精确「创业者」同一条规矩）。
    """
    if not isinstance(rec, dict) or str(rec.get("role") or "") != "专家":
        return None
    try:
        n = int(rec.get("seat"))
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= config.JUDGE_COUNT else None


def judge_donations_from_candidates(labels: dict | None,
                                    candidates_by_order: dict) -> dict:
    """右侧创业者候选里被标成「专家 N」的格位 → `{专家号: [(候选标识, 格位, 帧)]}`。

    用户 2026-09-19：「文化4 的专家5 在右侧最里面靠近投影仪的位置」。
    `judge_candidates` 只看画面**左区**（`cx < RIGHT_REGION_X`），坐在右侧的那
    一位一个候选都出不来——文化4 实测评审席 4/5、第 5 号空缺，而「未入席的候选」
    也是空的，人工根本无从指认他。

    他并没有丢：右区的人都进了创业者候选。出口就是逐格身份那一栏——把某个候选
    的某一格标成「专家 N」，那一帧归到第 N 位评审人名下。与左区评审人之间互相
    改判（`judge_cell_assignments`）是同一条通路，只是来源不同，因此两者产出的
    三元组同形，`registry` 合并之后一视同仁。

    **只收精确等于「专家」的角色**，且号必须落在 1..`JUDGE_COUNT`：号留空就是
    「不知道是哪一位」，钳到边界等于把这一帧记到另一个人头上，而一处不报错
    （与 `entryexit_fill_plan` 只认精确「创业者」同一条规矩）。

    按 (候选标识, 格位号) 排序后返回：2×2 装不下时被丢掉的是哪一格不能随字典
    顺序变。
    """
    from pipeline.entryexit import CELL_COUNT, cell_dets, cell_map_of_label

    out: dict[int, list] = {}
    if not labels:
        return out
    for order, cands in (candidates_by_order or {}).items():
        seg_labels = labels.get(str(order)) or labels.get(order) or {}
        if not isinstance(seg_labels, dict):
            continue
        for cand in cands or []:
            lab = seg_labels.get(cand.get("stable_id"))
            if not isinstance(lab, dict):
                continue
            cell_map, _report, _src = cell_map_of_label(lab, CELL_COUNT)
            if not cell_map:
                # 人数为 1 的候选不展开格位分配，人工只能在卡片顶上选身份+号
                # （实测文化4 entry/exit track 2：右侧那位评审人就是这种）。
                # 只认逐格标注的话，他标了等于没标。**逐格标过就以逐格为准**，
                # 两者都收会让同一帧进两次、2×2 里出现重复画面。
                n = _judge_seat_of(lab)
                if n is not None:
                    for k in range(1, len(cand.get("best_dets") or []) + 1):
                        for d in cell_dets(cand, k):
                            out.setdefault(n, []).append(
                                (str(cand.get("stable_id")), k, d))
                continue
            for cell, rec in (cell_map or {}).items():
                if not isinstance(rec, dict):
                    continue
                n = _judge_seat_of(rec)
                if n is None:
                    continue
                for d in cell_dets(cand, cell):
                    out.setdefault(n, []).append(
                        (str(cand.get("stable_id")), int(cell), d))
    for got in out.values():
        got.sort(key=lambda x: (x[0], x[1]))
    return out


def judge_number_conflicts(candidates: list[dict],
                           labels: dict | None) -> dict[int, list[str]]:
    """被人工指到**同一个号**的候选：`{号: [stable_id, ...]}`，没有冲突返回空。

    `judge_slots` 遇到这种情况抛 ValueError——挑一个取决于字典顺序，那条拒绝
    是对的。但抛异常同时**挡住了挑选页的渲染**，而挑选页正是改这个号的唯一
    地方：页面出不来 → 改不了 → 页面永远出不来（用户 2026-09-19 实测电子27
    就卡在这里）。所以先有一个只报告、不抛的版本，让调用方自己决定：登记表
    那一侧继续硬拒，挑选页降级并把冲突显示出来。
    """
    marks = judge_labels_of(labels)
    by_num: dict[int, list[str]] = {}
    for cand in candidates:
        rec = marks.get(cand["stable_id"])
        if not isinstance(rec, dict):
            continue
        if str(rec.get("role") or "") == config.JUDGE_ROLE_NOT_A_JUDGE:
            continue
        try:
            n = int(rec.get("judge"))
        except (TypeError, ValueError):
            continue
        if 1 <= n <= config.JUDGE_COUNT:
            by_num.setdefault(n, []).append(cand["stable_id"])
    return {n: sids for n, sids in by_num.items() if len(sids) > 1}


def without_conflicting_numbers(candidates: list[dict],
                                labels: dict | None) -> dict | None:
    """把冲突的那几位的**号**清掉，其余一字不动。原字典不改。

    只清号：逐格标注（`cells`）与补选（`picks`）是另一件工作，跟着一起清掉
    就是拿一个号的冲突去毁掉他半天的逐格核实。别人的号也不动——那是他们各自
    的工作，与这一处冲突无关。
    """
    conflicts = judge_number_conflicts(candidates, labels)
    if not conflicts:
        return labels
    bad = {sid for sids in conflicts.values() for sid in sids}
    out = dict(labels or {})
    judges = {}
    for sid, rec in (out.get("judges") or {}).items():
        if sid in bad and isinstance(rec, dict):
            rec = {k: v for k, v in rec.items() if k != "judge"}
        judges[sid] = rec
    out["judges"] = judges
    return out
