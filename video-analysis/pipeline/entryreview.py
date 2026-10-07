"""阶段 ④d·人工挑选页：把进出场候选（`pipeline.entryexit.entryexit_candidates`）
渲染成单文件 HTML，供人工逐段挑出谁是创业者、谁是工作人员/专家（Task 39）。

候选按框高排序展示，**不是「站立判定」**——两次框高阈值判据都被证伪
（就座创业者与进出场候选的框高分布完全重叠，见 `pipeline/entryexit.py`
模块顶部与 `config.py` 里 `ENTRYEXIT_TOP_N` 上方的完整记录），因此每段每个
窗口只展示前 `config.ENTRYEXIT_TOP_N` 个候选，超出部分被截断时页面必须
如实标出截断了多少（凡有截断必须可见）。

沿用 `pipeline/gallery.py` 已验证过的几条约定，不重新发明：
  - 图不嵌进 HTML，写成文件放在页面同目录的 `review_img/`（用户 2026-09-18）。
  - 图片显示缩放复用 `config.GALLERY_IMG_SCALE`（纯 CSS 缩放，不重采样像素，
    见 Task 36）。
  - 全部插值走 JS 里的 `esc()`，与 gallery.py 同一套转义/`_is_scalar` 白名单
    约定——该文件早期出过 XSS 与 bytes 污染 CSV 的缺陷，这里不重蹈覆辙。
  - 未标注时不得预填任何值（Task 32 的教训：预填等于把「没判定」伪装成
    「已判定」）——六个角色单选默认全部不选，座位号下拉默认「不确定」。
    「人数」下拉例外：默认值 1 是真实默认（大多数框确实只有一个人），
    不是「没判定」的伪装，Task 41 用户明确要求「人数下拉：1（默认）」。

Task 41（用户四条改动）：
  - 改动一：长宽比过滤局部检测（`pipeline.entryexit.entryexit_filtered_candidates`），
    被挡掉的候选不丢弃，渲染进每段末尾的可展开折叠区块。
  - 改动二：角色单选从四个扩到六个，新增「上一组创业者」「下一组创业者」。
  - 改动三：候选新增「人数」下拉（1/2/3），>1 时页面提示写备注，导出 JSON
    带 `n_people`/`note`；`entryexit_fill_plan` 相应把这类候选分流进
    `needs_resplit`，不参与自动补位。
  - 改动四：进场/退场区块各写一句朝向提示（`orientationHint`）——进场大概率
    背影、退场大概率正面，工作人员例外——供人工判断参考，不做自动判定。

Task 54（用户 2026-08-23）：
  - 需求一：格位分配从「写备注 + Python 侧解析」改成**结构化选择题**。
    人数选 N，页面当场展开 N 行「格位 / 身份 / 座位号」下拉，身份与座位号
    都可留空（留空 = 未判定，不参与补位），同一个人占两格时后一格选
    「同格位X」（只能引用更小的格位号，环在结构上不可能出现）。
    **实时性是这组改动的全部价值**：解析在 Python 侧时，用户填完要等下次
    跑流水线才知道结果——城市2组 order1 三个人早就写进备注了，目录仍是
    红的。现在 `determinedCount` 直接读 cells，填完立刻变绿。
    旧的 `note` 与解析器都**保留不删**（用户已按备注标完两组）：有 cells
    就用 cells、忽略 note；只有 note 时把解析结果预填进控件并提示
    「已由备注自动填充，请核对」；解析不出的把原文只读展示出来。
    优先级的唯一定义在 `entryexit.cell_map_of_label`，页面与之逐条对齐。

本模块只产出页面与读回工具，不接入 `assemble.py` 的正式流水线——先让用户
看到页面、确认候选靠谱，再谈接入（见 `pipeline.entryexit.entryexit_fill_plan`
docstring）。
"""
from __future__ import annotations

import hashlib
import html as _html
import json
from datetime import datetime
from pathlib import Path

from pipeline import config, paths
from pipeline.annotate import (build_evidence_sheet, cell_label_convention_text,
                              select_best_frames)
from pipeline.detect import Detection
from pipeline.entryexit import (
    CELL_COUNT,
    candidates_for_labels,
    CELL_ROLE_CHOICES,
    LABEL_RESERVED_KEYS,
    entryexit_candidate_totals,
    entryexit_candidates,
    entryexit_filtered_candidates,
    load_entry_labels,
    no_candidate_seats_of,
    note_to_cells,
    parse_cell_note_report,
)
from pipeline import judges as judges_mod
from pipeline.judges import (
    judge_candidates, judge_number_conflicts, judge_slots,
    realign_judge_labels, without_conflicting_numbers,
)
from pipeline.schedule import Segment


REVIEW_IMG_DIR = "review_img"


def _review_image_name(ref: tuple[int, str, int], media: str) -> str:
    """图片文件名。**由 ref 导出**，同一个候选每轮都落在同一个名字上。

    名字里带段号与类别，出问题时人一眼看得出是哪一段的哪一类图；判不清时
    他就是拿这个名字去 `frames/` 里对的。
    """
    order, kind, idx = ref
    ext = ".jpg" if media == "image/jpeg" else ".png"
    if kind == "judge":
        return f"judge{idx + 1:02d}{ext}"
    return f"seg{order:02d}_{kind}{idx:02d}{ext}"


def _write_review_images(out_path: Path, refs: list, candidates: list[dict]
                         ) -> tuple[int, int, int]:
    """把页面要用的图写进 `<页面同目录>/review_img/`，并在每项上记下相对路径。

    返回 (写了几张, 合计字节, 清掉几张上一轮遗留的)。

    为什么不再嵌进 HTML：一整组 264 张图 base64 进去有 80 MB，要压到 32 MB
    才装得下——而「人核对的图比模型看到的糊」正是本项目最忌讳的那一类
    （`test_downscaling_is_disclosed_on_the_page` 当初记的就是这件事）。写成
    文件之后压缩这一步整个消失，人看到的与送模型的是同一批字节。

    清理**只限这个目录**，且只清这一轮没写的那些：候选数量随重抽帧变化，留着
    的孤儿图既占盘，又让人以为页面还在用它。别处一个文件都不碰——`work/` 下
    其余每一样东西都带着人工标注的结果。
    """
    img_dir = out_path.parent / REVIEW_IMG_DIR
    img_dir.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    n_img = n_bytes = 0
    for ref, c in zip(refs, candidates):
        if not c.get("data"):
            c["src"] = ""
            continue
        name = _review_image_name(ref, c.get("media") or "image/png")
        (img_dir / name).write_bytes(c["data"])
        written.add(name)
        n_img += 1
        n_bytes += len(c["data"])
        c["src"] = f"{REVIEW_IMG_DIR}/{name}"
    n_stale = 0
    for old in img_dir.iterdir():
        if old.is_file() and old.name not in written:
            old.unlink()
            n_stale += 1
    return n_img, n_bytes, n_stale


def page_code_fp() -> str:
    """这份**挑选页代码**的指纹，随导出写进 `_meta.page_code_fp`。

    用户 2026-09-19 实测电子27：修好「专家的号被导出过滤掉」之后重出了页面，
    他重新导出，那五条专家的 seat 还是 null。我拿真页面在命令行上跑
    `buildLabelsPayload()`，导出的是 `seat: 5`——**页面是对的、文件是错的**
    两件事同时成立，而我没有任何依据判断这份文件出自哪份页面（浏览器多半
    还开着改版前那个标签页，或者吃了缓存）。只能请他再导一次，赌一把。

    `detections_fingerprint` 顶不上：它标的是检测结果，重出页面时检测没有
    重跑，指纹一字不变——恰恰在页面代码改了的时候完全不动。

    取 `_TEMPLATE` 的摘要：页面的行为全部来自这份模板，改了它才需要重新
    导出，替换进去的数据（候选、座位）变了并不影响「导出逻辑是哪一版」。
    """
    return hashlib.md5(_TEMPLATE.encode("utf-8")).hexdigest()[:16]


def render_entry_review(
    segments: list[Segment],
    dets: list[Detection],
    out_path: Path,
    seats_by_order: dict[int, dict[int, int]] | None = None,
    frame_w: int = 1440,
    frame_h: int = 1080,
    detections_fp: str = "",
    entry_labels: dict | None = None,
    labels_path: Path | None = None,
) -> None:
    """渲染 `entry_review.html`。

    `detections_fp`：本页候选所依据的那份 `detections.json` 的指纹
    （`entryexit.detections_fingerprint`），随导出的 JSON 一起写进
    `_meta.detections_fingerprint`，供 `entryexit.load_entry_labels` 判断
    「这份标注是不是这份检测的」（Task 46）。缺省空串——调用方没提供时
    页面照常可用，只是下游无从核对（`load_entry_labels` 会打印说明）。

    `seats_by_order`：来自阶段 4 `assign_segment_seats` 的 `{seg_order:
    {track_id: seat}}`，用于算每段「就座已定 M 人 / 还缺 K 人」。缺省
    `None`（等价于空——尚未跑过座位分配时页面仍可用，只是每段都显示为
    「已定0人」）。

    `frame_h`：画面高度，只用于算候选的框高占比（`config.
    ENTRYEXIT_MIN_HEIGHT_RATIO` 的绝对下限判据）。默认 1080，与
    `detect.detect_frames`、`pairing` 一贯的默认口径一致。

    每段一个区块，标题
        `#order 企业名 · 标注N人 · 就座已定M人 · 还缺K人`
    **段落一律按 order 升序**（Task 52 需求一，取代 Task 39 的「缺人的段排
    最前」）。用户 2026-08-23：「既然目录已用颜色区分标注完成状态，段落就按
    order 正常排序更好找」。缺人优先那一版的排序键是标注进度，而进度会随着
    用户边标边变——同一家企业标完两个候选就从第 2 位跳到第 9 位，按位置记
    住的路线全废。order 是唯一不随标注变化的稳定锚；「哪些还没标完」改由
    目录的颜色 + ⛔/⚠/✓ 标记承担（那套本就存在，且不受排序影响）。

    `entry_labels` / `labels_path`（Task 54 需求一）：既有标注，**只读**，
    用来做两件事，都在 `_file_label_prefill` 里：

      1. 把旧的自由文本备注解析成格位选择预填进控件（`note_to_cells`），
         页面提示「已由备注自动填充，请核对」——用户已按备注标完两组，
         不给迁移路径等于让他重标；
      2. 把文件里已有的 `cells` 补回页面（`hydrateCellsFromFile`）。
         `run_all._entry_labels_to_store` 目前只搬 role/seat/n_people/note，
         不搬 cells（那个文件不在本任务可改的范围），不补这一手，用户导出
         过的格位选择重开页面就没了，**而且不会报错**。

    两者都不传时按约定去 `out_path` 同目录找 `entry_labels_<SHEET>.json`
    （正式流水线里 `entry_review.html` 与标注文件正是同一个 work 目录）。
    读取走 `load_entry_labels` 的**同一套失效校验**：指纹不符时若能按视频
    内容重新对齐就用对齐后的，对不上一律整份不用——预填错一个候选，等于
    把甲写的格位分配显示在乙的卡片上（Task 46 那类错配，且不报错）。
    """
    seats_by_order = seats_by_order or {}

    per_segment_candidates: dict[int, list[dict]] = {
        segment.order: entryexit_candidates(dets, segment, frame_w, frame_h)
        for segment in segments
    }
    per_segment_totals: dict[int, dict[str, int]] = {
        segment.order: entryexit_candidate_totals(dets, segment, frame_w, frame_h)
        for segment in segments
    }
    # 改动一 + Task 52 需求三：被长宽比挡掉（判为局部检测/手）或被框高绝对
    # 下限挡掉（小到认不出是谁）的候选，都不丢弃，单独取出供页面折叠区块
    # 展示，并逐条标明是被哪条规则挡的——凡有过滤必须可见。
    per_segment_filtered: dict[int, list[dict]] = {
        segment.order: entryexit_filtered_candidates(dets, segment, frame_w, frame_h)
        for segment in segments
    }

    # --- 把每个候选（含被长宽比挡掉的）的 2×2 证据图攒成一个 candidates
    # 列表，最后交给 `_write_review_images` 一次写出。refs 用
    # (order, "cand"|"filtered", idx) 三元组区分两份列表，互不混淆。
    # 已定座位（用户 2026-08-23：只标缺失的座位）只出 2×2。
    #
    # 2026-09-20 去掉了卡片上那张就座单图（用户实测电子28：「本段已定座位
    # 会出现一个空白框是为何」——39 个全是灰方块）。它取自 `keyframes/`，
    # 而 `export_keyframes` 只在阶段 8 里调用，人工核实在阶段 5 之后：
    # **人需要它的时候它必然不存在**。何况它与「人核对的图 = 送模型的图」
    # 相冲突——送进 VLM 的是这张 2×2，单图是另一张模型看不到的裁剪。
    per_segment_filled: dict[int, list[dict]] = {}
    for segment in segments:
        rows = []
        for seat in sorted(set(seats_by_order.get(segment.order, {}).values())):
            # 已定座位的 2×2 证据图（用户 2026-08-24）：跟踪串号会让四格里
            # 某一格框到别人，而四格红框标的都是同一个 track——模型无从分辨，
            # 会把两个人的衣着混起来判断且不报错。必须让用户在标注页看到它。
            seat_tid = None
            for t, sn in (seats_by_order.get(segment.order) or {}).items():
                if sn == seat:
                    seat_tid = t
                    break
            sheet_dets = []
            if seat_tid is not None:
                own = [d for d in dets
                       if d.seg_order == segment.order and d.kind == "seated"
                       and d.track_id == seat_tid]
                if own:
                    sheet_dets = select_best_frames(own, n=CELL_COUNT)
            rows.append({"seat": seat, "sheet_dets": sheet_dets})
        per_segment_filled[segment.order] = rows

    img_candidates: list[dict] = []
    # 图落成文件、页面只引用（用户 2026-09-18：「HTML 引用原图，打开也快」）。
    # 此前一整组 264 张图以 base64 嵌进页面，80 MB 要压到 32 MB 才装得下——
    # 「人核对的图比模型看到的糊」正是本项目最忌讳的那一类。写成文件之后压缩
    # 这一步整个消失：人看到的与送模型的是同一批字节。顺带 base64 比原始字节
    # 胀 33%，浏览器要一口气解析 33 MB 的文档；改后页面只剩 1 MB 上下。
    refs: list[tuple[int, str, int]] = []
    for segment in segments:
        # 已定座位图与候选图纳入同一份体积预算，不另起一套
        for idx, row in enumerate(per_segment_filled[segment.order]):
            sd = row.get("sheet_dets") or []
            sheet = (build_evidence_sheet(
                        [d.frame_path for d in sd],
                        [(d.x1, d.y1, d.x2, d.y2) for d in sd],
                        label_cells=True)
                     if sd else None)
            img_candidates.append({"data": sheet, "media": "image/png"})
            refs.append((segment.order, "filledsheet", idx))
        # Task 49 · 需求一：挑选页的证据图带格位编号（1/2/3/4），用户在
        # 备注里按格位号写清「1 号格是创业者、2 号格是工作人员」。
        # VLM 用的那份走 assemble.py，用的是默认参数（不带编号）——多余的
        # 数字会干扰模型判读。
        for idx, cand in enumerate(per_segment_candidates[segment.order]):
            paths = [d.frame_path for d in cand["best_dets"]]
            boxes = [(d.x1, d.y1, d.x2, d.y2) for d in cand["best_dets"]]
            png = build_evidence_sheet(paths, boxes, label_cells=True) if paths else None
            img_candidates.append({"data": png, "media": "image/png"})
            refs.append((segment.order, "cand", idx))
        for idx, cand in enumerate(per_segment_filtered[segment.order]):
            paths = [d.frame_path for d in cand["best_dets"]]
            boxes = [(d.x1, d.y1, d.x2, d.y2) for d in cand["best_dets"]]
            png = build_evidence_sheet(paths, boxes, label_cells=True) if paths else None
            img_candidates.append({"data": png, "media": "image/png"})
            refs.append((segment.order, "filtered", idx))

    # 评审席（组级，用户 2026-09-17）：跨全部段汇集画面左侧的专家。
    # 2×2 走的是与创业者同一个 `build_evidence_sheet`，也和他们一起写出
    # ——另起一套出图路径必然日后走样，而走样的表现是某一类图静默变成叉。
    # 备选池切成若干张 2×2（用户 2026-09-18：「针对 2×2 不足的专家，重新给出
    # 新的挑选图，以实现 2×2 四张图」）。第一张就是原来那张——池的前四张与
    # 原 best_dets 一字不差，人已标好的格位号因此一个都不会错位。后面几张是
    # 备选：顶上来的可能还是同一个错的人，所以备选也必须看得见、标得了；只
    # 自动顶上不给看，等于把一张没人验过的脸送进模型。
    judge_cands = judge_candidates(dets, frame_w, frame_h)
    judge_sheets: dict[int, list[dict]] = {}
    for idx, c in enumerate(judge_cands):
        pool = c["pool_dets"]
        sheets = []
        for start in range(0, len(pool), CELL_COUNT):
            chunk = pool[start:start + CELL_COUNT]
            png = build_evidence_sheet(
                [d.frame_path for d in chunk],
                [(d.x1, d.y1, d.x2, d.y2) for d in chunk],
                label_cells=True, cell_labels=list(
                    range(start + 1, start + len(chunk) + 1))) if chunk else None
            img_candidates.append({"data": png, "media": "image/png"})
            refs.append((0, f"judge{idx}", len(sheets)))
            sheets.append({"cells": list(range(start + 1, start + len(chunk) + 1)),
                           "_ref": (0, f"judge{idx}", len(sheets))})
        judge_sheets[idx] = sheets

    n_img, img_bytes, n_stale = _write_review_images(out_path, refs, img_candidates)
    img_notice = (
        f'<p class="imgNotice">本页的图<b>引用同目录下的 {REVIEW_IMG_DIR}/</b>'
        f'（{n_img} 张，{img_bytes / 1048576:.1f} MB，与 VLM 实际拿到的是同一批'
        f'字节，未经任何压缩）。页面因此<b>不是自包含的单文件</b>——转交他人'
        f'标注请把整个目录一起打包。</p>')
    print(f"   图片 {n_img} 张已写入 {REVIEW_IMG_DIR}/"
          f"（{img_bytes / 1048576:.1f} MB，原图未压缩）"
          + (f"，清掉上一轮遗留 {n_stale} 张" if n_stale else ""), flush=True)
    imgs_by_ref = dict(zip(refs, img_candidates))

    def _img_fields(img: dict) -> dict:
        # `img` 现在是**相对页面的路径**，不再是 base64。页面侧一律经 `imgTag()`
        # 取用——四处各拼一遍 `<img>` 必然日后走样，而走样的表现是某一类图
        # 静默变成叉。
        return {"img": img.get("src") or ""}

    segments_json: list[dict] = []
    for segment in segments:
        occupied = len(seats_by_order.get(segment.order, {}))
        missing = max(0, segment.presenter_num - occupied)
        cand_list = []
        for idx, cand in enumerate(per_segment_candidates[segment.order]):
            img = imgs_by_ref[(segment.order, "cand", idx)]
            cand_list.append({
                "kind": cand["kind"],
                "track_id": cand["track_id"],
                # Task 46：导出与回填两侧都以 stable_id 为准，track_id 只
                # 作人工排查的参考信息（重检测后它全部改变）。
                "stable_id": cand["stable_id"],
                "n_frames": cand["n_frames"],
                "median_h": round(cand["median_h"], 1),
                "median_cx": round(cand["median_cx"], 1),
                # Task 49 · 需求三：位置排除按 (cx, y2) 判定「同一把椅子」，
                # 页面要在本地算出「这条排除会波及多少个候选」的预览，
                # 因此 y2 也必须随候选一起送到页面上。
                "median_y2": round(cand["median_y2"], 1),
                # Task 54：这个候选的证据图上**真正有画面**的格数。不足 4
                # 张时余下的是灰块，那些格位号 `entryexit.cell_dets` 一律
                # 返回空（宁可不补，不拿相邻那一格另一个人的画面顶上）。
                # 页面据此收窄「格位」下拉、并把超出的行排除在变绿判定之外
                # ——不带这个数字，用户能选一个根本没有画面的格位，页面算
                # 作「已确定」而流水线一个人也补不进去。
                "n_cells": min(CELL_COUNT, len(cand["best_dets"])),
                **_img_fields(img),
            })
        filtered_list = []
        for idx, cand in enumerate(per_segment_filtered[segment.order]):
            img = imgs_by_ref[(segment.order, "filtered", idx)]
            filtered_list.append({
                "kind": cand["kind"],
                "track_id": cand["track_id"],
                "n_frames": cand["n_frames"],
                "median_h": round(cand["median_h"], 1),
                "aspect": round(cand["aspect"], 2),
                # 与展示候选同一套字段（用户 2026-08-25：城市2 order 4 那个人
                # 就在折叠区里）。缺 stable_id 就无从存标注、重检测后也无从按
                # 视频内容重新对齐；缺 median_cx/y2 位置排除算不了；缺 n_cells
                # 格位下拉收不了口。
                "median_cx": round(cand["median_cx"], 1),
                "median_y2": round(cand["median_y2"], 1),
                "stable_id": cand["stable_id"],
                "n_cells": min(CELL_COUNT, len(cand["best_dets"])),
                # Task 52 需求三：被哪条规则挡的，逐条如实带上——用户看折叠
                # 区就是为了核实「这条规则有没有挡错人」，只给图不给原因，
                # 他没法判断该调哪个阈值。
                "height_ratio": round(cand["height_ratio"], 3),
                "filter_reasons": list(cand["filter_reasons"]),
                **_img_fields(img),
            })
        filled_list = []
        for idx, row in enumerate(per_segment_filled[segment.order]):
            sh = imgs_by_ref[(segment.order, "filledsheet", idx)]
            filled_list.append({"seat": row["seat"],
                                 "sheet": sh.get("src") or "",
                                 "n_cells": len(row.get("sheet_dets") or [])})
        # 尚空的座位号：1..presenter_num 里不在已定集合中的那些。
        # 座位号下拉只列这些 +「不确定」——此前可以指到已被占用的座位上，
        # 而 entryexit_fill_plan 会跳过它（不覆盖已占座位），用户标了却没
        # 生效且无任何提示。收窄选项从源头堵住这类静默失效。
        taken = set(seats_by_order.get(segment.order, {}).values())
        vacant = [k for k in range(1, segment.presenter_num + 1) if k not in taken]

        totals = per_segment_totals[segment.order]
        segments_json.append({
            "order": segment.order,
            "company": segment.company,
            "presenter_num": segment.presenter_num,
            # 各座位的**人工标注**性别（Presenter.gender_female，1=女）。
            # 必须与 VLM 的 gender_judged 分清：本项目曾把性别当「已知事实」
            # 喂进 PROMPT，导致模型判定失去独立性、据此做的一批校验全部作废。
            # presenters 为空（部分测试夹具）时给空字典，页面据此不显示性别。
            "seat_genders": {
                str(pr.seat_index): ("女" if pr.gender_female else "男")
                for pr in (segment.presenters or ())
            },
            "occupied": occupied,
            "missing": missing,
            "filled_seats": filled_list,
            "vacant_seats": vacant,
            "candidates": cand_list,
            "entry_total": totals["entry"],
            "exit_total": totals["exit"],
            "top_n": config.ENTRYEXIT_TOP_N,
            "filtered": filtered_list,
            "filtered_count": len(filtered_list),
            "min_aspect": config.ENTRYEXIT_MIN_ASPECT,
            "min_height_ratio": config.ENTRYEXIT_MIN_HEIGHT_RATIO,
            # 两个计数按「触发了这条规则」计，一个候选可能两条都触发，因此
            # 两数之和可能大于 filtered_count——折叠区的文案会把这点说清楚，
            # 不用互斥归类掩盖掉「这张两条都占」的事实。
            "filtered_aspect_count":
                sum(1 for c in filtered_list if "aspect" in c["filter_reasons"]),
            "filtered_height_count":
                sum(1 for c in filtered_list if "height" in c["filter_reasons"]),
        })

    # Task 52 需求一：一律按 order 升序。排序键必须与标注进度无关——按缺口
    # 排会让段落在用户边标边动（见 docstring）。缺人状态由目录的颜色与
    # ⛔/⚠/✓ 标记承担。
    segments_json.sort(key=lambda s: s["order"])

    safe_sheet = config.sanitize_filename_component(config.SHEET)

    # Task 54 需求一：既有标注里的备注 → 格位选择的预填表（只读，不写文件）。
    if entry_labels is None:
        if labels_path is None:
            labels_path = out_path.parent / f"entry_labels_{safe_sheet}.json"
        # 配对池必须是**展示 ∪ 折叠区**（`candidates_for_labels`），与
        # `registry` / `run_all` 逐字同一份。此前这里传的是展示候选
        # `per_segment_candidates`——它还套着 ENTRYEXIT_TOP_N 的**展示上限**，
        # 拿它配对会把标在折叠区、或重检测后滑出前 8 的那些一律判为配不上。
        #
        # 用户 2026-09-08 实测城市1：同一次运行里挑选页算出「50 条只对上 27
        # 条（54%），已作废」，登记表算出「对上 36 条（72%），保留」。整份被
        # 判作废之后 `FILE_LABELS` 变空，页面上 `hydrateCellsFromFile` 就恢复
        # 不了用户逐格选过的 `cells`——那份工作只剩在文件里，一旦从页面重新
        # 导出就没了，而且一处不报错。
        entry_labels = load_entry_labels(
            Path(labels_path),
            detections_fp=detections_fp or None,
            candidates_by_order={seg.order: candidates_for_labels(dets, seg,
                                                                  frame_w, frame_h)
                                 for seg in segments},
        )
    # 重抽之后把挪了位的评审人标注对回当前候选（见 `judges.realign_judge_labels`）。
    # 只在登记表那一侧对齐是不够的：人打开页面看到「没人标过」会重标一遍，而
    # 登记表里其实还留着旧的，两边各说各话且一处不报错。
    entry_labels, _jstat = realign_judge_labels(judge_cands, entry_labels)
    if _jstat["total"]:
        _m = f"   评审席标注：{_jstat['matched']}/{_jstat['total']} 条对上"
        if _jstat["moved"]:
            _m += f"（其中 {_jstat['moved']} 条按位置重新对齐）"
        if _jstat["dropped"]:
            _m += f"；{len(_jstat['dropped'])} 条找不到对应候选已丢弃：{_jstat['dropped']}"
        if _jstat["ambiguous"]:
            _m += f"；{len(_jstat['ambiguous'])} 条同时贴近两位候选、拒绝硬配：{_jstat['ambiguous']}"
        print(_m, flush=True)

    # 两位被指到同一号时**不能让页面渲染不出来**：挑选页正是改这个号的唯一
    # 地方，页面出不来 → 改不了 → 页面永远出不来（用户 2026-09-19 实测电子27
    # 卡在这里）。冲突的那几位降级为「未指号」（回到几何基线），其余人的号
    # 一字不动，并把冲突写在页面上。登记表那一侧继续硬拒——那里产出一份猜出来
    # 的座次就是静默出错。
    _conflicts = judge_number_conflicts(judge_cands, entry_labels)
    judge_conflict_note = ""
    if _conflicts:
        # 说**页面上看得到的东西**：几何基线第几位、位置、帧数。只印内部标识
        # 的话人无从下手——用户 2026-09-19 的原话是「人工不知道是哪两个 track
        # 冲突了」，而评审人候选是跨段按位置归并的，每一位由十来个 (段, track)
        # 拼成，根本没有「那个 track」可指。
        _at = {c["stable_id"]: (i, c) for i, c in enumerate(judge_cands, 1)}

        def _who(sid):
            hit = _at.get(sid)
            if hit is None:
                return sid
            i, c = hit
            return (f"几何基线第 {i} 位（位置 {c['median_cx']:.0f},"
                    f"{c['median_y2']:.0f}，{c['n_frames']} 帧）")

        parts = "；".join(
            f"专家 {n}：" + "、".join(_who(sid) for sid in sids)
            for n, sids in sorted(_conflicts.items()))
        judge_conflict_note = (
            f'<p class="judgeConflict">⚠ 有候选<b>同时被指为</b>同一个号，'
            f'这几位的号已暂时取消、退回几何基线，请在下面改清楚：{_html.escape(parts)}</p>')
        print(f"   ⚠ 评审席有重号，已暂时取消这几位的号：{parts}", flush=True)
        entry_labels = without_conflicting_numbers(judge_cands, entry_labels)

    slot_of = {}
    for n, c in judge_slots(judge_cands, entry_labels).items():
        if c is not None:
            slot_of[c["stable_id"]] = n
    judges_json = {
        "n_expected": config.JUDGE_COUNT,
        "not_a_judge": config.JUDGE_ROLE_NOT_A_JUDGE,
        "cell_nobody": judges_mod.JUDGE_CELL_NOBODY,
        "candidates": [
            {"stable_id": c["stable_id"],
             # 几何基线给的号（离镜头由近及远）；人工改号压过它。
             "judge": slot_of.get(c["stable_id"]),
             "median_cx": round(c["median_cx"], 1),
             "median_y2": round(c["median_y2"], 1),
             "median_h": round(c["median_h"], 1),
             "height_ratio": round(c["height_ratio"], 3),
             "n_frames": c["n_frames"],
             "n_segments": c["n_segments"],
             # 池里真正有画面的格数：没有帧的格不给标——标了也没有帧可丢，
             # 人却会以为自己已经核实过（与创业者候选的 `n_cells` 同一条口径）。
             "n_cells": len(c["pool_dets"]),
             "sheets": [{"cells": sh["cells"],
                         **_img_fields(imgs_by_ref[sh["_ref"]])}
                        for sh in judge_sheets[idx]],
             **_img_fields(imgs_by_ref[judge_sheets[idx][0]["_ref"]])}
            for idx, c in enumerate(judge_cands)
        ],
    }

    file_labels = _file_label_prefill(entry_labels)

    html = _TEMPLATE.replace("__IMG_SCALE__", str(config.GALLERY_IMG_SCALE))
    # 目录宽度只此一处来源，CSS 里的左栏宽与主内容左边距都由它算出，
    # 两处各写一份数字必然日后走样。
    html = html.replace("__TOC_W__", str(config.ENTRY_REVIEW_TOC_WIDTH_PX))
    # 正文左边距 = 目录宽 + 间隙。固定定位的目录不占文档流，body 不让位就会
    # 盖住候选图——而候选图正是本页的全部价值所在。
    html = html.replace("__TOC_PAD__",
                        str(config.ENTRY_REVIEW_TOC_WIDTH_PX + 24))
    html = html.replace("__SEGMENTS__", json.dumps(segments_json, ensure_ascii=False))
    html = html.replace(
        "__STORAGE_KEY__",
        json.dumps(f"dress_entry_review_v2::{config.SHEET}", ensure_ascii=False),
    )
    html = html.replace(
        "__EXPORT_FILENAME__",
        json.dumps(f"entry_labels_{safe_sheet}.json", ensure_ascii=False),
    )
    html = html.replace("__DETECTIONS_FP__", json.dumps(detections_fp))
    # 导出的来源标记，见 `page_code_fp`。生成时刻只给人读，判定一律看指纹。
    html = html.replace("__PAGE_CODE_FP__", json.dumps(page_code_fp()))
    html = html.replace(
        "__PAGE_BUILT_AT__",
        json.dumps(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # 格位约定的文案由 annotate 按**实际绘制顺序**算出，页面不另写一份
    # 数字——两处各写一份，改了绘制顺序而没改文案，用户会按错号写备注，
    # 而且没有任何一处会报错。
    html = html.replace("__CELL_MAP__",
                         json.dumps(cell_label_convention_text(), ensure_ascii=False))
    # Task 53 需求五：导出提示条的淡出时长，唯一来源在 config
    # （CSS 过渡时长与 setTimeout 都由它算出，不各写一份数字）。
    html = html.replace("__TOAST_MS__", json.dumps(config.ENTRY_REVIEW_TOAST_MS))
    html = html.replace("__SHEET_NAME__", json.dumps(config.SHEET, ensure_ascii=False))
    html = html.replace("__IMG_NOTICE__", img_notice)
    # Task 54：格位身份下拉的选项与格位总数都取自 entryexit，页面不另写一份
    # ——页面能选的角色与 `entryexit_fill_plan` 认得的角色一旦错开，表现只是
    # 「选了没反应」，没有任何一处会报错。
    html = html.replace("__CELL_ROLES__",
                         json.dumps(list(CELL_ROLE_CHOICES), ensure_ascii=False))
    html = html.replace("__MAX_CELLS__", json.dumps(CELL_COUNT))
    html = html.replace("__JUDGES__", json.dumps(judges_json, ensure_ascii=False))
    html = html.replace("__JUDGE_CONFLICT__", judge_conflict_note)
    html = html.replace("__FILE_LABELS__", json.dumps(file_labels, ensure_ascii=False))
    # Task 56C：文件里那份「本座位无可用画面」回填到页面。值一律归一成
    # int 列表——页面 JS 用 `indexOf(seat)` 判断（seat 来自 SEGMENTS 里的
    # int），字符串 "3" 与数字 3 在 indexOf 里不相等，勾选框会显示成空的，
    # 而且不报错。
    # 上次导出的评审席标记（不是评审人 / 逐格归属 / 补选）。页面只作**回填**：
    # localStorage 里没有这一项时才用它。用户 2026-09-21 实测电子28：标了
    # 「不是评审人」的那两位，换个浏览器/清过站点数据之后原样回到席位上，
    # 页面上没有任何痕迹说他们曾被否决——文件里其实写着，只是没人送回来。
    html = html.replace(
        "__FILE_JUDGES__",
        json.dumps((entry_labels or {}).get("judges") or {}, ensure_ascii=False))
    html = html.replace("__FILE_NO_CANDIDATE__",
                        json.dumps(_file_no_candidate(entry_labels),
                                   ensure_ascii=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def _file_no_candidate(entry_labels: dict | None) -> dict[str, list[int]]:
    """既有标注文件里的 `no_candidate_seats` → 页面回填表 `{"<order>": [座位号]}`。

    解析只在这一处（`entryexit.no_candidate_seats_of`），页面不另写一份 JS
    版的清洗逻辑——两处口径一旦错开，症状只是「勾选框看起来是空的」，
    没有任何一处会报错。
    """
    raw = (entry_labels or {}).get("no_candidate_seats") or {}
    out: dict[str, list[int]] = {}
    for order_key in raw:
        try:
            order = int(order_key)
        except (TypeError, ValueError):
            continue
        seats = sorted(no_candidate_seats_of(entry_labels, order))
        if seats:
            out[str(order)] = seats
    return out


def _file_label_prefill(entry_labels: dict | None) -> dict[str, dict]:
    """既有标注文件 → 页面用的预填表 `{"<order>::<稳定标识>": {...}}`。

    每项三个字段：

      · `note`        备注原文（页面只读展示；Task 54 起它不再参与补位判定）
      · `note_cells`  这条备注解析出来的格位选择（`entryexit.note_to_cells`），
                      页面据此预填控件并提示「已由备注自动填充，请核对」
      · `note_unparsed` 没能解析的短语原文——凡有未解析必须可见，用户据此
                      知道该手动补哪一格
      · `cells`       文件里**已有**的格位选择，页面用它回填 store
                      （`hydrateCellsFromFile`，理由见 `render_entry_review`）

    键的拼法与页面 JS 的 `candKey()` 完全一致（`f"{order}::{stable_id}"`）；
    两处各拼一份就会全表对不上任何候选，而症状只是「预填没生效」。

    解析只发生在这里（Python 侧那一个 `parse_cell_note`），页面不另写一份
    JS 解析器。
    """
    out: dict[str, dict] = {}
    for seg_order, seg_labels in (entry_labels or {}).items():
        if seg_order in LABEL_RESERVED_KEYS or not isinstance(seg_labels, dict):
            continue
        for sid, rec in seg_labels.items():
            if not isinstance(rec, dict):
                continue
            note = str(rec.get("note") or "")
            cells = rec.get("cells") if isinstance(rec.get("cells"), list) else []
            if not note and not cells:
                continue
            report = (parse_cell_note_report(note) if note
                      else {"parsed": {}, "unparsed_phrases": []})
            out[f"{seg_order}::{sid}"] = {
                "note": note,
                "note_cells": note_to_cells(note) if note else [],
                "note_unparsed": list(report["unparsed_phrases"]),
                "cells": cells,
            }
    return out


_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>进出场站立候选复核</title>
<style>
 :root{--img-scale: __IMG_SCALE__}
 body{font-family:-apple-system,"PingFang SC",sans-serif;margin:24px;background:#fafafa;color:#222}
 h2{border-bottom:2px solid #ddd;padding-bottom:6px;margin-top:36px}
 .seg-missing{color:#a33}
 .seg-complete{color:#888}
 .trunc-notice{background:#fff8e1;border-left:4px solid #e0a030;padding:6px 10px;
               font-size:13px;margin:6px 0}
 .cand{display:flex;gap:12px;background:#fff;border:1px solid #ddd;border-radius:8px;
       padding:10px;margin:8px 0;align-items:flex-start}
 /* 与画廊同一口径（Task 36）：按原尺寸的百分比缩放，不设像素上限。
    原为 calc(var(--img-scale) * 320px)，把 800x800 的证据图压到 256px——
    而本页的全部价值就在于让用户一眼认出候选是谁，压到那个尺寸就认不出了。
    height:auto 保持长宽比，避免方形证据图被压扁。*/
 .cand img{width:calc(var(--img-scale) * 100%);max-width:100%;
           height:auto;border-radius:4px;display:block}
 .cand .imgwrap{flex:0 0 62%;min-width:0}
 .noimg{width:100%;aspect-ratio:1/1;background:#eee;border-radius:4px}
 .info{flex:1;min-width:0}
 .meta{color:#666;font-size:13px;margin-bottom:6px}
 /* Task 46（用户 2026-08-21）：「因为 html 图片足够高，右侧选项一个选项
    一列更美观」——图片区已占 62% 宽、高度足够，右侧竖排每个选项独占一行
    更好读。原为 `.roles label{margin-right:14px}` 的横排。角色、座位号、
    人数三处风格一致：各自独占一行、标签左对齐。字号沿用 Task 36 的下限
    （≥14px，不许因为排版改动把字缩回去）。*/
 .roles label{display:block;font-size:14px;line-height:1.7;text-align:left}
 .roles label input{margin-right:6px}
 .filledBox{background:#eef6ee;border-left:4px solid #4a8c4a;padding:10px 12px;
            margin:8px 0;font-size:14px;border-radius:6px}
 /* 一座一行（用户 2026-09-19：「自动标好的图不够大，导致无法辨别，要求
    自动识别好的大小和其他候选图大小一致」）。原先是 180px 定宽、横排换行，
    2×2 缩到那个尺寸根本看不清衣服——而这一栏恰恰是要人核对「机器定的这个
    座位对不对」。
    尺寸不在这里定：卡片挂 `cand` 类、图挂 `imgwrap`，与创业者候选**共用同
    一条规则**。另抄一份百分比必然日后走样——2026-09-18 评审人卡片已经因此
    改过一次（2871f3a），当时只改了被点名的那一处，这一块漏了。*/
 .fsRow{margin-top:8px}
 .filledSeat{border-color:#cfe3cf}
 /* 改号（Task 56V）。选中后整块变色——它会改变这张图归谁，比「是否替换」
    更重，必须一眼看得出这张卡片被动过。 */
 .rsBox{margin-top:6px;font-size:12px;line-height:1.5}
 .rsOn{background:#fff3cd;border:1px solid #d39e00;border-radius:6px;padding:4px 6px}
 .rsHint{display:block;color:#666;font-size:11px;margin-top:2px}
 .seatReassign{font-size:12px;max-width:100%}
 .filledSeat img{width:100%;height:auto;border-radius:4px;display:block}
 .fsSeat{font-weight:700;font-size:13px;margin-bottom:4px}
 .fsNote{font-size:12px;color:#555;margin-top:2px}
 .fsSheet{margin-top:6px}
 .fsSheet img{width:100%;height:auto;border-radius:4px;display:block}
 .exRow{display:flex;flex-wrap:wrap;gap:4px 10px;margin-top:4px}
 .exCell{font-size:12px;white-space:nowrap}
 .seatWrap{margin-top:8px}
 .seatWrap label{display:block;font-size:14px;text-align:left}
 .seatWrap select{font-size:14px;padding:4px 6px;display:block;margin-top:4px}
 .peopleWrap{margin-top:8px}
 .peopleWrap label{display:block;font-size:14px;text-align:left}
 .peopleWrap select{font-size:14px;padding:4px 6px;display:block;margin-top:4px}
 .multiHint{background:#fdecec;border-left:4px solid #c0392b;padding:6px 10px;
            font-size:13px;margin-top:6px}
 /* Task 49 · 需求一：格位约定紧贴证据图下方——用户看图时就能看到编号
    含义，不必回头找说明。 */
 .cellMap{font-size:13px;color:#444;background:#f0f4f8;border-radius:4px;
          padding:4px 8px;margin-top:6px}
 /* Task 49 · 需求二：已定座位的否决/替代控件 */
 .ovrBox{margin-top:6px;text-align:left;border-top:1px dashed #9bb39b;padding-top:6px}
 .ovrBox label{display:block;font-size:14px;text-align:left;line-height:1.6}
 .ovrBox select,.ovrBox input{font-size:14px;width:100%;box-sizing:border-box;
                              margin-top:3px}
 .ovrOn{background:#fdecec;border-left:4px solid #c0392b;padding:4px 6px;
        border-radius:4px}
 /* Task 56C：尚未确定的座位 +「本座位无可用画面」勾选项。
    底色刻意与 .filledBox（已定，绿）拉开——这一栏说的是「还没定下来的」，
    与已定那栏并排出现，同色会让人以为这些座位也定好了。 */
 .ncBox{background:#fff7e6;border-left:4px solid #d98324;padding:10px 12px;
        margin:8px 0;font-size:14px;border-radius:6px}
 .ncRow{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:6px}
 .ncSeat{font-size:14px;white-space:nowrap}
 .ncOn{background:#f0e6f6;border-radius:4px;padding:1px 6px;font-weight:700}
 .ncHint{font-size:13px;color:#555;margin-top:6px;line-height:1.6}
 /* Task 53 · 需求二：标了创业者却没指定座位号。用与 .multiHint 同一套
    告警红——它和「这个框挤了多人」是同一类问题：**标了却补不进去**，而
    页面此前对它一声不吭（实测文化4组有 3 个这样的候选）。颜色之外另有
    文字（「未指定座位号」「不会补位」），不只靠颜色。 */
 .seatMissing{background:#fdecec;border-left:4px solid #c0392b;padding:6px 10px;
              font-size:13px;margin-top:6px;color:#8a1f14;font-weight:700}
 .multiHint textarea{width:100%;box-sizing:border-box;margin-top:6px;font-size:13px;
                      min-height:40px}
 /* Task 54 · 需求一：格位分配的选择题。一行三个控件（格位 / 身份 / 座位号），
    字号沿用 Task 36 的下限（≥14px，不许因为多塞了一个控件把字缩回去）。 */
 .ncAt{font-size:12px;padding:3px 6px;border:1px solid #c8a54a;border-radius:4px;
   margin:0 10px 0 -4px;min-width:210px}
 .judgeConflict{background:#fdecea;border-left:4px solid #c62828;padding:8px 12px;
   border-radius:6px;margin:8px 0}
 .judgeBox{background:#eef4ff;border:1px solid #b9cdf0;border-radius:8px;
   padding:10px 14px;margin:10px 0}
 /* 一位一行（用户 2026-09-18：「排成了一行，太小不容易分辨，一个专家一列，
    大小参考展示创业者的2*2图」）。尺寸**不在这里定**——卡片直接挂 `.cand`
    类，与创业者候选图共用同一条宽度规则；另抄一份必然日后走样，而走样的
    表现正是上一版：两边看着像一回事，实际一个 220px 定宽、一个 62%。 */
 .judgeRow{margin-top:8px}
 .judgeCard{border-color:#d6e0f2}
 .judgeDropped{background:#fff4e5;border-color:#e0b072}
 .judgeCard .meta{font-size:12px;color:#555;margin:4px 0}
 /* 尺寸只此一处来源：复用创业者候选那一列的 .imgwrap（flex:0 0 62%）。
    另抄一份百分比必然日后走样，而走样的表现是两边看着像一回事、实际一个
    62% 一个别的——2026-09-18 已经因此改过一次。这里只补上「卡片改成纵向
    堆叠之后 flex-basis 不再生效」那一点。*/
 .jsheetImg{margin:6px 0;flex:none;width:62%}
 .jspare{margin-top:8px;border-left:3px solid #d6e0f2;padding-left:8px}
 .jspare summary{cursor:pointer;font-size:12px;color:#456}
 .jcells{margin-top:6px;font-size:12px;color:#444}
 .jcell{display:inline-block;margin-right:12px;margin-top:4px}
 .jpick{display:inline-block;margin-left:4px;color:#456}
 .jcell select{font-size:12px}
 .jcellHint{color:#777;margin-top:4px}
 .judgeCard label{display:block;margin-top:4px}
 .judgeNum{font-weight:bold;font-size:15px;margin-bottom:4px}
 .judgeEmpty{background:#fff7e6;border:1px dashed #d9a441;
   border-radius:8px;padding:10px;margin:8px 0;font-size:13px;color:#7a5200}
 .judgeExtra{border-color:#d9a441;background:#fffdf5}
 .imgNotice{background:#fff3cd;border:1px solid #e0c86a;border-radius:6px;
   padding:8px 12px;margin:8px 0;font-size:13px;color:#6a4b00}
 .cellsBox{background:#fff;border:1px solid #d9c2c2;border-radius:4px;
           padding:6px 8px;margin-top:6px}
 .cellRow{display:flex;flex-wrap:wrap;gap:10px;align-items:center;
          padding:4px 0;border-bottom:1px dashed #eee}
 .cellRow:last-child{border-bottom:0}
 .cellRow label{font-size:14px;display:inline-flex;align-items:center;gap:4px}
 .cellRow select{font-size:14px;padding:3px 5px}
 /* 「同格位X」那一行的座位号是**继承**来的、下拉禁用：灰字明确表示
    「这里不是你能改的」，改了就会与被引用格矛盾。 */
 .cellRow select:disabled{color:#888;background:#f2f2f2}
 .cellInherit{font-size:13px;color:#666}
 .cellsPrefill{font-size:13px;color:#1b5e20;background:#e7f6e7;border-radius:3px;
               padding:3px 6px;margin-bottom:4px}
 /* 备注原文：只读展示，Task 54 起不再参与补位判定，但绝不隐藏——解析不出
    的那些全靠用户照着原文手动补选。 */
 .noteLegacy{font-size:13px;color:#333;background:#f7f7f7;border-left:3px solid #999;
             border-radius:3px;padding:5px 8px;margin-top:6px;line-height:1.7}
 .noteLegacy code{background:#fff;border-radius:3px;padding:1px 5px;
                  font-family:ui-monospace,Menlo,Consolas,monospace}
 .noteLegacyNo{margin-top:3px;color:#8a3b00}
 /* Task 52 需求四：备注框旁的可解析写法示例。示例本身用 <code> 排出来，
    与说明文字拉开距离——用户是照着抄的，抄错一个符号就解析不出来。 */
 /* Task 52 需求二：清除本条标注。做成弱化的次要样式（描边、非填充）——它
    是「撤销」，不该比角色单选更抢眼；但常驻可见，不藏在悬停里。 */
 .clearWrap{margin-top:8px;border-top:1px dashed #ccc;padding-top:6px;
            display:flex;align-items:center;gap:8px;flex-wrap:wrap}
 .clearBtn{margin-top:0;padding:4px 12px;font-size:13px;background:#fff;
           color:#a33;border:1px solid #a33}
 /* Task 52 需求三：折叠区里逐条写明被哪条规则挡下 */
 .filterReason{font-size:13px;color:#8a3b00;background:#fff8e1;
               border-left:3px solid #e0a030;padding:4px 8px;margin-top:4px;
               border-radius:3px}
 .kindHint{color:#555;font-size:13px;background:#eef4fb;border-left:4px solid #4a7fb5;
           padding:6px 10px;margin:6px 0}
 h3{margin-top:20px;margin-bottom:4px}
 /* 进场/退场两个标题（用户 2026-08-24：「几个字再大点，用红色，进行提醒」）。
    它们是分隔两个**时间窗口**的唯一标志——混过去就会把退场的人当成进场的人
    标，而那两拨人分属相邻的两家企业。 */
 .kindTitle{font-size:1.6rem;color:#c00;font-weight:700;
            border-bottom:2px solid #c00;padding-bottom:2px}
 .ovrHint{color:#666;font-size:12px;margin-left:6px}
 .saveHint{color:#333;font-size:12px;margin-left:10px}
 .filteredSection{margin-top:14px;background:#f3f3f3;border-radius:8px;padding:8px 12px}
 .filteredSection summary{cursor:pointer;font-size:13px;color:#555}
 .filteredCand{opacity:0.8}
 #progress-bar{font-weight:700}
 .exportbar{position:fixed;right:18px;bottom:18px;z-index:99;
            display:flex;align-items:center;gap:12px;
            background:#fff;border:1px solid #bbb;border-radius:10px;
            padding:10px 14px;box-shadow:0 2px 10px rgba(0,0,0,.18);
            font-size:14px}
 /* 给页面底部留出空间，避免最后一个候选被悬浮条遮住；左侧让出目录栏的
    宽度（__TOC_PAD__ = 目录宽 + 24px 空隙，两个数都由
    config.ENTRY_REVIEW_TOC_WIDTH_PX 算出，不各写一份） */
 body{padding-bottom:80px;padding-left:__TOC_PAD__px}
 /* ---- 企业目录（Task 47，用户 2026-08-21：「给各个企业名加个目录，且带
    颜色，一点滑到对应的标记处，方便定位和知道有哪些企业缺人」）----
    做成**左侧竖排固定栏**而非顶部条，理由见 render_entry_review docstring。
    自己可滚（60+ 家也放得下），锚在左边、导出条锚在右下角，互不遮挡。*/
 .toc{position:fixed;left:0;top:0;bottom:0;width:__TOC_W__px;z-index:98;
      overflow-y:auto;box-sizing:border-box;padding:12px 10px;
      background:#fff;border-inline-end:1px solid #ccc;font-size:13px}
 .toc-title{font-weight:700;font-size:14px;margin-bottom:2px}
 .toc-summary{color:#555;font-size:12px;margin-bottom:8px;line-height:1.5}
 .toc-row{display:block;text-decoration:none;color:#222;border-radius:6px;
          padding:5px 8px;margin-bottom:4px;border-left:4px solid transparent;
          line-height:1.45}
 /* 缺人状态配色。颜色之外一律另有文字标记（.toc-mark 的 ●●/●/○）——
    只靠颜色对色觉障碍者不可用，截图与打印后也失效。 */
 .toc-miss2{background:#fdecea;border-left:4px solid #c62828}
 .toc-miss1{background:#fff5e6;border-left:4px solid #e08a00}
 .toc-ok{background:#f4f4f4;border-left:4px solid #9e9e9e;color:#666}
 .toc-done-all{background:#e7f6e7;border-left:4px solid #2e7d32;color:#1b5e20}
 /* Task 56C 的第三态：「剩下的座位都已人工判定为没有可用画面」。
    紫灰色，与「已定（绿）」「缺人（红/橙）」「不缺人（灰）」四者都不同——
    用户靠目录颜色决定还要标哪些，这一档的语义是「这段人工已经看完了，
    但缺的那几个得靠加密抽帧再找」，与「还没标」必须一眼分得开。 */
 .toc-nocand{background:#f3ecf7;border-left:4px solid #7b4fa0;color:#4a2d63}
 .toc-nocand-n{display:block;font-size:12px;color:#5b3a75;font-weight:700;margin-top:2px}
 .toc-sex{display:block;font-size:12px;color:#555;margin-top:2px}
 /* Task 53 · 需求二：目录里的「标了创业者却没写座位号」计数 */
 .toc-warn{display:block;font-size:12px;color:#b3261e;font-weight:700;margin-top:2px}
 .seg-done-all{background:#e7f6e7;padding:6px 10px;border-radius:6px}
 .toc-current{outline:2px solid #4a7fb5;outline-offset:-2px}
 .toc-mark{display:inline-block;width:1.2em}
 .toc-name{font-size:13px}
 .toc-nums{display:block;font-size:12px}
 .toc-done{display:block;font-size:12px;color:#666;font-weight:400}
 /* 三档语义色。颜色**从来不是唯一区分**——每行还带图标（⛔/⚠/✓）、文字
    （缺2人/缺1人/不缺人）与字重差异，缺一不可：只靠颜色对色觉障碍者不可
    用，打印或截图成灰度后也全部失效，而这页正是边看边标、常要截图存证。*/
 .toc-short2{color:#b3261e;background:#fdecea;border-left-color:#b3261e;font-weight:700}
 .toc-short1{color:#8a5000;background:#fff5e0;border-left-color:#d98324;font-weight:600}
 .toc-ok{color:#5a6b5a;background:#f2f5f2;border-left-color:#9bb39b;font-weight:400}
 /* 当前所在企业：加粗描边 + 左侧箭头（同样不只靠颜色） */
 .toc-current{outline:2px solid #4a7fb5;outline-offset:-2px}
 .toc-current .toc-name::after{content:" ◀"}
 /* 窄屏（分屏、投影）下固定栏会把正文挤没，退回普通块级元素排在页首 */
 @media (max-width:900px){
   .toc{position:static;width:auto;max-height:40vh;border-right:0;
        border-bottom:1px solid #ccc}
   body{padding-left:0}
 }
 button{padding:8px 18px;font-size:14px;cursor:pointer;border-radius:6px;
        border:1px solid #444;background:#222;color:#fff;margin-top:16px}
 /* Task 53 · 需求五：导出后的页内提示条。用户原话：「再次导出的时候，
    没有任何提示，应该弹出提示」。
    **不用浏览器原生的模态弹窗**：它阻塞主线程、样式不可控、连续导出会
    叠一串框，还可能被弹窗拦截器吃掉——那就又回到「没有任何提示」。
    锚在导出按钮正上方（右下角），眼睛不用移开就能看到。默认
    `opacity:0` + `pointer-events:none`，加上 .toast-show 才显现，
    __TOAST_MS__ 毫秒后去掉这个类自行淡出。 */
 .toast{position:fixed;right:18px;bottom:78px;z-index:100;max-width:420px;
        background:#1b5e20;color:#fff;border-radius:8px;padding:10px 14px;
        font-size:13px;line-height:1.6;box-shadow:0 2px 12px rgba(0,0,0,.28);
        opacity:0;pointer-events:none;transition:opacity .25s ease}
 .toast-show{opacity:1}
</style>
<!-- 企业目录：常驻左栏。14 家企业里 11 家缺人时整页很长，没有导航很难定位；
     用户要标 60+ 组，导航的价值会放大。锚点用 <a href="#seg-N"> 而非 JS 滚动
     ——锚点在打印、右键新窗口、浏览器前进后退里都成立。 -->
<nav class="toc">
  <!-- 评审席排在企业目录**之前**（用户 2026-09-18）：它是全组共用的一块、挂在
       页首，也是最先要核的东西；排在十几行企业后面就要滚到底才看得见。 -->
  <div id="toc-judges"></div>
  <div class="toc-title" style="margin-top:10px">企业目录</div>
  <div class="toc-summary" id="toc-summary"></div>
  <div id="toc-list"></div>
</nav>
<h1>进出场站立候选复核</h1>
__IMG_NOTICE__
__JUDGE_CONFLICT__
<div id="judges"></div>
<p>进出场窗口占抽帧量的大半，其中站立候选可能是被前排遮住、就座画面编不了码的
创业者，也可能是中途走动的工作人员——请逐个候选判断。改动自动存入浏览器本地。</p>
<div id="main"></div>
<!-- 悬浮条：导出按钮与进度常驻可见（用户 2026-08-17）。候选按段分组、缺人段
     在前，整页往往很长，原先导出按钮只在页尾，标到一半想先存一份得滚到底。
     标注只存在 localStorage 里，浏览器一清就没了，随时可导也降低丢失风险。-->
<div class="exportbar">
  <span id="progress-bar"></span>
  <button onclick="exportLabels()">导出标注 JSON</button>
  <!-- 存到哪，必须写在按钮旁边（Task 56I）。浏览器只能弹「另存为」、选不了
       默认目录；用户多半会让它落进「下载」文件夹，而流水线在
       work/Output/<组>/ 那里找（Task 56G）。找不到的表现是「一条标注都没有」
       ——流水线照常跑完、结果全退回就座基线，一处不报错。这一句是这条链上
       唯一能拦住它的地方。 -->
  <span class="saveHint">↳ 存到本页所在的 <b>work/Output/&lt;组名&gt;/</b> 目录，
    覆盖同名文件即可</span>
</div>
<script>
const SEGMENTS=__SEGMENTS__;
const KEY=__STORAGE_KEY__;
const EXPORT_FILENAME=__EXPORT_FILENAME__;
// Task 46：本页候选所依据的那份检测结果的指纹，随导出写进 _meta，供
// load_entry_labels 判断「这份标注是不是这份检测的」。
const DETECTIONS_FP=__DETECTIONS_FP__;
// 导出这份标注的**页面代码**的指纹（见 Python 侧 `page_code_fp`）。
// 检测指纹标不出「页面换了新版」——重出页面时检测没重跑。
const PAGE_CODE_FP=__PAGE_CODE_FP__;
const PAGE_BUILT_AT=__PAGE_BUILT_AT__;
const SHEET_NAME=__SHEET_NAME__;
// Task 49 · 需求一：格位约定文案由 Python 侧按 build_evidence_sheet 的
// **实际绘制顺序**算出（annotate.cell_label_convention），页面不另写死
// 一份数字——约定与实际反过来，用户会按错号写备注、重新切分张冠李戴。
const CELL_MAP=__CELL_MAP__;
// Task 53 · 需求五：导出提示条显示多久后淡出（config.ENTRY_REVIEW_TOAST_MS）。
const TOAST_MS=__TOAST_MS__;
// 六个角色（Task 41 改动二）：后两个是相邻项目的人——进场窗口是段首前
// 25 秒，退场窗口是段尾后 30 秒，都可能录到上一组还没走完/下一组已经
// 进场的人。entryexit_fill_plan 只对精确等于「创业者」的标注补位，这
// 两个新角色与工作人员/专家/不确定一样一律不参与补位。
const ROLES=['创业者','工作人员','专家','不确定','上一组创业者','下一组创业者'];
// Task 54 · 需求一：格位分配（结构化选择题）用到的三份常量，全部由
// Python 侧注入，页面不另写死一份：
//   CELL_ROLES  身份下拉的选项（= entryexit.CELL_ROLE_CHOICES）
//   MAX_CELLS   2×2 证据图的格位数（= entryexit.CELL_COUNT）
//   FILE_LABELS 既有标注文件里的备注/格位选择 + 备注的解析结果，键与
//               candKey() 同一套拼法（见 entryreview._file_label_prefill）
const CELL_ROLES=__CELL_ROLES__;
const MAX_CELLS=__MAX_CELLS__;
// 评审席（组级，用户 2026-09-17）：同一批专家坐一整天，衣服不变，所以这份
// 数据不挂在任何一段底下——挂进段里同一张脸要认 N 遍。
const JUDGES=__JUDGES__;
const NOT_A_JUDGE=JUDGES.not_a_judge;
const JUDGE_CELL_NOBODY=JUDGES.cell_nobody;
const FILE_LABELS=__FILE_LABELS__;
// Task 56C：上次导出的 entry_labels 里那份「本座位无可用画面」。
// 只作**回填**用（localStorage 里没有该键时才生效）——不回填的后果
// 实测过（Task 49 那个只存 localStorage 的排除开关）：换台电脑/换个浏览器
// 打开，勾选框
// 看起来全是空的，用户会重标一遍，或者以为标记丢了。
const FILE_NO_CANDIDATE=__FILE_NO_CANDIDATE__;
// 上次导出的评审席标记，**只作回填**（见 judgeFileRec）。
const FILE_JUDGES=__FILE_JUDGES__;
const store=JSON.parse(localStorage.getItem(KEY)||'{}');

function esc(s){ return String(s??'').replace(/[&<>"']/g, c=>({
  '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
}[c])); }

// Task 46：候选的键一律用 stable_id（由「窗口类型 + 时间 + 框位置」导出，
// 重检测后仍然成立），不用 track_id——后者由 YOLO 跟踪器在检测时分配，
// 重跑检测即全部改变，用户标了 60+ 组之后会因一次重检测全部作废。
function candKey(seg,cand){ return seg.order+'::'+cand.stable_id; }

function save(ck,field,value){
  // 重新标这条候选 = 撤回之前的「清除」，墓碑让路（见 clearLabel）。
  delete store[clearedKey(ck)];
  store[ck]=store[ck]||{};
  store[ck][field]=value;
  localStorage.setItem(KEY,JSON.stringify(store));
  render();
}

// Task 52 需求二（用户 2026-08-23：「如果标了创业者之后，目前无法取消标注
// 了，如果下面有更好的图，希望取消先看到的图标注，给个取消标注的选择」）。
//
// 六个角色是 radio，选中之后 HTML 本身没有任何办法退回「全部未选」。这里
// 把整条记录 **delete 掉**，不是置成空字符串：空串与「没标过」是两回事——
// buildLabelsPayload 用 `!rec.role` 跳过，一条 role:'' 的记录在统计与导出
// 里都会被当成「标过、只是角色为空」，而用户的意思是「这条当我没标」。
//
// 必须走 render()：save() 那条路自带重渲染，删除若忘了重算，被取消的标注
// 仍会算进「已确定」、目录仍是绿的——用户会以为这段标完了。
function clearedKey(ck){ return 'cleared::' + ck; }

function mergePreload(preload){
  // PRELOAD（文件里那份）合并进 store 时**跳过被清除过的那些**。
  //
  // 用户 2026-09-09：「清除了标注也没生效」。`clearLabel` 只删 localStorage
  // 里的键，而页面每次打开都会把文件里那份盖回来——删除被悄悄撤销，页面上
  // 看不出发生过什么。这与「取消勾选『无可用画面』要写显式的 false、不能只
  // 删键」是同一条教训，当时没推广到这里。
  const out = {};
  Object.keys(preload || {}).forEach(k=>{
    if (store[clearedKey(k)]) return;      // 用户明确清除过，别盖回来
    out[k] = preload[k];
  });
  return out;
}

function clearLabel(ck){
  delete store[ck];
  // 留一个墓碑，好让下次打开页面时 PRELOAD 认得出「这条是用户删掉的」。
  // 用户重新标这条候选时它自动消失（见 save / setPeople / saveCell）。
  store[clearedKey(ck)] = {cleared: true};
  localStorage.setItem(KEY,JSON.stringify(store));
  render();
}

function seatOptions(vacant, seatVal, topUp){
  // 只列**尚空**的座位号 +「不确定」。此前列出 1..presenterNum 全部，
  // 用户可以把候选指到一个已被就座阶段占用的座位上，而
  // entryexit_fill_plan 会跳过它（不覆盖已占座位）——标了却没生效，
  // 且没有任何提示。收窄选项从源头堵住这类静默失效。
  //「不确定」始终保留：用户可能判断不出具体是几号位。
  let opts = `<option value="" ${seatVal===''?'selected':''}>不确定</option>`;
  vacant.forEach(n=>{
    // `topUp` 里的那些座位上**还坐着人**，指过去是把被勾掉的那一格补回来，
    // 不是换人。文案必须说破，否则用户以为自己在替换整个人。
    const tag = (topUp||[]).indexOf(n) >= 0 ? `${n}（补格位，不换人）` : `${n}`;
    opts += `<option value="${n}" ${String(seatVal)===String(n)?'selected':''}>${tag}</option>`;
  });
  return opts;
}

function seatOvrKey(seg, seat){ return 'seatovr::'+seg.order+'::'+seat; }

// 逐格排除的 store 键：**用 (order, seat) 而非 track_id**。座位号是人工确定
// 的、跨重检测稳定；track_id 由跟踪器分配，重跑检测即全变（Task 46 已因此把
// 标注标识改成由视频内容导出）。用 track_id 作键会让用户排掉的那一格在下次
// 重检测后静默失效，他得把每组重排一遍。
function cellExKey(seg, seat){ return 'cellex::'+seg.order+'::'+seat; }

// 已定座位改号（Task 56V，用户 2026-08-29：「城市2 order6 自动标的 seat1
// 实际为 seat2，这样自动标错的如何调整，否则标错了，无法将该图用于其他
// 座位」）。键同样是 **(order, 原座位号)**，与上面两条同源、跨重检测稳定。
//
// 键里的座位号是**这张卡片原本挂着的号**，不是改成的新号：逐格排除也是按
// 这个号存的，两者必须指同一张卡片，否则改完号之后勾掉的那一格会跑到别人
// 的排除表上，且一处不报错。
function reassignKey(seg, seat){ return 'reassign::'+seg.order+'::'+seat; }

function saveReassign(k, value){
  const v = parseInt(value, 10);
  // 空选项＝撤回。**整条 delete**，不是存 null：空记录会让导出那边以为
  // 「这个座位被人工处理过」（与 clearCand 同一条理由）。
  if (!v) delete store[k]; else store[k] = {seat: v};
  localStorage.setItem(KEY, JSON.stringify(store));
  render();
}

function toggleCellEx(k, cell, on){
  store[k] = store[k] || {cells: []};
  const cur = new Set(store[k].cells || []);
  if (on) cur.add(cell); else cur.delete(cell);
  store[k].cells = Array.from(cur).sort((a,b)=>a-b);
  localStorage.setItem(KEY, JSON.stringify(store));
  render();
}

// ---------------------------------------------------------------------------
// Task 56C（用户 2026-08-24）：「如果某个座位的人没有合适的候选图，能标记
// 下来，再抽帧二次有更多的备选吗」。
//
// 此前某座位在所有候选里都找不到合适画面时，用户没有任何办法把这件事记下来
// ——它只会一直显示为「未确定」，与「还没标」混作一谈，下次打开页面无从
// 分辨哪些是真找不到、哪些只是没标完。
//
// 键同样是 **(order, seat)**，与 seatOvrKey/cellExKey 同源：座位号跨重检测
// 稳定，而这个标记的唯一用途就是驱动加密抽帧、加密抽帧之后必然重检测——
// 用 track_id 作键的话，它会在自己触发的那次重检测里静默失效。
// ---------------------------------------------------------------------------
function noCandKey(seg, seat){ return 'nocand::'+seg.order+'::'+seat; }

function isNoCandidate(seg, seat){
  // 页面上的取舍**压过**文件里那份（`FILE_NO_CANDIDATE` 来自上次导出的
  // entry_labels）。取消勾选写的是显式的 false，不是删键——只删内存里的键，
  // 下次渲染又会从文件里读回 true，用户的「取消」被悄悄撤销。
  const rec = store[noCandKey(seg, seat)];
  if (rec && rec.no_candidate !== undefined) return !!rec.no_candidate;
  return (FILE_NO_CANDIDATE[String(seg.order)] || []).indexOf(seat) >= 0;
}

function toggleNoCandidate(order, seat, on){
  const seg = findSegByOrder(order); if (!seg) return;
  const k = noCandKey(seg, seat);
  store[k] = store[k] || {};
  store[k].no_candidate = !!on;
  localStorage.setItem(KEY, JSON.stringify(store));
  // 必须重渲染：目录的三态判定读的就是这个值，忘了重算等于勾了没反应
  // （Task 52 需求二在 clearLabel 上踩过同一个坑）。
  render();
}

function noCandidateAt(seg, seat){
  const rec = store[noCandKey(seg, seat)];
  return (rec && rec.at !== undefined) ? String(rec.at) : '';
}

function saveNoCandidateAt(order, seat, value){
  // 只存原文，**页面不解析**。解析只在 Python 的 `entryexit.parse_timestamp`
  // 一处——两处各写一份必然走样，而走样的表现是页面显示的秒数与真正抽帧的
  // 位置不是一个地方，且一处不报错。加密抽帧那一步会把它读到的秒数打印出来，
  // 写错了在那里看得见。
  const seg = findSegByOrder(order); if (!seg) return;
  const k = noCandKey(seg, seat);
  store[k] = store[k] || {};
  store[k].at = String(value || '');
  localStorage.setItem(KEY, JSON.stringify(store));
  // 这里**不重渲染**：重渲染会把输入框重建、光标跳走，用户打一半就被打断。
  // 勾选框那条要重渲染是因为目录的三态判定读它；时刻只影响导出，不影响任何
  // 页面上的判定。
}

function imgTag(src){
  // 图的**唯一**出口。`loading="lazy"` 让浏览器按需解码：一组两百多张图，
  // 一口气全解出来和开页面时只解可见的那几张，差别就在这一个属性上。
  return src ? `<img loading="lazy" src="${src}">` : '<div class="noimg"></div>';
}

function judgeKey(sid){ return 'judge::' + sid; }

function judgeOf(c){
  // 人工改的号压过几何基线（JUDGES 里带的那个）。显式存的空串 = 「这一位
  // 不指定号」，与「没动过」不同——与座位那一路同一条规矩。
  const rec = store[judgeKey(c.stable_id)];
  if (rec && rec.judge !== undefined){
    const v = parseInt(rec.judge, 10);
    return Number.isFinite(v) ? v : null;
  }
  return (c.judge === null || c.judge === undefined) ? null : parseInt(c.judge, 10);
}

function judgeField(c, field){
  // 这一位的某个字段：**localStorage 里有就用它，没有才回填文件里那份**。
  // 与 FILE_NO_CANDIDATE 同一条规矩，方向不能反——反了就是「取消了、刷新
  // 一下又回来」，本项目在「清除标注」上已经栽过一次。
  // 判据是「这个键在不在」，不是「值真不真」：显式存的空串（人工把
  // 「不是评审人」取消掉）必须压过文件里那条旧标记。
  const rec = store[judgeKey(c.stable_id)];
  if (rec && Object.prototype.hasOwnProperty.call(rec, field)) return rec[field];
  const f = FILE_JUDGES[c.stable_id];
  return (f && Object.prototype.hasOwnProperty.call(f, field)) ? f[field] : undefined;
}

function judgeIsExcluded(c){
  return String(judgeField(c, 'role') || '') === JUDGES.not_a_judge;
}

function judgeCellOf(c, k){
  // 第 k 格（1 起）人工说它是谁：'' = 本人（默认）、'1'..'N' = 其实是那一位、
  // 'x' = 其他人（工作人员、路过的），丢掉。
  //
  // 旧版是个勾选框，localStorage 里存的是布尔 true；用户 2026-09-18 11:30
  // 已经用那一版标完城市1 导出了。读不出它就等于把他那半天的工作静默清空，
  // 而页面上一个字都不会说——所以这里把 true 当作 'x'。
  const cells = judgeField(c, 'cells');
  const v = cells ? cells[String(k)] : undefined;
  if (v === true) return JUDGE_CELL_NOBODY;
  if (v === undefined || v === null || v === false) return '';
  return String(v);
}

function judgeCellCount(c){
  // 备选池里真有画面的格数。**不封顶在 MAX_CELLS**：MAX_CELLS 管的是「送进
  // 2×2 的有几格」，池里有十几格都要能标。封在 4 的话第 5 格起既不显示下拉、
  // 导出时也被跳过——人在页面上标了，文件里一个字都没有（2026-09-18 实测）。
  const n = (c && c.n_cells !== undefined) ? parseInt(c.n_cells, 10) : MAX_CELLS;
  return (Number.isFinite(n) && n > 0) ? n : MAX_CELLS;
}

function judgePicks(c){
  // 人工挑中组成 2×2 的那几格，按他挑的次序。与 Python 侧 `judges.judge_picks`
  // 同一条口径：最多 MAX_CELLS 格、去重、标成「其他人」的不作数。
  const fromAnywhere = judgeField(c, 'picks');
  const raw = Array.isArray(fromAnywhere) ? fromAnywhere : [];
  const out = [];
  raw.forEach(function(v){
    const k = parseInt(v, 10);
    if (!Number.isFinite(k) || k < 1) return;
    if (out.indexOf(k) >= 0 || out.length >= MAX_CELLS) return;
    if (judgeCellOf(c, k) !== '') return;
    out.push(k);
  });
  return out;
}

function judgeBaseCells(c){
  // 默认那四格里还算本人的。空出来的位置**不自动顶上**——自动顺序只按面积×
  // 清晰度往下顺，人却会以为那一格是他挑的。
  const out = [];
  const nc = Math.min(MAX_CELLS, judgeCellCount(c));
  for (let k = 1; k <= nc; k++){
    if (judgeCellOf(c, k) === '') out.push(k);
  }
  return out;
}

function judgeCellsInUse(c){
  // 现在真会进 2×2 的是哪几格：**默认那四格里还算本人的 + 人工补选的**
  // （用户 2026-09-18：「不需要总是勾选 4 个，默认选的被标记为其他人后，再从
  // 后面标的选中进 2×2 递补」）。与 Python 侧 `registry.build_seat_registry`
  // 同一条口径——两边不一致的表现是页面上打着星的那几格与模型实际看到的不是
  // 一回事，而一处不报错。
  //
  // 别人还回来的格不在这里：那几格来自另一位候选，星号标在**他**的卡片上。
  const base = judgeBaseCells(c);
  const out = base.slice();
  judgePicks(c).forEach(function(k){
    if (out.length < MAX_CELLS && out.indexOf(k) < 0) out.push(k);
  });
  return out;
}

function judgeDonationsTo(n){
  // 别人指过来的格：`[{from, cell}]`。与 Python 侧 `registry.build_seat_registry`
  // 同一条口径——先按 (来源标识, 格号) 排序再取，不能依赖字典顺序，否则装不下
  // 的是哪一格会随浏览器实现而变。
  const out = [];
  (JUDGES.candidates || []).forEach(function(c){
    const rec = store[judgeKey(c.stable_id)];
    const cells = (rec && rec.cells) || {};
    Object.keys(cells).forEach(function(k){
      const t = parseInt(cells[k], 10);
      const cell = parseInt(k, 10);
      if (Number.isFinite(t) && t === n && Number.isFinite(cell)){
        out.push({from: c.stable_id, cell: cell});
      }
    });
  });
  out.sort(function(a, b){
    if (a.from !== b.from) return a.from < b.from ? -1 : 1;
    return a.cell - b.cell;
  });
  return out;
}

function judgeFill(c, n, extra){
  // 这一位现在凑够几格：**本人的（默认四格里还算本人的 + 补选的）先占，
  // 别人还回来的补在后面**，与登记表逐字同一条口径（用户 2026-09-18 要求
  // 两边统一）。此前页面只算本人的：城市2 专家3 自己四格全标掉、收了两格，
  // 登记表说「2 格、还差 2」而页面说「四格齐」，人照着页面就以为标完了。
  const own = judgeCellsInUse(c);
  const got = (!extra && n) ? judgeDonationsTo(n) : [];
  const room = Math.max(0, MAX_CELLS - own.length);
  const taken = got.slice(0, room);
  return {own: own, donated: taken, unplaced: got.slice(room),
          total: own.length + taken.length};
}

function togglePick(sid, k, on){
  const key = judgeKey(sid);
  store[key] = store[key] || {};
  const cur = Array.isArray(store[key].picks) ? store[key].picks.slice() : [];
  const at = cur.indexOf(k);
  if (on && at < 0) cur.push(k);
  if (!on && at >= 0) cur.splice(at, 1);
  store[key].picks = cur;
  localStorage.setItem(KEY, JSON.stringify(store));
  render();
}

function saveJudgeCell(sid, k, who){
  const key = judgeKey(sid);
  store[key] = store[key] || {};
  store[key].cells = store[key].cells || {};
  store[key].cells[String(k)] = String(who || '');
  localStorage.setItem(KEY, JSON.stringify(store));
  render();
}

function saveJudge(sid, field, value){
  const k = judgeKey(sid);
  store[k] = store[k] || {};
  store[k][field] = value;
  localStorage.setItem(KEY, JSON.stringify(store));
  render();
}

function judgeSlots(){
  // {1..N: 候选|null} 外加没排上的。与 Python 的 `judges.judge_slots` 同一条
  // 口径：人工点过名的先落位，其余按几何顺序填空号。两边不一致的表现是页面
  // 显示的号与登记表里的号不是一回事，而一处不报错。
  const out = {};
  for (let n = 1; n <= JUDGES.n_expected; n++) out[n] = null;
  const rest = [];
  // 标了「不是评审人」的单独收着，**不丢**。用户 2026-09-20 实测电子28：
  // 这里原先是 `if (judgeIsExcluded(c)) return;`，整张卡片连同那个复选框一起
  // 消失，标记却留在 localStorage 里——页面上再没有一个控件能取消它，那几帧
  // 也没法复核。「删除是一种陈述，不是缺席」：否定要显式可见、可撤回。
  const excluded = [];
  (JUDGES.candidates || []).forEach(function(c){
    if (judgeIsExcluded(c)){ excluded.push(c); return; }
    const n = judgeOf(c);
    if (n && n >= 1 && n <= JUDGES.n_expected && !out[n]) out[n] = c;
    else rest.push(c);
  });
  for (let n = 1; n <= JUDGES.n_expected; n++){
    if (!out[n] && rest.length) out[n] = rest.shift();
  }
  return {slots: out, extra: rest, excluded: excluded};
}

function judgeCard(c, n, extra, titleText){
  // 用模板字符串（反引号）拼，**不要用 '...' 加反斜杠转义**：这段 JS 住在
  // Python 的普通三引号字符串里，`\'` 会被 Python 吞成 `'`，到了浏览器就是
  // 语法错误（2026-09-17 实测 node --check 直接红）。
  let opts = `<option value="">不指定</option>`;
  for (let i = 1; i <= JUDGES.n_expected; i++){
    opts += `<option value="${i}" ${String(n)===String(i)?'selected':''}>专家 ${i}</option>`;
  }
  const off = judgeIsExcluded(c);
  const sid = esc(c.stable_id);
  // 挂 `cand` 类 = 与创业者候选卡片同一套布局与图片尺寸（两列 flex，图占
  // 62%）。这正是用户要的「大小参考展示创业者的2*2图」，而且共用一条规则
  // 之后两边不会再各走各的。
  // 标题按卡片所在的那一栏说话：被标掉的那栏写「不是评审人」，写成
  // 「未入席的候选」会让人以为它还能排号。
  const title = titleText || (extra ? '未入席的候选' : `专家 ${n}`);
  // 逐格核实（用户 2026-09-18：「有的格子混进了其他专家」）。四格红框标的
  // 都是同一个候选，模型无从分辨哪格才是本人；勾掉的那一格连帧一起丢掉，
  // 登记表、核对页、交付包与送模型的那张同时干净（registry 统一滤一次）。
  // 备选池切成的每一张 2×2 各自带四个下拉，格位号接着往下排（图上画的就是
  // 这个号，见 annotate.build_evidence_sheet 的 cell_labels）。
  const sheets = (c.sheets && c.sheets.length) ? c.sheets
                 : [{cells: [1,2,3,4], img: c.img}];
  const base = judgeBaseCells(c);
  const picks = judgePicks(c);
  const use = judgeCellsInUse(c);
  let blocks = '';
  sheets.forEach(function(sh, si){
    let boxes = '';
    sh.cells.forEach(function(k){
      const cur = judgeCellOf(c, k);
      let o = `<option value="" ${cur===''?'selected':''}>本人</option>`;
      for (let j = 1; j <= JUDGES.n_expected; j++){
        o += `<option value="${j}" ${cur===String(j)?'selected':''}>其实是专家 ${j}</option>`;
      }
      o += `<option value="${JUDGE_CELL_NOBODY}" ${cur===JUDGE_CELL_NOBODY?'selected':''}>`
         + `其他人（丢掉）</option>`;
      // 默认那四格本来就在 2×2 里，不需要也不给勾——给了勾选框，人会以为
      // 不勾就不算数，反倒把好端端的几格漏掉。
      // 标成「其他人」的格同样不给挑：挑了也不作数（Python 侧一样会丢），
      // 让它可勾就是留个假象。
      const bad = judgeCellOf(c, k) !== '';
      const isBase = base.indexOf(k) >= 0;
      const on = picks.indexOf(k) >= 0;
      const full = use.length >= MAX_CELLS && !on;
      const pick = isBase
        ? `<span class="jpick">默认在 2×2 里</span>`
        : `<label class="jpick"><input type="checkbox" ${on?'checked':''}
             ${bad||full?'disabled':''}
             onchange="togglePick('${sid}',${k},this.checked)"> 补选进 2×2</label>`;
      boxes += `<label class="jcell">格${k}${use.indexOf(k)>=0?' ★':''}
        <select onchange="saveJudgeCell('${sid}',${k},this.value)">${o}</select>
        ${pick}</label>`;
    });
    // 备选默认收起：五位专家各摊开四张 2×2，页面长得没人看。缺格的那位
    // 自动展开——他才是需要翻备选的人。
    const spare = si > 0;
    const open = spare && use.length < MAX_CELLS ? ' open' : '';
    const body = `<div class="imgwrap jsheetImg">${imgTag(sh.img)}</div>
      <div class="jcells">${boxes}</div>`;
    blocks += spare
      ? `<details class="jspare"${open}><summary>备选 ${sh.cells[0]}-${sh.cells[sh.cells.length-1]}
          （当前 2×2 缺格时从这里往下顺着补）</summary>${body}</details>`
      : body;
  });
  // 还差几格必须写出来：少一格就导出，2×2 里会留个灰块，而人以为标完了。
  // 收进的格算在内（与登记表同一条口径）——不算的话这一位明明还缺，页面却
  // 显示「齐」。
  const fill = judgeFill(c, n, extra);
  const short = MAX_CELLS - fill.total;
  const state = `默认 ${base.length} 格`
    + (picks.length ? ` + 补选 ${use.length - base.length} 格` : '')
    + (fill.donated.length ? ` + 收进 ${fill.donated.length} 格` : '')
    + (short > 0 ? `，<b>还差 ${short} 格</b>` : '');
  const from = fill.donated.length
    ? `　收进：${fill.donated.map(d=>esc(d.from) + ' 的格 ' + d.cell).join('、')}。`
    : '';
  const left = fill.unplaced.length
    ? `　<b>没装下</b>：${fill.unplaced.map(d=>esc(d.from) + ' 的格 ' + d.cell).join('、')}
       （2×2 只有 ${MAX_CELLS} 格，本人的格先占）。`
    : '';
  const cellBoxes = `<div class="jcellHint">★ = 现在送模型的：${
      use.length ? '格 ' + use.join('、') : '本人一格都没有'}（${state}）。${from}${left}
    前四格默认就在里面；把不是本人的标掉之后，从下面的备选里勾「补选进 2×2」
    递补——只需补上缺的那几格。指给别的专家，那一帧就换到他名下。</div>${blocks}`;
  return `<div class="cand judgeCard${extra?' judgeExtra':''}" data-sid="${sid}"
    ${extra?'':`id="judge-${n}"`}>
    <div class="info">
      <div class="judgeNum">${title}</div>
      <div class="meta">${esc(c.n_frames)} 帧 · 出现在 ${esc(c.n_segments)} 段 ·
        框高 ${esc(c.median_h)}px · 位置 ${esc(c.median_cx)},${esc(c.median_y2)}</div>
      <label>编号 <select onchange="saveJudge('${sid}','judge',this.value)">
        ${opts}</select></label>
      <label><input type="checkbox" ${off?'checked':''}
        onchange="saveJudge('${sid}','role',this.checked?NOT_A_JUDGE:'')">
        不是评审人</label>
      ${cellBoxes}
    </div>
  </div>`;
}

function judgesSection(){
  // 席位恒为 N：空号必须**看得见**，否则人工分不清「确实没来」与「被漏检」。
  // 没排进席位的候选也一并列出——几何基线只按进深排，一个贴着画面边缘的人
  // 插进序列就会把后面的整体顶掉一位（实测城市2 就这样挤掉了真正的第 5 位）。
  const r = judgeSlots();
  let cards = '';
  let done = 0;
  for (let n = 1; n <= JUDGES.n_expected; n++){
    const c = r.slots[n];
    if (c){ done++; cards += judgeCard(c, n, false); }
    else {
      // 这一号是被人工标掉腾空的，还是本来就没候选？两者要分开说。
      // 用户 2026-09-21 误触「不是评审人」之后，看到的是「尚无候选——可能
      // 没到场，也可能被漏检」——读起来像检测漏了，完全看不出是自己刚点的，
      // 于是「找不到恢复的界面」。撤销要在**出事的地方**看得见。
      const mine = r.excluded.filter(function(c){ return judgeOf(c) === n; });
      cards += mine.length
        ? `<div class="judgeEmpty judgeDropped">专家 ${n}<br>
            这一位被标成了<b>不是评审人</b>，席位因此空着。<br>
            <a href="#judge-dropped">到下面取消这个标记 ↓</a></div>`
        : `<div class="judgeEmpty">专家 ${n}<br>尚无候选——这一位可能没到场，
        也可能被漏检了；可从下面未入席的候选里改号指过来。</div>`;
    }
  }
  const extras = r.extra.length
    ? `<div class="judgeRow"><b>未入席的候选</b>${r.extra.map(c=>judgeCard(c, judgeOf(c), true)).join('')}</div>`
    : '';
  // 被标掉的那些：照常列出（图、格位、复选框都在），取消勾选即可撤回。
  // 一个都没有时整栏不出现——常驻的空栏是纯噪声。
  const dropped = r.excluded.length
    ? `<details id="judge-dropped" class="judgeRow jdropped" open><summary><b>已标为「不是评审人」</b>
        （${r.excluded.length} 位）——取消勾选即可放回席位</summary>
       ${r.excluded.map(c=>judgeCard(c, judgeOf(c), true, '已标为「不是评审人」')).join('')}</details>`
    : '';
  return `<div class="judgeBox"><b>本组评审人</b>（${done}/${JUDGES.n_expected} 已排定）
    —— 画面左侧那一列，离镜头最近的是<b>专家 1</b>，最靠大屏幕的是专家
    ${JUDGES.n_expected}。他们整场不换位置，所以这一块是<b>全组共用</b>的，
    不按企业分。<div class="judgeRow">${cards}</div>${extras}${dropped}</div>`;
}

function undeterminedSeats(seg){
  // 还没定下来的座位号。判据只有一个来源 `determinedSeats`——另写一份
  // 「哪些算定了」的口径，页面上必然出现「目录说定了、这栏还让你标」。
  const done = determinedSeats(seg);
  const out = [];
  for (let k = 1; k <= seg.presenter_num; k++){
    if (done.indexOf(k) < 0) out.push(k);
  }
  return out;
}

function noCandidateSeats(seg){
  // 只认**仍未确定**的座位。用户先标「没画面」、后来又找到人补上时，那条
  // 陈旧标记自动作废——否则登记表会同时说「座位 2 已定」和「座位 2 没画面」，
  // 加密抽帧还会白跑一段其实已经不缺人的窗口。
  return undeterminedSeats(seg).filter(s=>isNoCandidate(seg, s));
}

function noCandidateSection(seg){
  const todo = undeterminedSeats(seg);
  if (!todo.length) return '';
  const g = seg.seat_genders || {};
  const items = todo.map(s=>{
    const on = isNoCandidate(seg, s);
    const sex = g[String(s)] ? '（应为'+esc(g[String(s)])+'）' : '';
    // 勾上之后才给填时刻的框：没勾就没有「重抽」这回事。
    const atBox = on
      ? `<input class="ncAt" type="text" value="${esc(noCandidateAt(seg, s))}"
           placeholder="重抽时刻，如 20:34 或 1234.5"
           onchange="saveNoCandidateAt(${seg.order},${s},this.value)">`
      : '';
    return `<label class="ncSeat${on?' ncOn':''}" data-seat="${esc(s)}">`
         + `<input type="checkbox" ${on?'checked':''} `
         + `onchange="toggleNoCandidate(${seg.order},${s},this.checked)"> `
         + `座位 ${esc(s)}${sex} 无可用画面</label>${atBox}`;
  }).join('');
  return `<div class="ncBox"><b>还没定下来的座位</b>（共 ${todo.length} 个）`
       + `<div class="ncRow">${items}</div>`
       + `<div class="ncHint">在所有候选（含上面的折叠区）里都找不到这个座位`
       + `的人时勾上它。勾上**不等于**这个座位定了，它只是把「确实没画面」`
       + `与「还没看」分开记下来；勾错了再点一下即可取消。`
       + `勾上之后旁边会出现一个<b>重抽时刻</b>框：你要是已经看到那个人在第几`
       + `分钟出现过，填进去（<code>20:34</code> 或 <code>1234.5</code>），`
       + `加密抽帧会直接对着那一带抽，比整段重抽准得多。`
       + `导出后：<code>python -m pipeline.densify</code>。</div></div>`;
}

// ---------------------------------------------------------------------------
// Task 56E · 「上/下一组创业者」：整张卡片流转到相邻段去标
//
// 用户 2026-08-24：「一旦选择『下一组创业者』该组图片可以立马流转到下一组图，
// 在下一组进行标注」。座位号必须在**目标段**选——那边才看得到该段已定的是
// 谁、还缺哪几个、各座位应为男应为女。在源段选等于凭空猜隔壁企业的编制。
//
// store 的键**始终挂在源段上**（`srcOrder::stable_id`）：Python 侧
// `entryexit_fill_plan` 是按「这条标注属于哪一段」去读的，那几帧也确实来自
// 源段的时间窗口。搬键会让流水线在目标段找不到对应候选，静默丢人。
// ---------------------------------------------------------------------------

const NEIGHBOUR_DELTA = {'上一组创业者': -1, '下一组创业者': +1};

function roleUsable(role, ownerSeg, cur, hostSeg){
  // 这个角色在这张卡片上**说得通**吗。
  //
  // **已经选中的那个永远留着**：藏起来的话一个都不选中，用户看不出自己此前标
  // 过什么、也没法改回去，而存着的值还在文件里——页面装作没有是最坏的一种。
  if (role === cur) return true;
  const d = NEIGHBOUR_DELTA[role];
  if (d === undefined) return true;          // 非方向角色一律留着
  const target = ownerSeg.order + d;

  // 借来的卡片（显示在 hostSeg 里）：只有两种身份说得通——**本段的**（指向
  // hostSeg 的那个方向）与**源段的**（普通「创业者」）。指向再远一段的那个与
  // 这张卡毫无关系，选了只会把人流到一个跟这张图不相干的企业名下。
  //
  // 用户 2026-08-26 实测城市2 order 16：卡片属于第 15 段、显示在第 16 段，
  // 上面还有「上一组创业者」（指向第 14 段）。上一版规则「目标段不存在就藏」
  // 只是**碰巧**修好了新能源2 那例——那张卡属于第 1 段，反方向指向第 0 段；
  // 而这里第 14 段确实存在，于是照旧显示。正确的口径与「存不存在」无关。
  if (hostSeg) return target === hostSeg.order;

  // 普通卡片：两个方向都是合法判断，只把**指向不存在那一段**的藏掉
  // （第一段的「上一组」、最后一段的「下一组」——选了也只会记一条未落位）。
  return !!findSegByOrder(target);
}

function roleLabel(role, ownerSeg, hostSeg){
  // 角色在**借来的卡片**上怎么显示。存进去的值一字不变，只改显示。
  //
  // 两件事都要写出来：**属于哪一段**（第 N 段，数字取 hostSeg.order，自动跟着
  // 走）与**图来自哪一边**（上/下一段）。只写段号，用户看不出这张图是从前面
  // 还是后面流过来的；只写方向，又要他自己去数当前是第几段。
  //
  // hostSeg 为空＝普通卡片，一律原样。算方向之前必须先判它——否则读
  // hostSeg.order 直接抛 TypeError，整页 JS 崩掉、一个候选都渲染不出来
  // （实测 21 条测试同时红）。
  if (!hostSeg) return role;
  const from = (ownerSeg.order < hostSeg.order) ? '上一段' : '下一段';
  if (NEIGHBOUR_DELTA[role] === hostSeg.order - ownerSeg.order){
    return `本段创业者（第 ${hostSeg.order} 段 · 图来自${from}）`;
  }
  if (role === '创业者') return `第 ${ownerSeg.order} 段的创业者`;
  return role;
}

function neighbourTargetOrder(seg, role){
  const d = NEIGHBOUR_DELTA[role];
  return (d === undefined) ? null : seg.order + d;
}

function flowTargetOf(seg, cand){
  // 这个候选流向哪一段。返回 {order, seg} —— `seg` 为 null 表示越界
  // （最后一段还标「下一组」）。不流转则返回 null。
  const r = store[candKey(seg, cand)];
  if (!r || !r.role) return null;
  const t = neighbourTargetOrder(seg, r.role);
  if (t === null) return null;
  return {order: t, seg: findSegByOrder(t)};
}

function borrowedCandidates(seg){
  // 别的段流转到本段的候选，[{src, cand}]。按 order 再按 stable_id 排序，
  // 保证渲染顺序稳定——顺序抖动会让用户每次刷新都要重新找自己标到哪了。
  const out = [];
  SEGMENTS.forEach(s=>{
    if (s.order === seg.order) return;
    // 展示候选 ∪ 折叠区：折叠区里的人一样可以标成「上/下一组创业者」，
    // 漏掉它的表现是源段写着「已流转到第 N 段」，而第 N 段里根本没有这张
    // 卡片——用户按提示过去找不到，座位号指不了，也没有任何报错。
    (s.candidates||[]).concat(s.filtered||[]).forEach(c=>{
      const t = flowTargetOf(s, c);
      if (t && t.seg && t.seg.order === seg.order) out.push({src: s, cand: c});
    });
  });
  out.sort((a,b)=> (a.src.order - b.src.order)
                   || (a.cand.stable_id < b.cand.stable_id ? -1 : 1));
  return out;
}

function seatPoolFor(ownerSeg, role){
  // 某个身份指的是**哪一段**的座位号池子。
  //
  // 「创业者」＝ownerSeg 自己的座位；「上/下一组创业者」＝相邻段的座位
  // （Python 侧 `_NEIGHBOUR_ROLE_DELTA` 就是按这个流转的）。此前一律取
  // ownerSeg 的空位池——把人指到了错的企业编制上，Python 按目标段的
  // presenter_num 校验后跳过，只打印一行未落位，用户很难注意到。
  //
  // 相邻段不存在（第一段标「上一组」、最后一段标「下一组」）时给空池：
  // 给一份本段的池子等于让他选一个必然落不了地的号。
  // 「专家」指的是**评审席**的号，不是本段的创业者座位（用户 2026-09-19）。
  // 文化4 的专家5 坐在右侧靠投影仪那头，`judge_candidates` 只看左区，他一个
  // 候选都出不来；右区的人都在创业者候选里，所以这一栏是指认他的唯一出口。
  // 此前这里给的是本段空着的创业者座位号，根本指不到评审席上。
  if (role === '专家'){
    const out = [];
    for (let i = 1; i <= JUDGES.n_expected; i++) out.push(i);
    return out;
  }
  const d = NEIGHBOUR_DELTA[role];
  if (d === undefined) return effectiveVacant(ownerSeg);
  const nb = findSegByOrder(ownerSeg.order + d);
  return nb ? effectiveVacant(nb) : [];
}

function seatReassignOf(seg){
  // 本段被人工改号的已定座位 `{原座位: 新座位}`。
  //
  // 判据与 Python 的 `entryexit.seat_reassign_of` **逐条一致**：脏值跳过、
  // 指向自己跳过、不另设上界。两边不一致的表现是页面算出一套座位账、流水线
  // 算出另一套，而没有任何一处报错。
  //
  // 语义（与 `registry` 的就座分支一致）：原座位**空出来**（那张图挪走了），
  // 目标座位**由这张图占着**，目标座位原来那位退回待标、不替他猜一个对调。
  const out = {};
  (seg.filled_seats||[]).forEach(f=>{
    const rec = store[reassignKey(seg, f.seat)];
    const v = rec && parseInt(rec.seat, 10);
    if (!Number.isFinite(v) || v < 1 || v === f.seat) return;
    out[f.seat] = v;
  });
  return out;
}

function reassignTargets(seg){
  const m = seatReassignOf(seg);
  return Object.keys(m).map(k=>m[k]);
}

function seatsHeldAfterReassign(seg){
  // 就座那一路此刻**实际占着**的座位号（已算进人工改号与人工否决）。
  // 「哪些还空着」「哪些已经定了」「性别栏怎么显示」共用这一份，各算各的
  // 必然出现「目录说定了、正文还让你标」。
  const moved = seatReassignOf(seg);
  const rejected = rejectedSeatsOf(seg);
  const out = new Set();
  (seg.filled_seats||[]).forEach(f=>{
    if (moved[f.seat] !== undefined) return;   // 图挪走了，本号空出来
    if (rejected.has(f.seat)) return;
    out.add(f.seat);
  });
  reassignTargets(seg).forEach(t=>out.add(t));
  return out;
}

function effectiveVacant(seg){
  // 座位号下拉里可选的座位 = 尚空的座位 + 被人工否决/指定替换的座位。
  // 少了后半截，用户「说了这个座位判错了」却没有任何办法把对的人指过去
  // ——一个只能否决、不能纠正的开关等于半个功能。
  //
  // 还要加上**勾了「排除格N」的已定座位**（用户 2026-08-25 实测文化4 order 8
  // 座位1、城市2 order 16 座位1）：剔掉一格之后想再指一张好图补上，下拉里那
  // 个座位却是灰的「已有人，不会覆盖」。那句提示在没有排除时是对的——自动
  // 递补不该顶掉就座结果；一旦勾了排除，座位上的人没换，只是他那张 2×2 少了
  // 一格，需要**同一个人**的另一张画面补上。补格位走
  // `entryexit.labelled_seat_candidates`（有意绕开占用判断），指过去是真的会
  // 生效的——下拉不让选才是错的。
  //
  // 第三类是**被改号挪走的原座位**（用户 2026-09-08 实测城市1 order 4）：
  // 「一旦勾选『这张图其实是座位2』后，应当将该座位释放出来以供选择」。
  // 那张图已经挂到别的号上了，这个号是真的空了，选不了就等于改完号那个人
  // 再也指不上。目标座位同样留在池子里——改号过来的图与用户在目标座位上标
  // 的候选是**合并**关系（Task 56W），拿掉它就把那条路堵死了。
  const extra=[];
  const moved = seatReassignOf(seg);
  (seg.filled_seats||[]).forEach(f=>{
    if (moved[f.seat] !== undefined){ extra.push(f.seat); return; }
    const ov = store[seatOvrKey(seg,f.seat)];
    if (ov && (ov.rejected || ov.replacement_stable_id)) extra.push(f.seat);
    else if (excludedCellsOf(seg, f.seat).length) extra.push(f.seat);
  });
  reassignTargets(seg).forEach(t=>extra.push(t));
  // 去重：同一个号可能由多条路进来（本就空着 + 又是改号目标）。
  return Array.from(new Set(seg.vacant_seats.concat(extra))).sort((a,b)=>a-b);
}

function excludedCellsOf(seg, seat){
  const r = store[cellExKey(seg, seat)];
  return (r && r.cells) || [];
}

function topUpSeats(seg){
  // 「指过去＝补格位」的那些座位（已定 + 勾了排除）。座位号下拉据此改写
  // 文案：用户看到的必须与实际发生的一致——指过去不会顶掉座位上的人。
  // 只算**此刻确实还有人坐着**的座位（`seatsHeldAfterReassign` 已扣掉被改号
  // 挪走的与被人工否决的）。不扣的后果，用户 2026-09-08 实测城市1 order 4：
  // 他先在座位 1 的 2×2 上勾了排除格 3、格 4，又勾了「这张图其实是座位 2」
  // 把整张挪走；之后在退场候选上标「创业者 · 座位 1」，下拉里却写着
  // 「1（补格位，不换人）」——而那条恰恰是座位 1 唯一的标注。文案与实际不符
  // 会让人以为自己指过去不生效。
  const held = seatsHeldAfterReassign(seg);
  const out = [];
  (seg.filled_seats||[]).forEach(f=>{
    if (!held.has(f.seat)) return;
    if (excludedCellsOf(seg, f.seat).length) out.push(f.seat);
  });
  // 改号的目标座位上已经有那张挪过来的图：指过去是与它合并，不是换人。
  reassignTargets(seg).forEach(t=>{ if (out.indexOf(t) < 0) out.push(t); });
  return out;
}

function seatOverrideControls(seg, seat){
  // 需求二（用户 2026-08-23）：「已标注的座位可能是错的，我人工核实时如何
  // 告诉是错的，或者画面没有我看到的清楚，舍弃掉用我指定的画面」。
  // 两个操作：① 判错（该座位视为空缺，可被递补）；② 指定替代候选。
  const k = seatOvrKey(seg, seat);
  const ov = store[k] || {};
  const rejected = !!ov.rejected;
  // 只剩一个开关（用户 2026-08-24 两次精简）。
  //
  // 文字用「是否替换」而不是「这个座位判错了」：这一栏多数时候不是「机器
  // 判错了」，而是「我找到了更清楚的一张」——写成判错会让人以为必须先认定
  // 它错才能换。
  //
  // 「改用我指定的候选」下拉已删：勾了这个开关，该座位即视为空缺，谁来补由
  // 用户在下面进出场候选里标的那份决定。同一件事让他在两个地方各说一遍，
  // 两处说法不一致时还得再定一套优先级——冗余本身就是缺陷来源。
  //
  // 备注填空框同样已删：`seat_overrides` 的消费方只读 `rejected`，那个框
  // 写什么都不影响任何结果。
  return `<div class="ovrBox${rejected?' ovrOn':''}">
    <label><input type="checkbox" ${rejected?'checked':''}
      onchange="save('${k}','rejected',this.checked)"> 是否替换</label>
    <span class="ovrHint">勾上＝这个座位视为空缺，请在下面的进出场候选里
      标出该由谁来补。</span>
  </div>`;
}

function seatReassignControl(seg, seat){
  // 「这张图其实是座位 N」。一步到位，不必先否决再去候选里重新指认同一个人
  // ——那个人恰好没有好用的进出场候选时，这张本来很好的就座图就此浪费。
  const k = reassignKey(seg, seat);
  const cur = (store[k] && store[k].seat) || 0;
  let opts = '<option value="">（不改，就是座位 ' + seat + '）</option>';
  for (let n = 1; n <= seg.presenter_num; n++){
    // 不列自己：那是个什么都不做的选项，留着只会让人以为自己点过了。
    if (n === seat) continue;
    opts += `<option value="${n}"${cur===n?' selected':''}>座位 ${n}</option>`;
  }
  return `<div class="rsBox${cur?' rsOn':''}">
    <label>这张图其实是
      <select class="seatReassign" onchange="saveReassign('${k}',this.value)">
        ${opts}
      </select></label>
    <span class="rsHint">改号后本座位变为空缺；目标座位原来那位会退回「待标」
      等你另行指认，不会替你猜一个对调。</span>
  </div>`;
}

function filledSeatsSection(seg){
  // 已定座位一栏，排在候选之前：用户只需标缺的那几个。
  // 同时它是参照物——同一家企业的人往往同时入场、衣着风格相近，
  // 看着已定者去找缺失者比凭空判断准得多。
  if (!seg.filled_seats || !seg.filled_seats.length){
    return `<div class="filledBox"><b>本段已定座位：</b>无（该段就座阶段一人未定）`
         + (seg.vacant_seats.length ? ` · 还缺：座位 ${seg.vacant_seats.join('、')}` : '')
         + `</div>`;
  }
  const cells = seg.filled_seats.map(f=>{
    const sex = (seg.seat_genders||{})[String(f.seat)];
    // 2×2 证据图 + 逐格排除（用户 2026-08-24）。跟踪串号会让四格里某一格
    // 框到别人；四格红框标的都是同一个 track，模型无从分辨哪格才是目标，
    // 会把两个人的衣着混起来判断且不报错。用户勾掉那一格即从 VLM 输入剔除。
    let sheetHtml = '';
    if (f.sheet){
      const k = cellExKey(seg, f.seat);
      const ex = (store[k] && store[k].cells) || [];
      let boxes = '';
      for (let c = 1; c <= (f.n_cells||0); c++){
        const on = ex.indexOf(c) >= 0;
        boxes += `<label class="exCell"><input type="checkbox" ${on?'checked':''} `
              + `onchange="toggleCellEx('${k}',${c},this.checked)"> 排除格${c}</label>`;
      }
      sheetHtml = `<div class="fsSheet">${imgTag(f.sheet)}`
                + `<div class="exRow">${boxes}</div>`
                + (ex.length >= (f.n_cells||0) && (f.n_cells||0) > 0
                    ? `<div class="warn">四格全被排除——该座位将没有可用证据图</div>` : '')
                + `</div>`;
    }
    // 与创业者候选同一套布局：左边一列图（imgwrap，占 62%），右边一列控件。
    return `<div class="cand filledSeat" data-seat="${esc(f.seat)}" `
         + `>`
         + `<div class="imgwrap">${sheetHtml}</div>`
         + `<div class="info">`
         + `<div class="fsSeat">座位 ${esc(f.seat)}`
         + (sex ? ` · ${esc(sex)}` : '') + `</div>`
         + `<div class="fsNote">已定；想换张更清楚的画面，用下面两项</div>`
         + seatReassignControl(seg, f.seat)
         + seatOverrideControls(seg, f.seat)
         + `</div></div>`;
  }).join('');
  const g = seg.seat_genders || {};
  const vac = seg.vacant_seats.length
    ? `还缺：` + seg.vacant_seats.map(n=>
        g[String(n)] ? `座位 ${n}（应为${g[String(n)]}）` : `座位 ${n}`).join('、')
    : '本段座位已全部确定，无需标注';
  return `<div class="filledBox"><b>本段已定座位（无需标注）</b> · ${esc(vac)}`
       + `<div class="fsRow">${cells}</div></div>`;
}

function peopleOptions(peopleVal){
  // 人数下拉 1..MAX_CELLS（＝2×2 的格数），默认「1」——与角色/座位号不同，
  // 1 是真实默认值（大多数框确实只有一个人），不是「没判定」的伪装
  // （Task 41 改动三）。
  //
  // 上限跟着格数走，不另写死（用户 2026-08-25：「每个 2*2 图片，人数的选择
  // 应该是最多可以选择 4 人吧，目前最多只能选 3 人」）。写死成 3 的后果是四个
  // 人的框标不全：第四个人没有格位行可填，只能眼看着他没人认领。
  let opts = '';
  for (let n=1;n<=MAX_CELLS;n++){
    opts += `<option value="${n}" ${String(peopleVal)===String(n)?'selected':''}>${n}</option>`;
  }
  return opts;
}


// ---------------------------------------------------------------------------
// Task 54 · 需求一：格位分配 = 结构化选择题（取代「写备注 + Python 侧解析」）
//
// 用户原话：「能放弃解析，转换为选择吗，这样就能精准实时匹配到座位人是否
// 识别，比如选人数为 2 时，就给出这两人身份和座位号的选择题，且可留空，根据
// 选出的空，精准识别，不需要解析了」。
//
// **实时性是这组改动的全部价值**：解析在 Python 侧，用户写完备注只能等下次
// 跑流水线才知道结果——城市2组 order1 三个人早就写进备注了，目录仍是红的。
// 选择题一填，下面这几个函数当场算出「这一段已确定几个座位」，目录立刻变绿。
//
// 语义必须与 `entryexit.cells_to_cell_map` / `cell_map_of_label` **逐条一致**
// （留空不算数、「同格位」只引用更小的格位、cells 全空退回 note/角色闸门）。
// 两边走样的表现是页面绿了而流水线一个人也没补进去，且不报错。
// ---------------------------------------------------------------------------

function peopleOf(rec){
  const v = (rec && rec.n_people!==undefined) ? parseInt(rec.n_people,10) : 1;
  return (Number.isFinite(v) && v>0) ? v : 1;
}

function cellCountOf(cand){
  // 这个候选的证据图上真正有画面的格数（不足 4 张时余下的是灰块，
  // entryexit.cell_dets 对那些格位一律返回空）。超出的格位不给选、不算数
  // ——否则用户选了一个没有画面的格位，页面算作已确定、流水线补不进去。
  const n = (cand && cand.n_cells!==undefined) ? parseInt(cand.n_cells,10) : MAX_CELLS;
  return (Number.isFinite(n) && n>0) ? Math.min(n, MAX_CELLS) : MAX_CELLS;
}

function cellRef(row){
  // 这一行有没有选「同格位X」；有则返回被引用的格位号，没有返回 null。
  if (!row || row.same_as===undefined || row.same_as===null || row.same_as==='') return null;
  const v = parseInt(row.same_as,10);
  return Number.isFinite(v) ? v : null;
}

function notePrefillRows(seg, cand, rec){
  // 旧备注的解析结果（Python 侧 entryexit.note_to_cells 算好送上来的，
  // 页面**不**另写一份解析器）。仅当这条记录确实存在、且备注一字未改时
  // 才用——用户在浏览器里改过备注之后，旧的解析结果可能已经不适用了，
  // 继续拿它预填等于替他做了个他不知道的判断。
  if (!rec) return null;
  const f = FILE_LABELS[candKey(seg,cand)];
  if (!f || !f.note_cells || !f.note_cells.length) return null;
  const cur = (rec.note!==undefined) ? String(rec.note) : String(f.note||'');
  if (cur.trim() !== String(f.note||'').trim()) return null;
  return f.note_cells;
}

function cellRowsOf(seg, cand, rec){
  // 这个候选此刻该显示哪几行格位分配，以及这些行是不是备注自动填充来的。
  //
  // 人数**明确**改回 1 时先收起来：一个框里一个人，逐格分配没有意义（用户
  // 2026-09-09 实测城市1 order 16：「选的是人数为 1，却还是出现了 4 个格位
  // 标注」——人数原先只管「要不要铺新行」，已经存着的 cells 照旧渲染）。
  // **不删 store 里的 cells**（`setPeople` 的注释记着这条教训：用户可能只是
  // 拨错了又拨回去，删掉等于把他的工作悄悄毁掉），只是不显示、不作数。
  // Python 侧 `cell_map_of_label` 同一条口径——两边不一致的话，页面藏起来了
  // 而流水线照旧按 cells 切分，一处不报错。
  // 只在**显式为 1** 时收起：`n_people` 整个不在时照旧显示。
  const _np = rec && rec.n_people;
  if (_np !== undefined && String(_np) === '1') return {rows: [], prefilled: false};
  // 优先级与 `entryexit.cell_map_of_label` 一致：store 里的 cells > 备注
  // 自动填充 > 按人数铺开的默认行（格位号默认 1..N，用户可改）。
  const max = cellCountOf(cand);
  // 行数 = 证据图的**实际格数**，不是人数。2×2 里站着 2 个人时，四格
  // 依然每一格都得说清是谁（常见是格1/3 一个人、格2/4 另一个人）——按
  // 人数只铺 2 行的话，格位 3/4 连行都没有，「同格位X」够不着，用户就
  // 只能眼看着两格无主。留空的行不参与判定（cellTargets 会丢掉没选身份
  // 的行），所以多铺出来的行不会把「未确定」误判成「已确定」。
  // 人数在这里只决定**要不要展开**格位分配这一件事。
  const want = peopleOf(rec) >= 2 ? max : 0;
  let rows = null, prefilled = false;
  if (rec && Array.isArray(rec.cells)){
    rows = rec.cells.map(r=>Object.assign({}, r));
  } else {
    const pre = notePrefillRows(seg, cand, rec);
    if (pre){ rows = pre.map(r=>Object.assign({}, r)); prefilled = true; }
  }
  if (!rows) rows = [];
  rows = rows.filter(r=>{
    const c = parseInt(r.cell,10);
    return Number.isFinite(c) && c>=1 && c<=max;
  });
  const used = {};
  rows.forEach(r=>{ used[parseInt(r.cell,10)] = true; });
  for (let k=1; k<=max && rows.length<want; k++){
    if (!used[k]){ rows.push({cell:k}); used[k] = true; }
  }
  return {rows: rows.slice(0, max), prefilled: prefilled};
}

function resolveCellRow(rows, row){
  // 解开「同格位X」的引用链；解不开返回 null（= 这一行视为空）。
  // 只能引用**更小**的格位号 → 每跳一次格位号严格减小 → 结构上不可能成环，
  // 不必事后检测环（与 entryexit._resolve_same_as 同一条不变式）。
  // MAX_CELLS 跳的上限纯属防御，正常永远触发不到。
  const byCell = {};
  rows.forEach(r=>{ const c=parseInt(r.cell,10);
                    if (byCell[c]===undefined) byCell[c]=r; });
  let cur = row, hops = 0;
  while (cur && cellRef(cur)!==null){
    if (hops++ > MAX_CELLS) return null;
    const ref = cellRef(cur), here = parseInt(cur.cell,10);
    if (!(ref < here) || byCell[ref]===undefined) return null;
    cur = byCell[ref];
  }
  return cur || null;
}

function cellTargets(seg, cand, rec){
  // 每一行**解引用之后**的最终判定：{cell, role, seat, ref}。身份留空的
  // 一律不返回（留空 = 未判定，不参与补位——与 Python 侧同一条规则）。
  const info = cellRowsOf(seg, cand, rec);
  const out = [];
  info.rows.forEach(r=>{
    const res = resolveCellRow(info.rows, r);
    if (!res) return;
    const role = String(res.role||'').trim();
    if (!role) return;
    const seat = parseInt(res.seat,10);
    out.push({cell: parseInt(r.cell,10), role: role,
              seat: Number.isFinite(seat) ? seat : null, ref: cellRef(r)});
  });
  return out;
}

function cellMapOf(seg, cand, rec){
  // 有内容才算「这个候选用 cells」；**全空等同于还没填**，退回备注/角色
  // 闸门（与 entryexit.cell_map_of_label 的优先级表逐条一致）——用户把人数
  // 拨到 2、控件刚展开还没选时，不能让他此前标的角色+座位号静默失效。
  const t = cellTargets(seg, cand, rec);
  return t.length ? t : null;
}

function presenterCellsWithoutSeat(seg, cand, rec){
  // 某几格标了创业者却没选座位号——与「整条候选标了创业者没选座位号」
  // 同一类问题：标了却补不进去（entryexit_fill_plan 的 needs_seat）。
  const cells = cellMapOf(seg, cand, rec);
  if (!cells) return [];
  return cells.filter(t=>t.role==='创业者' && t.seat===null).map(t=>t.cell);
}

function isPresenterWithoutSeat(rec, seg, cand){
  // Task 53 · 需求二的唯一判据，页面各处（卡片、目录、进度）共用一份——
  // 各写一份必然日后走样，而走样的表现是「卡片上红了，目录里没算」。
  //
  // 判据与 `determinedCount` 里那句 `Number.isFinite(parseInt(r.seat,10))`
  // 完全一致：座位号为空串 / 未定义 / 「不确定」都算没指定。
  //
  // Task 54：这个候选已经用格位选择题判定过时，整条候选的角色**不参与**
  // 任何判断（Python 侧同样忽略它），此处一律返回 false；那种情形下
  // 「哪一格没选座位号」由 presenterCellsWithoutSeat 逐格报告。
  if (seg && cand && cellMapOf(seg, cand, rec)) return false;
  if (!rec || rec.role !== '创业者') return false;
  return !Number.isFinite(parseInt(rec.seat, 10));
}

function segSeatMissingCount(seg){
  let n = 0;
  // 折叠区一并算进来：这个数字是目录上「N 处标了创业者没选座位号」的来源，
  // 漏掉折叠区就等于用户把目录扫成全绿，而那个人一个也补不进去。
  seg.candidates.concat(seg.filtered||[]).forEach(c=>{
    const rec = store[candKey(seg,c)];
    if (isPresenterWithoutSeat(rec, seg, c)) n++;
    n += presenterCellsWithoutSeat(seg, c, rec).length;
  });
  return n;
}

function cellIndexOptions(cur, max, taken){
  // `taken` = 被**别的行**占走的格位号。行数已等于格数（cellRowsOf），每格
  // 恰好一行，重号只会造出「两行都写着格位3、格位1 没人管」这种无声的错
  // ——两行都指同一张图，另一格的人则完全没被判定，且页面看不出异常。
  let o = '';
  for (let k=1;k<=max;k++){
    if (taken && taken[k] && String(cur)!==String(k)) continue;
    o += `<option value="${k}" ${String(cur)===String(k)?'selected':''}>${k}</option>`;
  }
  return o;
}

function cellRoleOptions(rows, i, ownerSeg, hostSeg){
  // 身份下拉。**「同格位X」追加在最前面**（用户追加需求：2×2 里同一个人
  // 常占两格，每格都重填身份+座位号比写备注还累），且只列**比本格小**的
  // 格位号——只能向前引用，环在结构上不可能出现，不靠事后检测。
  const here = parseInt(rows[i].cell,10);
  const ref = cellRef(rows[i]);
  const cur = ref===null ? String(rows[i].role||'') : '';
  let o = `<option value="" ${(ref===null && !cur)?'selected':''}>（留空 · 未判定）</option>`;
  // 可引用的格位号由**本格的格位号**决定（1..here-1），不遍历 rows。
  //
  // 用户实测「无法设置格位3 格位4同上」：原实现从 rows（当前已生成的行）
  // 里取可引用项，而行数由「人数」决定——人数选 2 时只有 2 行，用户把某行
  // 的格位号改到 3 或 4 后，那一行的下拉里仍只有另一行贡献的那一个选项，
  // 拿不到「同格位2/3」。改为按格位号生成，与行状态无关。
  // 格位 1 时循环一次都不执行，天然没有「同格位」可选（无更小的格位）。
  for (let c = 1; c < here; c++){
    o += `<option value="same:${c}" ${ref===c?'selected':''}>同格位${c}</option>`;
  }
  // `hostSeg` 非空＝这张卡片是**借来的**（在目标段里显示，见 borrowedCard）。
  // 此时把那个带方向的角色改写成「本段创业者（第N段）」——存进去的值一字不
  // 变（Python 只认「上/下一组创业者」），只是显示成用户此刻脑子里的说法。
  //
  // 用户 2026-08-25 实测新能源2 order 2：卡片流转过来了、提示写着「请到那一段
  // 指定座位号」，可他在那一段**选不了创业者身份和座位号**。因为选项里的
  // 「创业者」指的是**源段**（座位池也是源段的），要把人放进当前这一段得选
  // 那个带方向的词——而卡片就摆在当前段里，没有理由想到还要选方向。
  const dirRole = hostSeg ? roleTowards(hostSeg, ownerSeg) : null;
  CELL_ROLES.filter(r=>roleUsable(r, ownerSeg, cur, hostSeg)).forEach(r=>{
    const label = roleLabel(r, ownerSeg, hostSeg);
    o += `<option value="${esc(r)}" ${cur===r?'selected':''}>${esc(label)}</option>`;
  });
  return o;
}

function roleTowards(hostSeg, ownerSeg){
  // 从 ownerSeg 指向 hostSeg 的那个角色名（找不到返回 null）。
  const d = hostSeg.order - ownerSeg.order;
  const hit = Object.keys(NEIGHBOUR_DELTA).filter(k=>NEIGHBOUR_DELTA[k] === d);
  return hit.length ? hit[0] : null;
}

function cellSeatOptions(vacant, seatVal){
  // 只列**尚空**的座位（沿用 Task 48 的 effectiveVacant）——指到已被占用的
  // 座位，entryexit_fill_plan 会跳过（不覆盖已占座位），标了却不生效且没有
  // 任何提示。
  //
  // 唯一的例外：这一格**当前已经**选着一个已被占用的座位（多半是旧备注
  // 自动填充来的，例如「2/3/4是创业者1」而座位1 就座阶段已经定了）。这时
  // 把它作为**禁用**选项列出来并写明「已有人，不会覆盖」——直接不列会让
  // 下拉显示成「不确定」，与实际存着的值对不上，用户看不出发生了什么。
  const blank = (seatVal===''||seatVal===null||seatVal===undefined);
  let opts = `<option value="" ${blank?'selected':''}>不确定</option>`;
  const inVacant = vacant.some(n=>String(n)===String(seatVal));
  if (!blank && !inVacant){
    opts += `<option value="${esc(seatVal)}" selected disabled>`
          + `${esc(seatVal)}（已有人，不会覆盖）</option>`;
  }
  vacant.forEach(n=>{
    opts += `<option value="${n}" ${String(seatVal)===String(n)?'selected':''}>${n}</option>`;
  });
  return opts;
}

function cellsTaken(rows){
  // 各行占着的格位号。**不排除当前这一行**——`cellIndexOptions` 里那条
  // `String(cur)!==String(k)` 已经让本行自己的值恒可选，再排一次是死代码
  // （变异测试验证过：去掉 `j===i` 一个测试都不红）。而且留着 taken[cur]
  // 反倒对：旧标注里若两行重号，本行的值仍列得出来，不会变成空下拉把值丢了。
  const t = {};
  rows.forEach(r=>{ const c = parseInt(r.cell,10);
                    if (Number.isFinite(c)) t[c] = true; });
  return t;
}

function cellRowHtml(seg, cand, rows, i, hostSeg){
  const sid = esc(cand.stable_id);
  const row = rows[i];
  const ref = cellRef(row);
  const res = resolveCellRow(rows, row);
  const max = cellCountOf(cand);
  const call = (field)=>`saveCell(${seg.order},'${sid}',${i},'${field}',this.value)`;
  // 「同格位X」那一行：座位号继承自被引用的格位，下拉**禁用**并灰字显示
  // 继承来的值——能改就会与被引用格矛盾，而矛盾的两份值哪一份作数用户
  // 无从知道。存进 store 的只有引用（same_as），绝不复制一份值。
  let seatCtl;
  if (ref !== null){
    const inherited = (res && res.seat!==undefined && res.seat!==null && res.seat!=='')
                      ? res.seat : '';
    seatCtl = `<select disabled title="继承自格位 ${ref}">`
            + `<option>${inherited!=='' ? esc(inherited) : '（未判定）'}</option></select>`;
  } else {
    const seatVal = (row.seat===undefined||row.seat===null) ? '' : row.seat;
    // 池子跟着**这一行的身份**走，不是跟着卡片挂在哪一段走。
    seatCtl = `<select onchange="${call('seat')}">`
            + `${cellSeatOptions(seatPoolFor(seg, row.role), seatVal)}</select>`;
  }
  const inheritNote = ref!==null
    ? `<span class="cellInherit">继承自格位 ${ref}`
      + `${(res && res.role) ? '：' + esc(res.role) : '（该格尚未判定）'}</span>`
    : '';
  return `<div class="cellRow" data-cell="${esc(row.cell)}">
    <label>格位 <select onchange="${call('cell')}">
      ${cellIndexOptions(row.cell, max, cellsTaken(rows))}</select></label>
    <label>身份 <select onchange="${call('role')}">
      ${cellRoleOptions(rows, i, seg, hostSeg)}</select></label>
    <label>座位号 ${seatCtl}</label>
    ${inheritNote}
  </div>`;
}

function cellsBlock(seg, cand, rec, info, hostSeg){
  if (!info.rows.length) return '';
  const rows = info.rows.map((r,i)=>cellRowHtml(seg,cand,info.rows,i,hostSeg)).join('');
  const pre = info.prefilled
    ? `<div class="cellsPrefill">已由备注自动填充，请核对</div>` : '';
  const noSeat = presenterCellsWithoutSeat(seg, cand, rec);
  const warn = noSeat.length
    ? `<div class="seatMissing">⚠ 格位 ${noSeat.join('、')} 标了创业者但没选
       座位号——不会补位。选一个座位号；判断不出就把身份改回「留空」，
       别停在这个既算标过、又补不进去的中间状态。</div>`
    : '';
  return `<div class="cellsBox">${pre}${rows}${warn}</div>`;
}

function legacyNoteBlock(seg, cand, rec, info){
  // 备注原文只读展示：Task 54 起它**不再参与补位判定**，但绝不隐藏——
  // 解析不出的那些（「另两个」「左边那个」）全靠用户照着原文手动补选。
  const note = (rec && rec.note) ? String(rec.note) : '';
  if (!note.trim()) return '';
  const f = FILE_LABELS[candKey(seg,cand)] || {};
  const unparsed = f.note_unparsed || [];
  const auto = info.prefilled || (f.note_cells||[]).length>0;
  let extra = '';
  if (!auto){
    extra += `<div class="noteLegacyNo">这条备注没能自动填成上面的格位选择
      （相对指代如「另两个」「其余」「左边那个」一律不猜），请照原文逐格
      手动选择。</div>`;
  }
  if (unparsed.length){
    extra += `<div class="noteLegacyNo">其中 ${unparsed.length} 条短语未能解析：
      ${esc(unparsed.join('；'))}</div>`;
  }
  return `<div class="noteLegacy"><b>备注原文</b>（保留供参考，
    <b>不再参与补位判定</b>）：<code>${esc(note)}</code>${extra}</div>`;
}

function normalizeCells(rows){
  // 落盘前的归一：格位号/座位号转成数字，空值不写（留空 = 未判定），
  // 并清掉**孤儿引用与前向引用**。
  //
  // 孤儿引用正是「人数从 3 改回 2」会造出来的东西：被引用的那一行没了，
  // 引用它的行看着仍写着「继承自格位3」，实际什么都继承不到。留着它，
  // 页面与流水线各自按「空」处理，用户却以为那格已经标好了。
  const out = [];
  rows.forEach(r=>{
    const c = parseInt(r.cell,10);
    if (!Number.isFinite(c)) return;
    const o = {cell: c};
    const ref = cellRef(r);
    if (ref !== null){ o.same_as = ref; }
    else {
      if (r.role) o.role = r.role;
      const s = parseInt(r.seat,10);
      if (Number.isFinite(s)) o.seat = s;
    }
    out.push(o);
  });
  const present = {};
  out.forEach(r=>{ present[r.cell] = true; });
  out.forEach(r=>{
    if (r.same_as===undefined) return;
    if (!present[r.same_as] || !(r.same_as < r.cell)) delete r.same_as;
  });
  return out;
}

function findSegByOrder(order){
  const hit = SEGMENTS.filter(s=>String(s.order)===String(order));
  return hit.length ? hit[0] : null;
}

function findCandBySid(seg, sid){
  // **展示候选 ∪ 折叠区**，与 `determinedSeats` / `buildLabelsPayload` 用的
  // 是同一个池子（`seg.candidates.concat(seg.filtered||[])`）。
  //
  // 只找 seg.candidates 的后果，用户 2026-09-08 在城市1 order 2 实测到：
  // 折叠区那张 2×2 里挤了不止一个人，人数拨到 2 之后一行格位都没出来。
  // `setPeople` 与 `saveCell` 反查落空就 `return`，静默、无提示。而角色单选
  // 与座位号走的是 `save(ck, ...)`（直接拿键、不反查），一直是好的——于是
  // 折叠区的卡片看起来能标，实际只有一半能用。
  const pool = (seg.candidates||[]).concat(seg.filtered||[]);
  const hit = pool.filter(c=>c.stable_id===sid);
  return hit.length ? hit[0] : null;
}

function saveCell(order, sid, i, field, value){
  const seg = findSegByOrder(order); if (!seg) return;
  const cand = findCandBySid(seg, sid); if (!cand) return;
  const ck = candKey(seg, cand);
  delete store[clearedKey(ck)];          // 重新标 = 撤回之前的「清除」
  const rec = store[ck] = store[ck] || {};
  // 从**当前显示的**那几行出发（含默认行与备注自动填充来的行）：用户看到
  // 什么就存什么，一改就把自动填充的那份落实成他自己的选择。
  const rows = cellRowsOf(seg, cand, rec).rows;
  const row = rows[i]; if (!row) return;
  if (field === 'role'){
    if (String(value).indexOf('same:') === 0){
      row.same_as = parseInt(String(value).slice(5),10);
      // 引用只存引用，**绝不复制一份被引用格的值**：复制会在用户回头改
      // 源格时留下一份不跟着变的旧副本，且不报错。
      delete row.role; delete row.seat;
    } else {
      delete row.same_as;
      row.role = value;
    }
  } else if (field === 'cell'){
    row.cell = parseInt(value,10);
  } else {
    row[field] = value;
  }
  rec.cells = normalizeCells(rows);
  localStorage.setItem(KEY, JSON.stringify(store));
  render();
}

function setPeople(order, sid, value){
  const seg = findSegByOrder(order); if (!seg) return;
  const cand = findCandBySid(seg, sid); if (!cand) return;
  const ck = candKey(seg, cand);
  delete store[clearedKey(ck)];          // 重新标 = 撤回之前的「清除」
  const rec = store[ck] = store[ck] || {};
  const n = Math.max(1, parseInt(value,10) || 1);
  rec.n_people = String(n);
  // **不按人数截断 cells**。行数现在由证据图的格数决定（见 cellRowsOf），
  // 人数只决定要不要展开这块控件。曾经这里写着 `cells.slice(0, n)`：人数
  // 从 2 改 3 再改回 2，格位 3/4 已经填好的判定会被静默删掉，用户在页面上
  // 只看到那两行变空，没有任何提示。
  if (Array.isArray(rec.cells)) rec.cells = normalizeCells(rec.cells);
  localStorage.setItem(KEY, JSON.stringify(store));
  render();
}

function hydrateCellsFromFile(){
  // run_all 注入的 PRELOAD 按**整条记录**覆盖 store，而它拼出来的记录里
  // 没有 `cells`（run_all._entry_labels_to_store 只搬 role/seat/n_people/
  // note，那个文件不在 Task 54 可改的范围）。不补这一手，用户导出过的格位
  // 选择重开页面就没了，而且不会报错。
  //
  // 幂等且保守：只在记录**完全没有** cells 这个键时补。用户清空过的
  // （cells 是空数组）不再回填——那是他明确的操作，回填等于替他撤销。
  Object.keys(FILE_LABELS).forEach(ck=>{
    const f = FILE_LABELS[ck];
    if (!f || !Array.isArray(f.cells) || !f.cells.length) return;
    const rec = store[ck];
    if (!rec || Array.isArray(rec.cells)) return;
    rec.cells = f.cells.map(r=>Object.assign({}, r));
  });
}

function borrowedCard(hostSeg, srcSeg, cand){
  // 从相邻段流转过来的候选，渲染成**完整的候选卡片**（角色单选、人数、格位
  // 分配一个不少）+ 一条来源徽标。
  //
  // 用户 2026-08-25 实测（新能源2组 order 6）：那张 2×2 里挤了不止一个人，
  // 而这里原先只给一个「本段座位号」下拉——一张图只能指一个人，另外几个无从
  // 表达，用户只能眼看着他们没人认领。
  //
  // store 的键仍是**源段**那把（见 NEIGHBOUR_DELTA 上方的说明）；座位号池子
  // 由 `seatPoolFor` 按**每一行的身份**决定，与卡片显示在哪一段无关——所以
  // 这里直接复用 candidateCard，不另写一份会走样的简化版。
  const ck = candKey(srcSeg, cand);
  return `<div class="borrowWrap" data-ck="${esc(ck)}"
       data-src-order="${esc(srcSeg.order)}">
    <div class="borrowBadge">⇄ 来自第 ${esc(srcSeg.order)} 段
      ${esc(srcSeg.company)} 的${esc(cand.kind==='entry'?'进场':'退场')}候选
      · ${esc(cand.n_frames)}帧 · 逐格选「${esc(cand.kind==='entry'?'上':'下')}
      一组创业者」并指定<b>第 ${esc(hostSeg.order)} 段</b>的座位号</div>
    ${candidateCard(srcSeg, cand, hostSeg)}
    <div class="borrowBack">标错了？把角色改回「创业者」它就回到第
      ${esc(srcSeg.order)} 段。</div>
  </div>`;
}

function borrowedSection(seg){
  const rows = borrowedCandidates(seg);
  if (!rows.length) return '';
  return `<h3>从相邻段流转过来的候选（${rows.length}）</h3>`
       + `<div class="kindHint">这些人被标成「上/下一组创业者」，画面来自相邻`
       + `段的时间窗口。在这里指定他们在<b>本段</b>的座位号。</div>`
       + rows.map(r=>borrowedCard(seg, r.src, r.cand)).join('');
}

function candidateCard(seg,cand,hostSeg){
  const ck = candKey(seg,cand);
  const saved = store[ck]||{};
  const hasLabel = !!store[ck];
  const role = saved.role||'';
  const seatVal = saved.seat!==undefined ? saved.seat : '';
  const peopleVal = saved.n_people!==undefined ? saved.n_people : '1';
  // 借来的卡片（hostSeg 非空）上，角色单选与格位下拉走**同一套改写**：那个
  // 带方向的角色显示成「本段创业者（第N段 · 图来自X一段）」，原来的「创业者」
  // 显示成「第M段的创业者」。存进去的值一字不变（Python 只认带方向的那个）。
  //
  // 用户 2026-08-25：「流转卡片的角色是应该改为『本组创业者』吧」。上一轮只
  // 改了格位下拉，而**单人卡片根本不展开格位**——他看到的就是这排单选，上面
  // 写着「下一组创业者」，而卡片就摆在目标段里。
  const radios = ROLES.filter(r=>roleUsable(r, seg, role, hostSeg)).map(r=>
    `<label><input type="radio" name="role_${esc(ck)}" value="${esc(r)}"
      ${role===r?'checked':''} onchange="save('${ck}','role',this.value)"> ${esc(roleLabel(r, seg, hostSeg))}</label>`
  ).join('');
  // Task 56E：标成「上/下一组创业者」的，本段不再摆一张要人标的完整卡片
  // ——留着的话用户会在源段又选一遍座位号，而那个号属于**源段的编制**，
  // 与目标段的座位号是两码事，标了不生效且不报错。角色单选仍在，随时改回来。
  // asBorrowed = 正在目标段里渲染这张流转过来的卡片，此时要出完整卡片；
  // 否则（在源段里）出一张「已流转」的小卡片。
  // hostSeg 非空＝正在**目标段**里渲染这张流转过来的卡片，出完整卡片；
  // 否则（在源段里）出一张「已流转」的小卡片。
  const flow = hostSeg ? null : flowTargetOf(seg, cand);
  if (flow){
    const where = flow.seg
      ? `已流转到<b>第 ${flow.order} 段 ${esc(flow.seg.company)}</b>`
        + `，请到那一段的候选区里指定座位号`
      : `<span class="warn">没有第 ${flow.order} 段</span>——这一版标注`
        + `流转不过去（Python 侧会记一条未落位并打印）。请改选别的角色。`;
    return `<div class="cand flowedOut" data-ck="${esc(ck)}">
      ${imgTag(cand.img)}
      <div class="info">
        <div class="meta">${esc(cand.kind)} track ${esc(cand.track_id)} · ${esc(cand.n_frames)}帧</div>
        <div class="flowNote">⇄ ${where}</div>
        <div class="roles">${radios}</div>
        ${clearControl(ck, hasLabel)}
      </div></div>`;
  }
  // 座位号那一栏什么时候显示：本段的「创业者」，或者**流转过来的那一张**
  // （角色是「上/下一组创业者」，卡片正显示在目标段里）。
  //
  // 用户 2026-08-25 实测新能源2 order 2：流转过来的照片只有 1 个人时没法指
  // 座位号——这一栏原先写死 `role==='创业者'` 才显示，流转卡片的角色恰恰是
  // 那个带方向的词，整栏被藏起来；人数为 1 时又不展开格位分配，于是一个能填
  // 座位号的地方都没有。工作人员/专家仍不显示（他们本就不占座位）。
  //
  // 「专家」也要显示，而且给的是**评审席的号**（用户 2026-09-19 实测文化4
  // entry/exit track 2：「我选择了专家后，却没有弹出可供选择座位几的选项」）。
  // 文化4 的专家5 坐在右侧靠投影仪那头，只会出现在创业者候选里；人数为 1 时
  // 又不展开格位分配——这一栏藏起来，整张卡片就没有一个能填号的地方，那一位
  // 从此指认不了。池子与格位那一路共用 `seatPoolFor`，两处各写一份必然走样。
  const seatHost = hostSeg || seg;
  const isJudgeRole = (role === '专家');
  const showSeat = (role === '创业者') || isJudgeRole
                   || (!!hostSeg && NEIGHBOUR_DELTA[role] !== undefined);
  const seatOpts = isJudgeRole
    ? seatOptions(seatPoolFor(seatHost, role), seatVal, [])
    : seatOptions(effectiveVacant(seatHost), seatVal, topUpSeats(seatHost));
  const seatLabel = isJudgeRole ? '第几位专家' : '座位号';
  const seatSelect = `<div class="seatWrap" style="${showSeat?'':'display:none'}">
    <label>${seatLabel} <select onchange="save('${ck}','seat',this.value)">
      ${seatOpts}
    </select></label></div>`;
  // Task 54 · 需求一：人数一改，格位分配的选择题**当场**展开 N 行
  // （格位号默认 1..N，可改；身份与座位号都可留空）。
  const cellsInfo = cellRowsOf(seg, cand, saved);
  const isMulti = String(peopleVal) !== '1' || cellsInfo.rows.length > 0;
  const peopleSelect = `<div class="peopleWrap">
    <label>人数 <select onchange="setPeople(${seg.order},'${esc(cand.stable_id)}',this.value)">
      ${peopleOptions(peopleVal)}
    </select></label>
    ${isMulti ? `<div class="multiHint">⚠ 这个框可能挤了不止一个人——请按图上的
      格位编号（${esc(CELL_MAP)}）<b>逐格选出</b>每一格是谁。身份与座位号都可以
      留空，留空即「未判定」，不参与补位；同一个人占两格时，后一格选「同格位X」。
      ${cellsBlock(seg, cand, saved, cellsInfo, hostSeg)}
      ${legacyNoteBlock(seg, cand, saved, cellsInfo)}
      <textarea placeholder="补充说明（不参与补位判定），例：1号格那人被前排挡住看不清" onchange="save('${ck}','note',this.value)">${esc(saved.note||'')}</textarea>
    </div>` : ''}
  </div>`;
  // Task 53 · 需求二：标了创业者却没指定座位号 → 醒目提示。
  // 这类候选进的是 entryexit_fill_plan 的 needs_seat，**一个也补不进去**
  // （座位号按 median_cx 猜是几何推断，本项目已连续失败四次，不做）。
  // 页面此前对它一声不吭，用户以为标完了、那个座位一直空着。
  const seatMissing = isPresenterWithoutSeat(saved, seg, cand)
    ? `<div class="seatMissing">⚠ 已标为创业者但未指定座位号——不会补位。
       请在上面的「座位号」里选一个；判断不出是几号位就改标「不确定」，
       别让它停在这个既算标过、又补不进去的中间状态。</div>`
    : '';
  // 格位约定放在右栏顶部（紧挨着左边的证据图），不塞进图片那一列——
  // .cand 是两列 flex，往图片那一侧加同级元素会把布局挤成三列。
  return `<div class="cand" data-ck="${esc(ck)}">
    ${imgTag(cand.img)}
    <div class="info">
      <div class="cellMap">图上格位编号：${esc(CELL_MAP)}</div>
      <div class="meta">${esc(cand.kind)} track ${esc(cand.track_id)} · ${esc(cand.n_frames)}帧 ·
        框高 ${esc(cand.median_h)}px</div>
      <div class="roles">${radios}</div>
      ${seatSelect}
      ${seatMissing}
      ${peopleSelect}
      ${clearControl(ck, hasLabel)}
    </div></div>`;
}

function clearControl(ck, hasLabel){
  // Task 52 需求二。按钮**常驻**（未标注时也在，只是灰着说明「本条尚未
  // 标注」）：只在标过的候选上给按钮，用户得先随手点一个角色才能取消，
  // 等于绕圈子；而且按钮时有时无会让卡片高度跳动、更难扫读。
  return `<div class="clearWrap">
    <button type="button" class="clearBtn" onclick="clearLabel('${ck}')">清除本条标注</button>
    <span class="fsNote">${hasLabel
      ? '把这一条恢复成未标注（角色、座位号、人数、备注一并清掉）——下面有更好的图时用它'
      : '本条尚未标注'}</span>
  </div>`;
}

function filterReasonText(seg, cand){
  // Task 52 需求三：逐条说明这张图被**哪条**规则挡下。两条都触发时两条都
  // 写——写成「取第一条」会让用户据此调错阈值。
  const rs = cand.filter_reasons || [];
  const out = [];
  if (rs.indexOf('aspect') >= 0){
    out.push(`长宽比 ${esc(cand.aspect)} < ${esc(seg.min_aspect)}（疑似只拍到手/胳膊等局部）`);
  }
  if (rs.indexOf('height') >= 0){
    out.push(`画面过小：框高占比 ${esc(cand.height_ratio)} < ${esc(seg.min_height_ratio)}（证据图上认不出是谁）`);
  }
  return out.length ? out.join('；且 ') : '未记录原因';
}

function filteredCard(seg,cand){
  // 被过滤规则挡掉的候选——**同样可标注**（用户 2026-08-25 实测城市2
  // order 4：密抽之后那个人确实在候选里，只是框高占比 0.096、阈值 0.25，
  // 被挡进了折叠区）。
  //
  // 原设计是只读（Task 41 改动一），理由是「这批本就是给用户看一眼确认没有
  // 真人被误挡」——前提是被挡的都是手、或小到认不出。实际恰恰相反：真人就在
  // 里面，而用户没有任何办法把他标上，只能眼看着那个座位一直空着。
  //
  // 「被挡原因」保留在卡片顶部：用户要据此判断自己是不是在救一个本就该挡掉
  // 的东西（比如只拍到一只手）。data-reasons 让原因既能看也能被程序核对。
  //
  // 页面能标还不够——`entryexit_fill_plan` 必须也在折叠区里找得到这条标注，
  // 见 `entryexit.candidates_for_labels`。只改一边的表现是：用户标了、页面
  // 绿了、人一个也没补进去，且不报错。
  return `<div class="filteredCand" data-reasons="${esc((cand.filter_reasons||[]).join(' '))}">
    <div class="filterReason">被挡原因：${filterReasonText(seg, cand)}
      · 确认是真人就直接在下面标，标了照常补位</div>
    ${candidateCard(seg, cand)}
  </div>`;
}

function filteredSection(seg){
  // 凡有过滤必须可见：被挡掉的候选不静默丢弃，折进可展开的 <details>
  // 区块，默认收起（不干扰正常标注流程），但用户随时能点开核实里面是不是
  // 真的只有手、或者真的小到认不出（Task 41 改动一 + Task 52 需求三）。
  //
  // 两个计数按「触发了这条规则」算，一个候选可能两条都触发，因此两数之和
  // 可能大于总数——文案把这点说破，不用互斥归类掩盖掉。
  if (!seg.filtered_count) return '';
  const parts = [];
  if (seg.filtered_aspect_count) {
    parts.push(`长宽比 < ${seg.min_aspect} 的 ${seg.filtered_aspect_count} 个`);
  }
  if (seg.filtered_height_count) {
    parts.push(`画面过小（框高占比 < ${seg.min_height_ratio}）的 ${seg.filtered_height_count} 个`);
  }
  const both = seg.filtered_aspect_count + seg.filtered_height_count - seg.filtered_count;
  const bothNote = both > 0 ? `，其中 ${both} 个两条都触发` : '';
  // 里面已经标过的个数。用户 2026-08-25 实测城市2 order 4：在折叠区里认出人、
  // 标成创业者之后，页面一重绘那一区又收起来了——他看不到自己刚标的还在不在，
  // 每次都要手动展开去找。render() 是整段重绘的，`<details>` 不带 open 就默认
  // 收起。有标过的就默认展开，并把个数写进标题（收起时也看得到救回过几个）。
  const nLabelled = (seg.filtered||[]).filter(
    c => { const r = store[candKey(seg,c)];
           return r && (r.role || (r.cells||[]).length); }).length;
  // 展开与否比「已标几个」宽一档：只要这条记录被动过就算用户正在这一区里
  // 干活。用户 2026-09-08 实测——把「人数」拨到 2 之后这一区立马又折叠了，
  // 刚展开的四行格位连同那张图一起被卷回去。`setPeople` 只写 `n_people`，
  // 按「已标」判就是「没标过」，而他恰恰是要接着逐格勾选。
  //
  // 两个数分开是刻意的：标题里那个「其中 N 个已标」仍只数真正有身份判定的
  // ——把拨过人数的也算进去等于虚报进度。
  const nTouched = (seg.filtered||[]).filter(
    c => { const r = store[candKey(seg,c)];
           return r && (r.role || (r.cells||[]).length
                        || (r.n_people !== undefined && String(r.n_people) !== '1')
                        || String(r.note||'').trim()); }).length;
  const done = nLabelled ? `，<b>其中 ${nLabelled} 个已标</b>` : '';
  return `<details class="filteredSection"${nTouched ? ' open' : ''}>
    <summary>另有 ${esc(seg.filtered_count)} 个候选未列入（${esc(parts.join('、') + bothNote)}）${done}，点击展开逐条核对是否挡错了人</summary>
    ${seg.filtered.map(cand=>filteredCard(seg,cand)).join('')}
  </details>`;
}

function orientationHint(kind){
  // 用户人工编码时实际用的判据（大概率线索，不是绝对判据——工作人员进出
  // 反复，正面背影都有）：进场大概率拍到背影（往里走），退场大概率拍到
  // 正面（往外走）。Task 41 改动四：只写进页面辅助人工判断，不做自动
  // 朝向识别（控制器正在单独验证人脸/朝向判定的可行性）。
  if (kind === 'entry') {
    return '进场：创业者大概率是背影（正往里走）；工作人员不受此限，进出反复、方向不定，仅供参考，不是绝对判据。';
  }
  return '退场：创业者大概率是正面（正往外走）；工作人员不受此限，进出反复、方向不定，仅供参考，不是绝对判据。';
}

function segAnchorId(seg){
  // 锚点 id 一律由 order 派生。企业名含括号与全角字符，做 HTML id 不安全，
  // 而且跳不动时**不会报错**——只是点了没反应，属于最难发现的那类缺陷。
  return 'seg-' + seg.order;
}

function segHeader(seg){
  const cls = determinedCount(seg) >= seg.presenter_num
    ? 'seg-done-all'
    : (seg.missing>0 ? 'seg-missing' : 'seg-complete');
  return `<h2 id="${segAnchorId(seg)}" class="${cls}">#${esc(seg.order)} ${esc(seg.company)} · `
    + `标注${esc(seg.presenter_num)}人 · 就座已定${esc(seg.occupied)}人 · `
    + `还缺${esc(seg.missing)}人</h2>`;
}

function truncationNotice(seg){
  // 候选按框高排序展示，每段每个窗口最多 seg.top_n 个（展示上限，不是判据）。
  // 凡实际候选数超过这个上限，必须如实告知截断了多少，不能让用户以为
  // 看到的就是全部（entryexit_candidate_totals 给出截断前的真实总数）。
  const parts = [];
  if (seg.entry_total > seg.top_n) {
    parts.push(`进场候选共${seg.entry_total}个，按框高仅展示前${seg.top_n}个（隐藏${seg.entry_total-seg.top_n}个）`);
  }
  if (seg.exit_total > seg.top_n) {
    parts.push(`退场候选共${seg.exit_total}个，按框高仅展示前${seg.top_n}个（隐藏${seg.exit_total-seg.top_n}个）`);
  }
  if (!parts.length) return '';
  return `<div class="trunc-notice">⚠ ${parts.map(esc).join('；')}</div>`;
}

function progressText(){
  let total=0, done=0;
  SEGMENTS.forEach(seg=>seg.candidates.forEach(cand=>{
    total++;
    const ck = candKey(seg,cand);
    if (store[ck] && store[ck].role) done++;
  }));
  return `已标 ${done} / 共 ${total} 个候选`;
}

function rejectedSeatsOf(seg){
  // 被人工否决（判错）或指定了替代候选的已定座位——这两种都意味着
  // 「就座阶段那个结果不再作数」。
  //
  // 判据与 effectiveVacant 完全一致，且都读 store（页面此刻的状态），
  // 不读 run_all 注入的 PRELOAD——那份是「上次导出的文件」，用户刚勾的
  // 那一下还不在里面，以文件为准会让操作不生效且不报错。
  const out = new Set();
  (seg.filled_seats||[]).forEach(f=>{
    const ov = store[seatOvrKey(seg, f.seat)];
    if (ov && (ov.rejected || ov.replacement_stable_id)) out.add(f.seat);
  });
  return out;
}

function determinedSeats(seg){
  // 「已确定」的**座位号清单**（升序）。Task 56C 把原来的 determinedCount
  // 拆成这一份——「还没定下来的是哪几个」与「定下来几个」必须同一个口径，
  // 各算各的必然出现「目录说这段定齐了、正文还让你标座位 3」。
  // 返回数组而不是 Set：页面 JS 在 node 里跑测试时要 JSON 序列化，
  // Set 序列化出来是 `{}`（静默变成空），排查起来极费劲。
  //
  // 「已确定」= 就座已定 + 本页标为创业者且**已指定座位号**的 − 被否决的已定。
  //
  // 未指定座位号的**不算数**：用户真实标注里就有这种（标了创业者、seat
  // 留空），它进的是 needs_seat、根本补不进去。若算作已确定，用户会以为
  // 标完了而那个人其实丢了。
  //
  // 同一座位号被标两次只算一次——用户可能在进场和退场各标了同一个人。
  const seats = new Set(seatsHeldAfterReassign(seg));
  // 折叠区的候选一并扫（用户 2026-08-25 起它们也可标注）。不扫的表现是：
  // 用户在折叠区里认出并标定了人，目录还是红的、「还缺」照旧列着那个座位，
  // 他会以为没标上而反复重标。
  seg.candidates.concat(seg.filtered||[]).forEach(c=>{
    const r = store[candKey(seg,c)];
    // Task 54：格位选择题一填就算数，**不必等跑流水线**——这正是把备注换成
    // 选择题的全部价值（此前解析在 Python 侧，页面无从得知结果，用户明明
    // 把三个人都写进了备注、目录还是红的）。
    // cells 有内容时它就是权威，整条候选的 role/seat 一律不再看（Python 侧
    // `cell_map_of_label` 同样忽略）——两边口径不一致会让页面绿了而人没补上。
    const cells = cellMapOf(seg, c, r);
    if (cells){
      cells.forEach(t=>{ if (t.role==='创业者' && t.seat!==null) seats.add(t.seat); });
      return;
    }
    if (!r || r.role !== '创业者') return;
    const n = parseInt(r.seat, 10);
    if (Number.isFinite(n)) seats.add(n);
  });
  // Task 56E：别的段流转过来的人同样算本段已定。不算的话目标段的目录一直
  // 是红的、「还缺」照旧列着那个座位，用户会跑回去把同一个人重复标一遍。
  borrowedCandidates(seg).forEach(b=>{
    const r = store[candKey(b.src, b.cand)];
    const n = parseInt(r && r.seat, 10);
    if (Number.isFinite(n)) seats.add(n);
  });
  return Array.from(seats).sort((a,b)=>a-b);
}

function determinedCount(seg){ return determinedSeats(seg).length; }

function genderText(seg){
  // 各座位的**人工标注**性别。缺的座位标「应为X」——用户据此能直接排掉
  // 一半候选。presenters 为空时整段省略，不显示也不报错。
  const g = seg.seat_genders || {};
  if (!Object.keys(g).length) return '';
  const filled = seatsHeldAfterReassign(seg);
  const done = [], todo = [];
  for (let k=1;k<=seg.presenter_num;k++){
    const sex = g[String(k)] || '?';
    (filled.has(k) ? done : todo).push(`座位${k} ${sex}`);
  }
  const parts = [];
  if (done.length) parts.push('已定：' + done.join(' · '));
  if (todo.length) parts.push('缺：' + todo.map(t=>t.replace(/ (男|女|\?)$/,
                     (m,x)=>`（应为${x}）`)).join(' · '));
  return parts.join('　｜　') + '（性别为人工标注，非模型判定）';
}

function tocRow(seg){
  // 缺人状态除颜色外必须另有文字/图标标记：只靠颜色对色觉障碍者不可用，
  // 截图与打印后也失效。
  let cls, mark, nums;
  const allDone = determinedCount(seg) >= seg.presenter_num;
  // Task 56C：第三态「剩下的座位人工都看过了，确实没有可用画面」。
  // 它**不是**已定（那几个人还没找着），但也不该和「还没标」同色——
  // 用户靠目录颜色决定还要标哪些，两者同色他每翻一段都得回正文重看一遍。
  // 判据是「未定的座位全部被勾了」：只勾了一部分说明还有活要干，仍按缺人显示。
  const nNoCand = noCandidateSeats(seg).length;
  const allAccounted = !allDone && nNoCand > 0
                       && nNoCand >= undeterminedSeats(seg).length;
  if (allDone)               { cls='toc-done-all'; mark='✓'; }
  else if (allAccounted)     { cls='toc-nocand'; mark='⊘'; }
  else if (seg.missing >= 2) { cls='toc-miss2'; mark='●●'; }
  else if (seg.missing === 1){ cls='toc-miss1'; mark='●'; }
  else                       { cls='toc-ok';    mark='○'; }
  nums = `需${seg.presenter_num}人 · 已定${seg.occupied}人 · `
       + (allDone ? '座位已全部确定' : (seg.missing>0 ? `缺${seg.missing}人` : '不缺人'));
  // 「已定」来自就座阶段的座位分配（静态）；「已标」是本页的标注进度
  // （动态）。两者不是一回事——标了候选不等于人就补上了，必须分开显示。
  let done = 0;
  seg.candidates.forEach(c=>{ const r = store[candKey(seg,c)]; if (r && r.role) done++; });
  // Task 53 · 需求二：这一段有几条「标了创业者却没写座位号」。整页很长，
  // 不进目录用户根本扫不到是哪一段出了问题。
  const nMissSeat = segSeatMissingCount(seg);
  const missSeat = nMissSeat
    ? `<span class="toc-warn">⚠ ${nMissSeat} 条已标创业者但未指定座位号</span>`
    : '';
  // 勾了几个「无可用画面」要写出来（哪怕只勾了一部分）——那几个座位靠
  // 继续标是找不出来的，只能等加密抽帧，用户需要一眼看到还剩多少这种。
  const noCandLine = nNoCand
    ? `<span class="toc-nocand-n">⊘ ${nNoCand} 个座位已标为无可用画面</span>`
    : '';
  return `<a class="toc-row ${cls}" href="#${segAnchorId(seg)}" data-seg="${esc(seg.order)}">`
       + `<span class="toc-mark">${mark}</span>`
       + `<span class="toc-name">#${esc(seg.order)} ${esc(seg.company)}</span>`
       + `<span class="toc-nums">${esc(nums)}</span>`
       + `<span class="toc-done">已标 ${done}/${seg.candidates.length} 个候选</span>`
       + noCandLine
       + missSeat
       + (genderText(seg) ? `<span class="toc-sex">${esc(genderText(seg))}</span>` : '')
       + `</a>`;
}

function judgeTocHtml(){
  // 评审席在目录里的一栏（用户 2026-09-18）。评审席是全组共用的一块、挂在页首，
  // 而企业目录有十几行，人一滚下去就再也看不到它。
  //
  // 位置（cx,y2）必须写出来：席位号来自几何基线（只按进深排），一个贴着画面
  // 边缘的人插进序列就会把后面的整体顶掉一位——城市2 实测 `judge#35,668`
  // （cx=35，紧贴左边缘、266 帧）占了 3 号位，而真正那位在 cx=565、有 625 帧。
  // 把位置摆在目录里，人对着画面一眼就能看出谁坐错了号。
  const r = judgeSlots();
  let rows = '';
  for (let n = 1; n <= JUDGES.n_expected; n++){
    const c = r.slots[n];
    if (!c){
      rows += `<a class="toc-row toc-miss2" href="#judges">专家 ${n}
        <span class="toc-mark">尚无候选</span></a>`;
      continue;
    }
    const short = MAX_CELLS - judgeFill(c, n, false).total;
    const cls = short > 0 ? 'toc-miss1' : 'toc-done-all';
    rows += `<a class="toc-row ${cls}" href="#judge-${n}">专家 ${n}
      <span class="toc-mark">位置 ${esc(c.median_cx)},${esc(c.median_y2)}
      · ${short > 0 ? '差 ' + short + ' 格' : '四格齐'}</span></a>`;
  }
  // 没入席的候选只在**还有空号**时列进目录（用户 2026-09-18）。坐满之后那一行
  // 只是噪声；评审席区块里的卡片照旧列出来，想把人提回席位仍找得着。
  const anyEmpty = Object.keys(r.slots).some(function(k){ return !r.slots[k]; });
  const extra = (r.extra.length && anyEmpty)
    ? `<a class="toc-row toc-ok" href="#judges">未入席 ${r.extra.length} 位
       <span class="toc-mark">${r.extra.map(c=>esc(c.median_cx)+','+esc(c.median_y2)).join('；')}</span></a>`
    : '';
  // 被标掉的也列一行：评审席区块挂在页首，人一滚下去就再也看不到它，而误触
  // 的撤销恰恰是最需要找得回来的那一个。一个都没有时不出现。
  const dropped = r.excluded.length
    ? `<a class="toc-row toc-miss1" href="#judge-dropped">已标「不是评审人」 ${r.excluded.length} 位
       <span class="toc-mark">点此可取消</span></a>`
    : '';
  return `<div class="toc-title">评审席</div>
    <div class="toc-summary">画面左侧一列，离镜头最近的是专家 1</div>${rows}${extra}${dropped}`;
}

function renderToc(){
  const nMiss = SEGMENTS.filter(s=>s.missing>0).length;
  const nPeople = SEGMENTS.reduce((a,s)=>a+s.missing,0);
  document.getElementById('toc-summary').textContent =
    `${SEGMENTS.length} 家企业，${nMiss} 家缺人，共缺 ${nPeople} 人`;
  document.getElementById('toc-list').innerHTML = SEGMENTS.map(tocRow).join('');
  const jt = document.getElementById('toc-judges');
  if (jt) jt.innerHTML = judgeTocHtml();
}

function scrollTocRowIntoView(row){
  // Task 53 · 需求四：目录是 position:fixed + overflow-y:auto，段数多时
  // （用户要标 60+ 组）当前高亮的那一行经常在目录自己的可视区之外——
  // 高亮了等于没高亮。
  //
  // `block:'nearest'` 是刻意的：'start'/'center' 会把**整页**也一起拽走
  // （scrollIntoView 会滚动祖先链上每一个可滚动容器），而这个滚动是页面
  // 滚动**触发**的，互相拉扯会让页面卡住不动。'nearest' 在目标已经可见时
  // 什么都不做，正是这里要的语义。
  //
  // 特性检测 + try/catch，与本页 IntersectionObserver 那条同一套约定：
  // 老浏览器没有这个方法（或不认对象参数）时静默降级为不滚动，绝不能让
  // 一个 TypeError 打断回调、把高亮一起坏掉。
  try {
    if (row && typeof row.scrollIntoView === 'function') {
      row.scrollIntoView({block: 'nearest', inline: 'nearest'});
    }
  } catch (err) { /* 不滚就不滚，高亮照旧 */ }
}

function observeSections(){
  // 滚动时高亮当前所在企业。IntersectionObserver 不可用时静默降级为
  // 不高亮——这是锦上添花的功能，不该因它缺席而让整页 JS 崩掉。
  if (typeof IntersectionObserver === 'undefined') return;
  const rows = {};
  document.querySelectorAll('.toc-row').forEach(a=>{ rows[a.getAttribute('data-seg')] = a; });
  const io = new IntersectionObserver(entries=>{
    entries.forEach(en=>{
      if (!en.isIntersecting) return;
      const id = (en.target.id || '').replace('seg-','');
      Object.keys(rows).forEach(k=>rows[k].classList.remove('toc-current'));
      if (rows[id]) {
        rows[id].classList.add('toc-current');
        scrollTocRowIntoView(rows[id]);
      }
    });
  }, {rootMargin: '-20% 0px -70% 0px'});
  document.querySelectorAll('h2[id^="seg-"]').forEach(h=>io.observe(h));
}

function render(){
  hydrateCellsFromFile();
  document.getElementById('progress-bar').textContent = progressText();
  document.getElementById('judges').innerHTML = judgesSection();
  document.getElementById('main').innerHTML = SEGMENTS.map(seg=>{
    const entryCands = seg.candidates.filter(c=>c.kind==='entry');
    const exitCands = seg.candidates.filter(c=>c.kind==='exit');
    return segHeader(seg) + filledSeatsSection(seg) + noCandidateSection(seg)
      + truncationNotice(seg)
      + `<h3 class="kindTitle">进场候选</h3><div class="kindHint">${esc(orientationHint('entry'))}</div>`
      + entryCands.map(cand=>candidateCard(seg,cand)).join('')
      + `<h3 class="kindTitle">退场候选</h3><div class="kindHint">${esc(orientationHint('exit'))}</div>`
      + exitCands.map(cand=>candidateCard(seg,cand)).join('')
      + borrowedSection(seg)
      + filteredSection(seg);
  }).join('');
  renderToc();
  observeSections();
}
render();

// 导出的 JSON 本体只在这一个函数里拼（Task 46）。run_all 注入的「另存为」
// 升级版 exportLabels 只接管保存方式，payload 一律调用这里——两处各拼一份
// 必然日后走样（Ruling R39 的教训，也是本项目刚吃过的亏）。
function exportCells(seg, cand, rec){
  // 导出的 `cells`：用户此刻在页面上**看到的**那几行（含备注自动填充的
  // 那份——它与备注表达的是同一件事，一次导出就完成迁移），去掉整行留空
  // 的。归一与孤儿引用清理走 normalizeCells，不另写一份。
  const rows = normalizeCells(cellRowsOf(seg, cand, rec).rows);
  return rows.filter(r=>r.same_as!==undefined || r.role!==undefined
                        || r.seat!==undefined);
}

function buildLabelsPayload(){
  const out = {};
  let nCandidates = 0;
  SEGMENTS.forEach(seg=>{
    // 折叠区的候选一并导出（用户 2026-08-25 起它们也可标注）。漏掉的表现是
    // 用户在页面上标得好好的，导出时那条凭空消失，且不报错。
    seg.candidates.concat(seg.filtered||[]).forEach(cand=>{
      nCandidates++;
      const ck = candKey(seg,cand);
      const rec = store[ck];
      if (!rec) return;
      // Task 54：格位选择题本身就是一条完整的判定，**不要求先选整条候选的
      // 角色**——一个 2×2 里三个人分属不同座位时，「这个候选的角色」本就
      // 没有单一答案（Task 53 已在 Python 侧承认了这一点）。只按 role 过滤
      // 会把用户逐格选好的东西整条丢掉，且不报错。
      const cells = exportCells(seg, cand, rec);
      if (!rec.role && !cells.length) return;
      const segKey = String(seg.order);
      out[segKey] = out[segKey] || {};
      // 哪些角色带座位号：本段的「创业者」，以及「上/下一组创业者」——后者
      // 占的是**相邻段**的座位，同样要带。
      //
      // 用户 2026-08-25 实测新能源2：在流转过来的卡片上选了座位号，导出的
      // 文件里却是 seat: null，那条流转因此永远落不了地
      // （`entryexit_fill_plan` 的跨段分支要求 `lab.get("seat")`），而页面上
      // 看着好好的、一处不报错。根因就是这里写死了 `role==='创业者'`。
      //
      // 工作人员/专家/不确定仍不带：残留一个座位号会让下游以为「这个座位有
      // 人了」，而那个人根本不是创业者。
      // 「专家」也要带号（用户 2026-09-19 实测电子27：多处标成专家 5，文件里
      // 却全是 seat:null，`judge_donations_from_candidates` 一条都收不进去——
      // 标了等于没标，而一处不报错）。这一行上方的注释记着同一个 bug 之前为
      // 相邻角色发生过一次；这次是「专家」又漏了一遍。**凡是页面上会显示座位
      // 号控件的角色，这里都必须放行**——两处各列一份名单，必然再漏第三次。
      const seatRole = (rec.role==='创业者') || (rec.role==='专家')
                       || (NEIGHBOUR_DELTA[rec.role] !== undefined);
      const seatVal = (seatRole && rec.seat!==undefined && rec.seat!=='')
        ? parseInt(rec.seat,10) : null;
      const nPeople = rec.n_people!==undefined ? parseInt(rec.n_people,10) : 1;
      // 键是稳定标识；track_id 只作人工排查的参考信息写进值里，绝不用于匹配。
      const entry = {
        role: rec.role||'', seat: seatVal, n_people: nPeople,
        note: rec.note||'', track_id: cand.track_id,
      };
      // 空的一律不写：没填过格位的候选，导出的形状与 Task 46 起逐字相同。
      if (cells.length) entry.cells = cells;
      out[segKey][cand.stable_id] = entry;
    });
  });
  // Task 49 · 需求二：座位否决/替代。与段号平级的顶层键（`_meta` 同级）。
  const overrides = {};
  SEGMENTS.forEach(seg=>{
    (seg.filled_seats||[]).forEach(f=>{
      const rec = store[seatOvrKey(seg,f.seat)];
      if (!rec) return;
      const rejected = !!rec.rejected;
      const repl = rec.replacement_stable_id || null;
      const note = rec.note || '';
      // 三项全空说明用户碰过控件又撤回了，不写进文件——空条目会让下游
      // 以为「这个座位被人工处理过」。
      if (!rejected && !repl && !note) return;
      const k = String(seg.order);
      overrides[k] = overrides[k] || {};
      overrides[k][String(f.seat)] =
        {rejected: rejected, replacement_stable_id: repl, note: note};
    });
  });

  // 已定座位改号（Task 56V）：与段号平级的顶层键。写进某个段号底下会被当成
  // 一个候选的标注，静默失效。
  //
  // 判据只有 `seatReassignOf` 一份：页面的座位账（哪些空了、哪些定了、性别栏
  // 怎么显示）与导出必须同一条口径。各写一份的表现是页面显示成一套、文件里
  // 是另一套，而两边都不报错。
  const reassign = {};
  SEGMENTS.forEach(seg=>{
    const m = seatReassignOf(seg);
    Object.keys(m).forEach(src=>{
      const k = String(seg.order);
      reassign[k] = reassign[k] || {};
      reassign[k][String(src)] = m[src];
    });
  });

  // 逐格排除（用户 2026-08-24）：键是 (order, seat)，与 seat_overrides 同源。
  // 空数组不写——写了会让下游以为这个座位被处理过而其实没有。
  const cellEx = {};
  SEGMENTS.forEach(seg=>{
    (seg.filled_seats||[]).forEach(f=>{
      const rec = store[cellExKey(seg, f.seat)];
      const cells = (rec && rec.cells) || [];
      if (!cells.length) return;
      const k = String(seg.order);
      cellEx[k] = cellEx[k] || {};
      cellEx[k][String(f.seat)] = cells.slice().sort((a,b)=>a-b);
    });
  });


  // 「本座位无可用画面」（Task 56C）：键同样是 (order, seat)。只导出**仍未
  // 确定**的那些（见 noCandidateSeats）——后来定下来的座位那条标记自动作废。
  const noCand = {};
  SEGMENTS.forEach(seg=>{
    const seats = noCandidateSeats(seg);
    if (!seats.length) return;
    noCand[String(seg.order)] = seats.slice().sort((a,b)=>a-b);
  });

  const payload = Object.assign(
    {_meta: {detections_fingerprint: DETECTIONS_FP, sheet: SHEET_NAME,
             n_candidates: nCandidates,
             page_code_fp: PAGE_CODE_FP, page_built_at: PAGE_BUILT_AT}},
    out);
  // 空的一律不写：run_all 用「标注本体是不是空」判断这份文件可不可用，
  // 一个空的 seat_overrides 会让「一条标注都没有」的文件变成真值。
  if (Object.keys(overrides).length) payload.seat_overrides = overrides;
  if (Object.keys(cellEx).length) payload.cell_exclusions = cellEx;
  // 空的不写：挂一个空对象会让「一条标注都没有」的文件变成真值，下游打印
  // 「已启用」而实际什么都不做（与另外三条通道逐字同一条规矩）。
  if (Object.keys(reassign).length) payload.seat_reassign = reassign;
  if (Object.keys(noCand).length) payload.no_candidate_seats = noCand;
  // 重抽时刻（用户 2026-09-09）：**另起一条通道**，不塞进 no_candidate_seats
  // ——那一条的值是数组，另外三个组的既有文件里都是这个形状，改形状会让它们
  // 读不进来。只导出仍未确定、且确实填了东西的那些。
  const noCandNotes = {};
  SEGMENTS.forEach(seg=>{
    noCandidateSeats(seg).forEach(seat=>{
      const at = noCandidateAt(seg, seat).trim();
      if (!at) return;
      const k = String(seg.order);
      noCandNotes[k] = noCandNotes[k] || {};
      noCandNotes[k][String(seat)] = {at: at, text: ''};
    });
  });
  if (Object.keys(noCandNotes).length) payload.no_candidate_notes = noCandNotes;

  // 评审席（组级，用户 2026-09-17）：与段号平级的顶层键。只导出**人工动过
  // 的**那些——几何基线的号不写进文件，否则下游分不清「人看过并认可」与
  // 「没人碰过，这只是几何排序」，而那正是这份文件存在的意义。
  const judges = {};
  (JUDGES.candidates || []).forEach(function(c){
    const rec = store[judgeKey(c.stable_id)];
    if (!rec) return;
    const role = String(rec.role || '');
    const raw = (rec.judge === undefined) ? '' : String(rec.judge);
    const n = parseInt(raw, 10);
    // 逐格核实也算「人工动过」。只看号与身份的话，标完格、导出、重开页面，
    // 格子又全回来了——而一处不报错（用户 2026-09-18）。
    const cells = {};
    let nCells = 0;
    const nc = judgeCellCount(c);
    for (let k = 1; k <= nc; k++){
      const who = judgeCellOf(c, k);
      if (!who) continue;            // 本人：不留痕
      cells[String(k)] = (who === JUDGE_CELL_NOBODY) ? JUDGE_CELL_NOBODY
                                                     : parseInt(who, 10);
      nCells++;
    }
    const picks = Array.isArray(rec.picks)
      ? rec.picks.map(function(v){ return parseInt(v, 10); })
                 .filter(function(v){ return Number.isFinite(v) && v >= 1; })
      : [];
    if (!role && !raw && !nCells && !picks.length) return;
    judges[c.stable_id] = {judge: Number.isFinite(n) ? n : null, role: role};
    if (nCells) judges[c.stable_id].cells = cells;
    if (picks.length) judges[c.stable_id].picks = picks;
  });
  if (Object.keys(judges).length) payload.judges = judges;
  // Task 53 · 需求三：组级「这个位置一直是工作人员/专家」那个勾选框已按
  // 用户要求删除，它的导出字段不再写进文件。旧 localStorage 里残留的那几个
  // 键一律不读——写进去下游也不读了，留着只会让人以为它还生效。
  return payload;
}

// ---------------------------------------------------------------------------
// Task 53 · 需求五：导出后的页内提示条
//
// 用户 2026-08-23 原话：「再次导出的时候，没有任何提示，应该弹出提示」。
// 「再次」是关键——记住句柄之后导出连选择器都不弹，屏幕上什么都不动，
// 用户完全无法分辨「写成功了」与「按钮坏了」。
//
// 导出有三条路径，落地位置完全不同，文案必须能区分（用户需要知道文件到底
// 在哪）：
//   handle   记住的句柄直写（run_all 注入的升级版，最常走的一条）
//   picker   刚用选择器选的位置（同上）
//   download <a download> 回退 → 浏览器默认下载目录（本文件自带的这条，
//            以及升级版每一级都没成时的兜底）
//
// **不用浏览器原生的模态弹窗**：阻塞主线程、样式不可控、连续导出叠一串，
// 还可能被弹窗拦截器吃掉——那就又回到「没有任何提示」。
// ---------------------------------------------------------------------------

// 与段号平级、但不是段的顶层键。Python 侧的唯一来源是
// `entryexit.LABEL_RESERVED_KEYS`；这里按段展开去数时必须逐个跳过，否则
// 旁路通道会被算成标注条数。`no_candidate_seats` 的值还是**数组**，
// `Object.keys` 数的是下标，用户会看到一个凭空多出来的数字。
const PAYLOAD_RESERVED_KEYS = ['_meta', 'seat_overrides', 'cell_exclusions',
                               'no_candidate_seats'];

function countExportedLabels(payload){
  // 条数取自**真正写进文件的**那份 payload，不是候选总数、也不是 store 的
  // 键数：store 里躺着角色为空的记录、座位否决、以及上一版功能的残留，
  // 拿它计数会报出一个比实际多的数字，而用户会据此以为标注比实际多。
  let labels = 0, overrides = 0, noCand = 0;
  Object.keys(payload || {}).forEach(k=>{
    if (PAYLOAD_RESERVED_KEYS.indexOf(k) >= 0) return;
    labels += Object.keys(payload[k] || {}).length;
  });
  const ovr = (payload || {}).seat_overrides || {};
  Object.keys(ovr).forEach(k=>{ overrides += Object.keys(ovr[k] || {}).length; });
  const nc = (payload || {}).no_candidate_seats || {};
  Object.keys(nc).forEach(k=>{ noCand += (nc[k] || []).length; });
  return {labels: labels, overrides: overrides, no_candidate: noCand};
}

function exportToastText(where, name, payload){
  const n = countExportedLabels(payload);
  const what = `已导出 ${n.labels} 条标注`
             + (n.overrides ? ` + ${n.overrides} 条座位否决` : '')
             + (n.no_candidate ? ` + ${n.no_candidate} 个座位标为无可用画面` : '');
  if (where === 'handle'){
    return `✓ ${what} · 已直接写入上次记住的位置：${name}`;
  }
  if (where === 'picker'){
    return `✓ ${what} · 已保存到你刚选的位置：${name}`;
  }
  return `✓ ${what} · 已由浏览器下载到默认下载目录：${name}`
       + `（没看到就查浏览器的下载记录；流水线也会自动去下载目录找这个文件）`;
}

function showExportToast(where, name, payload){
  const msg = exportToastText(where, name, payload);
  // DOM 那一半整个包在 try 里：提示条是锦上添花，它坏了绝不能拖垮导出
  // （Task 45 有过前科——一处 JS 出错导致导出全程无效且无人发现）。
  try {
    let el = document.getElementById('export-toast');
    if (!el){
      el = document.createElement('div');
      el.id = 'export-toast';
      el.className = 'toast';
      if (document.body && document.body.appendChild) document.body.appendChild(el);
    }
    el.textContent = msg;
    el.className = 'toast toast-show';
    // 同一个元素复用 + 重置计时：连续导出不会叠出一串提示条。
    if (el._toastTimer) clearTimeout(el._toastTimer);
    el._toastTimer = setTimeout(function(){ el.className = 'toast'; }, TOAST_MS);
  } catch (err) { /* 提示没显示出来也不影响文件已经写成功 */ }
  // 返回文案供测试断言（不必造一整套 DOM 才能验文案对不对）
  return msg;
}
// 挂到 window 供 run_all 注入的升级版 exportLabels 调用——那一版接管了前
// 两条路径（它注入在本 <script> **之后**，所以这里赋值时它还没覆盖）。
// typeof 判断是为了让本脚本在 node 里（测试、node --check）也能跑。
if (typeof window !== 'undefined') window.showExportToast = showExportToast;

function exportLabels(){
  const payload = buildLabelsPayload();
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(payload)],
                                       {type:'application/json'}));
  a.download=EXPORT_FILENAME; a.click();
  showExportToast('download', EXPORT_FILENAME, payload);
}
</script>
"""


if __name__ == "__main__":
    import json as _json

    from pipeline.detect import load_detections
    from pipeline.schedule import load_segments_json

    seg_path = config.WORK / "segments.json"
    det_path = config.WORK / "detections.json"
    seats_path = config.WORK / "seats.json"
    out_path = paths.output_dir(config.WORK) / "entry_review.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    segments = load_segments_json(seg_path)
    dets = load_detections(det_path)

    seats_by_order: dict[int, dict[int, int]] = {}
    if seats_path.exists():
        raw = _json.loads(seats_path.read_text(encoding="utf-8"))
        seats_by_order = {
            int(o): {int(t): s for t, s in m.items()} for o, m in raw.items()
        }

    from pipeline.entryexit import detections_fingerprint

    render_entry_review(segments, dets, out_path, seats_by_order=seats_by_order,
                        detections_fp=detections_fingerprint(det_path))
    print(f"[entryreview] 写入 {out_path}")
