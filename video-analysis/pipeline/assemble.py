"""装配阶段：把上游产物合成每人一行的完整记录，并产出最终两张表。

本模块是 spec 第 7 节字段定义的唯一落地点——上游各模块只负责算，
这里负责按 spec 的字段名把结果拼起来。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline import config
from pipeline.annotate import (
    LAYOUT_HEADING, PersonAnnotation, ask_vlm, build_evidence_sheet,
    laplacian_sharpness, merge_tracks, select_best_frames,
)
from pipeline.colormath import (
    classify_chroma, classify_hue_6, classify_hue_11, classify_lightness,
    classify_warm_cool, choose_granularity, dominant_color, hue_margin_deg,
    hue_spread_deg, is_hue_valid, mean_hue_deg, rgb_to_hsv_deg, rgb_to_lab,
)
from pipeline.detect import Detection
from pipeline.parsing import inner_quality, parse_garments
from pipeline.schedule import Segment
from pipeline.entryexit import (
    apply_cell_exclusions, candidates_for_labels, cell_exclusions_of, labelled_seat_candidates,
    seat_reassign_of, topup_pairs,
    topup_frames,
)
from pipeline.seatmatch import match_seats_by_gender
from pipeline.whitebalance import find_reference_patch, von_kries_gains

# 单人 VLM 调用失败时，str(exc) 截断上限（Ruling R54）：中转服务的报错文本
# 有时会带回整段响应体，不截断会把它整坨糊进 CSV 的一格里。
_VLM_ERROR_MAX_LEN = 200

_EMPTY_COLOR = {
    "outer_hue_raw": None, "outer_sat_raw": None, "outer_val_raw": None,
    "outer_hue_corr": None, "outer_sat_corr": None, "outer_val_corr": None,
    "outer_lab_L": None, "outer_lab_a": None, "outer_lab_b": None,
    "outer_hue_valid": 0, "outer_color_stability": None, "n_frames_color": 0,
    "hue_class_pixel": "", "hue_class_11": "", "hue_class_6": "",
    "warm_cool": "", "lightness_class": "", "chroma_class": "",
    "hue_class_margin": None, "hue_class_borderline": 0,
    "inner_visible": 0, "inner_pixel_count": 0, "inner_quality": "poor",
    "sharpness_raw": 0.0, "wb_ok_ratio": 0.0,
}

# Task 34 · 改动二：像素轨关闭（config.PIXEL_COLOR_ENABLED=False，现为默认）
# 时，一律不写入结果的字段——全部依赖白平衡/主色提取/色彩空间转换那部分
# 已跳过的计算。集中在这一份列表里过滤，不论 pixel 字典来自真实测量、
# 缺帧占位（_EMPTY_COLOR）还是缺座位占位（_poor_annotation），都保证同一
# 套字段被同一处代码统一剔除，不会有的路径漏删、有的路径没删。
PIXEL_ONLY_FIELDS = (
    "outer_hue_corr", "outer_sat_corr", "outer_val_corr",
    "outer_lab_L", "outer_lab_a", "outer_lab_b",
    "hue_class_pixel", "hue_class_agree", "outer_hue_valid",
    # Task 35 · 改动二：实测三组 103 行 n_frames_color 全为 0（像素轨关闭
    # 后恒为 0，是死列）。像素轨重开时这个计数仍有意义，因此不从
    # _EMPTY_COLOR 里删掉这个键本身，只在关闭状态下靠这份过滤表把它挡在
    # 结果外——与其余像素轨专属字段走同一套机制，不另造一套过滤。
    "n_frames_color",
)


def _finalize_pixel_dict(pixel: dict) -> dict:
    """按 config.PIXEL_COLOR_ENABLED 统一过滤像素轨专属字段（Task 34）。

    调用点覆盖 measure_person_color 的全部返回路径（正常测量 / 缺帧占位）
    以及 _poor_annotation 的缺座位占位——保证不论走哪条路径，关闭时都不会
    有漏网的字段进 CSV。用 `pop`（而非把值置 None/""）是刻意的：rank.py 的
    `a.pixel.get("hue_class_agree", 1)` 这类调用依赖「键不存在才取默认值」，
    键存在但值为 None 会短路掉默认值，让"未测量"被误判成"测量后为假"。
    """
    if not config.PIXEL_COLOR_ENABLED:
        for f in PIXEL_ONLY_FIELDS:
            pixel.pop(f, None)
    return pixel


def normalize_sharpness(values: list[float]) -> list[float]:
    """把 Laplacian 方差归一到 0–1。

    用 95 分位数作分母而非最大值：单张异常锐利的帧不应把其余全部压到接近 0。
    结果必须钳制到 [0,1]——生产端也要做防御性钳制，不能只指望消费者
    （rank.py）单方面兜底（Ruling R5）。
    """
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    denom = float(np.percentile(arr, 95))
    if denom <= 0:
        return [0.5] * len(values)
    return [float(min(1.0, max(0.0, v / denom))) for v in arr]


def segment_gain(
    dets_of_one_segment: list[Detection], max_frames: int = 30
) -> tuple[float, float, float]:
    """一段一个白平衡增益：逐帧算出增益后取中位数。

    **粒度必须是段，不能是帧。** 实测（Task 7 完成后的跨任务验证）：
    逐帧各算各的会让同一件衣服在段内的色相离散度从 1.4–2.4° 恶化到
    4.1–5.0°——因为不同帧可能选中不同参照物（打印机／白纸／桌沿），
    各自真实颜色不同，遂注入噪声。改用段中位增益后为 1.0–2.1°，
    全部优于逐帧、4/5 优于不校正。

    而校正本身仍属必要：段间增益差异真实存在（实测 order 2 R=1.157/
    B=0.937、order 7 R=1.128/B=0.953、order 10 R=0.982/B=1.023），
    不校正则 order 2 的衣服会纯因灯光而比 order 10 显得更暖。
    """
    paths = sorted({d.frame_path for d in dets_of_one_segment
                    if d.kind == "seated"})[:max_frames]
    gains = []
    for pth in paths:
        try:
            img = np.asarray(Image.open(pth).convert("RGB"))
        except Exception:
            continue
        ref = find_reference_patch(img)
        if ref is not None:
            gains.append(von_kries_gains(ref))
    if not gains:
        return (1.0, 1.0, 1.0)
    med = np.median(np.asarray(gains), axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


def _crop(det: Detection) -> np.ndarray | None:
    try:
        img = np.asarray(Image.open(det.frame_path).convert("RGB"))
    except Exception:
        return None
    y1, y2 = max(0, int(det.y1)), min(img.shape[0], int(det.y2))
    x1, x2 = max(0, int(det.x1)), min(img.shape[1], int(det.x2))
    if y2 <= y1 or x2 <= x1:
        return None
    return img[y1:y2, x1:x2]


def _mask_pixels(crop: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None or not mask.any():
        return None
    return crop[mask].reshape(-1, 3)


def measure_person_color(
    dets: list[Detection],
    gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_frames: int = 12,
) -> dict:
    """在该人的若干帧上测颜色，跨帧聚合。

    只取最好的 max_frames 帧：分割模型每帧要跑一次推理，全量代价过高，
    而 spec 只要求跨帧平均以抵消幻灯片染色，十余帧已足够。

    Task 34 · 改动二：`config.PIXEL_COLOR_ENABLED=False`（现为默认）时跳过
    白平衡校正 / 服装分割（parse_garments）/ 主色提取 / 色彩空间转换这一整段
    计算——这部分是整个函数里唯一昂贵的部分（服装分割每帧跑一次分割模型
    推理），三组视频实测双轨一致率仅 33%/41%/43%，继续跑没有意义。只保留
    清晰度（Laplacian 方差，供排名用，且计算本身不依赖分割/白平衡）。
    """
    if not dets:
        return _finalize_pixel_dict(dict(_EMPTY_COLOR))

    chosen = select_best_frames(dets, n=max_frames)

    if not config.PIXEL_COLOR_ENABLED:
        sharps_only = []
        for d in chosen:
            crop = _crop(d)
            if crop is None or crop.size == 0:
                continue
            sharps_only.append(laplacian_sharpness(crop))
        out = dict(_EMPTY_COLOR)
        out["sharpness_raw"] = float(np.mean(sharps_only)) if sharps_only else 0.0
        return _finalize_pixel_dict(out)

    raw_h, raw_s, raw_v = [], [], []
    cor_h, cor_s, cor_v = [], [], []
    labs, sharps, inner_px, wb_flags = [], [], [], []
    parse_sources: list[str] = []
    inner_h, inner_s, inner_v, inner_labs = [], [], [], []

    for d in chosen:
        crop = _crop(d)
        if crop is None or crop.size == 0:
            continue
        sharps.append(laplacian_sharpness(crop))

        # 用**整段统一的增益**在浮点域校正，不逐帧各算各的、也不走
        # correct_image 的 uint8 量化路径。两个原因：
        # (1) 逐帧定位可能在不同帧选中不同参照物（打印机／白纸／桌沿），
        #     各自真实颜色不同，会注入噪声——实测使段内色相离散度由
        #     1.4–2.4° 恶化到 4.1–5.0°；改用段中位增益后为 1.0–2.1°。
        # (2) uint8 截断会把布料镜面高光压到 255，实测影响 0.49% 的衣服像素，
        #     且浅色衣物受影响大于深色，构成与明暗相关的系统偏差。
        corrected = crop.astype(np.float32) * np.asarray(gain, dtype=np.float32)
        wb_flags.append(1.0)

        g = parse_garments(crop)
        parse_sources.append(g.source)      # 记录本帧用的是模型还是整图兜底
        outer_raw = _mask_pixels(crop, g.outer)
        outer_cor = _mask_pixels(corrected, g.outer)
        if outer_raw is None or outer_cor is None:
            continue

        h, s, v = rgb_to_hsv_deg(dominant_color(outer_raw))
        raw_h.append(h); raw_s.append(s); raw_v.append(v)

        rgb_c = dominant_color(outer_cor)
        h, s, v = rgb_to_hsv_deg(rgb_c)
        cor_h.append(h); cor_s.append(s); cor_v.append(v)
        labs.append(rgb_to_lab(rgb_c))

        inner_px.append(g.inner_px)
        inner_cor = _mask_pixels(corrected, g.inner)
        if inner_cor is not None:
            ih, is_, iv = rgb_to_hsv_deg(dominant_color(inner_cor))
            inner_h.append(ih); inner_s.append(is_); inner_v.append(iv)
            inner_labs.append(rgb_to_lab(dominant_color(inner_cor)))

    if not cor_h:
        out = dict(_EMPTY_COLOR)
        out["sharpness_raw"] = float(np.mean(sharps)) if sharps else 0.0
        return _finalize_pixel_dict(out)

    hue_c = mean_hue_deg(cor_h)
    sat_c = float(np.mean(cor_s))
    val_c = float(np.mean(cor_v))
    lab = np.mean(np.asarray(labs), axis=0)

    cls11 = classify_hue_11(hue_c, sat_c, val_c)
    margin = hue_margin_deg(hue_c)
    stability = hue_spread_deg(cor_h)
    mean_inner_px = int(np.mean(inner_px)) if inner_px else 0

    out = {
        "outer_hue_raw": round(mean_hue_deg(raw_h), 2),
        "outer_sat_raw": round(float(np.mean(raw_s)), 2),
        "outer_val_raw": round(float(np.mean(raw_v)), 2),
        "outer_hue_corr": round(hue_c, 2),
        "outer_sat_corr": round(sat_c, 2),
        "outer_val_corr": round(val_c, 2),
        "outer_lab_L": round(float(lab[0]), 2),
        "outer_lab_a": round(float(lab[1]), 2),
        "outer_lab_b": round(float(lab[2]), 2),
        "outer_hue_valid": int(is_hue_valid(sat_c, val_c)),
        "outer_color_stability": round(stability, 2),
        "n_frames_color": len(cor_h),
        "hue_class_pixel": cls11,
        "hue_class_11": cls11,
        "hue_class_6": classify_hue_6(cls11),
        "warm_cool": classify_warm_cool(cls11),
        "lightness_class": classify_lightness(val_c),
        "chroma_class": classify_chroma(sat_c),
        "hue_class_margin": round(margin, 2),
        "hue_class_borderline": int(margin < config.HUE_BORDERLINE_MARGIN),
        "inner_visible": int(mean_inner_px > 0),
        "inner_pixel_count": mean_inner_px,
        "inner_quality": inner_quality(mean_inner_px),
        "sharpness_raw": float(np.mean(sharps)) if sharps else 0.0,
        "wb_ok_ratio": round(float(np.mean(wb_flags)), 3) if wb_flags else 0.0,
        # 分割来源：全部帧都走模型才算 model；有任一帧兜底则标出，供人工判断该行可信度
        "parse_source": ("model" if parse_sources and all(x == "model" for x in parse_sources)
                         else "kmeans_fallback"),
        "parse_fallback_frames": sum(1 for x in parse_sources if x != "model"),
    }

    if inner_h:
        ih = mean_hue_deg(inner_h)
        isat = float(np.mean(inner_s))
        ival = float(np.mean(inner_v))
        ilab = np.mean(np.asarray(inner_labs), axis=0)
        out.update({
            "inner_hue_corr": round(ih, 2),
            "inner_sat_corr": round(isat, 2),
            "inner_val_corr": round(ival, 2),
            "inner_lab_L": round(float(ilab[0]), 2),
            "inner_lab_a": round(float(ilab[1]), 2),
            "inner_lab_b": round(float(ilab[2]), 2),
            "inner_hue_class": classify_hue_11(ih, isat, ival),
        })
    return _finalize_pixel_dict(out)


def _poor_annotation(seg: Segment, seat: int) -> PersonAnnotation:
    """标注里有这个人、但画面里没找到时，仍占一行。

    spec 第 4 节：优先级是「准确度 > 样本保留 > 分辨率」，
    静默丢样本会让缺失率无法估计。
    """
    p = seg.presenters[seat - 1]
    a = PersonAnnotation(
        person_id=f"seg{seg.order:02d}_seat{seat}", seg_order=seg.order,
        company=seg.company, seat_index=seat, presenter_num=seg.presenter_num,
        gender_female=p.gender_female, speaks_pre=p.speaks_pre,
        speaks_qa=p.speaks_qa, seg_start=seg.start, seg_end=seg.end,
    )
    a.vlm = {"evidence_quality": "poor", "confidence": 0.0,
             "notes": "未在画面中定位到该座位"}
    a.pixel = _finalize_pixel_dict(dict(_EMPTY_COLOR))
    a.pixel["color_granularity_achieved"] = "无"
    a.pixel["hue_class_best"] = ""
    return a


def describe_layout(n_seated: int, n_entry: int = 0, n_exit: int = 0,
                    cell_index: int | None = None,
                    cell_indices: list[int] | None = None) -> str:
    """如实描述本次组图的构成（Task 26 · 需求一，续 Ruling R50；Task 42 扩展）。

    用户目视核对交付物后发现：证据图上排的进出场格里反复出现的是穿深色
    大衣、背对镜头来回走动的场地工作人员，不是创业者本人（order 2/6/12/
    13/15 座位1 五个不同项目都拍到了同一个人）。控制器排查过速度、路径长
    度、轨迹完整度等判别信号，没有一个能干净切分工作人员与正常人——与其
    继续调阈值，不如干脆不把进出场帧送入证据图，直接消灭这个风险来源。
    这个历史原因是 `n_entry=n_exit=0`（纯就座）分支至今仍然保守的依据：
    张数就是实际选中的就座帧数（1~4 张不等，不足 4 张不用进出场帧补位），
    没有进出场镜头，下装与整体站姿不可见，不要臆测。

    Task 42：项目即将把进出场的站立全身图混入证据图。原来的三句硬编码
    断言（不含进出场、就座画面为准绳、下装站姿一律不可见）只在"纯就座"
    这一种构成下成立，一旦证据图混入站立镜头就变成主动误导——站立镜头
    恰恰能看到下装与站姿，若仍告诉模型"看不到"，等于让它无视最有用的
    信息。`pipeline.annotate.PROMPT` 已改为把全部构成相关表述都委托给
    这里的 {layout} 文本，因此本函数现在要覆盖三种构成：

      1. 纯就座（n_entry=n_exit=0，向后兼容原签名/原调用点）：如上，
         保守——不含进出场画面，下装与站姿不可见，机位固定在左后侧方。
      2. 混合（n_seated>0 且 n_entry+n_exit>0）：明确点出哪几张是就座、
         哪几张是进出场站立镜头；**允许**据站立镜头判定下装/站姿（这是
         本次改动要解锁的核心行为）；机位方位随镜头变化（进场多背向、
         退场多正向），不再断言固定方位。
      3. 纯进出场（n_seated=0，仅在座位被前排完全挡住、没有可用就座帧
         时才会发生）：全部为站立镜头，下装/站姿可判定，同时提醒模型
         该人的就座画面本次不可用，避免模型去猜测/等待并不存在的就座
         证据。

    调用方目前仍只传 `n_seated`（`assemble_annotations` 尚未接入混合证据
    图，那是后续任务的范围）——保留原有单参数调用的完全向后兼容行为，是
    本函数扩展签名而非破坏式重写的原因。

    Task 43：三种构成的返回值统一以 `【{LAYOUT_HEADING}】` 开头——PROMPT 正文
    新增了「一律以上面『{LAYOUT_HEADING}』为准」这句，标题不带上，那句引用
    就没有明确指代对象。标题字面量取自 `pipeline.annotate.LAYOUT_HEADING`
    单一来源，不在这里另抄一份，防止两处各写一份、日后措辞漂移导致对不上
    （变异清单 (g)）。

    Task 51 · `cell_index`：第 4 种构成——**按格位切出的单张画面**。一个
    候选的框里挤了多个人时，用户在挑选页按证据图的格位编号写清哪一格是谁
    （`entryexit.parse_cell_note`），`entryexit.split_candidate_by_cells`
    据此把它切成若干个「只有一张图」的单人候选。这一张不再是四张拼图，
    构成文案必须如实说「共 1 张」并点明是第几格切出来的——沿用四张图的
    文案等于让模型去描述并不存在的另外三张画面，且下游无一处会报错
    （与 Task 42 那次「告诉模型下装不可见、而站立镜头里明明可见」同类）。
    传了 `cell_index` 就必须恰好是「1 张进出场镜头、0 张就座镜头」，否则
    抛 `ValueError`——参数组合自相矛盾时静默返回一段错的文案，比崩掉更糟。
    """
    heading = f"【{LAYOUT_HEADING}】"
    n_transition = n_entry + n_exit

    cells = ([int(c) for c in cell_indices] if cell_indices
             else ([int(cell_index)] if cell_index is not None else None))
    if cells:
        # Task 51：这几张是从一个「框里挤了多个人」的候选按格位切出来的。
        # 原候选的证据图是四张拼图、四张里坐着不同的人，用户在挑选页按格位
        # 号指明了哪一格是谁——沿用拼图的构成文案会让模型去描述并不存在的
        # 另外几张画面，且不报错（Ruling R50 那一类）。
        #
        # Task 56D（用户 2026-08-24：「登记表存多张就都给 VLM，增加信息量」）：
        # 同一个人可能占**多格**（「格位2 = 同格位1」），这几格都是他本人的
        # 证据，全部送进去。张数与格位数必须逐一对应：说了两格却只给一张图，
        # 等于让模型去找一张不存在的画面。参数自相矛盾时宁可崩掉，静默返回
        # 一段错的文案比崩掉更糟。
        cells = sorted(set(cells))
        if n_seated or n_transition != len(cells):
            raise ValueError(
                "cell_index/cell_indices 只用于按格位切出的进出场画面："
                f"进出场镜头张数应等于格位数 {len(cells)}、就座镜头应为 0，"
                f"实得 n_seated={n_seated}、n_entry={n_entry}、n_exit={n_exit}"
            )
        which = "进场" if n_entry else "退场"
        # 单格时**逐字**沿用 Task 51 的原文案：它同时是 Task 45 增量复用的
        # 「构成签名」，措辞动一个字就会让上一轮所有单格座位的结果失效重跑。
        if len(cells) == 1:
            what = (f"是从一张挤了多个人的合图里按编号第 {cells[0]} 格切出来的"
                    f"单张画面，图中只有该创业者本人")
        else:
            nums = "、".join(str(c) for c in cells)
            what = (f"是从一张挤了多个人的合图里按编号第 {nums} 格切出来的，"
                    f"这几格都是同一个人（该创业者本人），图中没有其他人")
        return (
            heading +
            f"本组图共 {len(cells)} 张站立的{which}镜头，{what}；"
            "该创业者本次没有其它可用画面，不要描述或推测本图之外的内容。"
            "站立镜头中下装（西裤/短裤等）与整体站姿清晰可见，可据此判定。"
            f"{which}镜头多为" + ("背向" if n_entry else "正向") + "拍摄，"
            "但不要假设固定的摄像机方位。"
        )

    if n_transition == 0:
        # 纯就座：与 Task 26 时的文案等价（含新增的机位一句，此前写死在
        # PROMPT 主体里，现在随构成移入这里）。
        return (
            heading +
            f"本组图共 {n_seated} 张就座镜头，全部来自该创业者本人，不含任何进出场画面。"
            "摄像机位于该人左后侧方。"
            "下装（西裤/短裤等）与整体站姿在本组图中一律不可见，不要据此推断或臆测下装。"
        )

    if n_seated == 0:
        # 纯进出场：座位被前排完全挡住、没有可用就座帧时的兜底构成。
        total = n_transition
        return (
            heading +
            f"本组图共 {total} 张站立的进出场镜头（进场 {n_entry} 张、退场 {n_exit} 张），"
            "全部来自该创业者本人；该创业者本次没有可用的就座镜头。"
            "站立镜头中下装（西裤/短裤等）与整体站姿清晰可见，可据此判定。"
            "机位方位随镜头变化：进场镜头多为背向拍摄，退场镜头多为正向拍摄，"
            "不要假设固定的摄像机方位。"
        )

    # 混合：就座 + 进出场站立镜头都在证据图里。
    total = n_seated + n_transition
    if n_entry and n_exit:
        trans_desc = f"另有 {n_entry} 张进场站立镜头、{n_exit} 张退场站立镜头"
    elif n_entry:
        trans_desc = f"另有 {n_entry} 张进场站立镜头"
    else:
        trans_desc = f"另有 {n_exit} 张退场站立镜头"
    return (
        heading +
        f"本组图共 {total} 张：其中 {n_seated} 张为该创业者本人的就座镜头，{trans_desc}。"
        "站立的进出场镜头中，下装（西裤/短裤等）与整体站姿清晰可见，可据此判定；"
        "就座镜头中下装与站姿仍不可见，不要据此臆测。"
        "机位方位随镜头变化：进场镜头多为背向拍摄，退场镜头多为正向拍摄，"
        "就座镜头位于该人左后侧方——不要假设整组图机位固定不变。"
    )


def _merge_hue_after_vlm_call(vlm: dict, pixel: dict) -> None:
    """VLM 调用（成功或失败）之后的双轨色相合并，原地修改 `pixel`。

    Task 45：从 `_vlm_and_hue_pass` 的 `if use_vlm:` 分支内原样抽出——增量
    打标命中缓存时（`assemble_annotations` 里的复用分支）需要对"上一轮的
    vlm 字典"跑完全相同的合并逻辑，两处必须共用同一份实现，不允许像
    `seats._ranked_tracks_dropping_expert` 提醒过的那样各写一份、日后走样
    （Ruling R39）。
    """
    if config.PIXEL_COLOR_ENABLED:
        if vlm.get("vlm_error"):
            # 调用失败，没有第二条轨可对：hue_class_agree 记 0（分歧），
            # 不是 1。use_vlm=False 分支记 1 是「刻意不调用」的诚实默认；
            # 这里是「调用了但失败」，没有 VLM 结论可比对，记 1 会把
            # "根本没对比过"伪装成"对比过且一致"，记 0 才诚实。
            pixel["hue_class_agree"] = 0
            pixel["hue_class_final"] = pixel.get("hue_class_pixel", "")
        else:
            agree, final = merge_tracks(
                pixel.get("hue_class_pixel", ""),
                vlm.get("outer_hue_class", ""),
            )
            pixel["hue_class_agree"] = int(agree)
            pixel["hue_class_final"] = final
    else:
        # Task 34 · 改动二：像素轨关闭，没有第二条轨可合并，hue_class_
        # final 直接采用 VLM 的色名（调用失败时留空，不能编造一个没有
        # 依据的颜色）。
        pixel["hue_class_final"] = (
            "" if vlm.get("vlm_error") else vlm.get("outer_hue_class", "")
        )


def _finalize_granularity(pixel: dict) -> dict:
    """粒度判定 + 按 `config.PIXEL_COLOR_ENABLED` 过滤，收尾步骤。

    Task 45：原 `_vlm_and_hue_pass` 末尾这段对三条分支（调用成功/调用失败/
    use_vlm=False）都是共用的，现独立抽出，供增量复用分支（跳过真正调用，
    直接拿上一轮的 vlm 字典）也走同一份收尾逻辑。
    """
    gran, best = choose_granularity(
        pixel.get("hue_class_pixel") or "灰",
        pixel.get("hue_class_margin") or 0.0,
        bool(pixel.get("hue_class_agree", 1)),
        pixel.get("outer_color_stability") or 0.0,
    )
    pixel["color_granularity_achieved"] = gran
    pixel["hue_class_best"] = best
    return _finalize_pixel_dict(pixel)


def _vlm_and_hue_pass(
    sheet: bytes, pixel: dict, context: dict, log_id: str, use_vlm: bool,
    vlm_failures: list[str],
) -> tuple[dict, dict]:
    """VLM 调用（含失败记录）+ 双轨色相合并 + 粒度判定，一份实现供正常
    候选人（第 1 步）与 Task 44 新增的进出场递补候选人共用（Task 44）。

    这段逻辑此前只内联在 `assemble_annotations` 的候选人循环里；递补候选人
    需要完全相同的失败处理、`hue_class_agree`/`hue_class_final` 语义、粒度
    判定，两处各写一份必然像 `seats._ranked_tracks_dropping_expert` 提醒的
    那样走样，因此抽成这个纯函数，行为与原内联代码逐字节相同。

    Task 45：内部的"合并"与"收尾"两段进一步抽成 `_merge_hue_after_vlm_call`
    / `_finalize_granularity`，供 `assemble_annotations` 的增量复用分支
    （跳过真正的 `ask_vlm` 调用，直接采用上一轮的 vlm 字典）复用，本函数
    行为逐字节不变。

    返回 (vlm 字典, 处理后的 pixel 字典——已含 hue_class_agree/final、
    color_granularity_achieved、hue_class_best，并已按
    config.PIXEL_COLOR_ENABLED 做过 `_finalize_pixel_dict` 过滤)。
    调用方须自行维护 `vlm_attempts` 计数（本函数不知道调用方的计数口径，
    只在 `use_vlm=True` 时才发起真正的调用）。
    """
    if use_vlm:
        try:
            # Task 34 · 改动一：context 不再含座位号——VLM 只需要「一个人」，
            # 红框已经指明了要描述谁；座位号既没必要，又会诱导模型把
            # 「座位1=主讲人」当成身份线索去回答 gender_judged 等本该独立
            # 判断的字段。
            vlm = ask_vlm(sheet, context)
        except Exception as exc:
            # 单人 VLM 调用失败不得拖垮整批连续调用（Ruling R54）。记显式
            # 失败态而非空字典/乐观默认值，人工复核在 CSV 里看得见。只取
            # 异常类型名与 str(exc)——不含 traceback、请求头、环境变量；
            # str(exc) 可能带中转服务的 URL 属可接受范围，但要截断，防止
            # 把整段响应体糊进一格 CSV。
            #
            # Task 35 · 改动三：可重试的服务端故障（5xx/429/连接错误/
            # VlmParseError）已在 `ask_vlm` 内部按指数退避重试过（最多
            # config.VLM_MAX_RETRIES 次），这里捕到的要么是重试耗尽后的最终
            # 失败（异常消息已含尝试次数，见 `ask_vlm` 的 docstring），要么
            # 是一开始就判定不可重试的 4xx（图太大/格式不对/鉴权失败），
            # 两种都应如实记录，不再在这一层额外重试。
            vlm_failures.append(log_id)
            msg = str(exc)
            if len(msg) > _VLM_ERROR_MAX_LEN:
                msg = msg[:_VLM_ERROR_MAX_LEN] + "…"
            vlm = {
                "evidence_quality": "poor",
                "confidence": 0.0,
                "vlm_error": f"{type(exc).__name__}: {msg}",
            }
        _merge_hue_after_vlm_call(vlm, pixel)
    else:
        vlm = {"evidence_quality": "fair", "confidence": 0.5}
        if config.PIXEL_COLOR_ENABLED:
            pixel["hue_class_agree"] = 1
            pixel["hue_class_final"] = pixel.get("hue_class_pixel", "")
        else:
            pixel["hue_class_final"] = ""

    pixel = _finalize_granularity(pixel)
    return vlm, pixel


def _can_reuse_prior_annotation(prior_row: dict | None, current_layout: str,
                                 force_vlm: bool) -> bool:
    """Task 45 增量打标复用判据。四条必须全部满足才复用，任一条不满足都
    重跑（宁可多算不可少算——见 `config.VLM_INCREMENTAL` 上方的完整说明）：

      1. 存在上一轮记录（`prior_row` 不是 None/空）。
      2. `vlm` 字典非空。
      3. `vlm_error` 为空——上次失败的人正是要重试的对象（brief 变异
         清单 (a)：增量判据不得忽略 vlm_error）。
      4. 上一轮存的 `layout` 与本轮完全一致——证据图构成变了，提示词就
         变了，复用旧判断等于拿错误前提下的结论充数（brief 变异清单 (b)）。
         旧文件没有 `layout` 键（Task 45 之前产出的 annotations.json）时，
         视为"构成未知"，保守重跑，不当作"和现在一样"。

    `force_vlm=True`（对应 `--force-vlm`）或 `config.VLM_INCREMENTAL=False`
    时整体短路为不可复用（brief 变异清单 (d)）。
    """
    if force_vlm or not config.VLM_INCREMENTAL:
        return False
    if not prior_row:
        return False
    vlm = prior_row.get("vlm") or {}
    if not vlm:
        return False
    if vlm.get("vlm_error"):
        return False
    if "layout" not in prior_row:
        return False
    if prior_row.get("layout") != current_layout:
        return False
    return True


def _track_y2(dets_of_track: list[Detection]) -> float:
    """轨迹全部检测框 bottom_y 的中位数，供性别匹配当「越靠前」的排序键。"""
    vals = sorted(d.bottom_y for d in dets_of_track)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def assemble_annotations(
    segments: list[Segment],
    dets: list[Detection],
    seats_by_order: dict[int, dict[int, int]],
    use_vlm: bool = True,
    pairs_by_order: dict[int, dict[int, dict]] | None = None,
    staff_ids: set[str] | None = None,
    entry_labels: dict | None = None,
    prior_annotations: list[dict] | None = None,
    force_vlm: bool = False,
) -> list[PersonAnnotation]:
    """把候选人测量、VLM 打标、性别匹配定座位号合成每人一条记录。

    use_vlm=False 用于测试与离线重算——跳过 API 调用，只保留像素轨。
    pairs_by_order 来自 Task 16 的 pair_transitions，缺省为 None 时
    **不使用任何进出场帧**（而不是退回按 track_id 猜，见 Ruling R18）。
    staff_ids 来自 Task 17 的 flag_suspect_staff，只置标志、不删除任何人。

    `seats_by_order` 的含义（Task 34 · 改动一）：**基线顺序**，不是最终座位
    号——由阶段 4 的纯几何排序（`assign_segment_seats` 默认 `gender_match=
    False`）给出，形如 `{track_id: baseline_seat}`。本函数内部分三步跑，
    彻底消灭「VLM 要性别、性别要座位、座位又要 VLM」这个先有鸡还是先有蛋
    的循环（旧流程必须先跑一遍拿到 gender_judged，再重跑阶段 4 重排座位、
    重跑阶段 7 的 VLM——两遍）：

      1. **候选人测量 + VLM 打标**（不知道、也不需要知道最终座位号）：
         按基线顺序逐个候选人测颜色、构建证据图、调用 `ask_vlm`——VLM 的
         PROMPT 已不再包含座位号（Task 34 · 改动一），红框已经指明了要
         描述谁，模型不需要、也不该知道座位号。
      2. **性别匹配定座位号**：候选人数与标注人数一致时，把每个候选人的
         `gender_judged`（刚从第 1 步拿到，同一次调用内产出，不必等
         "重跑"）与人工标注的性别序列一起交给
         `pipeline.seatmatch.match_seats_by_gender`，得到最终座位号。人数
         不等时（缺座位）保留基线顺序，交由下面第 3 步的缺座位占位处理，
         语义与旧的 `_poor_annotation` 完全一致。
      3. **按最终座位号产出 `PersonAnnotation`**：ground truth（gender_
         female/speaks_pre/speaks_qa）取最终座位号对应的 `seg.presenters`
         条目——这正是性别匹配存在的意义：候选人本身携带的信息（VLM 判定）
         决定了他该对应标注里的哪一位，而不是死板地按基线顺序对应。
      4. **进出场递补**（Task 44）：`entry_labels` 缺省为 None——用户尚未
         提交 `entry_review.html` 的标注时，本步骤完全不执行，结果与
         Task 44 之前逐字段相同（这是硬要求，见 brief 变异清单 (a)）。
         传入标注后，对每段仍空着的座位，调用
         `pipeline.entryexit.entryexit_fill_plan` 找出该补哪条进出场轨迹，
         用该轨迹的证据帧重新测颜色、构建证据图、调用 VLM——证据图构成
         走 `describe_layout` 的"纯进出场"分支（Task 42 第 3 种构成），
         绝不能沿用第 1 步就座候选人用的"纯就座"分支（那条断言"下装不
         可见"，会误导模型忽视站立镜头里明明可见的下装，见 brief 变异
         清单 (d)）。产出的行带三个标志字段：`source="entryexit"`、
         `seat_uncertain`（来自 fill_plan，用户是否明确指定了座位号）、
         `expert_removed`（该段是否发生过 `Innermost_expert` 剔除，见
         `pipeline.seats.expert_removed_tracks`）；正常就座行的这三个
         字段维持默认值 `"seated"`/`0`/`0`。
      5. **增量打标复用**（Task 45）：`prior_annotations` 缺省为 None——
         调用方不传上一轮结果时，本步骤完全不生效，结果与 Task 45 之前
         逐字段相同（同 Task 44 的向后兼容硬要求）。传入时（`run_all.py`
         阶段⑦会传上一轮 annotations.json 的原始 dict 列表），对每个候选
         按 `_can_reuse_prior_annotation` 的四条判据决定是否跳过真正的
         `ask_vlm` 调用、直接复用上一轮的 `vlm` 字典——存在的意义是让
         "递补带来的变化"与"VLM 自身运行间波动"能分离开看（VLM 存在真实
         波动，见 `config.VLM_INCREMENTAL` 上方的实测记录）。`force_vlm=
         True`（对应 CLI `--force-vlm`）无条件忽略缓存，全部重新调用。
         就座候选（第 1 步）按 `(seg_order, track_id)` 查上一轮记录——此时
         最终 person_id 尚未确定（性别匹配还没跑），track_id 才是跨轮次
         稳定的身份标记；递补候选（第 4 步）按 person_id（即目标座位号）
         查，因为那里座位号在调用 VLM 之前就已确定，并额外核对 track_id
         一致，防止本轮递补候选换了人却因 layout 恰好相同而被误判可复用。
    """
    if entry_labels is not None:
        from pipeline.entryexit import entryexit_candidates, entryexit_fill_plan
        from pipeline.seats import expert_removed_tracks

    # Task 45：把上一轮 annotations.json 的原始行按两种身份键分别建索引，
    # 供下面两处复用判定使用（见上方 docstring 第 5 点的键选择理由）。
    prior_seated_by_seg_tid: dict[tuple[int, int], dict] = {}
    prior_entryexit_by_person: dict[str, dict] = {}
    for row in (prior_annotations or []):
        pix = row.get("pixel") or {}
        tid = pix.get("track_id")
        seg_o = row.get("seg_order")
        if row.get("source") == "entryexit":
            pid = row.get("person_id")
            if pid:
                prior_entryexit_by_person[pid] = row
        elif tid is not None and seg_o is not None:
            prior_seated_by_seg_tid[(int(seg_o), int(tid))] = row

    out: list[PersonAnnotation] = []
    vlm_failures: list[str] = []
    vlm_attempts = 0
    gender_report_rows: list[dict] = []
    backfill_count = [0]
    n_reused = [0]
    n_vlm_calls = [0]
    needs_seat_total: list[dict] = []
    # 跨段流转（「上/下一组创业者」）要按 order 找相邻段，见下面 window 处。
    seg_by_order = {int(s.order): s for s in segments}

    for seg in segments:
        seats = seats_by_order.get(seg.order, {})
        by_baseline_seat = {v: k for k, v in seats.items()}
        # 补格位用的候选表：只含「用户亲手指名给某个座位的创业者」。与
        # registry 共用同一个函数，两处各算一份必然日后走样。
        # 相邻段标过来的（「上/下一组创业者」）同样是本段的补帧来源——与
        # `registry` 同一个函数、同一批邻段，两处各算一份必然走样（走样的
        # 表现是核对页与模型看到的画面对不上，且不报错）。
        topup_by_seat = labelled_seat_candidates(
            dets, seg, entry_labels,
            neighbours=[s for s in (seg_by_order.get(seg.order - 1),
                                    seg_by_order.get(seg.order + 1))
                        if s is not None])
        # 已定座位改号（Task 56V/56W）。**第 1 步就要拿到**：补帧要按改号后的
        # 座位号去查那个人的候选。它按基线座位号索引，不依赖尚未算出的最终
        # 座位号，所以可以提前算。第 2 步的落位另用同一份（见下）。
        _seg_reassign = seat_reassign_of(entry_labels, seg.order)
        # 整段一个增益，供本段所有人共用（见 segment_gain 的 docstring）
        gain = segment_gain([d for d in dets if d.seg_order == seg.order])

        # --- 第 1 步：逐候选人测量 + VLM 打标，此时尚未确定最终座位号 ---
        per_track: dict[int, dict] = {}
        for baseline_seat in range(1, seg.presenter_num + 1):
            tid = by_baseline_seat.get(baseline_seat)
            if tid is None:
                continue

            seated = [d for d in dets if d.seg_order == seg.order
                      and d.kind == "seated" and d.track_id == tid]

            # 进出场的 track_id 与就座的 track_id 不是同一套编号（Ruling R18）：
            # 跟踪器在每个 (seg_order, kind) 窗口都从 1 重新编号，同一项目内
            # entry 的 5 号与 seated 的 5 号不是同一个人。必须用 Task 16 的配对
            # 结果查出本座位对应的 entry/exit track_id，**绝不能直接拿 tid 去筛**
            # ——那样会把甲的入场照和乙的就座照拼进同一张证据图，且不报任何错。
            # 配对结果按**基线座位号**存放（pair_transitions 在阶段 4 之后、
            # 本函数之前跑，用的就是基线顺序），与最终座位号无关，这里必须用
            # baseline_seat 去查，不能用尚未算出的最终座位号。
            pair = (pairs_by_order or {}).get(seg.order, {}).get(baseline_seat, {})
            trans = []
            for kind, key in (("entry", "entry_track"), ("exit", "exit_track")):
                ttid = pair.get(key)
                if ttid is None:
                    continue          # 配不上就没有这一路：宁可少一张图，不能拼错人
                trans += [d for d in dets if d.seg_order == seg.order
                          and d.kind == kind and d.track_id == ttid]

            color = measure_person_color(seated or trans, gain=gain)
            pixel = dict(color)
            # 供 export_keyframes 按人筛帧用（R13）；绝不进 CSV——annotation_to_row
            # 只摊平标量，整数本身没问题，但要与 _evidence_png 那条二进制键放在
            # 一起说明：这里存的是普通 int，是安全的。
            pixel["track_id"] = tid

            best_seated = select_best_frames(seated, n=2) if seated else []
            best_trans = select_best_frames(trans, n=2) if trans else []
            best_frame_seated = str(best_seated[0].frame_path) if best_seated else ""
            best_frame_entry = str(best_trans[0].frame_path) if best_trans else ""

            # 证据图（2x2、红框标出目标）无论 use_vlm 与否都要构建（Task 25
            # 缺陷一）：人工复核（Ruling R13）需要它来判断"这行对应的确实是
            # 这个人穿了什么"，与要不要调 VLM 模型是两回事——--no-vlm 模式下
            # gallery.py 同样要靠它出缩略图，而不是把 1440x1080 整帧硬缩成
            # 520x390（那样创业者只占最右一小条，人根本看不清衣服）。
            # 字节暂存，供 export_keyframes 写盘核验用；写盘后必须从 pixel 中
            # 删除，绝不能让二进制内容进 CSV。存在 try/except 之外：调用失败
            # 时人工复核也要能看到本该送去的证据图。
            #
            # Task 26 · 需求一：只取该人的就座帧，绝不再混入进出场帧
            # （best_trans）——用户核对交付物发现进出场格里出现的是来回走动
            # 的场地工作人员，不是创业者本人，且控制器排查过全部可用判别
            # 信号都无法干净切分（见 describe_layout 的 docstring）。就座帧
            # 不足 4 张时按实际张数组图，不用进出场帧补位；build_evidence_
            # sheet 本身已支持不足 4 张时以灰块补齐，不会因此崩溃。
            # best_trans/best_frame_entry 不受影响：那两者仍照常导出到
            # keyframes/，只是不再进证据图。
            evidence_seated = select_best_frames(seated, n=4) if seated else []
            # 逐格排除与补格位都要在**算构成文案之前**做完（用户 2026-08-24 /
            # 2026-08-25）。此前排除在文案之后才应用，勾掉一格之后文案仍写着
            # 「共 4 张」而图里只有 3 张——模型被告知去看一张不存在的画面，
            # 且一处不报错。
            #
            # 用 baseline_seat：此处最终座位号（seat）还未算出，而用户在挑选页
            # 看到并勾选的正是这个阶段展示的座位号。
            _ex = cell_exclusions_of(entry_labels, seg.order, baseline_seat)
            _kept_seated = apply_cell_exclusions(evidence_seated, _ex)
            _all_excluded = bool(_ex) and not _kept_seated
            # 勾掉的那一格通常只是跟踪串号框错了人，不代表这位创业者没有别的
            # 画面——用**他自己那个人工标定的**进出场候选补回来，与 registry
            # 共用 labelled_seat_candidates / topup_frames（两处各算一份必然
            # 走样，走样的表现是核对页与模型看到的画面对不上，且不报错）。
            # **按改号后的座位号查**（Task 56W）：用户把那个人的进出场候选
            # 标成的是新号。用户 2026-08-29 实测城市2 order 6「自动标注的座位1
            # 改为座位2……应与后边人工标注的座位2进行合并」——按基线号查等于
            # 拿不到那条标注，两边的图各走各的，模型看到的比核对页少一半。
            _topup_seat = _seg_reassign.get(baseline_seat, baseline_seat)
            _evidence = topup_frames(_kept_seated,
                                     topup_by_seat.get(_topup_seat), want=4)
            # 构成**按最终那几张各自的 kind 数**，不按长度差、不按位置切。
            # 用户 2026-09-26 起多个来源轮流出格：四格就座 + 一条退场标注合成
            # 的仍是 4 张（2 就座 + 2 退场），张数不变多。此前这里用
            # 「比就座那几张多了几张」判断补没补站立镜头，多出 0 张就走纯就座
            # 那支，告诉模型「不含任何进出场画面，下装一律不可见」；用
            # `_evidence[len(_kept_seated):]` 切出「补进来的」也不再成立——
            # 就座的已不再全排在前面。两处都是在对模型说谎，且一处不报错。
            _n_seated_final = sum(1 for d in _evidence if d.kind == "seated")
            _n_entry_up = sum(1 for d in _evidence if d.kind == "entry")
            _n_topup = sum(1 for d in _evidence if d.kind != "seated")
            # 如实描述本次组图的构成（Ruling R50，Task 26 需求一续）。补进来的
            # 是站立全身镜头，文案必须走「混合」那一支——沿用纯就座那句「下装
            # 与站姿一律不可见」等于让模型无视最有用的信息（Task 42 同一件事）。
            # 这个文案本身就是 Task 45 增量复用判据要比对的"构成签名"，必须在
            # 决定是否复用之前算出来：格位一变、补帧一变，文案随之变，上一轮的
            # 结果就不会被误判为可复用。
            # 进场几张、退场几张**按补进来的帧自己数**，不去问候选的 kind：
            # 一个座位可能被标在好几条候选上（用户 2026-08-26），一进一退时
            # 拿第一条的 kind 去代表全部，文案就会把退场的说成进场。
            layout_text = describe_layout(
                _n_seated_final,
                n_entry=_n_entry_up,
                n_exit=_n_topup - _n_entry_up,
            ) if _n_topup else describe_layout(_n_seated_final)
            log_id = f"seg{seg.order:02d}_track{tid}"

            prior_row = prior_seated_by_seg_tid.get((seg.order, tid))
            if _can_reuse_prior_annotation(prior_row, layout_text, force_vlm):
                # Task 45 增量复用命中：跳过真正的 ask_vlm 调用，也不重建
                # 证据图字节（不调 build_evidence_sheet）——原图仍在
                # keyframes/ 磁盘上（export_keyframes 的复用路径会跳过重写，
                # 见其 docstring），不主动从磁盘读回塞进内存态 pixel（Ruling
                # R57：`_evidence_png` 本就是"待写盘"的暂存态，不该被反向
                # 填充）。双轨合并/粒度判定与"刚调用成功"走同一份函数，不
                # 另抄一份（Ruling R39）。
                n_reused[0] += 1
                vlm = dict(prior_row.get("vlm") or {})
                _merge_hue_after_vlm_call(vlm, pixel)
                pixel = _finalize_granularity(pixel)
            else:
                n_vlm_calls[0] += 1
                # 证据图（2x2、红框标出目标）无论 use_vlm 与否都要构建（Task 25
                # 缺陷一）：人工复核（Ruling R13）需要它来判断"这行对应的确实是
                # 这个人穿了什么"，与要不要调 VLM 模型是两回事——--no-vlm 模式下
                # gallery.py 同样要靠它出缩略图，而不是把 1440x1080 整帧硬缩成
                # 520x390（那样创业者只占最右一小条，人根本看不清衣服）。
                # 字节暂存，供 export_keyframes 写盘核验用；写盘后必须从 pixel 中
                # 删除，绝不能让二进制内容进 CSV。存在 try/except 之外：调用失败
                # 时人工复核也要能看到本该送去的证据图。
                #
                # Task 26 · 需求一：只取该人的就座帧，绝不再混入进出场帧
                # （best_trans）——用户核对交付物发现进出场格里出现的是来回走动
                # 的场地工作人员，不是创业者本人，且控制器排查过全部可用判别
                # 信号都无法干净切分（见 describe_layout 的 docstring）。就座帧
                # 不足 4 张时按实际张数组图，不用进出场帧补位；build_evidence_
                # sheet 本身已支持不足 4 张时以灰块补齐，不会因此崩溃。
                # best_trans/best_frame_entry 不受影响：那两者仍照常导出到
                # keyframes/，只是不再进证据图。
                # 逐格排除（用户 2026-08-24）：2×2 的四格来自同一条 track 的
                # 四帧，跟踪串号会让某一帧的框实际框到别人。四格红框标的都是
                # 同一个 track，模型无从分辨哪格才是目标，会把两个人的衣着混
                # 起来判断且不报错。用户在挑选页勾掉那一格，这里跳过它。
                # 排除与补格位已在上面算好（必须先于构成文案），这里直接用。
                _ev = _evidence
                if _all_excluded:
                    # 四格全被排除：如实报告没有可用证据，绝不静默塞一张
                    pixel["_evidence_png"] = None
                    pixel["cells_all_excluded"] = 1
                else:
                    pixel["_evidence_png"] = build_evidence_sheet(
                        [d.frame_path for d in _ev],
                        [(d.x1, d.y1, d.x2, d.y2) for d in _ev],
                    )
                sheet = pixel.get("_evidence_png")

                if use_vlm:
                    vlm_attempts += 1
                vlm, pixel = _vlm_and_hue_pass(
                    sheet, pixel,
                    {
                        "company": seg.company,
                        "presenter_num": seg.presenter_num,
                        "layout": layout_text,
                        # Task 35 · 改动三：只用于重试期间的进度打印
                        # （PROMPT.format 对多余键安全忽略，同 Task 29）。
                        "person_id": log_id,
                    },
                    log_id, use_vlm, vlm_failures,
                )

            per_track[tid] = {
                "baseline_seat": baseline_seat,
                "pixel": pixel,
                "vlm": vlm,
                "best_frame_seated": best_frame_seated,
                "best_frame_entry": best_frame_entry,
                "y2": _track_y2(seated or trans),
                "layout": layout_text,
            }

        # --- 第 2 步：性别匹配定最终座位号 ---
        # 候选人数与标注人数一致时才尝试重排——人数不等时性别多重集根本
        # 无法比较（`match_seats_by_gender` 自身也是这个约定），保留基线
        # 顺序，交给第 3 步的缺座位占位处理，与旧行为完全一致。
        final_seat_by_tid: dict[int, int] = {
            tid: info["baseline_seat"] for tid, info in per_track.items()
        }
        if seg.presenters and len(per_track) == seg.presenter_num:
            people = [
                {"track_id": tid, "y2": info["y2"],
                 "gender_judged": info["vlm"].get("gender_judged", "不确定")}
                for tid, info in sorted(per_track.items(),
                                        key=lambda kv: kv[1]["baseline_seat"])
            ]
            annotated_genders = [p.gender_female for p in seg.presenters]
            final_seat_by_tid = match_seats_by_gender(people, annotated_genders)
            gender_report_rows.append({
                "seg_order": seg.order, "people": people,
                "annotated_genders": annotated_genders,
            })

        # 已定座位改号（Task 56V，用户 2026-08-29 实测城市2 order 6）。在这里
        # 落地是因为**座位号只在这一处定下来**——第 3 步、递补、证据图、
        # 打标全都读 `final_seat_by_tid`。放到下游任何一处都会漏掉其余几处，
        # 表现是「核对页显示座位 3、模型产出的却记在座位 1 名下」，且不报错。
        #
        # 键按**基线座位号**查：用户在挑选页看到并改的就是那个号（`filled_
        # seats` 出自 `seats_by_order`），不是性别匹配之后的号。人工的改号
        # **压过**性别匹配——他是看着图说的，匹配只是统计推断。
        reassign = _seg_reassign
        if reassign:
            moved = {tid: reassign[info["baseline_seat"]]
                     for tid, info in per_track.items()
                     if info["baseline_seat"] in reassign}
            taken = set(moved.values())
            # 被顶掉的那一位**不自己找别的座位坐下**：用户只说了「这张图是
            # 座位 N」，没说被顶掉的是谁。替他猜一个是本项目最忌讳的那类错，
            # 落进缺座位占位（第 3 步）等他指认。
            #
            # `st not in taken` 这一半是**让位**，缺了它目标座位上会同时坐着
            # 两条 track，谁最终胜出取决于下一行 `by_final_seat` 反转时的
            # 字典插入顺序。曾按「update 恒后写、恒胜」把它当等价变异删过一
            # 次——那个理由只在依赖插入顺序时成立，把合并写成
            # `{**moved, **final}` 结果立刻反过来且一条测试都不红。靠插入
            # 顺序保正确性太脆，这里让结果与顺序无关。
            #
            # 代价是「合并顺序」那条变异就此成为等价变异（两边键不相交）。
            # 这是有意的取舍：宁可留一个等价变异，不要一个靠字典顺序的正确性。
            final_seat_by_tid = {tid: st for tid, st in final_seat_by_tid.items()
                                 if tid not in moved and st not in taken}
            final_seat_by_tid.update(moved)
            # 一个座位只能坐一个人。这条断言让**过滤**成为承重的那一条：
            # 少了它，目标座位上会同时坐着两条 track，谁胜出取决于下一行
            # `by_final_seat` 反转时的字典插入顺序——那样过滤与 `update` 顺序
            # 互为替代，同一时刻只有一条承重、另一条必是等价变异，留着就是
            # 一条假防线（变异测试两次确认过）。有了它，删掉过滤当场就炸。
            assert len(set(final_seat_by_tid.values())) == len(final_seat_by_tid), (
                f"第 {seg.order} 段改号后有座位坐了两个人：{final_seat_by_tid}")

        by_final_seat = {seat: tid for tid, seat in final_seat_by_tid.items()}

        # --- 第 3 步：按最终座位号产出 PersonAnnotation ---
        for seat in range(1, seg.presenter_num + 1):
            tid = by_final_seat.get(seat)
            if tid is None:
                out.append(_poor_annotation(seg, seat))
                continue

            info = per_track[tid]
            p = seg.presenters[seat - 1]
            a = PersonAnnotation(
                person_id=f"seg{seg.order:02d}_seat{seat}", seg_order=seg.order,
                company=seg.company, seat_index=seat,
                presenter_num=seg.presenter_num, gender_female=p.gender_female,
                speaks_pre=p.speaks_pre, speaks_qa=p.speaks_qa,
                seg_start=seg.start, seg_end=seg.end,
            )
            a.pixel = info["pixel"]
            a.vlm = info["vlm"]
            a.best_frame_seated = info["best_frame_seated"]
            a.best_frame_entry = info["best_frame_entry"]
            # 只置标志、绝不删除：被误标者要能在画廊点「恢复为创业者」找回（Ruling R19）
            a.suspect_staff = int(a.person_id in (staff_ids or set()))
            a.layout = info["layout"]   # Task 45：供下一轮增量复用判据比对
            out.append(a)

        # --- 第 4 步（Task 44）：进出场递补 ---
        #
        # entry_labels=None 是硬要求的向后兼容开关（brief 变异清单 (a)）：
        # 用户尚未提交 entry_review.html 的标注时，本步骤必须整段不执行，
        # 不能碰 out 里的任何一行。只有传了标注才走到这里。
        if entry_labels is not None:
            # 只补「本段确实还有空座位」——`by_final_seat` 恰好是本段已占用
            # 座位的反查表，直接喂给 entryexit_fill_plan 的 seats_by_order
            # 形参（该形参只看 .values() 取占用集合，见其 docstring）。
            seats_for_fill = {seg.order: {tid: seat for seat, tid in by_final_seat.items()}}
            # 传 order-1 / order / order+1 **三段**，与 `registry` 那边逐字
            # 同一套做法。
            #
            # 只传 `[seg]` 会让跨段流转（「上/下一组创业者」）**在打标这一步
            # 彻底失效**：源段的标注产生的计划项落在 `plan[目标段号]`，而本轮
            # 只读 `plan[seg.order]`；轮到目标段时又只传目标段自己，源段的
            # 标注根本不会被扫到。表现极隐蔽——登记表里有这个人（registry 用
            # 的就是三段窗口）、核对页里有他的图，最终 CSV 里却没有他这一行，
            # 且沿途只打印一条「没有第 N 段」，看上去像是数据本身缺一段。
            # 本函数里那段 `cross_segment` / `src_order` 的处理（Task 53 需求
            # 六）因此从写下起就一次也没执行到。
            #
            # 相邻段自己的普通补位项落在 `plan[相邻段号]`，下面只取
            # `plan[seg.order]`，不会串进来。
            window = [s for s in (seg_by_order.get(seg.order - 1), seg,
                                  seg_by_order.get(seg.order + 1)) if s is not None]
            # 这里**不给相邻段传座位占用表**，而且这是想清楚之后的决定：
            # 下面只取 `plan[seg.order]`，而目标段是否已有人由
            # `seats_for_fill[seg.order]`（本段的 by_final_seat，已含人工否决
            # 的结果）判定，相邻段那两份怎么填都影响不到它。填了反而不一致
            # ——本段用的是否决之后的占用，相邻段拿到的却是就座基线。
            # 变异测试确认过：把这几行去掉一个测试都不红（等价变异）。
            plan, _needs_resplit, seg_needs_seat = entryexit_fill_plan(
                window, dets, entry_labels, seats_for_fill,
                # 同 registry：三段窗口 + 整组段号，免得把「不在窗口里」误报
                # 成「没有这一段」。
                known_orders=[int(sg.order) for sg in segments],
            )
            # needs_seat 只收**本段**那几条：三段窗口会把相邻段的也算出来，
            # 全收进去会让同一条在跑到相邻段时被重复报一遍。
            needs_seat_total.extend(
                x for x in seg_needs_seat if int(x.get("order", 0)) == seg.order)

            # 人工在挑选页标为「这个座位判错了」的座位号。**必须在
            # `if entries:` 之外处理**——用户可能只否决、不指定替代候选，
            # 那时 plan 为空，若把标志写在 entries 分支里，用户说了判错却
            # 什么都不会发生且不报错（本轮实测复现过）。
            #
            # 下面那道 `seat in by_final_seat` 防御是为了不让**自动**递补
            # 悄悄改写人工可见的座位表；用户**显式否决**的座位不在此列。
            # 两者的区别在标注文件里有没有那条 override，不在于座位是否
            # 已被占用。
            from pipeline.entryexit import vacated_seats_of
            rejected = vacated_seats_of(entry_labels, seg.order)
            for row in out:
                if row.seg_order == seg.order and row.seat_index in rejected:
                    row.seat_rejected = 1

            entries = plan.get(seg.order, [])
            if entries:
                # 该段是否发生过专家剔除——与 assign_segment_seats 共用同一份
                # 判定（seats.expert_removed_tracks，Task 44），不是本函数
                # 另猜一遍。这是段级信息，本段全部行（含未被递补覆盖的就座
                # 行）一并标注，不止标在递补行上——否则同一段内，剔除专家
                # 这件事对没被递补覆盖的就座行而言就"查无实据"。
                seg_expert_removed = int(bool(expert_removed_tracks(dets, seg)))
                for row in out:
                    if row.seg_order == seg.order:
                        row.expert_removed = seg_expert_removed


                # 候选池同样覆盖三段窗口：跨段项的帧来自**源段**的时间窗口，
                # 用本段的候选池去找必然找不到——而找不到就 `continue`，那条
                # 流转会静默消失（registry 那边逐字同一条注释，同一个坑）。
                cand_by_key = {}
                for _s in window:
                    # 展示 ∪ 折叠：用户可能把人标在折叠区的候选上。
                    for c in candidates_for_labels(dets, _s):
                        cand_by_key.setdefault(f"{c['kind']}:{c['track_id']}", c)
                from pipeline.entryexit import cell_dets as _cell_dets
                for item in entries:
                    key = f"{item['kind']}:{item['track_id']}"
                    cand = cand_by_key.get(key)
                    # Task 56D：这一项占的全部格位（升序）。空 = 不是按格位
                    # 切出来的，那时传给 VLM 的是整条 track 的四格拼图，
                    # 「位置 i ↔ 格位 i」成立。非空则拿到的是一个**子集**，
                    # 两者的排除口径不同，下面必须分开处理。
                    item_cells = [int(c) for c in (item.get("cell_indices") or [])]
                    if not item_cells and item.get("cell_index"):
                        item_cells = [int(item["cell_index"])]
                    # Task 51：`cell_index` 表示这一项是从一个「框里挤了
                    # 多个人」的候选按格位切出来的——**只能用那一格的那张
                    # 图**。用整条 track 的四张证据帧去测颜色、拼证据图，
                    # 等于把四个不同的人混着编码（Ruling R14 那一类，且
                    # 不报错）。取哪一张与 `split_candidate_by_cells` 共用
                    # `entryexit.cell_dets`，不在这里另切一遍。
                    # 每一帧连同**它是第几格**一起带着走。此前格位号只隐含
                    # 在列表位置里（「位置 i ↔ 格位 i」），那个等式只有传完整
                    # 四格拼图时才成立；切分出来的座位拿到的是子集，位置与格
                    # 位号就此错开，而下面的逐格排除偏偏要按格位号判断。写成
                    # 显式的 (格位号, 帧) 之后，两条路径共用同一份数据，颜色
                    # 测量与证据图也不会各算一遍（各算一遍就会一处改了一处没
                    # 改，且不报错）。
                    if cand is None:
                        trans_pairs = []
                    elif item_cells:
                        # 用户 2026-08-24：「登记表存多张就都给 VLM，增加
                        # 信息量」。同一个人占多格（「格位2 = 同格位1」）时
                        # 这几格都是他本人的证据，全部送进去；此前只送
                        # `cell_index` 那一张代表格，另一张留在登记表里谁也
                        # 用不上。取哪一格用 `entryexit.cell_dets`，与
                        # `split_candidate_by_cells`、登记表同一处，不另切。
                        trans_pairs = [(c, d) for c in sorted(set(item_cells))
                                       for d in _cell_dets(cand, c)]
                    else:
                        # 整条 track 的四格拼图：位置号本来就是格位号。
                        trans_pairs = list(enumerate(cand["best_dets"], start=1))
                    trans_dets = [d for _c, d in trans_pairs]
                    if not trans_dets:
                        continue

                    seat = item["seat"]
                    if seat in by_final_seat and seat not in rejected:
                        # 防御性重复检查：entryexit_fill_plan 自身已经只吐出
                        # 空座位（见其 docstring 与 test_fill_plan_only_
                        # fills_unoccupied_seats），这里再判一次，绝不允许
                        # 递补覆盖已有的就座结果（brief 变异清单 (f)）。
                        continue

                    color = measure_person_color(trans_dets, gain=gain)
                    pixel = dict(color)
                    pixel["track_id"] = cand["track_id"]

                    # 证据图构成必须走进出场分支（Task 42 的第 3 种构成：
                    # 纯进出场，n_seated=0），绝不能沿用就座构成——那条分支
                    # 会告诉模型"下装不可见"，而递补候选恰恰是站立全身镜头、
                    # 下装可见（brief 变异清单 (d)，Task 42 报告详述过这个
                    # 缺陷）。这个文案也是 Task 45 增量复用判据要比对的
                    # "构成签名"。
                    #
                    # Task 51：按格位切出来的那种只有一张图，构成文案由
                    # `entryexit._plan_entry` 随计划一起给出（那里才知道
                    # 切自第几格）——沿用四张拼图的文案会让模型去描述并不
                    # 存在的另外三张画面。它同时是增量复用的"构成签名"：
                    # 换一格 = 换一段文案 = 不会误判可复用。
                    is_entry = cand["kind"] == "entry"

                    # 逐格排除 + 并入该座位**其余**的人工标注，两件事都要在
                    # 算构成文案之前做完：文案如实描述本次送了几张，它同时是
                    # Task 45 增量复用的「构成签名」。算在后面的话，上一轮那条
                    # 结果会因签名相同被误判可复用，而这一轮的图其实变了。
                    # **逐格排除在这一路一概不查**（Task 56W，与 registry
                    # 逐字同一条）：那排「排除格 N」复选框只长在**就座**证据
                    # 图上，说的是那张 2×2 的第几格。拿同一个 `(段, 座位)` 键
                    # 到这里再查一遍，剔掉的是另一张图上的东西——用户 2026-08-29
                    # 实测城市2 order 6 两格变一格，且一处不报错。
                    _ex = set()
                    _kept_pairs = list(trans_pairs)
                    # 用户 2026-08-26 实测文化4组 order 3：「单帧可能只选了一
                    # 个窗格，而在多帧分别选的窗格没有拼在一起，应该根据标注
                    # 尽可能全的拼成 2×2 四张图」。此前补帧只长在就座那一路
                    # 上，由人工标定的进出场候选定下来的座位拿到一格就收工。
                    #
                    # **排掉落位用的这一条**（按 stable_id）：它刚被逐格排除
                    # 筛过，再从它自己补一遍等于把用户勾掉的那一格搬回来。
                    _others = [sp for sp in (topup_by_seat.get(int(seat)) or [])
                               if (sp[0] or {}).get("stable_id")
                               != cand["stable_id"]]
                    _merged = topup_pairs(_kept_pairs, _others, want=4)
                    # 「并进了别条候选的画面没有」按**成员**判断，不按长度差：
                    # 多个来源轮流出格之后，本条可能让出格子，合成的张数不一定
                    # 变多（用户 2026-09-26）。按长度差会在「4 格本条 → 2 本条 +
                    # 2 别条」时判成没并，走「按编号第 N 格切出来、本次没有其它
                    # 可用画面」那支——既说错来源，还可能在格位数对不上时抛错。
                    _own = [d for _c, d in _kept_pairs]
                    _n_extra = sum(1 for _c, d in _merged if d not in _own)
                    _kept = [c for c, _d in _merged]
                    _td = [d for _c, d in _merged]

                    if _n_extra:
                        # 并了别条候选的画面之后，「第 N 格」这个说法就不成立
                        # 了——格位号是**各自那张**证据图上的编号，两条候选的
                        # 第 1 格是两个不同的画面。改用纯进出场那一支，如实说
                        # 共几张、进场几张退场几张。
                        _ne = sum(1 for d in _td if d.kind == "entry")
                        layout = describe_layout(0, n_entry=_ne,
                                                 n_exit=len(_td) - _ne)
                    elif item_cells and _kept and len(_kept) != len(item_cells):
                        # 剔掉几格，构成文案就得跟着改口——否则模型被告知有 N
                        # 张、却只收到 N-1 张，会去找那张已经被人工剔除的图
                        # （与 Task 42「告诉模型下装不可见、而站立镜头里明明
                        # 可见」同类，且一处也不会报错）。
                        layout = describe_layout(
                            0,
                            n_entry=len(_kept) if is_entry else 0,
                            n_exit=0 if is_entry else len(_kept),
                            cell_indices=_kept,
                        )
                    else:
                        layout = item.get("layout") or describe_layout(
                            0, n_entry=1 if is_entry else 0,
                            n_exit=0 if is_entry else 1,
                        )
                    person_id = f"seg{seg.order:02d}_seat{seat}"

                    # Task 45：递补候选按 person_id（即目标座位号）查上一轮
                    # 记录——座位号在这里已经确定，不必像第 1 步那样绕开
                    # "座位号要等 VLM 结果才能定"的鸡生蛋问题。额外核对
                    # track_id 一致：防止本轮 entry_labels 给同一座位换了
                    # 一个不同的候选，却因 person_id/layout 恰好相同而被
                    # 误判可复用——那样会把张三的衣着安在李四的记录名下。
                    prior_row = prior_entryexit_by_person.get(person_id)
                    prior_track_ok = (
                        prior_row is not None
                        and (prior_row.get("pixel") or {}).get("track_id")
                            == cand["track_id"]
                    )
                    if prior_track_ok and _can_reuse_prior_annotation(
                        prior_row, layout, force_vlm,
                    ):
                        n_reused[0] += 1
                        vlm = dict(prior_row.get("vlm") or {})
                        _merge_hue_after_vlm_call(vlm, pixel)
                        pixel = _finalize_granularity(pixel)
                    else:
                        n_vlm_calls[0] += 1
                        if _ex and not _td:
                            pixel["_evidence_png"] = None
                            pixel["cells_all_excluded"] = 1
                        else:
                            pixel["_evidence_png"] = build_evidence_sheet(
                                [d.frame_path for d in _td],
                                [(d.x1, d.y1, d.x2, d.y2) for d in _td],
                            )
                        sheet = pixel.get("_evidence_png")

                        if use_vlm:
                            vlm_attempts += 1
                        log_id = f"seg{seg.order:02d}_track{cand['track_id']}_entryexit"
                        vlm, pixel = _vlm_and_hue_pass(
                            sheet, pixel,
                            {
                                "company": seg.company,
                                "presenter_num": seg.presenter_num,
                                "layout": layout,
                                "person_id": log_id,
                            },
                            log_id, use_vlm, vlm_failures,
                        )

                    p = (seg.presenters[seat - 1]
                         if seg.presenters and seat <= len(seg.presenters) else None)
                    a = PersonAnnotation(
                        person_id=person_id, seg_order=seg.order,
                        company=seg.company, seat_index=seat,
                        presenter_num=seg.presenter_num,
                        gender_female=p.gender_female if p else 0,
                        speaks_pre=p.speaks_pre if p else 0,
                        speaks_qa=p.speaks_qa if p else 0,
                        seg_start=seg.start, seg_end=seg.end,
                    )
                    a.pixel = pixel
                    a.vlm = vlm
                    a.best_frame_seated = ""
                    a.best_frame_entry = str(trans_dets[0].frame_path)
                    a.suspect_staff = int(a.person_id in (staff_ids or set()))
                    # 递补三件套（Task 44 brief 变异清单 (b)(c)）：
                    a.source = "entryexit"
                    a.seat_uncertain = int(item.get("seat_uncertain", 0))
                    # Task 53 需求六：跨段流转的那几帧来自 src_order 那一段
                    a.cross_segment = int(item.get("cross_segment", 0) or 0)
                    a.src_seg_order = int(item.get("src_order") or 0)
                    a.expert_removed = seg_expert_removed
                    a.layout = layout   # Task 45：供下一轮增量复用判据比对

                    # 就地替换该座位原来的缺座位占位行，不新增、不重复
                    # （brief 变异清单 (f) 的另一半：位置替换而非追加，
                    # 保证每个座位号在 out 里最多出现一次）。
                    for i, row in enumerate(out):
                        if row.seg_order == seg.order and row.seat_index == seat:
                            out[i] = a
                            backfill_count[0] += 1
                            break

    # Task 45：凡有复用必须可见——不能让用户以为本轮 34 人全是新测的，
    # 其实 29 人原样冻结自上一轮。VLM_INCREMENTAL=False 时改打印"全量
    # 重跑"，不套用"复用 N/调用 M"这套措辞（哪怕 N 恰好等于 0，那句仍然
    # 意味着"本来会尝试复用、只是没命中"，与"整个机制都关掉了"是两回事）。
    if config.VLM_INCREMENTAL:
        print(f"   复用 {n_reused[0]} 人的既有标注，本轮实际调用 VLM {n_vlm_calls[0]} 人")
    else:
        print("   全量重跑：VLM_INCREMENTAL=False，本轮对全部候选人重新调用 VLM")

    if backfill_count[0]:
        print(f"   ⇄ 进出场递补 {backfill_count[0]} 人（衣着证据取自进出场画面，非就座画面）")

    # Task 45 追加改动四 · 缺陷二：座位号未指定的候选不参与递补（默认
    # config.BACKFILL_REQUIRE_SEAT=True），但不能静默丢弃——用户需要知道
    # 有几个候选卡在这一步，回挑选页补齐座位号才能让他们真正被用上。
    if needs_seat_total:
        print(f"   ⚠ {len(needs_seat_total)} 个已标为创业者但未指定座位号的候选"
              f"未参与递补，请在挑选页补齐座位号")

    norms = normalize_sharpness([float(a.pixel.get("sharpness_raw", 0.0)) for a in out])
    for a, n in zip(out, norms):
        a.pixel["sharpness_norm"] = round(n, 4)

    if vlm_failures:
        # 失败必须可见（既定原则），但不连坐：批已经跑完，这里只是把
        # 失败的人数汇总打印出来，具体是谁看 result.csv 的 vlm_error 列。
        print(f"   ⚠ VLM 调用失败 {len(vlm_failures)}/{vlm_attempts} 人，"
              f"详见 result.csv 的 vlm_error 列")

    if gender_report_rows:
        # Task 33 起的性别匹配标定报告，Task 34 挪到这里打印：不再需要
        # "重跑一次"才有 gender_judged 可看，本次调用内就能给出。
        from pipeline.seatmatch import gender_match_report
        for row in gender_match_report(gender_report_rows):
            conflict = "冲突" if row["gender_conflict"] else "正常"
            rearranged = "是" if row["rearranged"] else "否"
            print(f"   性别匹配 order {row['seg_order']:>2}：标注={row['annotated']} "
                  f"VLM={row['vlm']} 重排={rearranged} {conflict}")

    return out


def person_keyframe_dir(root: Path, seg_order: int, company: str,
                        seat_index: int) -> Path:
    """一个人的关键帧目录。写入端与读取端共用，避免两处各拼一次路径。

    画廊在内存里拿不到证据图时要从这里读回（Task 37），若两边各写一份
    拼接逻辑，改了净化规则就会一边能读、一边读不到，而表现只是"图没显示"
    这种看不出根因的症状。
    """
    safe = config.sanitize_filename_component(company)
    return root / f"{seg_order:02d}_{safe}" / f"seat{seat_index}"


def export_keyframes(
    annotations: list[PersonAnnotation],
    dets: list[Detection],
    out_root: Path,
    per_kind: int = 3,
) -> None:
    """按「企业 / 座位」导出全分辨率关键帧，供人工核验。

    原始帧目录是按窗口平铺的，人翻不动；这里只挑每人最好的几张，
    并额外画一张带检测框的整帧，让核验者能确认「这一行对应的确实是这个人」。

    写盘后从 pixel 中删除 `_evidence_png`（若存在）——它是暂存的二进制证据图，
    绝不能让它进 CSV（annotation_to_row 会摊平 pixel 的全部键）。
    """
    import cv2

    for a in annotations:
        # 净化规则集中在 config.sanitize_filename_component（Task 33），本处
        # 与 run_all.py 阶段 8/9 的产物文件名、画廊导出的人工校对版文件名
        # 共用同一份实现，不再各自另写一份字符集可能悄悄走样的净化逻辑。
        person_dir = person_keyframe_dir(out_root, a.seg_order, a.company,
                                         a.seat_index)
        person_dir.mkdir(parents=True, exist_ok=True)

        mine = [d for d in dets
                if d.seg_order == a.seg_order and str(d.frame_path) and
                d.track_id == a.pixel.get("track_id")]

        for kind in ("seated", "entry", "exit"):
            picks = select_best_frames([d for d in mine if d.kind == kind], n=per_kind)
            for i, d in enumerate(picks, 1):
                crop = _crop(d)
                if crop is None:
                    continue
                Image.fromarray(crop).save(
                    person_dir / f"crop_{kind}_{i:02d}_t{d.t:.1f}.jpg", quality=95
                )

        # 带框整帧：核验者据此确认人物身份与座位位置
        ctx = select_best_frames([d for d in mine if d.kind == "seated"], n=1)
        if ctx:
            d = ctx[0]
            img = np.asarray(Image.open(d.frame_path).convert("RGB")).copy()
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                          (255, 0, 0), 3)
            cv2.putText(img, f"seat{a.seat_index}", (int(d.x1), max(24, int(d.y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            Image.fromarray(img).save(person_dir / "context_full_frame.jpg", quality=92)

        # 交给 VLM 的那张 2×2 证据图也存一份：
        # 用户若不认同某个标注，需要看到模型当时看到的究竟是什么
        sheet = a.pixel.pop("_evidence_png", None)
        if sheet:
            (person_dir / "evidence_sent_to_model.png").write_bytes(sheet)


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _annotations_to_long_rows(annotations: list[PersonAnnotation]) -> list[dict]:
    """每人一行的长表展平（result.csv 与 result_wide.csv 共用同一份数据源）。

    硬性约束：result.csv 的列与行为不许改变。这里只是把 write_result_csv
    原本内联的展平逻辑抽出来复用，输出与此前逐字节相同。
    """
    from pipeline.gallery import annotation_to_row
    rows = []
    for a in annotations:
        r = annotation_to_row(a)
        r.pop("_thumb", None)
        r.pop("_evidence_png", None)
        rows.append(r)
    return rows


def write_result_csv(annotations: list[PersonAnnotation], path: Path) -> None:
    _write_csv(_annotations_to_long_rows(annotations), path)


def write_ranking_csv(rows: list[dict], path: Path) -> None:
    _write_csv(rows, path)


# ---------------------------------------------------------------------------
# Task 26 · 需求二 —— 一企业一行的宽表（用户原话：「每个企业为一行，依次
# 输出座位1、座位2…的相关值到不同列，如果相应座位没有，取缺失值即可」）。
#
# 新增产物 result_wide.csv，不改 result.csv 长表一个字节。座位数上限取 4：
# 用户的座位图最多 4 人，实测最多 3 人，第 4 组列会整列为空——保留，因为
# 用户说过人数分布 1–4 人。
#
# Task 35 · 改动四：宽表原有 9 个 team_* 团队层面聚合列（team_n_measured/
# team_formality_mean/lead/sd、team_hue_lead/modal/any_light/all_dark、
# team_confidence_mean）已按用户 2026-08-16 要求整体删除——用户全部不要，
# 不再输出、不再计算。`seatN_status` 三态列（已识别/未识别/无此座位）
# 用户明确要保留，不受这次删除影响。
# ---------------------------------------------------------------------------

MAX_SEATS_WIDE = 4

# 每个座位展开的字段，原样摘自长表对应列，不重算任何度量。
# 两处刻意的删减（用户 2026-08-16 定版）：
#
# 1. 不含 gender_female / speaks_pre / speaks_qa——它们是用户自己 dta 里
#    prst{N}Female / preprst{N} / qaprst{N} 的原样搬运，宽表里纯属冗余。
#    长表 result.csv 仍保留（每行自足，便于单独分发核对）。
# 2. 不含像素轨专属的连续色彩量。三组视频实测双轨一致率仅 33%/41%/43%，
#    新能源2组 30 人全判黑、L* 中位 1.3 已在传感器噪底。像素轨代码仍在，
#    config.PIXEL_COLOR_ENABLED 可重新开启，但宽表按用户要求固定为这一套
#    列、不随开关回退——宽表是给人看的最终产物，列集合稳定比可回溯更重要。
#    真要看像素轨的值，长表与 measure_person_color 都还在。
# Task 35 · 改动一：view_conflict 已从 WIDE_SEAT_FIELDS 移除——实测三组
# 103 行该字段 56 行为 2（无法比较）、47 行为空，无一行为 0 或 1，零信息量
# （PROMPT 已恒定要求模型记 2，因为证据图早已只含就座镜头、没有第二视角
# 可比，见 pipeline/annotate.py 里删掉的那段三态说明）。
# Task 35 · 改动二：n_frames_color 已从 WIDE_SEAT_FIELDS 移除——实测三组
# 103 行全为 0（像素轨关闭后恒为 0，是死列），过滤机制见 PIXEL_ONLY_FIELDS。
WIDE_SEAT_FIELDS = [
    "formality_1_5", "formality_label", "outer_garment_type", "has_tie",
    "hue_class_final", "outer_hue_class",
    "outer_saturation_1_5", "outer_brightness_1_5", "neutral_ref_found",
    "inner_color_class", "evidence_quality", "confidence",
    "sharpness_norm", "vlm_error",
]


def build_wide_rows(long_rows: list[dict]) -> list[dict]:
    """把每人一行的长表转成每企业一行的宽表。

    只做重排与聚合，不重新测量任何值——formality/颜色/置信度等全部原样
    摘自传入的长表行。座位号从长表的 seat_index 取，缺的座位（seat_index
    超过该企业 presenter_num）整组列留空字符串，绝不能填 0（0 是合法的
    有效值，例如 gender_female=0，用 0 占位会把"没测到"伪装成"测到了
    且为 0"）。
    """
    by_order: dict[int, list[dict]] = {}
    for r in long_rows:
        by_order.setdefault(int(r["order"]), []).append(r)

    wide_rows = []
    for order in sorted(by_order):
        seat_rows = by_order[order]
        by_seat_idx = {int(r["seat_index"]): r for r in seat_rows}
        first = seat_rows[0]

        row: dict = {
            "order": order,
            "company": first["company"],
            "presenter_num": first["presenter_num"],
            "seg_start": first["seg_start"],
            "seg_end": first["seg_end"],
        }

        try:
            pnum = int(float(first.get("presenter_num") or 0))
        except (TypeError, ValueError):
            pnum = 0

        for seat in range(1, MAX_SEATS_WIDE + 1):
            sr = by_seat_idx.get(seat)
            for field in WIDE_SEAT_FIELDS:
                row[f"seat{seat}_{field}"] = sr.get(field, "") if sr else ""

            # 空字符串有两种截然不同的含义，必须分开（用户 2026-08-16 指出）：
            # 「标注 3 人只测到 1 人」的 seat2 与「标注 1 人本就没有 seat2」的
            # seat2，此前在表里长得一模一样，无法判断是漏检还是本来没这个人。
            # 判据用用户的人工标注 presenter_num——本项目最可靠的先验。
            if seat > pnum:
                row[f"seat{seat}_status"] = "无此座位"
            elif sr and str(sr.get("formality_1_5", "")).strip():
                row[f"seat{seat}_status"] = "已识别"
            else:
                row[f"seat{seat}_status"] = "未识别"

        # Task 35 · 改动四：原先这里还计算 9 个 team_* 团队层面聚合列
        # （见已删除的 _to_float_or_none 辅助函数与模块顶部旧注释），
        # 用户 2026-08-16 要求整体删除，不再计算、不再输出。

        wide_rows.append(row)
    return wide_rows


def write_wide_csv(annotations: list[PersonAnnotation], path: Path) -> None:
    """需求二：一企业一行的宽表，写到 result_wide.csv（不动 result.csv）。"""
    _write_csv(build_wide_rows(_annotations_to_long_rows(annotations)), path)
