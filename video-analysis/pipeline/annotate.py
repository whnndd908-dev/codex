"""阶段 ⑤：VLM 打标与像素算法双轨合并。"""
from __future__ import annotations

import base64
import os
import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline import config
from pipeline.detect import Detection

# 模型名与端点集中在此，便于中转服务变更时单点调整。
# anthropic SDK 会自动读取环境变量 ANTHROPIC_API_KEY 与 ANTHROPIC_BASE_URL，
# 本项目把它们放在 dress/.env（权限 600，已被 .gitignore 排除）。
VLM_MODEL = os.environ.get("DRESS_VLM_MODEL", "claude-opus-5")

# 正式程度量表：用户定义的口径，唯一定义处——PROMPT、画廊下拉框
# （pipeline/gallery.py 的 formality 选项）、result.csv 的 formality_label 列
# 都从这里派生，不允许另抄一份字面量。本项目已在 validate.py 的规则名上因
# 多处重复定义而漂移过一次（Ruling R39），has_jacket 被用户裁定废弃后
# （无下游依赖：rank.py/validate.py 均不引用），改为让 formality 自解释，
# 同样的漂移风险必须从一开始就堵住。
# Task 43：用户逐字给出的量表修订——语义与旧五档一致（1↔运动、5↔成套），
# 只是把每档的边界例举得更具体，旧数据不作废。但第 3 档新增了「（原 Polo，
# 已改中文，见下）、商务连衣裙无外套」、第 4 档新增了「连衣裙+外套」——这
# 两处是旧文案完全没覆盖到的女装品类，日后核对旧数据时，凡涉及这两档的
# 女性样本都要留意：档位判定可能因这次措辞细化而改变，不能默认旧标注在
# 新口径下依旧成立。
#
# Task 43 追加（用户复核后要求）：提示词一律用中文，不留英文单词。
# - 第 4 档原文用户逐字给的是「单件西装外套/blazer」；但 blazer 本身就是
#   单件西装外套的英文说法，直译成中文「休闲外套」会与第 2 档的「休闲
#   夹克」撞车——模型分不清一件休闲外套该记 2 还是 4，直接污染正式度这个
#   核心变量。改用「西服式外套」这个中文说法消歧，同时去掉英文/中文的
#   同义重复（`test_formality_level_2_and_4_are_not_confusable` 守住不
#   撞车这条）。
# - 第 3 档的「Polo」改成「翻领短袖衫」（中文对 Polo 衫的常见称法），与
#   PROMPT 里 outer_garment_type 枚举、pipeline/gallery.py 下拉框选项三处
#   同步替换，避免 VLM 吐出中文新值而画廊选项里没有对应项（Task 32 修过
#   的「值与选项对不上」缺陷同款）。
# - 「T恤」保留不改：这里的「T」是字形描述（如同「V领」），不是英文单词，
#   `test_no_latin_words_in_formality_labels` / `test_garment_type_
#   options_have_no_latin_words` 均显式豁免这一个字符，不强改成「体恤」。
FORMALITY_LABELS = {
    1: "运动/健身服、背心、短裤、冲锋衣等功能性服装",
    2: "T恤、卫衣、牛仔外套、休闲夹克、印花上衣",
    3: "翻领短袖衫、衬衫（正装或休闲）无外套、针织衫/开衫、商务连衣裙无外套",
    4: "衬衫+西裤，或单件西装外套（西服式外套，与下装不成套）、连衣裙+外套",
    5: "成套西装/成套正式裙装（上下同料同色）",
}


def formality_label(value) -> str:
    """把 1-5 的正式度整数转成可读标签，如 "5 = 成套西装/成套正式裙装（上下同料同色）"。

    value 为 None/空字符串/不在 1-5 范围内（含 0）时返回空字符串——缺座位
    占位行与 VLM 调用失败时 formality_1_5 本身就是空，标签必须跟着空，
    不能替用户编造一个看起来合理的档位。
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return ""
    label = FORMALITY_LABELS.get(n)
    return f"{n} = {label}" if label else ""


_FORMALITY_SCALE_TEXT = "；".join(f"{n}={label}" for n, label in FORMALITY_LABELS.items())

# Task 43：饱和度/明度锚点，用户逐字给出。此前 schema 注释里各有一行简略
# 版本（"// 1=几乎无彩…"），本任务把详细锚点挪进正文、schema 注释只留字段名
# 与取值范围，两处不再并存——避免详略两份描述日后各自漂移、互相矛盾。
_SATURATION_ANCHORS = """饱和度（彩度）锚点——按「衣物固有色」判断，不看它在暖光下的样子：
1 = 几乎无彩：纯黑、纯白、各级灰、近黑藏青、燕麦白（HSV 的 S 约 <0.10）
2 = 很淡：米白、浅灰蓝、雾霾蓝、烟粉、浅卡其、浅驼（S 约 0.10–0.25）
3 = 中等：常规藏青、墨绿、酒红、砖红、中灰蓝、深棕（S 约 0.25–0.45）
4 = 鲜明：正红、宝蓝、明黄、草绿、亮紫、玫红（S 约 0.45–0.70）
5 = 非常饱和：荧光绿、大红、电光蓝、明橙等接近纯色（S 约 >0.70）"""

_BRIGHTNESS_ANCHORS = """明度锚点——以画面中的中性参照物（白纸/瓶标/幕布白/浅色桌牌）为「中性白 = 5」：
1 = 很暗，接近黑，与参照物反差极大
2 = 暗，如深藏青、深棕、炭灰
3 = 中等，如中蓝、中灰、砖红
4 = 亮，如浅蓝、浅灰、米色
5 = 很亮，与参照物明度相近或更亮"""

# Task 43：{layout} 段落的小标题，供 `pipeline.assemble.describe_layout` 与
# PROMPT 正文里「一律以上面『{LAYOUT_HEADING}』为准」那句共用同一个来源，
# 防止两处各写一份、日后有一处改了措辞而另一处没跟着改，导致正文的指代
# 落空（这正是本任务变异清单 (g) 要防的"差一个字"）。
LAYOUT_HEADING = "证据图构成说明"

# Task 29：「已知事实」段落曾经把 gender 也告诉模型，而 schema 里又问
# gender_judged——这是循环论证，模型只是在复读输入，用户提出的「性别对不上
# =识别错人」这条校验器因此完全失效（之前报的「26/27 一致」毫无意义）。
# 修法：从「已知事实」里删掉性别这一行，只保留企业/本组人数——
# gender_judged 仍在输出 schema 里（这是要独立判断的输出字段，不是删除
# 对象），只是不再把答案先喂给模型。调用方（annotate 装配层）目前仍会往
# context 里塞一个 'gender' 键——PROMPT.format(**ctx) 对没有对应占位符的
# 多余键是安全的（Python str.format 忽略未使用的关键字参数），不会因此
# KeyError 或崩溃，见 tests/test_annotate.py 里的对应测试。
#
# Task 34 · 改动一（架构性）：原先「已知事实」还有一行「座位号：{{seat_index}}
# （1 为最靠近镜头者，即主讲人）」。这一行既没必要——红框已经指明了要描述
# 谁，VLM 根本不需要知道座位号——又有害：它暗示「座位1=主讲人」，可能诱导
# 模型把座位号当成身份线索去回答 gender_judged 等本该独立判断的字段（与
# Task 29 删掉性别泄漏是同一个道理）。删掉这一行后，VLM 调用不再依赖任何
# 「座位已分配」的前提，装配层（pipeline/assemble.py）因此能把流程改成
# 单向、无环的「候选人 → VLM → 性别匹配定座位号」，不必再跑两遍
# （旧流程：第一遍出性别、第二遍重排座位再重跑 VLM）。
# `ask_vlm` 的 context 不再需要 `seat_index` 键——多余的键仍然安全（同上
# 的 str.format 行为），调用方（新流程）也确实不再传它。
# Task 42：本项目即将把进出场的站立全身图混入证据图（此前只有就座图）。
# 原 PROMPT 在这里硬编码了四句「证据图固定为就座镜头」类断言——这些断言
# 在 Task 26（只用就座帧）时是对的，但硬编码成常量后，一旦证据图构成
# 随组别变化（纯就座 / 混合 / 纯进出场），就会变成主动误导模型的假话，
# 其中「下装与站姿一律不可见，不要据此判定」这句最危险：站立镜头恰恰
# 能看到下装与站姿，这句话会让模型主动无视最有用的信息。
#
# 修法：凡是依赖「图里有什么」的表述，一律不再写死在 PROMPT 主体里，
# 全部改由 `pipeline.assemble.describe_layout` 按实际构成生成、经
# {layout} 占位符注入；PROMPT 主体只保留与构成无关的通用指令（只描述
# 红框内的人、中性参照物找法、输出 schema 等）。
#
# Task 43（用户逐字给出的修订）在此基础上继续四点改动：
# 1. 「会场灯光偏暖」不再是独立于 {layout} 的孤立一句，而是并入新增的
#    【拍摄条件】小节，并首次写明暖光偏色的**方向**——高估暖色系饱和度、
#    低估冷色系饱和度、压低整体明度——此前只说"偏暖"，没告诉模型该往
#    哪个方向校正，等于白说。新旧两句不并存：旧的独立句已删除，语义被
#    【拍摄条件】开头那句完全覆盖。
# 2. 饱和度/明度锚点从 schema 注释里的两行简写（"// 1=几乎无彩…"）搬进
#    正文，用具体色名/HSV 大致范围做锚点，schema 注释只保留字段名与
#    取值范围，两处不再并存。
# 3. 新增【只编码看得见的部位】《不要因上装推断下装》与【同一人多张图
#    不一致时】两条行为规则；前者点名"不要因为穿正装衬衫就默认下身是
#    西裤"，防止模型用统计先验替代对 {layout} 的实际遵守。
# 4. {layout} 段落现在带一个明确小标题 LAYOUT_HEADING（由
#    `pipeline.assemble.describe_layout` 与 DEFAULT_LAYOUT 共同写入，
#    见下方两处），使【只编码看得见的部位】里"以上面『{LAYOUT_HEADING}』
#    为准"这句有明确指代，而不是含糊地指"上面这段"。
PROMPT = ("""你在为一项创业路演研究做着装编码。图为同一位创业者的若干张证据截图。

{layout}

【拍摄条件】
会场灯光偏暖，画面整体偏暖色；这与镜头是就座还是站立拍摄无关。
暖光会**高估**红/橙/黄系衣物的饱和度，**低估**蓝/绿系衣物的饱和度，并整体压低明度。
你要判断的是衣物在中性日光下的**固有色**，不是它在这张图里呈现的样子。
请先在画面中找一个中性参照物（白色打印纸、矿泉水瓶标签、投影幕白底、浅色桌牌、白墙），
把它视为「中性白」，据此在心里做一次白平衡校正，再判断颜色。
若确实找不到任何中性参照物，把 neutral_ref_found 记 0，并在 notes 写「无参照物」，
两个量表仍照常给出，但按你的经验判断。

已知事实（可信，用于校验你的判断）：
- 企业：{company}
- 本组人数：{presenter_num}

【只描述红框内的人】
画面中可能出现邻座的其他人。每一条记录只描述该编号对应红框内的那个人，不要描述框外的人。
若红框内的人被严重遮挡或看不清，如实写 evidence_quality=poor。

【只编码看得见的部位】
下装类型、外套结构、是否成套、领带等，一律以上面「""" + LAYOUT_HEADING + """」为准——
它已如实告知本组每张图实际拍到和拍不到的部位。
说明中没有明确写「可见」的部位，一律填「不可见」，不要据此推断或臆测。
特别提醒：不要因为一个人上身穿正装衬衫就默认他下身是西裤。

【同一人多张图不一致时】
若同一人在不同图中着装不同（例如脱掉外套），以**最完整、最清晰的那张**为准，
并在 notes 注明「跨图不一致」。

""" + _SATURATION_ANCHORS + "\n\n" + _BRIGHTNESS_ANCHORS + """

请只输出 JSON，不要任何其他文字：
{{
  "formality_1_5": 1-5 的整数,
  "outer_garment_type": "西装外套|夹克|毛衣|衬衫|翻领短袖衫|T恤|连衣裙|其他",
  "has_tie": "有|无|不可见",
  "layer_count": 1 或 2,
  "outer_color_name": "具体中文色名，如 藏青、米白",
  "outer_hue_class": "红|橙|黄|绿|蓝|紫|粉|棕|黑|白|灰",
  "outer_saturation_1_5": 1-5 的整数,  // 饱和度，取值范围与含义见上方"饱和度锚点"
  "outer_brightness_1_5": 1-5 的整数,  // 明度，取值范围与含义见上方"明度锚点"
  "neutral_ref_found": 0 或 1,  // 是否在画面中找到了中性参照物（白纸/瓶标/幕布/桌牌等）
  "inner_visible": 0 或 1,
  "inner_color_class": "白|浅色|深色|蓝色系|花纹|不可见",
  "gender_judged": "男|女|不确定",
  "evidence_quality": "good|fair|poor",
  "confidence": 0.0-1.0,
  "notes": "简短说明，尤其是判不出来的原因、跨图不一致时选用了哪张图、有无中性参照物"
}}

正式程度量表：""" + _FORMALITY_SCALE_TEXT + """。

判不出就如实写 evidence_quality=poor 并在 notes 说明，不要猜。""")

# 评审人版的打标提示词（用户 2026-09-19：「阶段 7 的 VLM 打标接评审人」）。
#
# **由 PROMPT 切出来，不另抄一份。** 两套量表一旦有一点不同，跨组比较就是在比
# 两把尺子，而数字看上去一样合理——这正是本项目「一条规则只准有一个实现」那条
# 规矩要防的事。抄一份的话，日后有人调了创业者那一版的锚点，评审席这一版不会
# 跟着动，也没有任何一处会报错。
#
# 只有两处不同：
#   · 开头那句「图为同一位创业者的…」→「…评审人的…」
#   · 整块【已知事实】（企业、本组人数）删掉——评审人不属于任何一家企业，
#     留着就是让模型拿一家无关公司的信息去校验他的着装。
_JUDGE_HEAD = ("你在为一项创业路演研究做着装编码。图为同一位评审人的若干张"
               "证据截图。\n\n{layout}\n\n")
_COND_AT = "【拍摄条件】"
_FACTS_AT = "已知事实（可信，用于校验你的判断）："
_BOX_AT = "【只描述红框内的人】"


def _judge_prompt() -> str:
    rest = PROMPT[PROMPT.index(_COND_AT):]
    return _JUDGE_HEAD + rest[:rest.index(_FACTS_AT)] + rest[rest.index(_BOX_AT):]


JUDGE_PROMPT = _judge_prompt()


# ask_vlm 的 context 缺 layout 键时的默认值。必须保守：不宣称图中含进出场镜头，
# 否则模型会把两张就座图误当进场图去推断本不可见的下装（Ruling R50 连带缺陷，
# 见 order 10 退场侧 4 候选全被 Task 20 位置闸门拒掉、exit_track=None 的实例）。
#
# Task 42 补充：证据图构成即将从「恒为纯就座」变成「纯就座 / 混合 / 纯进出场」
# 三选一（`pipeline.assemble.describe_layout` 按实际构成生成对应文案）。这
# 让 DEFAULT_LAYOUT 的保守性要求变得更严格——它不但不能宣称"含"进出场镜头
# （原有理由不变），现在也不能宣称"不含"：三种构成里有两种确实含进出场镜头，
# 一句"不含"在那两种情况下会和 R50 当初要防的乐观宣称犯一样的错，只是方向
# 相反。下面这句"不要假设图中含进出场镜头；仅依据实际可见部位判断"本就没有
# 断言"不含"（它只是不替模型假设"含"），语义上已经是两不宣称，这里改用更
# 严格的标准复核过一遍，维持不变。
#
# Task 43：PROMPT 正文新增的"以上面「LAYOUT_HEADING」为准"这句要有明确
# 指代，DEFAULT_LAYOUT 作为 {layout} 缺失时的兜底替身，也必须带上同一个
# 小标题——否则 layout 缺失这种边界情况下，那句引用会指向一段没有标题的
# 文字，读起来仍然含糊。
DEFAULT_LAYOUT = (
    f"【{LAYOUT_HEADING}】"
    "各格镜头构成未知——不要假设图中含进出场镜头；仅依据画面中实际可见的部位判断，"
    "看不到的部位（例如下装）不要臆测"
)


@dataclass
class PersonAnnotation:
    person_id: str
    seg_order: int
    company: str
    seat_index: int
    presenter_num: int
    gender_female: int
    speaks_pre: int
    speaks_qa: int
    seg_start: int
    seg_end: int
    vlm: dict = field(default_factory=dict)
    pixel: dict = field(default_factory=dict)
    best_frame_entry: str = ""
    best_frame_seated: str = ""
    suspect_staff: int = 0
    # Task 44：进出场递补三件套——source 区分这一行的衣着证据来自就座画面
    # 还是进出场画面（"seated"|"entryexit"），seat_uncertain 标记座位号是否
    # 由算法猜出而非用户在复核页明确指定，expert_removed 标记本段是否发生
    # 过 Innermost_expert 剔除（见 pipeline.seats.expert_removed_tracks）。
    # 默认值就是此前唯一存在的行为（全部来自就座画面、座位号确定、未剔除
    # 任何人），`assemble_annotations(entry_labels=None)` 时这三个字段全部
    # 停在默认值，不改变既有结果一个字节。
    source: str = "seated"
    seat_uncertain: int = 0
    # 该座位被人工在挑选页标为「判错了」。区别于「没测到」——这一行是
    # 测到了但复核者否掉了，两者的处理方式不同，下游必须能分辨。
    seat_rejected: int = 0
    # Task 53 需求六：这一行的画面来自**相邻段**的时间窗口——用户在第 N 段
    # 的格位备注里写「下一场创业者2」，指的是第 N+1 段的人（`cross_segment=1`，
    # `src_seg_order` 记下 N）。默认 0/0 就是此前唯一存在的行为。
    # 必须落到 CSV：不记的话，拿 best_frame_* 的时间戳去核对本段的时间范围
    # 必然对不上，而没有任何一处会报错——本项目最坏的失败模式。
    cross_segment: int = 0
    src_seg_order: int = 0
    expert_removed: int = 0
    # Task 45：本行 VLM 调用时实际使用的证据图构成文案（`describe_layout`
    # 的返回值，原样存下，不只存摘要），供下一轮增量打标判断"构成是否
    # 变了"（见 pipeline.assemble 的 `_can_reuse_prior_annotation`）。默认
    # 空字符串——Task 45 之前产出的 annotations.json 里没有这个键，读取
    # 时按"构成未知"处理，保守重跑，不当作与当前构成相同。
    layout: str = ""


def _load_rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def laplacian_sharpness(img_rgb: np.ndarray) -> float:
    """Laplacian 方差，值越大越清晰。用于从数百帧里挑可用的那几张。"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_frames(dets: list[Detection], n: int = 2) -> list[Detection]:
    """按 bbox 面积与清晰度的乘积排序取前 n 张。

    面积大意味着人更近更完整；清晰度高意味着不是运动模糊的那一帧。
    """
    scored = []
    for d in dets:
        area = max(1.0, (d.x2 - d.x1) * (d.y2 - d.y1))
        try:
            img = _load_rgb(d.frame_path)
            crop = img[int(d.y1):int(d.y2), int(d.x1):int(d.x2)]
            sharp = laplacian_sharpness(crop) if crop.size else 0.0
        except Exception:
            sharp = 0.0
        scored.append((area * (1.0 + sharp), d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _s, d in scored[:n]]


# --- Task 49 · 需求一：2×2 拼图的格位编号 --------------------------------
#
# 一个候选的框里可能挤了不止一个人（Task 41 的「人数」下拉标 >1 时转入
# needs_resplit），但此前没有任何办法让用户指明「谁是谁」。给四格编上号，
# 用户就能在备注里写「1 号格是创业者、2 号格是工作人员」。
#
# 编号**默认不画**：VLM 用的证据图不该带多余的数字（会干扰模型判读，且
# 一旦带上，下游不会有任何一处报错），只有 `pipeline/entryreview.py` 渲染
# 的人工挑选页显式传 `label_cells=True`。
_CELL_LABEL_PAD = 8       # 徽标距格子左上角的内边距（px）
_CELL_LABEL_SIZE = 44     # 徽标边长（px），相对 400px 的格子约 11%
_CELL_LABEL_FONT_SIZE = 32


def _cell_origin(i: int, cell: int) -> tuple[int, int]:
    """第 i 格（0 起）左上角的像素坐标。**行优先**：
    0=左上、1=右上、2=左下、3=右下。

    贴图与画编号共用这一个函数，两处各算一遍必然日后走样——而走样的
    表现是「页面上写的格位约定与图上的数字对不上」，用户按错号写备注、
    重新切分时张冠李戴，全程无一处报错。
    """
    return (i % 2) * cell, (i // 2) * cell


def cell_label_convention() -> list[tuple[str, int]]:
    """格位约定：`[("左上", 1), ("右上", 2), ("左下", 3), ("右下", 4)]`。

    **由 `_cell_origin` 实际算出**，不是写死的文案——挑选页把它原样印在
    候选卡片上，绘制顺序一旦改动，页面上的说明跟着改，不会各说各话。
    """
    names = {(0, 0): "左上", (1, 0): "右上", (0, 1): "左下", (1, 1): "右下"}
    # cell=1 时 `_cell_origin` 返回的就是 (列, 行)
    return [(names[_cell_origin(i, 1)], i + 1) for i in range(4)]


def cell_label_convention_text() -> str:
    """格位约定的一行文案，例如「左上=1 右上=2 左下=3 右下=4」。"""
    return " ".join(f"{name}={n}" for name, n in cell_label_convention())


def _cell_label_font():
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=_CELL_LABEL_FONT_SIZE)
    except TypeError:      # Pillow < 10.1 的 load_default 不接受 size
        return ImageFont.load_default()


def _draw_cell_label(draw, ox: int, oy: int, n: int) -> None:
    """在左上角为 (ox, oy) 的那一格画出格位编号 `n`（白底黑字 + 黑边框）。

    白底黑字是为了在任何画面上都读得出来（会场灯光下人物裁剪明暗都有）；
    只占左上角 44×44px、画在 400px 的格子角上，人物主体在格子中央（贴图
    时居中），不会被遮住。
    """
    x0, y0 = ox + _CELL_LABEL_PAD, oy + _CELL_LABEL_PAD
    x1, y1 = x0 + _CELL_LABEL_SIZE, y0 + _CELL_LABEL_SIZE
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    font = _cell_label_font()
    text = str(n)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (x0 + (_CELL_LABEL_SIZE - (right - left)) / 2 - left,
         y0 + (_CELL_LABEL_SIZE - (bottom - top)) / 2 - top),
        text, fill=(0, 0, 0), font=font,
    )


def build_evidence_sheet(paths: list[Path], boxes: list[tuple],
                          margin_ratio: float = 0.25,
                          label_cells: bool = False,
                          cell_labels: list[int] | None = None) -> bytes:
    """把至多 4 张人物裁剪拼成 2×2 PNG，并在每格用红框标出目标人物。

    留边距再画框，是因为座位排密集、相邻两人的检测框必然交叠——实测
    order 2 座位3 的框与座位2 的框重叠，每格都会切进第二个人。不标出
    目标，VLM 只能自己猜描述谁，且下游无任何字段能反映这种歧义。
    不足 4 张时以灰块补齐。

    `label_cells`（Task 49 · 需求一）：在每格左上角画出格位编号
    （1/2/3/4，行优先，见 `_cell_origin`/`cell_label_convention`），供人工
    在备注里按格位号写清「1 号格是创业者、2 号格是工作人员」。
    **默认 False**——VLM 用的证据图不加编号，多余的数字会干扰模型判读；
    只有挑选页展示用的那份传 True。

    `cell_labels`：四格各画哪个号，缺省 `[1,2,3,4]`。评审人的备选池切成第二、
    第三张 2×2 时要画 5-8、9-12：图上还画 1/2/3/4 的话，人读到「格2」指的却是
    池里的第 6 格，他标的号与流水线用的号从此各说各话，而一处不报错。
    """
    cell = 400
    sheet = Image.new("RGB", (cell * 2, cell * 2), (32, 32, 32))
    for i in range(4):
        if i >= len(paths):
            continue
        img = Image.open(paths[i]).convert("RGB")
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        mw, mh = (x2 - x1) * margin_ratio, (y2 - y1) * margin_ratio
        cx1, cy1 = max(0, int(x1 - mw)), max(0, int(y1 - mh))
        cx2, cy2 = min(img.width, int(x2 + mw)), min(img.height, int(y2 + mh))
        crop = img.crop((cx1, cy1, cx2, cy2))
        ImageDraw.Draw(crop).rectangle(
            [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
            outline=(255, 40, 40), width=5,
        )
        crop.thumbnail((cell, cell))
        cell_x, cell_y = _cell_origin(i, cell)
        sheet.paste(crop, (cell_x + (cell - crop.width) // 2,
                           cell_y + (cell - crop.height) // 2))
    if label_cells:
        # 编号画在贴图之后，保证不被人物裁剪盖住；空格（灰块）也照样编号，
        # 格位约定才是完整的四格。
        nums = list(cell_labels) if cell_labels else [i + 1 for i in range(4)]
        draw = ImageDraw.Draw(sheet)
        for i in range(4):
            cell_x, cell_y = _cell_origin(i, cell)
            if i < len(nums):
                _draw_cell_label(draw, cell_x, cell_y, nums[i])
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


class VlmParseError(RuntimeError):
    """VLM 返回的文本无法解析为 JSON。绝不能静默返回空字典——本项目已因
    静默降级踩过三次坑，宁可在此处硬失败让人工介入。"""


def _extract_json_text(text: str) -> str:
    """从 VLM 回复中取出 JSON 主体。

    已观测到三种真实形态（同一份 prompt，三个不同模型）：
      - claude-opus-5     ```json\\n{...}\\n```   —— 带语言标记的围栏
      - claude-sonnet-5   {...}                    —— 裸 JSON，无围栏
      - claude-sonnet-4-5 ```\\n{...}\\n```         —— 无语言标记的围栏

    还需容忍模型在 JSON 前后加了说明文字的情况（未在三个模型上实测到，
    但 prompt 无法在协议层禁止，防御性处理更稳）。
    """
    t = text.strip()

    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", t, re.DOTALL)
    if fence:
        return fence.group(1).strip()

    # 没有围栏：可能是裸 JSON，也可能是「裸 JSON + 前后说明文字」。
    # 取第一个 '{' 到与之配对的最后一个 '}' 之间的内容。
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1].strip()

    return t


def _is_retryable_vlm_error(exc: Exception) -> bool:
    """判断一次 VLM 调用失败是否值得重试（Task 35 · 改动三）。

    可重试：
      - HTTP 5xx / 429（`anthropic.APIStatusError` 及其子类，靠
        `status_code` 判断，不逐个枚举子类名，防止漏掉中转服务返回的
        非标准子类）。
      - 连接错误 / 超时（`anthropic.APIConnectionError`，
        `APITimeoutError` 是它的子类，天然覆盖）。
      - `VlmParseError`——中转服务偶尔返回 HTML 错误页代替 JSON（实测
        `InternalServerError: <!DOCTYPE HTML ...>`），JSON 解析必然失败，
        表面是解析错误，本质仍是服务端故障，应重试而非直接放弃。

    不可重试：4xx 中除 429 外的一切（400/401/403/404/422 等）——图太大、
    格式不对、鉴权失败，重试多少次结果都一样，只会白白拖长运行时间。
    """
    from anthropic import APIConnectionError, APIStatusError

    if isinstance(exc, VlmParseError):
        return True
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None)
        return code == 429 or (code is not None and code >= 500)
    return False


def _call_vlm_once(client, sheet_png: bytes, fmt_context: dict,
                   prompt: str) -> dict:
    """单次 VLM 调用 + 解析，不含重试逻辑。供 `ask_vlm` 的重试循环复用。

    `prompt` 缺省用创业者那一版；评审席传 `JUDGE_PROMPT`（见它的注释）。
    """
    msg = client.messages.create(
        model=VLM_MODEL,
        # 1024 太紧：控制器实测复现（order 16 座位 2 的真实证据图）
        #   max_tokens=1024: stop=end_turn out_tokens=855（仅剩 17% 余量）
        #   max_tokens=3000: stop=end_turn out_tokens=567
        # notes 写长一点就撞上限，截断的 JSON 自然解析失败——这解释了控制器
        # 全量 16 段 27 次调用里 5 次 VlmParseError 的随机性（同一张图复现
        # 时反而成功）。2048 是正常回复（500–900 tokens）的两倍余量；本
        # prompt 要求「只输出 JSON」，设得更大只会在模型跑飞时浪费更多
        # token 与时间，不设更小是因为 855/1024 已证明不够。
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.b64encode(sheet_png).decode(),
                }},
                {"type": "text", "text": prompt.format(**fmt_context)},
            ],
        }],
    )
    # content[0] 不一定是文本块：中转服务对 claude-opus-5 在本 prompt 长度下
    # 会先返回一个 ThinkingBlock（扩展思考），文本块排在其后。实测过
    # （见 task-10-report.md Step 5）：content = [ThinkingBlock, TextBlock]。
    # 因此必须按 type 找文本块，不能假设下标 0。
    text_blocks = [b for b in msg.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        raise VlmParseError(
            "VLM 响应不含文本块，无法解析 JSON。"
            f"content 块类型: {[getattr(b, 'type', type(b).__name__) for b in msg.content]}"
        )
    raw = text_blocks[-1].text.strip()
    candidate = _extract_json_text(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise VlmParseError(
            "VLM 返回内容无法解析为 JSON。\n"
            f"--- 原始回复（前 2000 字符） ---\n{raw[:2000]}\n"
            f"--- 尝试解析的片段（前 2000 字符） ---\n{candidate[:2000]}\n"
            f"--- json.loads 错误 ---\n{e}"
        ) from e


def ask_vlm(sheet_png: bytes, context: dict,
            prompt: str | None = None) -> dict:
    """把 2x2 证据图交给 Claude 视觉打标，失败时按指数退避重试。

    端点与密钥从环境变量读取（ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY），
    本项目通过 dress/.env 提供，支持中转服务。

    Task 35 · 改动三：实测三组 103 人中 4 人失败，全部是服务端临时故障
    （503 No available channel ×3，HTML 错误页 ×1），「失败即放弃」白白丢
    掉这些本可挽回的样本。最多重试 `config.VLM_MAX_RETRIES` 次（即最多
    `VLM_MAX_RETRIES + 1` 次调用），间隔从 `config.VLM_RETRY_BASE_SEC` 秒
    起每次翻倍，只对 `_is_retryable_vlm_error` 判定为可重试的错误生效——
    4xx（除 429）立即失败，不浪费时间重试注定不变的结果。

    退避的 sleep 走 `time.sleep`（模块级引用，供测试 monkeypatch 掉，不
    真的睡）。多次重试后仍失败，抛出的异常消息里带上尝试次数与最后一次
    失败原因，供上层（`pipeline.assemble`）写进 `vlm_error` 列——调用失败
    与没检出人是两种不同的缺失原因，必须能从 CSV 里区分开。
    """
    from anthropic import Anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "未设置 ANTHROPIC_API_KEY。请先 `set -a && . ./.env && set +a`，"
            "或改用 run_all.py 的 --no-vlm 只跑像素轨。"
        )
    client = Anthropic()   # base_url 亦由 ANTHROPIC_BASE_URL 自动读取
    # layout 缺失时用保守默认值兜底：不能 KeyError，也不能默默填一个宣称
    # 有进出场镜头的乐观描述（见 DEFAULT_LAYOUT 的注释与 Ruling R50）。
    # 默认在这里定下来，往下传的永远是真正要用的那一份——让下游再 `or`
    # 一次的话，调用方无从断言「这一次用的是哪一版」，而两版量表混用是不
    # 报错的（见 JUDGE_PROMPT 的注释）。
    prompt = prompt or PROMPT
    fmt_context = {"layout": DEFAULT_LAYOUT, **context}
    # 仅用于重试期间的进度打印，不进 PROMPT——调用方（assemble.py）目前会
    # 塞一个 person_id 键，缺失时退回用企业名标识（同 str.format 对多余键
    # 安全忽略的既定行为，见 Task 29 的注释）。
    log_id = context.get("person_id") or context.get("company") or "?"

    for attempt in range(config.VLM_MAX_RETRIES + 1):
        try:
            return _call_vlm_once(client, sheet_png, fmt_context, prompt)
        except Exception as exc:
            is_last_attempt = attempt == config.VLM_MAX_RETRIES
            if not _is_retryable_vlm_error(exc):
                raise      # 4xx 等不可重试错误：立即失败，原因如实抛出
            if is_last_attempt:
                raise RuntimeError(
                    f"重试 {attempt} 次后仍失败: {type(exc).__name__}: {exc}"
                ) from exc
            wait = config.VLM_RETRY_BASE_SEC * (2 ** attempt)
            print(f"[VLM] {log_id} 第 {attempt + 1} 次重试，因 "
                  f"{type(exc).__name__}: {str(exc)[:120]}")
            time.sleep(wait)
    raise AssertionError("unreachable")   # 循环必然通过 return 或 raise 退出


# 11 类 = 6 个色相箱 + 棕/粉两个组合类 + 黑/白/灰三个无彩色类。
# 从 config 派生而非硬编码，保证与 colormath 的分类空间单一来源。
VALID_HUE_CLASSES = frozenset(
    [name for name, _lo, _hi in config.HUE_BINS] + ["棕", "粉", "黑", "白", "灰"]
)


def merge_tracks(pixel_cls: str, vlm_cls: str) -> tuple[bool, str]:
    """双轨合并。

    不一致时取 VLM 结果：人类视觉具备色彩恒常性，VLM 继承了这一能力，
    在本视频这种投影染色 + 钨丝灯偏色的场景下优于像素算法的粗糙对角校正。

    但 VLM 返回的色名未必在 11 类之内——实测模型会返回 `#18205e` 这类
    十六进制。非法值一律视为分歧并退回像素轨的可复现结果，绝不写入
    hue_class_final（spec 定义该字段为"人工裁决后的最终值"）。
    分歧会在画廊中标红，由人工裁决。
    """
    if vlm_cls not in VALID_HUE_CLASSES:
        return (False, pixel_cls)
    if pixel_cls == vlm_cls:
        return (True, pixel_cls)
    return (False, vlm_cls)


def face_belongs_to_box(face_box: tuple, person_box: tuple,
                         top_fraction: float | None = None) -> bool:
    """人脸框是否属于这个人（而非画进来的邻座）。

    Task 42：控制器此前测「进出场朝向线索能否用人脸判定」时，裁剪用了 10%
    外扩（build_evidence_sheet 的 margin_ratio 同款做法），结果邻座的脸也
    被算了进去。实测人脸在站立尺度有 93–130px，远高于当初唇动检测失败时
    的 28×42px——脸本身够大、技术可用，但那次测量因外扩混入邻座的脸，结论
    不可信。用户明确要求：去掉外扩，或只认位于框内上部三分之一的脸。本函数
    实现后一种（更宽松、不需要重新裁剪证据图）：判据是人脸中心必须落在
    person_box 内，且纵向位于该框上部 `config.FACE_TOP_FRACTION`（默认
    1/3）以内——邻座的脸即便侥幸落在横向范围内，人脸在合影里的位置通常
    不会挤在别人框的最上沿，这道纵向门槛能把它们筛掉。

    本函数只做纯几何判定，不引入任何人脸检测模型依赖，供将来的朝向检测
    任务复用；本任务不接入真实人脸模型，也不改动任何检测流程。

    face_box / person_box 均为 (x1, y1, x2, y2)。top_fraction 缺省时读
    `config.FACE_TOP_FRACTION`（每次调用时读取，不在函数定义时固化，改
    config 值后行为立即跟着变）。
    """
    if top_fraction is None:
        top_fraction = config.FACE_TOP_FRACTION
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box
    fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0

    if not (px1 <= fcx <= px2 and py1 <= fcy <= py2):
        return False

    top_band_bottom = py1 + top_fraction * (py2 - py1)
    return fcy <= top_band_bottom
