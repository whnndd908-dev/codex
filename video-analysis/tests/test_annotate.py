import io
from pathlib import Path
import numpy as np
import pytest
from pipeline.detect import Detection
from pipeline.annotate import (
    laplacian_sharpness, select_best_frames, merge_tracks, build_evidence_sheet,
    PROMPT, ask_vlm, DEFAULT_LAYOUT, face_belongs_to_box,
)


def test_sharpness_higher_for_detailed_image():
    blurry = np.full((100, 100, 3), 128, dtype=np.uint8)
    sharp = np.zeros((100, 100, 3), dtype=np.uint8)
    sharp[::2, :] = 255
    assert laplacian_sharpness(sharp) > laplacian_sharpness(blurry)


def test_select_best_frames_prefers_larger_boxes(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.annotate.laplacian_sharpness", lambda img: 1.0)
    monkeypatch.setattr("pipeline.annotate._load_rgb", lambda p: np.zeros((10, 10, 3), np.uint8))
    dets = [
        Detection(2, "seated", 0.0, Path("a.jpg"), 1, 0, 0, 50, 100, 0.9),
        Detection(2, "seated", 1.0, Path("b.jpg"), 1, 0, 0, 100, 400, 0.9),
        Detection(2, "seated", 2.0, Path("c.jpg"), 1, 0, 0, 60, 200, 0.9),
    ]
    best = select_best_frames(dets, n=2)
    assert [d.frame_path.name for d in best] == ["b.jpg", "c.jpg"]


def test_merge_tracks_agreement():
    agree, val = merge_tracks("蓝", "蓝")
    assert agree is True and val == "蓝"


def test_merge_tracks_disagreement_prefers_vlm():
    # VLM 具备色彩恒常性，偏色场景下优于像素算法
    agree, val = merge_tracks("紫", "蓝")
    assert agree is False and val == "蓝"


def test_build_evidence_sheet_returns_png(tmp_path):
    from PIL import Image
    paths = []
    for i in range(4):
        p = tmp_path / f"{i}.jpg"
        Image.fromarray(np.full((200, 200, 3), 40 * i, dtype=np.uint8)).save(p)
        paths.append(p)
    boxes = [(10, 10, 150, 190)] * 4
    png = build_evidence_sheet(paths, boxes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_evidence_sheet_pads_when_fewer_than_four(tmp_path):
    from PIL import Image
    p = tmp_path / "only.jpg"
    Image.fromarray(np.full((200, 200, 3), 90, dtype=np.uint8)).save(p)
    png = build_evidence_sheet([p], [(0, 0, 200, 200)])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_merge_tracks_rejects_illegal_vlm_class():
    # 实测 claude-opus-5 会返回 #18205e 这类十六进制而非中文色名。
    # 非法值必须视为分歧并退回像素轨，不得写入 hue_class_final。
    agree, val = merge_tracks("蓝", "#18205e")
    assert agree is False
    assert val == "蓝", "非法 VLM 值应退回像素轨结果"


def test_merge_tracks_accepts_all_eleven_legal_classes():
    for c in ("红", "橙", "黄", "绿", "蓝", "紫", "粉", "棕", "黑", "白", "灰"):
        agree, val = merge_tracks(c, c)
        assert agree is True and val == c


def test_evidence_sheet_marks_target_with_visible_box(tmp_path):
    # 红框必须真的画上去：检查目标框边缘附近存在红色像素
    from PIL import Image
    p = tmp_path / "f.jpg"
    Image.fromarray(np.full((400, 400, 3), 90, dtype=np.uint8)).save(p)
    png = build_evidence_sheet([p], [(100, 100, 300, 300)])
    arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    red = (arr[:, :, 0] > 180) & (arr[:, :, 1] < 90) & (arr[:, :, 2] < 90)
    assert red.sum() > 50, "证据图上应有可见的红色目标框"


# ---------------------------------------------------------------------------
# Task 21 · Ruling R50 —— 就座画面为着装准绳；组图 layout 须如实描述而非写死。
# ---------------------------------------------------------------------------

class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeMessages:
    def __init__(self, reply_json, capture):
        self._reply_json = reply_json
        self._capture = capture

    def create(self, **kwargs):
        self._capture["kwargs"] = kwargs
        # text 块固定是 content 列表里第二项（图片之后）
        text = kwargs["messages"][0]["content"][1]["text"]
        self._capture["text"] = text

        class _Resp:
            content = [_FakeTextBlock(self._reply_json)]

        return _Resp()


class _FakeAnthropic:
    def __init__(self, reply_json, capture):
        self.messages = _FakeMessages(reply_json, capture)


def _patch_anthropic(monkeypatch, reply_json):
    """拦住真实 API 调用，返回 (capture dict) 供断言实际发出的文本/收到的解析结果。"""
    capture = {}
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: _FakeAnthropic(reply_json, capture))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    return capture


def _basic_context(**extra):
    ctx = {"company": "测试公司", "seat_index": 1, "gender": "男", "presenter_num": 1}
    ctx.update(extra)
    return ctx


# ---------------------------------------------------------------------------
# Task 42 续：证据图即将混入进出场站立镜头。上面这条测试
# （test_prompt_states_seated_view_is_authoritative，原文断言 PROMPT 里必须
# 出现"就座"+"准绳/为准"）锁的正是本次要移除的硬编码断言——PROMPT 曾在这里
# 写死「outer_garment_type、formality_1_5 一律以就座画面为准绳判定」，这句
# 话一旦证据图混入站立镜头就是假话。改法是把"构成相关"的判断依据全部移入
# {layout}（pipeline.assemble.describe_layout 按实际构成生成），PROMPT 主体
# 只保留通用指令；因此原测试已随内容一起改写为下面两条，锁住新的等价不变量：
# 硬编码断言确实消失了，且确实转移到了 {layout} 机制上（不是干脆不说了）。
# ---------------------------------------------------------------------------

def test_prompt_body_has_no_hardcoded_composition_claims():
    """PROMPT 静态模板（{layout} 代入前）不得再硬编码依赖"证据图构成"的
    断言——纯就座、混合、纯进出场三种构成下，这些断言不会同时为真，写死
    在这里就必然在其余构成下变成主动误导模型的假话。"""
    assert "不含任何进出场画面" not in PROMPT
    assert "一律以就座画面为准绳判定" not in PROMPT
    assert "下装与整体站姿一律不可见" not in PROMPT
    assert "摄像机位于该人左后侧方" not in PROMPT
    # 外套相关的 schema 说明不是本次移除对象，必须仍然保留
    assert "外套" in PROMPT


def test_prompt_delegates_composition_judgement_to_layout():
    """确认硬编码断言确实转移到了 {layout} 占位符机制上，而不是干脆不说了
    （那样模型会瞎猜，比写死更糟）。"""
    assert "{layout}" in PROMPT
    assert "证据图构成说明" in PROMPT or "构成说明" in PROMPT


def test_ask_vlm_context_layout_is_interpolated(monkeypatch):
    capture = _patch_anthropic(monkeypatch, '{"formality_1_5": 3}')
    ctx = _basic_context(layout="4 张全部为就座镜头")
    ask_vlm(b"fake-png-bytes", ctx)
    sent = capture["text"]
    assert "4 张全部为就座镜头" in sent
    assert "上排为进出场" not in sent


def test_ask_vlm_layout_defaults_to_conservative_text(monkeypatch):
    capture = _patch_anthropic(monkeypatch, '{"formality_1_5": 3}')
    ctx = _basic_context()  # 不给 layout
    ask_vlm(b"fake-png-bytes", ctx)  # 不应抛 KeyError
    sent = capture["text"]
    assert "上排为进出场" not in sent
    # 默认值本身必须是保守的：不能宣称画面里确实有进出场镜头（写死的原句是
    # "上排为进出场全身镜头，下排为就座镜头"，默认值不得含其中任一半句）。
    assert "上排为进出场" not in DEFAULT_LAYOUT
    assert "下排为就座" not in DEFAULT_LAYOUT
    # 但必须明确表态"不要假设/不确定"，而不是默默留白让模型自行脑补。
    assert any(kw in DEFAULT_LAYOUT for kw in ("未知", "不确定", "不要假设"))
    assert DEFAULT_LAYOUT in sent


def test_default_layout_stays_conservative():
    """Task 42：三种证据图构成（纯就座/混合/纯进出场）都可能是 layout 缺失
    时的真相，DEFAULT_LAYOUT 既不能宣称含进出场镜头，也不能宣称不含——只能
    如实说"构成未知，仅依据实际可见部位判断"。变异清单(d)：改成宣称纯就座
    （例如加一句"不含进出场镜头"/"全部为就座镜头"）必须被这条测试抓住。"""
    assert "不含" not in DEFAULT_LAYOUT, "不应宣称不含进出场镜头"
    assert "构成未知" in DEFAULT_LAYOUT
    assert "不要假设" in DEFAULT_LAYOUT and "含进出场镜头" in DEFAULT_LAYOUT


def test_warm_tint_note_survives_all_layouts():
    """「画面偏暖色」描述的是会场灯光，对三种证据图构成同样成立，因此拆成
    独立于 {layout} 的固定句子。变异清单(e)：若被误挪进某个分支的 layout
    文案里、又在另一分支漏掉，这条测试必须能在三种构成下都抓到缺失。"""
    from pipeline.assemble import describe_layout
    layouts = [
        describe_layout(4, 0, 0),   # 纯就座
        describe_layout(2, 1, 1),   # 混合
        describe_layout(0, 2, 2),   # 纯进出场
    ]
    for layout_text in layouts:
        formatted = PROMPT.format(layout=layout_text, company="测试公司", presenter_num=1)
        assert "偏暖色" in formatted, f"偏暖色提示在此 layout 下丢失：{layout_text!r}"


# ---------------------------------------------------------------------------
# Task 42 · 要求二 —— 人脸归属判定，去掉 10% 外扩，改用「中心须在框内 +
# 位于框上部三分之一」的几何判据（供未来的进出场朝向检测复用）。
# ---------------------------------------------------------------------------

# 统一用同一个 person_box：宽 100（x 100~200）、高 200（y 100~300）。
# top_fraction 默认 0.34 时，上部三分之一带的下界 = 100 + 0.34*200 = 168。
_PERSON_BOX = (100.0, 100.0, 200.0, 300.0)


def test_face_center_outside_box_is_rejected():
    """脸中心 x 落在 person_box 外（邻座的脸），即便 y 落在上三分之一带内
    也必须拒绝——y 坐标特意选在带内，单独隔离出"中心须在框内"这一判据本身
    （变异清单(g)：若实现忽略了这条，仅按 y 判断，本用例会误判为通过）。"""
    face_box = (210.0, 110.0, 230.0, 130.0)   # 中心 (220, 120)：x 在框外，y 在上三分之一带内
    assert face_belongs_to_box(face_box, _PERSON_BOX) is False


def test_face_in_lower_two_thirds_is_rejected():
    """脸中心落在框下部三分之二内（多半是画进来的邻座），x 特意选在框内，
    单独隔离出"上部三分之一"这一判据本身（变异清单(f)：若实现忽略了这条
    约束、只检查中心是否在框内，本用例会误判为通过）。"""
    face_box = (140.0, 240.0, 160.0, 260.0)   # 中心 (150, 250)：x 在框内，y 在下三分之二
    assert face_belongs_to_box(face_box, _PERSON_BOX) is False


def test_face_in_top_third_is_accepted():
    face_box = (140.0, 110.0, 160.0, 130.0)   # 中心 (150, 120)：x 在框内，y 在上三分之一带内
    assert face_belongs_to_box(face_box, _PERSON_BOX) is True


def test_face_fraction_comes_from_config(monkeypatch):
    """FACE_TOP_FRACTION 必须每次调用时从 config 读取，不能在函数定义时
    固化——同一张脸，改小 config 值应被拒绝、改大应被接受（变异清单(h)）。"""
    from pipeline import config
    face_box = (140.0, 190.0, 160.0, 210.0)   # 中心 (150, 200)
    # 默认 0.34：带下界 168，中心 y=200 > 168，应拒绝。
    assert face_belongs_to_box(face_box, _PERSON_BOX) is False
    # 调大到 0.6：带下界 100+0.6*200=220 >= 200，应接受。
    monkeypatch.setattr(config, "FACE_TOP_FRACTION", 0.6)
    assert face_belongs_to_box(face_box, _PERSON_BOX) is True


# ---------------------------------------------------------------------------
# Task 43 · 用户逐字给出的 PROMPT 修订——续 Task 42 的 {layout} 结构。
#   一、FORMALITY_LABELS 换成更具体的五档文案（单一真源，PROMPT/画廊/CSV
#      三处派生）。
#   二、新增饱和度/明度锚点常量，从 schema 注释挪进正文。
#   三、四段新行为规则逐字整合：拍摄条件（含暖光偏差方向）、只描述红框
#      内的人、只编码看得见的部位（不要因上装推断下装）、跨图不一致处理。
#   四、describe_layout 的返回值加小标题，供正文那句"以上面『X』为准"
#      有明确指代。
# ---------------------------------------------------------------------------

def test_formality_labels_match_user_spec():
    """五档文案须与用户逐字给出的版本完全一致（变异清单(a)）——除第 3、4
    档两处经用户复核后明确要求的中文化改写：「Polo」→「翻领短袖衫」，
    「单件西装外套/blazer」→「单件西装外套（西服式外套）」（去英文单词，
    且避免与第 2 档「休闲夹克」撞车，理由见 FORMALITY_LABELS 上方注释）。
    其余三档、以及第 3/4 档里未被点名修改的部分，逐字不变。"""
    from pipeline.annotate import FORMALITY_LABELS
    expected = {
        1: "运动/健身服、背心、短裤、冲锋衣等功能性服装",
        2: "T恤、卫衣、牛仔外套、休闲夹克、印花上衣",
        3: "翻领短袖衫、衬衫（正装或休闲）无外套、针织衫/开衫、商务连衣裙无外套",
        4: "衬衫+西裤，或单件西装外套（西服式外套，与下装不成套）、连衣裙+外套",
        5: "成套西装/成套正式裙装（上下同料同色）",
    }
    assert FORMALITY_LABELS == expected


# 用户复核后追加要求：提示词一律用中文，不留英文单词（「T恤」的「T」是
# 字形描述，不算英文单词，显式豁免）。
_LATIN_WORD_RE = __import__("re").compile(r"[A-Za-z]{2,}")


def test_no_latin_words_in_formality_labels():
    """五档文案不含（长度≥2 的）英文单词——"T恤"的单字母"T"显式豁免。"""
    from pipeline.annotate import FORMALITY_LABELS
    for n, label in FORMALITY_LABELS.items():
        assert not _LATIN_WORD_RE.search(label), (
            f"档位 {n} 的文案仍含英文单词：{label!r}"
        )


def test_garment_type_options_have_no_latin_words():
    """PROMPT 里 outer_garment_type 的枚举同样不得含英文单词。"""
    import re
    m = re.search(r'"outer_garment_type":\s*"([^"]+)"', PROMPT)
    assert m, "未在 PROMPT 里找到 outer_garment_type 枚举"
    for opt in m.group(1).split("|"):
        assert not _LATIN_WORD_RE.search(opt), f"选项仍含英文单词：{opt!r}"


def test_prompt_and_gallery_garment_options_match():
    """PROMPT 里 outer_garment_type 枚举与画廊下拉框选项必须逐项一致
    （变异清单(j)：若只改了一处，这条必须抓住——比分别测两边更有效，因为
    分别测两边各自都能通过，只有放在一起比对才能看出"值与选项对不上"）。"""
    import re
    from pipeline import gallery

    m = re.search(r'"outer_garment_type":\s*"([^"]+)"', PROMPT)
    assert m
    prompt_opts = m.group(1).split("|")

    gallery_opts = dict(gallery._EDITABLE_STATIC)["outer_garment_type"]
    assert prompt_opts == gallery_opts


def test_formality_level_2_and_4_are_not_confusable():
    """第 2 档（含"休闲夹克"）与第 4 档（衬衫+西裤或单件西装外套）不得共享
    "休闲X外套/夹克"这类模糊称法——这是"blazer 不能直译成休闲外套"这条取舍
    要守住的不变量（变异清单(i)：若第 4 档写成"休闲外套"，会与第 2 档
    "休闲夹克"意思重叠，模型将无法判断一件休闲外套该记 2 还是 4）。"""
    from pipeline.annotate import FORMALITY_LABELS
    label2, label4 = FORMALITY_LABELS[2], FORMALITY_LABELS[4]
    assert "休闲夹克" in label2, "前提：第 2 档确实含「休闲夹克」"
    assert "休闲外套" not in label4, "第 4 档不应含「休闲外套」，会与第 2 档撞车"
    assert "休闲" not in label4, "第 4 档不应出现任何「休闲」系称法"
    assert label2 != label4


def test_prompt_contains_saturation_anchors():
    """饱和度锚点全文须出现在正文里，且 schema 注释里那行简略说明已删除
    （变异清单(c)：若锚点只留在 schema 注释、正文没有，这条必须抓住）。"""
    from pipeline.annotate import _SATURATION_ANCHORS
    assert _SATURATION_ANCHORS in PROMPT
    assert "1=几乎无彩（灰/黑/白）2=很淡 3=中等 4=鲜明 5=非常饱和" not in PROMPT, (
        "旧的 schema 简略说明不应与正文详细锚点并存"
    )


def test_prompt_contains_brightness_anchors():
    from pipeline.annotate import _BRIGHTNESS_ANCHORS
    assert _BRIGHTNESS_ANCHORS in PROMPT
    assert "1=很暗（近黑）2=暗 3=中等 4=亮 5=很亮（近白）" not in PROMPT, (
        "旧的 schema 简略说明不应与正文详细锚点并存"
    )


def test_prompt_states_warm_light_direction_of_bias():
    """本次修订最实质的信息：暖光高估暖色系、低估冷色系，压低明度——
    此前只说"偏暖"，没告诉模型该往哪个方向校正（变异清单(d)：写反必须
    被抓住，这里用逐字比对而非只查关键词，防止"高估"/"低估"两个词都在
    但配对给错颜色系的情况蒙混过关）。"""
    assert (
        "暖光会**高估**红/橙/黄系衣物的饱和度，**低估**蓝/绿系衣物的饱和度，"
        "并整体压低明度。"
    ) in PROMPT


def test_only_one_warm_light_sentence():
    """Task 42 遗留的独立"偏暖色"句（"会场灯光偏暖，各镜头画面整体偏暖色——
    这一点与镜头是就座还是站立拍摄的无关。"）已被新的【拍摄条件】小节取代，
    两句不得并存（变异清单(e)）。"""
    assert PROMPT.count("偏暖色") == 1, "「偏暖色」应只在【拍摄条件】里出现一次"
    assert "各镜头画面整体偏暖色" not in PROMPT, "旧句的措辞不应回潮"
    assert "这一点与镜头是就座还是站立拍摄的无关" not in PROMPT, "旧句的措辞不应回潮"


def test_layout_has_labelled_heading():
    """三种证据图构成的 layout 文本都要带「【证据图构成说明】」标题
    （变异清单(f)：若某种构成漏加标题，正文那句引用就会指代落空）。"""
    from pipeline.assemble import describe_layout
    for s in (describe_layout(4, 0, 0), describe_layout(2, 1, 1), describe_layout(0, 2, 2)):
        assert s.startswith("【证据图构成说明】"), f"缺少标题：{s!r}"


def test_prompt_references_layout_by_that_heading():
    """PROMPT 正文里"一律以上面「X」为准"的 X，必须与 describe_layout 实际
    写入证据图文本的标题逐字相同——只查关键词抓不住"差一个字"的漂移
    （变异清单(g)：这是本任务最容易做漏的地方，必须逐字比对，不能只判断
    两边是否都含"构成"或"说明"这类关键词）。"""
    import re
    from pipeline.assemble import describe_layout

    layout_text = describe_layout(4, 0, 0)
    m_heading = re.match(r"^【(.+?)】", layout_text)
    assert m_heading, f"describe_layout 输出应以【标题】开头：{layout_text!r}"
    heading = m_heading.group(1)

    m_ref = re.search(r"一律以上面「(.+?)」为准", PROMPT)
    assert m_ref, "PROMPT 正文应有『一律以上面「X」为准』这句引用"
    referenced = m_ref.group(1)

    assert referenced == heading, (
        f"PROMPT 引用的标题 {referenced!r} 与 describe_layout 实际写入的标题 "
        f"{heading!r} 逐字不一致，指代落空"
    )

    # DEFAULT_LAYOUT（layout 缺失时的兜底）也要用同一个标题，否则这条边界
    # 情况下引用同样落空。
    m_default_heading = re.match(r"^【(.+?)】", DEFAULT_LAYOUT)
    assert m_default_heading, f"DEFAULT_LAYOUT 也应以【标题】开头：{DEFAULT_LAYOUT!r}"
    assert m_default_heading.group(1) == heading


def test_prompt_warns_against_inferring_trousers():
    """不要因为上身穿正装衬衫就默认下身是西裤——防止模型用统计先验替代
    对 layout 的实际遵守。"""
    assert "不要因为一个人上身穿正装衬衫就默认他下身是西裤" in PROMPT


def test_prompt_handles_cross_image_inconsistency():
    """同一人多张图着装不一致时，以最完整清晰的那张为准，并在 notes 里
    注明——这是新增行为，schema 的 notes 字段说明也要跟着提到它，不能只在
    正文说了、schema 没有呼应。"""
    assert "若同一人在不同图中着装不同（例如脱掉外套），以**最完整、最清晰的那张**为准，" in PROMPT
    assert "并在 notes 注明「跨图不一致」。" in PROMPT
    notes_field_desc = PROMPT.split('"notes"', 1)[1].split("\n", 1)[0]
    assert "跨图不一致" in notes_field_desc, (
        "schema 里 notes 字段的说明也应提到跨图不一致，不能只在正文提"
    )


def test_no_hardcoded_composition_claims_regression():
    """护栏：Task 42 移除的四处依赖"证据图构成"的硬编码断言，不得在本次
    PROMPT 重写中回潮（变异清单(h)：「不含任何进出场画面」回潮）。"""
    assert "不含任何进出场画面" not in PROMPT
    assert "一律以就座画面为准绳判定" not in PROMPT
    assert "下装与整体站姿一律不可见" not in PROMPT
    assert "摄像机位于该人左后侧方" not in PROMPT


# ---------------------------------------------------------------------------
# Task 35 · 改动一：view_conflict 已整体删除（实测三组 103 行零信息量，
# 见 pipeline/assemble.py WIDE_SEAT_FIELDS 上方注释）。原「schema 声明
# 三态」「PROMPT 解释无法比较记 2」两条测试随字段一起删除——它们测的正是
# 已删除的内容，没有可迁移的旁路不变量（PROMPT 现在压根不提这个字段，
# 用下面 test_prompt_has_no_view_conflict_field 直接守住"不再出现"）。
# ---------------------------------------------------------------------------

def test_prompt_has_no_view_conflict_field():
    """view_conflict 的 schema 行、指引段落必须删干净，不留半句孤立解释。"""
    assert "view_conflict" not in PROMPT


# ---------------------------------------------------------------------------
# Task 23 · 缺陷一 —— max_tokens=1024 太紧，实测 order 16 座位 2 的真实证据图：
#   max_tokens=1024: stop=end_turn out_tokens=855 （仅剩 17% 余量）
#   max_tokens=3000: stop=end_turn out_tokens=567
# 855/1024 的余量太薄，notes 写长一点就把 JSON 尾部截断，parse 必然失败——
# 这正是控制器全量 16 段 27 次调用里 5 次 VlmParseError 的成因（同一张图
# 复现时反而成功，因为回复长度本身有波动）。
# ---------------------------------------------------------------------------

def test_ask_vlm_requests_enough_output_tokens(monkeypatch):
    capture = _patch_anthropic(monkeypatch, '{"formality_1_5": 3}')
    ask_vlm(b"fake-png-bytes", _basic_context())
    sent_max_tokens = capture["kwargs"]["max_tokens"]
    assert sent_max_tokens >= 2048, (
        f"max_tokens 太紧会截断长 notes 导致 JSON 解析失败，实得 {sent_max_tokens}"
    )


# ---------------------------------------------------------------------------
# Task 25 · 缺陷二（改稿）—— 用户看过 has_jacket 的修法后改主意：这个字段
# 干脆不要了（无下游依赖：rank.py/validate.py 都不用它），改为把 formality
# 量表每一档的含义说清楚，三处（PROMPT、画廊下拉框、result.csv）从
# FORMALITY_LABELS 单一来源派生，不允许各抄一份（Ruling R39 的教训）。
# ---------------------------------------------------------------------------

def test_prompt_has_no_has_jacket_field():
    assert "has_jacket" not in PROMPT, "has_jacket 字段已废弃，不应再出现在 PROMPT 里"


def test_formality_labels_defined_once():
    """PROMPT 里的正式度量表文本必须由 FORMALITY_LABELS 派生，不能是另抄
    的一份字面量——每一档的 "n=标签" 都要能在 PROMPT 里找到，且 PROMPT 里
    出现的正式度说明文字不能比 FORMALITY_LABELS 多一个字，也不能少一个。"""
    from pipeline.annotate import FORMALITY_LABELS
    assert len(FORMALITY_LABELS) == 5
    for n, label in FORMALITY_LABELS.items():
        assert f"{n}={label}" in PROMPT, (
            f"档位 {n} 的量表文本未在 PROMPT 中找到，源与 PROMPT 已漂移"
        )


def test_formality_label_function_formats_readable_string():
    from pipeline.annotate import formality_label, FORMALITY_LABELS
    assert formality_label(4) == f"4 = {FORMALITY_LABELS[4]}"
    assert formality_label("3") == f"3 = {FORMALITY_LABELS[3]}"


def test_formality_label_is_empty_for_missing_value():
    from pipeline.annotate import formality_label
    assert formality_label(None) == ""
    assert formality_label("") == ""
    assert formality_label(0) == "", "0 不在 1-5 量表范围内，不应编造标签"


# ---------------------------------------------------------------------------
# Task 29：PROMPT 曾把性别写进「已知事实」，而 schema 又问 gender_judged——
# 这是循环论证，模型只是复读输入，用户提出的「性别对不上=识别错人」这条
# 校验器因此完全失效。修法：从「已知事实」里删掉性别这一行，gender_judged
# 仍留在输出 schema 里（独立判断的输出字段，不是删除对象）。
# ---------------------------------------------------------------------------

def test_prompt_does_not_leak_gender():
    """「已知事实」段落不应再包含性别信息——不能把答案先告诉模型，
    再让模型的 gender_judged 输出去跟这个答案做无意义的自我比对。"""
    assert "{gender}" not in PROMPT, "PROMPT 不应再有 {gender} 占位符"
    facts_block = PROMPT.split("已知事实", 1)[1].split("画面中可能出现", 1)[0]
    assert "性别" not in facts_block, f"已知事实段落仍含性别信息：{facts_block!r}"


def test_gender_judged_still_in_schema():
    """删的是输入端的性别泄漏，不是输出字段——gender_judged 仍须由模型
    自主判断并输出。"""
    assert '"gender_judged"' in PROMPT


def test_ask_vlm_tolerates_extra_gender_key_in_context(monkeypatch):
    """调用方（annotate 装配层）目前仍会往 context 里塞一个 'gender' 键，
    PROMPT.format(**ctx) 对没有对应占位符的多余键必须安全忽略（Python
    str.format 的既有行为），既不能 KeyError 崩溃，也不能因为某种实现方式
    把这个多余键的值悄悄拼回发给模型的「已知事实」段落，导致性别信息又从
    后门泄漏（schema 里合法存在的 "gender_judged": "男|女|不确定" 选项
    文字不受影响，只检查「已知事实」这一段）。"""
    capture = _patch_anthropic(monkeypatch, '{"formality_1_5": 3}')
    ctx = _basic_context(gender="女")  # 显式携带 gender 键，模拟真实调用方
    ask_vlm(b"fake-png-bytes", ctx)    # 不应抛 KeyError 或任何异常
    sent = capture["text"]
    facts_block = sent.split("已知事实", 1)[1].split("画面中可能出现", 1)[0]
    assert "性别" not in facts_block
    assert "女" not in facts_block, "性别信息不应经由多余的 context 键从后门泄漏"


# ---------------------------------------------------------------------------
# Task 35 · 改动三 —— VLM 调用失败重试。实测三组 103 人中 4 人失败，全部是
# 服务端临时故障（InternalServerError: 503 - No available channel ×3，以及
# 中转返回 HTML 错误页导致 JSON 解析失败 ×1）。「失败即放弃」白白丢掉这些
# 本可挽回的样本，加指数退避重试。
# ---------------------------------------------------------------------------

import httpx
from anthropic import InternalServerError, BadRequestError

from pipeline import config


def _http_status_exc(cls, code: int, message: str):
    """构造一个真实的 anthropic.APIStatusError 子类实例（供测试模拟服务端
    报错）。用真实类而非自造的假异常，是为了让 `_is_retryable_vlm_error`
    的 isinstance/status_code 判断路径真被走到，不是靠鸭子类型蒙混过关。
    """
    resp = httpx.Response(status_code=code, request=httpx.Request("POST", "http://x"))
    return cls(message, response=resp, body=None)


class _SequenceMessages:
    """依次回放一串「行为」：字符串代表成功返回的原始回复文本，
    异常实例代表本次调用应抛出该异常。"""

    def __init__(self, behaviors, capture):
        self._behaviors = list(behaviors)
        self._capture = capture
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        self._capture.setdefault("calls", []).append(kwargs)
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, BaseException):
            raise behavior

        class _Resp:
            content = [_FakeTextBlock(behavior)]

        return _Resp()


class _SequenceAnthropic:
    def __init__(self, behaviors, capture):
        self.messages = _SequenceMessages(behaviors, capture)


def _patch_anthropic_sequence(monkeypatch, behaviors):
    """monkeypatch anthropic.Anthropic，让它按 behaviors 依次成功/报错。

    返回 (capture, client_holder)：capture 记录每次 create() 收到的
    kwargs；client_holder["client"] 是实际用到的那个 fake client 实例，
    供测试断言 call_count（调用了几次）。
    """
    capture: dict = {}
    client_holder: dict = {}

    def factory(**kw):
        client = _SequenceAnthropic(behaviors, capture)
        client_holder["client"] = client
        return client

    monkeypatch.setattr("anthropic.Anthropic", factory)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    return capture, client_holder


def _patch_sleep(monkeypatch):
    """monkeypatch pipeline.annotate.time.sleep，记录退避间隔而不真的睡。"""
    calls: list[float] = []
    monkeypatch.setattr("pipeline.annotate.time.sleep", lambda s: calls.append(s))
    return calls


def test_vlm_retry_config_defaults():
    """护栏：默认配置真的是「最多重试 3 次、2 秒起翻倍」，不是 0（=不重试）
    或某个被悄悄改小的值。补于变异测试(e)：`test_gives_up_after_max_
    retries_and_records_attempts` 用 config.VLM_MAX_RETRIES 动态生成行为
    列表，即便重试次数被改成 0 也会"自洽地"通过——需要这条硬编码断言直接
    锁住默认值本身。"""
    assert config.VLM_MAX_RETRIES == 3
    assert config.VLM_RETRY_BASE_SEC == 2.0


def test_retries_on_5xx_then_succeeds(monkeypatch, capsys):
    """5. 前两次 503、第三次成功：最终返回正常结果，且结果里没有 vlm_error
    （ask_vlm 本身不产出 vlm_error 键，那是 assemble.py 失败兜底才加的；
    这里断言的是"确实拿到了成功结果，不是失败态"）。"""
    sleeps = _patch_sleep(monkeypatch)
    behaviors = [
        _http_status_exc(InternalServerError, 503, "Error code: 503 - No available channel"),
        _http_status_exc(InternalServerError, 503, "Error code: 503 - No available channel"),
        '{"formality_1_5": 3}',
    ]
    _patch_anthropic_sequence(monkeypatch, behaviors)
    result = ask_vlm(b"fake-png-bytes", _basic_context(person_id="seg01_track1"))
    assert result == {"formality_1_5": 3}
    assert "vlm_error" not in result
    assert len(sleeps) == 2, "两次失败应睡两次，第三次成功后不再睡"

    out = capsys.readouterr().out
    assert "[VLM] seg01_track1 第 1 次重试" in out
    assert "[VLM] seg01_track1 第 2 次重试" in out


def test_does_not_retry_on_4xx(monkeypatch):
    """6. 400 只调用一次——防「无脑重试一切」。"""
    sleeps = _patch_sleep(monkeypatch)
    behaviors = [_http_status_exc(BadRequestError, 400, "bad request: image too large")]
    _capture, client_holder = _patch_anthropic_sequence(monkeypatch, behaviors)
    with pytest.raises(BadRequestError):
        ask_vlm(b"fake-png-bytes", _basic_context())
    assert client_holder["client"].messages.call_count == 1
    assert sleeps == [], "不可重试错误不应有任何退避睡眠"


def test_backoff_intervals_are_exponential(monkeypatch):
    """7. monkeypatch sleep，断言间隔序列确实是 2/4/8（不是固定间隔）。"""
    sleeps = _patch_sleep(monkeypatch)
    behaviors = [
        _http_status_exc(InternalServerError, 503, "503-1"),
        _http_status_exc(InternalServerError, 503, "503-2"),
        _http_status_exc(InternalServerError, 503, "503-3"),
        '{"formality_1_5": 1}',
    ]
    _patch_anthropic_sequence(monkeypatch, behaviors)
    ask_vlm(b"fake-png-bytes", _basic_context())
    assert sleeps == [2.0, 4.0, 8.0]


def test_gives_up_after_max_retries_and_records_attempts(monkeypatch):
    """8. 始终 503 时，调用 4 次后失败（config.VLM_MAX_RETRIES=3，最多重试
    3 次即最多 4 次调用），vlm_error（此处即抛出异常的 str()）含尝试次数。
    """
    _patch_sleep(monkeypatch)
    behaviors = [
        _http_status_exc(InternalServerError, 503, "Error code: 503 - No available channel")
        for _ in range(config.VLM_MAX_RETRIES + 1)
    ]
    _capture, client_holder = _patch_anthropic_sequence(monkeypatch, behaviors)
    with pytest.raises(Exception) as exc_info:
        ask_vlm(b"fake-png-bytes", _basic_context())
    assert client_holder["client"].messages.call_count == config.VLM_MAX_RETRIES + 1
    msg = str(exc_info.value)
    # 精确匹配"重试 N 次"这个短语，不能只查裸数字 3 是否出现——错误原文本身
    # 带着"503"，裸数字断言会被"503"里的那个"3"字符蒙混过关，测不出问题
    # （首次实现时踩过这个坑：把消息改成不含尝试次数的"VLM 调用最终失败:
    # ..."后，`"3" in msg` 因"503"仍判真，变异未被抓到）。
    assert f"重试 {config.VLM_MAX_RETRIES} 次" in msg, (
        f"异常消息应精确记录尝试次数（重试 {config.VLM_MAX_RETRIES} 次），实得：{msg!r}"
    )
    assert "InternalServerError" in msg and "503" in msg, (
        f"异常消息应保留最后一次失败原因，实得：{msg!r}"
    )


def test_html_error_page_is_treated_as_retryable(monkeypatch):
    """9. 中转服务返回 HTML 错误页（导致 VlmParseError）时应重试，
    不是立即放弃——这本质仍是服务端故障，不是 prompt/图片本身的问题。"""
    sleeps = _patch_sleep(monkeypatch)
    behaviors = [
        "<!DOCTYPE HTML><html><body>502 Bad Gateway</body></html>",
        '{"formality_1_5": 2}',
    ]
    _patch_anthropic_sequence(monkeypatch, behaviors)
    result = ask_vlm(b"fake-png-bytes", _basic_context())
    assert result == {"formality_1_5": 2}
    assert len(sleeps) == 1, "HTML 错误页应被当作可重试错误，重试一次后成功"


# --- 变异对照：4xx 中除 429 外，429 本身仍应可重试（不是所有 4xx 都不重试）---

def test_429_rate_limit_is_retryable(monkeypatch):
    """429 属于 4xx，但语义是限流而非请求本身有问题，必须可重试——防止
    `_is_retryable_vlm_error` 被简化成「4xx 一律不重试」这种过粗的判据。
    """
    from anthropic import RateLimitError
    sleeps = _patch_sleep(monkeypatch)
    behaviors = [
        _http_status_exc(RateLimitError, 429, "rate limited"),
        '{"formality_1_5": 4}',
    ]
    _patch_anthropic_sequence(monkeypatch, behaviors)
    result = ask_vlm(b"fake-png-bytes", _basic_context())
    assert result == {"formality_1_5": 4}
    assert len(sleeps) == 1


# ---------------------------------------------------------------------------
# Task 49 · 需求一：2×2 证据图的格位编号
#
# 一个候选的 2×2 拼图里可能挤了多个人。Task 41 加了「人数」下拉，标 >1 时
# 转入 needs_resplit，却没有任何办法让用户指明「谁是谁」。给四格编上号，
# 用户就能在备注里写「1 号格是创业者、2 号格是工作人员」。
#
# **默认不编号**：VLM 用的证据图不加，多余的数字会干扰模型判读；只有挑选页
# 展示用的那份加。
# ---------------------------------------------------------------------------

def _sheet_quadrants(png: bytes):
    """把证据图切成四象限，返回 {(col,row): 该象限左上角 100x100 的像素块}。"""
    from PIL import Image

    arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    h, w = arr.shape[:2]
    ch, cw = h // 2, w // 2
    return {(col, row): arr[row * ch: row * ch + 100, col * cw: col * cw + 100]
            for col in (0, 1) for row in (0, 1)}


def _badge_mask(patch):
    """象限左上角的白底徽标掩码（近白像素）。没画编号时全 False。"""
    return np.all(patch > 240, axis=-1)


def _make_cell_images(tmp_path, colours):
    from PIL import Image

    paths = []
    for i, c in enumerate(colours):
        p = tmp_path / f"cell{i}.jpg"
        Image.fromarray(np.full((200, 200, 3), c, dtype=np.uint8)).save(p, quality=100)
        paths.append(p)
    return paths


def test_evidence_sheet_labels_cells_when_asked(tmp_path):
    """label_cells=True 时四格都要画出编号（白底黑字），且四个编号互不相同
    ——四格全画同一个数字，用户照样指不出谁是谁。"""
    paths = _make_cell_images(tmp_path, [(90, 90, 90)] * 4)
    png = build_evidence_sheet(paths, [(10, 10, 190, 190)] * 4, label_cells=True)

    masks = {}
    for pos, patch in _sheet_quadrants(png).items():
        mask = _badge_mask(patch)
        assert mask.sum() > 150, (
            f"{pos} 格左上角没有白底编号徽标（近白像素仅 {mask.sum()} 个）"
        )
        ys, xs = np.nonzero(mask)
        inside = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        dark = np.all(inside < 90, axis=-1)
        assert dark.sum() > 10, f"{pos} 格的徽标里没有深色字形（只有白底没有数字）"
        masks[pos] = mask

    keys = list(masks)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not np.array_equal(masks[keys[i]], masks[keys[j]]), (
                f"{keys[i]} 与 {keys[j]} 两格画的是同一个数字——编号必须逐格不同"
            )


def test_evidence_sheet_unlabelled_by_default(tmp_path):
    """护栏：默认不加编号。VLM 用的证据图走的就是默认参数，多余的数字会
    干扰模型判读，而且它一旦被加上，下游不会有任何一处报错。"""
    paths = _make_cell_images(tmp_path, [(90, 90, 90)] * 4)
    png = build_evidence_sheet(paths, [(10, 10, 190, 190)] * 4)

    for pos, patch in _sheet_quadrants(png).items():
        assert _badge_mask(patch).sum() == 0, (
            f"默认参数下 {pos} 格出现了白底编号徽标——VLM 用的那份不该带编号"
        )


def test_cell_labels_can_start_from_a_given_number(tmp_path):
    """证据图的格位编号可以从任意号起（用户 2026-09-18 的评审人备选池）。

    备选池切成第二张、第三张 2×2 时，图上若还画 1/2/3/4，人读到「格2」指的
    却是池里的第 6 格——他标的号与流水线用的号从此各说各话，而一处不报错。
    """
    from PIL import Image
    from pipeline.annotate import build_evidence_sheet
    p = tmp_path / "f.png"
    Image.fromarray(np.full((120, 160, 3), 200, dtype=np.uint8)).save(p)
    boxes = [(10.0, 10.0, 60.0, 90.0)] * 4
    first = build_evidence_sheet([p] * 4, boxes, label_cells=True)
    second = build_evidence_sheet([p] * 4, boxes, label_cells=True,
                                  cell_labels=[5, 6, 7, 8])
    assert first != second, "换了起始号，图上的数字却一模一样"
    same = build_evidence_sheet([p] * 4, boxes, label_cells=True,
                                cell_labels=[1, 2, 3, 4])
    assert same == first, "显式传 1-4 应与默认完全一致"


# --- 评审人的打标提示词（用户 2026-09-19：「阶段 7 的 VLM 打标接评审人」）---
#
# 评审席的正式度必须与创业者**可比**——两套量表一旦有一点不同，跨组比较就是
# 在比两把尺子，而数字看上去一样合理。所以两个 prompt 共用同一段量表锚点、
# 同一套拍摄条件说明、同一份 JSON 结构，只有「这是谁」那一句和「已知事实」
# 那一块不同：评审人没有企业、没有本组人数。


def test_the_presenter_prompt_is_unchanged():
    """拆出评审人版本时，创业者那一版必须**一个字节都不变**。

    它是已经跑过四组的那一版；动了它，此前编码过的那些人就不可比了。
    """
    import hashlib
    from pipeline.annotate import PROMPT
    assert hashlib.sha256(PROMPT.encode()).hexdigest() == (
        "7525afed355cbeae282092c9cb5cde5d601e122ae426a8782ca2689d271604db"
    ), "创业者的 prompt 变了——此前编码过的四组就不可比了"


def test_the_judge_prompt_shares_every_scale_with_the_presenter_one():
    """量表锚点、拍摄条件、JSON 结构三块必须逐字相同。"""
    from pipeline import annotate as A
    for shared in (A._SATURATION_ANCHORS, A._BRIGHTNESS_ANCHORS):
        assert shared in A.JUDGE_PROMPT, shared[:40]
    # JSON 结构那一段（从「请只输出 JSON」到结尾）两边必须一致
    tail = A.PROMPT[A.PROMPT.index("请只输出 JSON"):]
    assert tail in A.JUDGE_PROMPT, "JSON 结构两边不一致"
    cond = A.PROMPT[A.PROMPT.index("【拍摄条件】"):A.PROMPT.index("已知事实")]
    assert cond in A.JUDGE_PROMPT, "拍摄条件说明两边不一致"


def test_the_judge_prompt_carries_no_company_or_headcount():
    """评审人没有企业、没有本组人数——留着这两个占位符会 KeyError 或填错。"""
    from pipeline.annotate import JUDGE_PROMPT
    assert "{company}" not in JUDGE_PROMPT
    assert "{presenter_num}" not in JUDGE_PROMPT
    assert "创业者" not in JUDGE_PROMPT.split("【拍摄条件】")[0], (
        "开头那句还把他当创业者")
    assert "评审" in JUDGE_PROMPT.split("【拍摄条件】")[0]


def test_the_judge_prompt_formats_with_only_a_layout():
    """只给 layout 就能格式化——评审人那一路不会有企业名可传。"""
    from pipeline.annotate import JUDGE_PROMPT
    out = JUDGE_PROMPT.format(layout="（测试）")
    assert "（测试）" in out
    # 不能断言「格式化后没有 {」——JSON 结构那段本来就是 {{ }}，format 之后
    # 正该变成单括号。要查的是**没有没填上的占位符**。
    assert "{layout}" not in out and "{company}" not in out


def test_ask_vlm_sends_the_prompt_it_was_given(monkeypatch):
    """`ask_vlm` 要能被指定用哪一份 prompt，默认仍是创业者那一版。"""
    from pipeline import annotate as A

    seen = {}

    def fake(client, png, ctx, prompt=None):
        seen["prompt"] = prompt
        return {"ok": 1}

    monkeypatch.setattr(A, "_call_vlm_once", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(A, "Anthropic", lambda *a, **k: object(), raising=False)
    import sys, types
    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda *a, **k: object()
    mod.APIConnectionError = type("APIConnectionError", (Exception,), {})
    mod.APIStatusError = type("APIStatusError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "anthropic", mod)

    A.ask_vlm(b"png", {"layout": "L"})
    assert seen["prompt"] is A.PROMPT
    A.ask_vlm(b"png", {"layout": "L"}, prompt=A.JUDGE_PROMPT)
    assert seen["prompt"] is A.JUDGE_PROMPT
