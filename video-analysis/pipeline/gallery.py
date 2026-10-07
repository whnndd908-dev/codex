"""阶段 ⑥：生成单文件 HTML 复核界面。

所有资源内联，不依赖外部 CDN——这份 HTML 要能离线打开，也要能长期归档。
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
import json

from pipeline import annotate, config
from pipeline.annotate import PersonAnnotation
from pipeline.validate import Violation

# 除 formality_1_5 外的下拉框选项，值即标签，静态即可。
#
# Task 43：outer_garment_type 的"Polo"改为中文"翻领短袖衫"（用户要求提示词
# 一律用中文），必须与 pipeline/annotate.py 的 PROMPT 里 outer_garment_type
# 枚举同步——两处各写一份字面量本就是 Ruling R39 提醒过的漂移风险，这里再
# 加一条 `tests/test_annotate.py::test_prompt_and_gallery_garment_options_
# match` 逐项比对锁住，防止某次只改了一处。
_EDITABLE_STATIC = [
    ("outer_garment_type", ["西装外套", "夹克", "毛衣", "衬衫", "翻领短袖衫", "T恤", "连衣裙", "其他"]),
    ("has_tie", ["有", "无", "不可见"]),
    ("hue_class_final", ["红", "橙", "黄", "绿", "蓝", "紫", "粉", "棕", "黑", "白", "灰"]),
    ("inner_color_class", ["白", "浅色", "深色", "蓝色系", "花纹", "不可见"]),
    ("evidence_quality", ["good", "fair", "poor"]),
]


def _editable_fields() -> list:
    """formality_1_5 的选项要显示每档含义（"3 = 翻领短袖衫、衬衫（正装或休闲）无外套…"），
    但存进 localStorage / 导出 CSV 的值必须仍是纯数字。此处产出
    (value, label) 二元组；(value, label) 之外的字段值=标签，二者都由
    前端 card() 统一处理。

    每次调用时从 `annotate.FORMALITY_LABELS` 现取（模块引用而非在此处
    另抄一份字面量），改动那份量表文案会立刻反映到这里，不会漂移
    （Ruling R39 的教训——规则名曾经在两处重复定义而彼此走样）。
    """
    formality_opts = [(str(n), f"{n} = {label}")
                       for n, label in annotate.FORMALITY_LABELS.items()]
    return [("formality_1_5", formality_opts)] + list(_EDITABLE_STATIC)


def _is_scalar(v) -> bool:
    """只有标量能安全进入 CSV 与 JSON。

    用白名单而非黑名单：原先写 `not isinstance(v, (dict, list))`，
    结果 bytes 漏了过去——证据图 PNG 字节会以 b'\\x89PNG...' 的形态
    污染 CSV 单元格，并让 json.dumps 直接崩溃。
    """
    return v is None or isinstance(v, (str, int, float, bool))


def annotation_to_row(a: PersonAnnotation) -> dict:
    """展平为 CSV 一行。嵌套的 vlm / pixel 字典摊开成同级列。"""
    row = {
        "person_id": a.person_id, "order": a.seg_order, "company": a.company,
        "seat_index": a.seat_index, "presenter_num": a.presenter_num,
        "gender_female": a.gender_female, "speaks_pre": a.speaks_pre,
        "speaks_qa": a.speaks_qa, "seg_start": a.seg_start, "seg_end": a.seg_end,
        "best_frame_entry": a.best_frame_entry, "best_frame_seated": a.best_frame_seated,
        "suspect_staff": a.suspect_staff,
        # Task 44：进出场递补三件套，一律落 CSV——不论这一行是不是递补来
        # 的，都要能看出「衣着证据来自哪里」，不能只在递补行才写这一列
        # （那样普通行缺列，导出 CSV 时列集合会因行而异）。
        "source": a.source, "seat_uncertain": a.seat_uncertain,
        "seat_rejected": a.seat_rejected,
        # Task 53 需求六：跨段流转。与上面几个同理，**不论是不是跨段来的
        # 都要写这两列**，否则列集合会因行而异。
        "cross_segment": a.cross_segment, "src_seg_order": a.src_seg_order,
        "expert_removed": a.expert_removed,
    }
    row.update({k: v for k, v in a.vlm.items() if _is_scalar(v)})
    row.update({k: v for k, v in a.pixel.items() if _is_scalar(v)})
    row.setdefault("hue_class_final", row.get("hue_class_best", ""))
    row.setdefault("human_verified", 0)
    row.setdefault("human_notes", "")
    # 人可读的正式度标签（"4 = 衬衫+西裤或商务外套（非成套）"），从
    # annotate.FORMALITY_LABELS 派生，不在此处另抄文案。formality_1_5 为空
    # （缺座位占位行 / VLM 调用失败）时该列也留空，不能替用户编造档位。
    row["formality_label"] = annotate.formality_label(row.get("formality_1_5"))
    return row


# ---------------------------------------------------------------------------
# 缩略图 —— 默认原尺寸内联，只有整份画廊预估体积超过 config.GALLERY_MAX_MB
# 才缩放（Task 32，要求一）。两条图片来源统一走同一套“先量后缩”流程：
#   - 证据图（build_evidence_sheet 产出的 2x2 红框拼图）：主路径，人物占比
#     远高于整帧（Task 25 缺陷一）。
#   - 整帧回退（best_frame_seated / best_frame_entry）：证据图缺失时用
#     （例如缺座位的占位行，压根没有 _evidence_png 可用）。
# ---------------------------------------------------------------------------


def _read_raw_image_bytes(path: str) -> tuple[bytes | None, str | None]:
    """读取整帧图片的原始字节，供原尺寸内联。

    做一次轻量解码校验（Image.open().verify()）而不改动字节本身——单张
    图片损坏不应让整份画廊无法生成，该行仍要能正常显示其余字段（沿用
    原 `_thumb_b64` 的容错约定）。
    """
    if not path:
        return None, None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None, None
    try:
        data = p.read_bytes()
    except OSError:
        return None, None
    from PIL import Image, UnidentifiedImageError
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError):
        return None, None
    media = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return data, media


def _b64_len(data: bytes) -> int:
    return len(base64.b64encode(data))


def _downscale_to_jpeg(data: bytes, max_dim: int, quality: int = 85) -> bytes | None:
    from PIL import Image, UnidentifiedImageError
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except (UnidentifiedImageError, OSError):
        return None


# 超限时逐档缩小的目标最大边长（px），从大到小依次尝试，缩到达标即停。
_DOWNSCALE_DIMS = [700, 500, 350, 250, 160]


def _fit_images_to_budget(candidates: list[dict], budget_bytes: int) -> tuple[bool, int, int]:
    """按“先缩最大的几张”的顺序把总体积压到 budget_bytes 以内。

    candidates 里每个元素形如 {"data": bytes|None, "media": str}，原地修改。
    返回 (是否发生了缩放, 缩放前总字节, 缩放后总字节)——字节数按 base64
    编码后的长度算，因为那才是实际写进 HTML 的体积。
    """
    sizes = {i: (_b64_len(c["data"]) if c["data"] else 0) for i, c in enumerate(candidates)}
    total_before = sum(sizes.values())
    total = total_before
    if total <= budget_bytes:
        return False, total_before, total_before

    downscaled = False
    for max_dim in _DOWNSCALE_DIMS:
        if total <= budget_bytes:
            break
        order = sorted(
            (i for i, c in enumerate(candidates) if c["data"]),
            key=lambda i: sizes[i], reverse=True,
        )
        for i in order:
            if total <= budget_bytes:
                break
            c = candidates[i]
            new_data = _downscale_to_jpeg(c["data"], max_dim)
            if new_data is None:
                continue
            new_size = _b64_len(new_data)
            if new_size < sizes[i]:
                total += new_size - sizes[i]
                sizes[i] = new_size
                c["data"] = new_data
                c["media"] = "image/jpeg"
                downscaled = True
    return downscaled, total_before, total


def _evidence_from_disk(a, keyframes_dir) -> bytes | None:
    """内存里没有证据图时，从 keyframes/ 读回那张交给模型的 2×2 拼图。

    `_evidence_png` 是二进制，写 annotations.json 时被刻意剥掉，于是想单独
    重跑阶段 8 换个页面样式，就得连阶段 7 的 VLM 一起重跑——几十秒的事拖成
    半小时，还白烧一遍 VLM 调用（Ruling R57 当时的补救是"阶段 8 只在不跑
    阶段 7 时才从 JSON 重载"，治标未治本）。

    但图一直都在盘上：export_keyframes 已把它存为
    evidence_sent_to_model.png（实测三组 27/27/30 张俱在）。路径由
    assemble.person_keyframe_dir 统一给出，写端读端共用一份拼接逻辑。

    读失败一律返回 None 交由整帧回退：画廊是核对工具，缺一张图不该让整页崩。
    """
    if keyframes_dir is None:
        return None
    try:
        from pipeline.assemble import person_keyframe_dir
        p = (person_keyframe_dir(Path(keyframes_dir), a.seg_order, a.company,
                                 a.seat_index) / "evidence_sent_to_model.png")
        return p.read_bytes() if p.is_file() else None
    except Exception:
        return None


def render_gallery(
    annotations: list[PersonAnnotation],
    violations: list[Violation],
    out_path: Path,
    keyframes_dir: Path | None = None,
) -> None:
    if keyframes_dir is None:
        keyframes_dir = config.WORK / "keyframes"
    rows = [annotation_to_row(a) for a in annotations]

    candidates: list[dict] = []
    for a in annotations:
        evidence = a.pixel.get("_evidence_png") or _evidence_from_disk(a, keyframes_dir)
        if evidence:
            candidates.append({"data": evidence, "media": "image/png"})
        else:
            data, media = _read_raw_image_bytes(a.best_frame_seated or a.best_frame_entry)
            candidates.append({"data": data, "media": media or "image/jpeg"})

    budget_bytes = int(config.GALLERY_MAX_MB * 1024 * 1024)
    downscaled, total_before, total_after = _fit_images_to_budget(candidates, budget_bytes)

    for r, c in zip(rows, candidates):
        r["_thumb"] = base64.b64encode(c["data"]).decode() if c["data"] else ""
        r["_thumb_type"] = c["media"]

    notice_html = ""
    if downscaled:
        msg = (
            f"为控制体积已缩放，原图见 keyframes/"
            f"（{total_before/1e6:.1f}MB \u2192 {total_after/1e6:.1f}MB，"
            f"上限 {config.GALLERY_MAX_MB}MB）"
        )
        print(f"[gallery] {msg}")
        notice_html = f'<p class="notice">\u26a0 {msg}</p>'

    viol = [{"rule": v.rule, "order": v.seg_order,
             "person_id": v.person_id, "detail": v.detail} for v in violations]

    html = _TEMPLATE.replace("__IMG_SCALE__", str(config.GALLERY_IMG_SCALE))
    html = html.replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
    html = html.replace("__VIOL__", json.dumps(viol, ensure_ascii=False))
    html = html.replace("__FIELDS__", json.dumps(_editable_fields(), ensure_ascii=False))
    # localStorage 键按视频命名空间隔离：person_id 形如 seg02_seat1，
    # 跨视频必然碰撞，同一浏览器打开第二份画廊时两份人工校对结果会互相覆盖。
    html = html.replace(
        "__STORAGE_KEY__", json.dumps(f"dress_review_v1::{config.SHEET}", ensure_ascii=False)
    )
    # 导出文件名带组名（Task 33）：三组视频的人工校对版都叫 result.csv 的话，
    # 分别下载后堆在同一个文件夹里会互相覆盖、也分不清谁是谁。净化规则与
    # export_keyframes/阶段⑧⑨产物文件名共用 config.sanitize_filename_component，
    # 不另写一份。
    safe_sheet = config.sanitize_filename_component(config.SHEET)
    html = html.replace(
        "__EXPORT_FILENAME__",
        json.dumps(f"result_人工校对_{safe_sheet}.csv", ensure_ascii=False),
    )
    html = html.replace("__FORMALITY_LEGEND__", _formality_legend_html())
    html = html.replace("__DOWNSCALE_NOTICE__", notice_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def _formality_legend_html() -> str:
    """把正式度量表的五档说明渲染成静态可见的 HTML（不依赖 JS 才能看到），
    放在页面顶部供复核时随时对照。文案取自 annotate.FORMALITY_LABELS——
    单一来源，不在此处另抄一份（Ruling R39）。
    """
    items = "".join(
        f"<li>{n} = {label}</li>" for n, label in annotate.FORMALITY_LABELS.items()
    )
    return f'<p><b>正式程度量表</b></p><ul style="margin-top:4px">{items}</ul>'


_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>着装编码复核</title>
<style>
 :root{--img-scale: __IMG_SCALE__}
 body{font-family:-apple-system,"PingFang SC",sans-serif;margin:24px;background:#fafafa;color:#222}
 h2{border-bottom:2px solid #ddd;padding-bottom:6px;margin-top:36px}
 .card{display:flex;gap:12px;background:#fff;border:1px solid #ddd;border-radius:8px;
       padding:10px;margin:8px 0;align-items:flex-start}
 .card.flag{border-color:#e03030;background:#fff6f6}
 .card.unidentified{background:#e9e9e9}
 /* 显示尺寸用 CSS 变量 --img-scale 缩放（Task 36 要求二），值来自
    config.GALLERY_IMG_SCALE，由 render_gallery() 在渲染时填入这个占位符
    （模板源码里搜 IMG_SCALE 那个替换点）。
    这是纯显示层缩放，内联的 base64 图片字节本身不被触碰——不要把这里
    改成在 Python 里用 Pillow 重新采样再内联，那会真的损失像素，且不可逆
    （用户放大浏览器或另存图片时就再也拿不回原分辨率了）。体积超限时的
    JPEG 重采样回退（config.GALLERY_MAX_MB，_fit_images_to_budget）是另一套
    独立机制，不要合并成一个。 */
 .card img{width:calc(var(--img-scale) * 100%);max-width:calc(var(--img-scale) * 100%);
           height:auto;border-radius:4px;display:block}
 .noimg{width:200px;height:150px;background:#eee;border-radius:4px;flex:none}
 .warn{color:#a33;font-weight:600;font-size:13px;margin-bottom:4px}
 .backfill-note{color:#1a6a3c;font-weight:600;font-size:13px;margin-bottom:4px}
 .notice{background:#fff8e1;border-left:4px solid #e0a030;padding:8px 12px;font-size:13px}
 .info{flex:1;min-width:0}
 .order-num{color:#888;font-weight:700;margin-right:6px}
 .fields{display:flex;flex-direction:column;gap:8px;margin-top:6px}
 .fields .field{display:flex;align-items:center;gap:8px}
 .fields .field label{width:210px;flex:none;font-size:14px;color:#666;white-space:nowrap}
 .fields select{font-size:14px;padding:4px 6px;flex:1;min-width:0}
 .fields .field.full input{flex:1;padding:6px;font-size:15px}
 .viol{background:#fff0f0;border-left:4px solid #e03030;padding:8px 12px;margin:6px 0;font-size:13px}
 button{padding:8px 18px;font-size:14px;cursor:pointer;border-radius:6px;
        border:1px solid #444;background:#222;color:#fff}
 .export-note{font-size:12px;color:#777;margin:4px 0 0}
 details{margin-top:24px}
</style>
<h1>着装编码复核</h1>
<p>改动自动存入浏览器本地。全部核完后点「导出人工校对版 CSV」。</p>
__DOWNSCALE_NOTICE__
__FORMALITY_LEGEND__
<button onclick="exportCsv()">导出人工校对版 CSV</button>
<p class="export-note">VLM 原始版本见 result.csv，本文件为校对后版本</p>
<div id="viol"></div>
<div id="main"></div>
<details><summary>疑似工作人员 / 被剔除</summary><div id="staff"></div></details>
<script>
const ROWS=__ROWS__, VIOL=__VIOL__, FIELDS=__FIELDS__;
const KEY = __STORAGE_KEY__;
const EXPORT_FILENAME = __EXPORT_FILENAME__;
const store=JSON.parse(localStorage.getItem(KEY)||'{}');

function save(pid,k,v){
  store[pid]=store[pid]||{}; store[pid][k]=v; store[pid].human_verified=1;
  localStorage.setItem(KEY,JSON.stringify(store));
  render();
}
function val(r,k){ return (store[r.person_id]&&store[r.person_id][k]!==undefined)
                        ? store[r.person_id][k] : (r[k]??''); }
function isStaff(r){ return !!val(r,'suspect_staff'); }
// 整行都没识别到人（formality_1_5 与 outer_garment_type 均为空）——这类行
// 的全部下拉框选项都会落在空占位项上，不能只靠“下拉框恰好是空”让人
// 自己发现，必须在卡片上给出显眼提示（Task 32 要求三）。
function isUnidentified(r){ return !val(r,'formality_1_5') && !val(r,'outer_garment_type'); }
function esc(s){ return String(s??'').replace(/[&<>"']/g, c=>({
  '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
}[c])); }

document.getElementById('viol').innerHTML =
  VIOL.map(v=>`<div class="viol"><b>${esc(v.rule)}</b> order ${esc(v.order)} ${esc(v.person_id)} \u2014 ${esc(v.detail)}</div>`).join('');

function card(r){
  const flagged = VIOL.some(v=>v.person_id===r.person_id) || isStaff(r);
  const unidentified = isUnidentified(r);
  // 选项元素既可以是裸字符串（值=标签），也可以是 [value,label] 二元组
  // （formality_1_5 用这个：下拉框显示每档含义，但存进 localStorage /
  // 导出 CSV 的必须仍是纯数字 value，不能是带含义的中文标签全文——
  // 否则会静默把脏值污染进 result.csv 的 formality_1_5 列）。
  // 每个 select 的第一项固定是空占位（value="" 的（未识别）），且字段值
  // 为空时它被选中——未识别的人不能因为 <select> 默认选中第一项而在界面
  // 上显示成某个具体选项，看起来像识别成功了（Task 32 要求三，真 bug）。
  // 占位项的 value 必须是空字符串而不是「（未识别）」文案本身，否则用户
  // 一旦在其他字段上触发 save()，这个占位文案会被整行存进 store 并在导出
  // CSV 时把该列污染成「（未识别）」这五个字，而不是空值。
  const fields = FIELDS.map(([k,opts])=>`<span class="field"><label>${esc(k)}</label>
    <select onchange="save('${r.person_id}','${k}',this.value)">
      <option value="" ${String(val(r,k))===''?'selected':''}>\uff08\u672a\u8bc6\u522b\uff09</option>
      ${opts.map(o=>{
        const ov = Array.isArray(o) ? o[0] : o;
        const ol = Array.isArray(o) ? o[1] : o;
        return `<option value="${esc(ov)}" ${String(val(r,k))===String(ov)?'selected':''}>${esc(ol)}</option>`;
      }).join('')}
    </select></span>`).join('');
  const staffBtn = isStaff(r)
    ? ` \u00b7 <button onclick="save('${r.person_id}','suspect_staff',0)">\u6062\u590d\u4e3a\u521b\u4e1a\u8005</button>`
    : '';
  const warn = unidentified
    ? '<div class="warn">\u26a0 \u672c\u884c\u672a\u8bc6\u522b\u5230\u4eba\uff0c\u4ee5\u4e0b\u9009\u9879\u5747\u4e3a\u7a7a\uff0c\u9700\u4eba\u5de5\u586b\u5199</div>'
    : '';
  // Task 44\uff1a\u9012\u8865\u884c\uff08\u8863\u7740\u8bc1\u636e\u6765\u81ea\u8fdb\u51fa\u573a\u753b\u9762\uff0c\u4e0d\u662f\u5c31\u5ea7\u753b\u9762\uff09\u5fc5\u987b\u5728\u5361\u7247\u4e0a
  // \u663e\u773c\u6807\u51fa\uff0c\u4e0d\u80fd\u8ba9\u7528\u6237\u8bef\u4ee5\u4e3a\u8fd9\u884c\u7684\u8863\u7740\u6765\u81ea\u5c31\u5ea7\u955c\u5934\uff08source='seated' \u624d
  // \u662f\u5c31\u5ea7\u6765\u6e90\uff0c\u89c1 pipeline.assemble.assemble_annotations \u7684\u4e09\u4ef6\u5957\u6807\u5fd7\uff09\u3002
  // seat_uncertain=1 \u65f6\u8ffd\u52a0\u5ea7\u4f4d\u53f7\u672a\u7ecf\u786e\u8ba4\u7684\u63d0\u793a\u2014\u2014\u90a3\u662f fill_plan \u731c\u7684\uff0c
  // \u4e0d\u662f\u7528\u6237\u5728\u590d\u6838\u9875\u660e\u786e\u6307\u5b9a\u7684\u3002
  const backfilled = String(val(r,'source')) === 'entryexit';
  const backfillNote = backfilled
    ? `<div class="backfill-note">\u21c4 \u8863\u7740\u53d6\u81ea\u8fdb\u51fa\u573a\u753b\u9762${
        Number(val(r,'seat_uncertain')) === 1 ? '\uff0c\u5ea7\u4f4d\u53f7\u672a\u7ecf\u786e\u8ba4' : ''
      }</div>`
    : '';
  return `<div class="card ${flagged?'flag':''}${unidentified?' unidentified':''}">
    ${r._thumb?`<img src="data:${r._thumb_type||'image/jpeg'};base64,${r._thumb}">`:'<div class="noimg"></div>'}
    <div class="info">
      ${warn}
      ${backfillNote}
      <span class="order-num">#${esc(r.order)}</span><b>${esc(r.company)}</b> \u00b7 \u5ea7\u4f4d ${esc(r.seat_index)}/${esc(r.presenter_num)} \u00b7
      ${r.gender_female?'\u5973':'\u7537'} \u00b7 \u7c92\u5ea6 ${esc(r.color_granularity_achieved||'-')}
      ${staffBtn}
      <div class="fields">${fields}
        <span class="field full"><label>human_notes</label>
          <input value="${esc(val(r,'human_notes'))}"
                 onchange="save('${r.person_id}','human_notes',this.value)"></span>
      </div>
    </div></div>`;
}

function render(){
  const groups={};
  ROWS.forEach(r=>{(groups[r.company]=groups[r.company]||[]).push(r)});
  document.getElementById('main').innerHTML = Object.entries(groups).map(([c,rs])=>
    `<h2>${esc(c)}</h2>` + rs.filter(r=>!isStaff(r)).map(card).join('')).join('');
  document.getElementById('staff').innerHTML =
    ROWS.filter(isStaff).map(card).join('') || '<p>\u65e0</p>';
}
render();

function exportCsv(){
  // human_edited_fields：该行被人工改过哪些字段（逗号分隔），方便比较人工
  // 校对版与 VLM 原始版 result.csv 之间到底动了哪里。human_verified 只是
  // 「本行有没有被碰过」的元标志，不是某个具体字段被改了，不计入这一列。
  const merged = ROWS.map(r=>{
    const edits = store[r.person_id] || {};
    const editedFields = Object.keys(edits).filter(k=>k!=='human_verified');
    return {...r, ...edits, human_edited_fields: editedFields.join(',')};
  });
  merged.forEach(r=>{delete r._thumb; delete r._thumb_type;});
  const cols=[...new Set(merged.flatMap(Object.keys))];
  const esc=v=>`"${String(v??'').replace(/"/g,'""')}"`;
  const csv=[cols.join(',')].concat(merged.map(r=>cols.map(c=>esc(r[c])).join(','))).join('\\n');
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob(['\\ufeff'+csv],{type:'text/csv'}));
  a.download=EXPORT_FILENAME; a.click();
}
</script>
"""
