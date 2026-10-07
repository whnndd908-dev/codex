"""Task 56F · 交付包：把「人工核实完、可以直接拿去打标」的东西单独拢一份。

用户 2026-08-24 原话：

    「将人工标记核实后用于 VLM 的文件单独放在 `/Users/ll/Downloads/dress/
      work/Output/` 各个组名下，这样方便由其他人标注，标注完成后将产出文件
      发我，我可以依赖最终产物进行 VLM 判定」。

这句话里最要紧的一点是**这份包会离开这台机器**。

登记表（`registry.py`）存的是「某一帧文件的某一个框」——路径指向本机的
`work/<组>/frames/`，那几万张帧不会跟着走。别人拿到一份只有路径的登记表，
打开什么也看不到；更糟的是他若在自己机器上重跑一遍抽帧检测，帧和框全变，
而登记表照样读得出来、数字照样生成（Task 46 的血教训就是这类静默错配）。

所以这里的规矩是：**证据图落成真正的 PNG 文件**，一个座位一张，与打标时
送进模型的那张**逐字节相同**（同一个 `annotate.build_evidence_sheet`、同一
批帧、同一组默认参数）。本模块绝不自己裁一份——两处各裁一份必然走样，而走
样的表现是人核对了一张模型没看过的图。

包的内容与用途：

    work/Output/<组名>/
      ├─ README.md                    怎么标、标完把**哪个文件**发回来
      ├─ manifest.json                这包产自哪一份标注（摘要）、缺哪些证据
      ├─ seat_registry_<组>.json      谁坐哪个座位（打标读它）
      ├─ entry_labels_<组>.json       当前那份人工标注（往返用，可能没有）
      ├─ entry_review.html            标注页（可能没有）
      ├─ review_img/                  标注页的图；页面引用它们，不能拆开
      └─ evidence/segNN_seatM.png     VLM 实际会看到的那张图

陈旧的登记表**不出包**：出了包，别人拿到的是旧图、按旧图标注，回来的结果
与当前标注对不上，而一路无人报错。判定复用 `registry.registry_staleness`，
不在这里另写一份（两处各写一份必然日后走样）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline import config, paths
from pipeline.annotate import build_evidence_sheet
from pipeline.registry import (
    REGISTRY_SCHEMA, RegistryStaleError, _preview_frames, registry_path,
    registry_staleness,
)

# 包里 evidence/ 下每张图的命名：段号补零到两位、座位号不补。
# 补零是为了在 Finder 里按名字排序时 seg02 排在 seg10 前面——收件人多半是
# 直接看文件夹，顺序乱了他会以为漏了段。
EVIDENCE_DIR = "evidence"


def bundle_dir(work_root: Path, group: str) -> Path:
    """某组的交付包目录 `<work_root>/Output/<组名>/`。

    组名用 work 下的目录名（如 `城市2-4.26`）而不是 SHEET：收件人认的是
    「哪一组视频」，而 SHEET 有时是视频文件名那种长串。

    位置由 `paths.output_dir` 单一决定——两处各拼一遍，改了一处没改另一处
    就会「包写到了 A、流水线去 B 找」，而两边都不报错。
    """
    return paths.output_dir(Path(work_root) / group)


def evidence_name(order: int, seat: int) -> str:
    return f"seg{int(order):02d}_seat{int(seat)}.png"


def judge_evidence_name(judge: int) -> str:
    """评审席的证据图名。不带段号——他是**组级**的，整场一个人一张
    （用户 2026-09-17）。"""
    return f"judge{int(judge)}.png"


def _missing_evidence_text(missing: list[str]) -> str:
    """缺图那一句。**分清座位与评审席**——评审人不是创业者座位。

    评审席恒有 5 个号（用户 2026-09-17），只到 4 位时第 5 号必然缺图；照旧
    印成「1 个座位没有证据图」会让人以为少了一个创业者去排查。
    """
    if not missing:
        return ""
    judges = [m for m in missing if m.startswith("judge")]
    seats = [m for m in missing if not m.startswith("judge")]
    parts = []
    if seats:
        parts.append(f"{len(seats)} 个座位没有证据图：{'、'.join(seats)}")
    if judges:
        parts.append(f"评审席有 {len(judges)} 个号没有证据图："
                     f"{'、'.join(judges)}（那一位可能没到场，人工核实）")
    return "；" + "；".join(parts)


def _write_readme(out: Path, sheet: str, n_seats: int, n_evidence: int,
                  missing: list[str], has_page: bool) -> None:
    """给**不是本项目的人**看的说明。

    要点名「送回 entry_labels_<组>.json 这一个文件」：不写清楚，收件人会把
    整个文件夹压缩回来，或者只回一张截图——两种都得再来回一趟。
    """
    lines = [
        f"# {sheet} · 创业者着装标注",
        "",
        "## 这个文件夹是什么",
        "",
        f"这一组视频里共 {n_seats} 个创业者座位。`evidence/` 下每张图对应一个",
        "座位，是后续交给模型判定着装的**那一张原图**（红框标出的就是要判定",
        "的那个人）。",
        "",
        "## 要做什么",
        "",
    ]
    if has_page:
        lines += [
            "1. 用浏览器打开 `entry_review.html`（双击即可，不需要联网）。",
            "   页面的图在同目录的 `review_img/` 里，**请保持整个文件夹在一起**"
            "——只把那个 html 单独拷出去，打开会满屏是叉。",
            "2. 按页面上的提示，把每个座位对应的人确认下来。",
            "3. 标完点页面右下角的「导出标注 JSON」。",
            "",
            "## 标完发回什么",
            "",
            f"**只需要 `entry_labels_{sheet}.json` 这一个文件。**",
            "",
            "不用回传整个文件夹，也不用回传截图——那个 JSON 里已经含全部结果。",
        ]
    else:
        lines += [
            "本次没有附标注页（`entry_review.html`）。请联系发件人索取。",
            "",
            "## 标完发回什么",
            "",
            f"**只需要 `entry_labels_{sheet}.json` 这一个文件。**",
        ]
    lines += [
        "",
        "## 文件清单",
        "",
        f"- `evidence/` — {n_evidence} 张证据图（每座位一张）",
        "- `review_img/` — 标注页用的图（页面引用，不必单独打开）",
        f"- `seat_registry_{sheet}.json` — 谁坐哪个座位（程序读，不必打开）",
        "- `manifest.json` — 这份包产自哪一次标注（程序读，不必打开）",
    ]
    if missing:
        lines += [
            "",
            "## 注意：以下座位／评审席没有证据图",
            "",
            "它们要么还没定下来，要么原始帧已不可用。**不是漏发**：",
            "",
        ] + [f"- {m}" for m in missing]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bundle(reg: dict, out_dir: Path, *, sheet: str,
                 work: Path | None = None) -> dict:
    """产出一组的交付包，返回摘要 dict。

    `work` 给了就顺带带上标注页与人工标注文件，并**先做一次陈旧校验**；
    不给（或目录不存在）时跳过校验，只出登记表与证据图——直接拿一份内存里
    的登记表出包（测试、或已经确认过新鲜度的调用方）是合法用法。
    """
    out_dir = Path(out_dir)
    meta = (reg or {}).get("_meta") or {}

    if work is not None and Path(work).is_dir():
        # 陈旧的登记表不出包。判定只有 registry_staleness 一个来源。
        st = registry_staleness(Path(work), sheet, reg_body=reg)
        if st.stale and st.reason in ("no_digest", "digest_mismatch"):
            raise RegistryStaleError(
                f"不出包：{st.message}"
                "（出了包，别人拿到的是旧图、按旧图标注，回来的结果与当前"
                "标注对不上，而一路无人报错）")

    out_dir.mkdir(parents=True, exist_ok=True)
    ev_dir = out_dir / EVIDENCE_DIR
    ev_dir.mkdir(exist_ok=True)

    n_evidence = 0
    missing: list[str] = []
    segs = (reg or {}).get("segments") or {}
    for order in sorted(segs, key=lambda k: int(k)):
        seg = segs[order] or {}
        for seat in sorted((seg.get("seats") or {}), key=lambda k: int(k)):
            rec = seg["seats"][seat] or {}
            name = evidence_name(int(order), int(seat))
            # 变量不叫 paths——本模块导入了 pipeline.paths，同名局部会把它
            # 遮蔽掉，而遮蔽处与使用处隔着几十行，报出来的是
            # 「'list' object has no attribute ...」这种看不出根因的错。
            frame_paths, boxes = _preview_frames(rec)
            if not frame_paths:
                # 静默少一张，收件人只会以为这个座位本来就不存在；而它其实
                # 是「还没定下来」或「帧文件不在了」，两种都得有人去处理。
                missing.append(name[:-4])
                continue
            try:
                png = build_evidence_sheet(frame_paths, boxes)
            except (OSError, ValueError):
                missing.append(name[:-4])
                continue
            (ev_dir / name).write_bytes(png)
            n_evidence += 1

    # 评审席（组级）。与座位那些走同一个 build_evidence_sheet、同一批帧——
    # 另裁一份就是「人核对的图与模型看到的不是同一张」。
    n_judges = 0
    for n, rec in sorted(((k, v) for k, v in
                          ((reg or {}).get("judges") or {}).get("slots", {}).items()),
                         key=lambda kv: int(kv[0])):
        if not rec:
            missing.append(judge_evidence_name(int(n))[:-4])
            continue
        frame_paths, boxes = _preview_frames(rec)
        if not frame_paths:
            missing.append(judge_evidence_name(int(n))[:-4])
            continue
        try:
            png = build_evidence_sheet(frame_paths, boxes)
        except (OSError, ValueError):
            missing.append(judge_evidence_name(int(n))[:-4])
            continue
        (ev_dir / judge_evidence_name(int(n))).write_bytes(png)
        n_judges += 1
    n_evidence += n_judges

    (out_dir / f"seat_registry_{sheet}.json").write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

    # Task 56G：**不再复制**。交付物本来就写在 Output（见 pipeline/paths.py），
    # 复制正好造出用户要消灭的那份重复：「Output 已经有的文件在原组文件夹下
    # 就不用有了」。这里只把还留在旧位置的**移动**过来，并如实报出移动了哪些。
    moved: list[str] = []
    if work is not None and Path(work).is_dir():
        moved = paths.migrate(Path(work), sheet)
    present = [n for n in paths.deliverable_names(sheet)
               if (out_dir / n).is_file()]

    manifest = {
        "sheet": sheet,
        "schema": meta.get("schema") or REGISTRY_SCHEMA,
        # 这包产自哪一份人工标注。包会离开本机再传回来，没有摘要就无从判断
        # 手上这份是不是最新的。
        "labels_digest": meta.get("labels_digest"),
        "labels_file": meta.get("labels_file"),
        "detections_fingerprint": meta.get("detections_fingerprint"),
        "built_at": meta.get("built_at"),
        "n_seats_expected": meta.get("n_seats_expected"),
        "n_seats_resolved": meta.get("n_seats_resolved"),
        "n_evidence": n_evidence,
        "missing_evidence": missing,
        "present": present,
        "migrated_from_group_dir": moved,
        "send_back": f"entry_labels_{sheet}.json",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_readme(out_dir / "README.md", sheet,
                  int(meta.get("n_seats_expected") or 0), n_evidence, missing,
                  "entry_review.html" in present)
    return manifest


def main(argv: list[str] | None = None) -> int:   # pragma: no cover - CLI 入口
    ap = argparse.ArgumentParser(
        prog="python -m pipeline.bundle",
        description="把人工核实完、可直接拿去打标的东西拢成一份交付包")
    ap.add_argument("--group", default=None,
                    help="work/ 下的组目录名（默认取当前 config.WORK 的名字）")
    ap.add_argument("--sheet", default=None, help="组名（默认取 config.SHEET）")
    ap.add_argument("--out", default=None, help="输出目录（默认 work/Output/<组名>/）")
    args = ap.parse_args(argv)

    work = (config.WORK.parent / args.group) if args.group else config.WORK
    sheet = args.sheet or config.SHEET
    if not work.is_dir():
        print(f"   ⚠ 没有 {work}")
        return 1
    rp = registry_path(work, sheet)
    if not rp.exists():
        print(f"   ⚠ 缺 {rp.name}，先跑 `python -m pipeline.registry --preview`")
        return 1
    reg = json.loads(rp.read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else bundle_dir(config.WORK.parent, work.name)
    try:
        man = write_bundle(reg, out, sheet=sheet, work=work)
    except RegistryStaleError as exc:
        print(f"   ⚠ {exc}")
        return 1
    print(f"   交付包：{out}")
    print(f"   证据图 {man['n_evidence']} 张"
          + _missing_evidence_text(man["missing_evidence"]))
    print(f"   包内交付物：{'、'.join(man['present']) or '（只有证据图）'}")
    if man["migrated_from_group_dir"]:
        print(f"   已从组目录移入（不再重复存储）："
              f"{'、'.join(man['migrated_from_group_dir'])}")
    dups = paths.duplicated(work, sheet)
    if dups:
        print(f"   ⚠ 两处都有、未自动处理：{'、'.join(dups)}"
              "（Output 那份作数；请自己看过再删组目录里的）")
    print(f"   对方标完发回：{man['send_back']}")
    return 0


if __name__ == "__main__":       # pragma: no cover - CLI 入口
    raise SystemExit(main())
