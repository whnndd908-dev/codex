"""Task 56G · 交付物的**唯一**落脚点。

用户 2026-08-24：

    「Output 已经有的文件在原组文件夹下就不用有了，即原来的位置放原始文件和
      过程文件，Output 放输出文件，不用重复存储」。

分工从此是一句话：

    work/<组>/          原始与过程文件——帧、检测、时间表、就座结果、打标结果
    work/Output/<组>/   交付物——标注页、人工标注、座位登记表、核对页、证据图

同一份东西存两处是这个项目反复吃亏的根源（Ruling R39 那一类）：两处迟早不
一致，而不一致时哪一份作数没人说得清，且**不报错**。此前 `bundle.py` 是把
组目录里的文件**复制**进 Output——那正好造出了用户要消灭的这份重复。

## 为什么读的时候还要看旧位置

用户手上已经有好几组标注躺在旧位置（`work/<组>/entry_labels_*.json`），那是
花了几小时的人工成果。改了规矩就当它不存在，表现是「一条标注都没有」，而
流水线照常跑完、结果全退回就座基线，一处也不报错。所以**读**一律先看
Output、没有再回落到组目录；**写**一律只写 Output。

两处都有时以 Output 为准，并且 `duplicated()` 要能把这件事报出来——静默二选
一等于把「哪一份作数」这个问题藏起来，而它迟早会以「我明明标过」的形式炸出来。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pipeline import config

OUTPUT_DIRNAME = "Output"


def output_dir(work: Path) -> Path:
    """某组的交付目录 `work/Output/<组名>/`。

    与各组目录**平级**，不是塞进组目录里：塞进去等于把交付物混回过程文件堆
    中，用户要发给别人时得先在几万张帧中间把它挑出来。
    """
    work = Path(work)
    return work.parent / OUTPUT_DIRNAME / work.name


def _safe(sheet: str) -> str:
    return config.sanitize_filename_component(sheet)


def deliverable_names(sheet: str) -> list[str]:
    """这一组的交付物文件名清单（不含 evidence/ 与 manifest/README）。

    单一来源：迁移、去重检查、包的组装都读它。分散手写必然漏掉后加的那个，
    而漏掉的表现是那份文件被永远留在旧位置、两处各改各的。
    """
    s = _safe(sheet)
    return ["entry_review.html", f"entry_labels_{s}.json",
            f"seat_registry_{s}.json", f"vlm_preview_{s}.html",
            f"judges_{s}.json"]


def _resolve(work: Path, name: str) -> Path:
    """读用：Output 里有就用 Output，没有回落到组目录（旧位置）。

    两处都没有时返回 **Output 那个路径**——调用方多半接着要写，写就该写到
    新位置；而 `.exists()` 为假这件事两条路径上都成立。
    """
    out = output_dir(work) / name
    if out.exists():
        return out
    legacy = Path(work) / name
    return legacy if legacy.exists() else out


def labels_path(work: Path, sheet: str) -> Path:
    return _resolve(work, f"entry_labels_{_safe(sheet)}.json")


def judges_path(work: Path, sheet: str) -> Path:
    """评审席的打标结果（用户 2026-09-19）。

    与 `annotations.json` **分开存**：那一份按 (段, 座位) 组织、每行都带企业
    与座位号，而评审人是组级的——没有段、没有企业、没有座位号。混进去就得给
    每一个下游消费者加一条「这一行 seg_order=0，特殊对待」，而漏加的那一处
    会把一位评审人当成某家企业的创业者统计进去，且不报错。
    """
    return _resolve(work, f"judges_{_safe(sheet)}.json")


def registry_path(work: Path, sheet: str) -> Path:
    return _resolve(work, f"seat_registry_{_safe(sheet)}.json")


def preview_path(work: Path, sheet: str) -> Path:
    return _resolve(work, f"vlm_preview_{_safe(sheet)}.html")


def review_page_path(work: Path) -> Path:
    return _resolve(work, "entry_review.html")


def duplicated(work: Path, sheet: str) -> list[str]:
    """两处都存在的交付物文件名。有就该清掉一份，别让它继续飘着。"""
    out = output_dir(work)
    return [n for n in deliverable_names(sheet)
            if (out / n).exists() and (Path(work) / n).exists()]


def migrate(work: Path, sheet: str) -> list[str]:
    """把还留在组目录里的交付物**移动**到 Output，返回实际移动了哪些。

    移动而不是复制：复制正好造出用户要消灭的那份重复。

    Output 里已经有同名文件时**不动**（既不覆盖也不删旧的）：覆盖掉的可能
    正是对的那一份——用户刚在 Output 里标完——而覆盖不报错。这种情况由
    `duplicated()` 报出来，让人自己看过再定。
    """
    work = Path(work)
    out = output_dir(work)
    moved: list[str] = []
    for name in deliverable_names(sheet):
        src = work / name
        if not src.is_file():
            continue
        dst = out / name
        if dst.exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append(name)
    return sorted(moved)
