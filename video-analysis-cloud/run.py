"""Apply local cloud paths before running an unmodified project module."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import shutil
import sys

PROJECT = Path('/workspace/video-analysis')
SETUP = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--dta', type=Path)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--panel')
    parser.add_argument('--sheet')
    parser.add_argument('--expected-segments', type=int)
    parser.add_argument('--expected-presenters', type=int)
    parser.add_argument('module')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    root = ns.root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ.setdefault('DRESS_FFMPEG', shutil.which('ffmpeg') or '/usr/bin/ffmpeg')
    os.environ.setdefault('DRESS_FFPROBE', shutil.which('ffprobe') or '/usr/bin/ffprobe')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(SETUP / 'yolo-config'))
    os.environ.setdefault('HF_HOME', str(SETUP / 'huggingface'))
    # Use the supported HTTPS downloader and the configured public CDN.
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    # Model fetching is a separate explicit setup step; reuse verified artifacts.
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)
    from pipeline import config
    config.ROOT = root
    config.DTA = (ns.dta or root / 'data' / '2017创新video.dta').resolve()
    config.VIDEO = (ns.video or root / 'data' / config.VIDEO.name).resolve()
    config.XLSX = root / 'data' / 'Book1.xlsx'
    config.WORK = (ns.work or root / 'work' / config.WORK.name).resolve()
    config.YOLO_MODEL = str(root / 'models' / 'yolo26x.pt')
    if ns.panel is not None:
        config.PANEL = ns.panel
    if ns.sheet is not None:
        config.SHEET = ns.sheet
    if ns.expected_segments is not None:
        config.EXPECTED_SEGMENT_COUNT = ns.expected_segments
    if ns.expected_presenters is not None:
        config.EXPECTED_PRESENTER_TOTAL = ns.expected_presenters
    needs_gpu = ns.module in {'smoke', 'pipeline.detect'}
    if ns.module in {'pipeline.prepare', 'pipeline.run_all'}:
        stages = next((ns.args[i + 1] for i, arg in enumerate(ns.args[:-1])
                       if arg == '--stages'), '1,2,3,4,5')
        needs_gpu = '3' in stages.split(',')
    if needs_gpu:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit('当前容器未挂载可用的 NVIDIA GPU。请挂载 GPU 后重试；'
                             '此启动入口按用户要求使用 CUDA 检测。'
                             '排期可独立运行：pipeline.run_all --stages 1。')
        os.environ.setdefault('DRESS_YOLO_DEVICE', '0')
    sys.argv = [ns.module, *ns.args]
    runpy.run_module(ns.module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
