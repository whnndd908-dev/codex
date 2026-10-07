#!/usr/bin/env bash
set -euo pipefail
cd /workspace/video-analysis
command -v python >/dev/null
command -v uv >/dev/null
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
test -f pipeline/config.py
test -f 'data/2017创新video.dta'
mkdir -p /workspace/video-analysis-cloud/uv-cache \
  /workspace/video-analysis-cloud/yolo-config \
  /workspace/video-analysis-cloud/huggingface models work
python -m venv .venv
export UV_CACHE_DIR=/workspace/video-analysis-cloud/uv-cache
uv pip sync /workspace/video-analysis-cloud/requirements.lock \
  --python .venv/bin/python --require-hashes \
  --index https://pypi.org/simple \
  --default-index https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-first-match
uv pip check --python .venv/bin/python
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
import tempfile
import urllib.request
import torch

weight = Path('/workspace/video-analysis/models/yolo26x.pt')
expected = '9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92'
if not weight.exists():
    url = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt'
    with tempfile.NamedTemporaryFile(dir=weight.parent, prefix='.download-', delete=False) as f:
        temp = Path(f.name)
        with urllib.request.urlopen(url, timeout=60) as r:
            while chunk := r.read(1024 * 1024):
                f.write(chunk)
    if hashlib.sha256(temp.read_bytes()).hexdigest() != expected:
        temp.unlink()
        raise SystemExit('YOLO weight checksum mismatch')
    temp.rename(weight)
assert hashlib.sha256(weight.read_bytes()).hexdigest() == expected, 'YOLO weight checksum mismatch'
assert torch.version.cuda == '12.6', torch.version.cuda
print('CUDA dependencies and YOLO weights verified; visible GPUs:', torch.cuda.device_count())
PY
