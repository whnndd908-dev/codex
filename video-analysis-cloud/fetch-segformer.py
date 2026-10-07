"""Fetch the documented public model after its network destinations are enabled."""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ['HF_HOME'] = '/workspace/video-analysis-cloud/huggingface'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
import logging

disable_progress_bars()
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
metadata = json.loads(Path('/workspace/video-analysis-cloud/segformer-source.json').read_text())
try:
    snapshot = Path(snapshot_download(metadata['repo'], revision=metadata['revision'],
                                     allow_patterns=['config.json', 'preprocessor_config.json',
                                                     'model.safetensors']))
except Exception as exc:
    # Signed redirect URLs are not needed in the setup report.
    print(f'Model download failed ({type(exc).__name__}). '
          'Check HTTPS access to huggingface.co and us.aws.cdn.hf.co.', file=sys.stderr)
    raise SystemExit(1)
expected = next(f['lfs']['sha256'] for f in metadata['files']
                if f['rfilename'] == 'model.safetensors')
actual = hashlib.sha256((snapshot / 'model.safetensors').read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit('SegFormer model checksum mismatch; do not use this artifact.')
# Pin the main cache ref to the verified revision for the source code's model ID.
ref = snapshot.parent.parent / 'refs' / 'main'
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(metadata['revision'])
print('SegFormer safetensors verified:', metadata['revision'])
