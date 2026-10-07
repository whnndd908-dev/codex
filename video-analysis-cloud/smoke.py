"""Exercise real decoding, YOLO tracking and evidence output on packaged sample media."""
from pathlib import Path
import hashlib
import io
import json
import subprocess

from PIL import Image
import torch
from ultralytics.utils import ASSETS

from pipeline import config
from pipeline.annotate import build_evidence_sheet
from pipeline.detect import detect_frames, save_detections, load_detections
from pipeline.frames import FFMPEG, extract_window, probe_duration, probe_fps
from pipeline.schedule import load_segments
from pipeline.windows import Window

OUT = Path('/workspace/video-analysis-cloud/smoke-output')
OUT.mkdir(exist_ok=True)
print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), flush=True)
segments = load_segments()
assert len(segments) == 17
assert sum(s.presenter_num for s in segments) == 42
sample = ASSETS / 'bus.jpg'
assert sample.is_file()
video = OUT / 'sample.mp4'
subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
                '-loop', '1', '-i', str(sample), '-t', '1', '-r', '10',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
assert abs(probe_duration(video) - 1.0) < 0.05
assert probe_fps(video) == 10.0
frames = extract_window(video, Window(1, 'entry', 0.2, 0.6, 0.2), OUT / 'frames')
assert len(frames) == 3, len(frames)
width, height = Image.open(frames[0].path).size
detections = detect_frames(frames, frame_h=height)
assert detections, 'YOLO must detect at least one person in the official bus sample'
assert {d.t for d in detections} == {f.t for f in frames}
save_detections(detections, OUT / 'detections.json')
assert load_detections(OUT / 'detections.json') == detections
chosen = detections[0]
png = build_evidence_sheet([chosen.frame_path], [(chosen.x1, chosen.y1, chosen.x2, chosen.y2)])
im = Image.open(io.BytesIO(png))
assert im.width > 0 and im.height > 0 and im.format == 'PNG'
(OUT / 'evidence.png').write_bytes(png)
report = {'schedule_segments':len(segments), 'presenters':42,
          'sample_media':'ultralytics packaged bus.jpg, repeated for one second',
          'video_seconds':probe_duration(video), 'decoded_frames':len(frames),
          'person_detections':len(detections), 'evidence_size':[im.width,im.height],
          'device':torch.cuda.get_device_name(0),
          'weight_sha256':hashlib.sha256(Path(config.YOLO_MODEL).read_bytes()).hexdigest()}
(OUT / 'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, ensure_ascii=False), flush=True)
