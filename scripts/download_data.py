import json
import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

# downloads the competition (~10 GB) and extracts every 5th train frame, all val and test frames
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.data import select_frames  # noqa: E402

competition = "multiple-object-video-detection-hw-2-ucu-ai-2026"
out = root / "data/raw"
archive = out / f"{competition}.zip"
if not archive.exists():
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition, path=out)
with zipfile.ZipFile(archive) as z:
    names = {"test_images.csv"} | {n for n in z.namelist() if n.startswith("test/images/")}
    for split, stride in [("train", 5), ("val", 1)]:
        coco = json.loads(z.read(f"{split}/annotations_coco.json"))
        names |= {f"{split}/annotations_coco.json"} | {f"{split}/{i['file_name']}" for i in select_frames(coco, stride)}
    for name in sorted(names):
        if not (out / name).exists():
            z.extract(name, out)
print(f"{len(names)} files in {out}")
