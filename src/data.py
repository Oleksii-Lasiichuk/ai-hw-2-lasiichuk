import json
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def load(path):
    with open(path) as f:
        return json.load(f)


def select_frames(coco, stride=1):
    clips = defaultdict(list)
    for image in sorted(coco["images"], key=lambda i: i["frame_id"]):
        clips[image["video"]].append(image)
    return [image for clip in sorted(clips) for image in clips[clip][::stride]]


def index_annotations(coco):
    index = defaultdict(list)
    for a in coco["annotations"]:
        index[a["image_id"]].append(a)
    return index


def ignore_mask(image, annotations):
    # crowd boxes are repeated once per class, so draw the unique ones
    mask = np.zeros((image["height"], image["width"]), bool)
    for x, y, w, h in {tuple(a["bbox"]) for a in annotations if a["iscrowd"]}:
        mask[int(y):math.ceil(y + h), int(x):math.ceil(x + w)] = True
    return mask


def export_yolo(coco, image_root, out, split, stride, max_side):
    # grey out ignore regions, drop boxes mostly inside them, shrink so the long side is max_side
    index = index_annotations(coco)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    def export(image):
        W, H = image["width"], image["height"]
        mask = ignore_mask(image, index[image["id"]])
        pixels = np.array(Image.open(image_root / image["file_name"]).convert("RGB"))
        pixels[mask] = 114
        picture = Image.fromarray(pixels)
        picture.thumbnail((max_side, max_side))
        name = f'{image["video"]}_{Path(image["file_name"]).stem}'
        picture.save(out / "images" / split / f"{name}.jpg", quality=95)
        lines = []
        for a in index[image["id"]]:
            x, y, w, h = a["bbox"]
            if a["iscrowd"] or mask[int(y):math.ceil(y + h), int(x):math.ceil(x + w)].mean() > 0.5:
                continue
            lines.append(f'{a["category_id"] - 1} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}')
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        list(pool.map(export, select_frames(coco, stride)))


def make_yolo_dataset(train_coco, val_coco, data, out, train_stride, val_stride, max_side):
    export_yolo(train_coco, data / "train", out, "train", train_stride, max_side)
    export_yolo(val_coco, data / "val", out, "val", val_stride, max_side)
    names = {c["id"] - 1: c["name"] for c in train_coco["categories"]}
    config = {"path": str(out), "train": "images/train", "val": "images/val", "names": names}
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return out / "data.yaml"
