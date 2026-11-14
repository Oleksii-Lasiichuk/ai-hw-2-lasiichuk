import json
import math
import os
from collections import Counter, defaultdict
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


def split_clips(train_coco, val_coco, fold=None, folds=5):
    # file names become relative to the data root; with a fold, all clips are pooled and that fold's clips are held out
    rooted = lambda coco, split, shift: ([{**i, "id": i["id"] + shift, "file_name": f'{split}/{i["file_name"]}'} for i in coco["images"]],
                                         [{**a, "id": a["id"] + shift, "image_id": a["image_id"] + shift} for a in coco["annotations"]])
    if fold is None:
        return [{**coco, **dict(zip(["images", "annotations"], rooted(coco, split, 0)))}
                for coco, split in [(train_coco, "train"), (val_coco, "val")]]
    shift = max(i["id"] for i in train_coco["images"]) + 1  # validation ids move past the training ids
    shift = max(shift, max(a["id"] for a in train_coco["annotations"]) + 1)
    (ti, ta), (vi, va) = rooted(train_coco, "train", 0), rooted(val_coco, "val", shift)
    images, annotations = ti + vi, ta + va
    frames = Counter(i["video"] for i in images)
    sizes, assigned = [0] * folds, {}
    for clip in sorted(frames, key=lambda c: (-frames[c], c)):  # largest clips first, each to the smallest fold
        k = int(np.argmin(sizes))
        assigned[clip], sizes[k] = k, sizes[k] + frames[clip]
    held = {i["id"] for i in images if assigned[i["video"]] == fold}
    pick = lambda keep: {**train_coco, "images": [i for i in images if (i["id"] in held) == keep],
                         "annotations": [a for a in annotations if (a["image_id"] in held) == keep]}
    return pick(False), pick(True)


def tile_origins(width, height, tile):
    # crops (x, y, w, h) of tile px along every side longer than 1.25 tiles, with at least 20% overlap;
    # a shorter side stays whole, and a frame short on both sides gets no crops
    def spans(size):
        if size <= 1.25 * tile:
            return [(0, size)]
        return [(int(s), tile) for s in np.linspace(0, size - tile, math.ceil((size - tile) / (0.8 * tile)) + 1)]
    xs, ys = spans(width), spans(height)
    return [] if len(xs) == len(ys) == 1 else [(x, y, w, h) for y, h in ys for x, w in xs]


def clip_order(images):
    # file name -> (file names of its clip ordered by frame, its position there)
    clips = defaultdict(list)
    for image in sorted(images, key=lambda i: i["frame_id"]):
        clips[image["video"]].append(image["file_name"])
    return {f: (files, t) for files in clips.values() for t, f in enumerate(files)}


def read_frame(root, image, order=None, temporal=0):
    # RGB pixels, or with temporal = k the grey frames t-k, t, t+k of the same clip as the three channels
    if not temporal:
        return np.array(Image.open(root / image["file_name"]).convert("RGB"))
    files, t = order[image["file_name"]]
    return np.dstack([np.array(Image.open(root / files[min(max(t + d, 0), len(files) - 1)]).convert("L"))
                      for d in (-temporal, 0, temporal)])


def write_frames(images, root, out, temporal):
    # temporal inputs for prediction, same file names under out; 4:4:4 JPEG keeps the three frames apart
    order = clip_order(images)

    def write(image):
        (out / image["file_name"]).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(read_frame(root, image, order, temporal)).save(out / image["file_name"], quality=95, subsampling=0)

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        list(pool.map(write, images))


def export_yolo(coco, image_root, out, split, stride, max_side, tile=None, temporal=0):
    # grey out ignore regions, drop boxes mostly inside them, shrink so the long side is max_side;
    # tile also saves crops at original resolution (boxes kept if at least half visible), temporal as in read_frame
    index = index_annotations(coco)
    order = clip_order(coco["images"]) if temporal else None
    options = dict(quality=95, subsampling=0) if temporal else dict(quality=95)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    def save(name, pixels, boxes, crop):
        x0, y0, w0, h0 = crop
        picture = Image.fromarray(pixels[y0:y0 + h0, x0:x0 + w0])
        picture.thumbnail((max_side, max_side))
        picture.save(out / "images" / split / f"{name}.jpg", **options)
        lines = []
        for c, x, y, w, h in boxes:
            if (w0, h0) != (pixels.shape[1], pixels.shape[0]):
                x1, y1, x2, y2 = max(x, x0), max(y, y0), min(x + w, x0 + w0), min(y + h, y0 + h0)
                if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 0.5 * w * h:
                    continue
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
            lines.append(f"{c} {(x - x0 + w / 2) / w0:.6f} {(y - y0 + h / 2) / h0:.6f} {w / w0:.6f} {h / h0:.6f}")
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))

    def export(image):
        W, H = image["width"], image["height"]
        mask = ignore_mask(image, index[image["id"]])
        pixels = read_frame(image_root, image, order, temporal)
        pixels[mask] = 114
        boxes = []
        for a in index[image["id"]]:
            x, y, w, h = a["bbox"]
            if not a["iscrowd"] and not mask[int(y):math.ceil(y + h), int(x):math.ceil(x + w)].mean() > 0.5:
                boxes.append((a["category_id"] - 1, x, y, w, h))
        name = f'{image["video"]}_{Path(image["file_name"]).stem}'
        save(name, pixels, boxes, (0, 0, W, H))
        for x, y, w, h in tile_origins(W, H, tile) if tile else []:
            save(f"{name}_{x}_{y}", pixels, boxes, (x, y, w, h))

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        list(pool.map(export, select_frames(coco, stride)))


def make_yolo_dataset(train_coco, val_coco, root, out, train_stride, val_stride, max_side, tile=None, temporal=0):
    # file names relative to root (see split_clips); validation frames are never tiled
    export_yolo(train_coco, root, out, "train", train_stride, max_side, tile, temporal)
    export_yolo(val_coco, root, out, "val", val_stride, max_side, None, temporal)
    names = {c["id"] - 1: c["name"] for c in train_coco["categories"]}
    config = {"path": str(out), "train": "images/train", "val": "images/val", "names": names}
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return out / "data.yaml"
