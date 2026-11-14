import functools
import time

import cv2
import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.ops import batched_nms
from ultralytics.utils import nms

from src.data import tile_origins

# Ultralytics stops NMS after 2 s + 0.05 s/img and returns no boxes for the rest of the batch;
# 10k candidates instead of 30k give identical boxes on the densest val frames (149 objects)
nms.non_max_suppression = functools.partial(nms.non_max_suppression, max_time_img=1e3, max_nms=10000)


def predict_yolo(model, images, imgsz, batch=16):
    # images: [(image_id, path)] -> {image_id: (xyxy, scores, classes)} in original pixels, mean ms per image
    # Ultralytics decodes a whole list of paths up front, so it gets one batch at a time
    dets, ms = {}, []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        results = model.predict([str(p) for _, p in chunk], imgsz=imgsz, conf=0.001, iou=0.7, max_det=500,
                                half=torch.cuda.is_available(), verbose=False)
        for (image_id, _), r in zip(chunk, results):
            boxes = r.boxes.xyxy.cpu().numpy()
            keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
            dets[image_id] = boxes[keep], r.boxes.conf.cpu().numpy()[keep], r.boxes.cls.cpu().numpy().astype(int)[keep]
            ms.append(sum(r.speed.values()))
    return dets, float(np.mean(ms))


@torch.no_grad()
def predict_tiled(model, images, imgsz, tile):
    # same output as predict_yolo from the whole frame plus its original-resolution crops (tile_origins);
    # crop boxes touching an inner crop edge belong to cut objects and are dropped, the rest go through class-wise NMS;
    # model is an Ultralytics model or a torchvision detector (imgsz unused, the model resizes itself)
    dets, ms = {}, []
    for image_id, path in images:
        frame = cv2.imread(str(path))  # BGR, as Ultralytics reads files
        H, W = frame.shape[:2]
        crops = [(0, 0, W, H)] + tile_origins(W, H, tile)
        if hasattr(model, "predict"):
            out = model.predict([np.ascontiguousarray(frame[y:y + h, x:x + w]) for x, y, w, h in crops], imgsz=imgsz,
                                conf=0.001, iou=0.7, max_det=500, half=torch.cuda.is_available(), verbose=False)
            results = [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()) for r in out]
            ms.append(sum(sum(r.speed.values()) for r in out))
        else:
            model.eval()
            device, start = next(model.parameters()).device, time.time()
            pixels = torch.from_numpy(frame[..., ::-1].copy()).to(device).permute(2, 0, 1).float() / 255
            with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
                out = model([pixels[:, y:y + h, x:x + w] for x, y, w, h in crops])
            results = [(r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(), r["labels"].cpu().numpy() - 1) for r in out]
            ms.append((time.time() - start) * 1000)
        parts = []
        for (x, y, w, h), (b, s, c) in zip(crops, results):
            whole = ((b[:, 0] > 2) | (x == 0)) & ((b[:, 1] > 2) | (y == 0)) & ((b[:, 2] < w - 2) | (x + w == W)) & ((b[:, 3] < h - 2) | (y + h == H))
            parts.append(np.c_[b[whole] + [x, y, x, y], s[whole], c[whole]])
        a = np.concatenate(parts)
        a = a[(a[:, 2] - a[:, 0] >= 1) & (a[:, 3] - a[:, 1] >= 1)]
        keep = batched_nms(torch.from_numpy(a[:, :4]).float(), torch.from_numpy(a[:, 4]).float(), torch.from_numpy(a[:, 5]).long(), 0.6)
        a = a[keep[:500].numpy()]
        dets[image_id] = a[:, :4], a[:, 4], a[:, 5].astype(int)
    return dets, float(np.mean(ms))


def load_image(path, device):
    # JPEG decoded on the GPU when there is one; float CHW in [0, 1]
    decoder = "cuda" if device == "cuda" else "cpu"
    return decode_jpeg(read_file(str(path)), mode=ImageReadMode.RGB, device=decoder).to(device).float() / 255


@torch.no_grad()
def predict_torchvision(model, images, device, batch=4):
    # same output as predict_yolo; torchvision rescales boxes to original pixels, labels 1-8 become 0-7
    model.eval()
    dets, ms = {}, []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        pixels = [load_image(p, device) for _, p in chunk]
        start = time.time()
        with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            results = model(pixels)
        if device == "cuda":
            torch.cuda.synchronize()
        ms.append((time.time() - start) * 1000 / len(chunk))
        for (image_id, _), r in zip(chunk, results):
            dets[image_id] = r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(), r["labels"].cpu().numpy() - 1
    return dets, float(np.mean(ms))
