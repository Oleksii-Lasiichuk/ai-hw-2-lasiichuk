import functools

import numpy as np
import torch
from ultralytics.utils import nms

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
