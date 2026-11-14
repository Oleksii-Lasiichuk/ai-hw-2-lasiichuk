import gc
import json

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

EMPTY = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))


def to_dets(df):
    # columns image_id, c (0-7), s, x, y, w, h -> {image_id: (xyxy, scores, classes)}; boxes under 1 px dropped
    df = df[(df.w >= 1) & (df.h >= 1)]
    return {i: (np.c_[g.x, g.y, g.x + g.w, g.y + g.h], g.s.to_numpy(), g.c.to_numpy()) for i, g in df.groupby("image_id")}


def read_val(path, keep=lambda i: True, offset=0):
    # val_predictions.json; keep and offset select and renumber the validation frames of a fold run (see split_clips)
    preds = json.loads(path.read_text())
    df = pd.DataFrame([(p["image_id"] - offset, p["category_id"] - 1, p["score"], *p["bbox"]) for p in preds if keep(p["image_id"])],
                      columns=["image_id", "c", "s", "x", "y", "w", "h"])
    del preds
    gc.collect()
    return to_dets(df)


def read_test(path):
    # submission.csv rows "class score x y w h ..." per image
    rows = []
    for image_id, s in pd.read_csv(path).itertuples(index=False):
        a = np.array(s.split(), float).reshape(-1, 6)
        rows.append(pd.DataFrame({"image_id": image_id, "c": a[:, 0].astype(int), "s": a[:, 1], "x": a[:, 2], "y": a[:, 3], "w": a[:, 4], "h": a[:, 5]}))
    return to_dets(pd.concat(rows))


def fuse(parts, size, weights=None, iou=0.7):
    # weighted boxes fusion per image on boxes normalised to [0, 1]; the 500 best fused boxes are kept
    out = {}
    for i, (w, h) in size.items():
        scale = np.array([w, h, w, h])
        ps = [p.get(i, EMPTY) for p in parts]
        boxes, scores, classes = weighted_boxes_fusion([np.clip(p[0] / scale, 0, 1) for p in ps], [p[1] for p in ps], [p[2] for p in ps],
                                                       weights=weights, iou_thr=iou, skip_box_thr=0.001)
        top = np.argsort(-scores)[:500]
        out[i] = boxes[top] * scale, scores[top], classes[top].astype(int)
    return out


def box_iou(a, b):
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def rescore(dets, clips, k=2, a=0.2, iou=0.5):
    # temporal rescoring: mix each score with the mean score of its best same-class match (IoU >= iou)
    # in each of the k frames before and after it, 0 where there is none; boxes are not moved, added or removed
    out = {}
    for ids in clips:
        for t, i in enumerate(ids):
            boxes, scores, classes = dets.get(i, EMPTY)
            support = []
            for n in ids[max(t - k, 0):t] + ids[t + 1:t + 1 + k]:
                nb, ns, nc = dets.get(n, EMPTY)
                if len(nb) == 0:
                    support.append(np.zeros(len(boxes)))
                    continue
                o = box_iou(boxes, nb) * (classes[:, None] == nc[None])
                support.append(np.where(o.max(1) >= iou, ns[o.argmax(1)], 0))
            out[i] = boxes, (1 - a) * scores + a * np.mean(support, 0), classes
    return out
