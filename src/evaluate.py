import copy

from faster_coco_eval import COCO, COCOeval_faster


def to_coco(dets):
    return [{"image_id": image_id, "category_id": int(c) + 1, "score": round(float(s), 5),
             "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]}
            for image_id, (boxes, scores, classes) in dets.items()
            for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores, classes)]


# COCO metrics with ignore regions as iscrowd: https://cocodataset.org/#detection-eval
def evaluate(coco, dets):
    gt = COCO(copy.deepcopy(coco))
    ev = COCOeval_faster(gt, gt.loadRes(to_coco(dets)), "bbox")
    ev.params.imgIds = sorted(dets)
    ev.params.maxDets = [1, 10, 500]
    ev.evaluate()
    ev.accumulate()
    p = ev.eval["precision"][..., -1]  # [iou, recall, class, area] at maxDets=500
    ap = lambda x: float(x[x > -1].mean())
    return {"mAP": ap(p[..., 0]), "AP50": ap(p[0, ..., 0]),
            "AP_small": ap(p[..., 1]), "AP_medium": ap(p[..., 2]), "AP_large": ap(p[..., 3])}
