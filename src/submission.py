import pandas as pd

# the grader rejects an empty PredictionString as null, so an image without boxes gets one near-zero-score box
EMPTY = "0 0.00001 0.00 0.00 1.00 1.00"


def write_submission(test, dets, path):
    # PredictionString: "class score x y w h ..." with top-left corner in original pixels
    rows = []
    for image_id in test.image_id:
        boxes, scores, classes = dets[image_id]
        rows.append(" ".join(f"{c} {s:.5f} {x1:.2f} {y1:.2f} {x2 - x1:.2f} {y2 - y1:.2f}"
                             for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), classes.tolist()))
                    or EMPTY)
    pd.DataFrame({"image_id": test.image_id, "PredictionString": rows}).to_csv(path, index=False)
