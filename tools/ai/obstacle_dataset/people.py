"""Complete person labels on real photos (v4).

Real datasets often leave people unlabelled (LOCO has no person class at all),
and an unlabelled person teaches the model that a person is background. Two
detectors trained on other data look for people: the COCO-trained YOLO11m and
the open-vocabulary YOLO-World-L (asked for "person"; it found 94-98 % of the
people in the LOCO test against 82 % for our v3, see AI_REVIEW.md). Their
boxes are merged; a found person that overlaps no labelled person is added.

Both detectors are kept out of our own training data's origin: they only
label, they do not decide what our model sees.
"""
import numpy as np

CONF = 0.5            # each detector's own confidence threshold
MERGE_IOU = 0.6       # boxes of the two detectors closer than this are one person
NEW_IOU = 0.3         # a found person overlapping a labelled one less than this is new


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class PersonFinder:
    def __init__(self, coco='yolo11m.pt', world='yolov8l-worldv2.pt'):
        from ultralytics import YOLO
        self.coco = YOLO(coco)
        self.world = YOLO(world)
        self.world.set_classes(['person'])

    def find(self, image):
        """Person boxes (x0, y0, x1, y1, conf) in pixels, both detectors merged."""
        boxes = []
        for model, kw in ((self.coco, dict(classes=[0])), (self.world, {})):
            r = model.predict(image, conf=CONF, verbose=False, **kw)[0].boxes
            boxes += [(*b, c) for b, c in zip(r.xyxy.cpu().numpy().tolist(), r.conf.cpu().numpy().tolist())]
        boxes.sort(key=lambda b: -b[4])
        kept = []
        for b in boxes:
            if not kept or iou([b[:4]], [k[:4] for k in kept]).max() < MERGE_IOU:
                kept.append(b)
        return kept

    def missing(self, image, labelled):
        """Found people that overlap none of the `labelled` person boxes (x0, y0, x1, y1)."""
        found = self.find(image)
        if not labelled:
            return found
        return [f for f in found if iou([f[:4]], labelled).max() < NEW_IOU]
