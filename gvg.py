import numpy as np


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


class GVGEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

        self.no_target_tn = 0
        self.no_target_fp = 0
        self.no_target_total = 0

        self.target_tp = 0
        self.target_fp = 0
        self.target_fn = 0
        self.target_total_images = 0

    def add_sample(self, pred_boxes, gt_boxes):
        if len(gt_boxes) == 0:
            self.no_target_total += 1
            if len(pred_boxes) == 0:
                self.no_target_tn += 1
            else:
                self.no_target_fp += 1
            return
        self.target_total_images += 1

        if len(pred_boxes) == 0:
            self.target_fn += len(gt_boxes)
            return

        matched_gt_indices = set()
        local_tp = 0
        local_fp = 0

        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt_indices:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= self.iou_threshold:
                local_tp += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                local_fp += 1

        local_fn = len(gt_boxes) - len(matched_gt_indices)

        self.target_tp += local_tp
        self.target_fp += local_fp
        self.target_fn += local_fn
