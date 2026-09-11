import numpy as np


def calculate_iou(boxA, boxB):
    """Klasyczna funkcja obliczająca Intersection over Union dla dwóch ramek."""
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
        """
        Dodaje wynik z pojedynczego zdjęcia do globalnych statystyk.
        pred_boxes: Lista ramek wygenerowanych przez model (po odcięciu przez próg Confidence!)
        gt_boxes: Lista ramek prawdziwych (może być pusta [])
        """
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

    def get_results(self):
        """Oblicza końcowe metryki i zwraca słownik z wynikami."""

        no_target_accuracy = 0.0
        if self.no_target_total > 0:
            no_target_accuracy = self.no_target_tn / self.no_target_total

        precision = 0.0
        recall = 0.0
        f1 = 0.0

        if (self.target_tp + self.target_fp) > 0:
            precision = self.target_tp / (self.target_tp + self.target_fp)

        if (self.target_tp + self.target_fn) > 0:
            recall = self.target_tp / (self.target_tp + self.target_fn)

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "No-Target Total": self.no_target_total,
            "No-Target Correct Refusals (TN%)": round(no_target_accuracy * 100, 2),
            "No-Target False Alarms (FP%)": round((1 - no_target_accuracy) * 100, 2),
            "Target Total Images": self.target_total_images,
            "Target Precision": round(precision, 4),
            "Target Recall": round(recall, 4),
            "Target F1-Score": round(f1, 4),
        }
