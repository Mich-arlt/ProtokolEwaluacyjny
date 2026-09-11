import os
import csv
import json
import pickle
import torch
import argparse
import pandas as pd
import datetime
import gc
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from transformers import AutoProcessor, OmDetTurboForObjectDetection
import time

COCO_IMG_DIR = "train2014/train2014"
COCO_2017_IMG_DIR = "lvis/val2017"
COCO_INSTANCES_JSON = "annotations_trainval2014/annotations/instances_train2014.json"
REFCOCO_PKL = "refcoco/refs(unc).p"
REFCOCO_PLUS_PKL = "refcoco+/refcoco+/refcoco+/refs(unc).p"
REFCOCOG_PKL = "refcocog/refcocog/refs(umd).p"
GREFCOCO_JSON = "grefcoco/grefs(unc).json"

REFERIT_IMG_DIR = "referit/referit/images"
REFERIT_INSTANCES_JSON = "referit/referit/instances.json"
REFERIT_PKL = "referit/referit/refs(berkeley).p"

FLICKR_IMG_DIR = "flickr/flickr/flickr30k_images/flickr30k_images"
FLICKR_ANN_JSON = "flickr/flickr/annotations.json"

VG_IMG_DIR = "visual_genome"
VG_REGIONS_JSON = "visual_genome/region_descriptions.json"
LVIS_ANN_JSON = "lvis/lvis_v1_val.json"


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
        local_tp, local_fp = 0, 0

        for pred_box in pred_boxes:
            best_iou, best_gt_idx = 0, -1
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt_indices:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, i

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
        no_target_accuracy = (
            self.no_target_tn / self.no_target_total
            if self.no_target_total > 0
            else 0.0
        )
        precision = (
            self.target_tp / (self.target_tp + self.target_fp)
            if (self.target_tp + self.target_fp) > 0
            else 0.0
        )
        recall = (
            self.target_tp / (self.target_tp + self.target_fn)
            if (self.target_tp + self.target_fn) > 0
            else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "No-Target Total": self.no_target_total,
            "No-Target Correct Refusals (TN%)": round(no_target_accuracy * 100, 2),
            "No-Target False Alarms (FP%)": round((1 - no_target_accuracy) * 100, 2),
            "Target Total Images": self.target_total_images,
            "Target Precision": round(precision, 4),
            "Target Recall": round(recall, 4),
            "Target F1-Score": round(f1, 4),
        }


def generuj_wykresy_zasobow(
    wszystkie_historie, dataset_name, conf_thresh, folder_wykresow
):
    import seaborn as sns

    if len(wszystkie_historie) < 1:
        return

    print("\n[RYSOWANIE] Generowanie wykresów zasobożerności (VRAM i FPS)...")

    dane_wykres = []
    for model_name, historia in wszystkie_historie.items():
        if not historia:
            continue

        if "FPS" not in historia[0] or "VRAM_MB" not in historia[0]:
            print(
                f"[INFO] Pomijam {model_name} na wykresach zasobów (brak danych FPS/VRAM)."
            )
            continue

        avg_fps = sum(w["FPS"] for w in historia) / len(historia)
        max_vram = max(w["VRAM_MB"] for w in historia)

        dane_wykres.append(
            {
                "Model": model_name.upper(),
                "Średnie FPS": avg_fps,
                "Szczytowy VRAM (MB)": max_vram,
            }
        )

    if not dane_wykres:
        return

    df = pd.DataFrame(dane_wykres)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(data=df, x="Model", y="Średnie FPS", palette="Blues_d")
    plt.title(
        f"Średnia szybkość inferencji (FPS) - {dataset_name.upper()}",
        fontsize=14,
        pad=15,
    )
    plt.ylabel("Klatki na sekundę (FPS)", fontsize=12)
    plt.xlabel("Model", fontsize=12)

    for p in ax1.patches:
        ax1.annotate(
            f"{p.get_height():.1f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            folder_wykresow, f"zasoby_FPS_{dataset_name}_conf{conf_thresh}.png"
        ),
        dpi=300,
    )
    plt.close()
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(data=df, x="Model", y="Szczytowy VRAM (MB)", palette="Reds_d")
    plt.title(
        f"Szczytowe zużycie pamięci VRAM - {dataset_name.upper()}", fontsize=14, pad=15
    )
    plt.ylabel("VRAM (Megabajty)", fontsize=12)
    plt.xlabel("Model", fontsize=12)

    for p in ax2.patches:
        ax2.annotate(
            f"{p.get_height():.0f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            folder_wykresow, f"zasoby_VRAM_{dataset_name}_conf{conf_thresh}.png"
        ),
        dpi=300,
    )
    plt.close()

    print(f"[SUKCES] Wykresy zasobów (FPS i VRAM) zapisane w: {folder_wykresow}/")


def generuj_wykresy_percentylowe(
    wszystkie_historie, dataset_name, conf_thresh, folder_wykresow
):
    metryki = [
        "IoU_Top1",
        "IoU_Top5",
        "Top1_Acc_0.5",
        "Top5_Acc_0.5",
        "Top1_Acc_0.7",
        "Top5_Acc_0.7",
        "Top1_Acc_0.9",
        "Top5_Acc_0.9",
    ]

    print(
        f"\n[RYSOWANIE] Generowanie {len(metryki)} wykresów percentylowych do folderu '{folder_wykresow}'..."
    )

    percentiles = np.arange(0, 101, 1)

    for metryka in metryki:
        plt.figure(figsize=(10, 6))

        dane_znalezione = False
        for model_name, historia in wszystkie_historie.items():
            if not historia:
                continue

            if metryka not in historia[0]:
                continue

            dane_znalezione = True
            wartosci = [w[metryka] for w in historia]

            p_values = np.percentile(wartosci, percentiles)

            plt.plot(percentiles, p_values, linewidth=2.5, label=model_name.upper())

        if not dane_znalezione:
            plt.close()
            continue

        plt.title(
            f"Wykres percentylowy {metryka} - {dataset_name.upper()} (Conf: {conf_thresh})"
        )
        plt.xlabel("Percentyl (%)")

        if "Acc" in metryka:
            plt.ylabel("Skuteczność (%)")
            plt.ylim(0, 105)
        else:
            plt.ylabel("Wartość IoU")
            plt.ylim(0, 1.05)

        plt.xlim(0, 100)

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        nazwa_pliku = os.path.join(
            folder_wykresow,
            f"percentyle_{metryka}_{dataset_name}_conf{conf_thresh}.png",
        )
        plt.savefig(nazwa_pliku)
        plt.close()

    print(
        f"[SUKCES] Wykresy percentylowe bezpiecznie wylądowały w folderze: {folder_wykresow}/"
    )


def generuj_wykresy_gvg(folder_excel, dataset_name, conf_thresh, folder_wykresow):
    csv_file = os.path.join(folder_excel, "wyniki_gvg_master.csv")
    if not os.path.exists(csv_file):
        print(f"[INFO] Brak pliku {csv_file}. Pomijam rysowanie wykresów GVG.")
        return

    df = pd.read_csv(csv_file)
    df_filtered = df[(df["Zbiór"] == dataset_name) & (df["Conf_Thresh"] == conf_thresh)]

    if df_filtered.empty:
        print(
            f"[INFO] Brak danych GVG dla zbioru {dataset_name} i progu {conf_thresh}."
        )
        return

    print(
        f"\n[RYSOWANIE] Generowanie wykresów słupkowych GVG do folderu '{folder_wykresow}'..."
    )
    sns.set_theme(style="whitegrid")

    metryki_gvg = [
        ("No-Target Correct Refusals (TN%)", "Poprawne odmowy (%)", 105),
        ("Target F1-Score", "Wartość F1-Score", 1.05),
        ("Target Precision", "Precyzja", 1.05),
        ("Target Recall", "Czułość", 1.05),
    ]

    for kolumna, os_y, limit in metryki_gvg:
        if kolumna not in df_filtered.columns:
            continue
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_filtered, x="Model", y=kolumna, hue="Split", palette="viridis"
        )
        plt.title(
            f"GVG: {kolumna} - {dataset_name.upper()} (Conf: {conf_thresh})",
            fontsize=14,
        )
        plt.xlabel("Model", fontsize=12)
        plt.ylabel(os_y, fontsize=12)
        plt.ylim(0, limit)
        plt.legend(title="Split")
        plt.tight_layout()

        bezpieczna_nazwa = (
            kolumna.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "proc")
        )
        plt.savefig(
            os.path.join(
                folder_wykresow,
                f"gvg_{bezpieczna_nazwa}_{dataset_name}_conf{conf_thresh}.png",
            ),
            dpi=300,
        )
        plt.close()


def calculate_iou(box1, box2):
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / (box1_area + box2_area - intersection_area)


def calculate_max_iou_for_multiple_targets(pred_box, gt_boxes):
    if not gt_boxes:
        return 0.0
    return max([calculate_iou(pred_box, gt) for gt in gt_boxes])


def load_dataset(dataset_name, target_split):
    dataset = []
    print(f"-> Rozpoczynam parsowanie zbioru: {dataset_name.upper()}")

    if dataset_name in ["refcoco", "refcoco+", "refcocog"]:
        with open(COCO_INSTANCES_JSON, "r") as f:
            coco_data = json.load(f)

        img_id_to_filename = {
            img["id"]: img["file_name"] for img in coco_data["images"]
        }
        ann_id_to_bbox = {ann["id"]: ann["bbox"] for ann in coco_data["annotations"]}

        if dataset_name == "refcoco":
            pkl_path = REFCOCO_PKL
        elif dataset_name == "refcoco+":
            pkl_path = REFCOCO_PLUS_PKL
        else:
            pkl_path = REFCOCOG_PKL

        with open(pkl_path, "rb") as f:
            refs_data = pickle.load(f)

        for ref in refs_data:
            if ref["split"] != target_split:
                continue
            img_id = ref["image_id"]
            ann_id = ref["ann_id"]
            if img_id not in img_id_to_filename or ann_id not in ann_id_to_bbox:
                continue
            img_path = os.path.join(COCO_IMG_DIR, img_id_to_filename[img_id])
            x, y, w, h = ann_id_to_bbox[ann_id]
            for sentence in ref["sentences"]:
                dataset.append(
                    {
                        "image_path": img_path,
                        "prompt": sentence["sent"],
                        "gt_boxes": [[x, y, x + w, y + h]],
                    }
                )

    elif dataset_name == "referit":
        with open(REFERIT_INSTANCES_JSON, "r") as f:
            coco_data = json.load(f)
        img_id_to_filename = {
            img["id"]: img["file_name"] for img in coco_data["images"]
        }
        ann_id_to_bbox = {ann["id"]: ann["bbox"] for ann in coco_data["annotations"]}
        with open(REFERIT_PKL, "rb") as f:
            refs_data = pickle.load(f)
        for ref in refs_data:
            if ref["split"] != target_split:
                continue
            img_id, ann_id = ref["image_id"], ref["ann_id"]
            if img_id not in img_id_to_filename or ann_id not in ann_id_to_bbox:
                continue
            img_path = os.path.join(REFERIT_IMG_DIR, img_id_to_filename[img_id])
            x, y, w, h = ann_id_to_bbox[ann_id]
            for sentence in ref["sentences"]:
                dataset.append(
                    {
                        "image_path": img_path,
                        "prompt": sentence["sent"],
                        "gt_boxes": [[x, y, x + w, y + h]],
                    }
                )

    elif dataset_name == "grefcoco":
        with open(GREFCOCO_JSON, "r") as f:
            grefs_data = json.load(f)
        with open(COCO_INSTANCES_JSON, "r") as f:
            coco_data = json.load(f)
        img_id_to_filename = {
            img["id"]: img["file_name"] for img in coco_data["images"]
        }
        ann_id_to_bbox = {ann["id"]: ann["bbox"] for ann in coco_data["annotations"]}

        for gref in grefs_data:
            if gref.get("split", "val") != target_split:
                continue

            img_id = gref["image_id"]
            if img_id not in img_id_to_filename:
                continue

            img_path = os.path.join(COCO_IMG_DIR, img_id_to_filename[img_id])
            formatted_boxes = []
            for ann_id in gref.get("ann_id", []):
                if ann_id in ann_id_to_bbox:
                    b = ann_id_to_bbox[ann_id]
                    formatted_boxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])

            for sentence in gref["sentences"]:
                dataset.append(
                    {
                        "image_path": img_path,
                        "prompt": sentence["sent"],
                        "gt_boxes": formatted_boxes,
                    }
                )

    elif dataset_name == "flickr30k":
        with open(FLICKR_ANN_JSON, "r", encoding="utf-8") as f:
            flickr_data = json.load(f)
        for item in flickr_data:
            if item.get("split", "val") != target_split:
                continue
            img_path = os.path.join(FLICKR_IMG_DIR, item["image_name"])
            full_sentence = item["prompt"]
            for phrase_data in item.get("phrases", []):
                dataset.append(
                    {
                        "image_path": img_path,
                        "prompt": phrase_data["phrase_text"],
                        "full_sentence": full_sentence,
                        "gt_boxes": phrase_data["bboxes"],
                        "is_flickr": True,
                    }
                )

    elif dataset_name == "lvis":
        with open(LVIS_ANN_JSON, "r", encoding="utf-8") as f:
            lvis_data = json.load(f)
        img_id_to_filename = {
            img["id"]: f"{img['id']:012d}.jpg" for img in lvis_data["images"]
        }
        cat_id_to_name = {c["id"]: c["name"] for c in lvis_data["categories"]}

        grouped_anns = {}
        for ann in lvis_data["annotations"]:
            key = (ann["image_id"], ann["category_id"])
            if key not in grouped_anns:
                grouped_anns[key] = []
            x, y, w, h = ann["bbox"]
            grouped_anns[key].append([x, y, x + w, y + h])

        for (img_id, cat_id), boxes in grouped_anns.items():
            if img_id not in img_id_to_filename:
                continue
            img_path = os.path.join(COCO_2017_IMG_DIR, img_id_to_filename[img_id])
            if not os.path.exists(img_path):
                continue
            dataset.append(
                {
                    "image_path": img_path,
                    "prompt": cat_id_to_name[cat_id],
                    "gt_boxes": boxes,
                }
            )

    elif dataset_name == "visual_genome":
        with open(VG_REGIONS_JSON, "r") as f:
            vg_data = json.load(f)
        base_vg_dir = os.path.dirname(VG_REGIONS_JSON)
        img_dir_1 = os.path.join(base_vg_dir, "VG_100K")
        img_dir_2 = os.path.join(base_vg_dir, "VG_100K_2")

        for img_entry in vg_data:
            img_id = img_entry.get("image_id") or img_entry.get("id")
            path_1 = os.path.join(img_dir_1, f"{img_id}.jpg")
            path_2 = os.path.join(img_dir_2, f"{img_id}.jpg")

            if os.path.exists(path_1):
                img_path = path_1
            elif os.path.exists(path_2):
                img_path = path_2
            else:
                continue

            for region in img_entry["regions"][:5]:
                dataset.append(
                    {
                        "image_path": img_path,
                        "prompt": region["phrase"],
                        "gt_boxes": [
                            [
                                region["x"],
                                region["y"],
                                region["x"] + region["width"],
                                region["y"] + region["height"],
                            ]
                        ],
                        "dataset": "visual_genome",
                    }
                )

    MAX_SAMPLES = 10000
    if len(dataset) > MAX_SAMPLES:
        print(
            f"-> UWAGA: Ocinam {len(dataset)} celów do bezpiecznego limitu: {MAX_SAMPLES}"
        )
        dataset = dataset[:MAX_SAMPLES]

    print(f"-> Parsowanie zakończone. Gotowych próbek do testu: {len(dataset)}")
    return dataset


def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()


def run_evaluation_omdet(
    dataset,
    device,
    dataset_name,
    split_name,
    conf_thresh,
    folder_excel,
    glowny_folder_bledow,
    glowny_folder_sukcesow,
):
    total_samples = len(dataset)
    if total_samples == 0:
        return []

    total_iou_top1 = 0.0
    t1_at_05, t1_at_07, t1_at_09, t5_at_05, t5_at_07, t5_at_09 = 0, 0, 0, 0, 0, 0
    historia_wynikow = []

    nazwa_pliku_excel = os.path.join(
        folder_excel, f"wyniki_omdet_{dataset_name}_{split_name}_conf{conf_thresh}.xlsx"
    )
    katalog_bledow = os.path.join(
        glowny_folder_bledow, f"bledy_omdet_{dataset_name}_{split_name}"
    )
    os.makedirs(katalog_bledow, exist_ok=True)

    katalog_sukcesow = os.path.join(
        glowny_folder_sukcesow, f"sukcesy_omdet_{dataset_name}_{split_name}"
    )
    os.makedirs(katalog_sukcesow, exist_ok=True)

    zapisane_bledy = 0
    zapisane_sukcesy = 0

    is_grefcoco = dataset_name.lower() == "grefcoco"
    if is_grefcoco:
        gvg_metrics = GVGEvaluator(iou_threshold=0.5)

    print(
        f"\n{'='*80}\n MODEL: OMDET-TURBO | ZBIÓR: {dataset_name.upper()} ({split_name}) | PRÓBKI: {total_samples} | CONF: {conf_thresh}\n{'='*80}"
    )

    print("-> Ładowanie modelu OmDet-Turbo (Wersja Natywna!)...")
    processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
    model = OmDetTurboForObjectDetection.from_pretrained(
        "omlab/omdet-turbo-swin-tiny-hf"
    ).to(device)
    print("-> Model załadowany!")

    for i, data in enumerate(dataset):
        img_path, prompt, gt_boxes = (
            data["image_path"],
            data["prompt"],
            data["gt_boxes"],
        )

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        best_iou_top1, best_iou_top5 = 0.0, 0.0
        boxes = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        start_time = time.perf_counter()
        try:
            klasy = [prompt]
            inputs = processor(images=image, text=klasy, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                text_labels=klasy,
                target_sizes=[image.size[::-1]],
                threshold=conf_thresh,
                nms_threshold=0.3,
            )[0]

            all_boxes = results["boxes"].cpu().numpy()
            all_scores = results["scores"].cpu().numpy()

            if len(all_boxes) > 0:
                boxes = all_boxes[(-all_scores).argsort()]

        except Exception as e:
            pass

        end_time = time.perf_counter()

        inference_time = end_time - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0.0

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        if is_grefcoco:
            gvg_metrics.add_sample(pred_boxes=boxes, gt_boxes=gt_boxes)

        if len(boxes) > 0:
            best_iou_top1 = calculate_max_iou_for_multiple_targets(boxes[0], gt_boxes)
            best_iou_top5 = max(
                [
                    calculate_max_iou_for_multiple_targets(box, gt_boxes)
                    for box in boxes[:5]
                ]
            )

            if best_iou_top1 == 0.0 and zapisane_bledy < 10:
                try:
                    img_error = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(img_error)
                    for gt in gt_boxes:
                        draw.rectangle(gt, outline="green", width=5)
                    draw.rectangle(boxes[0], outline="red", width=5)
                    bezpieczny_prompt = "".join(
                        [c for c in prompt if c.isalpha() or c.isdigit() or c == " "]
                    ).rstrip()[:30]
                    nazwa_obrazka = os.path.join(
                        katalog_bledow,
                        f"blad_{zapisane_bledy+1}_{bezpieczny_prompt}.jpg",
                    )
                    img_error.save(nazwa_obrazka)
                    zapisane_bledy += 1
                except Exception:
                    pass

            elif best_iou_top1 >= 0.95 and zapisane_sukcesy < 20:
                try:
                    img_success = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(img_success)

                    for gt in gt_boxes:
                        draw.rectangle(gt, outline="green", width=5)

                    for box in boxes:
                        draw.rectangle(box, outline="blue", width=3)

                    bezpieczny_prompt = "".join(
                        [c for c in prompt if c.isalpha() or c.isdigit() or c == " "]
                    ).rstrip()[:30]

                    nazwa_obrazka = os.path.join(
                        katalog_sukcesow,
                        f"sukces_{zapisane_sukcesy+1}_{bezpieczny_prompt}.jpg",
                    )
                    img_success.save(nazwa_obrazka)
                    zapisane_sukcesy += 1
                except Exception as e:
                    pass
            if zapisane_sukcesy == 20:
                break

        total_iou_top1 += best_iou_top1
        if best_iou_top1 >= 0.5:
            t1_at_05 += 1
        if best_iou_top1 >= 0.7:
            t1_at_07 += 1
        if best_iou_top1 >= 0.9:
            t1_at_09 += 1
        if best_iou_top5 >= 0.5:
            t5_at_05 += 1
        if best_iou_top5 >= 0.7:
            t5_at_07 += 1
        if best_iou_top5 >= 0.9:
            t5_at_09 += 1

        historia_wynikow.append(
            {
                "Krok": i + 1,
                "IoU_Top1": best_iou_top1,
                "IoU_Top5": best_iou_top5,
                "Czas_Inferencji_s": inference_time,
                "FPS": fps,
                "VRAM_MB": vram_mb,
                "Top1_Acc_0.5": (t1_at_05 / (i + 1)) * 100,
                "Top5_Acc_0.5": (t5_at_05 / (i + 1)) * 100,
                "Top1_Acc_0.7": (t1_at_07 / (i + 1)) * 100,
                "Top5_Acc_0.7": (t5_at_07 / (i + 1)) * 100,
                "Top1_Acc_0.9": (t1_at_09 / (i + 1)) * 100,
                "Top5_Acc_0.9": (t5_at_09 / (i + 1)) * 100,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == total_samples:
            current_miou_t1 = total_iou_top1 / (i + 1)

            print(
                f"[OMDET] Krok {i + 1}/{total_samples} |IoU(T1): {best_iou_top1:.3f} | T1@0.5: {(t1_at_05/(i+1))*100:.1f}% | T5@0.5: {(t5_at_05/(i+1))*100:.1f}%"
            )
            pd.DataFrame(historia_wynikow).to_excel(nazwa_pliku_excel, index=False)

    print(f"\n[ZAKOŃCZONO] Zapisano: {nazwa_pliku_excel}")

    if is_grefcoco:
        raport = gvg_metrics.get_results()
        raport_z_tagami = {
            "Model": "omdet",
            "Zbiór": dataset_name,
            "Split": split_name,
            "Conf_Thresh": conf_thresh,
        }
        raport_z_tagami.update(raport)

        csv_file = os.path.join(folder_excel, "wyniki_gvg_master.csv")
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=raport_z_tagami.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(raport_z_tagami)

        print(f"[GVG] Zapisano raport z halucynacjami i F1 do: {csv_file}")

    del model, processor
    clear_vram()
    return historia_wynikow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "refcoco+",
            "refcoco",
            "refcocog",
            "referit",
            "grefcoco",
            "flickr30k",
            "visual_genome",
            "lvis",
        ],
        default="grefcoco",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    VALID_SPLITS = {
        "refcoco+": ["train", "val", "testA", "testB"],
        "refcoco": ["train", "val", "testA", "testB"],
        "refcocog": ["train", "val", "test"],
        "referit": ["train", "val", "test", "trainval"],
        "grefcoco": ["train", "val", "testA", "testB"],
        "flickr30k": ["train", "val", "test"],
        "visual_genome": ["train", "test", "val"],
        "lvis": ["val"],
    }

    if args.split not in VALID_SPLITS[args.dataset]:
        print(
            f"\n[BŁĄD KRYTYCZNY] Zbiór '{args.dataset}' nie obsługuje podzbioru '{args.split}'. Dozwolone: {VALID_SPLITS[args.dataset]}\n"
        )
        exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        dataset = load_dataset(args.dataset, args.split)
    except FileNotFoundError as e:
        print(f"\n[BŁĄD PLIKU] Brak danych! Szczegóły: {e}")
        exit()

    now = datetime.datetime.now()
    SESSION_TIMESTAMP = now.strftime("%Y-%m-%d_%H-%M-%S")

    bezpieczny_dataset = args.dataset.replace("+", "plus")

    FOLDER_EXCEL = (
        f"wyniki_omdet_excel_{bezpieczny_dataset}_conf{args.conf}_{SESSION_TIMESTAMP}"
    )
    FOLDER_BLEDOW = f"wszystkie_omdet_bledy_{bezpieczny_dataset}_conf{args.conf}_{SESSION_TIMESTAMP}"
    FOLDER_SUKCESOW = f"wszystkie_omdet_sukcesy_{bezpieczny_dataset}_conf{args.conf}_{SESSION_TIMESTAMP}"
    FOLDER_WYKRESOW = f"wykresy_omdet_wynikowe_{bezpieczny_dataset}_conf{args.conf}_{SESSION_TIMESTAMP}"

    os.makedirs(FOLDER_EXCEL, exist_ok=True)
    os.makedirs(FOLDER_BLEDOW, exist_ok=True)
    os.makedirs(FOLDER_SUKCESOW, exist_ok=True)
    os.makedirs(FOLDER_WYKRESOW, exist_ok=True)

    wszystkie_wyniki = {}

    wszystkie_wyniki["omdet"] = run_evaluation_omdet(
        dataset,
        device,
        args.dataset,
        args.split,
        args.conf,
        FOLDER_EXCEL,
        FOLDER_BLEDOW,
        FOLDER_SUKCESOW,
    )

    if args.dataset == "grefcoco":
        generuj_wykresy_gvg(FOLDER_EXCEL, args.dataset, args.conf, FOLDER_WYKRESOW)
    else:
        generuj_wykresy_percentylowe(
            wszystkie_wyniki, args.dataset, args.conf, FOLDER_WYKRESOW
        )
