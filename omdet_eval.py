import os
import json
import pickle
import torch
import argparse
import pandas as pd
import datetime
import gc
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from transformers import AutoProcessor, OmDetTurboForObjectDetection

COCO_IMG_DIR = r".\train2014\train2014"
COCO_2017_IMG_DIR = r".\lvis\val2017"
COCO_INSTANCES_JSON = r".\annotations_trainval2014\annotations\instances_train2014.json"
REFCOCO_PLUS_PKL = r".\refcoco+\refcoco+\refs(unc).p"
REFCOCOG_PKL = r".\refcocog\refcocog\refs(umd).p"
GREFCOCO_JSON = r".\grefcoco\grefs(unc).json"

REFERIT_IMG_DIR = r".\referit\referit\images"
REFERIT_INSTANCES_JSON = r".\referit\referit\instances.json"
REFERIT_PKL = r".\referit\referit\refs(berkeley).p"

FLICKR_IMG_DIR = r".\flickr\flickr30k_images"
FLICKR_ANN_JSON = r".\flickr\annotations.json"
LVIS_ANN_JSON = r".\lvis\lvis_v1_val.json"


def generuj_wykresy_liniowe(
    wszystkie_historie, dataset_name, conf_thresh, folder_wykresow
):
    if not wszystkie_historie:
        return

    metryki = [
        "mIoU_Top1",
        "Top1_Acc_0.5",
        "Top5_Acc_0.5",
        "Top1_Acc_0.7",
        "Top5_Acc_0.7",
        "Top1_Acc_0.9",
        "Top5_Acc_0.9",
    ]

    print(
        f"\n[RYSOWANIE] Generowanie {len(metryki)} wykresów do folderu '{folder_wykresow}'..."
    )

    for metryka in metryki:
        plt.figure(figsize=(10, 6))

        for model_name, historia in wszystkie_historie.items():
            if not historia:
                continue
            kroki = [w["Krok"] for w in historia]
            wartosci = [w[metryka] for w in historia]
            plt.plot(
                kroki,
                wartosci,
                marker="o",
                linewidth=2,
                label=model_name.upper(),
                color="#9b59b6",
            )

        plt.title(f"Przebieg {metryka} - {dataset_name.upper()} (Conf: {conf_thresh})")
        plt.xlabel("Przetworzone próbki (Krok)")

        if "Acc" in metryka:
            plt.ylabel("Skuteczność (%)")
            plt.ylim(0, 105)
        else:
            plt.ylabel("Wartość mIoU")
            plt.ylim(0, 1.05)

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        nazwa_pliku = os.path.join(
            folder_wykresow, f"liniowy_{metryka}_{dataset_name}_conf{conf_thresh}.png"
        )
        plt.savefig(nazwa_pliku)
        plt.close()

    print(f"[SUKCES] Wykresy bezpiecznie wylądowały w folderze: {folder_wykresow}/")


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area


def calculate_max_iou_for_multiple_targets(pred_box, gt_boxes):
    if not gt_boxes:
        return 0.0
    return max([calculate_iou(pred_box, gt) for gt in gt_boxes])


def load_dataset(dataset_name, target_split):
    dataset = []
    print(f"-> Rozpoczynam parsowanie zbioru: {dataset_name.upper()}")

    if dataset_name in ["refcoco+", "refcocog"]:
        with open(COCO_INSTANCES_JSON, "r") as f:
            coco_data = json.load(f)
        img_id_to_filename = {
            img["id"]: img["file_name"] for img in coco_data["images"]
        }
        ann_id_to_bbox = {ann["id"]: ann["bbox"] for ann in coco_data["annotations"]}
        pkl_path = REFCOCO_PLUS_PKL if dataset_name == "refcoco+" else REFCOCOG_PKL
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
            cat_name = cat_id_to_name[cat_id]
            dataset.append(
                {"image_path": img_path, "prompt": cat_name, "gt_boxes": boxes}
            )

    MAX_SAMPLES = 200
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
            print(f"DEBUG BŁĘDU (Pętla): {e}")
            pass

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
            elif best_iou_top1 >= 0.95 and zapisane_sukcesy < 10:
                try:
                    img_success = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(img_success)
                    for gt in gt_boxes:
                        draw.rectangle(gt, outline="green", width=5)

                    draw.rectangle(boxes[0], outline="blue", width=3)

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
                "Top1_Acc_0.5": (t1_at_05 / (i + 1)) * 100,
                "Top5_Acc_0.5": (t5_at_05 / (i + 1)) * 100,
                "Top1_Acc_0.7": (t1_at_07 / (i + 1)) * 100,
                "Top5_Acc_0.7": (t5_at_07 / (i + 1)) * 100,
                "Top1_Acc_0.9": (t1_at_09 / (i + 1)) * 100,
                "Top5_Acc_0.9": (t5_at_09 / (i + 1)) * 100,
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == total_samples:
            print(
                f"[OMDET] Krok {i + 1}/{total_samples} |IoU(T1): {best_iou_top1:.3f} | T1@0.5: {(t1_at_05/(i+1))*100:.1f}% | T5@0.5: {(t5_at_05/(i+1))*100:.1f}%"
            )
            pd.DataFrame(historia_wynikow).to_excel(nazwa_pliku_excel, index=False)

    print(f"\n[ZAKOŃCZONO] Zapisano: {nazwa_pliku_excel}")

    del model, processor
    clear_vram()
    return historia_wynikow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["refcoco+", "refcocog", "lvis"],
        default="refcoco+",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        dataset = load_dataset(args.dataset, args.split)
    except FileNotFoundError as e:
        print(f"\n[BŁĄD PLIKU] Brak danych! Szczegóły: {e}")
        exit()

    now = datetime.datetime.now()
    SESSION_TIMESTAMP = now.strftime("%Y-%m-%d_%H-%M-%S")

    FOLDER_EXCEL = f"wyniki_excel_{SESSION_TIMESTAMP}"
    FOLDER_BLEDOW = f"wszystkie_bledy_{SESSION_TIMESTAMP}"
    FOLDER_SUKCESOW = f"wszystkie_sukcesy_{SESSION_TIMESTAMP}"
    FOLDER_WYKRESOW = f"wykresy_wynikowe_{SESSION_TIMESTAMP}"

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

    generuj_wykresy_liniowe(wszystkie_wyniki, args.dataset, args.conf, FOLDER_WYKRESOW)
