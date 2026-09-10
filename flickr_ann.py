import os
import json
import re
import xml.etree.ElementTree as ET

ANNOTATIONS_DIR = "Annotations"
SENTENCES_DIR = "Sentences"
OUTPUT_JSON = "annotations.json"

dataset = []

for filename in os.listdir(SENTENCES_DIR):
    if not filename.endswith(".txt"):
        continue

    img_id = filename.split(".")[0]
    img_name = f"{img_id}.jpg"
    xml_path = os.path.join(ANNOTATIONS_DIR, f"{img_id}.xml")

    if not os.path.exists(xml_path):
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()
    entities = {}

    for obj in root.findall("object"):
        names = obj.findall("name")
        bndbox = obj.find("bndbox")
        if not bndbox:
            continue

        x = int(bndbox.find("xmin").text)
        y = int(bndbox.find("ymin").text)
        x2 = int(bndbox.find("xmax").text)
        y2 = int(bndbox.find("ymax").text)

        for name in names:
            ent_id = name.text
            if ent_id not in entities:
                entities[ent_id] = []
            entities[ent_id].append([x, y, x2, y2])

    with open(os.path.join(SENTENCES_DIR, filename), "r", encoding="utf-8") as f:
        for line in f:
            matches = re.findall(r"\[/EN#(\d+)[^\s]*\s([^\]]+)\]", line)
            clean_sentence = re.sub(r"\[/EN#\d+[^\s]*\s([^\]]+)\]", r"\1", line).strip()

            phrases_data = []
            for ent_id, phrase in matches:
                if ent_id in entities:
                    phrases_data.append(
                        {"phrase_text": phrase.strip(), "bboxes": entities[ent_id]}
                    )
            if phrases_data:
                last_digit = int(img_id[-1])
                split_name = (
                    "test" if last_digit == 9 else "val" if last_digit == 8 else "train"
                )

                dataset.append(
                    {
                        "image_name": img_name,
                        "prompt": clean_sentence,
                        "phrases": phrases_data,
                        "split": split_name,
                    }
                )

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Gotowe! Utworzono {OUTPUT_JSON} zawierający {len(dataset)} pełnych zdań.")
