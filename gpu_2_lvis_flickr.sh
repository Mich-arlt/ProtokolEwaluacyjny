#!/bin/bash
sed -i 's/\r$//' "$0"
source .venv_3.12/bin/activate
export CUDA_VISIBLE_DEVICES=2

echo "[GPU 2] Start: FLICKR30K i LVIS"

# --- FLICKR30K VAL ---
python3 all_models.py --florence --kosmos2 --dataset flickr30k --split test
python3 all_models.py --yolo --dino --owlv2 --dataset flickr30k --split test --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset flickr30k --split test --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset flickr30k --split test --conf 0.50

# --- LVIS VAL ---
python3 all_models.py --florence --kosmos2 --dataset lvis --split val
python3 all_models.py --yolo --dino --owlv2 --dataset lvis --split val --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset lvis --split val --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset lvis --split val --conf 0.50

# # --- REFCOCOG VAL ---
python3 all_models.py --florence --kosmos2 --dataset refcocog --split test
python3 all_models.py --yolo --dino --owlv2 --dataset refcocog --split test --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset refcocog --split test --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset refcocog --split test --conf 0.50

# # --- VG VAL ---
python3 all_models.py --florence --kosmos2 --dataset visual_genome --split val
python3 all_models.py --yolo --dino --owlv2 --dataset visual_genome --split val --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset visual_genome --split val --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset visual_genome --split val --conf 0.50

echo "[GPU 2] ZAKONCZONO!"