#!/bin/bash
sed -i 's/\r$//' "$0"

export CUDA_VISIBLE_DEVICES=1

echo "[GPU 1] Start: GREFCOCO i REFERIT"

# --- GREFCOCO VAL ---
python3 all_models.py --florence --kosmos2 --dataset grefcoco --split val
python3 all_models.py --yolo --dino --owlv2 --dataset grefcoco --split val --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset grefcoco --split val --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset grefcoco --split val --conf 0.50

# --- GREFCOCO TEST A ---
python3 all_models.py --florence --kosmos2 --dataset grefcoco --split testA
python3 all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testA --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testA --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testA --conf 0.50

# --- GREFCOCO TEST B ---
python3 all_models.py --florence --kosmos2 --dataset grefcoco --split testB
python3 all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testB --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testB --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testB --conf 0.50

# --- REFERIT VAL ---
python3 all_models.py --florence --kosmos2 --dataset referit --split test
python3 all_models.py --yolo --dino --owlv2 --dataset referit --split test --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset referit --split test --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset referit --split test --conf 0.50

echo "[GPU 1] ZAKONCZONO!"