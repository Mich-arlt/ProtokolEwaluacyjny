#!/bin/bash
sed -i 's/\r$//' "$0"
source .venv_3.12/bin/activate
export CUDA_VISIBLE_DEVICES=0

echo "[GPU 0] Start: RefCOCO+ i RefCOCOg"

# --- REFCOCO+ VAL ---
python3 all_models.py --florence --kosmos2 --dataset refcoco --split val
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco --split val --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco --split val --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco --split val --conf 0.50

# --- REFCOCO+ TEST A ---
python3 all_models.py --florence --kosmos2 --dataset refcoco --split testA
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco --split testA --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco --split testA --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco --split testA --conf 0.50

# --- REFCOCO+ TEST B ---
python3 all_models.py --florence --kosmos2 --dataset refcoco --split testB
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco --split testB --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco --split testB --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco --split testB --conf 0.50


# --- REFCOCO+ VAL ---
python3 all_models.py --florence --kosmos2 --dataset refcoco+ --split val
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split val --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split val --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split val --conf 0.50

# --- REFCOCO+ TEST A ---
python3 all_models.py --florence --kosmos2 --dataset refcoco+ --split testA
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testA --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testA --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testA --conf 0.50

# --- REFCOCO+ TEST B ---
python3 all_models.py --florence --kosmos2 --dataset refcoco+ --split testB
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testB --conf 0.15
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testB --conf 0.25
python3 all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testB --conf 0.50




echo "[GPU 0] ZAKONCZONO!"