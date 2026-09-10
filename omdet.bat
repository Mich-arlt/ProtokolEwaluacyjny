@echo off

::PRZEJDZ DO FOLDERU PROJEKTU 
cd /d "%~dp0"

:: --- AKTYWACJA SRODOWISKA ---
call .venv_3.12\Scripts\activate
:: ----------------------------

echo [INFO] Srodowisko wirtualne aktywowane. Rozpoczynam benchmark...

:: ==========================================
:: 1. ZBIÓR REFCOCO+ 
:: ==========================================
echo [REFCOCO+] Rozpoczynanie serii testow...
:: VAL

python omdet_eval.py --dataset refcoco+ --split val --conf 0.15
python omdet_eval.py --dataset refcoco+ --split val --conf 0.25
python omdet_eval.py --dataset refcoco+ --split val --conf 0.50
:: TEST A (LUDZIE)

python omdet_eval.py --dataset refcoco+ --split testA --conf 0.15
python omdet_eval.py --dataset refcoco+ --split testA --conf 0.25
python omdet_eval.py --dataset refcoco+ --split testA --conf 0.50
:: TEST B (OBIEKTY)

python omdet_eval.py --dataset refcoco+ --split testB --conf 0.15
python omdet_eval.py --dataset refcoco+ --split testB --conf 0.25
python omdet_eval.py --dataset refcoco+ --split testB --conf 0.50

:: ==========================================
:: 1. ZBIÓR REFCOCO
:: ==========================================
echo [REFCOCO] Rozpoczynanie serii testow...
:: VAL

python omdet_eval.py --dataset refcoco --split val --conf 0.15
python omdet_eval.py --dataset refcoco --split val --conf 0.25
python omdet_eval.py --dataset refcoco --split val --conf 0.50
:: TEST A (LUDZIE)

python omdet_eval.py --dataset refcoco --split testA --conf 0.15
python omdet_eval.py --dataset refcoco --split testA --conf 0.25
python omdet_eval.py --dataset refcoco --split testA --conf 0.50
:: TEST B (OBIEKTY)

python omdet_eval.py --dataset refcoco --split testB --conf 0.15
python omdet_eval.py --dataset refcoco --split testB --conf 0.25
python omdet_eval.py --dataset refcoco --split testB --conf 0.50

:: ==========================================
:: 2. ZBIÓR GREFCOCO (GVG)
:: ==========================================
echo [GREFCOCO] Rozpoczynanie serii testow...
:: VAL
python omdet_eval.py --dataset grefcoco --split val --conf 0.15
python omdet_eval.py --dataset grefcoco --split val --conf 0.25
python omdet_eval.py --dataset grefcoco --split val --conf 0.50

:: TEST A
python omdet_eval.py --dataset grefcoco --split testA --conf 0.15
python omdet_eval.py --dataset grefcoco --split testA --conf 0.25
python omdet_eval.py --dataset grefcoco --split testA --conf 0.50

:: TEST B
python omdet_eval.py --dataset grefcoco --split testB --conf 0.15
python omdet_eval.py --dataset grefcoco --split testB --conf 0.25
python omdet_eval.py --dataset grefcoco --split testB --conf 0.50


:: ==========================================
:: 3. ZBIÓR REFCOCOG
:: ==========================================
echo [REFCOCOG] Rozpoczynanie serii testow...

python omdet_eval.py --dataset refcocog --split test --conf 0.15
python omdet_eval.py --dataset refcocog --split test --conf 0.25
python omdet_eval.py --dataset refcocog --split test --conf 0.50


:: ==========================================
:: 4. ZBIÓR REFERITGAME
:: ==========================================
echo [REFERIT] Rozpoczynanie serii testow...

python omdet_eval.py --dataset referit --split test --conf 0.15
python omdet_eval.py --dataset referit --split test --conf 0.25
python omdet_eval.py --dataset referit --split test --conf 0.50


:: ==========================================
:: 5. ZBIÓR FLICKR30K ENTITIES
:: ==========================================
echo [FLICKR30K] Rozpoczynanie serii testow...

python omdet_eval.py --dataset flickr30k --split test --conf 0.15
python omdet_eval.py --dataset flickr30k --split test --conf 0.25
python omdet_eval.py --dataset flickr30k --split test --conf 0.50

:: ==========================================
:: 6. ZBIÓR VISUAL GENOME
:: ==========================================
echo [VISUAL GENOME] Rozpoczynanie serii testow...

python omdet_eval.py --dataset visual_genome --split val --conf 0.15
python omdet_eval.py --dataset visual_genome --split val --conf 0.25
python omdet_eval.py --dataset visual_genome --split val --conf 0.50

:: ==========================================
:: 5. ZBIÓR LVIS
:: ==========================================
echo [LVIS] Rozpoczynanie serii testow...

python omdet_eval.py --dataset lvis --split val --conf 0.15
python omdet_eval.py --dataset lvis --split val --conf 0.25
python omdet_eval.py --dataset lvis --split val --conf 0.50


echo [FINISZ] Wszystkie zaplanowane testy zostaly wykonane. Pliki XLSX czekaja w folderze.
pause