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
python all_models.py --florence --kosmos2 --dataset refcoco+ --split val
python all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split val --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split val --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split val --conf 0.50
:: TEST A (LUDZIE)
python all_models.py --florence --kosmos2 --dataset refcoco+ --split testA
python all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testA --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testA --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testA --conf 0.50
:: TEST B (OBIEKTY)
python all_models.py --florence --kosmos2 --dataset refcoco+ --split testB
python all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testB --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testB --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset refcoco+ --split testB --conf 0.50

:: ==========================================
:: 1. ZBIÓR REFCOCO
:: ==========================================
echo [REFCOCO] Rozpoczynanie serii testow...
:: VAL
python all_models.py --florence --kosmos2 --dataset refcoco --split val
python all_models.py --yolo --dino --owlv2 --dataset refcoco --split val --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset refcoco --split val --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset refcoco --split val --conf 0.50
:: TEST A (LUDZIE)
python all_models.py --florence --kosmos2 --dataset refcoco --split testA
python all_models.py --yolo --dino --owlv2 --dataset refcoco --split testA --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset refcoco --split testA --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset refcoco --split testA --conf 0.50
:: TEST B (OBIEKTY)
python all_models.py --florence --kosmos2 --dataset refcoco --split testB
python all_models.py --yolo --dino --owlv2 --dataset refcoco --split testB --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset refcoco --split testB --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset refcoco --split testB --conf 0.50

:: ==========================================
:: 2. ZBIÓR GREFCOCO (GVG)
:: ==========================================
echo [GREFCOCO] Rozpoczynanie serii testow...
:: VAL
python all_models.py --florence --kosmos2 --dataset grefcoco --split val
python all_models.py --yolo --dino --owlv2 --dataset grefcoco --split val --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset grefcoco --split val --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset grefcoco --split val --conf 0.50

:: TEST A
python all_models.py --florence --kosmos2 --dataset grefcoco --split testA
python all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testA --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testA --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testA --conf 0.50

:: TEST B
python all_models.py --florence --kosmos2 --dataset grefcoco --split testB
python all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testB --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testB --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset grefcoco --split testB --conf 0.50


:: ==========================================
:: 3. ZBIÓR REFCOCOG
:: ==========================================
echo [REFCOCOG] Rozpoczynanie serii testow...
python all_models.py --florence --kosmos2 --dataset refcocog --split test
python all_models.py --yolo --dino --owlv2 --dataset refcocog --split test --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset refcocog --split test --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset refcocog --split test --conf 0.50


:: ==========================================
:: 4. ZBIÓR REFERITGAME
:: ==========================================
echo [REFERIT] Rozpoczynanie serii testow...
python all_models.py --florence --kosmos2 --dataset referit --split test
python all_models.py --yolo --dino --owlv2 --dataset referit --split test --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset referit --split test --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset referit --split test --conf 0.50


:: ==========================================
:: 5. ZBIÓR FLICKR30K ENTITIES
:: ==========================================
echo [FLICKR30K] Rozpoczynanie serii testow...
python all_models.py --florence --kosmos2 --dataset flickr30k --split test
python all_models.py --yolo --dino --owlv2 --dataset flickr30k --split test --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset flickr30k --split test --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset flickr30k --split test --conf 0.50

:: ==========================================
:: 6. ZBIÓR VISUAL GENOME
:: ==========================================
echo [VISUAL GENOME] Rozpoczynanie serii testow...
python all_models.py --florence --kosmos2 --dataset visual_genome --split val
python all_models.py --yolo --dino --owlv2 --dataset visual_genome --split val --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset visual_genome --split val --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset visual_genome --split val --conf 0.50

:: ==========================================
:: 5. ZBIÓR LVIS
:: ==========================================
echo [LVIS] Rozpoczynanie serii testow...
python all_models.py --florence --kosmos2 --dataset lvis --split val
python all_models.py --yolo --dino --owlv2 --dataset lvis --split val --conf 0.15
python all_models.py --yolo --dino --owlv2 --dataset lvis --split val --conf 0.25
python all_models.py --yolo --dino --owlv2 --dataset lvis --split val --conf 0.50


echo [FINISZ] Wszystkie zaplanowane testy zostaly wykonane. Pliki XLSX czekaja w folderze.
pause