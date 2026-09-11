Skrypt jest w pełni sterowany z poziomu terminala. Aby zadziałał, musisz podać przynajmniej jedną flagę modelu.
Reszta flag jest opcjonalna i posiada wartości domyślne.

### 1. Wybór modeli (Możesz łączyć dowolną liczbę!)
    --yolo     : Uruchamia test dla modelu z rodziny YOLO-World (Promptable 1-Stage).
    --florence : Uruchamia test dla modelu Microsoft Florence-2-large (VLM).
    --dino     : Uruchamia test dla modelu Grounding DINO (End-to-End DETR).
    --owlv2    : Uruchamia test dla modelu Google OWLv2 (Vision-Transformer OVD).
    --kosmos2  : Uruchamia test dla modelu Microsoft Kosmos-2 (MLLM).
    --omdet    : Uruchamia test dla modelu OmDet-Turbo (najnowszy End-to-End DETR).

### 2. Wybór zbioru danych (--dataset)
Domyślnie skrypt używa refcoco+. Aby to zmienić, użyj flagi --dataset:
    --dataset refcoco+ (domyślnie) - Opisy oparte o wygląd (zakaz używania słów kierunkowych).
    --dataset refcocog - Bardzo długie, złożone opisy detali.
    --dataset referit - Krótkie hasła bazujące na lokalizacji.
    --dataset grefcoco - Wiele fraz, wiele odpowiedzi, obiektów może być wiele, albo może ich w ogóle nie być.
    --dataset flickr30k - Zdanie z wieloma frazami -> ramka dla każdej frazy (Phrase Grounding).
    --dataset visual_genome - Gęsta reprezentacja przestrzeni i długi ogon pojęć (miliony unikalnych atrybutów).

### 3. Wybór podzbioru testowego (--split)
Domyślnie skrypt używa podzbioru val. Aby zmienić trudność/kategorię testu, wpisz --split.
Skrypt posiada wbudowany słownik dozwolonych podzbiorów. Jeśli podasz zły split dla danego zbioru, 
program natychmiast przerwie działanie.

Zbiór danych(--dataset) | Dostępne flagi --split               | Opis podzbiorów
------------------------|--------------------------------------|--------------------------------------------------
refcoco+                | train, val, testA, testB             | testA: tylko ludzie, testB: obiekty bez ludzi
refcocog                | train, val, test                     | Klasyczny 
grefcoco                | train, val, testA, testB             | testA: tylko ludzie, testB: obiekty bez ludzi
referit                 | train, val, test                     | Klasyczny
flickr30k               | train, val, test                     | Klasyczny
visual_genome           | val                                  | val: pierwsze obrazy

### 4. Wybór progu pewności (--conf)
Określa minimalną pewność modelu wymaganą do zatwierdzenia ramki (domyślnie 0.25). 
Modele nie uwzględniające progu pominą flagę.
    --conf 0.75

### 5. Zapis wyników i ograniczenia sprzętowe (Ważne!)
    * Generowanie Excela: Skrypt co 50 testowanych obiektów automatycznie zapisuje/nadpisuje plik .xlsx z pełną historią (mIoU, Top-1, Top-5, progi: 0.5, 0.7, 0.9).
    * Limit bezpieczeństwa: Zbiory giganty (jak VG czy Flickr30k) są automatycznie ucinane do maksymalnie 10 000 próbek na test, aby zapobiec wielodniowemu obciążeniu karty graficznej.

### 6. Przykłady Użycia

    1. Szybki test jednego modelu (Domyślne ustawienia: RefCOCO+, split val):
        python eval_models.py --dino
    
    2. Porównanie DINO i Florence na trudnym zbiorze z ludźmi (RefCOCO+, split testA):
        python eval_models.py --dino --florence --split testA
    
    3. Test najnowszego YOLO na długich opisach (RefCOCOg, split test):
        python eval_models.py --yolo --dataset refcocog --split test
    
    4. Pojedynek modeli OVD (OWLv2) i MLLM (Kosmos-2) ze zwiększonym progiem pewności na Visual Genome:
        python test.py --owlv2 --kosmos2 --dataset visual_genome --split val --conf 0.5
    
    5. Pełny benchmark wszystkich 6 modeli naraz na zbiorze ReferIt:
        python eval_models.py --yolo --florence --dino --owlv2 --kosmos2 --omdet --dataset referit --split test