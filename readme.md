Skrypt jest w pełni sterowany z poziomu terminala. Aby zadziałał, musisz podać przynajmniej jedną flagę modelu.
Reszta flag jest opcjonalna i posiada wartości domyślne.

### 1. Wybór modeli (Możesz łączyć dowolną liczbę!)
    --yolo
    --florence
    --dino    
    --owlv2  
    --kosmos2  
    --omdet   

### 2. Wybór zbioru danych (--dataset)
Domyślnie skrypt używa refcoco+. Aby to zmienić, użyj flagi --dataset:
    --dataset refcoco(domyślnie) 
    --dataset refcoco+ 
    --dataset refcocog 
    --dataset referit 
    --dataset grefcoco 
    --dataset flickr30k
    --dataset visual_genome

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
    * Generowanie Excela: Skrypt co 100 testowanych obiektów automatycznie zapisuje/nadpisuje plik .xlsx z pełną historią (mIoU, Top-1, Top-5, progi: 0.5, 0.7, 0.9).
    * Limit bezpieczeństwa: Zbiory giganty (jak VG czy Flickr30k) są automatycznie ucinane do maksymalnie 10 000 próbek na test, aby zapobiec wielodniowemu obciążeniu karty graficznej.

### 6. Przykłady Użycia

    1. Szybki test jednego modelu (Domyślne ustawienia: RefCOCO+, split val):
        python all_models.py --dino
    
    2. Porównanie DINO i Florence na trudnym zbiorze z ludźmi (RefCOCO+, split testA):
        python all_models.py --dino --florence --split testA
    
    3. Test najnowszego YOLO na długich opisach (RefCOCOg, split test):
        python all_models.py --yolo --dataset refcocog --split test
    
    4. Pojedynek modeli OVD (OWLv2) i MLLM (Kosmos-2) ze zwiększonym progiem pewności na Visual Genome:
        python all.py --owlv2 --kosmos2 --dataset visual_genome --split val --conf 0.5
    
    5. Pełny benchmark wszystkich 5 modeli naraz na zbiorze ReferIt:
        python all_models.py --yolo --florence --dino --owlv2 --kosmos2 --dataset referit --split test

### 7. Model OmDet-Turbo
    Dla modelu OmDet-Turbo potrzebne jest zainstalowanie requirements_omdet.
    python omdet_eval.py --omdet --dataset refcocog --split test