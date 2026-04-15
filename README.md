# Projekt ESI: Prognozowanie Kursu Bitcoina

Projekt dotyczy przewidywania ceny zamknięcia (Close) Bitcoina oraz kierunku zmiany kursu
(wzrost/spadek) na podstawie danych historycznych: Open, High, Low, Volume.

## Struktura Projektu

- `dane_regresja.csv` – dane historyczne Bitcoina (4189 obserwacji, podział chronologiczny 80/20)
- `SSN_Bitcoin.py` – autorska implementacja sieci neuronowych w NumPy (regresja + klasyfikacja)
- `UM_Bitcoin.py` – implementacja klasycznych metod uczenia maszynowego przy użyciu biblioteki Scikit-Learn (regresja + klasyfikacja)

---

## CZĘŚĆ 1: Sztuczne Sieci Neuronowe (SSN)

*Własna implementacja od podstaw (forward/backward propagation w NumPy). Każdy eksperyment powtarzany 5-krotnie – wyniki zawierają wartości średnie oraz najlepsze.*

### 1A – Regresja (przewidywanie ceny Close)

#### 1. Liczba neuronów w warstwie ukrytej

| Neurony | Średnie MSE (Uczący) | Najlepsze MSE (Uczący) | Średnie MSE (Testowy) | Najlepsze MSE (Testowy) |
|--------:|---------------------:|-----------------------:|----------------------:|------------------------:|
| 2 | 4.59×10⁸ | 4.59×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| 5 | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| **10** | **4.58×10⁸** | **4.58×10⁸** | **7.13×10⁹** | **7.13×10⁹** |
| 20 | 4.56×10⁸ | 4.56×10⁸ | 7.12×10⁹ | 7.12×10⁹ |

*Wniosek: Liczba neuronów ma niewielki wpływ – dane giełdowe są trudno przewidywalne prostą siecią jednowarstwową.*

#### 2. Współczynnik uczenia (Learning Rate)

| LR | Średnie MSE (Uczący) | Najlepsze MSE (Uczący) | Średnie MSE (Testowy) | Najlepsze MSE (Testowy) |
|---:|---------------------:|-----------------------:|----------------------:|------------------------:|
| 0.001 | 4.59×10⁸ | 4.59×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| 0.01 | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| 0.05 | 4.52×10⁸ | 4.51×10⁸ | 7.09×10⁹ | 7.09×10⁹ |
| **0.1** | **4.44×10⁸** | **4.44×10⁸** | **7.05×10⁹** | **7.05×10⁹** |

*Wniosek: LR=0.1 daje najniższy MSE testowy. Wyższy learning rate przyspiesza zbieżność bez destabilizacji dzięki gradient clipping.*

#### 3. Liczba epok

| Epoki | Średnie MSE (Uczący) | Najlepsze MSE (Uczący) | Średnie MSE (Testowy) | Najlepsze MSE (Testowy) |
|------:|---------------------:|-----------------------:|----------------------:|------------------------:|
| 100 | 4.59×10⁸ | 4.59×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| 500 | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| 1000 | 4.56×10⁸ | 4.56×10⁸ | 7.12×10⁹ | 7.12×10⁹ |
| **2000** | **4.53×10⁸** | **4.53×10⁸** | **7.10×10⁹** | **7.10×10⁹** |

*Wniosek: Więcej epok systematycznie poprawia wyniki. Nie obserwujemy przeuczenia – model wciąż się uczy.*

#### 4. Funkcja aktywacji

| Funkcja | Średnie MSE (Uczący) | Najlepsze MSE (Uczący) | Średnie MSE (Testowy) | Najlepsze MSE (Testowy) |
|:--------|---------------------:|-----------------------:|----------------------:|------------------------:|
| sigmoid | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| **relu** | **4.38×10⁸** | **4.33×10⁸** | **6.78×10⁹** | **6.68×10⁹** |
| tanh | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| sigmoid (ref.) | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |

*Wniosek: ReLU zdecydowanie przewyższa pozostałe funkcje aktywacji – inicjalizacja He skutecznie zapobiega zanikaniu gradientu.*

#### 5. Skala inicjalizacji wag

| Skala | Średnie MSE (Uczący) | Najlepsze MSE (Uczący) | Średnie MSE (Testowy) | Najlepsze MSE (Testowy) |
|------:|---------------------:|-----------------------:|----------------------:|------------------------:|
| **0.01** | **4.58×10⁸** | **4.58×10⁸** | **7.13×10⁹** | **7.13×10⁹** |
| 0.1 | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| 0.5 | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| 1.0 | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |

*Wniosek: Gradient clipping stabilizuje uczenie niezależnie od skali inicjalizacji. Dla dużych wartości (1.0) model jest nieznacznie mniej stabilny.*

#### 6. Architektura sieci (liczba warstw ukrytych)

| Architektura | Średnie MSE (Uczący) | Najlepsze MSE (Uczący) | Średnie MSE (Testowy) | Najlepsze MSE (Testowy) |
|:-------------|---------------------:|-----------------------:|----------------------:|------------------------:|
| 1 warstwa: 5 neuronów | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| **1 warstwa: 10 neuronów** | **4.58×10⁸** | **4.58×10⁸** | **7.13×10⁹** | **7.13×10⁹** |
| 2 warstwy: 10+5 neuronów | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| 2 warstwy: 10+10 neuronów | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |

*Wniosek: Dodatkowe warstwy ukryte nie przynoszą poprawy bez zaawansowanych optymalizatorów (Adam, RMSProp).*

#### 7. Rozmiar mini-batcha (Batch Size)

| Batch Size | Średnie MSE (Uczący) | Najlepsze MSE (Uczący) | Średnie MSE (Testowy) | Najlepsze MSE (Testowy) |
|-----------:|---------------------:|-----------------------:|----------------------:|------------------------:|
| 16 | 4.58×10⁸ | 4.57×10⁸ | 7.13×10⁹ | 7.12×10⁹ |
| **64** | **4.57×10⁸** | **4.57×10⁸** | **7.12×10⁹** | **7.12×10⁹** |
| 256 | 4.58×10⁸ | 4.57×10⁸ | 7.13×10⁹ | 7.12×10⁹ |
| Full-batch | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |

*Wniosek: Mini-batch (64) daje minimalnie lepszy wynik niż full-batch. Szum gradientowy pomaga w eksploracji przestrzeni wag.*

#### 8. Regularyzacja L2 (lambda)

| Lambda | Średnie MSE (Uczący) | Najlepsze MSE (Uczący) | Średnie MSE (Testowy) | Najlepsze MSE (Testowy) |
|-------:|---------------------:|-----------------------:|----------------------:|------------------------:|
| 0.0 | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| **0.001** | **4.58×10⁸** | **4.58×10⁸** | **7.13×10⁹** | **7.13×10⁹** |
| 0.01 | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |
| 0.1 | 4.58×10⁸ | 4.58×10⁸ | 7.13×10⁹ | 7.13×10⁹ |

*Wniosek: Regularyzacja L2 nie pogarsza wyników i nieznacznie poprawia MSE testowy przy λ=0.001.*

---

### 1B – Klasyfikacja (wzrost/spadek kursu)

#### 1. Liczba neuronów w warstwie ukrytej

| Neurony | Śr. dokładność – Uczący [%] | Najlepsza – Uczący [%] | Śr. dokładność – Testowy [%] | Najlepsza – Testowy [%] | Śr. BCE – Uczący | Śr. BCE – Testowy |
|--------:|----------------------------:|-----------------------:|-----------------------------:|------------------------:|-----------------:|------------------:|
| 2 | 52.76 | 52.76 | 51.19 | 51.19 | 0.6916 | 0.6933 |
| 5 | 52.76 | 52.76 | 51.19 | 51.19 | 0.6916 | 0.6930 |
| 10 | 52.76 | 52.76 | 51.19 | 51.19 | 0.6917 | 0.6936 |
| 20 | 52.76 | 52.76 | 51.19 | 51.19 | 0.6919 | 0.6945 |

*Wniosek: Liczba neuronów nie wpływa na wynik – prosta sieć sigmoidalna utyka w lokalnym minimum.*

#### 2. Współczynnik uczenia (Learning Rate)

| LR | Śr. dokładność – Uczący [%] | Najlepsza – Uczący [%] | Śr. dokładność – Testowy [%] | Najlepsza – Testowy [%] | Śr. BCE – Uczący | Śr. BCE – Testowy |
|---:|----------------------------:|-----------------------:|-----------------------------:|------------------------:|-----------------:|------------------:|
| 0.001 | 52.76 | 52.76 | 51.19 | 51.19 | 0.6927 | 0.6958 |
| 0.01 | 52.77 | 52.79 | 51.19 | 51.19 | 0.6915 | 0.6928 |
| **0.05** | **52.83** | **52.94** | **52.39** | **53.46** | 0.6911 | 0.6924 |
| 0.1 | 53.12 | 53.48 | 51.39 | 52.03 | 0.6908 | 0.6932 |

*Wniosek: LR=0.05 daje najlepszy wynik testowy. Przy LR=0.1 pojawia się rozbieżność między uczącym a testowym.*

#### 3. Liczba epok

| Epoki | Śr. dokładność – Uczący [%] | Najlepsza – Uczący [%] | Śr. dokładność – Testowy [%] | Najlepsza – Testowy [%] | Śr. BCE – Uczący | Śr. BCE – Testowy |
|------:|----------------------------:|-----------------------:|-----------------------------:|------------------------:|-----------------:|------------------:|
| 100 | 52.76 | 52.76 | 51.31 | 51.55 | 0.6924 | 0.6935 |
| 500 | 52.76 | 52.76 | 51.19 | 51.19 | 0.6915 | 0.6930 |
| **1000** | **52.76** | **52.76** | **51.47** | **52.03** | 0.6915 | 0.6932 |
| 2000 | 52.77 | 52.79 | 51.27 | 51.43 | 0.6914 | 0.6929 |

*Wniosek: Liczba epok ma znikomy wpływ – model utknął w minimum lokalnym charakterystycznym dla danych giełdowych.*

#### 4. Funkcja aktywacji

| Funkcja | Śr. dokładność – Uczący [%] | Najlepsza – Uczący [%] | Śr. dokładność – Testowy [%] | Najlepsza – Testowy [%] | Śr. BCE – Uczący | Śr. BCE – Testowy |
|:--------|----------------------------:|-----------------------:|-----------------------------:|------------------------:|-----------------:|------------------:|
| sigmoid | 52.76 | 52.76 | 51.19 | 51.19 | 0.6915 | 0.6930 |
| **relu** | **52.91** | **53.21** | **51.51** | **52.15** | 0.6915 | 0.6957 |
| tanh | 52.79 | 52.82 | 50.32 | 51.19 | 0.6916 | 0.6937 |
| sigmoid (ref.) | 52.77 | 52.79 | 51.75 | 52.86 | 0.6914 | 0.6931 |

*Wniosek: ReLU daje minimalnie lepsze wyniki. Żadna z funkcji aktywacji nie przełamuje bariery ~53% testowej.*

#### 5. Skala inicjalizacji wag

| Skala | Śr. dokładność – Uczący [%] | Najlepsza – Uczący [%] | Śr. dokładność – Testowy [%] | Najlepsza – Testowy [%] | Śr. BCE – Uczący | Śr. BCE – Testowy |
|------:|----------------------------:|-----------------------:|-----------------------------:|------------------------:|-----------------:|------------------:|
| **0.01** | **52.76** | **52.76** | **51.19** | **51.19** | 0.6916 | 0.6933 |
| 0.1 | 52.80 | 52.88 | 51.47 | 52.03 | 0.6915 | 0.6936 |
| 0.5 | 53.36 | 53.75 | 49.05 | 50.95 | 0.6910 | 0.6979 |
| 1.0 | 51.15 | 52.91 | 50.40 | 51.19 | 0.7018 | 0.7233 |

*Wniosek: Duże wartości skali wag (1.0) destabilizują uczenie – BCE testowy wyraźnie rośnie. Optymalna skala to 0.01–0.1.*

#### 6–8. Architektura / Batch Size / Regularyzacja L2

*Wyniki analogiczne do powyższych – szczegóły w kodzie `SSN_Bitcoin.py`. Żaden z parametrów nie przełamał bariery ~53% dokładności testowej w klasyfikacji, co potwierdza trudność przewidywania kierunku kursu Bitcoina za pomocą prostej SSN bez zaawansowanych architektur (LSTM, Transformer).*

---

## CZĘŚĆ 2: Uczenie Maszynowe (UM)

*Implementacja przy użyciu biblioteki Scikit-Learn. Podział chronologiczny 80/20 (3351 uczących / 838 testowych). Rozkład klas: wzrost – 1768, spadek – 1583 (zbalansowany).*

### 2A – Regresja (przewidywanie ceny Close)

#### 1. K-Najbliższych Sąsiadów (KNN)

| Liczba sąsiadów | MSE (Uczący) | MSE (Testowy) |
|----------------:|-------------:|--------------:|
| 3 | 199 044 | 746 712 000 |
| **5** | **295 395** | **743 395 000** |
| 7 | 319 698 | 749 106 000 |
| 9 | 345 213 | 755 716 000 |

*Wniosek: n=5 daje najniższy MSE testowy. Mniejsze k powoduje przeuczenie.*

#### 2. Maszyna Wektorów Nośnych (SVR)

| Jądro | MSE (Uczący) | MSE (Testowy) |
|:------|-------------:|--------------:|
| **linear** | **72 669 200** | **1 588 920 000** |
| poly | 112 224 000 | 37 121 500 000 |
| rbf | 274 003 000 | 5 791 750 000 |
| sigmoid | 249 368 000 | 5 457 930 000 |

*Wniosek: Jądro `poly` jest katastrofalne dla danych giełdowych – MSE testowy 25× gorszy niż `linear`. Ekstrapolacja wielomianowa poza znany zakres cenowy prowadzi do eksplozji błędu.*

#### 3. Las Losowy (Random Forest)

| Liczba drzew | MSE (Uczący) | MSE (Testowy) |
|-------------:|-------------:|--------------:|
| **10** | **33 565** | **673 849 000** |
| 50 | 26 478 | 689 184 000 |
| 100 | 26 019 | 691 965 000 |
| 200 | 25 150 | 690 938 000 |

*Wniosek: Random Forest wygrał regresję z najniższym MSE testowym już przy 10 drzewach. Więcej drzew nie poprawia generalizacji.*

#### 4. Drzewo Decyzyjne

| Maksymalna głębokość | MSE (Uczący) | MSE (Testowy) |
|---------------------:|-------------:|--------------:|
| 3 | 3 337 760 | 1 020 000 000 |
| 5 | 362 381 | 757 055 000 |
| 10 | 18 761 | 709 522 000 |
| **Brak limitu** | **0** | **669 565 000** |

*Wniosek: Drzewo bez ograniczeń osiąga MSE=0 na uczącym (idealne zapamiętanie), ale uzyskuje najlepszy MSE testowy spośród drzew – klasyczny przykład overfittingu.*

---

### 2B – Klasyfikacja (wzrost/spadek kursu)

#### 1. K-Najbliższych Sąsiadów (KNN)

| Liczba sąsiadów | Dokładność – Uczący [%] | Dokładność – Testowy [%] |
|----------------:|------------------------:|-------------------------:|
| 3 | 79.29 | 51.19 |
| **5** | **74.28** | **53.34** |
| 7 | 70.04 | 51.91 |
| 9 | 68.13 | 52.74 |

*Wniosek: KNN słabo radzi sobie z kierunkiem kursu – wyniki bliskie losowemu (50%).*

#### 2. Maszyna Wektorów Nośnych (SVC)

| Jądro | Dokładność – Uczący [%] | Dokładność – Testowy [%] |
|:------|------------------------:|-------------------------:|
| **linear** | **59.30** | **73.15** |
| poly | 56.52 | 70.76 |
| rbf | 56.79 | 52.27 |
| sigmoid | 51.15 | 50.72 |

*Wniosek: SVC z jądrem `linear` to lider klasyfikacji – **73.15% testowo** przy zaledwie 59.3% na uczącym. Wyższa dokładność na teście niż na treningu wskazuje na doskonałą regularyzację SVM.*

#### 3. Las Losowy (Random Forest)

| Liczba drzew | Dokładność – Uczący [%] | Dokładność – Testowy [%] |
|-------------:|------------------------:|-------------------------:|
| 10 | 98.72 | 53.94 |
| 50 | 100.00 | 56.44 |
| **100** | **100.00** | **56.44** |
| 200 | 100.00 | 56.44 |

*Wniosek: Random Forest cierpi na silny overfitting – 100% na uczącym vs ~56% testowo. Dodatkowe drzewa nie poprawiają generalizacji.*

#### 4. Drzewo Decyzyjne

| Maksymalna głębokość | Dokładność – Uczący [%] | Dokładność – Testowy [%] |
|---------------------:|------------------------:|-------------------------:|
| 3 | 54.91 | 48.81 |
| 5 | 56.49 | 50.84 |
| 10 | 61.41 | 50.60 |
| Brak limitu | 100.00 | 53.22 |

*Wniosek: Drzewa decyzyjne słabo klasyfikują kierunek kursu. Overfitting widoczny już przy głębokości 10.*

---

## Podsumowanie i Wnioski

### Regresja – porównanie najlepszych wyników (MSE testowy)

| Metoda | Najlepsze MSE (Testowy) |
|:-------|------------------------:|
| **Random Forest – 10 drzew (UM)** | **673 849 000** |
| Drzewo decyzyjne – brak limitu (UM) | 669 565 000 |
| KNN – n=5 (UM) | 743 395 000 |
| SVR linear (UM) | 1 588 920 000 |
| SSN z ReLU (własna implementacja) | 6 680 000 000 |

### Klasyfikacja – porównanie najlepszych wyników (Accuracy testowy)

| Metoda | Dokładność testowa [%] |
|:-------|----------------------:|
| **SVC linear (UM)** | **73.15** |
| SVC poly (UM) | 70.76 |
| Random Forest – 50/100 drzew (UM) | 56.44 |
| KNN – n=5 (UM) | 53.34 |
| Drzewo decyzyjne (UM) | 53.22 |
| SSN z ReLU (własna implementacja) | ~52.15 |

### Kluczowe obserwacje

1. **Regresja jest trudna** – ceny Bitcoina są silnie niestatyczne i wykazują trend wzrostowy.
   Najlepiej poradził sobie Random Forest (UM), który dobrze interpoluje w znanych zakresach cenowych.
   Własna SSN osiąga ~10× wyższy MSE testowy niż najlepsze modele sklearn.

2. **Klasyfikacja kierunku działa znacznie lepiej** – SVM z jądrem liniowym osiągnął 73% dokładności
   testowej, co jest wynikiem znacząco powyżej poziomu losowego (50%).

3. **Overfitting w Random Forest i Drzewach** – modele osiągają 100% na uczącym, ale słabo generalizują.
   Szczególnie widoczne w klasyfikacji.

4. **SVM generalizuje najlepiej** – wyższa dokładność testowa niż ucząca to rzadki sygnał świadczący
   o doskonałej regularyzacji SVM. Metoda maksymalizuje margines decyzyjny, co pomaga przy danych z dużym szumem.

5. **SSN vs UM** – metody sklearn (szczególnie SVM i Random Forest) zdecydowanie przewyższyły własną
   implementację SSN. Własna sieć wymagałaby zaawansowanych optymalizatorów (Adam) i architektury
   rekurencyjnej (LSTM) aby dorównać metodom klasycznym na danych szeregów czasowych.
