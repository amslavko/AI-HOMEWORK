Raport końcowy
Analiza i predykcja spalania kalorii

1. Cel opracowania
Zadaniem projektu było przygotowanie modelu regresyjnego służącego do szacowania liczby spalonych kalorii. Model wykorzystuje informacje dotyczące przebiegu treningu, w szczególności jego czas trwania oraz tętno, a także podstawowe dane opisujące użytkownika: wiek, wagę i wzrost.

2. Wstępna analiza danych
Zbiór danych obejmował siedem zmiennych wejściowych: Sex, Age, Height, Weight, Duration, Heart_Rate, Body_Temp.

Kluczowe obserwacje:
- Najsilniejszy związek ze zmienną Calories wykazują czas trwania aktywności fizycznej oraz tętno.
- Zależności pomiędzy tymi cechami a liczbą spalonych kalorii mają charakter w przybliżeniu liniowy.
- Pozostałe zmienne mają mniejszy wpływ bezpośredni, jednak poprawiają jakość predykcji.

3. Charakterystyka zastosowanego rozwiązania
Problem został rozwiązany przy użyciu wielowarstwowej sieci neuronowej typu MLP (Multi-Layer Perceptron), zaimplementowanej w środowisku PyTorch.

Architektura modelu:
- Warstwa wejściowa: 7 neuronów.
- Warstwy ukryte:
  - pierwsza warstwa zawierająca 64 neurony z aktywacją ReLU i Dropout,
  - druga warstwa z 32 neuronami, również z funkcją ReLU i Dropout.
- Warstwa wyjściowa: pojedynczy neuron generujący wartość predykcji.

Zastosowanie Dropoutu miało na celu ograniczenie przeuczenia sieci.

4. Trening modelu i wyniki
Uczenie modelu przeprowadzono w ciągu 30 epok. W trakcie treningu zaobserwowano regularny spadek wartości funkcji straty, co świadczy o poprawnym procesie uczenia.

Uzyskane rezultaty:
- Początkowa faza treningu (epoka 1): RMSLE około 0.17, Loss około 0.51
- Końcowa faza treningu (epoki 21–30): RMSLE około 0.07, Loss około 0.07

Model osiągnął stabilizację wyników po około 20 epokach.

5. Podsumowanie wyników
Przeprowadzone eksperymenty potwierdzają skuteczność zaproponowanego podejścia. Osiągnięta niska wartość błędu RMSLE wskazuje na dobrą dokładność predykcji. Przyjęta architektura sieci neuronowej okazała się wystarczająca dla rozważanego problemu.

6. Część teoretyczna

Zadanie 1 – Wyznaczanie pochodnych
Dla sieci o wejściu [2, 3], wartości docelowej 5, wagach równych 1.0 oraz biasach o wartości 1.0 wyznaczono następujące pochodne funkcji straty:
- pochodna względem wag wyjściowych: 96
- pochodna względem biasu wyjściowego: 16
- pochodne wag warstwy ukrytej: x1 = 32, x2 = 48
- pochodna względem biasów warstwy ukrytej: 16

Czy sieć neuronowa może się uczyć przy inicjalizacji wszystkich parametrów zerami?
Nie.
W takim przypadku:
- wyjście sieci przyjmuje wartość 0,
- gradienty wag są równe 0,
- nie zachodzi propagacja błędu do wcześniejszych warstw,
- występuje problem symetrii neuronów.

Zadanie 2 – Zagadnienia teoretyczne

1. Zastosowanie sieci neuronowych
Sieci neuronowe są wykorzystywane w sytuacjach, gdy dane są złożone, relacje trudne do jednoznacznego opisania oraz wymagane jest uogólnienie na nowe dane.

2. Funkcje aktywacji
Funkcje aktywacji umożliwiają wprowadzenie nieliniowości do modelu. Bez nich nawet sieć wielowarstwowa zachowywałaby się jak model liniowy.

3. Mechanizm Dropout
Dropout jest techniką regularyzacji polegającą na losowym dezaktywowaniu neuronów podczas treningu, co ogranicza ryzyko przeuczenia.

7. Autorzy
Kryzhanivskyi Stanislav – 29209
Volodymyr Nosko – 29176
Artsiom Slavko – 29164
Shumeiko Artem – 29227