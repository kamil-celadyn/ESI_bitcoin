import numpy as np
import pandas as pd
import os

# =====================================================================
# 1. KLASA SIECI NEURONOWEJ 
# =====================================================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size=1):
        """
        Inicjalizacja wag i biasów dla problemu regresji.
        Wagi losowane z rozkładu normalnego (małe wartości), biasy to zera.
        """
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))

    def sigmoid(self, z):
        """Funkcja aktywacji dla warstwy ukrytej."""
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """Propagacja w przód (obliczanie odpowiedzi sieci)."""
        # Przejście z wejścia do warstwy ukrytej
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        
        # Przejście z warstwy ukrytej na wyjście.
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.Z2 
            
        return self.A2

    def compute_loss(self, A2, Y):
        """Obliczanie błędu średniokwadratowego (MSE)."""
        m = Y.shape[1]
        loss = (1/m) * np.sum(np.square(A2 - Y))
        return loss

    def backward(self, X, Y, learning_rate):
        """Propagacja wsteczna (Backpropagation) i aktualizacja wag."""
        m = X.shape[1]
        
        # Gradienty dla warstwy wyjściowej
        dZ2 = self.A2 - Y 
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Gradienty dla warstwy ukrytej (z pochodną funkcji sigmoid)
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.A1 * (1 - self.A1) 
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # Aktualizacja wag algorytmem Gradient Descent
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


# =====================================================================
# 2. FUNKCJE POMOCNICZE
# =====================================================================
def standardize(X_train, X_test):
    """
    Standaryzacja danych (Z-score normalization).
    Parametry (średnia i odchylenie) są liczone TYLKO na zbiorze uczącym,
    aby nie przenosić informacji z testowego (data leakage).
    """
    mean = np.mean(X_train, axis=1, keepdims=True)
    std = np.std(X_train, axis=1, keepdims=True)
    epsilon = 1e-8 # Zabezpieczenie przed dzieleniem przez 0
    
    X_train_scaled = (X_train - mean) / (std + epsilon)
    X_test_scaled = (X_test - mean) / (std + epsilon)
    return X_train_scaled, X_test_scaled

def prepare_data(file_name, test_ratio=0.2):
    """Wczytuje z pliku CSV, dzieli na zbiory i standaryzuje."""
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"BŁĄD: Brak pliku '{file_name}'. Upewnij się, że plik jest w tym samym folderze co skrypt.")
    
    print(f"Wczytywanie danych z pliku: {file_name}...")
    # Wczytujemy plik. Zakładamy, że pierwsza linia to nazwy kolumn.
    df = pd.read_csv(file_name)

    # Losowy podział na zbiór uczący i testowy (np. 80/20)
    train_df = df.sample(frac=(1 - test_ratio), random_state=42)
    test_df = df.drop(train_df.index)

    # Przygotowanie macierzy X (cechy) i Y (zmienna docelowa - ostatnia kolumna)
    # .T odwraca macierz, by wymiary to: (liczba_cech, liczba_obserwacji)
    X_train_raw = train_df.iloc[:, :-1].to_numpy().T
    Y_train = train_df.iloc[:, -1:].to_numpy().T
    
    X_test_raw = test_df.iloc[:, :-1].to_numpy().T
    Y_test = test_df.iloc[:, -1:].to_numpy().T

    # Skalowanie cech
    X_train, X_test = standardize(X_train_raw, X_test_raw)
    
    return X_train, Y_train, X_test, Y_test


# =====================================================================
# 3. SILNIK EKSPERYMENTÓW
# =====================================================================
def run_experiments(X_train, Y_train, X_test, Y_test, param_name, param_values, n_repeats=5, epochs=1000, lr=0.01):
    """
    Testuje różne wartości danego parametru, uśredniając wyniki z kilku prób.
    """
    results = []
    input_features = X_train.shape[0]
    
    print(f"\n--- ROZPOCZYNAM EKSPERYMENTY DLA REGRESJI ---")
    print(f"Badany parametr: {param_name}")
    
    for val in param_values:
        print(f"Testowanie wariantu: {param_name} = {val}...")
        
        train_metrics = []
        test_metrics = []
        
        for attempt in range(n_repeats):
            # Dynamiczne przypisanie badanego parametru
            hidden_size = val if param_name == 'hidden_size' else 10
            current_lr = val if param_name == 'learning_rate' else lr
            
            # Tworzymy nową sieć (losowa inicjalizacja wag)
            nn = NeuralNetwork(input_size=input_features, hidden_size=hidden_size)
            
            # Pętla ucząca
            for _ in range(epochs):
                nn.forward(X_train)
                nn.backward(X_train, Y_train, learning_rate=current_lr)
            
            # Zapisanie ostatecznego błędu (MSE) po wyuczeniu
            A2_train = nn.forward(X_train)
            A2_test = nn.forward(X_test)
            
            train_metrics.append(nn.compute_loss(A2_train, Y_train))
            test_metrics.append(nn.compute_loss(A2_test, Y_test))
        
        # Zapisujemy wartości średnie i najlepsze (najmniejszy błąd MSE)
        results.append({
            f'Wartość parametru ({param_name})': val,
            'Średnie MSE (Uczący)': np.mean(train_metrics),
            'Najlepsze MSE (Uczący)': np.min(train_metrics),
            'Średnie MSE (Testowy)': np.mean(test_metrics),
            'Najlepsze MSE (Testowy)': np.min(test_metrics)
        })
        
    return pd.DataFrame(results)


# =====================================================================
# 4. URUCHOMIENIE
# =====================================================================
if __name__ == "__main__":
    # 1. Przygotuj swój plik CSV. Ostatnia kolumna musi być wartością przewidywaną (Y).
    nazwa_pliku = 'dane_regresja.csv' 
    
    try:
        # 2. Załadowanie i obróbka danych
        X_train, Y_train, X_test, Y_test = prepare_data(nazwa_pliku, test_ratio=0.2)
        
        # 3. Przeprowadzenie analizy wpływu parametrów (np. liczby neuronów)
        # UWAGA: Minimum 4 różne wartości wg. wytycznych
        badane_wartosci = [2, 5, 10, 20] 
        
        tabela_wynikow = run_experiments(
            X_train, Y_train, X_test, Y_test, 
            param_name='hidden_size',    #Do zmiany na 'learning_rate' jeśli chcemy testować wpływ learning rate itd
            param_values=badane_wartosci, 
            n_repeats=5,                 # Liczba powtórzeń dla każdej wartości parametru
            epochs=1000,                 # Liczba epok w pojedynczym cyklu uczenia
            lr=0.01                      # Domyślny learning rate (jeśli nie jest badanym parametrem)
        )
        
        # 4. Wyświetlenie zestawienia
        print("\nGOTOWE ZESTAWIENIE DO RAPORTU:")
        print(tabela_wynikow.to_markdown(index=False))
        
    except Exception as e:
        print(f"\nWystąpił błąd: {e}")