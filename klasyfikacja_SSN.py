import numpy as np
import pandas as pd
import os

# =====================================================================
# 1. KLASA SIECI NEURONOWEJ (KLASYFIKACJA BINARNA)
# =====================================================================
class NeuralNetworkClassifier:
    def __init__(self, input_size, hidden_size, output_size=1):
        """
        Inicjalizacja wag i biasów dla problemu klasyfikacji binarnej.
        Wagi losowane z rozkładu normalnego (małe wartości), biasy to zera.
        """
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))

        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))

    def sigmoid(self, z):
        """Funkcja aktywacji sigmoidalna – używana w obu warstwach."""
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """Propagacja w przód."""
        # Warstwa ukryta
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.sigmoid(self.Z1)

        # Warstwa wyjściowa – sigmoid daje prawdopodobieństwo klasy 1
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def compute_loss(self, A2, Y):
        """Binary Cross-Entropy – standardowa funkcja straty dla klasyfikacji binarnej."""
        m = Y.shape[1]
        epsilon = 1e-8  # zapobiega log(0)
        loss = -(1 / m) * np.sum(
            Y * np.log(A2 + epsilon) + (1 - Y) * np.log(1 - A2 + epsilon)
        )
        return loss

    def compute_accuracy(self, A2, Y):
        """Dokładność klasyfikacji (Accuracy)."""
        predictions = (A2 >= 0.5).astype(int)
        accuracy = np.mean(predictions == Y) * 100
        return accuracy

    def backward(self, X, Y, learning_rate):
        """Propagacja wsteczna i aktualizacja wag (Gradient Descent)."""
        m = X.shape[1]

        # Gradienty warstwy wyjściowej
        dZ2 = self.A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Gradienty warstwy ukrytej (pochodna sigmoidy)
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.A1 * (1 - self.A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Aktualizacja wag
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


# =====================================================================
# 2. FUNKCJE POMOCNICZE
# =====================================================================
def standardize(X_train, X_test):
    """
    Standaryzacja Z-score. Parametry liczone TYLKO na zbiorze uczącym,
    aby uniknąć wycieku danych (data leakage).
    """
    mean = np.mean(X_train, axis=1, keepdims=True)
    std  = np.std(X_train,  axis=1, keepdims=True)
    epsilon = 1e-8

    X_train_scaled = (X_train - mean) / (std + epsilon)
    X_test_scaled  = (X_test  - mean) / (std + epsilon)
    return X_train_scaled, X_test_scaled


def prepare_data(file_name, test_ratio=0.2):
    """
    Wczytuje dane Bitcoina i tworzy problem klasyfikacji binarnej:
    Y = 1 jeśli cena Close jest wyższa niż Open tego samego dnia (wzrost),
    Y = 0 w przeciwnym razie (brak wzrostu / spadek).
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(
            f"BŁĄD: Brak pliku '{file_name}'. Upewnij się, że plik jest w tym samym folderze co skrypt."
        )

    print(f"Wczytywanie danych z pliku: {file_name}...")
    df = pd.read_csv(file_name)

    # --- Tworzenie etykiety klasyfikacyjnej ---
    # 1 = Close > Open (cena wzrosła w ciągu dnia), 0 = Close <= Open
    df['Target'] = (df['Close'] > df['Open']).astype(int)

    # Cechy wejściowe: Open, High, Low, Volume (bez Close, żeby nie było wycieku)
    feature_cols = ['Open', 'High', 'Low', 'Volume']
    X_all = df[feature_cols].values
    Y_all = df['Target'].values

    # Podział na zbiór uczący i testowy (zachowanie kolejności czasowej)
    split_idx = int(len(df) * (1 - test_ratio))
    X_train_raw = X_all[:split_idx].T   # (cechy x obserwacje)
    Y_train      = Y_all[:split_idx].reshape(1, -1)
    X_test_raw   = X_all[split_idx:].T
    Y_test       = Y_all[split_idx:].reshape(1, -1)

    X_train, X_test = standardize(X_train_raw, X_test_raw)

    print(f"Liczba przykładów uczących : {X_train.shape[1]}")
    print(f"Liczba przykładów testowych: {X_test.shape[1]}")
    print(f"Rozkład klas (uczący) – wzrost: {int(Y_train.sum())}, spadek: {Y_train.shape[1] - int(Y_train.sum())}")

    return X_train, Y_train, X_test, Y_test


# =====================================================================
# 3. SILNIK EKSPERYMENTÓW
# =====================================================================
def run_experiments(X_train, Y_train, X_test, Y_test,
                    param_name, param_values,
                    n_repeats=5, epochs=1000, lr=0.01):
    """
    Testuje różne wartości wybranego parametru, uśredniając wyniki z kilku prób
    (sieć neuronowa to proces niedeterministyczny – losowa inicjalizacja wag).
    """
    results = []
    input_features = X_train.shape[0]

    print(f"\n--- EKSPERYMENTY DLA KLASYFIKACJI ---")
    print(f"Badany parametr: {param_name}\n")

    for val in param_values:
        print(f"  Testowanie: {param_name} = {val} ...")

        train_losses, test_losses = [], []
        train_accs,   test_accs   = [], []

        for _ in range(n_repeats):
            # Dynamiczne przypisanie badanego parametru
            hidden_size  = val  if param_name == 'hidden_size'   else 10
            current_lr   = val  if param_name == 'learning_rate' else lr
            current_ep   = val  if param_name == 'epochs'        else epochs

            nn = NeuralNetworkClassifier(input_size=input_features, hidden_size=hidden_size)

            # Pętla ucząca
            for _ in range(current_ep):
                nn.forward(X_train)
                nn.backward(X_train, Y_train, learning_rate=current_lr)

            # Ewaluacja
            A2_train = nn.forward(X_train)
            A2_test  = nn.forward(X_test)

            train_losses.append(nn.compute_loss(A2_train, Y_train))
            test_losses.append( nn.compute_loss(A2_test,  Y_test))
            train_accs.append(  nn.compute_accuracy(A2_train, Y_train))
            test_accs.append(   nn.compute_accuracy(A2_test,  Y_test))

        results.append({
            f'Wartość parametru ({param_name})': val,
            'Średnia dokładność – Uczący [%]':  round(np.mean(train_accs),  2),
            'Najlepsza dokładność – Uczący [%]': round(np.max(train_accs),  2),
            'Średnia dokładność – Testowy [%]':  round(np.mean(test_accs),  2),
            'Najlepsza dokładność – Testowy [%]':round(np.max(test_accs),   2),
            'Średni BCE – Uczący':               round(np.mean(train_losses),4),
            'Średni BCE – Testowy':              round(np.mean(test_losses), 4),
        })

    return pd.DataFrame(results)


# =====================================================================
# 4. URUCHOMIENIE
# =====================================================================
if __name__ == "__main__":
    nazwa_pliku = 'dane_regresja.csv'   # ten sam plik co przy regresji

    try:
        X_train, Y_train, X_test, Y_test = prepare_data(nazwa_pliku, test_ratio=0.2)

        # ------------------------------------------------------------------
        # EKSPERYMENT 1: Wpływ liczby neuronów w warstwie ukrytej
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 1: Liczba neuronów w warstwie ukrytej")
        print("="*65)
        tabela1 = run_experiments(
            X_train, Y_train, X_test, Y_test,
            param_name   = 'hidden_size',
            param_values = [2, 5, 10, 20],
            n_repeats    = 5,
            epochs       = 1000,
            lr           = 0.01
        )
        print("\nWYNIKI:")
        print(tabela1.to_markdown(index=False))

        # ------------------------------------------------------------------
        # EKSPERYMENT 2: Wpływ współczynnika uczenia (learning rate)
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 2: Współczynnik uczenia (learning rate)")
        print("="*65)
        tabela2 = run_experiments(
            X_train, Y_train, X_test, Y_test,
            param_name   = 'learning_rate',
            param_values = [0.001, 0.01, 0.05, 0.1],
            n_repeats    = 5,
            epochs       = 1000,
            lr           = 0.01
        )
        print("\nWYNIKI:")
        print(tabela2.to_markdown(index=False))

        # ------------------------------------------------------------------
        # EKSPERYMENT 3: Wpływ liczby epok
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 3: Liczba epok uczenia")
        print("="*65)
        tabela3 = run_experiments(
            X_train, Y_train, X_test, Y_test,
            param_name   = 'epochs',
            param_values = [100, 500, 1000, 2000],
            n_repeats    = 5,
            epochs       = 1000,
            lr           = 0.01
        )
        print("\nWYNIKI:")
        print(tabela3.to_markdown(index=False))

        print("\n✅ Analiza zakończona.")

    except Exception as e:
        print(f"\nWystąpił błąd: {e}")
