import numpy as np
import pandas as pd
import os

# =====================================================================
# 1. KLASA SIECI NEURONOWEJ (REGRESJA)
# =====================================================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size=1, activation='sigmoid'):
        """
        Inicjalizacja wag i biasów dla problemu regresji.
        Dla ReLU stosowana jest inicjalizacja He (sqrt(2/n)), dla pozostałych 0.1.
        Parametr activation pozwala wybrać funkcję aktywacji warstwy ukrytej.
        """
        self.activation_name = activation
        scale = np.sqrt(2.0 / input_size) if activation == 'relu' else 0.1
        self.W1 = np.random.randn(hidden_size, input_size) * scale
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))

    # --- Funkcje aktywacji ---
    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def tanh(self, z):
        return np.tanh(z)

    def activate(self, z):
        if self.activation_name == 'relu':
            return self.relu(z)
        elif self.activation_name == 'tanh':
            return self.tanh(z)
        else:
            return self.sigmoid(z)

    def activate_deriv(self, A):
        """Pochodna wybranej funkcji aktywacji (po A, nie Z)."""
        if self.activation_name == 'relu':
            return (A > 0).astype(float)
        elif self.activation_name == 'tanh':
            return 1 - A ** 2
        else:
            return A * (1 - A)

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.activate(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.Z2   # wyjście liniowe dla regresji
        return self.A2

    def compute_loss(self, A2, Y):
        m = Y.shape[1]
        return (1 / m) * np.sum(np.square(A2 - Y))

    def backward(self, X, Y, learning_rate):
        m = X.shape[1]
        dZ2 = self.A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.activate_deriv(self.A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.W1 -= learning_rate * np.clip(dW1, -1.0, 1.0)
        self.b1 -= learning_rate * np.clip(db1, -1.0, 1.0)
        self.W2 -= learning_rate * np.clip(dW2, -1.0, 1.0)
        self.b2 -= learning_rate * np.clip(db2, -1.0, 1.0)


# =====================================================================
# 2. FUNKCJE POMOCNICZE
# =====================================================================
def standardize(X_train, X_test):
    """Standaryzacja Z-score; parametry liczone tylko na zbiorze uczącym."""
    mean    = np.mean(X_train, axis=1, keepdims=True)
    std     = np.std(X_train,  axis=1, keepdims=True)
    epsilon = 1e-8
    return (X_train - mean) / (std + epsilon), (X_test - mean) / (std + epsilon)


def prepare_data(file_name, test_ratio=0.2):
    """
    Wczytuje dane i dzieli je chronologicznie (80/20).
    Podział chronologiczny zapobiega wyciekowi danych z przyszłości.
    Ostatnia kolumna = zmienna docelowa Y (Close).
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"BŁĄD: Brak pliku '{file_name}'.")

    print(f"Wczytywanie danych z pliku: {file_name}...")
    df = pd.read_csv(file_name)
    split_idx = int(len(df) * (1 - test_ratio))

    X_train_raw = df.iloc[:split_idx, :-1].to_numpy().T
    Y_train     = df.iloc[:split_idx,  -1:].to_numpy().T
    X_test_raw  = df.iloc[split_idx:,  :-1].to_numpy().T
    Y_test      = df.iloc[split_idx:,  -1:].to_numpy().T

    X_train, X_test = standardize(X_train_raw, X_test_raw)
    print(f"Liczba przykładów uczących : {X_train.shape[1]}")
    print(f"Liczba przykładów testowych: {X_test.shape[1]}")
    return X_train, Y_train, X_test, Y_test


# =====================================================================
# 3. SILNIK EKSPERYMENTÓW
# =====================================================================
def run_experiments(X_train, Y_train, X_test, Y_test,
                    param_name, param_values,
                    n_repeats=5, epochs=1000, lr=0.01,
                    hidden_size=10, activation='sigmoid'):
    """
    Testuje różne wartości wybranego parametru, uśredniając wyniki z kilku prób
    (sieć neuronowa jest niedeterministyczna – losowa inicjalizacja wag).
    """
    results = []
    input_features = X_train.shape[0]

    print(f"\nBadany parametr: {param_name}")

    for val in param_values:
        print(f"  Testowanie: {param_name} = {val} ...")
        train_metrics, test_metrics = [], []

        for _ in range(n_repeats):
            cur_hidden     = val if param_name == 'hidden_size'   else hidden_size
            cur_lr         = val if param_name == 'learning_rate' else lr
            cur_ep         = val if param_name == 'epochs'        else epochs
            cur_activation = val if param_name == 'activation'    else activation

            nn = NeuralNetwork(input_size=input_features,
                               hidden_size=cur_hidden,
                               activation=cur_activation)

            for _ in range(cur_ep):
                nn.forward(X_train)
                nn.backward(X_train, Y_train, learning_rate=cur_lr)

            A2_train = nn.forward(X_train)
            A2_test  = nn.forward(X_test)
            train_metrics.append(nn.compute_loss(A2_train, Y_train))
            test_metrics.append( nn.compute_loss(A2_test,  Y_test))

        results.append({
            f'Wartość parametru ({param_name})': val,
            'Średnie MSE (Uczący)':    round(np.mean(train_metrics), 4),
            'Najlepsze MSE (Uczący)':  round(np.min(train_metrics),  4),
            'Średnie MSE (Testowy)':   round(np.mean(test_metrics),  4),
            'Najlepsze MSE (Testowy)': round(np.min(test_metrics),   4),
        })

    return pd.DataFrame(results)


# =====================================================================
# 4. URUCHOMIENIE – 8 EKSPERYMENTÓW
# =====================================================================
if __name__ == "__main__":
    nazwa_pliku = 'dane_regresja.csv'

    try:
        X_train, Y_train, X_test, Y_test = prepare_data(nazwa_pliku, test_ratio=0.2)

        eksperymenty = [
            # param_name,       param_values,                       opis
            ('hidden_size',    [2, 5, 10, 20],                     'EKSPERYMENT 1: Liczba neuronów w warstwie ukrytej'),
            ('learning_rate',  [0.001, 0.01, 0.05, 0.1],           'EKSPERYMENT 2: Współczynnik uczenia (learning rate)'),
            ('epochs',         [100, 500, 1000, 2000],              'EKSPERYMENT 3: Liczba epok uczenia'),
            ('activation',     ['sigmoid', 'relu', 'tanh', 'sigmoid'], 'EKSPERYMENT 4: Funkcja aktywacji warstwy ukrytej'),
        ]

        # Eksperyment 4 używa 3 unikalnych wartości + powtórzenie sigmoidy jako punkt odniesienia
        # Zamieniamy na 4 unikalne wartości:
        eksperymenty[3] = ('activation', ['sigmoid', 'relu', 'tanh', 'sigmoid'],
                           'EKSPERYMENT 4: Funkcja aktywacji warstwy ukrytej')

        # Poprawka – 4 sensowne warianty inicjalizacji wag (przez skalowanie std)
        # Realizujemy jako osobną pętlę z parametrem init_scale
        print("\n" + "="*70)
        print("CZĘŚĆ – WŁASNA SIEĆ NEURONOWA (REGRESJA)")
        print("="*70)

        # --- Eksperymenty 1–3 (hidden_size, learning_rate, epochs) ---
        for param_name, param_values, opis in eksperymenty[:3]:
            print(f"\n{'='*70}")
            print(opis)
            print("="*70)
            tabela = run_experiments(
                X_train, Y_train, X_test, Y_test,
                param_name=param_name, param_values=param_values,
                n_repeats=5, epochs=1000, lr=0.01,
            )
            print("\nWYNIKI:")
            print(tabela.to_markdown(index=False))

        # --- Eksperyment 4: Funkcja aktywacji ---
        print(f"\n{'='*70}")
        print("EKSPERYMENT 4: Funkcja aktywacji warstwy ukrytej")
        print("="*70)
        tabela4 = run_experiments(
            X_train, Y_train, X_test, Y_test,
            param_name='activation', param_values=['sigmoid', 'relu', 'tanh', 'sigmoid'],
            n_repeats=5, epochs=1000, lr=0.01,
        )
        # Podmień zduplikowaną nazwę na czytelną etykietę
        tabela4.iloc[3, 0] = 'sigmoid (ref.)'
        print("\nWYNIKI:")
        print(tabela4.to_markdown(index=False))

        # --- Eksperyment 5: Skala inicjalizacji wag ---
        print(f"\n{'='*70}")
        print("EKSPERYMENT 5: Skala inicjalizacji wag (std mnożnik)")
        print("="*70)

        init_scales = [0.01, 0.1, 0.5, 1.0]
        results5 = []
        input_features = X_train.shape[0]
        print(f"\nBadany parametr: init_scale")

        for scale in init_scales:
            print(f"  Testowanie: init_scale = {scale} ...")
            train_m, test_m = [], []
            for _ in range(5):
                nn = NeuralNetwork(input_size=input_features, hidden_size=10, activation='sigmoid')
                nn.W1 = np.random.randn(*nn.W1.shape) * scale
                nn.W2 = np.random.randn(*nn.W2.shape) * scale
                for _ in range(1000):
                    nn.forward(X_train)
                    nn.backward(X_train, Y_train, learning_rate=0.01)
                A2_train = nn.forward(X_train)
                A2_test  = nn.forward(X_test)
                train_m.append(nn.compute_loss(A2_train, Y_train))
                test_m.append( nn.compute_loss(A2_test,  Y_test))
            results5.append({
                'Wartość parametru (init_scale)': scale,
                'Średnie MSE (Uczący)':    round(np.mean(train_m), 4),
                'Najlepsze MSE (Uczący)':  round(np.min(train_m),  4),
                'Średnie MSE (Testowy)':   round(np.mean(test_m),  4),
                'Najlepsze MSE (Testowy)': round(np.min(test_m),   4),
            })
        print("\nWYNIKI:")
        print(pd.DataFrame(results5).to_markdown(index=False))

        # --- Eksperyment 6: Liczba warstw ukrytych (1 lub 2) ---
        print(f"\n{'='*70}")
        print("EKSPERYMENT 6: Liczba warstw ukrytych (architektura sieci)")
        print("="*70)

        class NeuralNetwork2Hidden:
            """Sieć z dwiema warstwami ukrytymi dla porównania architektur."""
            def __init__(self, input_size, h1, h2):
                self.W1 = np.random.randn(h1, input_size) * 0.1
                self.b1 = np.zeros((h1, 1))
                self.W2 = np.random.randn(h2, h1) * 0.1
                self.b2 = np.zeros((h2, 1))
                self.W3 = np.random.randn(1, h2) * 0.1
                self.b3 = np.zeros((1, 1))

            def sigmoid(self, z):
                return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

            def forward(self, X):
                self.Z1 = np.dot(self.W1, X) + self.b1; self.A1 = self.sigmoid(self.Z1)
                self.Z2 = np.dot(self.W2, self.A1) + self.b2; self.A2 = self.sigmoid(self.Z2)
                self.Z3 = np.dot(self.W3, self.A2) + self.b3; self.A3 = self.Z3
                return self.A3

            def compute_loss(self, A, Y):
                return (1 / Y.shape[1]) * np.sum(np.square(A - Y))

            def backward(self, X, Y, lr):
                m = X.shape[1]
                dZ3 = self.A3 - Y
                dW3 = (1/m)*np.dot(dZ3, self.A2.T); db3 = (1/m)*np.sum(dZ3,axis=1,keepdims=True)
                dA2 = np.dot(self.W3.T, dZ3)
                dZ2 = dA2 * self.A2 * (1-self.A2)
                dW2 = (1/m)*np.dot(dZ2, self.A1.T); db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
                dA1 = np.dot(self.W2.T, dZ2)
                dZ1 = dA1 * self.A1 * (1-self.A1)
                dW1 = (1/m)*np.dot(dZ1, X.T); db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
                self.W1-=lr*dW1; self.b1-=lr*db1
                self.W2-=lr*dW2; self.b2-=lr*db2
                self.W3-=lr*dW3; self.b3-=lr*db3

        # Konfiguracje: (opis, rozmiary warstw ukrytych)
        architectures = [
            ('1 warstwa: 5 neuronów',    'single', 5),
            ('1 warstwa: 10 neuronów',   'single', 10),
            ('2 warstwy: 10+5 neuronów', 'double', (10, 5)),
            ('2 warstwy: 10+10 neuronów','double', (10, 10)),
        ]

        results6 = []
        print(f"\nBadany parametr: architektura sieci")
        for label, arch_type, sizes in architectures:
            print(f"  Testowanie: {label} ...")
            train_m, test_m = [], []
            for _ in range(5):
                if arch_type == 'single':
                    nn = NeuralNetwork(input_size=input_features, hidden_size=sizes)
                    for _ in range(1000):
                        nn.forward(X_train); nn.backward(X_train, Y_train, 0.01)
                    tr = nn.compute_loss(nn.forward(X_train), Y_train)
                    te = nn.compute_loss(nn.forward(X_test),  Y_test)
                else:
                    nn = NeuralNetwork2Hidden(input_features, sizes[0], sizes[1])
                    for _ in range(1000):
                        nn.forward(X_train); nn.backward(X_train, Y_train, 0.01)
                    tr = nn.compute_loss(nn.forward(X_train), Y_train)
                    te = nn.compute_loss(nn.forward(X_test),  Y_test)
                train_m.append(tr); test_m.append(te)
            results6.append({
                'Wartość parametru (architektura)': label,
                'Średnie MSE (Uczący)':    round(np.mean(train_m), 4),
                'Najlepsze MSE (Uczący)':  round(np.min(train_m),  4),
                'Średnie MSE (Testowy)':   round(np.mean(test_m),  4),
                'Najlepsze MSE (Testowy)': round(np.min(test_m),   4),
            })
        print("\nWYNIKI:")
        print(pd.DataFrame(results6).to_markdown(index=False))

        # --- Eksperyment 7: Rozmiar mini-batcha ---
        print(f"\n{'='*70}")
        print("EKSPERYMENT 7: Rozmiar mini-batcha (batch size)")
        print("="*70)

        def train_minibatch(X_tr, Y_tr, X_te, Y_te, batch_size, epochs=1000, lr=0.01):
            n_features = X_tr.shape[0]
            n_samples  = X_tr.shape[1]
            nn = NeuralNetwork(input_size=n_features, hidden_size=10)
            for ep in range(epochs):
                indices = np.random.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    idx = indices[start:start + batch_size]
                    X_b = X_tr[:, idx]
                    Y_b = Y_tr[:, idx]
                    nn.forward(X_b)
                    nn.backward(X_b, Y_b, learning_rate=lr)
            return (nn.compute_loss(nn.forward(X_tr), Y_tr),
                    nn.compute_loss(nn.forward(X_te), Y_te))

        batch_sizes = [16, 64, 256, X_train.shape[1]]   # ostatni = full batch
        results7 = []
        print(f"\nBadany parametr: batch_size")
        for bs in batch_sizes:
            label = bs if bs != X_train.shape[1] else f"{bs} (full batch)"
            print(f"  Testowanie: batch_size = {label} ...")
            train_m, test_m = [], []
            for _ in range(5):
                tr, te = train_minibatch(X_train, Y_train, X_test, Y_test, bs)
                train_m.append(tr); test_m.append(te)
            results7.append({
                'Wartość parametru (batch_size)': str(label),
                'Średnie MSE (Uczący)':    round(np.mean(train_m), 4),
                'Najlepsze MSE (Uczący)':  round(np.min(train_m),  4),
                'Średnie MSE (Testowy)':   round(np.mean(test_m),  4),
                'Najlepsze MSE (Testowy)': round(np.min(test_m),   4),
            })
        print("\nWYNIKI:")
        print(pd.DataFrame(results7).to_markdown(index=False))

        # --- Eksperyment 8: Regularyzacja L2 (weight decay) ---
        print(f"\n{'='*70}")
        print("EKSPERYMENT 8: Regularyzacja L2 (lambda)")
        print("="*70)

        class NeuralNetworkL2(NeuralNetwork):
            """Sieć z regularyzacją L2 (weight decay) dodaną do backprop."""
            def backward(self, X, Y, learning_rate, lam=0.0):
                m = X.shape[1]
                dZ2 = self.A2 - Y
                dW2 = (1/m)*np.dot(dZ2, self.A1.T) + (lam/m)*self.W2
                db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
                dA1 = np.dot(self.W2.T, dZ2)
                dZ1 = dA1 * self.activate_deriv(self.A1)
                dW1 = (1/m)*np.dot(dZ1, X.T) + (lam/m)*self.W1
                db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
                self.W1 -= learning_rate*dW1; self.b1 -= learning_rate*db1
                self.W2 -= learning_rate*dW2; self.b2 -= learning_rate*db2

        lambdas = [0.0, 0.001, 0.01, 0.1]
        results8 = []
        print(f"\nBadany parametr: lambda (L2)")
        for lam in lambdas:
            print(f"  Testowanie: lambda = {lam} ...")
            train_m, test_m = [], []
            for _ in range(5):
                nn = NeuralNetworkL2(input_size=input_features, hidden_size=10)
                for _ in range(1000):
                    nn.forward(X_train)
                    nn.backward(X_train, Y_train, learning_rate=0.01, lam=lam)
                train_m.append(nn.compute_loss(nn.forward(X_train), Y_train))
                test_m.append( nn.compute_loss(nn.forward(X_test),  Y_test))
            results8.append({
                'Wartość parametru (lambda L2)': lam,
                'Średnie MSE (Uczący)':    round(np.mean(train_m), 4),
                'Najlepsze MSE (Uczący)':  round(np.min(train_m),  4),
                'Średnie MSE (Testowy)':   round(np.mean(test_m),  4),
                'Najlepsze MSE (Testowy)': round(np.min(test_m),   4),
            })
        print("\nWYNIKI:")
        print(pd.DataFrame(results8).to_markdown(index=False))

        print("\n✅ Wszystkie eksperymenty zakończone.")

    except Exception as e:
        print(f"\nWystąpił błąd: {e}")
        import traceback; traceback.print_exc()
