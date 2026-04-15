import numpy as np
import pandas as pd
import os

# =====================================================================
# 1. KLASA SIECI NEURONOWEJ (KLASYFIKACJA BINARNA)
# =====================================================================
class NeuralNetworkClassifier:
    def __init__(self, input_size, hidden_size, output_size=1, activation='sigmoid'):
        """
        Inicjalizacja wag i biasów dla klasyfikacji binarnej.
        Parametr activation określa funkcję aktywacji warstwy ukrytej.
        Warstwa wyjściowa zawsze używa sigmoidy (prawdopodobieństwo klasy 1).
        """
        self.activation_name = activation
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def tanh_fn(self, z):
        return np.tanh(z)

    def activate(self, z):
        if self.activation_name == 'relu':
            return self.relu(z)
        elif self.activation_name == 'tanh':
            return self.tanh_fn(z)
        else:
            return self.sigmoid(z)

    def activate_deriv(self, A):
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
        self.A2 = self.sigmoid(self.Z2)   # wyjście = prawdopodobieństwo
        return self.A2

    def compute_loss(self, A2, Y):
        """Binary Cross-Entropy."""
        m = Y.shape[1]
        eps = 1e-8
        return -(1/m) * np.sum(Y*np.log(A2+eps) + (1-Y)*np.log(1-A2+eps))

    def compute_accuracy(self, A2, Y):
        return np.mean((A2 >= 0.5).astype(int) == Y) * 100

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


# =====================================================================
# 2. SIEĆ Z DWIEMA WARSTWAMI UKRYTYMI
# =====================================================================
class NeuralNetworkClassifier2Hidden:
    def __init__(self, input_size, h1, h2):
        self.W1 = np.random.randn(h1, input_size) * 0.1; self.b1 = np.zeros((h1, 1))
        self.W2 = np.random.randn(h2, h1) * 0.1;         self.b2 = np.zeros((h2, 1))
        self.W3 = np.random.randn(1,  h2) * 0.1;         self.b3 = np.zeros((1,  1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X):
        self.Z1=np.dot(self.W1,X)+self.b1;   self.A1=self.sigmoid(self.Z1)
        self.Z2=np.dot(self.W2,self.A1)+self.b2; self.A2=self.sigmoid(self.Z2)
        self.Z3=np.dot(self.W3,self.A2)+self.b3; self.A3=self.sigmoid(self.Z3)
        return self.A3

    def compute_loss(self, A, Y):
        m=Y.shape[1]; eps=1e-8
        return -(1/m)*np.sum(Y*np.log(A+eps)+(1-Y)*np.log(1-A+eps))

    def compute_accuracy(self, A, Y):
        return np.mean((A>=0.5).astype(int)==Y)*100

    def backward(self, X, Y, lr):
        m=X.shape[1]
        dZ3=self.A3-Y
        dW3=(1/m)*np.dot(dZ3,self.A2.T); db3=(1/m)*np.sum(dZ3,axis=1,keepdims=True)
        dA2=np.dot(self.W3.T,dZ3); dZ2=dA2*self.A2*(1-self.A2)
        dW2=(1/m)*np.dot(dZ2,self.A1.T); db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)
        dA1=np.dot(self.W2.T,dZ2); dZ1=dA1*self.A1*(1-self.A1)
        dW1=(1/m)*np.dot(dZ1,X.T);  db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)
        self.W1-=lr*dW1; self.b1-=lr*db1
        self.W2-=lr*dW2; self.b2-=lr*db2
        self.W3-=lr*dW3; self.b3-=lr*db3


# =====================================================================
# 3. FUNKCJE POMOCNICZE
# =====================================================================
def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=1, keepdims=True)
    std  = np.std(X_train,  axis=1, keepdims=True)
    eps  = 1e-8
    return (X_train-mean)/(std+eps), (X_test-mean)/(std+eps)


def prepare_data(file_name, test_ratio=0.2):
    """
    Wczytuje dane i tworzy klasyfikację binarną:
    Y=1 gdy Close > Open (wzrost), Y=0 w przeciwnym razie.
    Podział chronologiczny – bez data leakage.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"BŁĄD: Brak pliku '{file_name}'.")

    print(f"Wczytywanie danych z pliku: {file_name}...")
    df = pd.read_csv(file_name)
    df['Target'] = (df['Close'] > df['Open']).astype(int)

    feature_cols = ['Open', 'High', 'Low', 'Volume']
    X_all = df[feature_cols].values
    Y_all = df['Target'].values

    split_idx = int(len(df) * (1 - test_ratio))
    X_train_raw = X_all[:split_idx].T;   Y_train = Y_all[:split_idx].reshape(1,-1)
    X_test_raw  = X_all[split_idx:].T;   Y_test  = Y_all[split_idx:].reshape(1,-1)

    X_train, X_test = standardize(X_train_raw, X_test_raw)
    print(f"Liczba przykładów uczących : {X_train.shape[1]}")
    print(f"Liczba przykładów testowych: {X_test.shape[1]}")
    print(f"Rozkład klas (uczący) – wzrost: {int(Y_train.sum())}, spadek: {Y_train.shape[1]-int(Y_train.sum())}")
    return X_train, Y_train, X_test, Y_test


# =====================================================================
# 4. SILNIK EKSPERYMENTÓW (ogólny)
# =====================================================================
def run_experiments(X_train, Y_train, X_test, Y_test,
                    param_name, param_values,
                    n_repeats=5, epochs=1000, lr=0.01,
                    hidden_size=10, activation='sigmoid', lam=0.0):
    results = []
    input_features = X_train.shape[0]
    print(f"\nBadany parametr: {param_name}")

    for val in param_values:
        print(f"  Testowanie: {param_name} = {val} ...")
        train_losses, test_losses, train_accs, test_accs = [], [], [], []

        for _ in range(n_repeats):
            cur_hidden = val if param_name == 'hidden_size'   else hidden_size
            cur_lr     = val if param_name == 'learning_rate' else lr
            cur_ep     = val if param_name == 'epochs'        else epochs
            cur_act    = val if param_name == 'activation'    else activation
            cur_lam    = val if param_name == 'lambda_l2'     else lam

            nn = NeuralNetworkClassifier(input_size=input_features,
                                         hidden_size=cur_hidden,
                                         activation=cur_act)
            for _ in range(cur_ep):
                nn.forward(X_train)
                nn.backward(X_train, Y_train, learning_rate=cur_lr, lam=cur_lam)

            A2_tr = nn.forward(X_train); A2_te = nn.forward(X_test)
            train_losses.append(nn.compute_loss(A2_tr, Y_train))
            test_losses.append( nn.compute_loss(A2_te, Y_test))
            train_accs.append(  nn.compute_accuracy(A2_tr, Y_train))
            test_accs.append(   nn.compute_accuracy(A2_te, Y_test))

        results.append({
            f'Wartość parametru ({param_name})':         val,
            'Średnia dokładność – Uczący [%]':           round(np.mean(train_accs),  2),
            'Najlepsza dokładność – Uczący [%]':         round(np.max(train_accs),   2),
            'Średnia dokładność – Testowy [%]':          round(np.mean(test_accs),   2),
            'Najlepsza dokładność – Testowy [%]':        round(np.max(test_accs),    2),
            'Średni BCE – Uczący':                       round(np.mean(train_losses),4),
            'Średni BCE – Testowy':                      round(np.mean(test_losses), 4),
        })

    return pd.DataFrame(results)


# =====================================================================
# 5. URUCHOMIENIE – 8 EKSPERYMENTÓW
# =====================================================================
if __name__ == "__main__":
    nazwa_pliku = 'dane_regresja.csv'

    try:
        X_train, Y_train, X_test, Y_test = prepare_data(nazwa_pliku, test_ratio=0.2)
        input_features = X_train.shape[0]

        # ------------------------------------------------------------------
        # EKSPERYMENT 1: Liczba neuronów w warstwie ukrytej
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 1: Liczba neuronów w warstwie ukrytej")
        print("="*65)
        t1 = run_experiments(X_train,Y_train,X_test,Y_test,
                             'hidden_size',[2,5,10,20],n_repeats=5,epochs=1000,lr=0.01)
        print("\nWYNIKI:"); print(t1.to_markdown(index=False))

        # ------------------------------------------------------------------
        # EKSPERYMENT 2: Współczynnik uczenia
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 2: Współczynnik uczenia (learning rate)")
        print("="*65)
        t2 = run_experiments(X_train,Y_train,X_test,Y_test,
                             'learning_rate',[0.001,0.01,0.05,0.1],n_repeats=5,epochs=1000,lr=0.01)
        print("\nWYNIKI:"); print(t2.to_markdown(index=False))

        # ------------------------------------------------------------------
        # EKSPERYMENT 3: Liczba epok
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 3: Liczba epok uczenia")
        print("="*65)
        t3 = run_experiments(X_train,Y_train,X_test,Y_test,
                             'epochs',[100,500,1000,2000],n_repeats=5,epochs=1000,lr=0.01)
        print("\nWYNIKI:"); print(t3.to_markdown(index=False))

        # ------------------------------------------------------------------
        # EKSPERYMENT 4: Funkcja aktywacji
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 4: Funkcja aktywacji warstwy ukrytej")
        print("="*65)
        t4 = run_experiments(X_train,Y_train,X_test,Y_test,
                             'activation',['sigmoid','relu','tanh','sigmoid'],
                             n_repeats=5,epochs=1000,lr=0.01)
        t4.iloc[3,0] = 'sigmoid (ref.)'
        print("\nWYNIKI:"); print(t4.to_markdown(index=False))

        # ------------------------------------------------------------------
        # EKSPERYMENT 5: Skala inicjalizacji wag
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 5: Skala inicjalizacji wag (std mnożnik)")
        print("="*65)
        init_scales = [0.01, 0.1, 0.5, 1.0]
        results5 = []
        print(f"\nBadany parametr: init_scale")
        for scale in init_scales:
            print(f"  Testowanie: init_scale = {scale} ...")
            tr_l, te_l, tr_a, te_a = [], [], [], []
            for _ in range(5):
                nn = NeuralNetworkClassifier(input_size=input_features, hidden_size=10)
                nn.W1 = np.random.randn(*nn.W1.shape) * scale
                nn.W2 = np.random.randn(*nn.W2.shape) * scale
                for _ in range(1000):
                    nn.forward(X_train); nn.backward(X_train, Y_train, 0.01)
                A2_tr=nn.forward(X_train); A2_te=nn.forward(X_test)
                tr_l.append(nn.compute_loss(A2_tr,Y_train)); te_l.append(nn.compute_loss(A2_te,Y_test))
                tr_a.append(nn.compute_accuracy(A2_tr,Y_train)); te_a.append(nn.compute_accuracy(A2_te,Y_test))
            results5.append({
                'Wartość parametru (init_scale)':    scale,
                'Średnia dokładność – Uczący [%]':   round(np.mean(tr_a),2),
                'Najlepsza dokładność – Uczący [%]': round(np.max(tr_a),2),
                'Średnia dokładność – Testowy [%]':  round(np.mean(te_a),2),
                'Najlepsza dokładność – Testowy [%]':round(np.max(te_a),2),
                'Średni BCE – Uczący':               round(np.mean(tr_l),4),
                'Średni BCE – Testowy':              round(np.mean(te_l),4),
            })
        print("\nWYNIKI:"); print(pd.DataFrame(results5).to_markdown(index=False))

        # ------------------------------------------------------------------
        # EKSPERYMENT 6: Liczba warstw ukrytych (architektura)
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 6: Liczba warstw ukrytych (architektura sieci)")
        print("="*65)
        architectures = [
            ('1 warstwa: 5 neuronów',    'single', 5),
            ('1 warstwa: 10 neuronów',   'single', 10),
            ('2 warstwy: 10+5 neuronów', 'double', (10,5)),
            ('2 warstwy: 10+10 neuronów','double', (10,10)),
        ]
        results6 = []
        print(f"\nBadany parametr: architektura sieci")
        for label, arch_type, sizes in architectures:
            print(f"  Testowanie: {label} ...")
            tr_l, te_l, tr_a, te_a = [], [], [], []
            for _ in range(5):
                if arch_type == 'single':
                    nn = NeuralNetworkClassifier(input_size=input_features, hidden_size=sizes)
                    for _ in range(1000):
                        nn.forward(X_train); nn.backward(X_train,Y_train,0.01)
                    A2_tr=nn.forward(X_train); A2_te=nn.forward(X_test)
                    tr_l.append(nn.compute_loss(A2_tr,Y_train)); te_l.append(nn.compute_loss(A2_te,Y_test))
                    tr_a.append(nn.compute_accuracy(A2_tr,Y_train)); te_a.append(nn.compute_accuracy(A2_te,Y_test))
                else:
                    nn = NeuralNetworkClassifier2Hidden(input_features, sizes[0], sizes[1])
                    for _ in range(1000):
                        nn.forward(X_train); nn.backward(X_train,Y_train,0.01)
                    A3_tr=nn.forward(X_train); A3_te=nn.forward(X_test)
                    tr_l.append(nn.compute_loss(A3_tr,Y_train)); te_l.append(nn.compute_loss(A3_te,Y_test))
                    tr_a.append(nn.compute_accuracy(A3_tr,Y_train)); te_a.append(nn.compute_accuracy(A3_te,Y_test))
            results6.append({
                'Wartość parametru (architektura)':  label,
                'Średnia dokładność – Uczący [%]':   round(np.mean(tr_a),2),
                'Najlepsza dokładność – Uczący [%]': round(np.max(tr_a),2),
                'Średnia dokładność – Testowy [%]':  round(np.mean(te_a),2),
                'Najlepsza dokładność – Testowy [%]':round(np.max(te_a),2),
                'Średni BCE – Uczący':               round(np.mean(tr_l),4),
                'Średni BCE – Testowy':              round(np.mean(te_l),4),
            })
        print("\nWYNIKI:"); print(pd.DataFrame(results6).to_markdown(index=False))

        # ------------------------------------------------------------------
        # EKSPERYMENT 7: Rozmiar mini-batcha
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 7: Rozmiar mini-batcha (batch size)")
        print("="*65)
        batch_sizes = [16, 64, 256, X_train.shape[1]]
        results7 = []
        print(f"\nBadany parametr: batch_size")
        for bs in batch_sizes:
            label = bs if bs != X_train.shape[1] else f"{bs} (full batch)"
            print(f"  Testowanie: batch_size = {label} ...")
            tr_l, te_l, tr_a, te_a = [], [], [], []
            for _ in range(5):
                n_samples = X_train.shape[1]
                nn = NeuralNetworkClassifier(input_size=input_features, hidden_size=10)
                for ep in range(1000):
                    idx = np.random.permutation(n_samples)
                    for start in range(0, n_samples, bs):
                        b = idx[start:start+bs]
                        nn.forward(X_train[:,b]); nn.backward(X_train[:,b], Y_train[:,b], 0.01)
                A2_tr=nn.forward(X_train); A2_te=nn.forward(X_test)
                tr_l.append(nn.compute_loss(A2_tr,Y_train)); te_l.append(nn.compute_loss(A2_te,Y_test))
                tr_a.append(nn.compute_accuracy(A2_tr,Y_train)); te_a.append(nn.compute_accuracy(A2_te,Y_test))
            results7.append({
                'Wartość parametru (batch_size)':    str(label),
                'Średnia dokładność – Uczący [%]':   round(np.mean(tr_a),2),
                'Najlepsza dokładność – Uczący [%]': round(np.max(tr_a),2),
                'Średnia dokładność – Testowy [%]':  round(np.mean(te_a),2),
                'Najlepsza dokładność – Testowy [%]':round(np.max(te_a),2),
                'Średni BCE – Uczący':               round(np.mean(tr_l),4),
                'Średni BCE – Testowy':              round(np.mean(te_l),4),
            })
        print("\nWYNIKI:"); print(pd.DataFrame(results7).to_markdown(index=False))

        # ------------------------------------------------------------------
        # EKSPERYMENT 8: Regularyzacja L2 (lambda)
        # ------------------------------------------------------------------
        print("\n" + "="*65)
        print("EKSPERYMENT 8: Regularyzacja L2 (lambda)")
        print("="*65)
        t8 = run_experiments(X_train,Y_train,X_test,Y_test,
                             'lambda_l2',[0.0,0.001,0.01,0.1],
                             n_repeats=5,epochs=1000,lr=0.01)
        print("\nWYNIKI:"); print(t8.to_markdown(index=False))

        print("\n✅ Wszystkie eksperymenty zakończone.")

    except Exception as e:
        print(f"\nWystąpił błąd: {e}")
        import traceback; traceback.print_exc()
