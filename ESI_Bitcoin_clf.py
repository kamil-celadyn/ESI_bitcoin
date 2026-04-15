import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# =====================================================================
# 1. FUNKCJA PRZYGOTOWUJĄCA DANE
# =====================================================================
def load_and_prepare_data(filepath):
    """
    Wczytuje dane Bitcoina i tworzy problem klasyfikacji binarnej:
    Y = 1 jeśli cena Close jest wyższa niż Open tego samego dnia (wzrost),
    Y = 0 w przeciwnym razie (brak wzrostu / spadek).
    Podział chronologiczny – bez data leakage.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BŁĄD: Brak pliku {filepath}!")

    df = pd.read_csv(filepath)

    # Tworzenie etykiety klasyfikacyjnej
    df['Target'] = (df['Close'] > df['Open']).astype(int)

    # Cechy: Open, High, Low, Volume (bez Close, żeby nie było wycieku)
    feature_cols = ['Open', 'High', 'Low', 'Volume']
    X = df[feature_cols].values
    y = df['Target'].values

    # Podział chronologiczny 80/20
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Liczba przykładów uczących : {len(X_train)}")
    print(f"Liczba przykładów testowych: {len(X_test)}")
    print(f"Rozkład klas (uczący) – wzrost: {y_train.sum()}, spadek: {len(y_train) - y_train.sum()}")

    # Standaryzacja – parametry tylko ze zbioru uczącego
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# =====================================================================
# 2. SILNIK EKSPERYMENTÓW ML
# =====================================================================
def run_ml_experiments(X_train, X_test, y_train, y_test, model_class, param_name, param_values):
    results = []

    for val in param_values:
        if 'random_state' in model_class().get_params():
            params = {param_name: val, 'random_state': 42}
        else:
            params = {param_name: val}

        model = model_class(**params)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred) * 100
        test_acc  = accuracy_score(y_test,  y_test_pred)  * 100

        results.append({
            'Model':            model.__class__.__name__,
            'Badany parametr':  f"{param_name} = {val}",
            'Dokładność (Uczący) [%]':  round(train_acc, 2),
            'Dokładność (Testowy) [%]': round(test_acc,  2),
        })

    return pd.DataFrame(results)


# =====================================================================
# 3. URUCHOMIENIE ANALIZY DLA KLASYFIKACJI
# =====================================================================
if __name__ == "__main__":
    print("="*70)
    print("CZĘŚĆ 2 PROJEKTU – UCZENIE MASZYNOWE (KLASYFIKACJA)")
    print("="*70)
    nazwa_pliku = 'dane_regresja.csv'

    try:
        print(f"\n[Wczytywanie danych z pliku {nazwa_pliku}...]")
        X_train, X_test, y_train, y_test = load_and_prepare_data(nazwa_pliku)

        # 4 metody klasyfikacji, każda z 4 wartościami parametru
        clf_experiments = [
            (KNeighborsClassifier, 'n_neighbors', [3, 5, 7, 9]),
            (SVC,                  'kernel',       ['linear', 'poly', 'rbf', 'sigmoid']),
            (RandomForestClassifier,'n_estimators',[10, 50, 100, 200]),
            (DecisionTreeClassifier,'max_depth',   [3, 5, 10, None]),
        ]

        df_all_results = pd.DataFrame()

        print("[Trwa trenowanie modeli i badanie parametrów... To może potrwać chwilę.]\n")
        for model_class, param_name, param_values in clf_experiments:
            df_res = run_ml_experiments(
                X_train, X_test, y_train, y_test,
                model_class, param_name, param_values
            )
            df_all_results = pd.concat([df_all_results, df_res], ignore_index=True)

        print("\n" + "="*70)
        print("WYNIKI DLA KLASYFIKACJI (Dokładność – Accuracy):")
        print("="*70)
        print(df_all_results.to_markdown(index=False))

    except Exception as e:
        print(f"\nWystąpił błąd: {e}")
