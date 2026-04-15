import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


# =====================================================================
# 1. WCZYTYWANIE DANYCH
# =====================================================================
def load_regression_data(filepath):
    """
    Regresja: przewidywanie ceny Close na podstawie Open, High, Low, Volume.
    Podział chronologiczny 80/20 – bez data leakage.
    """
    df = pd.read_csv(filepath)
    split_idx = int(len(df) * 0.8)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"  Zbiór uczący : {len(X_train)} przykładów")
    print(f"  Zbiór testowy: {len(X_test)} przykładów")
    return X_train_s, X_test_s, y_train, y_test


def load_classification_data(filepath):
    """
    Klasyfikacja binarna: Y=1 gdy Close > Open (wzrost), Y=0 w przeciwnym razie.
    Podział chronologiczny 80/20 – bez data leakage.
    """
    df = pd.read_csv(filepath)
    df['Target'] = (df['Close'] > df['Open']).astype(int)

    feature_cols = ['Open', 'High', 'Low', 'Volume']
    X = df[feature_cols].values
    y = df['Target'].values

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"  Zbiór uczący : {len(X_train)} przykładów")
    print(f"  Zbiór testowy: {len(X_test)} przykładów")
    print(f"  Rozkład klas (uczący) – wzrost: {y_train.sum()}, spadek: {len(y_train) - y_train.sum()}")
    return X_train_s, X_test_s, y_train, y_test


# =====================================================================
# 2. SILNIK EKSPERYMENTÓW
# =====================================================================
def run_regression_experiments(X_train, X_test, y_train, y_test,
                                model_class, param_name, param_values):
    results = []
    for val in param_values:
        try:
            if 'random_state' in model_class().get_params():
                params = {param_name: val, 'random_state': 42}
            else:
                params = {param_name: val}
            model = model_class(**params)
            model.fit(X_train, y_train)
            train_mse = mean_squared_error(y_train, model.predict(X_train))
            test_mse  = mean_squared_error(y_test,  model.predict(X_test))
        except Exception as e:
            train_mse = test_mse = float('nan')
        results.append({
            'Model':         model_class.__name__,
            'Parametr':      f"{param_name} = {val}",
            'MSE (Uczący)':  round(train_mse, 2),
            'MSE (Testowy)': round(test_mse,  2),
        })
    return pd.DataFrame(results)


def run_classification_experiments(X_train, X_test, y_train, y_test,
                                    model_class, param_name, param_values):
    results = []
    for val in param_values:
        try:
            if 'random_state' in model_class().get_params():
                params = {param_name: val, 'random_state': 42}
            else:
                params = {param_name: val}
            model = model_class(**params)
            model.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
            test_acc  = accuracy_score(y_test,  model.predict(X_test))  * 100
        except Exception as e:
            train_acc = test_acc = float('nan')
        results.append({
            'Model':                    model_class.__name__,
            'Parametr':                 f"{param_name} = {val}",
            'Dokładność – Uczący [%]':  round(train_acc, 2),
            'Dokładność – Testowy [%]': round(test_acc,  2),
        })
    return pd.DataFrame(results)


# =====================================================================
# 3. GŁÓWNA CZĘŚĆ PROGRAMU
# =====================================================================
if __name__ == "__main__":
    nazwa_pliku = 'dane_regresja.csv'

    if not os.path.exists(nazwa_pliku):
        raise FileNotFoundError(f"BŁĄD: Brak pliku {nazwa_pliku}!")

    # ==================================================================
    # CZĘŚĆ A – REGRESJA
    # ==================================================================
    print("="*70)
    print("CZĘŚĆ 2A – UCZENIE MASZYNOWE: REGRESJA")
    print("Cel: przewidywanie ceny zamknięcia (Close) Bitcoina")
    print("="*70)

    X_train_r, X_test_r, y_train_r, y_test_r = load_regression_data(nazwa_pliku)

    reg_experiments = [
        (KNeighborsRegressor,   'n_neighbors', [3, 5, 7, 9]),
        (SVR,                   'kernel',      ['linear', 'poly', 'rbf', 'sigmoid']),
        (RandomForestRegressor, 'n_estimators',[10, 50, 100, 200]),
        (DecisionTreeRegressor, 'max_depth',   [3, 5, 10, None]),
    ]

    all_reg = pd.DataFrame()
    print("\n[Trwa trenowanie modeli regresji...]\n")
    for model_class, param_name, param_values in reg_experiments:
        print(f">>> {model_class.__name__} – badany parametr: {param_name}")
        df = run_regression_experiments(
            X_train_r, X_test_r, y_train_r, y_test_r,
            model_class, param_name, param_values
        )
        print(df.to_markdown(index=False))
        print()
        all_reg = pd.concat([all_reg, df], ignore_index=True)

    print("\nZBIORCZE WYNIKI – REGRESJA:")
    print(all_reg.to_markdown(index=False))

    # ==================================================================
    # CZĘŚĆ B – KLASYFIKACJA
    # ==================================================================
    print("\n" + "="*70)
    print("CZĘŚĆ 2B – UCZENIE MASZYNOWE: KLASYFIKACJA")
    print("Cel: przewidywanie kierunku kursu Bitcoina (wzrost/spadek)")
    print("="*70)

    X_train_c, X_test_c, y_train_c, y_test_c = load_classification_data(nazwa_pliku)

    clf_experiments = [
        (KNeighborsClassifier,   'n_neighbors', [3, 5, 7, 9]),
        (SVC,                    'kernel',      ['linear', 'poly', 'rbf', 'sigmoid']),
        (RandomForestClassifier, 'n_estimators',[10, 50, 100, 200]),
        (DecisionTreeClassifier, 'max_depth',   [3, 5, 10, None]),
    ]

    all_clf = pd.DataFrame()
    print("\n[Trwa trenowanie modeli klasyfikacji...]\n")
    for model_class, param_name, param_values in clf_experiments:
        print(f">>> {model_class.__name__} – badany parametr: {param_name}")
        df = run_classification_experiments(
            X_train_c, X_test_c, y_train_c, y_test_c,
            model_class, param_name, param_values
        )
        print(df.to_markdown(index=False))
        print()
        all_clf = pd.concat([all_clf, df], ignore_index=True)

    print("\nZBIORCZE WYNIKI – KLASYFIKACJA:")
    print(all_clf.to_markdown(index=False))

    print("\n✅ Wszystkie eksperymenty zakończone.")
