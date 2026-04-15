import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# =====================================================================
# 1. FUNKCJA PRZYGOTOWUJĄCA DANE
# =====================================================================
def load_and_prepare_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BŁĄD: Brak pliku {filepath}!")
        
    df = pd.read_csv(filepath)
    
    # Zakładamy, że ostatnia kolumna to nasza zmienna docelowa (Y) - np. cena Close
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Podział na zbiór uczący (80%) i testowy (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standaryzacja danych
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# =====================================================================
# 2. SILNIK EKSPERYMENTÓW ML
# =====================================================================
def run_ml_experiments(X_train, X_test, y_train, y_test, model_class, param_name, param_values):
    results = []
    
    for val in param_values:
        # Konfiguracja modelu z odpowiednim parametrem
        # Dodajemy random_state tam, gdzie to możliwe, aby wyniki były powtarzalne
        if 'random_state' in model_class().get_params():
            params = {param_name: val, 'random_state': 42}
        else:
            params = {param_name: val}
            
        model = model_class(**params)
        
        # Trening modelu
        model.fit(X_train, y_train)
        
        # Predykcje
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Obliczanie błędu MSE (dla regresji)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
            
        results.append({
            'Model': model.__class__.__name__,
            f'Badany parametr': f"{param_name} = {val}",
            'MSE (Uczący)': train_mse,
            'MSE (Testowy)': test_mse
        })
        
    return pd.DataFrame(results)

# =====================================================================
# 3. URUCHOMIENIE ANALIZY DLA REGRESJI
# =====================================================================
if __name__ == "__main__":
    print("="*70)
    print("CZĘŚĆ 2 PROJEKTU - UCZENIE MASZYNOWE (TYLKO REGRESJA)")
    print("="*70)
    nazwa_pliku = 'dane_regresja.csv'
    
    try:
        print(f"\n[Wczytywanie danych z pliku {nazwa_pliku}...]")
        X_train, X_test, y_train, y_test = load_and_prepare_data(nazwa_pliku)
        
        # Definiujemy 4 metody i po 4 wartości parametrów do przetestowania
        reg_experiments = [
            (KNeighborsRegressor, 'n_neighbors', [3, 5, 7, 9]),               # KNN: Liczba sąsiadów
            (SVR, 'kernel', ['linear', 'poly', 'rbf', 'sigmoid']),            # SVM: Rodzaj jądra
            (RandomForestRegressor, 'n_estimators', [10, 50, 100, 200]),      # Las Losowy: Liczba drzew
            (DecisionTreeRegressor, 'max_depth', [3, 5, 10, None])            # Drzewo: Maksymalna głębokość
        ]
        
        df_all_results = pd.DataFrame()
        
        print("[Trwa trenowanie modeli i badanie parametrów... To może potrwać chwilę.]")
        for model_class, param_name, param_values in reg_experiments:
            df_res = run_ml_experiments(X_train, X_test, y_train, y_test, model_class, param_name, param_values)
            df_all_results = pd.concat([df_all_results, df_res], ignore_index=True)
            
        print("\n" + "="*70)
        print("WYNIKI DLA REGRESJI (Błąd Średniokwadratowy - MSE):")
        print("="*70)
        # Znowu używamy formatowania markdown do ładnej tabelki
        print(df_all_results.to_markdown(index=False))
        
    except Exception as e:
        print(f"\nWystąpił błąd: {e}")