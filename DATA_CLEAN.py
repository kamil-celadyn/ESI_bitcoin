import pandas as pd
df = pd.read_csv('dane_regresja.csv')
new_df = df.dropna()
nowa_kolejnosc = ['Open', 'High', 'Low', 'Volume', 'Close']
df = df[nowa_kolejnosc]
df.to_csv('dane_regresja.csv', index=False)