import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

if len(sys.argv) < 2:
    print("Uso: python trasponi.py <file_trasposto.xlsx> [file_output.xlsx]")
    sys.exit(1)

path_in  = sys.argv[1]
path_out = sys.argv[2] if len(sys.argv) > 2 else path_in.rsplit('.', 1)[0] + '_tr.xlsx'

df_raw = pd.read_excel(path_in, header=None)
print(f"File letto: {df_raw.shape[0]} righe × {df_raw.shape[1]} colonne")

# Struttura attesa:
#   riga 0, col 0 : etichetta "Samples" (o simile)
#   riga 0, col 1+: nomi campioni (F1, F2, …, SD6)
#   riga 1, col 0 : etichetta "Groups" (o simile)
#   riga 1, col 1+: nomi gruppo (F, F, F, …, SD, SD)
#   righe 2+, col 0   : nomi variabili (composti)
#   righe 2+, col 1+  : valori numerici

data_cols    = list(df_raw.columns[1:])          # tutte le colonne dati
sample_names = [str(v) for v in df_raw.iloc[0, 1:].values]
groups       = [str(v) for v in df_raw.iloc[1, 1:].values]

# Righe variabili: dalla riga 2, con nome non vuoto in col 0
var_rows = df_raw.iloc[2:].copy()
valid    = (var_rows.iloc[:, 0].notna() &
            (var_rows.iloc[:, 0].astype(str).str.strip() != ''))
var_rows = var_rows[valid]

var_names   = var_rows.iloc[:, 0].astype(str).str.strip().tolist()
data_matrix = var_rows[data_cols].apply(pd.to_numeric, errors='coerce').values
# shape: (n_variabili, n_campioni) → trasposta → (n_campioni, n_variabili)

result = pd.DataFrame(data_matrix.T, columns=var_names)
result.insert(0, 'Campione', sample_names)
result.insert(1, 'Gruppo',   groups)

result.to_excel(path_out, index=False)

print(f"Salvato: {path_out}")
print(f"Shape:   {result.shape[0]} campioni × {result.shape[1]} colonne")
print(f"Campioni: {list(result['Campione'])}")
print(f"Gruppi unici: {sorted(result['Gruppo'].unique())}")
print(f"\nPrime 6 righe (prime 4 colonne):")
print(result.iloc[:6, :4].to_string())
