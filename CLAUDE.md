# PCA GUI — Note per Claude

## Progetto
GUI desktop in Python (tkinter) per l'Analisi delle Componenti Principali.
File unico: `pca_gui.pyw` — classe `PCAApp`.
Avviato con doppio clic su Windows (`.pyw` = nessuna finestra console).

## Dipendenze
```
numpy, pandas, scikit-learn, matplotlib, openpyxl
adjustText  # opzionale — evita sovrapposizione etichette nel biplot
```

## Struttura della classe PCAApp

| Metodo | Responsabilità |
|---|---|
| `__init__` | Tutti i `tk.Var` (stato dell'app) |
| `_build_ui` | Costruisce l'intera UI; pannello sinistro + pannello destro scorrevoli |
| `_load_file` | Legge il file raw (header=None) in `self.df_raw`, chiama `_apply_orientation` |
| `_load_example` | Carica dataset sintetico 3×6×8, apre spreadsheet |
| `_apply_orientation` | Trasforma `df_raw` → `self.df` in base al flag righe/colonne |
| `_normalize_rows` | Formato standard: prima riga = intestazioni |
| `_transpose_data` | Formato trasposto: campioni per colonne → DataFrame standard |
| `_populate_controls` | Aggiorna checkbox variabili, campioni, colori dopo caricamento |
| `_update_samples` | Rigenera la lista campioni da escludere |
| `_update_colors` | Rigenera i color picker per gruppo |
| `_run_pca` | Wrapper try/except che chiama `_run_pca_inner` |
| `_run_pca_inner` | Pre-elaborazione + PCA + disegno 3 grafici |
| `_on_canvas_resize` | Ridisegna la figura ad ogni resize della finestra |
| `_draw_ax_A` | Disegna **scree plot** su un axes dato |
| `_draw_ax_B` | Disegna **loadings bar chart** su un axes dato |
| `_draw_ax_C` | Disegna **biplot** su un axes dato |
| `_draw_pca` | Compone i tre pannelli nella figura (usa i tre metodi sopra) |
| `_save_figure` | Salva figura completa in PDF + PNG + SVG |
| `_save_panels` | Salva i tre pannelli come file separati (PDF + PNG + SVG × 3) |
| `_export_data` | Esporta scores/loadings/varianza in Excel (3 fogli) |
| `_export_biplot_excel` | Esporta dati biplot-ready per Excel (6 fogli + istruzioni) |
| `_open_data_view` | Apre Toplevel con spreadsheet editabile |
| `_log` / `_show_log` | Log interno con finestra stile terminale (WARNING in giallo, ERROR in rosso) |

## Layout UI
- **Sinistra** (290px, scorrevole): carica file, orientamento, bottoni azione, colonne speciali, pre-elaborazione, variabili, campioni da escludere
- **Destra** (270px, scorrevole): colori gruppi, colori PC, componenti, opzioni grafici
- **Centro**: figura matplotlib embedded (`FigureCanvasTkAgg`)
- **Basso**: barra di stato

### Pannello sinistro — sezioni dall'alto
1. Carica file + selettore orientamento (righe / colonne)
2. Bottoni: Visualizza/Modifica dati · Dati esempio
3. Bottoni azione: Esegui PCA · Salva figura · Salva pannelli · Esporta dati · Esporta biplot Excel · Log/Messaggi
4. Colonne speciali (campioni, gruppi)
5. Pre-elaborazione (scalatura, n. componenti)
6. Variabili per la PCA (checkbox scrollabili con lettera a–z)
7. Campioni da escludere (checkbox scrollabili)

### Pannello destro — sezioni dall'alto
1. Colori gruppi (color picker per gruppo)
2. Colori PC (color picker per PC_A e PC_B del grafico B)
3. Componenti da visualizzare (Biplot X/Y, Loadings A/B)
4. Opzioni grafici (soglia loading, sigma ellisse, soglia scree, max PC scree, scala frecce, dim. marker, checkbox: etichette campioni / frecce+etichette variabili / ellissi gruppi / biplot scala 1:1)

## Stato persistente
- `self.df` — DataFrame corrente (già trasformato secondo l'orientamento)
- `self.df_raw` — dati grezzi letti con `header=None` (conservati per ri-applicare l'orientamento senza riaprire il file)
- `self._loaded_path` — percorso dell'ultimo file caricato
- `self.data_orientation` — `tk.StringVar`: `'righe'` | `'colonne'`
- `self.equal_aspect` — `tk.BooleanVar(value=True)`: biplot scala 1:1
- `self._last_pca` — dict con tutti i risultati dell'ultima PCA:
  `scores, loadings, var_exp, features, samples, types, colors, group_markers,`
  `threshold, bx, by, la, lb, sigma, n_pc, groups, lx, ly, scale, msize,`
  `show_slbl, show_vlbl, show_ell, scree_thr, scree_max, plot_title,`
  `pc_color_a, pc_color_b, equal_aspect, var_letters`
- `self.group_color_vars` — dict `{gruppo: hex_color}`
- `self._var_letters` — dict `{col: lettera}` (a–z poi A–Z), assegnato in `_populate_controls`
- `self._log_lines`, `self._log_win` — log interno

## Grafici prodotti — layout GridSpec 2×2

```
┌──────────────┬──────────────────────────┐
│ A — Scree    │  B — Loadings            │  height_ratio 1, width_ratio [1, 2.5]
├──────────────┴──────────────────────────┤
│         C — Biplot (pieno)              │  height_ratio 1.5
└─────────────────────────────────────────┘
```

- **A** (riga 0, col 0): Scree plot — barre varianza + curva cumulata + linee PC selezionate
- **B** (riga 0, col 1): Loadings bar chart PC_a e PC_b — titolo "B | Loadings"
- **C** (riga 1, intera): Biplot — scores + frecce loadings + etichette lettere + ellissi sigma — titolo "C | Biplot"

### Ordine logico di lettura
Scree → Loadings → Biplot (spiega quante PC, cosa significano, poi dove cadono i campioni)

### Layout tecnico
- `tight_layout(pad=1.2)` — ricalcolato ad ogni resize tramite `_on_canvas_resize`
- **Biplot scala 1:1**: `ax.set_aspect('equal', adjustable='box')` — ridimensiona il box fisico
  mantenendo i `xlim/ylim` invariati; le etichette restano sempre entro i limiti calcolati.
  **Non usare `adjustable='datalim'`**: con `tight_layout` cambia i limiti in modo
  imprevedibile causando etichette fuori dal grafico.
- **adjustText**: dopo la chiamata, ripristina `xlim/ylim` e clamp le posizioni delle
  etichette con `np.clip(..., xmin+4%, xmax-4%)` per evitare overflow.

## Etichette variabili nel biplot
- Ogni variabile numerica riceve una lettera (a–z, poi A–Z) in `_populate_controls`
- Nella checkbox: `"a — NomeVariabile"`
- Nel biplot: la lettera appare all'estremità della freccia (fontsize=8, bold)
- `_var_letters` è salvato in `_last_pca['var_letters']` per coerenza tra ridisegni

## Marcatori gruppi
`['o', 's', '^', 'D', 'v', 'h']` — fino a 6 gruppi con simboli distinti.
La legenda mostra sia il colore che il simbolo (`plt.Line2D` con marker).

## Metodi di scalatura (dizionario `SCALERS`)
`z-score`, `centratura`, `min-max`, `robust`, `nessuna`
Aggiungere nuovi metodi solo qui, senza toccare `_run_pca_inner`.

## Gestione valori mancanti
In `_run_pca_inner`, prima della scalatura:
- conta NaN con `np.isnan(X).sum()`
- imputa con la media di colonna (`np.nanmean`)
- logga un WARNING con il numero di celle imputate
- aggiunge avviso in barra di stato

## Fogli Excel esportati da `_export_biplot_excel`
| Foglio | Contenuto |
|---|---|
| Scores_per_gruppo | Wide format, una colonna PC per gruppo |
| Frecce_loadings | Coppie (0,0)→(lx·scala, ly·scala) + separatori NaN |
| Ellissi_gruppi | 361 punti parametrici per gruppo + separatori NaN |
| Loadings_barchart | Loadings PC_a e PC_b + flag dominante |
| Varianza | Varianza spiegata e cumulata |
| Istruzioni | Guida passo-passo con parametri usati |

## Convenzioni
- Indici PC sempre **0-based** internamente; UI e label usano 1-based
- `bx`, `by` = indici 0-based degli assi biplot; `la`, `lb` = loadings chart
- Mousewheel: `bind_all` attivato su `<Enter>` del canvas, rimosso su `<Leave>`
- I canvas interni (variabili, campioni) reindirizzano il mousewheel a sé stessi
- Rilevamento automatico colonne in `_populate_controls`: primo match vince;
  keyword: `['campion', 'sample', 'campione']` per campioni,
  `['tipo', 'type', 'group', 'gruppo']` per gruppi.
  **Non usare 'id' o 'name'** (sottostringhe di nomi variabili es. "acid").

## Formati file Excel supportati

### Formato standard — modalità "righe"
Prima riga = intestazioni, righe successive = un campione per riga.
```
Campione | Gruppo | Variabile1 | Variabile2 | …
F1       | F      | 38.5       | 466.4      | …
F2       | F      | 0.03       | 370.7      | …
```

### Formato trasposto — modalità "colonne"
Campioni disposti per colonne. Struttura rilevata automaticamente da `_transpose_data`:
```
Samples  | F1   | F2   | … | SD6
Groups   | F    | F    | … | SD
Composto1| 38.5 | 0.03 | … | 0.07
Composto2| 466  | 370  | … | 240
```
- **Struttura A** (rilevata se riga 1, col 0 contiene 'group'/'gruppo'):
  riga 0 = nomi campioni, riga 1 = nomi gruppo, righe 2+ = variabili
- **Struttura B** (altrimenti):
  riga 0 = nomi gruppo (con eventuali NaN separatori), riga 1 = nomi campioni

### Script di conversione
`trasponi.py` — converte il file trasposto (struttura A) in formato standard,
salva `*_tr.xlsx`. Utile per condividere dati con colleghi.
