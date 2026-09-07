#!/usr/bin/env python3
"""
PCA GUI — Analisi delle Componenti Principali
Autore: generato con Claude per Francesco
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

SCALERS = {
    'z-score':    lambda: StandardScaler(),
    'centratura': lambda: StandardScaler(with_std=False),
    'min-max':    lambda: MinMaxScaler(),
    'robust':     lambda: RobustScaler(),
    'nessuna':    None,
}


class PCAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PCA — Analisi Componenti Principali")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f5f5f5')

        self.df = None
        self.df_raw = None          # dati grezzi (header=None) per ri-applicare l'orientamento
        self._loaded_path = None    # percorso ultimo file caricato
        self.data_orientation = tk.StringVar(value='righe')   # 'righe' | 'colonne'
        self.features = []
        self.var_checks = {}
        self.sample_checks = {}

        # Colonne speciali
        self.group_col  = tk.StringVar()
        self.sample_col = tk.StringVar()

        # Titolo figura (plain str, impostato al caricamento del file)
        self.plot_title = "PCA"

        # Selezione componenti
        self.pc_biplot_x = tk.IntVar(value=1)
        self.pc_biplot_y = tk.IntVar(value=2)
        self.pc_load_a   = tk.IntVar(value=1)
        self.pc_load_b   = tk.IntVar(value=2)

        # Pre-elaborazione
        self.scaling_method = tk.StringVar(value='z-score')
        self.n_components   = tk.IntVar(value=0)          # 0 = tutte

        # Opzioni grafici
        self.threshold         = tk.DoubleVar(value=0.3)
        self.ellipse_sigma     = tk.IntVar(value=2)
        self.scree_threshold   = tk.IntVar(value=70)
        self.scree_max_pc      = tk.IntVar(value=8)
        self.biplot_scale      = tk.DoubleVar(value=8.0)
        self.marker_size       = tk.IntVar(value=40)
        self.show_sample_labels = tk.BooleanVar(value=True)
        self.show_var_labels    = tk.BooleanVar(value=True)
        self.show_ellipses      = tk.BooleanVar(value=True)
        self.equal_aspect       = tk.BooleanVar(value=True)

        # Colori barre grafico C
        self.pc_color_a = '#008000'   # verde  R=0   G=128 B=0
        self.pc_color_b = '#FF8000'   # arancio R=255 G=128 B=0

        # Log
        self._log_lines = []
        self._log_win   = None

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Pannello sinistro (scorrevole) ───────────────────────────
        left_outer = tk.Frame(self.root, bg='#f5f5f5', width=290)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)
        left_outer.pack_propagate(False)

        _lsb = ttk.Scrollbar(left_outer, orient='vertical')
        _lsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._ctrl_canvas = tk.Canvas(left_outer, bg='#f5f5f5',
                                      highlightthickness=0,
                                      yscrollcommand=_lsb.set)
        self._ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _lsb.config(command=self._ctrl_canvas.yview)

        left = tk.Frame(self._ctrl_canvas, bg='#f5f5f5')
        _lwin = self._ctrl_canvas.create_window((0, 0), window=left, anchor='nw')
        left.bind('<Configure>', lambda e: self._ctrl_canvas.configure(
            scrollregion=self._ctrl_canvas.bbox('all')))
        self._ctrl_canvas.bind('<Configure>',
                               lambda e: self._ctrl_canvas.itemconfig(_lwin, width=e.width))

        def _mw_ctrl(ev):
            self._ctrl_canvas.yview_scroll(int(-1 * (ev.delta / 120)), 'units')
        self._ctrl_canvas.bind('<Enter>',
                               lambda e: self._ctrl_canvas.bind_all('<MouseWheel>', _mw_ctrl))
        self._ctrl_canvas.bind('<Leave>',
                               lambda e: self._ctrl_canvas.unbind_all('<MouseWheel>'))

        # ── Pannello destro opzioni (scorrevole) ─────────────────────
        right_outer = tk.Frame(self.root, bg='#f5f5f5', width=270)
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        right_outer.pack_propagate(False)

        _rsb = ttk.Scrollbar(right_outer, orient='vertical')
        _rsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._right_canvas = tk.Canvas(right_outer, bg='#f5f5f5',
                                       highlightthickness=0,
                                       yscrollcommand=_rsb.set)
        self._right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _rsb.config(command=self._right_canvas.yview)

        rpanel = tk.Frame(self._right_canvas, bg='#f5f5f5')
        _rwin = self._right_canvas.create_window((0, 0), window=rpanel, anchor='nw')
        rpanel.bind('<Configure>', lambda e: self._right_canvas.configure(
            scrollregion=self._right_canvas.bbox('all')))
        self._right_canvas.bind('<Configure>',
                                lambda e: self._right_canvas.itemconfig(_rwin, width=e.width))

        def _mw_right(ev):
            self._right_canvas.yview_scroll(int(-1 * (ev.delta / 120)), 'units')
        self._right_canvas.bind('<Enter>',
                                lambda e: self._right_canvas.bind_all('<MouseWheel>', _mw_right))
        self._right_canvas.bind('<Leave>',
                                lambda e: self._right_canvas.unbind_all('<MouseWheel>'))

        # ── Figura centrale ──────────────────────────────────────────
        center = tk.Frame(self.root, bg='white')
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = plt.figure(figsize=(12, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('resize_event', self._on_canvas_resize)

        # ── Status bar ───────────────────────────────────────────────
        self.status = tk.Label(self.root, text="Carica un file per iniziare.",
                               font=('Helvetica', 9), bg='#ddd',
                               fg='#555', anchor='w', padx=8)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # ═══════════════ PANNELLO SINISTRO ═══════════════════════════

        tk.Label(left, text="PCA — Controllo", font=('Helvetica', 13, 'bold'),
                 bg='#f5f5f5', fg='#333').pack(anchor='w', pady=(0, 8))

        tk.Button(left, text="📂  Carica Excel / CSV",
                  command=self._load_file,
                  bg='#2166ac', fg='white', font=('Helvetica', 11, 'bold'),
                  relief=tk.FLAT, padx=10, pady=6,
                  cursor='hand2').pack(fill=tk.X)

        self.file_label = tk.Label(left, text="Nessun file caricato",
                                   font=('Helvetica', 9), bg='#f5f5f5',
                                   fg='#888', wraplength=270, anchor='w')
        self.file_label.pack(anchor='w', pady=(2, 2))

        # ── Orientamento dati ────────────────────────────────────────
        orient_frame = tk.LabelFrame(left, text="Orientamento dati",
                                     bg='#f5f5f5', font=('Helvetica', 9, 'bold'),
                                     fg='#555', padx=6, pady=4)
        orient_frame.pack(fill=tk.X, pady=(0, 4))
        tk.Label(orient_frame, text="Campioni disposti per:",
                 bg='#f5f5f5', font=('Helvetica', 9)).pack(anchor='w')
        orient_btn_row = tk.Frame(orient_frame, bg='#f5f5f5')
        orient_btn_row.pack(anchor='w')
        for val, lab, tip in [
            ('righe',   '↔  righe (standard)',   'Una riga = un campione'),
            ('colonne', '↕  colonne (trasposto)', 'Una colonna = un campione'),
        ]:
            tk.Radiobutton(orient_btn_row, text=lab, variable=self.data_orientation,
                           value=val, bg='#f5f5f5', font=('Helvetica', 9),
                           command=self._apply_orientation,
                           cursor='hand2').pack(anchor='w', pady=1)

        btn_row2 = tk.Frame(left, bg='#f5f5f5')
        btn_row2.pack(fill=tk.X, pady=(0, 4))
        tk.Button(btn_row2, text="📋  Visualizza / Modifica dati",
                  command=self._open_data_view,
                  bg='#555577', fg='white', font=('Helvetica', 9),
                  relief=tk.FLAT, padx=8, pady=4,
                  cursor='hand2').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(btn_row2, text="📄  Dati esempio",
                  command=self._load_example,
                  bg='#888855', fg='white', font=('Helvetica', 9),
                  relief=tk.FLAT, padx=6, pady=4,
                  cursor='hand2').pack(side=tk.LEFT)

        # ── Bottoni azione ───────────────────────────────────────────
        tk.Button(left, text="▶  Esegui PCA",
                  command=self._run_pca,
                  bg='#d4a017', fg='white', font=('Helvetica', 12, 'bold'),
                  relief=tk.FLAT, padx=10, pady=8,
                  cursor='hand2').pack(fill=tk.X, pady=(0, 3))

        tk.Button(left, text="💾  Salva figura (PDF + PNG + SVG)",
                  command=self._save_figure,
                  bg='#4d9b8f', fg='white', font=('Helvetica', 10),
                  relief=tk.FLAT, padx=10, pady=5,
                  cursor='hand2').pack(fill=tk.X, pady=(0, 3))

        tk.Button(left, text="🖼  Salva pannelli separati",
                  command=self._save_panels,
                  bg='#3a7a8f', fg='white', font=('Helvetica', 10),
                  relief=tk.FLAT, padx=10, pady=5,
                  cursor='hand2').pack(fill=tk.X, pady=(0, 3))

        tk.Button(left, text="📊  Esporta dati (Excel)",
                  command=self._export_data,
                  bg='#5a7a3a', fg='white', font=('Helvetica', 10),
                  relief=tk.FLAT, padx=10, pady=5,
                  cursor='hand2').pack(fill=tk.X, pady=(0, 3))

        tk.Button(left, text="📈  Esporta biplot per Excel",
                  command=self._export_biplot_excel,
                  bg='#7a3a5a', fg='white', font=('Helvetica', 10),
                  relief=tk.FLAT, padx=10, pady=5,
                  cursor='hand2').pack(fill=tk.X, pady=(0, 3))

        tk.Button(left, text="📋  Log / Messaggi",
                  command=self._show_log,
                  bg='#888888', fg='white', font=('Helvetica', 10),
                  relief=tk.FLAT, padx=10, pady=5,
                  cursor='hand2').pack(fill=tk.X, pady=(0, 8))

        ttk.Separator(left, orient='horizontal').pack(fill=tk.X, pady=(0, 8))

        # ── Colonne speciali ─────────────────────────────────────────
        col_frame = tk.LabelFrame(left, text="Colonne speciali",
                                  bg='#f5f5f5', font=('Helvetica', 10, 'bold'),
                                  fg='#555', padx=6, pady=6)
        col_frame.pack(fill=tk.X, pady=4)

        tk.Label(col_frame, text="Campioni:", bg='#f5f5f5',
                 font=('Helvetica', 9)).grid(row=0, column=0, sticky='w', pady=2)
        self.sample_menu = ttk.Combobox(col_frame, textvariable=self.sample_col,
                                        width=16, state='readonly')
        self.sample_menu.grid(row=0, column=1, padx=4)

        tk.Label(col_frame, text="Gruppi:", bg='#f5f5f5',
                 font=('Helvetica', 9)).grid(row=1, column=0, sticky='w', pady=2)
        self.group_menu = ttk.Combobox(col_frame, textvariable=self.group_col,
                                       width=16, state='readonly')
        self.group_menu.grid(row=1, column=1, padx=4)

        # ── Pre-elaborazione ─────────────────────────────────────────
        pre_frame = tk.LabelFrame(left, text="Pre-elaborazione",
                                  bg='#f5f5f5', font=('Helvetica', 10, 'bold'),
                                  fg='#555', padx=6, pady=6)
        pre_frame.pack(fill=tk.X, pady=4)

        tk.Label(pre_frame, text="Scalatura:", bg='#f5f5f5',
                 font=('Helvetica', 9)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Combobox(pre_frame, textvariable=self.scaling_method,
                     values=list(SCALERS.keys()),
                     state='readonly', width=12,
                     font=('Helvetica', 9)).grid(row=0, column=1, columnspan=2,
                                                  padx=4, sticky='w')

        tk.Label(pre_frame, text="N. componenti:", bg='#f5f5f5',
                 font=('Helvetica', 9)).grid(row=1, column=0, sticky='w', pady=2)
        tk.Spinbox(pre_frame, from_=0, to=50, textvariable=self.n_components,
                   width=4, font=('Helvetica', 9)).grid(row=1, column=1, padx=4, sticky='w')
        tk.Label(pre_frame, text="(0 = tutte)", bg='#f5f5f5',
                 font=('Helvetica', 8), fg='#888').grid(row=1, column=2, sticky='w')

        # ── Variabili ────────────────────────────────────────────────
        var_outer = tk.LabelFrame(left, text="Variabili per la PCA",
                                  bg='#f5f5f5', font=('Helvetica', 10, 'bold'),
                                  fg='#555', padx=6, pady=6)
        var_outer.pack(fill=tk.X, pady=4)

        btn_row = tk.Frame(var_outer, bg='#f5f5f5')
        btn_row.pack(fill=tk.X, pady=(0, 4))
        tk.Button(btn_row, text="Seleziona tutto", command=self._select_all,
                  font=('Helvetica', 8), bg='#eee', relief=tk.FLAT,
                  cursor='hand2').pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Deseleziona tutto", command=self._deselect_all,
                  font=('Helvetica', 8), bg='#eee', relief=tk.FLAT,
                  cursor='hand2').pack(side=tk.LEFT, padx=2)

        self.var_canvas = tk.Canvas(var_outer, bg='#f5f5f5',
                                    highlightthickness=0, height=200)
        _vsb = ttk.Scrollbar(var_outer, orient='vertical',
                              command=self.var_canvas.yview)
        self.var_frame = tk.Frame(self.var_canvas, bg='#f5f5f5')
        self.var_frame.bind('<Configure>',
                            lambda e: self.var_canvas.configure(
                                scrollregion=self.var_canvas.bbox('all')))
        self.var_canvas.create_window((0, 0), window=self.var_frame, anchor='nw')
        self.var_canvas.configure(yscrollcommand=_vsb.set)
        self.var_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)

        def _mw_var(ev):
            self.var_canvas.yview_scroll(int(-1 * (ev.delta / 120)), 'units')
        self.var_canvas.bind('<Enter>',
                             lambda e: self.var_canvas.bind_all('<MouseWheel>', _mw_var))
        self.var_canvas.bind('<Leave>',
                             lambda e: self._ctrl_canvas.bind_all('<MouseWheel>', _mw_ctrl))

        # ── Campioni da escludere ────────────────────────────────────
        excl_frame = tk.LabelFrame(left, text="Campioni da escludere",
                                   bg='#f5f5f5', font=('Helvetica', 10, 'bold'),
                                   fg='#555', padx=6, pady=6)
        excl_frame.pack(fill=tk.X, pady=4)

        self.excl_canvas = tk.Canvas(excl_frame, bg='#f5f5f5',
                                     highlightthickness=0, height=140)
        _esb = ttk.Scrollbar(excl_frame, orient='vertical',
                              command=self.excl_canvas.yview)
        self.excl_frame = tk.Frame(self.excl_canvas, bg='#f5f5f5')
        self.excl_frame.bind('<Configure>',
                             lambda e: self.excl_canvas.configure(
                                 scrollregion=self.excl_canvas.bbox('all')))
        self.excl_canvas.create_window((0, 0), window=self.excl_frame, anchor='nw')
        self.excl_canvas.configure(yscrollcommand=_esb.set)
        self.excl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _esb.pack(side=tk.RIGHT, fill=tk.Y)

        def _mw_excl(ev):
            self.excl_canvas.yview_scroll(int(-1 * (ev.delta / 120)), 'units')
        self.excl_canvas.bind('<Enter>',
                              lambda e: self.excl_canvas.bind_all('<MouseWheel>', _mw_excl))
        self.excl_canvas.bind('<Leave>',
                              lambda e: self._ctrl_canvas.bind_all('<MouseWheel>', _mw_ctrl))

        # ═══════════════ PANNELLO DESTRO OPZIONI ═════════════════════

        # ── Colori gruppi ────────────────────────────────────────────
        color_frame = tk.LabelFrame(rpanel, text="Colori gruppi",
                                    bg='#f5f5f5', font=('Helvetica', 10, 'bold'),
                                    fg='#555', padx=6, pady=6)
        color_frame.pack(fill=tk.X, pady=4)
        self.group_color_vars = {}
        self.color_labels = {}
        self.color_inner = tk.Frame(color_frame, bg='#f5f5f5')
        self.color_inner.pack(fill=tk.X)

        # ── Colori PC (grafico C) ────────────────────────────────────
        pc_col_frame = tk.LabelFrame(rpanel, text="Colori PC (grafico C)",
                                     bg='#f5f5f5', font=('Helvetica', 10, 'bold'),
                                     fg='#555', padx=6, pady=6)
        pc_col_frame.pack(fill=tk.X, pady=4)

        for which, label, attr in [('a', 'PC A', 'pc_color_a_lbl'),
                                    ('b', 'PC B', 'pc_color_b_lbl')]:
            row = tk.Frame(pc_col_frame, bg='#f5f5f5')
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, width=6, bg='#f5f5f5',
                     font=('Helvetica', 9)).pack(side=tk.LEFT)
            current = self.pc_color_a if which == 'a' else self.pc_color_b
            lbl = tk.Label(row, bg=current, width=4, relief=tk.RIDGE, cursor='hand2')
            lbl.pack(side=tk.LEFT, padx=4)
            setattr(self, attr, lbl)
            lbl.bind('<Button-1>', lambda e, w=which: self._pick_pc_color(w))

        # ── Selezione componenti ─────────────────────────────────────
        pc_frame = tk.LabelFrame(rpanel, text="Componenti da visualizzare",
                                 bg='#f5f5f5', font=('Helvetica', 10, 'bold'),
                                 fg='#555', padx=6, pady=6)
        pc_frame.pack(fill=tk.X, pady=4)

        def _isb(parent, var):
            return tk.Spinbox(parent, from_=1, to=20, textvariable=var,
                              width=4, font=('Helvetica', 9))

        tk.Label(pc_frame, text="Biplot  X:", bg='#f5f5f5',
                 font=('Helvetica', 9)).grid(row=0, column=0, sticky='w', pady=2)
        _isb(pc_frame, self.pc_biplot_x).grid(row=0, column=1, padx=2)
        tk.Label(pc_frame, text="Y:", bg='#f5f5f5',
                 font=('Helvetica', 9)).grid(row=0, column=2, padx=(6, 0))
        _isb(pc_frame, self.pc_biplot_y).grid(row=0, column=3, padx=2)

        tk.Label(pc_frame, text="Loadings A:", bg='#f5f5f5',
                 font=('Helvetica', 9)).grid(row=1, column=0, sticky='w', pady=2)
        _isb(pc_frame, self.pc_load_a).grid(row=1, column=1, padx=2)
        tk.Label(pc_frame, text="B:", bg='#f5f5f5',
                 font=('Helvetica', 9)).grid(row=1, column=2, padx=(6, 0))
        _isb(pc_frame, self.pc_load_b).grid(row=1, column=3, padx=2)

        # ── Opzioni grafici ──────────────────────────────────────────
        opt = tk.LabelFrame(rpanel, text="Opzioni grafici",
                            bg='#f5f5f5', font=('Helvetica', 10, 'bold'),
                            fg='#555', padx=6, pady=6)
        opt.pack(fill=tk.X, pady=4)

        def _row(parent, r, label, widget_fn):
            tk.Label(parent, text=label, bg='#f5f5f5',
                     font=('Helvetica', 9)).grid(row=r, column=0, sticky='w', pady=2)
            widget_fn(parent).grid(row=r, column=1, padx=6, sticky='w')

        def _spx(parent, var, lo, hi, inc):
            return tk.Spinbox(parent, from_=lo, to=hi, increment=inc,
                              textvariable=var, width=6, font=('Helvetica', 9))

        _row(opt, 0, "Soglia |loading|:",
             lambda p: _spx(p, self.threshold, 0.1, 0.9, 0.05))
        _row(opt, 1, "Sigma ellisse:",
             lambda p: _spx(p, self.ellipse_sigma, 1, 3, 1))
        _row(opt, 2, "Soglia varianza scree (%):",
             lambda p: _spx(p, self.scree_threshold, 50, 99, 5))
        _row(opt, 3, "Max PC nello scree:",
             lambda p: _spx(p, self.scree_max_pc, 2, 20, 1))
        _row(opt, 4, "Scala frecce biplot:",
             lambda p: _spx(p, self.biplot_scale, 0.5, 16.0, 0.5))
        _row(opt, 5, "Dim. marker:",
             lambda p: _spx(p, self.marker_size, 10, 400, 10))

        cb_frame = tk.Frame(opt, bg='#f5f5f5')
        cb_frame.grid(row=6, column=0, columnspan=2, sticky='w', pady=(6, 0))
        for var, label in [
            (self.show_sample_labels, "Etichette campioni"),
            (self.show_var_labels,    "Frecce + etichette variabili"),
            (self.show_ellipses,      "Ellissi gruppi"),
            (self.equal_aspect,       "Biplot scala 1:1 (assi proporzionali)"),
        ]:
            tk.Checkbutton(cb_frame, text=label, variable=var,
                           bg='#f5f5f5', font=('Helvetica', 9),
                           anchor='w').pack(fill=tk.X, pady=1)

    # ── Dati di esempio ──────────────────────────────────────────────────
    def _load_example(self):
        """
        Carica un dataset di esempio con 3 gruppi × 6 campioni e 8 variabili.
        Mostra la finestra dati così l'utente vede come devono essere organizzati
        i dati di input nel formato standard (campioni per righe).
        """
        rng = np.random.default_rng(42)

        gruppi   = ['Gruppo_A'] * 6 + ['Gruppo_B'] * 6 + ['Gruppo_C'] * 6
        campioni = ([f'A{i}' for i in range(1, 7)] +
                    [f'B{i}' for i in range(1, 7)] +
                    [f'C{i}' for i in range(1, 7)])

        # Tre cluster ben separati nello spazio delle 8 variabili
        mu = {
            'Gruppo_A': np.array([8.0,  2.0,  5.0,  9.0,  1.0,  3.0,  7.0,  4.0]),
            'Gruppo_B': np.array([2.0,  8.0,  3.0,  2.0,  9.0,  7.0,  1.0,  6.0]),
            'Gruppo_C': np.array([5.0,  5.0,  9.0,  4.0,  5.0,  1.0,  4.0,  9.0]),
        }
        variabili = ['Var_1', 'Var_2', 'Var_3', 'Var_4',
                     'Var_5', 'Var_6', 'Var_7', 'Var_8']

        rows = []
        for camp, grp in zip(campioni, gruppi):
            vals = mu[grp] + rng.normal(0, 0.6, size=8)
            rows.append([camp, grp] + list(np.round(vals, 3)))

        df = pd.DataFrame(rows, columns=['Campione', 'Gruppo'] + variabili)

        # Costruisce anche df_raw nel formato standard (prima riga = intestazioni)
        header = pd.DataFrame([df.columns.tolist()], columns=df.columns)
        self.df_raw = pd.concat([header, df], ignore_index=True)
        self.df_raw.columns = range(len(self.df_raw.columns))

        self.df = df
        self._loaded_path = None
        self.data_orientation.set('righe')
        self.plot_title = "Dati_esempio"
        self.file_label.config(
            text="[dati di esempio — 3 gruppi × 6 campioni × 8 variabili]",
            fg='#888855')

        self._populate_controls()
        self._set_status(
            "Dati di esempio caricati. "
            "Premi '📋 Visualizza / Modifica dati' per vedere il formato atteso, "
            "oppure '▶ Esegui PCA' per provare subito.")
        self._open_data_view()   # apre automaticamente lo spreadsheet

    # ── Caricamento file ─────────────────────────────────────────────────
    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("Tutti", "*.*")])
        if not path:
            return
        try:
            # Legge sempre senza header per conservare i dati grezzi
            self.df_raw = (pd.read_csv(path, header=None)
                           if path.endswith('.csv')
                           else pd.read_excel(path, header=None))
            self._loaded_path = path
            filename = path.split('/')[-1]
            self.file_label.config(text=filename, fg='#2166ac')
            self.plot_title = filename.rsplit('.', 1)[0]
            self._apply_orientation()   # costruisce self.df e aggiorna i controlli
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile caricare il file:\n{e}")

    def _apply_orientation(self):
        """Trasforma df_raw → self.df in base all'orientamento scelto."""
        if self.df_raw is None:
            return
        try:
            if self.data_orientation.get() == 'righe':
                self.df = self._normalize_rows(self.df_raw)
            else:
                self.df = self._transpose_data(self.df_raw)
            self._populate_controls()
            self._set_status(
                f"File caricato ({self.data_orientation.get()}): "
                f"{self.df.shape[0]} campioni, {self.df.shape[1]} colonne.")
        except Exception as e:
            messagebox.showerror("Errore orientamento",
                                 f"Impossibile leggere i dati con questo orientamento:\n{e}")

    def _normalize_rows(self, df_raw):
        """
        Formato standard: prima riga = intestazioni, righe successive = campioni.
        Riproduce il comportamento di pd.read_excel(path) con header=0.
        """
        df = df_raw.copy()
        # Prima riga diventa intestazione; gestisce duplicati come pandas
        new_cols = []
        seen = {}
        for v in df.iloc[0]:
            name = str(v) if pd.notna(v) else 'Unnamed'
            count = seen.get(name, 0)
            seen[name] = count + 1
            new_cols.append(name if count == 0 else f'{name}.{count}')
        df.columns = new_cols
        df = df.iloc[1:].reset_index(drop=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df

    def _transpose_data(self, df_raw):
        """
        Formato trasposto (campioni per colonne). Due strutture supportate:

        Struttura A — righe intestazione esplicite (es. file HPLC):
          - riga 0, col 0 : etichetta 'Samples' (o simile, ignorata)
          - riga 0, col 1+: nomi campioni (F1, F2, …)
          - riga 1, col 0 : etichetta 'Groups' (o simile, ignorata)
          - riga 1, col 1+: nomi gruppo (F, F, F, …, SD)
          - righe 2+, col 0: nomi variabili
          - righe 2+, col 1+: dati

        Struttura B — senza intestazioni esplicite:
          - riga 0, col 1+: nomi gruppo (NaN nelle colonne separatori)
          - riga 1, col 1+: nomi campioni (o tutti NaN → auto-genera)
          - righe 2+, col 0: nomi variabili
          - righe 2+, col 1+: dati

        Rilevamento automatico: se riga 1, col 0 contiene 'group'/'gruppo'
        si usa la struttura A, altrimenti la struttura B.
        """
        label1 = str(df_raw.iloc[1, 0]).lower() if pd.notna(df_raw.iloc[1, 0]) else ''
        use_structure_a = any(k in label1 for k in ['group', 'gruppo', 'groups'])

        data_cols = list(df_raw.columns[1:])

        if use_structure_a:
            # Struttura A: riga 0 = campioni, riga 1 = gruppi
            sample_names = [str(v) if pd.notna(v) else f"campione_{i+1}"
                            for i, v in enumerate(df_raw.iloc[0, 1:].values)]
            groups       = [str(v) if pd.notna(v) else ''
                            for v in df_raw.iloc[1, 1:].values]
            data_start   = 2
        else:
            # Struttura B: riga 0 = gruppi (con eventuali NaN separatori),
            #              riga 1 = campioni (o tutti NaN → auto-genera)
            row0 = df_raw.iloc[0]
            row1 = df_raw.iloc[1]
            data_cols = [c for c in df_raw.columns[1:] if pd.notna(row0[c])]
            if not data_cols:
                raise ValueError(
                    "Nessuna colonna dati trovata.\n"
                    "Nel formato 'colonne' la prima riga deve contenere "
                    "le etichette gruppo (es. F, FD, C, …).")
            groups       = [str(row0[c]) for c in data_cols]
            snames_raw   = [row1[c] for c in data_cols]
            if all(pd.isna(v) for v in snames_raw):
                counter = {}
                sample_names = []
                for g in groups:
                    counter[g] = counter.get(g, 0) + 1
                    sample_names.append(f"{g}_{counter[g]}")
            else:
                sample_names = [str(v) if pd.notna(v) else f"campione_{i+1}"
                                for i, v in enumerate(snames_raw)]
            data_start = 2

        # Righe variabili: dalla riga data_start, con nome non vuoto in col 0
        var_rows = df_raw.iloc[data_start:].copy()
        valid = (var_rows.iloc[:, 0].notna() &
                 (var_rows.iloc[:, 0].astype(str).str.strip() != ''))
        var_rows = var_rows[valid]
        if var_rows.empty:
            raise ValueError("Nessuna variabile trovata.")

        var_names   = var_rows.iloc[:, 0].astype(str).str.strip().tolist()
        data_matrix = var_rows[data_cols].apply(
            pd.to_numeric, errors='coerce').values   # shape (n_vars, n_samples)

        # Trasposta → (n_samples, n_vars)
        result = pd.DataFrame(data_matrix.T, columns=var_names)
        result.insert(0, 'Campione', sample_names)
        result.insert(1, 'Gruppo',   groups)
        return result

    def _populate_controls(self):
        cols     = list(self.df.columns)
        num_cols = list(self.df.select_dtypes(include=[np.number]).columns)

        self.sample_menu['values'] = cols
        self.group_menu['values']  = cols
        # Rilevamento automatico: primo match vince; keyword brevi ('id','name')
        # escluse perché sono sottostringhe di nomi variabili ('acid', ecc.)
        detected_sample = False
        detected_group  = False
        for c in cols:
            cl = c.lower()
            if not detected_sample and any(
                    k in cl for k in ['campion', 'sample', 'campione']):
                self.sample_col.set(c)
                detected_sample = True
            if not detected_group and any(
                    k in cl for k in ['tipo', 'type', 'group', 'gruppo']):
                self.group_col.set(c)
                detected_group = True
            if detected_sample and detected_group:
                break

        # Assegna una lettera a ogni variabile numerica (a-z poi A-Z)
        import string
        _alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase)
        self._var_letters = {
            col: (_alphabet[i] if i < len(_alphabet) else str(i))
            for i, col in enumerate(num_cols)
        }

        for w in self.var_frame.winfo_children():
            w.destroy()
        self.var_checks = {}
        for col in num_cols:
            var   = tk.BooleanVar(value=True)
            lbl   = self._var_letters.get(col, '')
            tk.Checkbutton(self.var_frame, text=f"{lbl} — {col}", variable=var,
                           bg='#f5f5f5', font=('Helvetica', 9),
                           anchor='w').pack(fill=tk.X, pady=1)
            self.var_checks[col] = var

        self._update_samples()
        self.sample_menu.bind('<<ComboboxSelected>>', self._update_samples)
        self.group_menu.bind('<<ComboboxSelected>>', self._update_colors)
        self._update_colors()

    def _update_samples(self, event=None):
        for w in self.excl_frame.winfo_children():
            w.destroy()
        self.sample_checks = {}
        sc = self.sample_col.get()
        if sc and sc in self.df.columns:
            for s in self.df[sc].astype(str).drop_duplicates():
                var = tk.BooleanVar(value=False)
                tk.Checkbutton(self.excl_frame, text=s, variable=var,
                               bg='#f5f5f5', font=('Helvetica', 9),
                               anchor='w').pack(fill=tk.X, pady=1)
                self.sample_checks[s] = var

    def _update_colors(self, event=None):
        for w in self.color_inner.winfo_children():
            w.destroy()
        self.group_color_vars = {}
        self.color_labels = {}
        gc = self.group_col.get()
        if gc and gc in self.df.columns:
            default_colors = ['#2166ac', '#d4a017', '#4d9b8f',
                              '#b35806', '#762a83', '#1b7837']
            for i, g in enumerate(self.df[gc].dropna().unique()):
                color = default_colors[i % len(default_colors)]
                row = tk.Frame(self.color_inner, bg='#f5f5f5')
                row.pack(fill=tk.X, pady=2)
                tk.Label(row, text=str(g), width=8,
                         bg='#f5f5f5', font=('Helvetica', 9)).pack(side=tk.LEFT)
                clbl = tk.Label(row, bg=color, width=4,
                                relief=tk.RIDGE, cursor='hand2')
                clbl.pack(side=tk.LEFT, padx=4)
                self.group_color_vars[str(g)] = color
                self.color_labels[str(g)] = clbl
                clbl.bind('<Button-1>',
                          lambda e, grp=str(g): self._pick_color(grp))

    def _pick_color(self, group):
        from tkinter import colorchooser
        c = colorchooser.askcolor(color=self.group_color_vars[group],
                                  title=f"Colore per {group}")[1]
        if c:
            self.group_color_vars[group] = c
            self.color_labels[group].config(bg=c)

    def _pick_pc_color(self, which):
        from tkinter import colorchooser
        current = self.pc_color_a if which == 'a' else self.pc_color_b
        c = colorchooser.askcolor(color=current,
                                  title=f"Colore PC {'A' if which == 'a' else 'B'}")[1]
        if c:
            if which == 'a':
                self.pc_color_a = c
                self.pc_color_a_lbl.config(bg=c)
            else:
                self.pc_color_b = c
                self.pc_color_b_lbl.config(bg=c)

    def _select_all(self):
        for v in self.var_checks.values():
            v.set(True)

    def _deselect_all(self):
        for v in self.var_checks.values():
            v.set(False)

    # ── Log ─────────────────────────────────────────────────────────────
    def _log(self, msg, level='INFO'):
        import datetime
        line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {level}: {msg}"
        self._log_lines.append(line)
        # Aggiorna la finestra log se è aperta
        if self._log_win and self._log_win.winfo_exists():
            self._log_win._txt.configure(state='normal')
            self._log_win._txt.insert('end', line + '\n')
            if level in ('ERROR', 'WARNING'):
                start = self._log_win._txt.index('end - 2 lines')
                end   = self._log_win._txt.index('end - 1 chars')
                self._log_win._txt.tag_add(level, start, end)
            self._log_win._txt.configure(state='disabled')
            self._log_win._txt.see('end')

    def _show_log(self):
        if self._log_win and self._log_win.winfo_exists():
            self._log_win.lift()
            return
        win = tk.Toplevel(self.root)
        win.title("Log / Messaggi")
        win.geometry("780x340")
        win.configure(bg='#1e1e1e')
        self._log_win = win

        tb = tk.Frame(win, bg='#1e1e1e')
        tb.pack(fill=tk.X, padx=6, pady=(6, 2))
        tk.Button(tb, text="🗑  Pulisci", font=('Helvetica', 9),
                  bg='#444', fg='white', relief=tk.FLAT,
                  command=lambda: (
                      self._log_lines.clear(),
                      txt.configure(state='normal'),
                      txt.delete('1.0', 'end'),
                      txt.configure(state='disabled'))
                  ).pack(side=tk.LEFT, padx=2)

        frame = tk.Frame(win, bg='#1e1e1e')
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        vsb = ttk.Scrollbar(frame, orient='vertical')
        hsb = ttk.Scrollbar(frame, orient='horizontal')
        txt = tk.Text(frame, bg='#1e1e1e', fg='#cccccc',
                      font=('Courier', 9), wrap='none', state='disabled',
                      yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                      selectbackground='#555')
        vsb.config(command=txt.yview)
        hsb.config(command=txt.xview)
        txt.tag_configure('ERROR',   foreground='#ff6666')
        txt.tag_configure('WARNING', foreground='#ffcc44')
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        txt.pack(fill=tk.BOTH, expand=True)
        win._txt = txt

        # Riempi con i messaggi già presenti
        txt.configure(state='normal')
        for line in self._log_lines:
            txt.insert('end', line + '\n')
            if '] ERROR:' in line:
                s = txt.index('end - 2 lines')
                txt.tag_add('ERROR', s, txt.index('end - 1 chars'))
            elif '] WARNING:' in line:
                s = txt.index('end - 2 lines')
                txt.tag_add('WARNING', s, txt.index('end - 1 chars'))
        txt.configure(state='disabled')
        txt.see('end')

    # ── PCA ─────────────────────────────────────────────────────────────
    def _run_pca(self):
        if self.df is None:
            messagebox.showwarning("Attenzione", "Carica prima un file.")
            return

        features = [f for f, v in self.var_checks.items() if v.get()]
        if len(features) < 2:
            messagebox.showwarning("Attenzione", "Seleziona almeno 2 variabili.")
            return

        try:
            self._run_pca_inner(features)
        except Exception:
            tb_str = traceback.format_exc()
            self._log(tb_str, level='ERROR')
            self._set_status("❌  Errore durante la PCA — vedi Log / Messaggi")
            messagebox.showerror(
                "Errore PCA",
                "Si è verificato un errore.\n"
                "Apri '📋 Log / Messaggi' per i dettagli.\n\n"
                + tb_str[-600:])   # ultimi 600 char nel popup
            self._show_log()

    def _run_pca_inner(self, features):
        """Corpo effettivo della PCA — separato per isolare il try/except."""
        # Indici PC (0-based)
        bx = self.pc_biplot_x.get() - 1
        by = self.pc_biplot_y.get() - 1
        la = self.pc_load_a.get() - 1
        lb = self.pc_load_b.get() - 1

        excluded = {s for s, v in self.sample_checks.items() if v.get()}
        sc = self.sample_col.get()
        gc = self.group_col.get()

        df = self.df.copy()
        if sc and sc in df.columns:
            df = df[~df[sc].astype(str).isin(excluded)]

        X = df[features].values.astype(float)

        # ── Gestione valori mancanti ──────────────────────────────────
        n_nan = int(np.isnan(X).sum())
        if n_nan > 0:
            # Imputa con la media di colonna
            col_means = np.nanmean(X, axis=0)
            nan_idx   = np.where(np.isnan(X))
            X[nan_idx] = np.take(col_means, nan_idx[1])
            self._log(
                f"Valori mancanti: {n_nan} celle imputate con la media di colonna.",
                level='WARNING')

        samples = (df[sc].astype(str).values if sc
                   else np.arange(len(df)).astype(str))
        types   = (df[gc].astype(str).values if gc
                   else np.array([''] * len(df)))

        self._log(f"Avvio PCA: {len(features)} variabili, {len(samples)} campioni, "
                  f"scalatura={self.scaling_method.get()}")

        # ── Pre-elaborazione ──────────────────────────────────────────
        method    = self.scaling_method.get()
        scaler_fn = SCALERS.get(method)
        X_proc    = scaler_fn().fit_transform(X) if scaler_fn else X.copy()

        n_comp_val = self.n_components.get()
        n_comp     = (None if n_comp_val == 0
                      else min(n_comp_val, min(X.shape)))
        pca      = PCA(n_components=n_comp)
        scores   = pca.fit_transform(X_proc)
        loadings = pca.components_
        var_exp  = pca.explained_variance_ratio_ * 100
        n_pc     = len(var_exp)

        self._log(f"PCA completata: {n_pc} componenti, "
                  f"varianza spiegata PC1={var_exp[0]:.1f}% "
                  f"PC2={var_exp[1]:.1f}% (cumul={var_exp[0]+var_exp[1]:.1f}%)")

        # Validazione indici PC
        bad = [(n, i) for n, i in [('Biplot X', bx), ('Biplot Y', by),
                                    ('Loadings A', la), ('Loadings B', lb)]
               if i >= n_pc]
        if bad:
            raise ValueError(
                "Indice PC fuori range:\n" +
                "\n".join(f"  {n}: PC{i+1} non disponibile (max PC{n_pc})"
                          for n, i in bad))

        colors        = self.group_color_vars
        markers_list  = ['o', 's', '^', 'D', 'v', 'h']
        groups        = list(dict.fromkeys(types))
        group_markers = {g: markers_list[i % len(markers_list)]
                         for i, g in enumerate(groups)}

        self._last_pca = dict(
            scores=scores, loadings=loadings, var_exp=var_exp,
            features=features, samples=samples, types=types,
            colors=colors, group_markers=group_markers,
            threshold=self.threshold.get(),
            bx=bx, by=by, la=la, lb=lb,
            sigma=self.ellipse_sigma.get(),
            n_pc=n_pc,
            groups=groups,
            lx=loadings[bx],
            ly=loadings[by],
            scale=self.biplot_scale.get(),
            msize=self.marker_size.get(),
            show_slbl=self.show_sample_labels.get(),
            show_vlbl=self.show_var_labels.get(),
            show_ell=self.show_ellipses.get(),
            scree_thr=self.scree_threshold.get(),
            scree_max=self.scree_max_pc.get(),
            plot_title=self.plot_title or "PCA",
            pc_color_a=self.pc_color_a,
            pc_color_b=self.pc_color_b,
            equal_aspect=self.equal_aspect.get(),
            var_letters={f: self._var_letters.get(f, f) for f in features},
        )

        self._draw_pca(self.fig)
        self.canvas.draw()

        above = [f for f, l in zip(features, loadings[la])
                 if abs(l) > self._last_pca['threshold']]
        msg = (f"Scalatura: {method}  |  "
               f"PC{bx+1}={var_exp[bx]:.1f}%  PC{by+1}={var_exp[by]:.1f}%  "
               f"(cumul.={var_exp[bx]+var_exp[by]:.1f}%)  |  "
               f"n variabili: {len(features)}  n campioni: {len(samples)}  |  "
               f"Dominanti PC{la+1} (|l|>{self._last_pca['threshold']}): "
               f"{', '.join(above) if above else 'nessuna'}")
        if n_nan > 0:
            msg += f"  |  ⚠ {n_nan} NaN imputati"
        self._set_status(msg)
        self._log("Grafico aggiornato.")

    def _on_canvas_resize(self, event):
        """Ridisegna la figura quando la finestra viene ridimensionata,
        così tight_layout ricalcola i margini per la nuova dimensione."""
        if hasattr(self, '_last_pca'):
            self._draw_pca(self.fig)
            self.canvas.draw_idle()

    # ── Disegno pannelli individuali ────────────────────────────────────
    def _draw_ax_A(self, ax, d):
        var_exp  = d['var_exp'];  n_pc = d['n_pc']
        bx, by, la, lb = d['bx'], d['by'], d['la'], d['lb']
        scree_thr = d['scree_thr'];  scree_max = d['scree_max']
        n_bars = min(scree_max, n_pc)
        pcs    = np.arange(1, n_bars + 1)
        cumvar = np.cumsum(var_exp)
        ax.bar(pcs, var_exp[:n_bars], color='#6baed6',
               edgecolor='white', width=0.6)
        ax.plot(pcs, cumvar[:n_bars], 'o-', color='#d4a017',
                lw=2, ms=7, label='Cumulative %')
        ax.axhline(scree_thr, color='gray', ls=':', lw=1.2,
                   label=f'{scree_thr}% threshold')
        for idx, col in [(bx, '#2166ac'), (by, '#d4a017'),
                         (la, '#4d9b8f'), (lb, '#b35806')]:
            if idx < n_bars:
                ax.axvline(idx + 1, color=col, ls='--', lw=1.0, alpha=0.6)
        ax.set_xlabel('Principal Component', fontsize=10)
        ax.set_ylabel('Explained Variance (%)', fontsize=10)
        ax.set_title('A  |  Scree Plot', fontsize=12,
                     fontweight='bold', loc='left')
        ax.legend(fontsize=8)
        ax.set_xticks(pcs)
        ax.tick_params(labelsize=9)

    def _draw_ax_B(self, ax, d):
        """B — Loadings bar chart (PC_a e PC_b)."""
        la, lb   = d['la'], d['lb']
        loadings = d['loadings'];  var_exp  = d['var_exp']
        features = d['features'];  threshold = d['threshold']
        n_feat   = len(features)
        x_pos    = np.arange(n_feat)
        w        = 0.35
        ax.bar(x_pos - w / 2, loadings[la], w,
               label=f'PC{la+1} ({var_exp[la]:.1f}%)',
               color=d['pc_color_a'], edgecolor='white', alpha=0.85)
        ax.bar(x_pos + w / 2, loadings[lb], w,
               label=f'PC{lb+1} ({var_exp[lb]:.1f}%)',
               color=d['pc_color_b'], edgecolor='white', alpha=0.75)
        ax.axhline(0, color='black', lw=0.8)
        ax.axhline( threshold, color='gray', ls=':', lw=1.2,
                   label=f'|loading| = {threshold:.2f}')
        ax.axhline(-threshold, color='gray', ls=':', lw=1.2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(features, fontsize=8, rotation=30, ha='right')
        ax.set_ylabel('Loading', fontsize=10)
        ax.set_title(f'B  |  Loadings  PC{la+1} & PC{lb+1}',
                     fontsize=12, fontweight='bold', loc='left')
        mid  = max(1, n_feat // 2)
        la_v = loadings[la];  lb_v = loadings[lb]
        def _peak(arr, positive):
            v = arr if positive else -arr
            return float(v.max()) if len(v) else 0.0
        corner_weights = {
            'upper left':  max(_peak(la_v[:mid], True),  _peak(lb_v[:mid], True)),
            'lower left':  max(_peak(la_v[:mid], False), _peak(lb_v[:mid], False)),
            'upper right': max(_peak(la_v[mid:], True),  _peak(lb_v[mid:], True)),
            'lower right': max(_peak(la_v[mid:], False), _peak(lb_v[mid:], False)),
        }
        ax.legend(fontsize=9, loc=min(corner_weights, key=corner_weights.get))
        ax.tick_params(labelsize=9)

    def _draw_ax_C(self, ax, d):
        """C — Biplot (scores + frecce loadings + ellissi sigma)."""
        bx, by = d['bx'], d['by']
        scores   = d['scores'];   loadings = d['loadings']
        var_exp  = d['var_exp'];  features = d['features']
        samples  = d['samples'];  types    = d['types']
        colors   = d['colors'];   groups   = d['groups']
        group_markers = d['group_markers']
        sigma     = d['sigma'];   scale   = d['scale']
        msize     = d['msize']
        show_slbl = d['show_slbl']
        show_vlbl = d['show_vlbl']
        show_ell  = d['show_ell']
        lx = d['lx'];  ly = d['ly']
        n_feat = len(features)

        all_x = list(scores[:, bx]) + [lx[i] * scale * 1.3
                                        for i in range(n_feat)]
        all_y = list(scores[:, by]) + [ly[i] * scale * 1.3
                                        for i in range(n_feat)]
        for g in groups:
            sc_g = scores[types == g]
            if len(sc_g) > 1:
                cx = sc_g[:, bx].mean();  cy = sc_g[:, by].mean()
                all_x += [cx - sc_g[:, bx].std() * sigma,
                           cx + sc_g[:, bx].std() * sigma]
                all_y += [cy - sc_g[:, by].std() * sigma,
                           cy + sc_g[:, by].std() * sigma]
        margin = 0.6
        xmin = min(all_x) - margin
        xmax = max(all_x) + margin
        ymin = min(all_y) - margin
        ymax = max(all_y) + margin
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if d.get('equal_aspect', False):
            # adjustable='box': ridimensiona il box assi mantenendo i data limits
            # invariati → le etichette restano sempre entro i limiti calcolati
            ax.set_aspect('equal', adjustable='box')

        var_letters = d.get('var_letters', {})
        if show_vlbl:
            for i in range(n_feat):
                ax.annotate('', xy=(lx[i] * scale, ly[i] * scale),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle='-|>',
                                           color='#555555', lw=1.6,
                                           mutation_scale=10))
            var_texts = []
            for i, feat in enumerate(features):
                label = var_letters.get(feat, feat)
                txt = ax.text(lx[i] * scale * 1.18, ly[i] * scale * 1.18,
                              label, fontsize=8, color='#333333',
                              ha='center', va='center', fontweight='bold')
                var_texts.append(txt)
            if HAS_ADJUSTTEXT and var_texts:
                adjust_text(var_texts, ax=ax,
                            expand_text=(1.2, 1.2), force_text=(0.3, 0.3),
                            only_move={'text': 'xy'},
                            arrowprops=dict(arrowstyle='-',
                                           color='#dddddd', lw=0.6))
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ix = 0.04 * (xmax - xmin)
                iy = 0.04 * (ymax - ymin)
                for txt in var_texts:
                    tx, ty = txt.get_position()
                    txt.set_position((
                        float(np.clip(tx, xmin + ix, xmax - ix)),
                        float(np.clip(ty, ymin + iy, ymax - iy))))

        if show_ell:
            for g in groups:
                sc_g = scores[types == g]
                col  = colors.get(g, '#888888')
                if len(sc_g) > 1:
                    cx = sc_g[:, bx].mean();  cy = sc_g[:, by].mean()
                    w  = sc_g[:, bx].std() * 2 * sigma
                    h  = sc_g[:, by].std() * 2 * sigma
                    ax.add_patch(Ellipse((cx, cy), w, h,
                                         alpha=0.07, color=col, zorder=1))
                    ax.add_patch(Ellipse((cx, cy), w, h,
                                         fill=False, edgecolor=col,
                                         lw=1.4, linestyle='--',
                                         alpha=0.55, zorder=2))

        for t, sc_pt in zip(types, scores):
            ax.scatter(sc_pt[bx], sc_pt[by],
                       c=colors.get(t, '#888888'),
                       marker=group_markers.get(t, 'o'),
                       s=msize, zorder=5, edgecolors='white', linewidths=0.8)

        if show_slbl:
            for s, t, sc_pt in zip(samples, types, scores):
                ax.text(sc_pt[bx] + 0.04, sc_pt[by] + 0.04, s,
                        fontsize=8, color=colors.get(t, '#888888'),
                        fontweight='bold', zorder=6)

        ax.axhline(0, color='lightgray', lw=0.8, ls='--')
        ax.axvline(0, color='lightgray', lw=0.8, ls='--')
        ax.set_xlabel(f'PC{bx+1} ({var_exp[bx]:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC{by+1} ({var_exp[by]:.1f}%)', fontsize=10)
        ax.set_title(f'C  |  Biplot  PC{bx+1} vs PC{by+1}',
                     fontsize=12, fontweight='bold', loc='left')
        ax.tick_params(labelsize=9)
        handles = [
            plt.Line2D([0], [0],
                       marker=group_markers.get(g, 'o'),
                       color='w',
                       markerfacecolor=colors.get(g, '#888'),
                       markeredgecolor=colors.get(g, '#888'),
                       markersize=7, label=g)
            for g in groups if g
        ]
        if handles:
            ax.legend(handles=handles, fontsize=9, loc='lower right')

    # ── Disegno PCA (indipendente dalla figura) ──────────────────────────
    def _draw_pca(self, fig):
        d = self._last_pca

        fig.clear()

        # Layout: A (scree) e B (loadings) affiancati nella riga superiore;
        # C (biplot) occupa tutta la riga inferiore.
        # Ordine logico: scree → loadings → biplot.
        gs = fig.add_gridspec(2, 2,
                              width_ratios=[1, 2.5],
                              height_ratios=[1, 1.5])

        ax_A = fig.add_subplot(gs[0, 0])   # scree    — riga superiore sinistra
        ax_B = fig.add_subplot(gs[0, 1])   # loadings — riga superiore destra
        ax_C = fig.add_subplot(gs[1, :])   # biplot   — riga inferiore intera

        self._draw_ax_A(ax_A, d)
        self._draw_ax_B(ax_B, d)
        self._draw_ax_C(ax_C, d)

        try:
            fig.tight_layout(pad=1.2)
        except Exception:
            pass

    # ── Salva figura ─────────────────────────────────────────────────────
    def _save_figure(self):
        if not hasattr(self, '_last_pca'):
            messagebox.showwarning("Attenzione", "Esegui prima la PCA.")
            return
        stem = self._last_pca['plot_title'].replace(' ', '_') or "PCA_result"
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("PNG", "*.png"),
                       ("SVG", "*.svg"), ("Tutti", "*.*")],
            initialfile=stem)
        if not path:
            return
        base = path.rsplit('.', 1)[0]
        fig_out = plt.figure(figsize=(16, 10))
        self._draw_pca(fig_out)
        for ext in ('.pdf', '.png', '.svg'):
            fig_out.savefig(base + ext, bbox_inches='tight', dpi=300)
        plt.close(fig_out)
        self._set_status(f"Salvato: {base}.pdf / .png / .svg")

    # ── Salva pannelli separati ───────────────────────────────────────────
    def _save_panels(self):
        if not hasattr(self, '_last_pca'):
            messagebox.showwarning("Attenzione", "Esegui prima la PCA.")
            return
        stem = self._last_pca['plot_title'].replace(' ', '_') or "PCA_result"
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("Tutti", "*.*")],
            initialfile=stem,
            title="Scegli cartella e nome base — verranno creati _A/B/C .pdf .png .svg")
        if not path:
            return
        base = path.rsplit('.', 1)[0]
        d = self._last_pca
        n_feat = len(d['features'])

        panel_specs = [
            ('A', self._draw_ax_A, (6, 5)),
            ('B', self._draw_ax_B, (max(6, n_feat * 0.55 + 1.5), 5)),
            ('C', self._draw_ax_C, (8, 7)),
        ]
        saved = []
        for label, draw_fn, figsize in panel_specs:
            fig_p = plt.figure(figsize=figsize)
            ax = fig_p.add_subplot(1, 1, 1)
            draw_fn(ax, d)
            fig_p.tight_layout()
            for ext in ('.pdf', '.png', '.svg'):
                fig_p.savefig(f"{base}_{label}{ext}",
                              bbox_inches='tight', dpi=300)
            plt.close(fig_p)
            saved.append(f"{base}_{label}")
        self._set_status(
            f"Pannelli salvati (PDF+PNG+SVG): {', '.join(saved)}")

    # ── Esporta risultati PCA ─────────────────────────────────────────────
    def _export_data(self):
        if not hasattr(self, '_last_pca'):
            messagebox.showwarning("Attenzione", "Esegui prima la PCA.")
            return
        d    = self._last_pca
        stem = (self._last_pca['plot_title']).replace(' ', '_') or "PCA_data"
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("Tutti", "*.*")],
            initialfile=stem)
        if not path:
            return
        try:
            pc_cols = [f'PC{i+1}' for i in range(d['n_pc'])]

            df_scores = pd.DataFrame(d['scores'], columns=pc_cols)
            df_scores.insert(0, 'Sample', d['samples'])
            if any(d['types']):
                df_scores.insert(1, 'Group', d['types'])

            df_loadings = pd.DataFrame(d['loadings'].T,
                                       index=d['features'], columns=pc_cols)
            df_loadings.index.name = 'Variable'

            df_var = pd.DataFrame({
                'PC': pc_cols,
                'Explained_variance_%': d['var_exp'],
                'Cumulative_%': np.cumsum(d['var_exp']),
            })

            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                df_scores.to_excel(writer, sheet_name='Scores', index=False)
                df_loadings.to_excel(writer, sheet_name='Loadings')
                df_var.to_excel(writer, sheet_name='Varianza', index=False)

            self._set_status(f"Dati esportati: {path}")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile salvare:\n{e}")

    # ── Spreadsheet interno ───────────────────────────────────────────────
    def _open_data_view(self):
        if self.df is None:
            messagebox.showwarning("Attenzione", "Carica prima un file.")
            return

        cols = list(self.df.columns)
        win  = tk.Toplevel(self.root)
        win.title("Dati — Visualizza e modifica")
        win.geometry("1000x560")
        win.configure(bg='#f5f5f5')

        # Toolbar
        tb = tk.Frame(win, bg='#f5f5f5', pady=6, padx=8)
        tb.pack(fill=tk.X)

        def apply_edits():
            rows = [list(tree.item(i, 'values')) for i in tree.get_children()]
            try:
                new_df = pd.DataFrame(rows, columns=cols)
                for col in cols:
                    try:
                        new_df[col] = pd.to_numeric(new_df[col])
                    except (ValueError, TypeError):
                        pass
                self.df = new_df
                self._populate_controls()
                self._set_status(f"Dati aggiornati: {len(new_df)} righe.")
            except Exception as e:
                messagebox.showerror("Errore", str(e), parent=win)

        def save_to_file():
            rows   = [list(tree.item(i, 'values')) for i in tree.get_children()]
            df_out = pd.DataFrame(rows, columns=cols)
            for col in cols:
                try:
                    df_out[col] = pd.to_numeric(df_out[col])
                except (ValueError, TypeError):
                    pass
            path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"),
                           ("Tutti", "*.*")],
                parent=win)
            if not path:
                return
            try:
                if path.endswith('.csv'):
                    df_out.to_csv(path, index=False)
                else:
                    df_out.to_excel(path, index=False, engine='openpyxl')
                self._set_status(f"Dati salvati: {path}")
            except Exception as e:
                messagebox.showerror("Errore", str(e), parent=win)

        bkw = dict(font=('Helvetica', 10), relief=tk.FLAT,
                   padx=8, pady=4, cursor='hand2')
        tk.Button(tb, text="✅  Applica al programma", command=apply_edits,
                  bg='#2166ac', fg='white', **bkw).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="💾  Salva su file", command=save_to_file,
                  bg='#4d9b8f', fg='white', **bkw).pack(side=tk.LEFT, padx=4)
        ttk.Separator(tb, orient='vertical').pack(
            side=tk.LEFT, fill=tk.Y, padx=10, pady=4)
        tk.Button(tb, text="＋ Riga",
                  command=lambda: tree.insert('', 'end',
                                              values=[''] * len(cols)),
                  bg='#e0e0e0', **bkw).pack(side=tk.LEFT, padx=2)
        tk.Button(tb, text="－ Riga",
                  command=lambda: [tree.delete(i) for i in tree.selection()],
                  bg='#e0e0e0', **bkw).pack(side=tk.LEFT, padx=2)

        # Treeview
        frame = tk.Frame(win, bg='white')
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        vsb  = ttk.Scrollbar(frame, orient='vertical')
        hsb  = ttk.Scrollbar(frame, orient='horizontal')
        tree = ttk.Treeview(frame, columns=cols, show='headings',
                            yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                            selectmode='extended')
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        for col in cols:
            tree.heading(col, text=col, anchor='w')
            sample      = self.df[col].head(200).fillna('').astype(str)
            max_len_val = sample.str.len().max()
            max_chars   = max(len(str(col)),
                              int(max_len_val) if pd.notna(max_len_val) else 0)
            tree.column(col, width=min(max(max_chars * 9 + 16, 70), 280),
                        minwidth=50)

        tree.tag_configure('odd',  background='#f0f4f8')
        tree.tag_configure('even', background='white')
        for i, (_, row) in enumerate(self.df.iterrows()):
            tree.insert('', 'end',
                        values=[str(v) if pd.notna(v) else '' for v in row],
                        tags=('odd' if i % 2 else 'even',))

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True)

        # Editing inline
        entry_widget = [None]
        edit_info    = [None]
        entry_var    = tk.StringVar()

        def commit(e=None):
            if not entry_widget[0]:
                return
            try:
                if not entry_widget[0].winfo_exists():
                    entry_widget[0] = None
                    return
            except tk.TclError:
                entry_widget[0] = None
                return
            item, col_idx = edit_info[0]
            values = list(tree.item(item, 'values'))
            values[col_idx] = entry_var.get()
            tree.item(item, values=values)
            entry_widget[0].destroy()
            entry_widget[0] = None

        def cancel(e=None):
            if entry_widget[0]:
                try:
                    entry_widget[0].destroy()
                except tk.TclError:
                    pass
                entry_widget[0] = None

        def on_double_click(event):
            if tree.identify_region(event.x, event.y) != 'cell':
                return
            commit()
            item   = tree.identify_row(event.y)
            col_id = tree.identify_column(event.x)
            if not item:
                return
            bbox = tree.bbox(item, col_id)
            if not bbox:
                return
            x, y, width, height = bbox
            col_idx = int(col_id.lstrip('#')) - 1
            entry_var.set(str(tree.item(item, 'values')[col_idx]))
            entry = tk.Entry(tree, textvariable=entry_var,
                             font=('Helvetica', 9), relief=tk.FLAT, bd=1,
                             highlightthickness=1,
                             highlightcolor='#2166ac',
                             highlightbackground='#2166ac')
            entry.place(x=x, y=y, width=width, height=height)
            entry.focus_set()
            entry.select_range(0, 'end')
            entry_widget[0] = entry
            edit_info[0]    = (item, col_idx)
            entry.bind('<Return>',   commit)
            entry.bind('<Escape>',   cancel)
            entry.bind('<FocusOut>', commit)

        tree.bind('<Double-1>', on_double_click)
        tree.bind('<Button-1>', lambda e: commit())

    # ── Esporta biplot-ready per Excel ───────────────────────────────────
    def _export_biplot_excel(self):
        if not hasattr(self, '_last_pca'):
            messagebox.showwarning("Attenzione", "Esegui prima la PCA.")
            return

        d     = self._last_pca
        bx    = d['bx'];  by = d['by']
        la    = d['la'];  lb = d['lb']
        scale = self.biplot_scale.get()
        sigma = d['sigma']
        lx    = d['loadings'][bx]
        ly    = d['loadings'][by]
        scores   = d['scores']
        features = d['features']
        samples  = d['samples']
        types    = d['types']
        var_exp  = d['var_exp']
        groups   = list(dict.fromkeys(types))
        pcbx = f'PC{bx+1}';  pcby = f'PC{by+1}'
        pcla = f'PC{la+1}';  pclb = f'PC{lb+1}'

        stem = (self._last_pca['plot_title']).replace(' ', '_') or 'PCA'
        path = filedialog.asksaveasfilename(
            defaultextension='.xlsx',
            filetypes=[('Excel', '*.xlsx')],
            initialfile=f'{stem}_Excel_biplot')
        if not path:
            return

        try:
            with pd.ExcelWriter(path, engine='openpyxl') as writer:

                # ── 1. Scores per gruppo (wide) ──────────────────────
                max_n  = max((int(np.sum(types == g)) for g in groups), default=0)
                frames = []
                for g in groups:
                    mask = types == g
                    s_g  = samples[mask];  sc_g = scores[mask];  n = len(s_g)
                    pad  = max_n - n
                    frames.append(pd.DataFrame({
                        f'Campione [{g}]': list(s_g) + [''] * pad,
                        f'{pcbx} [{g}]':  list(sc_g[:, bx]) + [np.nan] * pad,
                        f'{pcby} [{g}]':  list(sc_g[:, by]) + [np.nan] * pad,
                    }))
                df_wide = pd.concat(frames, axis=1) if frames else pd.DataFrame()
                df_wide.to_excel(writer, sheet_name='Scores_per_gruppo', index=False)

                # ── 2. Frecce loadings ───────────────────────────────
                rows = []
                for i, feat in enumerate(features):
                    rows.append({'X': 0,             'Y': 0,             'Variabile': ''})
                    rows.append({'X': lx[i] * scale, 'Y': ly[i] * scale, 'Variabile': feat})
                    rows.append({'X': np.nan,         'Y': np.nan,        'Variabile': ''})
                pd.DataFrame(rows).to_excel(
                    writer, sheet_name='Frecce_loadings', index=False)

                # ── 3. Ellissi gruppi ────────────────────────────────
                t_pts    = np.linspace(0, 2 * np.pi, 361)
                ell_rows = []
                for g in groups:
                    mask = types == g;  sc_g = scores[mask]
                    if len(sc_g) > 1:
                        cx = sc_g[:, bx].mean();  cy = sc_g[:, by].mean()
                        rx = sc_g[:, bx].std() * sigma
                        ry = sc_g[:, by].std() * sigma
                        for t in t_pts:
                            ell_rows.append({'Gruppo': g,
                                             'X': cx + rx * np.cos(t),
                                             'Y': cy + ry * np.sin(t)})
                        ell_rows.append({'Gruppo': '', 'X': np.nan, 'Y': np.nan})
                if ell_rows:
                    pd.DataFrame(ell_rows).to_excel(
                        writer, sheet_name='Ellissi_gruppi', index=False)

                # ── 4. Loadings barchart ─────────────────────────────
                pd.DataFrame({
                    'Variabile': features,
                    pcla: d['loadings'][la],
                    pclb: d['loadings'][lb],
                    f'Dominante {pcla} (|l|>{d["threshold"]})': [
                        'SÌ' if abs(v) > d['threshold'] else ''
                        for v in d['loadings'][la]],
                }).to_excel(writer, sheet_name='Loadings_barchart', index=False)

                # ── 5. Varianza ──────────────────────────────────────
                pc_cols = [f'PC{i+1}' for i in range(d['n_pc'])]
                pd.DataFrame({
                    'PC': pc_cols,
                    'Varianza spiegata (%)': var_exp,
                    'Cumulata (%)': np.cumsum(var_exp),
                }).to_excel(writer, sheet_name='Varianza', index=False)

                # ── 6. Istruzioni ────────────────────────────────────
                instr = [
                    ('ISTRUZIONI — Come ricreare il biplot in Excel', ''),
                    ('', ''),
                    ('═══ PARTE 1 — Punti campioni ═══', ''),
                    ('1.', 'Vai al foglio "Scores_per_gruppo"'),
                    ('2.', f'Seleziona le colonne "{pcbx} [Gruppo1]" e "{pcby} [Gruppo1]"'),
                    ('3.', 'Inserisci → Grafico → Dispersione XY'),
                    ('4.', 'Per ogni gruppo aggiuntivo: tasto destro sul grafico → Seleziona dati → Aggiungi serie'),
                    ('  →', f'Valori X = colonna {pcbx} [GruppoN]   |   Valori Y = colonna {pcby} [GruppoN]'),
                    ('5.', 'Colora ogni serie con il colore del gruppo (tasto destro → Formato serie)'),
                    ('', ''),
                    ('═══ PARTE 2 — Frecce variabili ═══', ''),
                    ('6.', 'Seleziona dati → Aggiungi serie dal foglio "Frecce_loadings"'),
                    ('  →', 'Valori X = colonna X   |   Valori Y = colonna Y'),
                    ('7.', 'Tasto destro sulla serie → Cambia tipo grafico → Linea'),
                    ('8.', 'Formato serie → Linea → Tipo freccia fine: scegli punta freccia'),
                    ('  →', 'Indicatori: Nessuno'),
                    ('9.', 'Aggiungi etichette dati alla serie → Formato etichetta'),
                    ('  →', '"Valore da celle" → seleziona colonna Variabile del foglio Frecce_loadings'),
                    ('  →', 'Deseleziona "Valore X" e "Valore Y"'),
                    ('  →', 'Elimina manualmente le etichette vuote (quelle sui punti di origine)'),
                    ('', ''),
                    ('═══ PARTE 3 — Ellissi gruppi ═══', ''),
                    ('10.', 'Foglio "Ellissi_gruppi": aggiungi una serie per gruppo'),
                    ('   →', 'Filtra per Gruppo, usa le colonne X e Y → tipo Linea, nessun indicatore'),
                    ('   →', 'La riga con valori vuoti separa i gruppi (Excel la tratta come interruzione)'),
                    ('   →', 'Opzione rapida: aggiungi tutto X e Y come unica serie linea — le righe NaN'),
                    ('      ', 'interrompono automaticamente il tratto tra un gruppo e l\'altro'),
                    ('', ''),
                    ('═══ PARTE 4 — Grafico loadings ═══', ''),
                    ('11.', 'Nuovo grafico dal foglio "Loadings_barchart"'),
                    ('   →', f'Seleziona Variabile + {pcla} + {pclb} → Istogramma raggruppato'),
                    ('   →', 'Per la soglia: aggiungi serie costante = ±soglia come linea sovrapposta'),
                    ('', ''),
                    ('═══ PARAMETRI USATI ═══', ''),
                    ('Assi biplot:',          f'{pcbx} (X)   vs   {pcby} (Y)'),
                    ('Scala frecce:',         str(scale)),
                    ('Sigma ellisse:',        str(sigma)),
                    (f'Varianza {pcbx}:',     f'{var_exp[bx]:.1f}%'),
                    (f'Varianza {pcby}:',     f'{var_exp[by]:.1f}%'),
                    (f'Varianza {pcbx}+{pcby}:', f'{var_exp[bx]+var_exp[by]:.1f}%'),
                    ('Soglia |loading|:',     str(d['threshold'])),
                    ('Loadings barchart:',    f'{pcla}   e   {pclb}'),
                ]
                df_instr = pd.DataFrame(instr, columns=['Passo', 'Dettaglio'])
                df_instr.to_excel(writer, sheet_name='Istruzioni', index=False)

                ws = writer.sheets['Istruzioni']
                ws.column_dimensions['A'].width = 26
                ws.column_dimensions['B'].width = 72

            self._set_status(f"Esportato biplot Excel: {path}")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile salvare:\n{e}")

    def _set_status(self, msg):
        self.status.config(text=msg)


if __name__ == '__main__':
    root = tk.Tk()
    app = PCAApp(root)
    root.mainloop()
