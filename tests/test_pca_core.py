"""Test di PCA GUI: caricamento/orientamento dei dati e risultati della PCA.

L'app è una GUI tkinter, quindi i test creano una finestra nascosta (se tkinter o il display non
sono disponibili vengono saltati) e leggono i risultati da `_last_pca`, senza guardare i grafici.

    python -m unittest discover -s tests -v
"""
import importlib.util
import os
import shutil
import tempfile
import tkinter as tk
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
PYW = os.path.join(HERE, "..", "pca_gui.pyw")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _r = tk.Tk()
    _r.destroy()
    TK_OK = True
except tk.TclError:
    TK_OK = False


def make_app(mod):
    root = tk.Tk()
    root.withdraw()
    with mock.patch.object(mod.messagebox, "showwarning"):
        app = mod.PCAApp(root)
    return root, app


def load_example(mod, app):
    """Carica i dati di esempio senza aprire la finestra del foglio dati."""
    with mock.patch.object(app, "_open_data_view"):
        app._load_example()
    return [f for f, v in app.var_checks.items() if v.get()]


@unittest.skipUnless(TK_OK, "tkinter/display non disponibile")
class PcaResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module(PYW, "pca_gui_under_test")

    def setUp(self):
        self.root, self.app = make_app(self.mod)
        self.addCleanup(self.root.destroy)
        self.features = load_example(self.mod, self.app)

    def run_pca(self):
        with mock.patch.object(self.mod.messagebox, "showerror"):
            self.app._run_pca_inner(self.features)
        return self.app._last_pca

    def test_example_has_expected_shape(self):
        self.assertEqual(self.app.df.shape, (18, 10))
        self.assertEqual(len(self.features), 8)

    def test_variance_and_orthonormal_loadings(self):
        d = self.run_pca()
        self.assertAlmostEqual(d["var_exp"].sum(), 100.0, places=6)
        self.assertTrue(np.all(np.diff(d["var_exp"]) <= 1e-9))      # decrescente
        np.testing.assert_allclose(d["loadings"] @ d["loadings"].T,
                                   np.eye(d["n_pc"]), atol=1e-9)

    def test_matches_sklearn_on_standardized_data(self):
        d = self.run_pca()
        X = self.app.df[self.features].to_numpy(float)
        ref = PCA().fit(StandardScaler().fit_transform(X))
        np.testing.assert_allclose(d["var_exp"], ref.explained_variance_ratio_ * 100, atol=1e-9)
        # i segni delle componenti sono arbitrari: si confrontano i valori assoluti
        np.testing.assert_allclose(np.abs(d["loadings"]), np.abs(ref.components_), atol=1e-9)

    def test_three_separated_groups_are_separated_on_pc1_pc2(self):
        d = self.run_pca()
        self.assertGreater(d["var_exp"][:2].sum(), 70.0)
        pts = d["scores"][:, :2]
        groups = np.array(d["types"])
        cent = {g: pts[groups == g].mean(axis=0) for g in dict.fromkeys(groups)}
        spread = max(np.linalg.norm(pts[groups == g] - cent[g], axis=1).max() for g in cent)
        names = list(cent)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                self.assertGreater(np.linalg.norm(cent[names[i]] - cent[names[j]]), spread)

    def test_excluded_sample_is_left_out(self):
        self.app.sample_checks["A1"].set(True)
        d = self.run_pca()
        self.assertEqual(len(d["samples"]), 17)
        self.assertNotIn("A1", list(d["samples"]))

    def test_missing_values_are_imputed_and_logged(self):
        self.app.df.loc[0, "Var_3"] = np.nan
        d = self.run_pca()
        self.assertTrue(np.isfinite(d["scores"]).all())
        self.assertTrue(any("WARNING" in l and "imputate" in l for l in self.app._log_lines))

    def test_scaling_none_equals_plain_pca(self):
        self.app.scaling_method.set("nessuna")
        d = self.run_pca()
        X = self.app.df[self.features].to_numpy(float)
        np.testing.assert_allclose(d["var_exp"], PCA().fit(X).explained_variance_ratio_ * 100,
                                   atol=1e-9)

    def test_out_of_range_pc_index_is_an_error(self):
        self.app.n_components.set(2)
        self.app.pc_biplot_y.set(5)
        with self.assertRaises(ValueError):
            self.app._run_pca_inner(self.features)


@unittest.skipUnless(TK_OK, "tkinter/display non disponibile")
class DataLayouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module(PYW, "pca_gui_layouts")

    def setUp(self):
        self.root, self.app = make_app(self.mod)
        self.addCleanup(self.root.destroy)

    def standard_raw(self):
        return pd.DataFrame([["Campione", "Gruppo", "V1", "V2"],
                             ["a1", "A", 1.0, 2.0],
                             ["a2", "A", 3.0, 4.0],
                             ["b1", "B", 5.0, 6.0]])

    def transposed_raw(self):
        return pd.DataFrame([["Samples", "a1", "a2", "b1"],
                             ["Groups", "A", "A", "B"],
                             ["V1", 1.0, 3.0, 5.0],
                             ["V2", 2.0, 4.0, 6.0]])

    def test_standard_layout_without_deprecation_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")        # un FutureWarning qui diventa un errore
            df = self.app._normalize_rows(self.standard_raw())
        self.assertEqual(list(df.columns), ["Campione", "Gruppo", "V1", "V2"])
        self.assertTrue(pd.api.types.is_numeric_dtype(df["V1"]))
        self.assertEqual(df["Campione"].tolist(), ["a1", "a2", "b1"])

    def test_transposed_layout_matches_standard_layout(self):
        std = self.app._normalize_rows(self.standard_raw())
        tr = self.app._transpose_data(self.transposed_raw())
        self.assertEqual(tr["Campione"].tolist(), std["Campione"].tolist())
        self.assertEqual(tr["Gruppo"].tolist(), std["Gruppo"].tolist())
        np.testing.assert_allclose(tr[["V1", "V2"]].to_numpy(float),
                                   std[["V1", "V2"]].to_numpy(float))

    def test_transposed_layout_without_sample_names_generates_them(self):
        raw = pd.DataFrame([[np.nan, "F", "F", "SD"],
                            [np.nan, np.nan, np.nan, np.nan],
                            ["V1", 1.0, 2.0, 3.0]])
        tr = self.app._transpose_data(raw)
        self.assertEqual(tr["Campione"].tolist(), ["F_1", "F_2", "SD_1"])

    def test_duplicate_headers_are_made_unique(self):
        raw = pd.DataFrame([["x", "x", "y"], [1, 2, 3]])
        self.assertEqual(list(self.app._normalize_rows(raw).columns), ["x", "x.1", "y"])


@unittest.skipUnless(TK_OK, "tkinter/display non disponibile")
class WithoutPlotStyleKit(unittest.TestCase):
    """L'app deve funzionare anche senza il repo fratello PlotStyleKit."""

    def test_runs_when_the_sibling_repo_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            isolated = os.path.join(d, "PCA")
            os.makedirs(isolated)
            shutil.copy(PYW, os.path.join(isolated, "pca_gui.pyw"))
            mod = load_module(os.path.join(isolated, "pca_gui.pyw"), "pca_gui_isolated")
            self.assertIsNone(mod.origin_style)
            root, app = make_app(mod)
            try:
                features = load_example(mod, app)
                app._run_pca_inner(features)
                self.assertEqual(app._last_pca["n_pc"], 8)
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
