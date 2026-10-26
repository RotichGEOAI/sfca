# -*- coding: utf-8 -*-
"""Stand-alone dialog for the accessibility regression (QGIS 4 / Qt6). Runs independently
of the 2SFCA tool: any polygon layer holding an accessibility field will do."""
import csv
import math
import os
import traceback

import numpy as np

from qgis.core import Qgis, QgsProject, QgsVectorLayer
from qgis.PyQt.QtCore import QCoreApplication, Qt
from qgis.PyQt.QtGui import QFont
from qgis.PyQt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import regression_core as rc
from . import regression_qgis as rq
from . import spatial_models as sm

FILTER = "Vector files (*.shp *.geojson *.json *.gpkg);;All files (*)"
SAME = "— same layer as the accessibility layer —"


def poly_layers():
    return [l for l in QgsProject.instance().mapLayers().values()
            if isinstance(l, QgsVectorLayer) and l.isValid()
            and l.geometryType() == Qgis.GeometryType.Polygon]


def fill_fields(combo, layer, optional=False, prefer=()):
    combo.blockSignals(True)
    combo.clear()
    if optional:
        combo.addItem("— none —", None)
    if layer is not None:
        for f in layer.fields():
            combo.addItem("%s%s" % (f.name(), "" if f.isNumeric() else "  (text)"), f.name())
    for want in prefer:
        for i in range(combo.count()):
            d = combo.itemData(i)
            if d and want in d.lower():
                combo.setCurrentIndex(i)
                combo.blockSignals(False)
                return
    combo.blockSignals(False)


class PolyPicker(QWidget):
    """Combo of polygon layers in the project + Browse… for .shp/.geojson files."""

    def __init__(self, owner, allow_same=False):
        super().__init__()
        self.owner, self.allow_same = owner, allow_same
        self.combo = QComboBox()
        self.combo.setMinimumWidth(240)
        b = QPushButton("Browse…")
        b.clicked.connect(self.browse)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.combo, 1)
        lay.addWidget(b)

    def refresh(self, select_id=None):
        keep = select_id or self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        if self.allow_same:
            self.combo.addItem(SAME, None)
        for l in poly_layers():
            self.combo.addItem(l.name(), l.id())
        i = self.combo.findData(keep)
        if i >= 0:
            self.combo.setCurrentIndex(i)
        self.combo.blockSignals(False)
        self.combo.currentIndexChanged.emit(self.combo.currentIndex())

    def layer(self):
        d = self.combo.currentData()
        return QgsProject.instance().mapLayer(d) if d else None

    def browse(self):
        path, _f = QFileDialog.getOpenFileName(self, "Select polygon file", "", FILTER)
        if not path:
            return
        lyr = QgsVectorLayer(path, os.path.splitext(os.path.basename(path))[0], "ogr")
        if not lyr.isValid() or lyr.geometryType() != Qgis.GeometryType.Polygon:
            QMessageBox.warning(self, "Could not load", "That file is not a valid polygon layer.")
            return
        QgsProject.instance().addMapLayer(lyr)
        self.owner.refresh_layers(select={id(self): lyr.id()})


class PredictorBox(QGroupBox):
    def __init__(self, owner, title, hint, prefer, log_default):
        super().__init__(title)
        self.owner, self.label, self.prefer = owner, title, prefer
        self.setCheckable(True)
        self.setChecked(True)
        self.picker = PolyPicker(owner, allow_same=True)
        self.field = QComboBox()
        self.join = QComboBox()
        self.join.addItem("Spatial: ED point-on-surface falls inside the polygon", "spatial")
        self.join.addItem("Attribute: match an ID field on both layers", "id")
        self.base_id = QComboBox()
        self.pred_id = QComboBox()
        self.log = QCheckBox("Log-transform  ln(1 + x)  (numeric fields only)")
        self.log.setChecked(log_default)
        f = QFormLayout(self)
        f.addRow(QLabel(hint))
        f.addRow("Shapefile / layer", self.picker)
        f.addRow("Attribute field", self.field)
        f.addRow("Join method", self.join)
        f.addRow("ID field on accessibility layer", self.base_id)
        f.addRow("ID field on this layer", self.pred_id)
        f.addRow("", self.log)
        self.picker.combo.currentIndexChanged.connect(self.reload_fields)
        self.join.currentIndexChanged.connect(self._join_changed)
        self._join_changed()

    def target_layer(self):
        return self.picker.layer() or self.owner.base.layer()

    def reload_fields(self, *_):
        lyr = self.target_layer()
        fill_fields(self.field, lyr, prefer=self.prefer)
        fill_fields(self.pred_id, lyr, optional=True, prefer=("id", "code", "guid"))

    def _join_changed(self, *_):
        on = self.join.currentData() == "id"
        self.base_id.setEnabled(on)
        self.pred_id.setEnabled(on)

    def spec(self):
        if not self.isChecked():
            return None
        return {"label": self.label, "layer": self.picker.layer(),
                "field": self.field.currentData(), "join": self.join.currentData(),
                "pred_id_field": self.pred_id.currentData(), "log": self.log.isChecked()}


class RegressionDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("GP Accessibility — Regression (density, rurality, deprivation)")
        self.resize(900, 900)
        self._res = self._spec = self._data = None
        self._sp = None
        self._gwr_layer = None
        self._gwr_map = []
        self._cur = None
        self._summary = None
        self._lisa_inp = self._lisa = None
        self._lisa_tabs = None
        self._cancel = False

        self.base = PolyPicker(self)
        self.outcome = QComboBox()
        self.base_id = QComboBox()
        g0 = QGroupBox("1. Accessibility (outcome) — any polygon layer with an accessibility field")
        f0 = QFormLayout(g0)
        f0.addRow(QLabel("Use the 2SFCA output (field sf_per10k) or any other shapefile that "
                         "holds an accessibility measure per area."))
        f0.addRow("Accessibility layer", self.base)
        f0.addRow("Outcome field", self.outcome)
        f0.addRow("ID field (for attribute joins)", self.base_id)

        self.dens = PredictorBox(self, "Population density",
                                 "Numeric, e.g. persons per km2.", ("dens", "pop"), True)
        self.rural = PredictorBox(self, "Rurality",
                                  "Numeric (e.g. % rural, distance to town) or a text/code "
                                  "classification (dummy-coded automatically).",
                                  ("rural", "urban", "class", "type"), False)
        self.depr = PredictorBox(self, "Deprivation",
                                 "Numeric index, score, rank or decile (e.g. Pobal HP score).",
                                 ("depriv", "hp", "imd", "score", "decile"), False)
        for pb in (self.dens, self.rural, self.depr):
            pb.join.currentIndexChanged.connect(self._sync_base_id)

        g4 = QGroupBox("2. Model options")
        f4 = QFormLayout(g4)
        self.log_y = QCheckBox("Log-transform outcome  ln(1 + y)  (skewed access scores)")
        self.drop0 = QCheckBox("Exclude EDs with zero access (no GP in catchment)")
        self.robust = QCheckBox("Robust (HC3) standard errors")
        self.robust.setChecked(True)
        self.knn = QSpinBox()
        self.knn.setRange(3, 20)
        self.knn.setValue(8)
        self.perm = QSpinBox()
        self.perm.setRange(99, 9999)
        self.perm.setValue(499)
        self.add_res = QCheckBox("Add styled residual layer to the project")
        self.add_res.setChecked(True)
        for w in (self.log_y, self.drop0, self.robust, self.add_res):
            f4.addRow("", w)
        f4.addRow("Neighbours for Moran's I", self.knn)
        f4.addRow("Moran's I permutations", self.perm)

        g5 = QGroupBox("3. Spatial models (use when residuals are spatially clustered)")
        f5 = QFormLayout(g5)
        f5.addRow(QLabel("Lagrange-multiplier tests always run and say which model fits best. "
                         "Weights = the nearest-neighbour count set above."))
        self.m_lag = QCheckBox("Spatial lag model (outcome in one ED depends on neighbours' outcomes)")
        self.m_err = QCheckBox("Spatial error model (clustered unexplained variation)")
        self.m_gwr = QCheckBox("Geographically weighted regression, GWR (coefficients vary across space)")
        self.bw_mode = QComboBox()
        self.bw_mode.addItem("GWR bandwidth: select automatically (AICc)", "auto")
        self.bw_mode.addItem("GWR bandwidth: fixed number of nearest EDs", "fixed")
        self.bw_n = QSpinBox()
        self.bw_n.setRange(20, 5000)
        self.bw_n.setValue(100)
        self.bw_n.setEnabled(False)
        self.bw_mode.currentIndexChanged.connect(
            lambda *_: self.bw_n.setEnabled(self.bw_mode.currentData() == "fixed"))
        for w in (self.m_lag, self.m_err, self.m_gwr):
            f5.addRow("", w)
        f5.addRow("", self.bw_mode)
        f5.addRow("Fixed bandwidth (neighbours)", self.bw_n)

        inner = QWidget()
        iv = QVBoxLayout(inner)
        g6 = QGroupBox("4. Local Moran's I (LISA): hot spots, cold spots and spatial outliers")
        f6 = QFormLayout(g6)
        f6.addRow(QLabel("Maps where similar values cluster. Works on any numeric field of the "
                         "accessibility layer, or on the OLS residuals after a regression run."))
        self.lisa_var = QComboBox()
        self.lisa_perm = QSpinBox()
        self.lisa_perm.setRange(99, 9999)
        self.lisa_perm.setValue(999)
        self.lisa_alpha = QComboBox()
        for a in (0.05, 0.01, 0.001):
            self.lisa_alpha.addItem(str(a), a)
        self.lisa_fdr = QCheckBox("Correct for multiple testing (false discovery rate, recommended)")
        self.lisa_fdr.setChecked(True)
        self.lisa_btn = QPushButton("Run Local Moran's I")
        self.lisa_btn.clicked.connect(self.run_lisa)
        f6.addRow("Variable", self.lisa_var)
        f6.addRow("Permutations", self.lisa_perm)
        f6.addRow("Significance level", self.lisa_alpha)
        f6.addRow("", self.lisa_fdr)
        f6.addRow("", self.lisa_btn)
        for g in (g0, self.dens, self.rural, self.depr, g4, g5, g6):
            iv.addWidget(g)
        iv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Variable", "Coef.", "Std. beta", "Std. err",
                                              "t", "p", "CI low", "CI high", "VIF"])
        self.table.setMinimumHeight(150)
        self.view = QComboBox()
        self.view.addItem("Show: OLS summary (HC3 robust errors, interpretation)", "ols")
        self.view.addItem("Show: model fit and diagnostics", "fit")
        self.view.currentIndexChanged.connect(self._view_changed)
        self.gwr_var = QComboBox()
        self.gwr_var.setEnabled(False)
        self.gwr_btn = QPushButton("Map GWR coefficient")
        self.gwr_btn.setEnabled(False)
        self.gwr_btn.clicked.connect(self.map_gwr)
        vrow = QHBoxLayout()
        vrow.addWidget(self.view, 1)
        vrow.addWidget(self.gwr_var, 1)
        vrow.addWidget(self.gwr_btn)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.report.setFont(mono)
        self.report.setMinimumHeight(220)

        self.run_btn = QPushButton("Run regression")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._do_cancel)
        self.save_txt = QPushButton("Save report (.txt)")
        self.save_csv = QPushButton("Save table shown (.csv)")
        self.save_sum = QPushButton("Save regression summary (.csv)")
        self.save_sum.setEnabled(False)
        self.save_sum.clicked.connect(self.do_save_summary)
        self.close_btn = QPushButton("Close")
        self.save_txt.setEnabled(False)
        self.save_csv.setEnabled(False)
        self.run_btn.clicked.connect(self.run)
        self.save_txt.clicked.connect(self.do_save_txt)
        self.save_csv.clicked.connect(self.do_save_csv)
        self.close_btn.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addStretch(1)
        for b in (self.run_btn, self.cancel_btn, self.save_sum, self.save_txt, self.save_csv, self.close_btn):
            row.addWidget(b)

        main = QVBoxLayout(self)
        main.addWidget(scroll, 3)
        main.addLayout(vrow)
        main.addWidget(self.table, 1)
        main.addWidget(self.bar)
        main.addWidget(self.report, 2)
        main.addLayout(row)

        self.base.combo.currentIndexChanged.connect(self._base_changed)
        self.refresh_layers()

    # ---------------- layers / fields ----------------
    def _pickers(self):
        return [self.base, self.dens.picker, self.rural.picker, self.depr.picker]

    def refresh_layers(self, select=None):
        select = select or {}
        for p in self._pickers():
            p.refresh(select.get(id(p)))

    def _base_changed(self, *_):
        lyr = self.base.layer()
        fill_fields(self.outcome, lyr, prefer=("sf_per10k", "per10k", "access"))
        fill_fields(self.base_id, lyr, optional=True, prefer=("guid", "id", "code"))
        fill_fields(self.lisa_var, lyr, prefer=("sf_per10k", "per10k", "access"))
        self.lisa_var.addItem("OLS residuals (run the regression first)", "__resid__")
        for pb in (self.dens, self.rural, self.depr):
            pb.reload_fields()

    def _sync_base_id(self, *_):
        pass

    # ---------------- run ----------------
    def _build_spec(self):
        base = self.base.layer()
        preds = [s for s in (self.dens.spec(), self.rural.spec(), self.depr.spec()) if s]
        for s in preds:
            if not s["field"]:
                raise rq.RegressionError("Choose an attribute field for '%s'." % s["label"])
        return {"base": base, "outcome_field": self.outcome.currentData(),
                "base_id_field": self.base_id.currentData(), "log_outcome": self.log_y.isChecked(),
                "drop_zero": self.drop0.isChecked(), "predictors": preds}

    def run(self):
        self.report.clear()
        try:
            spec = self._build_spec()
            self.report.setPlainText("Assembling data…")
            data = rq.assemble(spec)
            res = rq.fit(data, robust=self.robust.isChecked(),
                         k_neighbours=self.knn.value(), permutations=self.perm.value())
            text = rq.make_report(spec, data, res)
        except (rq.RegressionError, ValueError) as e:
            self.report.setPlainText("ERROR: %s" % e)
            QMessageBox.warning(self, "Regression", str(e))
            return
        except Exception:  # noqa: BLE001
            self.report.setPlainText("UNEXPECTED ERROR:\n" + traceback.format_exc())
            return
        self._spec, self._data, self._res = spec, data, res
        self._sp = None
        self.report.setPlainText(text)
        self.view.blockSignals(True)
        while self.view.count() > 2:
            self.view.removeItem(2)
        self.view.setCurrentIndex(0)
        self.view.blockSignals(False)
        self._summary = rq.summary_for(spec, res)
        self._fill_generic(self._summary[0], self._summary[1], 4)
        self.save_txt.setEnabled(True)
        self.save_csv.setEnabled(True)
        self.save_sum.setEnabled(True)
        if self.add_res.isChecked():
            try:
                lyr = rq.residual_layer(spec["base"], data, res)
                QgsProject.instance().addMapLayer(lyr)
            except Exception as e:  # noqa: BLE001
                self.report.appendPlainText("\n(Residual layer failed: %s)" % e)

        # ---- spatial diagnostics and models (always LM tests; models if ticked) ----
        self._cancel = False
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        try:
            opts = {"k": self.knn.value(), "lag": self.m_lag.isChecked(),
                    "error": self.m_err.isChecked(), "gwr": self.m_gwr.isChecked(),
                    "bandwidth": self.bw_n.value() if self.bw_mode.currentData() == "fixed" else None,
                    "perms": self.perm.value()}
            sp = rq.fit_spatial(data, opts, self._progress)
            self._sp = sp
            self.report.appendPlainText(rq.spatial_report(res, sp))
            for key, label in (("lag", "Spatial lag"), ("error", "Spatial error"),
                               ("gwr", "GWR coefficient summary")):
                if key in sp:
                    self.view.addItem("Show: %s" % label, key)
            if "gwr" in sp:
                self._add_gwr_layer(spec, data, sp["gwr"])
        except sm.Cancelled:
            self.report.appendPlainText("\nSpatial models cancelled.")
        except (ValueError, np.linalg.LinAlgError) as e:
            self.report.appendPlainText("\nSpatial models failed: %s" % e)
        except Exception:  # noqa: BLE001
            self.report.appendPlainText("\nSPATIAL MODEL ERROR:\n" + traceback.format_exc())
        finally:
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.bar.setValue(1000)
        self.iface.messageBar().pushMessage(
            "Regression", "Model fitted on %d EDs (R² = %.2f)." % (res["n"], res["r2"]),
            Qgis.MessageLevel.Success, 8)

    # ---------------- spatial helpers ----------------
    def _progress(self, frac, msg):
        self.bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self.bar.setFormat(msg)
        QCoreApplication.processEvents()
        return not self._cancel

    def _do_cancel(self):
        self._cancel = True

    def _add_gwr_layer(self, spec, data, g):
        lyr, mapping = rq.gwr_layer(spec["base"], data, g)
        self._gwr_layer, self._gwr_map = lyr, mapping
        self.gwr_var.clear()
        for j, (label, bf, tf) in enumerate(mapping):
            if j > 0:                                   # skip the intercept
                self.gwr_var.addItem(label, (bf, tf, label))
        self.gwr_var.setEnabled(True)
        self.gwr_btn.setEnabled(True)
        QgsProject.instance().addMapLayer(lyr)
        self.map_gwr()

    def map_gwr(self):
        if self._gwr_layer is None or self.gwr_var.currentData() is None:
            return
        bf, tf, label = self.gwr_var.currentData()
        rq.style_gwr(self._gwr_layer, bf, tf, label)

    def _view_changed(self, *_):
        key = self.view.currentData()
        if key == "ols" and self._summary is not None:
            self._fill_generic(self._summary[0], self._summary[1], 4)
        elif key == "fit" and self._summary is not None:
            self._fill_generic(self._summary[2], self._summary[3], None)
        elif key == "lisa_sum" and self._lisa_tabs is not None:
            self._fill_generic(self._lisa_tabs[0], self._lisa_tabs[1], None)
        elif key == "lisa_det" and self._lisa_tabs is not None:
            self._fill_generic(self._lisa_tabs[2], self._lisa_tabs[3], 4)
        elif key in ("lag", "error", "gwr") and self._sp is not None:
            head, rows = rq.spatial_table_rows(key, self._sp)
            self._fill_generic(head, rows, 4 if key in ("lag", "error") else None)

    def _fill_table(self, res):
        head = ["Variable", "Coef.", "Std. beta", "Std. err", "t", "p", "CI low", "CI high", "VIF"]
        self._fill_generic(head, rq.table_rows(res), 5)

    def _fill_generic(self, head, rows, pcol=None):
        self._cur = (head, rows)
        self.table.clear()
        self.table.setColumnCount(len(head))
        self.table.setHorizontalHeaderLabels(head)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                if isinstance(v, float):
                    txt = "%.3g" % v if c == pcol else "%.4g" % v
                else:
                    txt = "" if v is None else str(v)
                it = QTableWidgetItem(txt)
                if c > 0:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if pcol is not None and c == pcol and isinstance(v, float) and v < 0.05:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()

    # ---------------- Local Moran's I ----------------
    def run_lisa(self):
        base = self.base.layer()
        var = self.lisa_var.currentData()
        try:
            if base is None or not var:
                raise rq.RegressionError("Choose the accessibility layer and a variable.")
            id_field = self.base_id.currentData()
            if var == "__resid__":
                if self._res is None or self._data is None:
                    raise rq.RegressionError("Run the regression first to analyse its residuals.")
                inp = rq.lisa_inputs_from_residuals(self._spec["base"], self._data, self._res, id_field)
                base = self._spec["base"]
            else:
                inp = rq.lisa_inputs_from_layer(base, var, id_field)
        except rq.RegressionError as e:
            QMessageBox.warning(self, "Local Moran's I", str(e))
            return
        self._cancel = False
        self.lisa_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        try:
            r = rq.run_lisa(inp, self.knn.value(), self.lisa_perm.value(),
                            self.lisa_alpha.currentData(), self.lisa_fdr.isChecked(), self._progress)
            self._lisa_inp, self._lisa = inp, r
            self._lisa_tabs = rq.lisa_tables(inp, r)
            self.report.appendPlainText(rq.lisa_report(inp, r))
            lyr = rq.lisa_layer(base, inp, r)
            QgsProject.instance().addMapLayer(lyr)
            for key, label in (("lisa_sum", "Local Moran's I: cluster summary"),
                               ("lisa_det", "Local Moran's I: results by ED")):
                if self.view.findData(key) < 0:
                    self.view.addItem("Show: %s" % label, key)
            self.view.setCurrentIndex(self.view.findData("lisa_sum"))
            self.save_csv.setEnabled(True)
            self.iface.messageBar().pushMessage(
                "Local Moran's I", "%d hot-spot, %d cold-spot and %d outlier EDs."
                % (r["counts"]["HH"], r["counts"]["LL"], r["counts"]["HL"] + r["counts"]["LH"]),
                Qgis.MessageLevel.Success, 8)
        except sm.Cancelled:
            self.report.appendPlainText("\nLocal Moran's I cancelled.")
        except ValueError as e:
            QMessageBox.warning(self, "Local Moran's I", str(e))
        except Exception:  # noqa: BLE001
            self.report.appendPlainText("\nLISA ERROR:\n" + traceback.format_exc())
        finally:
            self.lisa_btn.setEnabled(True)
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.bar.setValue(1000)

    def do_save_summary(self):
        if self._summary is None:
            return
        path, _f = QFileDialog.getSaveFileName(self, "Save regression summary",
                                               "gp_access_regression_summary.csv", "CSV (*.csv)")
        if path:
            try:
                rq.write_summary_csv(path, self._summary)
            except OSError as e:
                QMessageBox.warning(self, "Save failed", str(e))

    # ---------------- save ----------------
    def do_save_txt(self):
        path, _f = QFileDialog.getSaveFileName(self, "Save report", "gp_access_regression.txt",
                                               "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.report.toPlainText() + "\n")

    def do_save_csv(self):
        if self._cur is None:
            return
        path, _f = QFileDialog.getSaveFileName(self, "Save table shown",
                                               "gp_access_regression_table.csv", "CSV (*.csv)")
        if not path:
            return
        head, rows = self._cur
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            for row in rows:
                w.writerow(["" if v is None else v for v in row])
