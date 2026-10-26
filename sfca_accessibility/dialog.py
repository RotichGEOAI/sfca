# -*- coding: utf-8 -*-
"""Main dialog (QGIS 4 / Qt6 / PyQt6). Built in code, no .ui file, scoped enums only."""
import csv
import os
import traceback

from qgis.core import (Qgis, QgsCoordinateReferenceSystem, QgsCoordinateTransform,
                       QgsProject, QgsVectorLayer)
from qgis.gui import QgsFileWidget, QgsProjectionSelectionWidget
from qgis.PyQt.QtCore import Qt, QCoreApplication
from qgis.PyQt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
    QWidget)

from . import analysis, core
from .stats_dialog import StatsDialog

GP, ED, ROAD, POLY = "gp", "ed", "road", "poly"
FILTERS = {
    GP: "GP locations (*.csv *.txt *.shp *.geojson *.json *.gpkg);;All files (*)",
    ED: "Electoral divisions (*.shp *.geojson *.json *.gpkg);;All files (*)",
    ROAD: "Roads (*.shp *.geojson *.json *.gpkg);;All files (*)",
    POLY: "Polygon layers (*.shp *.geojson *.json *.gpkg);;All files (*)",
}
GEOM = {GP: Qgis.GeometryType.Point, ED: Qgis.GeometryType.Polygon,
        ROAD: Qgis.GeometryType.Line, POLY: Qgis.GeometryType.Polygon}


class CsvPointsDialog(QDialog):
    """Ask which columns hold X / Y and the CRS of a CSV file."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CSV point settings")
        self.delim, headers = analysis.csv_headers(path)
        gx, gy = analysis.guess_xy(headers)
        self.cx, self.cy = QComboBox(), QComboBox()
        for c in (self.cx, self.cy):
            c.addItems(headers)
        if gx:
            self.cx.setCurrentText(gx)
            self.cy.setCurrentText(gy)
        elif len(headers) > 1:
            self.cy.setCurrentIndex(1)
        self.crs = QgsProjectionSelectionWidget()
        self.crs.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
        form = QFormLayout()
        form.addRow("X / longitude field", self.cx)
        form.addRow("Y / latitude field", self.cy)
        form.addRow("Coordinates are in CRS", self.crs)
        form.addRow(QLabel("Tip: WGS84 lon/lat = EPSG:4326; Irish Transverse Mercator "
                           "eastings/northings = EPSG:2157."))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(bb)

    def values(self):
        return self.cx.currentText(), self.cy.currentText(), self.crs.crs().authid(), self.delim


class LayerPicker(QWidget):
    """Combo of project layers of one geometry type + a Browse… button for files."""

    def __init__(self, kind, owner, key=None):
        super().__init__()
        self.kind, self.owner = kind, owner
        self.key = key or kind          # lets one dialog hold several pickers of one kind
        self.combo = QComboBox()
        self.combo.setMinimumWidth(260)
        self.btn = QPushButton("Browse…")
        self.btn.clicked.connect(self.browse)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.combo, 1)
        lay.addWidget(self.btn)

    def refresh(self, select_id=None):
        keep = select_id or self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsVectorLayer) and lyr.isValid() \
                    and lyr.geometryType() == GEOM[self.kind]:
                self.combo.addItem(lyr.name(), lyr.id())
        i = self.combo.findData(keep)
        if i >= 0:
            self.combo.setCurrentIndex(i)
        self.combo.blockSignals(False)
        self.combo.currentIndexChanged.emit(self.combo.currentIndex())

    def layer(self):
        return QgsProject.instance().mapLayer(self.combo.currentData()) \
            if self.combo.currentData() else None

    def browse(self):
        path, _f = QFileDialog.getOpenFileName(self, "Select file", "", FILTERS[self.kind])
        if not path:
            return
        try:
            if path.lower().endswith((".csv", ".txt")):
                if self.kind != GP:
                    raise analysis.AnalysisError("CSV is only supported for GP locations.")
                dlg = CsvPointsDialog(path, self)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                xf, yf, crs, delim = dlg.values()
                lyr = analysis.load_points_csv(path, xf, yf, crs, delim)
            else:
                lyr = analysis.load_vector_file(path)
            if lyr.geometryType() != GEOM[self.kind]:
                raise analysis.AnalysisError(
                    "That file has the wrong geometry type for this input "
                    "(expected %s)." % {GP: "points", ED: "polygons", ROAD: "lines",
                                        POLY: "polygons"}[self.kind])
            QgsProject.instance().addMapLayer(lyr)
            self.owner.refresh_layers(select={self.key: lyr.id()})
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Could not load file", str(e))


def fill_fields(combo, layer, numeric_only=False, optional=False, prefer=()):
    combo.blockSignals(True)
    combo.clear()
    if optional:
        combo.addItem("— none —", None)
    if layer is not None:
        for f in layer.fields():
            if numeric_only and not f.isNumeric():
                continue
            combo.addItem(f.name(), f.name())
    for want in prefer:
        for i in range(combo.count()):
            d = combo.itemData(i)
            if d and want in d.lower():
                combo.setCurrentIndex(i)
                combo.blockSignals(False)
                return
    combo.blockSignals(False)


class SfcaDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("GP Accessibility — Two-Step Floating Catchment Area (2SFCA)")
        self.resize(760, 820)
        self._cancel = False
        self._running = False

        # ---------------- inputs ----------------
        self.gp = LayerPicker(GP, self)
        self.gp_name = QComboBox()
        self.gp_cap = QComboBox()
        self.gp_cap.setToolTip("Optional numeric field giving each provider's capacity, e.g. "
                               "GP full-time equivalents, number of GPs at the practice, or "
                               "panel size. Leave as 'none' to count every point as 1 GP.")
        self.gp_cap_missing = QComboBox()
        self.gp_cap_missing.addItem("Skip that GP", "skip")
        self.gp_cap_missing.addItem("Count it as 1", "one")
        self.gp_cap_missing.addItem("Count it as 0", "zero")
        self.cap_info = QLabel("")
        self.cap_info.setWordWrap(True)
        self.ed = LayerPicker(ED, self)
        self.pop = QComboBox()
        self.group = QComboBox()
        self.group.setToolTip("Optional: add a table of access statistics for each value of "
                              "this field (e.g. county, urban/rural class).")
        self.surface = QCheckBox("Use point-on-surface instead of centroid "
                                 "(guarantees the point lies inside the polygon)")
        self.road = LayerPicker(ROAD, self)
        self.speed_f = QComboBox()
        self.class_f = QComboBox()
        self.oneway_f = QComboBox()

        g1 = QGroupBox("1. Supply — GP locations (CSV, Shapefile, GeoJSON)")
        f1 = QFormLayout(g1)
        f1.addRow("GP layer / file", self.gp)
        f1.addRow("Name field (optional)", self.gp_name)
        f1.addRow("Provider capacity field (optional)", self.gp_cap)
        f1.addRow("If capacity is blank / non-numeric", self.gp_cap_missing)
        f1.addRow("", self.cap_info)

        g2 = QGroupBox("2. Demand — electoral divisions with population (Shapefile, GeoJSON)")
        f2 = QFormLayout(g2)
        f2.addRow("ED layer / file", self.ed)
        f2.addRow("Population field", self.pop)
        f2.addRow("Summarise statistics by (optional)", self.group)
        f2.addRow("", self.surface)

        g3 = QGroupBox("3. Road network (line Shapefile / GeoJSON)")
        f3 = QFormLayout(g3)
        f3.addRow("Roads layer / file", self.road)
        f3.addRow("Speed field, km/h (optional)", self.speed_f)
        f3.addRow("Road-class field (optional, e.g. OSM 'highway')", self.class_f)
        f3.addRow("One-way field (optional, driving only)", self.oneway_f)

        # ---------------- parameters ----------------
        self.mode = QComboBox()
        for k, v in core.MODE_PROFILES.items():
            self.mode.addItem(v["label"], k)
        self.thr = QDoubleSpinBox()
        self.thr.setRange(1, 180)
        self.thr.setValue(15)
        self.thr.setSuffix(" min")
        self.decay = QComboBox()
        self.decay.addItem("Binary (classic 2SFCA)", "binary")
        self.decay.addItem("Gaussian distance decay (enhanced 2SFCA)", "gaussian")
        self.def_kmh = QDoubleSpinBox()
        self.def_kmh.setRange(0, 130)
        self.def_kmh.setValue(0)
        self.def_kmh.setSpecialValueText("mode default")
        self.def_kmh.setSuffix(" km/h")
        self.overrides = QLineEdit()
        self.overrides.setPlaceholderText("e.g. motorway=100, primary=60, residential=30")
        self.access_kmh = QDoubleSpinBox()
        self.access_kmh.setRange(1, 60)
        self.access_kmh.setValue(10)
        self.access_kmh.setSuffix(" km/h")
        self.snap = QDoubleSpinBox()
        self.snap.setRange(50, 20000)
        self.snap.setValue(2000)
        self.snap.setSuffix(" m")
        self.crs = QgsProjectionSelectionWidget()
        self.ncls = QSpinBox()
        self.ncls.setRange(3, 9)
        self.ncls.setValue(5)
        self.cmeth = QComboBox()
        self.cmeth.addItem("Quantile", "quantile")
        self.cmeth.addItem("Equal interval", "equal")

        g4 = QGroupBox("4. Parameters")
        f4 = QFormLayout(g4)
        f4.addRow("Transport mode", self.mode)
        f4.addRow("Catchment threshold (travel time)", self.thr)
        f4.addRow("Catchment weighting", self.decay)
        f4.addRow("Default road speed", self.def_kmh)
        f4.addRow("Class speed overrides", self.overrides)
        f4.addRow("Off-network access speed (point → road)", self.access_kmh)
        f4.addRow("Max. snap distance to road", self.snap)
        f4.addRow("Analysis CRS (projected, metres)", self.crs)
        f4.addRow("Choropleth classes", self.ncls)
        f4.addRow("Classification", self.cmeth)

        # ---------------- outputs ----------------
        self.out = QgsFileWidget()
        self.out.setStorageMode(QgsFileWidget.StorageMode.SaveFile)
        self.out.setFilter("GeoPackage (*.gpkg)")
        self.out.setDialogTitle("Save results as GeoPackage")
        self.mk_layout = QCheckBox("Create a print layout (A4 landscape)")
        self.mk_layout.setChecked(True)
        self.pdf = QgsFileWidget()
        self.pdf.setStorageMode(QgsFileWidget.StorageMode.SaveFile)
        self.pdf.setFilter("PDF (*.pdf)")
        self.pdf.setDialogTitle("Export map as PDF")

        g5 = QGroupBox("5. Outputs")
        f5 = QFormLayout(g5)
        f5.addRow("Results GeoPackage (blank = temporary layers)", self.out)
        f5.addRow("", self.mk_layout)
        f5.addRow("Export choropleth map PDF (optional)", self.pdf)

        # ---------------- run area ----------------
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(140)
        self.run_btn = QPushButton("Run 2SFCA")
        self.stats_btn = QPushButton("Show statistics")
        self.stats_btn.setEnabled(False)
        self.stats_btn.clicked.connect(self.show_stats)
        self._tables = None
        self._stats_dlg = None
        self.cancel_btn = QPushButton("Cancel run")
        self.cancel_btn.setEnabled(False)
        self.close_btn = QPushButton("Close")
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn.clicked.connect(self._do_cancel)
        self.close_btn.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addStretch(1)
        for b in (self.run_btn, self.stats_btn, self.cancel_btn, self.close_btn):
            row.addWidget(b)

        inner = QWidget()
        iv = QVBoxLayout(inner)
        for g in (g1, g2, g3, g4, g5):
            iv.addWidget(g)
        iv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        main = QVBoxLayout(self)
        main.addWidget(scroll, 1)
        main.addWidget(self.bar)
        main.addWidget(self.log)
        main.addLayout(row)

        self.gp.combo.currentIndexChanged.connect(self._gp_changed)
        self.gp_cap.currentIndexChanged.connect(self._cap_changed)
        self.ed.combo.currentIndexChanged.connect(self._ed_changed)
        self.road.combo.currentIndexChanged.connect(self._road_changed)
        self.road.combo.currentIndexChanged.connect(self._suggest_crs)
        self.ed.combo.currentIndexChanged.connect(self._suggest_crs)
        self.refresh_layers()

    # ---------------- layer / field handling ----------------
    def refresh_layers(self, select=None):
        select = select or {}
        for kind, w in ((GP, self.gp), (ED, self.ed), (ROAD, self.road)):
            w.refresh(select.get(kind))

    def _gp_changed(self, *_):
        lyr = self.gp.layer()
        fill_fields(self.gp_name, lyr, optional=True, prefer=("name", "practice"))
        # every field is offered (numeric text such as "1.5" is parsed too); a likely
        # capacity column (fte, capacity, gps, doctors, panel...) is pre-selected
        fill_fields(self.gp_cap, lyr, optional=True,
                    prefer=("fte", "capacity", "n_gp", "num_gp", "gps", "doctor",
                            "practitioner", "panel", "headcount"))
        self._cap_changed()

    def _cap_changed(self, *_):
        """Show a quick summary of the chosen capacity field."""
        lyr, fld = self.gp.layer(), self.gp_cap.currentData()
        self.gp_cap_missing.setEnabled(bool(fld))
        if lyr is None or not fld:
            self.cap_info.setText("Every GP point will count as 1 provider.")
            return
        vals, bad = [], 0
        for k, f in enumerate(lyr.getFeatures()):
            if k >= 50000:
                break
            c, st = core.parse_capacity(f[fld], "skip")
            if st == "ok":
                vals.append(c)
            else:
                bad += 1
        if not vals:
            self.cap_info.setText("No numeric values found in '%s'. Pick another field." % fld)
            return
        self.cap_info.setText(
            "'%s': %d numeric values, total %.2f, min %.2f, max %.2f%s. Each GP supplies its "
            "own value (e.g. FTE) instead of 1." % (
                fld, len(vals), sum(vals), min(vals), max(vals),
                ", %d blank/invalid" % bad if bad else ""))

    def _ed_changed(self, *_):
        fill_fields(self.pop, self.ed.layer(), numeric_only=True,
                    prefer=("pop", "total", "persons", "t1_1agett"))
        fill_fields(self.group, self.ed.layer(), optional=True)
        self.group.setCurrentIndex(0)

    def _road_changed(self, *_):
        lyr = self.road.layer()
        fill_fields(self.speed_f, lyr, numeric_only=True, optional=True,
                    prefer=("maxspeed", "speed", "kmh"))
        fill_fields(self.class_f, lyr, optional=True, prefer=("highway", "fclass", "class"))
        fill_fields(self.oneway_f, lyr, optional=True, prefer=("oneway",))
        # optional selectors default to none unless a sensible name was found
        for c, keys in ((self.speed_f, ("maxspeed", "speed", "kmh")),
                        (self.class_f, ("highway", "fclass", "class")),
                        (self.oneway_f, ("oneway",))):
            d = c.currentData()
            if d is None or not any(k in d.lower() for k in keys):
                c.setCurrentIndex(0)

    def _suggest_crs(self, *_):
        """Default analysis CRS: a projected CRS from the inputs, else the UTM zone."""
        cur = self.crs.crs()
        if cur.isValid() and not cur.isGeographic():
            return
        for lyr in (self.ed.layer(), self.road.layer(), self.gp.layer()):
            if lyr is not None and lyr.crs().isValid() and not lyr.crs().isGeographic():
                self.crs.setCrs(lyr.crs())
                return
        lyr = self.ed.layer()
        if lyr is not None and lyr.crs().isValid():
            try:
                wgs = QgsCoordinateReferenceSystem("EPSG:4326")
                c = QgsCoordinateTransform(lyr.crs(), wgs, QgsProject.instance()) \
                    .transform(lyr.extent().center())
                zone = int((c.x() + 180) // 6) + 1
                self.crs.setCrs(QgsCoordinateReferenceSystem(
                    "EPSG:%d" % ((32600 if c.y() >= 0 else 32700) + zone)))
            except Exception:  # noqa: BLE001
                pass

    # ---------------- run ----------------
    def _say(self, msg):
        self.log.appendPlainText(msg)
        QCoreApplication.processEvents()

    def _progress(self, frac, msg):
        self.bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self.bar.setFormat(msg)
        QCoreApplication.processEvents()
        return not self._cancel

    def _do_cancel(self):
        self._cancel = True
        self._say("Cancelling…")

    def _params(self):
        gp, ed, rd = self.gp.layer(), self.ed.layer(), self.road.layer()
        if gp is None or ed is None or rd is None:
            raise analysis.AnalysisError("Select a GP layer, an electoral-division layer and a "
                                         "roads layer.")
        if not self.pop.currentData():
            raise analysis.AnalysisError("Select the population field.")
        out = self.out.filePath().strip()
        if out and not out.lower().endswith(".gpkg"):
            out += ".gpkg"
        return {
            "work_crs": self.crs.crs(), "mode": self.mode.currentData(),
            "threshold": self.thr.value(),
            "gp_layer": gp, "gp_cap_field": self.gp_cap.currentData(),
            "gp_cap_missing": self.gp_cap_missing.currentData(),
            "gp_name_field": self.gp_name.currentData(),
            "ed_layer": ed, "pop_field": self.pop.currentData(),
            "use_surface": self.surface.isChecked(),
            "group_field": self.group.currentData(),
            "roads_layer": rd, "speed_field": self.speed_f.currentData(),
            "class_field": self.class_f.currentData(),
            "oneway_field": self.oneway_f.currentData(),
            "default_kmh": self.def_kmh.value() or None,
            "speed_overrides": self.overrides.text(),
            "access_kmh": self.access_kmh.value(), "snap_max": self.snap.value(),
            "decay": self.decay.currentData(), "n_classes": self.ncls.value(),
            "class_method": self.cmeth.currentData(), "out_gpkg": out,
        }

    def show_stats(self):
        if not self._tables:
            return
        self._stats_dlg = StatsDialog(self._tables, "GP accessibility statistics (2SFCA)", self)
        self._stats_dlg.show()
        self._stats_dlg.raise_()

    def _set_running(self, on):
        self._running = on
        self.run_btn.setEnabled(not on)
        self.cancel_btn.setEnabled(on)
        self.close_btn.setEnabled(not on)

    def run(self):
        self.log.clear()
        self._cancel = False
        try:
            p = self._params()
        except analysis.AnalysisError as e:
            QMessageBox.warning(self, "Check inputs", str(e))
            return
        self._set_running(True)
        try:
            res = analysis.run_analysis(p, self._say, self._progress)
            proj = QgsProject.instance()
            proj.addMapLayer(res["ed_layer"])
            proj.addMapLayer(res["supply_layer"])
            self.iface.mapCanvas().setExtent(
                QgsCoordinateTransform(res["ed_layer"].crs(), proj.crs(), proj)
                .transformBoundingBox(res["ed_layer"].extent()))
            self.iface.mapCanvas().refresh()
            self._say(analysis.format_summary(res["summary"], res, p["threshold"]))
            self._tables = res["tables"]
            self.stats_btn.setEnabled(True)
            if res.get("stat_files"):
                self._say("Statistics saved next to the GeoPackage (%d files)."
                          % len(res["stat_files"]))
            self.show_stats()
            if self.mk_layout.isChecked() or self.pdf.filePath().strip():
                pdf = self.pdf.filePath().strip() or None
                if pdf and not pdf.lower().endswith(".pdf"):
                    pdf += ".pdf"
                title = "Access to GPs by electoral division (2SFCA)"
                sub = ("%s · %g-minute road catchment · %s weighting · GPs per 10,000 "
                       "population" % (res["mode_label"], p["threshold"], p["decay"]))
                try:
                    unit = ("GP capacity (%s) per 10,000 population" % p["gp_cap_field"]
                            if p["gp_cap_field"] else "GPs per 10,000 population")
                    analysis.make_layout(res["ed_layer"], title, sub, pdf, unit)
                    self._say("Print layout created (Project ▸ Layouts)."
                              + ("  PDF: %s" % pdf if pdf else ""))
                except Exception as e:  # noqa: BLE001
                    self._say("Layout step failed: %s (the choropleth layer is still styled "
                              "on the canvas)." % e)
            self.bar.setValue(1000)
            self.iface.messageBar().pushMessage(
                "2SFCA", "Accessibility computed for %d electoral divisions."
                % res["summary"]["n_demand_snapped"], Qgis.MessageLevel.Success, 8)
        except core.Cancelled:
            self._say("Run cancelled.")
        except analysis.AnalysisError as e:
            self._say("ERROR: %s" % e)
            QMessageBox.warning(self, "2SFCA", str(e))
        except Exception:  # noqa: BLE001
            self._say("UNEXPECTED ERROR:\n" + traceback.format_exc())
        finally:
            self._set_running(False)

    def closeEvent(self, ev):
        if self._running:
            ev.ignore()
            return
        super().closeEvent(ev)
