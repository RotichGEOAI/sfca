# -*- coding: utf-8 -*-
"""Results window with the 2SFCA statistics tables (QGIS 4 / Qt6)."""
import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout)

from . import stats_core


class StatsDialog(QDialog):
    def __init__(self, tables, title="GP accessibility statistics", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 560)
        self.tables = tables
        self.tabs = QTabWidget()
        self.grids = []
        for name, head, rows in tables:
            grid = QTableWidget(len(rows), len(head))
            grid.setHorizontalHeaderLabels([str(h) for h in head])
            grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            grid.setAlternatingRowColors(True)
            for r, row in enumerate(rows):
                for c, v in enumerate(row):
                    it = QTableWidgetItem("" if v is None else str(v))
                    if c > 0:
                        it.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                            Qt.AlignmentFlag.AlignVCenter)
                    grid.setItem(r, c, it)
            grid.resizeColumnsToContents()
            grid.horizontalHeader().setStretchLastSection(True)
            self.tabs.addTab(grid, name)
            self.grids.append(grid)

        b_copy = QPushButton("Copy this table")
        b_csv = QPushButton("Export all as CSV…")
        b_txt = QPushButton("Save text report…")
        b_close = QPushButton("Close")
        b_copy.clicked.connect(self.copy_current)
        b_csv.clicked.connect(self.export_csv)
        b_txt.clicked.connect(self.save_txt)
        b_close.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addStretch(1)
        for b in (b_copy, b_csv, b_txt, b_close):
            row.addWidget(b)
        lay = QVBoxLayout(self)
        lay.addWidget(self.tabs)
        lay.addLayout(row)

    def copy_current(self):
        i = self.tabs.currentIndex()
        _n, head, rows = self.tables[i]
        text = "\n".join("\t".join(str(c) for c in r) for r in [head] + rows)
        QApplication.clipboard().setText(text)

    def export_csv(self):
        path, _f = QFileDialog.getSaveFileName(
            self, "Choose a file name prefix", "gp_accessibility", "CSV (*.csv)")
        if not path:
            return
        prefix = os.path.splitext(path)[0]
        try:
            files = stats_core.write_csvs(self.tables, prefix)
        except OSError as e:
            QMessageBox.warning(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Exported", "%d CSV files written, e.g.\n%s"
                                % (len(files), files[0]))

    def save_txt(self):
        path, _f = QFileDialog.getSaveFileName(self, "Save report",
                                               "gp_accessibility_report.txt", "Text (*.txt)")
        if path:
            try:
                stats_core.write_text(self.tables, path)
            except OSError as e:
                QMessageBox.warning(self, "Save failed", str(e))
