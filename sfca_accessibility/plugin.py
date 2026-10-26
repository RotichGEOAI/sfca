# -*- coding: utf-8 -*-
"""Plugin entry point (QGIS 4 / Qt6): two independent tools, 2SFCA and regression."""
import os

from qgis.PyQt.QtGui import QIcon

try:                                    # Qt6: QAction lives in QtGui
    from qgis.PyQt.QtGui import QAction
except ImportError:                     # pragma: no cover (Qt5 fallback)
    from qgis.PyQt.QtWidgets import QAction

MENU = "&GP Accessibility (2SFCA)"


class SfcaPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.actions = []
        self.dlg = None
        self.reg_dlg = None

    def _add(self, text, tip, slot):
        icon = QIcon(os.path.join(os.path.dirname(__file__), "icon.svg"))
        a = QAction(icon, text, self.iface.mainWindow())
        a.setToolTip(tip)
        a.triggered.connect(slot)
        self.iface.addPluginToMenu(MENU, a)
        self.iface.addToolBarIcon(a)
        self.actions.append(a)

    def initGui(self):
        self._add("GP Accessibility (2SFCA)…",
                  "Two-step floating catchment area accessibility to GPs", self.run)
        self._add("GP Accessibility Regression…",
                  "Regress GP accessibility on density, rurality and deprivation "
                  "(runs independently)", self.run_regression)

    def unload(self):
        for a in self.actions:
            self.iface.removePluginMenu(MENU, a)
            self.iface.removeToolBarIcon(a)
            a.deleteLater()
        self.actions = []
        for d in (self.dlg, self.reg_dlg):
            if d is not None:
                d.close()
        self.dlg = self.reg_dlg = None

    @staticmethod
    def _show(dlg):
        dlg.refresh_layers()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def run(self):
        from .dialog import SfcaDialog
        if self.dlg is None:
            self.dlg = SfcaDialog(self.iface, self.iface.mainWindow())
        self._show(self.dlg)

    def run_regression(self):
        from .regression_dialog import RegressionDialog
        if self.reg_dlg is None:
            self.reg_dlg = RegressionDialog(self.iface, self.iface.mainWindow())
        self.reg_dlg.refresh_layers()
        self.reg_dlg.show()
        self.reg_dlg.raise_()
        self.reg_dlg.activateWindow()
