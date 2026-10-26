# -*- coding: utf-8 -*-
def classFactory(iface):
    from .plugin import SfcaPlugin
    return SfcaPlugin(iface)
