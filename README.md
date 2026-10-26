# GP Accessibility (2SFCA) – QGIS 4 plugin

Two-step floating catchment area analysis of GP accessibility by electoral division (Ireland),
road-network catchments (default 15 min; driving, cycling, walking), provider capacity field,
statistics tables, a choropleth + print layout, and an independent regression tool
(HC3 summary table, standardised betas, VIF, Breusch-Pagan, spatial lag/error, GWR, Local Moran's I).

Requires **QGIS 4.0+ (Qt6 / PyQt6)**. Full documentation: [sfca_accessibility/README.md](sfca_accessibility/README.md).

## Install
Download `sfca_accessibility.zip` from the latest CI run (Actions → Artifacts) or a release, then
QGIS → Plugins → Manage and Install Plugins → Install from ZIP.
Or copy the `sfca_accessibility/` folder into your QGIS profile's `python/plugins/` directory.

## Develop
```
pip install numpy
python -m unittest discover -s sfca_accessibility/tests -v
```
The engines (`core.py`, `stats_core.py`, `regression_core.py`, `spatial_models.py`) are pure
numpy and tested without QGIS. The dialogs and QGIS glue were written for the QGIS 4 API but have
not yet been run in a live QGIS 4 session – please open an issue with the log text if something fails.

Licence: GPL-2.0-or-later (see LICENSE). Before publishing, replace `YOUR-USER` and the author
email in `sfca_accessibility/metadata.txt`.
