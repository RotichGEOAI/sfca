# GP Accessibility (2SFCA) – QGIS 4.0+ plugin

Two-step floating catchment area (2SFCA) accessibility to GPs by electoral division,
using road-network travel time (default 15 minutes).

**Requires QGIS 4.0 or later (Qt6 / PyQt6).** `metadata.txt`: `qgisMinimumVersion=4.0`.

## Install
Plugins ▸ Manage and Install Plugins ▸ Install from ZIP ▸ `sfca_accessibility.zip`.
Open from the menu *Plugins ▸ GP Accessibility (2SFCA)* or the toolbar icon.

## Inputs
| Input | Formats | Notes |
|---|---|---|
| GP locations (supply) | CSV, Shapefile, GeoJSON (GPKG also works) | CSV: choose X/Y columns + CRS. Optional **provider capacity field** (GP FTE, GPs per practice, panel size); none = 1 per GP. Blank/non-numeric values can be skipped, counted as 1 or as 0 |
| Electoral divisions (demand) | Shapefile, GeoJSON | Polygons with a numeric population field; converted to centroids (or point-on-surface) |
| Roads | Shapefile, GeoJSON (line) | Must be **noded at junctions**. Optional speed (km/h), class (OSM `highway`/`fclass`) and one-way fields |

## Method
1. **Step 1** – for each GP *j*: `R_j = S_j / Σ_k (P_k · f(t_kj))` over EDs *k* with `t_kj ≤ 15 min`.
2. **Step 2** – for each ED *i*: `A_i = Σ_j (R_j · f(t_ij))` over GPs with `t_ij ≤ 15 min`.
`f` = 1 (binary, classic) or a Gaussian decay. Results are shown as GPs per 10,000 people.
Travel time = shortest path on the road graph (Dijkstra) + an off-network access leg
(point → nearest road node at the *access speed*, default 10 km/h, max snap 2 km).

## Modes
Driving (class speeds, one-ways), cycling (15 km/h cap, motorways excluded),
walking (5 km/h cap, motorways excluded). Public transport is not modelled (needs GTFS).

## Outputs
* `GP accessibility by ED (2SFCA)` – styled choropleth. Fields: `sf_pop, sf_nGP, sf_per10k,
  sf_mint, sf_idx, sf_edge, sf_snap, sf_class`
* `GP supply (R_j)` – GP points with ratio `sf_R10k`
* **Statistics window + tables** (see below)
* Optional GeoPackage + CSV, A4 print layout and PDF

## Accessibility statistics (tables, not just maps)
After each run a *GP accessibility statistics* window opens (re-open with **Show statistics**).
All figures are population-weighted, so they describe people rather than areas:
* **Summary** – benchmark, mean/median/SD, 10th/25th/75th/90th percentiles, P90/P10 ratio, Gini
  coefficient, % of people with no GP in catchment, % below half the benchmark, mean minutes to nearest GP
* **Map classes**, **Access vs benchmark** (none / very low / low / average / high / very high),
  **GPs reachable** (0, 1, 2-3 …), **Time to nearest GP** (with cumulative %)
* **By <field>** – optional breakdown by any ED field (county, urban/rural class …)
* **Providers** – GP-level ratios, catchment loads, GPs with no demand or heavily loaded
Export from the window (CSV per table, or one text report) or automatically next to the
results GeoPackage (`<name>_stats_*.csv`, `<name>_stats_report.txt`).

## Provider capacity
Pick any field in the GP layer as capacity (S_j). Each GP then supplies that amount in step 1
(`R_j = S_j / Σ P_k`) instead of 1, so a 6-GP practice outweighs a single-handed one. A likely
column (fte, capacity, gps, doctors, panel…) is pre-selected and the dialog shows its total, min
and max. Numeric text such as `1,5` is accepted; negatives are always skipped. Results are then
supply units per 10,000 people (GP FTE per 10,000 when the field is FTE).

## Regression tool (runs independently)
*Plugins ▸ GP Accessibility ▸ GP Accessibility Regression…* – OLS of an accessibility field on
population density, rurality and deprivation. Each predictor comes from its own shapefile (or the
accessibility layer itself) with a picked attribute field, joined spatially (ED point-on-surface
inside the polygon) or by ID. Text rurality classes are dummy-coded. Options: ln(1+x) transforms,
log outcome, drop zero-access EDs, HC3 robust errors. Output: coefficients, standardised betas,
VIF, R²/adj. R², F, AIC/BIC, Breusch-Pagan, Moran's I of residuals, plain-language reading,
TXT/CSV export and a styled residual layer. Needs no 2SFCA run – any polygon layer with an
accessibility field works.

**Spatial models (for clustered residuals).** LM-lag/LM-error tests (plus robust versions) always run
and recommend a model. Tick any of: *spatial lag* (ML, with total effects β/(1−ρ)), *spatial error*
(ML), *GWR* (adaptive bisquare kernel, bandwidth by AICc or fixed, local coefficient/pseudo-t/local-R²
layer with a significance map per predictor). Weights: row-standardised k-nearest neighbours. A model
comparison table (R², log-lik, AIC, residual Moran's I) shows which model absorbs the clustering.
Pure numpy; ~10 s for 3,400 EDs. Caveat: GWR with sparse dummy categories can be unstable – prefer
numeric predictors.

**Regression summary table (v1.4).** The default view shows one table per predictor: coefficient,
HC3 robust SE, robust t/p, stars, 95% CI, standardised beta (with a Weak/Moderate/Strong label), VIF
(with an OK/Moderate/High flag), classical SE/p for comparison and a plain-English line. A second view,
*model fit and diagnostics*, gives n, R², adjusted R², F, AIC/BIC, Breusch-Pagan test with a verdict,
highest VIF and Moran's I of residuals, each with "what it means". *Save regression summary (.csv)*
writes both blocks and notes in one file (UTF-8 with BOM, opens cleanly in Excel).

**Local Moran's I (LISA).** Pick any numeric field of the accessibility layer (or the OLS residuals
after a run), neighbours k, permutations (default 999), significance level and optional
Benjamini-Hochberg FDR correction. Output layer classes: High-High hot spot, Low-Low cold spot,
High-Low and Low-High spatial outliers, Not significant. A summary table and a by-ED table are added
to the view list and can be saved with *Save table shown*.

## Tests
`python -m unittest discover -s tests -v` (pure-Python engine, no QGIS needed).

## Status
Engine unit-tested. The QGIS GUI layer was written against the QGIS 4 / Qt6 API but has not
been executed inside a live QGIS 4 session; report any API error from the log pane.
