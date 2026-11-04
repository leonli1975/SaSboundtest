# Reproducibility package

**Paper.** *Multicomponent and Competing-Clutter Mismatch in the SαS CRLB for LFM*
Guocai Li, College of Electrical Engineering, Sichuan University.
Target journal: Signal Processing (Elsevier).

This package contains the code, the JSON results behind every table and figure,
the figure scripts, and the LaTeX sources. **The raw radar data are not
redistributed**; they are referenced in Section 2 and must be obtained from the
original data provider.

---

## 1. What is included

```
submission_package/
├── README.md                 this file
├── requirements.txt          Python dependencies
├── MANIFEST.txt              SHA-256 of every file in the package
├── data/README.md            how to obtain and place the raw data
├── sim/                      synthetic experiments (CRLB, estimators, GELCT)
├── results/                  JSON results read by the tables/figures
├── gelct_lfm/                GELCT reproduction (Liu et al., IEEE Syst. J. 2021)
├── T2exp/                    MTDSP T2 experiments (code + results, no .mat)
└── paper/                    LaTeX sources, figures, bibliography, class/bst
```

`T2exp/*.mat` (~3.3 GB) are intentionally absent. `T2exp/README.md` documents the
MAT layout and the ten file names; Section 2 below explains how to obtain them.

## 2. Raw data (not included)

The real-data experiments use the **MTDSP** X-band sea-clutter dataset (T2, LFM
waveform, HH/VV, scan/stare modes). Please cite:

> N. Liu, J. Li, G. Wang, B. Chen, Z. Cao, Y. Dong, J. Guan, X. Jiang,
> Z. Zhang, W. Xue, "Radar sea-detection experiments and target characteristic
> data acquisition: a multi-source observation dataset of maritime targets,"
> *Journal of Radars*, vol. 14, no. 3, pp. 754–780, 2025.
> https://doi.org/10.12000/JR25001

Access is through the Maritime Target Data Sharing Program described in that
paper. The data are **not** included here for two reasons: size (~3.3 GB) and
redistribution terms.

**Placement.** Download the ten files and put them in `T2exp/` with their
original names:

```
MTDSP2025_HH_413020540.mat   MTDSP2025_VV_413020540.mat
MTDSP2025_HH_412331720.mat   MTDSP2025_VV_412331720.mat
MTDSP2025_HH_pure.mat        MTDSP2025_VV_pure.mat
MTDSP2022_HH_stare.mat       MTDSP2022_VV_stare.mat
MTDSP2020_HH_stare.mat       MTDSP2019_HH_scan.mat
```

The scripts read them from `T2exp/` (see `T2exp/README.md` for the variable
layout: `pc`, `if_raw`, `clutter_cells`, `target_cells`, `grazing_deg`, `meta`).
Only the four files `MTDSP2022_HH_stare`, `MTDSP2022_VV_stare`,
`MTDSP2025_HH_pure`, `MTDSP2025_VV_pure` are needed for Tables 5–6 and
Figures 6–8; the fast-time check additionally uses `if_raw` from the two 2025
`_pure` files. The 2019/2020 files have no `if_raw` and no mast height, so their
`grazing_deg` is NaN by construction.

## 3. Setup

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+, NumPy/SciPy/Matplotlib. The LaTeX sources compile with a standard
TeX Live (`pdflatex` + `bibtex`); `paper/` already contains the class and `.bst`
files used (`cas-sc.cls`, `elsarticle-num-doi.bst`, `cas-model2-names.bst`).

Synthetic scripts write into `results/`, and the figure scripts write into
`paper/figures/`. All paths are relative to this package, so no environment
variables are needed.

## 4. What reproduces what

### Figures

| Fig. | File | Script / data | Notes |
|---|---|---|---|
| 1 | `fig_crlb_dev.pdf` | data: `sim/kappa_dense.py` → `results/kappa_dense.json`; `sim/crlb_lfm.py` → `results/crlb_lfm.json` | relative deviation (dB) of real κ and complex J; the committed PDF is the final styling |
| 2 | `fig2_estimators.pdf` | `sim/unified_single.py`; `sim/paper_figures.py` (fig2) | `results/unified_single.json` |
| 3 | `fig8_snr.pdf` | `sim/snr_sweep.py` | `results/snr_sweep.json`; final styling from that JSON |
| 4 | `fig_block_crlb.pdf` | `sim/kappa_multicomponent.py` | `results/crlb_multicomponent.json` |
| 5(a) | `fig4_multicomponent.pdf` | `sim/unified_multi_df.py`; `sim/paper_figures.py` (fig4) | `results/unified_multi_df.json` |
| 5(b) | `fig5_alpha.pdf` | `sim/unified_multi_alpha.py`; `sim/paper_figures.py` (fig5) | `results/unified_multi_alpha.json` |
| 6 | `fig9_t2_alpha.pdf` | `T2exp/_exp4/make_t2_figs.py` | `T2exp/_verify/step2_alpha.csv` (no `.mat` needed) |
| 7 | `fig10_t2_estimator.pdf` | `T2exp/_exp4/make_t2_figs.py` | `T2exp/_exp4/exp3_rerun.json` (no `.mat` needed) |
| 8 | `fig11_t2_two_ray.pdf` | data: `T2exp/_exp4/step4_two_ray_inject.py` | `step4_two_ray_inject.json`; final styling from that JSON |
| 9 | `fig3_frft.pdf` | `sim/estimators_frft.py`; `sim/paper_figures.py` (fig3) | `results/estimators_frft.json` |

### Tables

| Table | Content | Source |
|---|---|---|
| 1 | CRLB and CRLB ratios | `results/kappa_dense.json`, `results/kappa_real.json`, `results/crlb_lfm.json` |
| 2 | Estimator efficiency (LS/Huber/MLE/GELCT) | `sim/unified_single.py`, `sim/gelct_vs_single.py` |
| 3 | Trimmed MSE vs SNR | `sim/snr_sweep.py` → `results/snr_sweep.json` |
| 4 | Single vs two component | `sim/unified_multi_df.py`, `sim/unified_multi_a13_df24.py`, `sim/unified_multi_df_500.py`, `sim/unified_multi_p3.py` |
| 5 | T2 real clutter (LS/Oracle/GELCT) | `T2exp/_exp4/exp3_rerun_v2.py` (needs `.mat`), `T2exp/_exp4/exp6_gelct_real.py` |
| 6 | Two-component fit on real clutter | `T2exp/_exp4/exp5_two_component_real.py` |
| 7 | GELCT vs refinements, LS collapse | `gelct_lfm/experiments/collapse_test.py`, `gelct_lfm/experiments/gelct_refine_vs_oracle.py` |

### Re-running the real-data experiments

```
cd T2exp
python _exp4/exp3_rerun_v2.py      # LS/Huber/Oracle, Table 5 rows
python _exp4/exp6_gelct_real.py    # adds GELCT to Table 5
python _exp4/exp5_two_component_real.py   # Table 6 (2-component fit)
python _exp4/step4_two_ray_inject.py      # Fig. 8 data
python _exp4/make_t2_figs.py              # Figs. 6-7 (from cached JSON)
```

`make_t2_figs.py` and the alpha figure do not require the `.mat` files, because
their inputs are cached in `T2exp/_verify/` and `T2exp/_exp4/`.

## 5. Numerical settings and caveats

- Synthetic single-component runs: `N = 256`, 100 trials, seeds `100000 + i`,
  95% bootstrap confidence intervals; MAD normalized by the CRLB standard
  deviation.
- Multicomponent runs: 100 trials (25 in the earlier exploratory `p3` run, which
  is reported as such); separated and sub-Rayleigh spacings as stated in the
  tables.
- Real-data runs: 104 rows per file (8 clutter cells × 13 blocks), +2 dB per
  sample unless stated; the GELCT is initialized by its own full-window chirp
  search.
- Monte Carlo results can differ in the last digits across BLAS/CPU; the JSON
  files are the values used in the manuscript.
- `paper/figures/*.pdf` are the final versions used in the manuscript. Figures
  1, 3 and 8 were styled directly from their JSON data; the scripts above
  regenerate the underlying numbers and the comparable plots.

## 6. License and contact

Code is provided for review and reproducibility. No license is granted for
redistribution of the MTDSP data; cite the dataset paper above. Contact:
Guocai Li, liguocai@scu.edu.cn (ORCID 0000-0002-3191-146X).
