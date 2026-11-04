# Raw data — how to obtain and place it

The radar data used in Section 6 (Tables 5–6, Figures 6–8) are **not** included
in this package. They are obtained from the MTDSP data-sharing program and must
be cited.

## Citation

> N. Liu, J. Li, G. Wang, B. Chen, Z. Cao, Y. Dong, J. Guan, X. Jiang,
> Z. Zhang, W. Xue, "Radar sea-detection experiments and target characteristic
> data acquisition: a multi-source observation dataset of maritime targets,"
> *Journal of Radars*, vol. 14, no. 3, pp. 754–780, 2025.
>
> DOI: https://doi.org/10.12000/JR25001

The DOI above is the canonical link to the dataset description and to the access
procedure of the Maritime Target Data Sharing Program.

## Why the files are not included

- Total size of the ten files is about 3.3 GB.
- The data are distributed under the program's terms, not under this package's
  terms.

## Files expected by the scripts

Place the files in `T2exp/` (the directory that contains `_exp4/`), keeping the
original names:

| File | Radar / polarization | Mode | Used for |
|---|---|---|---|
| `MTDSP2022_HH_stare.mat` | SPPR50P HH | stare | Table 5, Figures 7–8 |
| `MTDSP2022_VV_stare.mat` | SPPR50P VV | stare | Table 5 |
| `MTDSP2025_HH_pure.mat` | SPPR50P HH | scan | Table 5, fast-time check |
| `MTDSP2025_VV_pure.mat` | SPPR50P VV | scan | Table 5, fast-time check |
| `MTDSP2025_HH_412331720.mat` | SPPR50P HH | scan | α statistics |
| `MTDSP2025_VV_412331720.mat` | SPPR50P VV | scan | α statistics |
| `MTDSP2025_HH_413020540.mat` | SPPR50P HH | scan | α statistics |
| `MTDSP2025_VV_413020540.mat` | SPPR50P VV | scan | α statistics |
| `MTDSP2020_HH_stare.mat` | 2018 radar HH | stare | α statistics |
| `MTDSP2019_HH_scan.mat` | 2018 radar HH | scan | α statistics |

## MAT layout (summary)

Each file is a MATLAB struct (see `T2exp/README.md` for the full table):

- `pc` — pulse × range-cell complex data, pulse-compressed;
- `if_raw` — pulse × 1760 int16, pre-compression IF (2025 files only);
- `clutter_cells` — target-free range cells (1-based);
- `target_cells`, `strong_cells`, `land_sectors_deg` — annotations;
- `grazing_deg` — grazing angle (NaN for 2019/2020, where mast height is unknown);
- `meta` — 44 fields (PRF, waveform, bandwidth, sea state, radar height, target, …).

If a different acquisition is used, keep the same variable names; the scripts
only rely on `pc`, `clutter_cells`, and (for the fast-time check) `if_raw`.

## Quick check

After placing the files:

```
cd T2exp
python _exp4/make_t2_figs.py      # should regenerate Figs. 6-7 from cached JSON
python -c "import scipy.io as s; d=s.loadmat('MTDSP2022_HH_stare.mat',variable_names=['pc','clutter_cells']); print(d['pc'].shape, d['clutter_cells'].size)"
```

The cached JSON in `T2exp/_verify/` and `T2exp/_exp4/` lets you rebuild Figures 6
and 7 (and inspect the recorded numbers) without downloading the `.mat` files.
