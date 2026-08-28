# SPEC — figure code

Analysis code for the SPEC (Solid-Phase Extraction Capture) manuscript. Every figure
and supplementary panel in the paper is produced by a script in this repository, from
the DIA-NN search outputs deposited with the paper.

The repository is self-contained: it bundles the analysis library it needs, so nothing
outside this folder has to be installed first.

```
SPEC_code/
├── spec_config.py                    the two paths, both defaulting into this folder
├── pyproject.toml                    package metadata (lower bounds)
├── environment.yml                   the exact versions every figure was produced with
├── dist/                             pre-built wheel, for sharing
├── src/spec_analytics/               the analysis library
├── figure1/ … figure6/scripts/       one script per panel or panel group
└── supplementary_figure1/ … 7/scripts/
```

---

## 1. Install

Python **3.12 or newer**. From a shell, in this folder:

```bash
pip install -e .
```

For the exact versions the published figures were produced with, create the pinned
environment first:

```bash
mamba env create -f environment.yml
conda activate spec-paper
pip install -e .
```

That installs `spec_analytics` and its dependencies (numpy, pandas, scipy,
scikit-learn, statsmodels, matplotlib, seaborn, pyarrow, matplotlib-venn, pyteomics,
directlfq). `-e` means editable: the installed package points back at
`src/spec_analytics`, so the code you read is the code that runs.

**This step is required before any script will work.** The scripts do
`import spec_analytics as core`, which fails until the package is installed.

### Sharing without this folder

A colleague who only wants the library can install the pre-built wheel — one file, no
repository, no package index:

```bash
pip install dist/spec_analytics-0.2.0-py3-none-any.whl
```

To rebuild it after changing the library:

```bash
pip wheel --no-deps -w dist .
```

Reproducing figures needs the folder as well, since the figure scripts are run from
here rather than installed.

### Check it worked

```bash
python -c "import spec_analytics as core; print(core.__file__)"
```

should print a path inside `src/spec_analytics`.

---

## 2. Point it at the data

All paths come from `spec_config.py`, which defines exactly two roots. Both default
to this folder, so **the normal case needs no configuration**: unpack the deposited
archive here and the input folders land where the scripts look.

| variable | default | meaning |
|---|---|---|
| `SPEC_DATA_ROOT` | this folder | the input tree: one folder per figure, each containing an `input/` directory of search outputs |
| `SPEC_OUTPUT_ROOT` | `./output` | where figures, source data and caches are written |

```
SPEC_code/
├── figure2/input/H032_E305/SPEC/200ng/report.parquet
├── figure2/scripts/panel_c_pg_vs_input.py
├── figure4/input/H032_E127/single-shot_SPEC/report.parquet
└── …
```

Set the variables only to keep the data or the outputs somewhere else — a different
disk, or outside the repository:

```bash
# Windows
set SPEC_DATA_ROOT=D:\SPEC_data
set SPEC_OUTPUT_ROOT=D:\SPEC_figures

# macOS / Linux
export SPEC_DATA_ROOT=/data/SPEC
export SPEC_OUTPUT_ROOT=/data/SPEC_figures
```

The input tree must keep the layout it is deposited in either way; `input/` folders
and `output/` are git-ignored, so the data drop never shows up as untracked changes.

**Nothing is ever written into the input tree.** Every output goes under
`SPEC_OUTPUT_ROOT`, so the deposited data cannot be modified by running the code.

---

## 3. Run a script

Each script is standalone and is run **from its own directory**:

```bash
cd figure2/scripts
python panel_b_sample_volume_peptides.py
```

It prints its summary numbers to the screen and writes three files to
`SPEC_OUTPUT_ROOT/figure2/`:

| file | what it is |
|---|---|
| `panel_b_sample_volume_peptides.pdf` | the panel, vector, with editable text |
| `panel_b_sample_volume_peptides.png` | the same at 300 dpi |
| `panel_b_sample_volume_peptides_sourcedata.csv` | the exact values plotted |

Summary tables are printed rather than saved, because they are cheap aggregations of
what is already in the source-data file.

### A script is named after the panel it draws, not the folder it sits in

Every script's file name carries the panel it produces, so
`figure3/scripts/panel_c_channel_overlap.py` draws Fig. 3c and
`supplementary_figure4/scripts/suppl4_a_recovery_volume.py` draws
Supplementary Fig. 4a.

Six of them sit in a folder that does not match that name, and deliberately:
`spec_config` resolves a script's **input** directory from the folder it lives in,
so a supplementary panel built from a main figure's search output has to stay
beside that figure's data. Those six write across into the supplement:

| script | draws | writes into |
|---|---|---|
| `figure1/scripts/suppl2_a_lc_comparison.py` | Suppl. Fig. 2a | `supplementary_figure2/` |
| `figure1/scripts/suppl2_b_lc_correlation.py` | Suppl. Fig. 2b | `supplementary_figure2/` |
| `figure2/scripts/suppl3_ab_kinetics.py` | Suppl. Fig. 3a, b | `supplementary_figure3/` |
| `figure2/scripts/suppl4_b_volume_protein_groups.py` | Suppl. Fig. 4b | `supplementary_figure4/` |
| `figure2/scripts/suppl4_c_input_protein_groups.py` | Suppl. Fig. 4c | `supplementary_figure4/` |
| `figure2/scripts/suppl4_d_detergent_protein_groups.py` | Suppl. Fig. 4d | `supplementary_figure4/` |

### Or use the notebooks

`notebooks/` holds two, and they are the easiest way in:

| notebook | what it does |
|---|---|
| `00_getting_started.ipynb` | checks the install, prints the resolved paths, reports which figure inputs are present, then builds one panel. Under a minute — run this first. |
| `01_all_figures.ipynb` | rebuilds every panel in the paper, one cell per figure, showing each panel inline. Roughly 15 minutes end to end. Generated together with the scripts, so it cannot name a script this repository does not carry. |

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/00_getting_started.ipynb
```

They run the same panel scripts in a subprocess rather than duplicating any
analysis code, so a notebook cannot disagree with the script or with the source
data. They are stored unexecuted — an executed copy would embed every 300 dpi
panel — so expect empty cells until you run them.

To run everything in one figure:

```bash
cd figure3/scripts
for f in *.py; do python "$f"; done          # bash
Get-ChildItem *.py | ForEach-Object { python $_.Name }   # PowerShell
```

---

## 4. Order of execution

Most scripts are independent and can be run in any order. Five dependencies must be
respected:

1. **`figure5` is a pipeline.** Run in order:
   `01_load_and_filter.py` → `02_fiber_types.py` → then any of the numbered panel
   scripts (`05`, `08`, `11`–`15`). The later steps read caches written by the
   first two, and fail without them.
2. **`supplementary_figure6/scripts/prep_digestion_efficiency.py`** must run before
   that figure's panel. `protein_membrane_class.csv` and the protein-group matrix it
   also needs are provided in `supplementary_figure6/input/`, because they derive
   from an annotation step in the source experiment rather than from this code.
3. **`supplementary_figure3`** and **`supplementary_figure4`** have no `input/` of
   their own: both read `figure2`'s input tree through
   `figure2/scripts/common_figure2.py`, so `figure2`'s input must be present.
4. **`supplementary_figure2/scripts/supplement_reproducibility.py`** reads
   `supplementary_figure1`'s input tree, and panels 2a and 2b are drawn by
   `figure1/scripts/suppl2_a_lc_comparison.py` and `suppl2_b_lc_correlation.py`
   (see the table in section 3).
5. **`figure1/scripts/panel_c_confinement.py`** reads
   `figure1/input/confinement_profile.npz`, which is committed here rather than
   downloaded — it is measured off the manuscript's artwork, not a search output,
   so it has no place in the MassIVE deposit and `.gitignore` exempts it. The two
   `figure1/scripts/prep_*.py` scripts that produce it read the manuscript's
   Illustrator file, which is not part of the deposit; they are included as
   provenance for the measurement, not as a step you have to run.

Files named `common_*.py` are imported by the panel scripts, not run directly.

---

## 5. Caches

Some scripts cache an expensive intermediate in `SPEC_OUTPUT_ROOT/<figure>/data/`.
Delete a cache file to force a recompute. Two are slow to rebuild:

- the E256 peptide tables (`supplementary_figure3`) — several minutes
- the single-fiber quantification (`figure5/01_load_and_filter.py`) — directLFQ over
  164 runs

Everything else recomputes in seconds to a minute.

---

## 6. Reproducibility check

`figure5/scripts/15_outlier_filter_sensitivity.py` re-runs the single-fiber
differential analysis with the technical-outlier filter disabled and reports how far
the regulated set, the fold changes and the marker calls move. It writes to
`SPEC_OUTPUT_ROOT/figure5/sensitivity_no_outlier_filter/` and does not disturb the
figure outputs.

---

## 7. Troubleshooting

| symptom | cause |
|---|---|
| `ModuleNotFoundError: No module named 'spec_analytics'` | the package is not installed — see step 1 |
| `ValueError: assignment destination is read-only`, inside directLFQ | pandas 3 is installed. directLFQ 0.3.3 declares no upper bound and breaks there; `pyproject.toml` caps it at `<3`, so this only reaches an environment that already had pandas 3. `pip install "pandas<3"` |
| `FileNotFoundError` on a `report.parquet` | `SPEC_DATA_ROOT` is wrong, or the input tree is not laid out as above |
| a `figure5` panel script fails immediately | run `01_load_and_filter.py` and `02_fiber_types.py` first |
| a figure looks stale after changing the data | delete the matching file in `SPEC_OUTPUT_ROOT/<figure>/data/` |

---

## Library scope

`src/spec_analytics` is the subset of the analysis library these figures reach — 18
of its 34 modules. The raw-file subpackage is not included: no figure in this paper
opens a raw file. Nor are the plotting modules for rank curves, set overlaps,
intensity distributions and annotated spectra, or the `species`, `reshape`,
`pipeline`, `plate_map`, `msqueue`, `peaks` and `diann` modules. The retained unit
tests run against exactly what is vendored:

```bash
python -m pytest unit_tests -q
```

Tests needing a PEAKS fixture skip: every search output in this paper is DIA-NN.

---

## Data

The DIA-NN search outputs are deposited in MassIVE. Download that tree and point
`SPEC_DATA_ROOT` at it.
