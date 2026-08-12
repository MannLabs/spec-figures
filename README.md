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
└── supplementary_figure1/ … 6/scripts/
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
python panel_b_sample_volume.py
```

It prints its summary numbers to the screen and writes three files to
`SPEC_OUTPUT_ROOT/figure2/`:

| file | what it is |
|---|---|
| `panel_b_sample_volume.pdf` | the panel, vector, with editable text |
| `panel_b_sample_volume.png` | the same at 300 dpi |
| `panel_b_sample_volume_sourcedata.csv` | the exact values plotted |

Summary tables are printed rather than saved, because they are cheap aggregations of
what is already in the source-data file.

To run everything in one figure:

```bash
cd figure3/scripts
for f in *.py; do python "$f"; done          # bash
Get-ChildItem *.py | ForEach-Object { python $_.Name }   # PowerShell
```

---

## 4. Order of execution

Most scripts are independent and can be run in any order. Four dependencies must be
respected:

1. **`figure5` is a pipeline.** Run in order:
   `01_load_and_filter.py` → `02_fiber_types.py` → then any of the numbered panel
   scripts (`05`, `07`, `08`, `11`–`14`). The later steps read caches written by the
   first two, and fail without them.
2. **`supplementary_figure5/scripts/prep_*.py`** must run before that figure's panels.
3. **`supplementary_figure3`** imports `figure2/scripts/common_figure2.py` and reads
   `figure2`'s input tree, so `figure2`'s input must be present.
4. **`supplementary_figure2/scripts/supplement_reproducibility.py`** reads
   `supplementary_figure1`'s input tree.

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
| `FileNotFoundError` on a `report.parquet` | `SPEC_DATA_ROOT` is wrong, or the input tree is not laid out as above |
| a `figure5` panel script fails immediately | run `01_load_and_filter.py` and `02_fiber_types.py` first |
| a figure looks stale after changing the data | delete the matching file in `SPEC_OUTPUT_ROOT/<figure>/data/` |

---

## Data

The DIA-NN search outputs are deposited in MassIVE. Download that tree and point
`SPEC_DATA_ROOT` at it.
