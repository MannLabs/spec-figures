"""Build compact, self-contained inputs for supplementary figure 4.

Provenance, not a required step: the four files this writes are part of the
deposited data and already sit in `supplementary_figure4/input/`. It is kept so
the derivation from the single-fiber report is reproducible. Outputs go under
SPEC_OUTPUT_ROOT, never into the input tree.

The membrane-class annotation is the one input that is not derived here: it was
built once from the UniProt subcellular-location API in the original experiment
folder. Point MEMBRANE_CLASS_SOURCE at that file to rebuild it; without it the
step is skipped and the deposited copy stands.
"""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import spec_config as _cfg  # noqa: E402

FIG5_SCRIPTS = Path(_cfg.REPO_ROOT) / 'figure5' / 'scripts'
DEPOSITED = Path(_cfg.cross_input('supplementary_figure4'))
OUT = Path(_cfg.output_dir(__file__, 'prep'))
MEMBRANE_CLASS_SOURCE = Path(
    os.environ.get('SPEC_MEMBRANE_CLASS_CSV',
                   DEPOSITED / 'protein_membrane_class.csv'))

sys.path.insert(0, str(FIG5_SCRIPTS))
import common as C  # noqa: E402

df = C.load_filtered(columns=['run', 'protein_group', 'genes', 'pg_intensity'])
pg_wide, gene_of = C.pg_log2_matrix(df)
sample_info = C.load_typed()
print(f'protein-group matrix: {pg_wide.shape[0]:,} groups x {pg_wide.shape[1]} fibers')

keep = pg_wide.index
print(f'protein groups kept (no coverage filter): {len(keep):,}')

pg_wide.to_parquet(OUT / 'pg_log2_matrix.parquet')
pd.DataFrame({'protein_group': keep,
              'gene': [gene_of.get(p, '') for p in keep]}
             ).to_csv(OUT / 'protein_genes.csv', index=False)
sample_info.to_parquet(OUT / 'sample_info_typed.parquet')
if MEMBRANE_CLASS_SOURCE.exists():
    shutil.copyfile(MEMBRANE_CLASS_SOURCE, OUT / 'protein_membrane_class.csv')
else:
    print(f'membrane-class annotation not found at {MEMBRANE_CLASS_SOURCE}; '
          'skipped (set SPEC_MEMBRANE_CLASS_CSV to rebuild it)')

for p in sorted(OUT.iterdir()):
    print(f'  {p.name:34s} {p.stat().st_size / 1e6:7.2f} MB')

print(f'\nwritten to {OUT}\nthe deposited copies the panels read are in {DEPOSITED}')

if not (OUT / 'protein_membrane_class.csv').exists():
    raise SystemExit(0)
cls = pd.read_csv(OUT / 'protein_membrane_class.csv')
print('\nmembrane classes in the annotation file:')
print(cls['membrane_class'].value_counts().to_string())
print('\ncoverage of the plotted set:',
      f"{cls.set_index('protein_group')['membrane_class'].reindex(keep).notna().mean():.1%}")
