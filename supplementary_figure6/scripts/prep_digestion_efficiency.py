"""Per-fiber fully-cleaved precursor rate, cached for the digestion-efficiency panel."""
import os
import sys
import pandas as pd
import pyarrow.parquet as pq
import spec_analytics as core
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
REPORT = _cfg.cross_input('figure5', 'report.parquet')
INPUT = _cfg.input_dir(__file__)
OUT = os.path.join(INPUT, 'per_fiber_digestion.csv')
QVALUE = 0.01
BATCH_ROWS = 500_000
COLUMNS = ['Run', 'Stripped.Sequence', 'Precursor.Quantity', 'Q.Value',
           'PG.Q.Value', 'Decoy']
def main():
    available = set(pq.ParquetFile(REPORT).schema_arrow.names)
    columns = [c for c in COLUMNS if c in available]
    memo, acc = {}, {}
    reader = pq.ParquetFile(REPORT)
    print(f'streaming {reader.metadata.num_rows:,} rows in {BATCH_ROWS:,}-row batches')
    for i, batch in enumerate(reader.iter_batches(batch_size=BATCH_ROWS,
                                                  columns=columns)):
        d = batch.to_pandas()
        mask = (d['Q.Value'] <= QVALUE) & (d['PG.Q.Value'] <= QVALUE) \
            & (d['Precursor.Quantity'] > 0)
        if 'Decoy' in d:
            mask &= d['Decoy'] == 0
        d = d[mask]
        if d.empty:
            continue
        for seq in d['Stripped.Sequence'].unique():
            if seq not in memo:
                memo[seq] = core.count_missed_cleavages(seq, protease='trypsin')
        is_mc0 = d['Stripped.Sequence'].map(memo) == 0
        for run, grp in d.assign(is_mc0=is_mc0).groupby('Run', sort=False):
            a = acc.setdefault(run, [0, 0, 0.0, 0.0])
            a[0] += len(grp)
            a[1] += int(grp['is_mc0'].sum())
            a[2] += float(grp['Precursor.Quantity'].sum())
            a[3] += float(grp.loc[grp['is_mc0'], 'Precursor.Quantity'].sum())
        if (i + 1) % 5 == 0:
            print(f'  {(i + 1) * BATCH_ROWS:,} rows, {len(acc)} runs, '
                  f'{len(memo):,} distinct sequences')
    out = pd.DataFrame([
        {'run': run, 'precursors': n, 'mc0_precursors': n0,
         'mc0_by_count': 100 * n0 / n, 'mc0_by_intensity': 100 * q0 / q}
        for run, (n, n0, q, q0) in acc.items()]).sort_values('run')
    os.makedirs(INPUT, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f'\n{len(out)} runs, {len(memo):,} distinct stripped sequences')
    print(out[['mc0_by_count', 'mc0_by_intensity']].describe().round(2).to_string())
    print(f'wrote {OUT}')
if __name__ == '__main__':
    main()
