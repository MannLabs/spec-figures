"""In-silico digestion, sequence properties (GRAVY, coverage), FASTA loading,
and per-protein-group info. Extracted from _core.py (REFACTOR_PLAN.md step 2);
behaviour unchanged."""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd


_PROTEASE_SITES = {
    'trypsin': 'RK',
    'lysc': 'K',
    'argc': 'R',
    'chymotrypsin': 'FWY',
    'gluc': 'ED',
}


def count_missed_cleavages(sequence, protease='trypsin'):
    """Count missed cleavages within `sequence` (excluding the C-terminus)."""
    if not sequence:
        return 0
    sites = _PROTEASE_SITES.get(protease.lower(), 'RK')
    return sum(1 for aa in sequence[:-1] if aa in sites)


def digest_protein(sequence, *, protease='trypsin', max_missed_cleavages=1,
                   keil_rule=False):
    """In-silico digest `sequence` into peptide tuples (start, end, peptide).

    Cuts after every cleavage residue by default. `max_missed_cleavages`
    controls how many missed cleavages each generated peptide may carry; the
    set of returned peptides therefore includes the fully-cleaved set plus any
    longer peptides that span up to `max_missed_cleavages` extra cut sites.

    Pass `keil_rule=True` to suppress cuts before Proline for trypsin/LysC/ArgC
    (the Keil 1992 observation). Off by default — most modern search engines
    treat Pro-blocking as optional, and disabling it gives a slightly more
    permissive theoretical coverage.
    """
    if not sequence:
        return []
    sites = _PROTEASE_SITES.get(protease.lower(), 'RK')
    apply_keil = keil_rule and protease.lower() in ('trypsin', 'lysc', 'argc')
    cut_points = [0]
    for i in range(len(sequence) - 1):
        if sequence[i] in sites:
            if apply_keil and sequence[i + 1] == 'P':
                continue
            cut_points.append(i + 1)
    cut_points.append(len(sequence))

    peptides = []
    for i in range(len(cut_points) - 1):
        for mc in range(max_missed_cleavages + 1):
            j = i + 1 + mc
            if j >= len(cut_points):
                break
            start, end = cut_points[i], cut_points[j]
            peptides.append((start, end, sequence[start:end]))
    return peptides


def theoretical_coverage(
    sequence,
    *,
    protease='trypsin',
    max_missed_cleavages=1,
    min_peptide_length=6,
    max_peptide_length=40,
    keil_rule=False,
):
    """Maximum sequence coverage achievable by an in-silico tryptic digest.

    Sums the residues of `sequence` covered by any peptide in the digest whose
    length falls in [min_peptide_length, max_peptide_length], and returns the
    percentage. Equivalent to "if every detectable peptide had been observed,
    what would coverage have been?"
    """
    if not sequence:
        return 0.0
    L = len(sequence)
    covered = bytearray(L)
    for start, end, _ in digest_protein(sequence, protease=protease,
                                        max_missed_cleavages=max_missed_cleavages,
                                        keil_rule=keil_rule):
        plen = end - start
        if plen < min_peptide_length or plen > max_peptide_length:
            continue
        for k in range(start, end):
            covered[k] = 1
    return 100.0 * sum(covered) / L


# Kyte-Doolittle hydropathy index (Kyte & Doolittle, J. Mol. Biol. 1982).
_KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


def gravy(sequence):
    """Grand average of hydropathy (GRAVY, Kyte-Doolittle) for a sequence.

    Returns the mean hydropathy across all residues, ignoring unknown letters.
    NaN for empty input or sequences containing only unrecognised characters.
    More positive = more hydrophobic; more negative = more hydrophilic.
    """
    if not sequence:
        return float('nan')
    vals = [_KYTE_DOOLITTLE[a] for a in sequence if a in _KYTE_DOOLITTLE]
    if not vals:
        return float('nan')
    return sum(vals) / len(vals)


def compute_theoretical_coverage(
    protein_info,
    protein_sequences,
    *,
    protease='trypsin',
    max_missed_cleavages=1,
    min_peptide_length=6,
    max_peptide_length=40,
    keil_rule=False,
):
    """Theoretical coverage per row of `protein_info` (uses representative_protein).

    Returns a Series aligned to `protein_info.index`. Rows whose representative
    protein is missing from `protein_sequences` get NaN.
    """
    # Cache by leader id so we don't re-digest the same protein per condition.
    cache = {}
    out = []
    for leader in protein_info['representative_protein']:
        if leader not in cache:
            seq = protein_sequences.get(leader)
            if seq is None or len(seq) == 0:
                cache[leader] = float('nan')
            else:
                cache[leader] = theoretical_coverage(
                    seq,
                    protease=protease,
                    max_missed_cleavages=max_missed_cleavages,
                    min_peptide_length=min_peptide_length,
                    max_peptide_length=max_peptide_length,
                    keil_rule=keil_rule,
                )
        out.append(cache[leader])
    return pd.Series(out, index=protein_info.index, name='theoretical_coverage_pct')


# ============================================================================
# Protein-level info (sequence coverage and peptides per protein group)
# ============================================================================

def load_protein_sequences(path: str) -> dict[str, str]:
    """Load `{protein_id: AA_sequence}` from FASTA or DIA-NN protein_description.tsv.

    Accepted inputs:
      * FASTA (`.fa`, `.fasta`) — UniProt-style headers `>sp|ACC|NAME ...` and
        `>tr|ACC|NAME ...` are parsed to use the bare accession as the key.
        Other headers fall back to the first whitespace-delimited token.
      * DIA-NN `protein_description.tsv` — uses `Protein.Id` as the key and
        `Sequence` as the value. This file is written by DIA-NN alongside the
        report and works as a sequence source for any engine whose protein
        groups use the same UniProt accessions.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.tsv':
        tab = pd.read_csv(path, sep='\t')
        if 'Protein.Id' not in tab.columns or 'Sequence' not in tab.columns:
            raise ValueError(
                f'TSV {path!r} missing required columns Protein.Id / Sequence; '
                f'got {list(tab.columns)}'
            )
        return dict(zip(tab['Protein.Id'].astype(str), tab['Sequence'].astype(str)))

    out: dict[str, str] = {}
    cur_id: str | None = None
    cur_seq: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('>'):
                if cur_id is not None:
                    out[cur_id] = ''.join(cur_seq)
                header = line[1:].strip()
                m = re.match(r'(?:sp|tr)\|([^|]+)\|', header)
                cur_id = m.group(1) if m else header.split()[0]
                cur_seq = []
            elif line:
                cur_seq.append(line)
        if cur_id is not None:
            out[cur_id] = ''.join(cur_seq)
    return out


def compute_protein_info(
    df,
    sample_info,
    protein_sequences,
    *,
    group_col='condition2',
    hue_col=None,
):
    """Per-(protein_group x group) summary: peptide count and sequence coverage.

    For each value of `sample_info[group_col]`, restricts to that group's runs
    and computes for every detected `protein_group`:
      * `n_peptides` — number of unique stripped-sequence peptides
        (peptidoforms collapsed; `sequence` column).
      * `coverage_pct` — percentage of the leader protein's sequence covered
        by those peptides. The leader is the first id in `protein_group`
        (semicolon-separated when the group has multiple). Coverage is
        computed by finding all (overlap-allowed) substring matches of each
        peptide in the leader sequence and counting unique covered residues.
      * `protein_length` — length of the leader sequence.
      * `representative_protein` — the leader id used.

    Pass `group_col=None` for a single combined row per protein group across
    all runs in `df` (group label `'all'`).

    Pass `hue_col=` to partition by both `group_col` and `hue_col`; the
    output gains a `hue` column and one row per (protein_group, group, hue).

    Protein groups whose leader id is not in `protein_sequences` get NaN
    `coverage_pct` and `protein_length`, but `n_peptides` is still reported.
    """
    if group_col is None:
        partitions = [(('all', None), set(df['run'].dropna().unique()))]
    else:
        partitions = []
        keys = [group_col] + ([hue_col] if hue_col else [])
        for combo, sub in sample_info.groupby(keys, sort=False):
            if not isinstance(combo, tuple):
                combo = (combo,)
            grp = combo[0]
            hue = combo[1] if hue_col else None
            partitions.append(((grp, hue), set(sub['run'])))

    # First non-empty gene per protein_group, evaluated globally so it stays
    # consistent across condition rows.
    gene_map = {}
    if 'genes' in df.columns:
        gene_map = (df.assign(_g=df['genes'].astype(str).str.split(';').str[0])
                      .groupby('protein_group')['_g']
                      .agg(lambda s: next((v for v in s if v), ''))
                      .to_dict())

    rows = []
    for (grp, hue), runs in partitions:
        sub = df[df['run'].isin(runs)].dropna(subset=['protein_group', 'sequence'])
        if sub.empty:
            continue
        peptides_by_pg = (
            sub.groupby('protein_group')['sequence']
               .agg(lambda s: frozenset(s.unique()))
        )
        for pg, peps in peptides_by_pg.items():
            leader = str(pg).split(';')[0]
            seq = protein_sequences.get(leader)
            if seq is not None and len(seq) > 0:
                covered = bytearray(len(seq))
                for pep in peps:
                    if not pep:
                        continue
                    start = 0
                    while True:
                        idx = seq.find(pep, start)
                        if idx == -1:
                            break
                        for i in range(idx, idx + len(pep)):
                            covered[i] = 1
                        start = idx + 1  # allow overlapping matches
                cov_pct = 100.0 * sum(covered) / len(seq)
                prot_len = len(seq)
            else:
                cov_pct = np.nan
                prot_len = np.nan
            row = {
                'protein_group': pg,
                'gene': gene_map.get(pg, ''),
                'group': grp,
                'representative_protein': leader,
                'protein_length': prot_len,
                'n_peptides': len(peps),
                'coverage_pct': cov_pct,
            }
            if hue_col:
                row['hue'] = hue
            rows.append(row)
    return pd.DataFrame(rows)
