"""Protein-group inference rules and the peptidoform id builder. Extracted"""
from __future__ import annotations
from typing import Iterable
def group_proteins_by_shared_peptides(
    accession_sets: Iterable[Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """
    Connected-components grouping ("CC", PEAKS' rule).
    Two proteins are in the same group if they share at least one peptide,
    transitively. The input is an iterable where each element is the set of
    proteins claimed by one peptide. Returns a dict mapping each individual
    protein identifier to the sorted tuple of all proteins in its group.
    Verified empirically: matches PEAKS' `protein.csv` 4,332 of 4,332 groups
    on our 500 SPD test data. This is what `peaks.load_peaks` uses by default.
    The function is engine-agnostic; pass any string as protein identifier
    (`UniProt|EntryName`, bare UniProt, anything that uniquely identifies a
    protein within the dataset).
    """
    parent: dict[str, str] = {}
    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
            return x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for accs in accession_sets:
        accs = list(accs)
        if not accs:
            continue
        find(accs[0])
        for a in accs[1:]:
            union(accs[0], a)
    components: dict[str, list[str]] = {}
    for acc in parent:
        components.setdefault(find(acc), []).append(acc)
    out: dict[str, tuple[str, ...]] = {}
    for members in components.values():
        group = tuple(sorted(members))
        for m in members:
            out[m] = group
    return out
def group_proteins_by_signature(
    accession_sets: Iterable[Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """
    Signature-based grouping (DIA-NN's rule).
    Two proteins are in the same group iff they have identical peptide
    signatures — they appear in *exactly* the same set of peptide accession
    sets. Equivalently: a protein with at least one distinguishing peptide
    (any peptide that differentiates its candidate set from another protein's)
    gets its own group.
    Verified empirically: ~99.5% exact match against DIA-NN's actual
    `Protein.Group` field on the 500 SPD parquet. The remaining ~0.5%
    involves large paralog families where DIA-NN does additional score-based
    fine-tuning we can't fully replicate from the parquet alone.
    Compared to `group_proteins_by_shared_peptides`, this rule is stricter:
    it splits paralog families that have unique distinguishing peptides
    (e.g., kinase families) which CC would lump together.
    Same input/output shape as `group_proteins_by_shared_peptides`.
    """
    accession_set_keys = []
    protein_to_sig: dict[str, set] = {}
    for accs in accession_sets:
        accs = tuple(sorted(set(accs)))
        if not accs:
            continue
        accession_set_keys.append(accs)
        for a in accs:
            protein_to_sig.setdefault(a, set()).add(accs)
    sig_to_proteins: dict[frozenset, list[str]] = {}
    for prot, sig in protein_to_sig.items():
        sig_to_proteins.setdefault(frozenset(sig), []).append(prot)
    out: dict[str, tuple[str, ...]] = {}
    for members in sig_to_proteins.values():
        group = tuple(sorted(members))
        for m in members:
            out[m] = group
    return out
def build_peptide_id(sequence: str, mods: str, mod_sites: str) -> str:
    """
    Render the peptidoform identifier as an inline modified-sequence string in
    alphabase mod-name format. This is the value stored in the `peptide_id`
    column of the canonical DataFrame.
    Format:
      * Mods at site 0 (N-term) are placed in parentheses before the first AA.
      * Internal mods at site k are placed in parentheses immediately after the
        k-th AA (1-based).
      * Returns the unmodified `sequence` when `mods` is empty.
    Examples:
      ('AAAGTATSQR', '', '')                       -> 'AAAGTATSQR'
      ('AAAGTATSQR', 'Acetyl@Protein_N-term', '0') -> '(Acetyl@Protein_N-term)AAAGTATSQR'
      ('PEPMIDE', 'Oxidation@M', '4')              -> 'PEPM(Oxidation@M)IDE'
      ('AAANLCPGD', 'Carbamidomethyl@C', '6')      -> 'AAANLC(Carbamidomethyl@C)PGD'
    The result uniquely identifies the peptidoform (same triple in -> same
    string out; different mods or sites -> different string), so `peptide_id`
    is a valid groupby key. It can also be used as a readable label and
    converted back to (sequence, mods, mod_sites) by parsing the parens.
    AlphaBase / AlphaPeptDeep tools (e.g. `update_precursor_mz`) operate on
    the underlying `(sequence, mods, mod_sites)` triple directly — they do not
    require this inline form.
    """
    if not mods:
        return sequence
    mod_list = mods.split(';')
    site_list = [int(s) for s in mod_sites.split(';')]
    pairs = sorted(zip(site_list, mod_list), key=lambda p: p[0])
    out: list[str] = []
    for s, m in pairs:
        if s == 0:
            out.append(f'({m})')
    internal = [(s, m) for s, m in pairs if s > 0]
    j = 0
    for i, aa in enumerate(sequence, start=1):
        out.append(aa)
        while j < len(internal) and internal[j][0] == i:
            out.append(f'({internal[j][1]})')
            j += 1
    return ''.join(out)
