"""Paths for the SPEC figure code.

Two roots, both derived from this file's own location so a fresh clone runs with
no configuration, and both overridable by environment variable:

    SPEC_DATA_ROOT      the input tree, laid out as one folder per figure, each
                        containing an `input/` directory of search outputs.
                        Defaults to the repository root, which is where the
                        deposited data unpacks to: <repo>/figure2/input/...
    SPEC_OUTPUT_ROOT    where figures, source data and caches are written.
                        Defaults to <repo>/output, which is git-ignored.

Set them only to keep the data or the outputs outside the repository:

    set SPEC_DATA_ROOT=D:\\SPEC_data                 (Windows)
    export SPEC_DATA_ROOT=/data/SPEC                (macOS / Linux)

Everything else is derived. A script asks for its own figure's input or output by
passing `__file__`; the figure name is taken from the directory the script lives in.
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.environ.get('SPEC_DATA_ROOT', REPO_ROOT)
OUTPUT_ROOT = os.environ.get(
    'SPEC_OUTPUT_ROOT', os.path.join(REPO_ROOT, 'output'))


def figure_of(script_path):
    """'.../figure2/scripts/panel_b.py' -> 'figure2'."""
    return os.path.basename(
        os.path.dirname(os.path.dirname(os.path.abspath(script_path))))


def input_dir(script_path, *parts):
    """Input directory of the figure this script belongs to."""
    return os.path.join(DATA_ROOT, figure_of(script_path), 'input', *parts)


def output_dir(script_path, *parts):
    """Output directory of the figure this script belongs to, created on demand."""
    d = os.path.join(OUTPUT_ROOT, figure_of(script_path), *parts)
    os.makedirs(d, exist_ok=True)
    return d


def data_dir(script_path, *parts):
    """Cache directory, kept beside the outputs rather than in the input tree."""
    d = os.path.join(OUTPUT_ROOT, figure_of(script_path), 'data')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, *parts)


def cross_input(figure, *parts):
    """Input directory of another figure, for the panels that read one."""
    return os.path.join(DATA_ROOT, figure, 'input', *parts)
