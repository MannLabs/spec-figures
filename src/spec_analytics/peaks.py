"""Back-compat shim -> spec_analytics.io.peaks (REFACTOR_PLAN.md step 3).

The PEAKS loader now lives in the io subpackage. This keeps
`spec_analytics.peaks` and `import peaks` (via the repo-root shim) working
until callers migrate to `spec_analytics.io.peaks`.
"""

from .io.peaks import *  # noqa: F401,F403
from .io.peaks import load_peaks  # noqa: F401
