"""Back-compat shim -> spec_analytics.io.diann (REFACTOR_PLAN.md step 3).

The DIA-NN loader now lives in the io subpackage. This keeps
`spec_analytics.diann` and `import diann` (via the repo-root shim) working
until callers migrate to `spec_analytics.io.diann`.
"""

from .io.diann import *  # noqa: F401,F403
from .io.diann import load_diann  # noqa: F401
