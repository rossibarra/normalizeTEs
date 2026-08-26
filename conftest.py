"""Make the checkout importable so a bare `pytest` behaves like `python -m pytest`.

Under pytest's default `prepend` import mode the repository root is not placed on
`sys.path`, so `import normalize_tes` fails unless the suite is started with
`python -m pytest`, which prepends the working directory itself. Doing it here
means both invocations resolve the package the same way.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
