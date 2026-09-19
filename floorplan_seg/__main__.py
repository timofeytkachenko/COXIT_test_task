"""Allow ``python -m floorplan_seg``."""

import sys

from .cli import main

sys.exit(main())
