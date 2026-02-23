"""Test configuration - add source directories to sys.path."""

import sys
from pathlib import Path

# Add experiment source directories to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "adversarial"))
sys.path.insert(0, str(ROOT / "experiments"))
