from pathlib import Path
from .feyngraph import _WOLFRAM_ENABLED
import os

def import_wolfram() -> str:
    """Return the Wolfram command to import FeynGraph"""
    if _WOLFRAM_ENABLED:
        prefix = Path(__file__).parent.parent.parent.parent.parent.absolute()
        return f"Get[\"{os.path.join(prefix, 'share', 'FeynGraph', 'feyngraph.m')}\"]"
    else:
        return "Error: FeynGraph was built without Wolfram support. Please rebuild with '-F wolfram-bindings'."
