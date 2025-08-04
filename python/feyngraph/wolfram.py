from pathlib import Path
from .feyngraph import _WOLFRAM_ENABLED

def import_wolfram() -> str:
    """Return the Wolfram command to import FeynGraph"""
    if _WOLFRAM_ENABLED:
        return f"Get[\"{Path(__file__).parent.absolute()}/feyngraph.m\"]"
    else:
        return "Error: FeynGraph was built without Wolfram support. Please rebuild with '-F wolfram-bindings'."
