from __future__ import absolute_import

from .function_library import acsc, asec, complexconjugate, cot, csc, im, re, sec

# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.1.0 for Linux x86 (64-bit) (July 16, 2024)
# Date: Tue 29 Apr 2025 17:51:04
from .object_library import Lorentz, all_lorentz

try:
    import form_factors as ForFac
except ImportError:
    pass

FFFF1 = Lorentz(
    name="FFFF1", spins=[2, 2, 2, 2], structure="Gamma(-1,2,3)*Gamma(-1,4,1)"
)

FFFF2 = Lorentz(
    name="FFFF2", spins=[2, 2, 2, 2], structure="Gamma(-1,2,1)*Gamma(-1,4,3)"
)
