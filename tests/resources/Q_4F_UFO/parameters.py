from __future__ import absolute_import

from .function_library import acsc, asec, complexconjugate, cot, csc, im, re, sec

# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.1.0 for Linux x86 (64-bit) (July 16, 2024)
# Date: Tue 29 Apr 2025 17:51:04
from .object_library import Parameter, all_parameters

# This is a default parameter object representing 0.
ZERO = Parameter(name="ZERO", nature="internal", type="real", value="0.0", texname="0")

# User-defined parameters.

cuu = Parameter(
    name="cuu",
    nature="external",
    type="real",
    value=1,
    texname="\\text{cuu}",
    lhablock="FRBlock",
    lhacode=[2],
)
