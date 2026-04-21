from __future__ import absolute_import

from . import couplings as C
from . import lorentz as L
from . import particles as P

# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.1.0 for Linux x86 (64-bit) (July 16, 2024)
# Date: Tue 29 Apr 2025 17:51:04
from .object_library import Vertex, all_vertices

V_1 = Vertex(
    name="V_1",
    particles=[P.u__tilde__, P.u, P.u__tilde__, P.u],
    color=["Identity(1,2)*Identity(3,4)", "Identity(1,4)*Identity(2,3)"],
    lorentz=[L.FFFF1, L.FFFF2],
    couplings={(1, 0): C.GC_1, (0, 1): C.GC_1},
)
