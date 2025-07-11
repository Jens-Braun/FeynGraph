from __future__ import absolute_import
# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.1.0 for Linux x86 (64-bit) (July 16, 2024)
# Date: Tue 29 Apr 2025 17:51:04


from .object_library import all_vertices, Vertex
from . import particles as P
from . import couplings as C
from . import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.G, P.G, P.G ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.VVV1 ],
             couplings = {(0,0):C.GC_4})

V_2 = Vertex(name = 'V_2',
             particles = [ P.G, P.G, P.G, P.G ],
             color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
             lorentz = [ L.VVVV1, L.VVVV2, L.VVVV3 ],
             couplings = {(1,1):C.GC_6,(0,0):C.GC_6,(2,2):C.GC_6})

V_3 = Vertex(name = 'V_3',
             particles = [ P.d__tilde__, P.d, P.d__tilde__, P.d ],
             color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
             lorentz = [ L.FFFF1, L.FFFF2 ],
             couplings = {(1,0):C.GC_1,(0,1):C.GC_1})

V_4 = Vertex(name = 'V_4',
             particles = [ P.d__tilde__, P.d, P.u__tilde__, P.u ],
             color = [ 'Identity(1,2)*Identity(3,4)' ],
             lorentz = [ L.FFFF2 ],
             couplings = {(0,0):C.GC_2})

V_5 = Vertex(name = 'V_5',
             particles = [ P.u__tilde__, P.u, P.u__tilde__, P.u ],
             color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
             lorentz = [ L.FFFF1, L.FFFF2 ],
             couplings = {(1,0):C.GC_3,(0,1):C.GC_3})

V_6 = Vertex(name = 'V_6',
             particles = [ P.d__tilde__, P.d, P.G ],
             color = [ 'T(3,2,1)' ],
             lorentz = [ L.FFV1 ],
             couplings = {(0,0):C.GC_5})

V_7 = Vertex(name = 'V_7',
             particles = [ P.u__tilde__, P.u, P.G ],
             color = [ 'T(3,2,1)' ],
             lorentz = [ L.FFV1 ],
             couplings = {(0,0):C.GC_5})

