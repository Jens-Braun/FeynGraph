# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.1.0 for Microsoft Windows (64-bit) (July 16, 2024)
# Date: Fri 22 Nov 2024 22:31:21


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.G, P.G, P.G ],
             color = [ 'f(1,2,3)' ],
             lorentz = [ L.VVV1 ],
             couplings = {(0,0):C.GC_1})

V_2 = Vertex(name = 'V_2',
             particles = [ P.G, P.G, P.G, P.G ],
             color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
             lorentz = [ L.VVVV1, L.VVVV2, L.VVVV3 ],
             couplings = {(1,1):C.GC_3,(0,0):C.GC_3,(2,2):C.GC_3})

V_3 = Vertex(name = 'V_3',
             particles = [ P.u__tilde__, P.u, P.G ],
             color = [ 'T(3,2,1)' ],
             lorentz = [ L.FFV1 ],
             couplings = {(0,0):C.GC_2})

V_4 = Vertex(name = 'V_4',
             particles = [ P.c__tilde__, P.c, P.G ],
             color = [ 'T(3,2,1)' ],
             lorentz = [ L.FFV1 ],
             couplings = {(0,0):C.GC_2})

V_5 = Vertex(name = 'V_5',
             particles = [ P.t__tilde__, P.t, P.G ],
             color = [ 'T(3,2,1)' ],
             lorentz = [ L.FFV1 ],
             couplings = {(0,0):C.GC_2})

