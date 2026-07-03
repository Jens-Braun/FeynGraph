# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 13.0.1 for Linux x86 (64-bit) (January 29, 2022)
# Date: Mon 8 Jan 2024 12:54:06


from object_library import all_vertices, all_CTvertices, Vertex, CTVertex
import particles as P
import CT_couplings as C
import lorentz as L

CTV_25 = CTVertex(name = 'CTV_25',
	type = 'UV',
        particles = [ P.g, P.g, P.G0, P.G0 ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.VVSS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_150})

CTV_26 = CTVertex(name = 'CTV_26',
	type = 'UV',
        particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.VVSS2 ],
        loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_150})

CTV_27 = CTVertex(name = 'CTV_27',
	type = 'UV',
        particles = [ P.g, P.g, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.VVS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_187})

CTV_28 = CTVertex(name = 'CTV_28',
	type = 'UV',
        particles = [ P.g, P.g, P.H, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.VVSS2 ],
        loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_150})

CTV_30 = CTVertex(name = 'CTV_30',
	type = 'UV',
        particles = [ P.g, P.g, P.g, P.G0, P.G0 ],
	color = [ 'f(1,2,3)' ],
	lorentz = [ L.VVVSS1 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_244})

CTV_31 = CTVertex(name = 'CTV_31',
	type = 'UV',
        particles = [ P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
	color = [ 'f(1,2,3)' ],
	lorentz = [ L.VVVSS1 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_244})

CTV_32 = CTVertex(name = 'CTV_32',
	type = 'UV',
        particles = [ P.g, P.g, P.g, P.H ],
	color = [ 'f(1,2,3)' ],
	lorentz = [ L.VVVS1 ],
        loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_245})

CTV_33 = CTVertex(name = 'CTV_33',
	type = 'UV',
        particles = [ P.g, P.g, P.g, P.H, P.H ],
	color = [ 'f(1,2,3)' ],
	lorentz = [ L.VVVSS1 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_244})

CTV_35 = CTVertex(name = 'CTV_35',
	type = 'UV',
        particles = [ P.g, P.g, P.g, P.g, P.G0, P.G0 ],
	color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
	lorentz = [ L.VVVVSS1, L.VVVVSS2, L.VVVVSS3 ],
        loop_particles = [ [ P.g ] ],
        couplings = {(1,1,0):C.UVGC_247, (0,0,0):C.UVGC_247, (2,2,0):C.UVGC_247})

CTV_36 = CTVertex(name = 'CTV_36',
	type = 'UV',
        particles = [ P.g, P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
	color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
	lorentz = [ L.VVVVSS1, L.VVVVSS2, L.VVVVSS3 ],
        loop_particles = [ [ P.g ] ],
        couplings = {(1,1,0):C.UVGC_247, (0,0,0):C.UVGC_247, (2,2,0):C.UVGC_247})

CTV_37 = CTVertex(name = 'CTV_37',
	type = 'UV',
        particles = [ P.g, P.g, P.g, P.g, P.H ],
	color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
	lorentz = [ L.VVVVS1, L.VVVVS2, L.VVVVS3 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(1,1,0):C.UVGC_248, (0,0,0):C.UVGC_248, (2,2,0):C.UVGC_248})

CTV_38 = CTVertex(name = 'CTV_38',
	type = 'UV',
        particles = [ P.g, P.g, P.g, P.g, P.H, P.H ],
	color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
	lorentz = [ L.VVVVSS1, L.VVVVSS2, L.VVVVSS3 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(1,1,0):C.UVGC_247, (0,0,0):C.UVGC_247, (2,2,0):C.UVGC_247})

CTV_48 = CTVertex(name = 'CTV_48',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.G0, P.G0, P.G0 ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_377, (0,1,0):C.UVGC_146})

CTV_57 = CTVertex(name = 'CTV_57',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.G0, P.G0 ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSS1, L.FFSS2 ],
        loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_379, (0,1,0):C.UVGC_185})

CTV_66 = CTVertex(name = 'CTV_66',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.G0, P.G__minus__, P.G__plus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_376, (0,1,0):C.UVGC_147})

CTV_75 = CTVertex(name = 'CTV_75',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.G__minus__, P.G__plus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSS1, L.FFSS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_379, (0,1,0):C.UVGC_185})

CTV_84 = CTVertex(name = 'CTV_84',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.G0, P.G0, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_374, (0,1,0):C.UVGC_148})

CTV_102 = CTVertex(name = 'CTV_102',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFS3, L.FFS4 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_378, (0,1,0):C.UVGC_159})

CTV_111 = CTVertex(name = 'CTV_111',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.G0, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSS1, L.FFSS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_381, (0,1,0):C.UVGC_184})

CTV_120 = CTVertex(name = 'CTV_120',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.G__minus__, P.G__plus__, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_374, (0,1,0):C.UVGC_148})

CTV_129 = CTVertex(name = 'CTV_129',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.G0, P.H, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_376, (0,1,0):C.UVGC_147})

CTV_138 = CTVertex(name = 'CTV_138',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.H, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSS1, L.FFSS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_380, (0,1,0):C.UVGC_186})

CTV_147 = CTVertex(name = 'CTV_147',
	type = 'UV',
        particles = [ P.b__tilde__, P.b, P.H, P.H, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	loop_particles = [ [ P.g ] ],
        couplings = {(0,0,0):C.UVGC_375, (0,1,0):C.UVGC_149})

