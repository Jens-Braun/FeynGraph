# This file was automatically created by FeynRules 2.3.31
# Mathematica version: 11.2.0 for Mac OS X x86 (64-bit) (September 11, 2017)
# Date: Wed 9 May 2018 16:32:47


from .object_library import all_vertices, all_CTvertices, Vertex, CTVertex
from . import particles as P
from . import CT_couplings as C
from . import lorentz as L


V_1 = CTVertex(name = 'V_1',
               type = 'R2',
               particles = [ P.g, P.g, P.g ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVV1, L.VVV2, L.VVV4, L.VVV6, L.VVV7, L.VVV8 ],
               loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ] ],
               couplings = {(0,0,0):C.R2GC_157_40,(0,0,1):C.R2GC_157_41,(0,1,0):C.R2GC_160_42,(0,1,1):C.R2GC_160_43,(0,2,0):C.R2GC_160_42,(0,2,1):C.R2GC_160_43,(0,3,0):C.R2GC_157_40,(0,3,1):C.R2GC_157_41,(0,4,0):C.R2GC_157_40,(0,4,1):C.R2GC_157_41,(0,5,0):C.R2GC_160_42,(0,5,1):C.R2GC_160_43})

V_2 = CTVertex(name = 'V_2',
               type = 'R2',
               particles = [ P.g, P.g, P.g, P.g ],
               color = [ 'd(-1,1,3)*d(-1,2,4)', 'd(-1,1,3)*f(-1,2,4)', 'd(-1,1,4)*d(-1,2,3)', 'd(-1,1,4)*f(-1,2,3)', 'd(-1,2,3)*f(-1,1,4)', 'd(-1,2,4)*f(-1,1,3)', 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)', 'Identity(1,2)*Identity(3,4)', 'Identity(1,3)*Identity(2,4)', 'Identity(1,4)*Identity(2,3)' ],
               lorentz = [ L.VVVV2, L.VVVV3, L.VVVV4 ],
               loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ] ],
               couplings = {(2,0,0):C.R2GC_114_11,(2,0,1):C.R2GC_114_12,(0,0,0):C.R2GC_114_11,(0,0,1):C.R2GC_114_12,(6,0,0):C.R2GC_117_16,(6,0,1):C.R2GC_167_49,(4,0,0):C.R2GC_112_7,(4,0,1):C.R2GC_112_8,(3,0,0):C.R2GC_112_7,(3,0,1):C.R2GC_112_8,(8,0,0):C.R2GC_113_9,(8,0,1):C.R2GC_113_10,(7,0,0):C.R2GC_118_18,(7,0,1):C.R2GC_166_48,(5,0,0):C.R2GC_112_7,(5,0,1):C.R2GC_112_8,(1,0,0):C.R2GC_112_7,(1,0,1):C.R2GC_112_8,(11,0,0):C.R2GC_116_14,(11,0,1):C.R2GC_116_15,(10,0,0):C.R2GC_116_14,(10,0,1):C.R2GC_116_15,(9,0,1):C.R2GC_115_13,(2,1,0):C.R2GC_114_11,(2,1,1):C.R2GC_114_12,(0,1,0):C.R2GC_114_11,(0,1,1):C.R2GC_114_12,(4,1,0):C.R2GC_112_7,(4,1,1):C.R2GC_112_8,(3,1,0):C.R2GC_112_7,(3,1,1):C.R2GC_112_8,(8,1,0):C.R2GC_113_9,(8,1,1):C.R2GC_168_50,(6,1,0):C.R2GC_163_44,(6,1,1):C.R2GC_163_45,(7,1,0):C.R2GC_118_18,(7,1,1):C.R2GC_118_19,(5,1,0):C.R2GC_112_7,(5,1,1):C.R2GC_112_8,(1,1,0):C.R2GC_112_7,(1,1,1):C.R2GC_112_8,(11,1,0):C.R2GC_116_14,(11,1,1):C.R2GC_116_15,(10,1,0):C.R2GC_116_14,(10,1,1):C.R2GC_116_15,(9,1,1):C.R2GC_115_13,(2,2,0):C.R2GC_114_11,(2,2,1):C.R2GC_114_12,(0,2,0):C.R2GC_114_11,(0,2,1):C.R2GC_114_12,(4,2,0):C.R2GC_112_7,(4,2,1):C.R2GC_112_8,(3,2,0):C.R2GC_112_7,(3,2,1):C.R2GC_112_8,(8,2,0):C.R2GC_113_9,(8,2,1):C.R2GC_165_47,(6,2,0):C.R2GC_117_16,(6,2,1):C.R2GC_117_17,(7,2,0):C.R2GC_164_46,(7,2,1):C.R2GC_114_12,(5,2,0):C.R2GC_112_7,(5,2,1):C.R2GC_112_8,(1,2,0):C.R2GC_112_7,(1,2,1):C.R2GC_112_8,(11,2,0):C.R2GC_116_14,(11,2,1):C.R2GC_116_15,(10,2,0):C.R2GC_116_14,(10,2,1):C.R2GC_116_15,(9,2,1):C.R2GC_115_13})

V_3 = CTVertex(name = 'V_3',
               type = 'R2',
               particles = [ P.t__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               loop_particles = [ [ [P.b, P.g, P.t] ] ],
               couplings = {(0,0,0):C.R2GC_177_55,(0,1,0):C.R2GC_178_56})

V_4 = CTVertex(name = 'V_4',
               type = 'R2',
               particles = [ P.b__tilde__, P.b, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               loop_particles = [ [ [P.b, P.g] ] ],
               couplings = {(0,0,0):C.R2GC_154_36})

V_5 = CTVertex(name = 'V_5',
               type = 'R2',
               particles = [ P.b__tilde__, P.b, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2 ],
               loop_particles = [ [ [P.b, P.g] ] ],
               couplings = {(0,0,0):C.R2GC_153_35})

V_6 = CTVertex(name = 'V_6',
               type = 'R2',
               particles = [ P.b__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS4 ],
               loop_particles = [ [ [P.b, P.g, P.t] ] ],
               couplings = {(0,0,0):C.R2GC_179_57,(0,1,0):C.R2GC_176_54})

V_7 = CTVertex(name = 'V_7',
               type = 'R2',
               particles = [ P.t__tilde__, P.t, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               loop_particles = [ [ [P.g, P.t] ] ],
               couplings = {(0,0,0):C.R2GC_180_58})

V_8 = CTVertex(name = 'V_8',
               type = 'R2',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS2 ],
               loop_particles = [ [ [P.g, P.t] ] ],
               couplings = {(0,0,0):C.R2GC_181_59})

V_9 = CTVertex(name = 'V_9',
               type = 'R2',
               particles = [ P.u__tilde__, P.u, P.a ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               loop_particles = [ [ [P.g, P.u] ] ],
               couplings = {(0,0,0):C.R2GC_122_23})

V_10 = CTVertex(name = 'V_10',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_122_23})

V_11 = CTVertex(name = 'V_11',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_122_23})

V_12 = CTVertex(name = 'V_12',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_119_20})

V_13 = CTVertex(name = 'V_13',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_119_20})

V_14 = CTVertex(name = 'V_14',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_119_20})

V_15 = CTVertex(name = 'V_15',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_120_21})

V_16 = CTVertex(name = 'V_16',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_120_21})

V_17 = CTVertex(name = 'V_17',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_120_21})

V_18 = CTVertex(name = 'V_18',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_120_21})

V_19 = CTVertex(name = 'V_19',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_120_21})

V_20 = CTVertex(name = 'V_20',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_120_21})

V_21 = CTVertex(name = 'V_21',
                type = 'R2',
                particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.d, P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_143_28})

V_22 = CTVertex(name = 'V_22',
                type = 'R2',
                particles = [ P.s__tilde__, P.u, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.g, P.s, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_145_30})

V_23 = CTVertex(name = 'V_23',
                type = 'R2',
                particles = [ P.d__tilde__, P.c, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.c, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_139_24})

V_24 = CTVertex(name = 'V_24',
                type = 'R2',
                particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_141_26})

V_25 = CTVertex(name = 'V_25',
                type = 'R2',
                particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_173_52})

V_26 = CTVertex(name = 'V_26',
                type = 'R2',
                particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.d, P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_144_29})

V_27 = CTVertex(name = 'V_27',
                type = 'R2',
                particles = [ P.c__tilde__, P.d, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.c, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_140_25})

V_28 = CTVertex(name = 'V_28',
                type = 'R2',
                particles = [ P.u__tilde__, P.s, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.g, P.s, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_146_31})

V_29 = CTVertex(name = 'V_29',
                type = 'R2',
                particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_142_27})

V_30 = CTVertex(name = 'V_30',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_173_52})

V_31 = CTVertex(name = 'V_31',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_100_1,(0,1,0):C.R2GC_175_53})

V_32 = CTVertex(name = 'V_32',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_100_1,(0,1,0):C.R2GC_175_53})

V_33 = CTVertex(name = 'V_33',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_100_1,(0,1,0):C.R2GC_175_53})

V_34 = CTVertex(name = 'V_34',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_151_33,(0,1,0):C.R2GC_152_34})

V_35 = CTVertex(name = 'V_35',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_151_33,(0,1,0):C.R2GC_152_34})

V_36 = CTVertex(name = 'V_36',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_151_33,(0,1,0):C.R2GC_152_34})

V_37 = CTVertex(name = 'V_37',
                type = 'R2',
                particles = [ P.u__tilde__, P.u ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_121_22})

V_38 = CTVertex(name = 'V_38',
                type = 'R2',
                particles = [ P.c__tilde__, P.c ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_121_22})

V_39 = CTVertex(name = 'V_39',
                type = 'R2',
                particles = [ P.t__tilde__, P.t ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_172_51,(0,2,0):C.R2GC_172_51,(0,1,0):C.R2GC_121_22,(0,3,0):C.R2GC_121_22})

V_40 = CTVertex(name = 'V_40',
                type = 'R2',
                particles = [ P.d__tilde__, P.d ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_121_22})

V_41 = CTVertex(name = 'V_41',
                type = 'R2',
                particles = [ P.s__tilde__, P.s ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_121_22})

V_42 = CTVertex(name = 'V_42',
                type = 'R2',
                particles = [ P.b__tilde__, P.b ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_150_32,(0,2,0):C.R2GC_150_32,(0,1,0):C.R2GC_121_22,(0,3,0):C.R2GC_121_22})

V_43 = CTVertex(name = 'V_43',
                type = 'R2',
                particles = [ P.g, P.g ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VV1, L.VV2, L.VV3 ],
                loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                couplings = {(0,0,2):C.R2GC_156_39,(0,1,0):C.R2GC_87_60,(0,1,3):C.R2GC_87_61,(0,2,1):C.R2GC_155_37,(0,2,2):C.R2GC_155_38})

V_44 = CTVertex(name = 'V_44',
                type = 'R2',
                particles = [ P.g, P.g, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVS1 ],
                loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                couplings = {(0,0,0):C.R2GC_88_62,(0,0,1):C.R2GC_88_63})

V_45 = CTVertex(name = 'V_45',
                type = 'R2',
                particles = [ P.g, P.g, P.W__minus__, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVVV2, L.VVVV3, L.VVVV4 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.d] ], [ [P.c, P.s] ], [ [P.d, P.u] ], [ [P.s, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_101_2,(0,0,1):C.R2GC_101_3,(0,0,2):C.R2GC_101_4,(0,0,3):C.R2GC_101_5,(0,0,4):C.R2GC_101_6,(0,1,0):C.R2GC_101_2,(0,1,1):C.R2GC_101_3,(0,1,2):C.R2GC_101_4,(0,1,3):C.R2GC_101_5,(0,1,4):C.R2GC_101_6,(0,2,0):C.R2GC_101_2,(0,2,1):C.R2GC_101_3,(0,2,2):C.R2GC_101_4,(0,2,3):C.R2GC_101_5,(0,2,4):C.R2GC_101_6})

V_46 = CTVertex(name = 'V_46',
                type = 'R2',
                particles = [ P.a, P.g, P.g, P.Z ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.VVVV2, L.VVVV3, L.VVVV4 ],
                loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_92_70,(0,0,1):C.R2GC_92_71,(0,1,0):C.R2GC_92_70,(0,1,1):C.R2GC_92_71,(0,2,0):C.R2GC_92_70,(0,2,1):C.R2GC_92_71})

V_47 = CTVertex(name = 'V_47',
                type = 'R2',
                particles = [ P.g, P.g, P.Z, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVVV2, L.VVVV3, L.VVVV4 ],
                loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_95_76,(0,0,1):C.R2GC_95_77,(0,1,0):C.R2GC_95_76,(0,1,1):C.R2GC_95_77,(0,2,0):C.R2GC_95_76,(0,2,1):C.R2GC_95_77})

V_48 = CTVertex(name = 'V_48',
                type = 'R2',
                particles = [ P.a, P.a, P.g, P.g ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.VVVV2, L.VVVV3, L.VVVV4 ],
                loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_90_66,(0,0,1):C.R2GC_90_67,(0,1,0):C.R2GC_90_66,(0,1,1):C.R2GC_90_67,(0,2,0):C.R2GC_90_66,(0,2,1):C.R2GC_90_67})

V_49 = CTVertex(name = 'V_49',
                type = 'R2',
                particles = [ P.g, P.g, P.g, P.Z ],
                color = [ 'd(1,2,3)', 'f(1,2,3)' ],
                lorentz = [ L.VVVV1, L.VVVV2, L.VVVV3, L.VVVV4 ],
                loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                couplings = {(1,0,0):C.R2GC_94_74,(1,0,1):C.R2GC_94_75,(0,1,0):C.R2GC_93_72,(0,1,1):C.R2GC_93_73,(0,2,0):C.R2GC_93_72,(0,2,1):C.R2GC_93_73,(0,3,0):C.R2GC_93_72,(0,3,1):C.R2GC_93_73})

V_50 = CTVertex(name = 'V_50',
                type = 'R2',
                particles = [ P.a, P.g, P.g, P.g ],
                color = [ 'd(2,3,4)' ],
                lorentz = [ L.VVVV2, L.VVVV3, L.VVVV4 ],
                loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_91_68,(0,0,1):C.R2GC_91_69,(0,1,0):C.R2GC_91_68,(0,1,1):C.R2GC_91_69,(0,2,0):C.R2GC_91_68,(0,2,1):C.R2GC_91_69})

V_51 = CTVertex(name = 'V_51',
                type = 'R2',
                particles = [ P.g, P.g, P.H, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                couplings = {(0,0,0):C.R2GC_89_64,(0,0,1):C.R2GC_89_65})

V_52 = CTVertex(name = 'V_52',
                type = 'R2',
                particles = [ P.g, P.g, P.G0, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                couplings = {(0,0,0):C.R2GC_89_64,(0,0,1):C.R2GC_89_65})

V_53 = CTVertex(name = 'V_53',
                type = 'R2',
                particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_99_78})

V_54 = CTVertex(name = 'V_54',
                type = 'UV',
                particles = [ P.g, P.g, P.g ],
                color = [ 'f(1,2,3)' ],
                lorentz = [ L.VVV1, L.VVV2, L.VVV3, L.VVV4, L.VVV5, L.VVV6, L.VVV7, L.VVV8 ],
                loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.t] ] ],
                couplings = {(0,0,0):C.UVGC_157_60,(0,0,1):C.UVGC_157_61,(0,0,2):C.UVGC_157_62,(0,0,3):C.UVGC_111_7,(0,0,4):C.UVGC_157_63,(0,1,0):C.UVGC_160_68,(0,1,1):C.UVGC_160_69,(0,1,2):C.UVGC_160_70,(0,1,3):C.UVGC_160_71,(0,1,4):C.UVGC_160_72,(0,3,0):C.UVGC_160_68,(0,3,1):C.UVGC_160_69,(0,3,2):C.UVGC_162_75,(0,3,3):C.UVGC_110_5,(0,3,4):C.UVGC_160_72,(0,5,0):C.UVGC_157_60,(0,5,1):C.UVGC_157_61,(0,5,2):C.UVGC_159_66,(0,5,3):C.UVGC_159_67,(0,5,4):C.UVGC_157_63,(0,6,0):C.UVGC_157_60,(0,6,1):C.UVGC_157_61,(0,6,2):C.UVGC_158_64,(0,6,3):C.UVGC_158_65,(0,6,4):C.UVGC_157_63,(0,7,0):C.UVGC_160_68,(0,7,1):C.UVGC_160_69,(0,7,2):C.UVGC_161_73,(0,7,3):C.UVGC_161_74,(0,7,4):C.UVGC_160_72,(0,2,2):C.UVGC_110_4,(0,2,3):C.UVGC_110_5,(0,4,2):C.UVGC_111_6,(0,4,3):C.UVGC_111_7})

V_55 = CTVertex(name = 'V_55',
                type = 'UV',
                particles = [ P.g, P.g, P.g, P.g ],
                color = [ 'd(-1,1,3)*d(-1,2,4)', 'd(-1,1,3)*f(-1,2,4)', 'd(-1,1,4)*d(-1,2,3)', 'd(-1,1,4)*f(-1,2,3)', 'd(-1,2,3)*f(-1,1,4)', 'd(-1,2,4)*f(-1,1,3)', 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)', 'Identity(1,2)*Identity(3,4)', 'Identity(1,3)*Identity(2,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.VVVV2, L.VVVV3, L.VVVV4 ],
                loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.t] ] ],
                couplings = {(2,0,3):C.UVGC_113_11,(2,0,4):C.UVGC_113_10,(0,0,3):C.UVGC_113_11,(0,0,4):C.UVGC_113_10,(6,0,0):C.UVGC_166_87,(6,0,2):C.UVGC_166_88,(6,0,3):C.UVGC_167_92,(6,0,4):C.UVGC_167_93,(6,0,5):C.UVGC_166_91,(4,0,3):C.UVGC_112_8,(4,0,4):C.UVGC_112_9,(3,0,3):C.UVGC_112_8,(3,0,4):C.UVGC_112_9,(8,0,3):C.UVGC_113_10,(8,0,4):C.UVGC_113_11,(7,0,0):C.UVGC_166_87,(7,0,2):C.UVGC_166_88,(7,0,3):C.UVGC_166_89,(7,0,4):C.UVGC_166_90,(7,0,5):C.UVGC_166_91,(5,0,3):C.UVGC_112_8,(5,0,4):C.UVGC_112_9,(1,0,3):C.UVGC_112_8,(1,0,4):C.UVGC_112_9,(11,0,3):C.UVGC_116_14,(11,0,4):C.UVGC_116_15,(10,0,3):C.UVGC_116_14,(10,0,4):C.UVGC_116_15,(9,0,3):C.UVGC_115_12,(9,0,4):C.UVGC_115_13,(2,1,3):C.UVGC_113_11,(2,1,4):C.UVGC_113_10,(0,1,3):C.UVGC_113_11,(0,1,4):C.UVGC_113_10,(4,1,3):C.UVGC_112_8,(4,1,4):C.UVGC_112_9,(3,1,3):C.UVGC_112_8,(3,1,4):C.UVGC_112_9,(8,1,0):C.UVGC_168_94,(8,1,2):C.UVGC_168_95,(8,1,3):C.UVGC_168_96,(8,1,4):C.UVGC_168_97,(8,1,5):C.UVGC_168_98,(6,1,0):C.UVGC_163_76,(6,1,3):C.UVGC_163_77,(6,1,4):C.UVGC_163_78,(6,1,5):C.UVGC_163_79,(7,1,1):C.UVGC_117_16,(7,1,3):C.UVGC_118_18,(7,1,4):C.UVGC_118_19,(5,1,3):C.UVGC_112_8,(5,1,4):C.UVGC_112_9,(1,1,3):C.UVGC_112_8,(1,1,4):C.UVGC_112_9,(11,1,3):C.UVGC_116_14,(11,1,4):C.UVGC_116_15,(10,1,3):C.UVGC_116_14,(10,1,4):C.UVGC_116_15,(9,1,3):C.UVGC_115_12,(9,1,4):C.UVGC_115_13,(2,2,3):C.UVGC_113_11,(2,2,4):C.UVGC_113_10,(0,2,3):C.UVGC_113_11,(0,2,4):C.UVGC_113_10,(4,2,3):C.UVGC_112_8,(4,2,4):C.UVGC_112_9,(3,2,3):C.UVGC_112_8,(3,2,4):C.UVGC_112_9,(8,2,0):C.UVGC_165_82,(8,2,2):C.UVGC_165_83,(8,2,3):C.UVGC_165_84,(8,2,4):C.UVGC_165_85,(8,2,5):C.UVGC_165_86,(6,2,1):C.UVGC_117_16,(6,2,3):C.UVGC_117_17,(6,2,4):C.UVGC_115_12,(7,2,0):C.UVGC_163_76,(7,2,3):C.UVGC_164_80,(7,2,4):C.UVGC_164_81,(7,2,5):C.UVGC_163_79,(5,2,3):C.UVGC_112_8,(5,2,4):C.UVGC_112_9,(1,2,3):C.UVGC_112_8,(1,2,4):C.UVGC_112_9,(11,2,3):C.UVGC_116_14,(11,2,4):C.UVGC_116_15,(10,2,3):C.UVGC_116_14,(10,2,4):C.UVGC_116_15,(9,2,3):C.UVGC_115_12,(9,2,4):C.UVGC_115_13})

V_56 = CTVertex(name = 'V_56',
                type = 'UV',
                particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.UVGC_177_111,(0,0,2):C.UVGC_177_112,(0,0,1):C.UVGC_177_113,(0,1,0):C.UVGC_178_114,(0,1,2):C.UVGC_178_115,(0,1,1):C.UVGC_178_116})

V_57 = CTVertex(name = 'V_57',
                type = 'UV',
                particles = [ P.b__tilde__, P.b, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS1 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_154_53})

V_58 = CTVertex(name = 'V_58',
                type = 'UV',
                particles = [ P.b__tilde__, P.b, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS2 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_153_52})

V_59 = CTVertex(name = 'V_59',
                type = 'UV',
                particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS4 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.UVGC_179_117,(0,0,2):C.UVGC_179_118,(0,0,1):C.UVGC_179_119,(0,1,0):C.UVGC_176_108,(0,1,2):C.UVGC_176_109,(0,1,1):C.UVGC_176_110})

V_60 = CTVertex(name = 'V_60',
                type = 'UV',
                particles = [ P.t__tilde__, P.t, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS1 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.UVGC_180_120})

V_61 = CTVertex(name = 'V_61',
                type = 'UV',
                particles = [ P.t__tilde__, P.t, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS2 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.UVGC_181_121})

V_62 = CTVertex(name = 'V_62',
                type = 'UV',
                particles = [ P.u__tilde__, P.u, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.UVGC_122_23,(0,1,0):C.UVGC_103_2,(0,2,0):C.UVGC_103_2})

V_63 = CTVertex(name = 'V_63',
                type = 'UV',
                particles = [ P.c__tilde__, P.c, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_122_23,(0,1,0):C.UVGC_103_2,(0,2,0):C.UVGC_103_2})

V_64 = CTVertex(name = 'V_64',
                type = 'UV',
                particles = [ P.t__tilde__, P.t, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.UVGC_122_23,(0,1,0):C.UVGC_170_100,(0,2,0):C.UVGC_170_100})

V_65 = CTVertex(name = 'V_65',
                type = 'UV',
                particles = [ P.d__tilde__, P.d, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_119_20,(0,1,0):C.UVGC_105_3,(0,2,0):C.UVGC_105_3})

V_66 = CTVertex(name = 'V_66',
                type = 'UV',
                particles = [ P.s__tilde__, P.s, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.UVGC_119_20,(0,1,0):C.UVGC_105_3,(0,2,0):C.UVGC_105_3})

V_67 = CTVertex(name = 'V_67',
                type = 'UV',
                particles = [ P.b__tilde__, P.b, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_119_20,(0,1,0):C.UVGC_148_47,(0,2,0):C.UVGC_148_47})

V_68 = CTVertex(name = 'V_68',
                type = 'UV',
                particles = [ P.u__tilde__, P.u, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.u] ], [ [P.t] ] ],
                couplings = {(0,0,4):C.UVGC_120_21,(0,1,0):C.UVGC_123_24,(0,1,1):C.UVGC_123_25,(0,1,2):C.UVGC_123_26,(0,1,3):C.UVGC_123_27,(0,1,5):C.UVGC_123_28,(0,1,4):C.UVGC_123_29,(0,2,0):C.UVGC_123_24,(0,2,1):C.UVGC_123_25,(0,2,2):C.UVGC_123_26,(0,2,3):C.UVGC_123_27,(0,2,5):C.UVGC_123_28,(0,2,4):C.UVGC_123_29})

V_69 = CTVertex(name = 'V_69',
                type = 'UV',
                particles = [ P.c__tilde__, P.c, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.g] ], [ [P.g] ], [ [P.ghG] ], [ [P.t] ] ],
                couplings = {(0,0,2):C.UVGC_120_21,(0,1,0):C.UVGC_123_24,(0,1,1):C.UVGC_123_25,(0,1,3):C.UVGC_123_26,(0,1,4):C.UVGC_123_27,(0,1,5):C.UVGC_123_28,(0,1,2):C.UVGC_123_29,(0,2,0):C.UVGC_123_24,(0,2,1):C.UVGC_123_25,(0,2,3):C.UVGC_123_26,(0,2,4):C.UVGC_123_27,(0,2,5):C.UVGC_123_28,(0,2,2):C.UVGC_123_29})

V_70 = CTVertex(name = 'V_70',
                type = 'UV',
                particles = [ P.t__tilde__, P.t, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.t] ], [ [P.t] ] ],
                couplings = {(0,0,4):C.UVGC_120_21,(0,1,0):C.UVGC_123_24,(0,1,1):C.UVGC_123_25,(0,1,2):C.UVGC_123_26,(0,1,3):C.UVGC_123_27,(0,1,5):C.UVGC_123_28,(0,1,4):C.UVGC_171_101,(0,2,0):C.UVGC_123_24,(0,2,1):C.UVGC_123_25,(0,2,2):C.UVGC_123_26,(0,2,3):C.UVGC_123_27,(0,2,5):C.UVGC_123_28,(0,2,4):C.UVGC_171_101})

V_71 = CTVertex(name = 'V_71',
                type = 'UV',
                particles = [ P.d__tilde__, P.d, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.g] ], [ [P.g] ], [ [P.ghG] ], [ [P.t] ] ],
                couplings = {(0,0,2):C.UVGC_120_21,(0,1,0):C.UVGC_123_24,(0,1,1):C.UVGC_123_25,(0,1,3):C.UVGC_123_26,(0,1,4):C.UVGC_123_27,(0,1,5):C.UVGC_123_28,(0,1,2):C.UVGC_123_29,(0,2,0):C.UVGC_123_24,(0,2,1):C.UVGC_123_25,(0,2,3):C.UVGC_123_26,(0,2,4):C.UVGC_123_27,(0,2,5):C.UVGC_123_28,(0,2,2):C.UVGC_123_29})

V_72 = CTVertex(name = 'V_72',
                type = 'UV',
                particles = [ P.s__tilde__, P.s, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.b] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.s] ], [ [P.t] ] ],
                couplings = {(0,0,4):C.UVGC_120_21,(0,1,0):C.UVGC_123_24,(0,1,1):C.UVGC_123_25,(0,1,2):C.UVGC_123_26,(0,1,3):C.UVGC_123_27,(0,1,5):C.UVGC_123_28,(0,1,4):C.UVGC_123_29,(0,2,0):C.UVGC_123_24,(0,2,1):C.UVGC_123_25,(0,2,2):C.UVGC_123_26,(0,2,3):C.UVGC_123_27,(0,2,5):C.UVGC_123_28,(0,2,4):C.UVGC_123_29})

V_73 = CTVertex(name = 'V_73',
                type = 'UV',
                particles = [ P.b__tilde__, P.b, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1, L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.b] ], [ [P.b, P.g] ], [ [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.t] ] ],
                couplings = {(0,0,1):C.UVGC_120_21,(0,1,0):C.UVGC_123_24,(0,1,2):C.UVGC_123_25,(0,1,3):C.UVGC_123_26,(0,1,4):C.UVGC_123_27,(0,1,5):C.UVGC_123_28,(0,1,1):C.UVGC_149_48,(0,2,0):C.UVGC_123_24,(0,2,2):C.UVGC_123_25,(0,2,3):C.UVGC_123_26,(0,2,4):C.UVGC_123_27,(0,2,5):C.UVGC_123_28,(0,2,1):C.UVGC_149_48})

V_74 = CTVertex(name = 'V_74',
                type = 'UV',
                particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                couplings = {(0,0,0):C.UVGC_143_38,(0,0,1):C.UVGC_143_39})

V_75 = CTVertex(name = 'V_75',
                type = 'UV',
                particles = [ P.s__tilde__, P.u, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.g, P.s], [P.g, P.u] ], [ [P.g, P.s, P.u] ] ],
                couplings = {(0,0,0):C.UVGC_145_42,(0,0,1):C.UVGC_145_43})

V_76 = CTVertex(name = 'V_76',
                type = 'UV',
                particles = [ P.d__tilde__, P.c, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.c, P.d, P.g] ], [ [P.c, P.g], [P.d, P.g] ] ],
                couplings = {(0,0,1):C.UVGC_139_30,(0,0,0):C.UVGC_139_31})

V_77 = CTVertex(name = 'V_77',
                type = 'UV',
                particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.UVGC_141_34,(0,0,1):C.UVGC_141_35})

V_78 = CTVertex(name = 'V_78',
                type = 'UV',
                particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.UVGC_173_103,(0,0,2):C.UVGC_173_104,(0,0,1):C.UVGC_173_105})

V_79 = CTVertex(name = 'V_79',
                type = 'UV',
                particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                couplings = {(0,0,0):C.UVGC_144_40,(0,0,1):C.UVGC_144_41})

V_80 = CTVertex(name = 'V_80',
                type = 'UV',
                particles = [ P.c__tilde__, P.d, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.c, P.d, P.g] ], [ [P.c, P.g], [P.d, P.g] ] ],
                couplings = {(0,0,1):C.UVGC_140_32,(0,0,0):C.UVGC_140_33})

V_81 = CTVertex(name = 'V_81',
                type = 'UV',
                particles = [ P.u__tilde__, P.s, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.g, P.s], [P.g, P.u] ], [ [P.g, P.s, P.u] ] ],
                couplings = {(0,0,0):C.UVGC_146_44,(0,0,1):C.UVGC_146_45})

V_82 = CTVertex(name = 'V_82',
                type = 'UV',
                particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.UVGC_142_36,(0,0,1):C.UVGC_142_37})

V_83 = CTVertex(name = 'V_83',
                type = 'UV',
                particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.UVGC_173_103,(0,0,2):C.UVGC_173_104,(0,0,1):C.UVGC_173_105})

V_84 = CTVertex(name = 'V_84',
                type = 'UV',
                particles = [ P.t__tilde__, P.t, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.UVGC_174_106,(0,1,0):C.UVGC_175_107})

V_85 = CTVertex(name = 'V_85',
                type = 'UV',
                particles = [ P.b__tilde__, P.b, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV2, L.FFV3 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_151_50,(0,1,0):C.UVGC_152_51})

V_86 = CTVertex(name = 'V_86',
                type = 'UV',
                particles = [ P.u__tilde__, P.u ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1, L.FF3, L.FF5 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.UVGC_121_22,(0,1,0):C.UVGC_102_1,(0,2,0):C.UVGC_102_1})

V_87 = CTVertex(name = 'V_87',
                type = 'UV',
                particles = [ P.c__tilde__, P.c ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1, L.FF3, L.FF5 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_121_22,(0,1,0):C.UVGC_102_1,(0,2,0):C.UVGC_102_1})

V_88 = CTVertex(name = 'V_88',
                type = 'UV',
                particles = [ P.t__tilde__, P.t ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.UVGC_172_102,(0,2,0):C.UVGC_172_102,(0,1,0):C.UVGC_169_99,(0,3,0):C.UVGC_169_99})

V_89 = CTVertex(name = 'V_89',
                type = 'UV',
                particles = [ P.d__tilde__, P.d ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1, L.FF3, L.FF5 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_121_22,(0,1,0):C.UVGC_102_1,(0,2,0):C.UVGC_102_1})

V_90 = CTVertex(name = 'V_90',
                type = 'UV',
                particles = [ P.s__tilde__, P.s ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1, L.FF3, L.FF5 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.UVGC_121_22,(0,1,0):C.UVGC_102_1,(0,2,0):C.UVGC_102_1})

V_91 = CTVertex(name = 'V_91',
                type = 'UV',
                particles = [ P.b__tilde__, P.b ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_150_49,(0,2,0):C.UVGC_150_49,(0,1,0):C.UVGC_147_46,(0,3,0):C.UVGC_147_46})

V_92 = CTVertex(name = 'V_92',
                type = 'UV',
                particles = [ P.g, P.g ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VV1, L.VV3 ],
                loop_particles = [ [ [P.b] ], [ [P.g] ], [ [P.ghG] ], [ [P.t] ] ],
                couplings = {(0,0,0):C.UVGC_156_56,(0,0,1):C.UVGC_156_57,(0,0,2):C.UVGC_156_58,(0,0,3):C.UVGC_156_59,(0,1,0):C.UVGC_155_54,(0,1,3):C.UVGC_155_55})

