#
#
#


from .object_library import all_vertices, all_CTvertices, Vertex, CTVertex
from . import particles as P
from . import CT_couplings as C
from . import lorentz as L

# strong counpling renormalisation

CTV_36 = CTVertex(name = 'CTV_36',
                  type = 'UV',
                  particles = [ P.g, P.g, P.g ],
                  color = [ 'f(1,2,3)' ],
                  lorentz = [ L.VVV1 ],
                  loop_particles = [[[P.d],[P.u],[P.s],[P.c],[P.b],[P.t],[P.g]]],
                  couplings = {(0,0,0):C.UVGC_10})

CTV_37 = CTVertex(name = 'CTV_37',
                  type = 'UV',
                  particles = [ P.g, P.g, P.g, P.g ],
                  color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
                  lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
                  loop_particles = [[[P.d],[P.u],[P.s],[P.c],[P.b],[P.t],[P.g]]],
                  couplings = {(1,1,0):C.UVGC_12,(0,0,0):C.UVGC_12,(2,2,0):C.UVGC_12})

CTV_107 = CTVertex(name = 'CTV_107',
               type = 'UV',
               particles = [ P.u__tilde__, P.u, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles = [[[P.d],[P.u],[P.s],[P.c],[P.b],[P.t],[P.g]]],
               couplings = {(0,0,0):C.UVGC_11})

CTV_108 = CTVertex(name = 'CTV_108',
               type = 'UV',
               particles = [ P.c__tilde__, P.c, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles = [[[P.d],[P.u],[P.s],[P.c],[P.b],[P.t],[P.g]]],
               couplings = {(0,0,0):C.UVGC_11})

CTV_109 = CTVertex(name = 'CTV_109',
               type = 'UV',
               particles = [ P.t__tilde__, P.t, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles = [[[P.d],[P.u],[P.s],[P.c],[P.b],[P.t],[P.g]]],
               couplings = {(0,0,0):C.UVGC_11})

CTV_125 = CTVertex(name = 'CTV_125',
               type = 'UV',
               particles = [ P.d__tilde__, P.d, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles = [[[P.d],[P.u],[P.s],[P.c],[P.b],[P.t],[P.g]]],
               couplings = {(0,0,0):C.UVGC_11})

CTV_126 = CTVertex(name = 'CTV_126',
                   type = 'UV',
               particles = [ P.s__tilde__, P.s, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles = [[[P.d],[P.u],[P.s],[P.c],[P.b],[P.t],[P.g]]],
               couplings = {(0,0,0):C.UVGC_11})

CTV_127 = CTVertex(name = 'CTV_127',
                   type = 'UV',
               particles = [ P.b__tilde__, P.b, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               loop_particles = [[[P.d],[P.u],[P.s],[P.c],[P.b],[P.t],[P.g]]],
               couplings = {(0,0,0):C.UVGC_11})

# yukawa renormalisation

CTV_43 = CTVertex(name = 'CTV_43',
                  type = 'UV',
              particles = [ P.b__tilde__, P.b, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
                  loop_particles = [[[P.g]]],
              couplings = {(0,0,0):C.UVGC_76})

CTV_44 = CTVertex(name = 'CTV_44',
                  type = 'UV',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4 ],
                  loop_particles = [[[P.g]]],
              couplings = {(0,0,0):C.UVGC_77})

CTV_48 = CTVertex(name = 'CTV_48',
                  type = 'UV',
              particles = [ P.t__tilde__, P.t, P.G0 ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS2 ],
                  loop_particles = [[[P.g]]],
              couplings = {(0,0,0):C.UVGC_79})

CTV_49 = CTVertex(name = 'CTV_49',
                  type = 'UV',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS4 ],
                  loop_particles = [[[P.g]]],
              couplings = {(0,0,0):C.UVGC_78})

# quark mass renormalisation

CTV_MB = CTVertex(name = 'CTV_MB',
                  type = 'UV',
              particles = [ P.b__tilde__, P.b ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.MASSCT ],
              loop_particles = [[[P.g],[P.b]]],
              couplings = {(0,0,0):C.UVGC_MB})

CTV_MT = CTVertex(name = 'CTV_MT',
                  type = 'UV',
              particles = [ P.t__tilde__, P.t ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.MASSCT ],
              loop_particles = [[[P.g],[P.b]]],
              couplings = {(0,0,0):C.UVGC_MT})

# finite renormalisation for axial-vector current

CTV_110 = CTVertex(name = 'CTV_110',
                 type = 'UV',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_35})

CTV_111 = CTVertex(name = 'CTV_111',
               type = 'UV',
               particles = [ P.s__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_36})

CTV_112 = CTVertex(name = 'CTV_112',
               type = 'UV',
               particles = [ P.b__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_37})

CTV_113 = CTVertex(name = 'CTV_113',
               type = 'UV',
               particles = [ P.d__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_38})

CTV_114 = CTVertex(name = 'CTV_114',
               type = 'UV',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_39})

CTV_115 = CTVertex(name = 'CTV_115',
               type = 'UV',
               particles = [ P.b__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_40})

CTV_116 = CTVertex(name = 'CTV_116',
               type = 'UV',
               particles = [ P.d__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_41})

CTV_117 = CTVertex(name = 'CTV_117',
               type = 'UV',
               particles = [ P.s__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_42})

CTV_118 = CTVertex(name = 'CTV_118',
               type = 'UV',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_43})

CTV_119 = CTVertex(name = 'CTV_119',
               type = 'UV',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_Zu})

CTV_120 = CTVertex(name = 'CTV_120',
               type = 'UV',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_Zu})

CTV_121 = CTVertex(name = 'CTV_121',
               type = 'UV',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_Zu})

CTV_128 = CTVertex(name = 'CTV_128',
               type = 'UV',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_84})

CTV_129 = CTVertex(name = 'CTV_129',
               type = 'UV',
               particles = [ P.c__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_87})

CTV_130 = CTVertex(name = 'CTV_130',
               type = 'UV',
               particles = [ P.t__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_90})

CTV_131 = CTVertex(name = 'CTV_131',
               type = 'UV',
               particles = [ P.u__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_85})

CTV_132 = CTVertex(name = 'CTV_132',
               type = 'UV',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_88})

CTV_133 = CTVertex(name = 'CTV_133',
               type = 'UV',
               particles = [ P.t__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_91})

CTV_134 = CTVertex(name = 'CTV_134',
               type = 'UV',
               particles = [ P.u__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_86})

CTV_135 = CTVertex(name = 'CTV_135',
               type = 'UV',
               particles = [ P.c__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_89})

CTV_136 = CTVertex(name = 'CTV_136',
               type = 'UV',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_92})

CTV_137 = CTVertex(name = 'CTV_137',
               type = 'UV',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_Zd})

CTV_138 = CTVertex(name = 'CTV_138',
               type = 'UV',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_Zd})

CTV_139 = CTVertex(name = 'CTV_139',
               type = 'UV',
               particles = [ P.b__tilde__, P.b, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFVaxialCT ],
               loop_particles = [[[P.g]]],
               couplings = {(0,0,0):C.UVaxialGC_Zd})
