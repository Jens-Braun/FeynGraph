# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.1.0 for Microsoft Windows (64-bit) (July 16, 2024)
# Date: Fri 22 Nov 2024 22:31:21


from __future__ import division
from object_library import all_particles, Particle
import parameters as Param

import propagators as Prop

u = Particle(pdg_code = 9000001,
             name = 'u',
             antiname = 'u~',
             spin = 2,
             color = 3,
             mass = Param.MU,
             width = Param.ZERO,
             texname = 'u',
             antitexname = 'u~',
             charge = 2/3)

u__tilde__ = u.anti()

c = Particle(pdg_code = 9000002,
             name = 'c',
             antiname = 'c~',
             spin = 2,
             color = 3,
             mass = Param.MC,
             width = Param.WC,
             texname = 'c',
             antitexname = 'c~',
             charge = 2/3)

c__tilde__ = c.anti()

t = Particle(pdg_code = 9000003,
             name = 't',
             antiname = 't~',
             spin = 2,
             color = 3,
             mass = Param.MT,
             width = Param.WT,
             texname = 't',
             antitexname = 't~',
             charge = 2/3)

t__tilde__ = t.anti()

G = Particle(pdg_code = 9000004,
             name = 'G',
             antiname = 'G',
             spin = 3,
             color = 8,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'G',
             antitexname = 'G',
             charge = 0)

