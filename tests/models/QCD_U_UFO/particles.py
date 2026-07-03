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
             charge = 0)

u__tilde__ = u.anti()

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

