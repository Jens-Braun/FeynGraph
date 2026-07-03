# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.1.0 for Linux x86 (64-bit) (July 16, 2024)
# Date: Tue 29 Apr 2025 17:51:04


from __future__ import absolute_import, division

from . import parameters as Param
from . import propagators as Prop
from .object_library import Particle, all_particles

u = Particle(
    pdg_code=9000001,
    name="u",
    antiname="u~",
    spin=2,
    color=3,
    mass=Param.ZERO,
    width=Param.ZERO,
    texname="u",
    antitexname="u~",
    charge=0,
)

u__tilde__ = u.anti()
