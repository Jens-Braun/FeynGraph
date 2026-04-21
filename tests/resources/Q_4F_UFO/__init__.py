from __future__ import absolute_import

from . import particles
from . import couplings
from . import lorentz
from . import parameters
from . import vertices
from . import coupling_orders
#import write_param_card
from . import propagators
from . import object_library
from . import function_library


all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_orders = coupling_orders.all_orders
all_functions = function_library.all_functions
all_propagators = propagators.all_propagators

try:
   from . import decays
except ImportError:
   pass
else:
   all_decays = decays.all_decays

try:
   import form_factors
except ImportError:
   pass
else:
   all_form_factors = form_factors.all_form_factors

try:
   import CT_vertices
except ImportError:
   pass
else:
   all_CTvertices = CT_vertices.all_CTvertices


gauge = [0]


__author__ = "J. Braun"
__date__ = "29.04.2025"
