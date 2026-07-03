from __future__ import absolute_import
# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.1.0 for Linux x86 (64-bit) (July 16, 2024)
# Date: Tue 29 Apr 2025 17:51:04


from .object_library import all_couplings, Coupling

from .function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot



GC_1 = Coupling(name = 'GC_1',
                value = '2*cdd*complex(0,1)',
                order = {'NP':1,'QCD':2})

GC_2 = Coupling(name = 'GC_2',
                value = 'cud1*complex(0,1)',
                order = {'NP':1,'QCD':2})

GC_3 = Coupling(name = 'GC_3',
                value = '2*cuu*complex(0,1)',
                order = {'NP':1,'QCD':2})

GC_4 = Coupling(name = 'GC_4',
                value = '-gs',
                order = {'QCD':1})

GC_5 = Coupling(name = 'GC_5',
                value = '-(complex(0,1)*gs)',
                order = {'QCD':1})

GC_6 = Coupling(name = 'GC_6',
                value = 'complex(0,1)*gs**2',
                order = {'QCD':2})

