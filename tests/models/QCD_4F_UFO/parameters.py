from __future__ import absolute_import
# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.1.0 for Linux x86 (64-bit) (July 16, 2024)
# Date: Tue 29 Apr 2025 17:51:04



from .object_library import all_parameters, Parameter


from .function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
gs = Parameter(name = 'gs',
               nature = 'external',
               type = 'real',
               value = 1.2,
               texname = 'g_s',
               lhablock = 'FRBlock',
               lhacode = [ 1 ])

cuu = Parameter(name = 'cuu',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\text{cuu}',
                lhablock = 'FRBlock',
                lhacode = [ 2 ])

cdd = Parameter(name = 'cdd',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\text{cdd}',
                lhablock = 'FRBlock',
                lhacode = [ 3 ])

cud1 = Parameter(name = 'cud1',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{cud1}',
                 lhablock = 'FRBlock',
                 lhacode = [ 4 ])

cud8 = Parameter(name = 'cud8',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{cud8}',
                 lhablock = 'FRBlock',
                 lhacode = [ 5 ])

