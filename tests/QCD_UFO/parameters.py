# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.1.0 for Microsoft Windows (64-bit) (July 16, 2024)
# Date: Fri 22 Nov 2024 22:31:21



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

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

MU = Parameter(name = 'MU',
               nature = 'external',
               type = 'real',
               value = 1.2,
               texname = 'M',
               lhablock = 'MASS',
               lhacode = [ 9000001 ])

MC = Parameter(name = 'MC',
               nature = 'external',
               type = 'real',
               value = 5.7,
               texname = '\\text{MC}',
               lhablock = 'MASS',
               lhacode = [ 9000002 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 20.4,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 9000003 ])

WC = Parameter(name = 'WC',
               nature = 'external',
               type = 'real',
               value = 8,
               texname = '\\text{WC}',
               lhablock = 'DECAY',
               lhacode = [ 9000002 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 120,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 9000003 ])

