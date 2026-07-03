# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 13.0.1 for Linux x86 (64-bit) (January 29, 2022)
# Date: Mon 8 Jan 2024 12:54:06



from object_library import all_parameters, Parameter, all_CTparameters, CTParameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

#mueft = Parameter(name = 'mueft',
#                  nature = 'external',
#                  type = 'real',
#                  value = 100.,
#                  texname = '\\mu_\\text{EFT}',
#                  lhablock = 'FRBlock',
#                  lhacode = [ 700 ])

CphiG_CT = CTParameter(name = 'CphiG_CT',
                  type = 'real',
                  value = {
                     -1: 'aS/2/cmath.pi*(-7/2*CphiG)',
                     0: 'dred*aS/2/cmath.pi*(CphiG/2)'
                  },
                  texname = '\\delta\\text{CphiG\\_CT}')

Cbphi_CT = CTParameter(name = 'Cbphi_CT',
                     type = 'real',
                     value = {
                     -1:'aS/2/cmath.pi*(-2*Cbphi+8*yb*CphiG)',
                     0:'dred*aS/2/cmath.pi*(-2/3*Cbphi+8/3*yb*CphiG)'
                     },
                     texname = '\\delta\\text{Cbphi\\_CT}')
