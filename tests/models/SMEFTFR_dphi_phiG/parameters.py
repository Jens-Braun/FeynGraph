# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 13.3.1 for Linux x86 (64-bit) (July 24, 2023)
# Date: Wed 7 May 2025 13:35:26



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
mueft = Parameter(name = 'mueft',
                   nature = 'external',
                   type = 'real',
                   value = 100.,
                   texname = '\\mu_\\text{EFT}',
                   lhablock = 'FRBlock',
                   lhacode = [ 700 ])

dred = Parameter(name = 'dred',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = 'DRED',
                   lhablock = 'FRBlock',
                   lhacode = [ 701 ])

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.000011638081097590524,
               texname = 'G_F',
               lhablock = 'FRBlock',
               lhacode = [ 2 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 0.00754852887435655,
                  texname = '\\alpha _{\\text{em}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 31 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1176,
               texname = '\\alpha _s',
               lhablock = 'FRBlock',
               lhacode = [ 34 ])

MG0 = Parameter(name = 'MG0',
                nature = 'external',
                type = 'real',
                value = 91.1876,
                texname = 'M_{\\text{G0}}',
                lhablock = 'FRBlock',
                lhacode = [ 52 ])

MGP = Parameter(name = 'MGP',
                nature = 'external',
                type = 'real',
                value = 80.379,
                texname = 'M_{\\text{GP}}',
                lhablock = 'FRBlock',
                lhacode = [ 53 ])

MgZ = Parameter(name = 'MgZ',
                nature = 'external',
                type = 'real',
                value = 91.1876,
                texname = 'M_{\\text{etaZ}}',
                lhablock = 'FRBlock',
                lhacode = [ 54 ])

MgW = Parameter(name = 'MgW',
                nature = 'external',
                type = 'real',
                value = 80.379,
                texname = 'M_{\\text{etaW}}',
                lhablock = 'FRBlock',
                lhacode = [ 55 ])

xiW = Parameter(name = 'xiW',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\xi _W',
                lhablock = 'FRBlock',
                lhacode = [ 56 ])

xiZ = Parameter(name = 'xiZ',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\xi _Z',
                lhablock = 'FRBlock',
                lhacode = [ 57 ])

CphiG = Parameter(name = 'CphiG',
                  nature = 'external',
                  type = 'real',
                  value = 6.85863e-9,
                  texname = 'C^{\\text{$\\phi $G}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 72 ])

Lam = Parameter(name = 'Lam',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\frac{1}{\\Lambda ^2}',
                lhablock = 'FRBlock',
                lhacode = [ 73 ])

ymtau = Parameter(name = 'ymtau',
                   nature = 'external',
                   type = 'real',
                   value = 1.77686,
                   texname = '\\text{ymtau}',
                   lhablock = 'FRBlock3',
                   lhacode = [ 3, 3 ])

ymb = Parameter(name = 'ymb',
                   nature = 'external',
                   type = 'real',
                   value = 4.7,
                   texname = '\\text{ymb}',
                   lhablock = 'FRBlock4',
                   lhacode = [ 3, 3 ])

ymt = Parameter(name = 'ymt',
                   nature = 'external',
                   type = 'real',
                   value = 172.76,
                   texname = '\\text{ymt}',
                   lhablock = 'FRBlock5',
                   lhacode = [ 3, 3 ])

Cbphi = Parameter(name = 'Cbphi',
                     nature = 'external',
                     type = 'real',
                     value = 5.48497e-9,
                     texname = '\\text{Cbphi}',
                     lhablock = 'FRBlock9',
                     lhacode = [ 3, 3 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MW = Parameter(name = 'MW',
               nature = 'external',
               type = 'real',
               value = 80.379,
               texname = '\\text{MW}',
               lhablock = 'MASS',
               lhacode = [ 24 ])

MLT = Parameter(name = 'MLT',
                nature = 'external',
                type = 'real',
                value = 1.77686,
                texname = '\\text{MLT}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

MQT = Parameter(name = 'MQT',
                nature = 'external',
                type = 'real',
                value = 172.76,
                texname = '\\text{MQT}',
                lhablock = 'MASS',
                lhacode = [ 6 ])

MQB = Parameter(name = 'MQB',
                nature = 'external',
                type = 'real',
                value = 4.7,
                texname = '\\text{MQB}',
                lhablock = 'MASS',
                lhacode = [ 5 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 125.35,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])

GAMZ = Parameter(name = 'GAMZ',
                 nature = 'external',
                 type = 'real',
                 value = 2.4952,
                 texname = '\\text{GAMZ}',
                 lhablock = 'DECAY',
                 lhacode = [ 23 ])

GAMW = Parameter(name = 'GAMW',
                 nature = 'external',
                 type = 'real',
                 value = 2.085,
                 texname = '\\text{GAMW}',
                 lhablock = 'DECAY',
                 lhacode = [ 24 ])

WLT = Parameter(name = 'WLT',
                nature = 'external',
                type = 'real',
                value = 2.25e-12,
                texname = '\\text{WLT}',
                lhablock = 'DECAY',
                lhacode = [ 15 ])

WQT = Parameter(name = 'WQT',
                nature = 'external',
                type = 'real',
                value = 1.35,
                texname = '\\text{WQT}',
                lhablock = 'DECAY',
                lhacode = [ 6 ])

GAMH = Parameter(name = 'GAMH',
                 nature = 'external',
                 type = 'real',
                 value = 0.00575,
                 texname = '\\text{GAMH}',
                 lhablock = 'DECAY',
                 lhacode = [ 25 ])

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aEWM1)*cmath.sqrt(cmath.pi)',
               texname = 'q_e')

GS = Parameter(name = 'GS',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
               texname = 'G_s')

I2a33 = Parameter(name = 'I2a33',
                  nature = 'internal',
                  type = 'real',
                  value = '0+ ymb',
                  texname = '\\text{I2a33}')

I3a33 = Parameter(name = 'I3a33',
                  nature = 'internal',
                  type = 'real',
                  value = '0+ ymt',
                  texname = '\\text{I3a33}')

I4a33 = Parameter(name = 'I4a33',
                  nature = 'internal',
                  type = 'real',
                  value = '0+ ymt',
                  texname = '\\text{I4a33}')

I5a33 = Parameter(name = 'I5a33',
                  nature = 'internal',
                  type = 'real',
                  value = '0+ ymb',
                  texname = '\\text{I5a33}')

I6a33 = Parameter(name = 'I6a33',
                  nature = 'internal',
                  type = 'real',
                  value = '0+ ymtau',
                  texname = '\\text{I6a33}')

I8a33 = Parameter(name = 'I8a33',
                  nature = 'internal',
                  type = 'real',
                  value = '0+ ymtau',
                  texname = '\\text{I8a33}')

ytau = Parameter(name = 'ytau',
                   nature = 'internal',
                   type = 'real',
                   value = 'cmath.sqrt(cmath.sqrt(8)*Gf)*ymtau',
                   texname = '\\text{ytau}')

yb = Parameter(name = 'yb',
                   nature = 'internal',
                   type = 'real',
                   value = 'cmath.sqrt(cmath.sqrt(8)*Gf)*ymb',
                   texname = '\\text{yb}')

yt = Parameter(name = 'yt',
                   nature = 'internal',
                   type = 'real',
                   value = 'cmath.sqrt(cmath.sqrt(8)*Gf)*ymt',
                   texname = '\\text{yt}')
