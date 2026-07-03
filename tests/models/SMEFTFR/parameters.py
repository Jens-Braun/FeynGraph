# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 13.2.1 for Linux x86 (64-bit) (January 27, 2023)
# Date: Mon 8 Dec 2025 20:55:51



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
alphas = Parameter(name = 'alphas',
                   nature = 'external',
                   type = 'real',
                   value = 0.1176,
                   texname = '\\alpha _s',
                   lhablock = 'FRBlock',
                   lhacode = [ 1 ])

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.000011638081097590524,
               texname = 'G_F',
               lhablock = 'FRBlock',
               lhacode = [ 2 ])

AEM = Parameter(name = 'AEM',
                nature = 'external',
                type = 'real',
                value = 0.0072973525692838015,
                texname = '\\alpha _{\\text{em}}',
                lhablock = 'FRBlock',
                lhacode = [ 3 ])

Zmass = Parameter(name = 'Zmass',
                  nature = 'external',
                  type = 'real',
                  value = 91.1876,
                  texname = 'M_Z',
                  lhablock = 'FRBlock',
                  lhacode = [ 4 ])

Wmass = Parameter(name = 'Wmass',
                  nature = 'external',
                  type = 'real',
                  value = 80.379,
                  texname = 'M_W',
                  lhablock = 'FRBlock',
                  lhacode = [ 5 ])

Hmass = Parameter(name = 'Hmass',
                  nature = 'external',
                  type = 'real',
                  value = 125.35,
                  texname = 'M_H',
                  lhablock = 'FRBlock',
                  lhacode = [ 6 ])

tmass = Parameter(name = 'tmass',
                  nature = 'external',
                  type = 'real',
                  value = 172.76,
                  texname = 'm_t',
                  lhablock = 'FRBlock',
                  lhacode = [ 7 ])

cmass = Parameter(name = 'cmass',
                  nature = 'external',
                  type = 'real',
                  value = 1.27,
                  texname = 'm_c',
                  lhablock = 'FRBlock',
                  lhacode = [ 8 ])

umass = Parameter(name = 'umass',
                  nature = 'external',
                  type = 'real',
                  value = 0.0025499999999999997,
                  texname = 'm_u',
                  lhablock = 'FRBlock',
                  lhacode = [ 9 ])

bmass = Parameter(name = 'bmass',
                  nature = 'external',
                  type = 'real',
                  value = 4.7,
                  texname = 'm_b',
                  lhablock = 'FRBlock',
                  lhacode = [ 10 ])

smass = Parameter(name = 'smass',
                  nature = 'external',
                  type = 'real',
                  value = 0.101,
                  texname = 'm_s',
                  lhablock = 'FRBlock',
                  lhacode = [ 11 ])

dmass = Parameter(name = 'dmass',
                  nature = 'external',
                  type = 'real',
                  value = 0.00504,
                  texname = 'm_d',
                  lhablock = 'FRBlock',
                  lhacode = [ 12 ])

taumass = Parameter(name = 'taumass',
                    nature = 'external',
                    type = 'real',
                    value = 1.77686,
                    texname = 'm_{\\tau }',
                    lhablock = 'FRBlock',
                    lhacode = [ 13 ])

mumass = Parameter(name = 'mumass',
                   nature = 'external',
                   type = 'real',
                   value = 0.1056583755,
                   texname = 'm_{\\mu }',
                   lhablock = 'FRBlock',
                   lhacode = [ 14 ])

emass = Parameter(name = 'emass',
                  nature = 'external',
                  type = 'real',
                  value = 0.00051099895,
                  texname = 'm_e',
                  lhablock = 'FRBlock',
                  lhacode = [ 15 ])

lamT = Parameter(name = 'lamT',
                 nature = 'external',
                 type = 'real',
                 value = 0.22537,
                 texname = '\\lambda _T',
                 lhablock = 'FRBlock',
                 lhacode = [ 16 ])

AT = Parameter(name = 'AT',
               nature = 'external',
               type = 'real',
               value = 0.828,
               texname = 'A_T',
               lhablock = 'FRBlock',
               lhacode = [ 17 ])

rhoT = Parameter(name = 'rhoT',
                 nature = 'external',
                 type = 'real',
                 value = 0.194,
                 texname = '\\rho _T',
                 lhablock = 'FRBlock',
                 lhacode = [ 18 ])

etaT = Parameter(name = 'etaT',
                 nature = 'external',
                 type = 'real',
                 value = 0.391,
                 texname = '\\eta _T',
                 lhablock = 'FRBlock',
                 lhacode = [ 19 ])

theta12 = Parameter(name = 'theta12',
                    nature = 'external',
                    type = 'real',
                    value = 0.5836381018669038,
                    texname = '\\theta _{12}',
                    lhablock = 'FRBlock',
                    lhacode = [ 20 ])

theta13 = Parameter(name = 'theta13',
                    nature = 'external',
                    type = 'real',
                    value = 0.14957471689591406,
                    texname = '\\theta _{13}',
                    lhablock = 'FRBlock',
                    lhacode = [ 21 ])

theta23 = Parameter(name = 'theta23',
                    nature = 'external',
                    type = 'real',
                    value = 0.8587019919812102,
                    texname = '\\theta _{23}',
                    lhablock = 'FRBlock',
                    lhacode = [ 22 ])

delta = Parameter(name = 'delta',
                  nature = 'external',
                  type = 'real',
                  value = 3.3859562021615193,
                  texname = '\\delta',
                  lhablock = 'FRBlock',
                  lhacode = [ 23 ])

taumu = Parameter(name = 'taumu',
                  nature = 'external',
                  type = 'real',
                  value = 3.337862504e18,
                  texname = '\\tau _{\\mu }',
                  lhablock = 'FRBlock',
                  lhacode = [ 24 ])

mB = Parameter(name = 'mB',
               nature = 'external',
               type = 'real',
               value = 5.27932,
               texname = 'm^{\\text{B+}}',
               lhablock = 'FRBlock',
               lhacode = [ 25 ])

mBs = Parameter(name = 'mBs',
                nature = 'external',
                type = 'real',
                value = 5.36689,
                texname = 'm_{\\text{Bs}}',
                lhablock = 'FRBlock',
                lhacode = [ 26 ])

mBd = Parameter(name = 'mBd',
                nature = 'external',
                type = 'real',
                value = 5.27963,
                texname = 'm_{\\text{Bd}}',
                lhablock = 'FRBlock',
                lhacode = [ 27 ])

mK = Parameter(name = 'mK',
               nature = 'external',
               type = 'real',
               value = 0.493677,
               texname = 'm^{\\text{K+}}',
               lhablock = 'FRBlock',
               lhacode = [ 28 ])

mPi = Parameter(name = 'mPi',
                nature = 'external',
                type = 'real',
                value = 0.13957061,
                texname = 'm^{\\text{$\\pi $+}}',
                lhablock = 'FRBlock',
                lhacode = [ 29 ])

S1 = Parameter(name = 'S1',
               nature = 'external',
               type = 'real',
               value = 2.3124,
               texname = 'S_1\\left(m_W\\right)',
               lhablock = 'FRBlock',
               lhacode = [ 30 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 0.00754852887435655,
                  texname = '\\alpha _{\\text{em}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 31 ])

G1 = Parameter(name = 'G1',
               nature = 'external',
               type = 'real',
               value = 0.34940513159160064,
               texname = 'G_1',
               lhablock = 'FRBlock',
               lhacode = [ 32 ])

GW = Parameter(name = 'GW',
               nature = 'external',
               type = 'real',
               value = 0.6521849654370863,
               texname = 'G_W',
               lhablock = 'FRBlock',
               lhacode = [ 33 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1176,
               texname = '\\alpha _s',
               lhablock = 'FRBlock',
               lhacode = [ 34 ])

vev = Parameter(name = 'vev',
                nature = 'external',
                type = 'real',
                value = 246.49142270898864,
                texname = 'v',
                lhablock = 'FRBlock',
                lhacode = [ 35 ])

hlambda = Parameter(name = 'hlambda',
                    nature = 'external',
                    type = 'real',
                    value = 0.2586098447591928,
                    texname = '\\lambda',
                    lhablock = 'FRBlock',
                    lhacode = [ 36 ])

mz = Parameter(name = 'mz',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = 'M_Z',
               lhablock = 'FRBlock',
               lhacode = [ 37 ])

mw = Parameter(name = 'mw',
               nature = 'external',
               type = 'real',
               value = 80.379,
               texname = 'M_W',
               lhablock = 'FRBlock',
               lhacode = [ 38 ])

mh = Parameter(name = 'mh',
               nature = 'external',
               type = 'real',
               value = 125.35,
               texname = 'M_H',
               lhablock = 'FRBlock',
               lhacode = [ 39 ])

mqt = Parameter(name = 'mqt',
                nature = 'external',
                type = 'real',
                value = 172.76,
                texname = 'm_t',
                lhablock = 'FRBlock',
                lhacode = [ 40 ])

mqc = Parameter(name = 'mqc',
                nature = 'external',
                type = 'real',
                value = 1.27,
                texname = 'm_c',
                lhablock = 'FRBlock',
                lhacode = [ 41 ])

mqu = Parameter(name = 'mqu',
                nature = 'external',
                type = 'real',
                value = 0.0025499999999999997,
                texname = 'm_u',
                lhablock = 'FRBlock',
                lhacode = [ 42 ])

mqb = Parameter(name = 'mqb',
                nature = 'external',
                type = 'real',
                value = 4.7,
                texname = 'm_b',
                lhablock = 'FRBlock',
                lhacode = [ 43 ])

mqs = Parameter(name = 'mqs',
                nature = 'external',
                type = 'real',
                value = 0.101,
                texname = 'm_s',
                lhablock = 'FRBlock',
                lhacode = [ 44 ])

mqd = Parameter(name = 'mqd',
                nature = 'external',
                type = 'real',
                value = 0.00504,
                texname = 'm_d',
                lhablock = 'FRBlock',
                lhacode = [ 45 ])

mlt = Parameter(name = 'mlt',
                nature = 'external',
                type = 'real',
                value = 1.77686,
                texname = 'm_{\\tau }',
                lhablock = 'FRBlock',
                lhacode = [ 46 ])

mlm = Parameter(name = 'mlm',
                nature = 'external',
                type = 'real',
                value = 0.1056583755,
                texname = 'm_{\\mu }',
                lhablock = 'FRBlock',
                lhacode = [ 47 ])

mle = Parameter(name = 'mle',
                nature = 'external',
                type = 'real',
                value = 0.00051099895,
                texname = 'm_e',
                lhablock = 'FRBlock',
                lhacode = [ 48 ])

mve = Parameter(name = 'mve',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'm_{\\text{$\\nu $e}}',
                lhablock = 'FRBlock',
                lhacode = [ 49 ])

mvm = Parameter(name = 'mvm',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'm_{\\nu \\mu }',
                lhablock = 'FRBlock',
                lhacode = [ 50 ])

mvt = Parameter(name = 'mvt',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'm_{\\nu \\tau }',
                lhablock = 'FRBlock',
                lhacode = [ 51 ])

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

xiA = Parameter(name = 'xiA',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\xi _A',
                lhablock = 'FRBlock',
                lhacode = [ 58 ])

xiG = Parameter(name = 'xiG',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\xi _G',
                lhablock = 'FRBlock',
                lhacode = [ 59 ])

g1norm = Parameter(name = 'g1norm',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = 'Z_{\\text{g\'}}',
                   lhablock = 'FRBlock',
                   lhacode = [ 60 ])

gwnorm = Parameter(name = 'gwnorm',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = 'Z_g',
                   lhablock = 'FRBlock',
                   lhacode = [ 61 ])

gsnorm = Parameter(name = 'gsnorm',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = 'Z_{\\text{gs}}',
                   lhablock = 'FRBlock',
                   lhacode = [ 62 ])

Hnorm = Parameter(name = 'Hnorm',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = 'Z_h',
                  lhablock = 'FRBlock',
                  lhacode = [ 63 ])

G0norm = Parameter(name = 'G0norm',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = 'Z_{\\text{G0}}',
                   lhablock = 'FRBlock',
                   lhacode = [ 64 ])

GPnorm = Parameter(name = 'GPnorm',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = 'Z_{\\text{G+}}',
                   lhablock = 'FRBlock',
                   lhacode = [ 65 ])

Wnorm = Parameter(name = 'Wnorm',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = 'Z_W',
                  lhablock = 'FRBlock',
                  lhacode = [ 66 ])

Gnorm = Parameter(name = 'Gnorm',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = 'Z_{\\text{gl}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 67 ])

AZnorm11 = Parameter(name = 'AZnorm11',
                     nature = 'external',
                     type = 'real',
                     value = 0.8814685330022941,
                     texname = 'Z_{\\text{$\\gamma $Z}}^{11}',
                     lhablock = 'FRBlock',
                     lhacode = [ 68 ])

AZnorm12 = Parameter(name = 'AZnorm12',
                     nature = 'external',
                     type = 'real',
                     value = 0.4722427610104611,
                     texname = 'Z_{\\text{$\\gamma $Z}}^{12}',
                     lhablock = 'FRBlock',
                     lhacode = [ 69 ])

AZnorm21 = Parameter(name = 'AZnorm21',
                     nature = 'external',
                     type = 'real',
                     value = -0.4722427610104611,
                     texname = 'Z_{\\text{$\\gamma $Z}}^{21}',
                     lhablock = 'FRBlock',
                     lhacode = [ 70 ])

AZnorm22 = Parameter(name = 'AZnorm22',
                     nature = 'external',
                     type = 'real',
                     value = 0.8814685330022941,
                     texname = 'Z_{\\text{$\\gamma $Z}}^{22}',
                     lhablock = 'FRBlock',
                     lhacode = [ 71 ])

CG = Parameter(name = 'CG',
               nature = 'external',
               type = 'real',
               value = 0,
               texname = 'C^G',
               lhablock = 'FRBlock',
               lhacode = [ 72 ])

CGtilde = Parameter(name = 'CGtilde',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = 'C^{\\text{Gtilde}}',
                    lhablock = 'FRBlock',
                    lhacode = [ 73 ])

Cphi = Parameter(name = 'Cphi',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'C^{\\phi }',
                 lhablock = 'FRBlock',
                 lhacode = [ 74 ])

CphiB = Parameter(name = 'CphiB',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = 'C^{\\text{$\\phi $B}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 75 ])

CphiBox = Parameter(name = 'CphiBox',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = 'C^{\\text{$\\phi $Box}}',
                    lhablock = 'FRBlock',
                    lhacode = [ 76 ])

CphiBtilde = Parameter(name = 'CphiBtilde',
                       nature = 'external',
                       type = 'real',
                       value = 0,
                       texname = 'C^{\\text{$\\phi $Btilde}}',
                       lhablock = 'FRBlock',
                       lhacode = [ 77 ])

CphiD = Parameter(name = 'CphiD',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = 'C^{\\text{$\\phi $D}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 78 ])

CphiG = Parameter(name = 'CphiG',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = 'C^{\\text{$\\phi $G}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 79 ])

CphiGtilde = Parameter(name = 'CphiGtilde',
                       nature = 'external',
                       type = 'real',
                       value = 0,
                       texname = 'C^{\\text{$\\phi $Gtilde}}',
                       lhablock = 'FRBlock',
                       lhacode = [ 80 ])

CphiW = Parameter(name = 'CphiW',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = 'C^{\\text{$\\phi $W}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 81 ])

CphiWB = Parameter(name = 'CphiWB',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = 'C^{\\text{$\\phi $WB}}',
                   lhablock = 'FRBlock',
                   lhacode = [ 82 ])

CphiWtilde = Parameter(name = 'CphiWtilde',
                       nature = 'external',
                       type = 'real',
                       value = 0,
                       texname = 'C^{\\text{$\\phi $Wtilde}}',
                       lhablock = 'FRBlock',
                       lhacode = [ 83 ])

CphiWtildeB = Parameter(name = 'CphiWtildeB',
                        nature = 'external',
                        type = 'real',
                        value = 0,
                        texname = 'C^{\\text{$\\phi $WtildeB}}',
                        lhablock = 'FRBlock',
                        lhacode = [ 84 ])

CW = Parameter(name = 'CW',
               nature = 'external',
               type = 'real',
               value = 0,
               texname = 'C^W',
               lhablock = 'FRBlock',
               lhacode = [ 85 ])

CWtilde = Parameter(name = 'CWtilde',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = 'C^{\\text{Wtilde}}',
                    lhablock = 'FRBlock',
                    lhacode = [ 86 ])

Lam = Parameter(name = 'Lam',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\frac{1}{\\Lambda ^2}',
                lhablock = 'FRBlock',
                lhacode = [ 87 ])

Cephi1x1 = Parameter(name = 'Cephi1x1',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cephi1x1}',
                     lhablock = 'FRBlock10',
                     lhacode = [ 1, 1 ])

Cephi1x2 = Parameter(name = 'Cephi1x2',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cephi1x2}',
                     lhablock = 'FRBlock10',
                     lhacode = [ 1, 2 ])

Cephi1x3 = Parameter(name = 'Cephi1x3',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cephi1x3}',
                     lhablock = 'FRBlock10',
                     lhacode = [ 1, 3 ])

Cephi2x1 = Parameter(name = 'Cephi2x1',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cephi2x1}',
                     lhablock = 'FRBlock10',
                     lhacode = [ 2, 1 ])

Cephi2x2 = Parameter(name = 'Cephi2x2',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cephi2x2}',
                     lhablock = 'FRBlock10',
                     lhacode = [ 2, 2 ])

Cephi2x3 = Parameter(name = 'Cephi2x3',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cephi2x3}',
                     lhablock = 'FRBlock10',
                     lhacode = [ 2, 3 ])

Cephi3x1 = Parameter(name = 'Cephi3x1',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cephi3x1}',
                     lhablock = 'FRBlock10',
                     lhacode = [ 3, 1 ])

Cephi3x2 = Parameter(name = 'Cephi3x2',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cephi3x2}',
                     lhablock = 'FRBlock10',
                     lhacode = [ 3, 2 ])

Cephi3x3 = Parameter(name = 'Cephi3x3',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cephi3x3}',
                     lhablock = 'FRBlock10',
                     lhacode = [ 3, 3 ])

Cuphi1x1 = Parameter(name = 'Cuphi1x1',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cuphi1x1}',
                     lhablock = 'FRBlock11',
                     lhacode = [ 1, 1 ])

Cuphi1x2 = Parameter(name = 'Cuphi1x2',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cuphi1x2}',
                     lhablock = 'FRBlock11',
                     lhacode = [ 1, 2 ])

Cuphi1x3 = Parameter(name = 'Cuphi1x3',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cuphi1x3}',
                     lhablock = 'FRBlock11',
                     lhacode = [ 1, 3 ])

Cuphi2x1 = Parameter(name = 'Cuphi2x1',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cuphi2x1}',
                     lhablock = 'FRBlock11',
                     lhacode = [ 2, 1 ])

Cuphi2x2 = Parameter(name = 'Cuphi2x2',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cuphi2x2}',
                     lhablock = 'FRBlock11',
                     lhacode = [ 2, 2 ])

Cuphi2x3 = Parameter(name = 'Cuphi2x3',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cuphi2x3}',
                     lhablock = 'FRBlock11',
                     lhacode = [ 2, 3 ])

Cuphi3x1 = Parameter(name = 'Cuphi3x1',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cuphi3x1}',
                     lhablock = 'FRBlock11',
                     lhacode = [ 3, 1 ])

Cuphi3x2 = Parameter(name = 'Cuphi3x2',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cuphi3x2}',
                     lhablock = 'FRBlock11',
                     lhacode = [ 3, 2 ])

Cuphi3x3 = Parameter(name = 'Cuphi3x3',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cuphi3x3}',
                     lhablock = 'FRBlock11',
                     lhacode = [ 3, 3 ])

fmv1x1 = Parameter(name = 'fmv1x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmv1x1}',
                   lhablock = 'FRBlock2',
                   lhacode = [ 1, 1 ])

fmv1x2 = Parameter(name = 'fmv1x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmv1x2}',
                   lhablock = 'FRBlock2',
                   lhacode = [ 1, 2 ])

fmv1x3 = Parameter(name = 'fmv1x3',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmv1x3}',
                   lhablock = 'FRBlock2',
                   lhacode = [ 1, 3 ])

fmv2x1 = Parameter(name = 'fmv2x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmv2x1}',
                   lhablock = 'FRBlock2',
                   lhacode = [ 2, 1 ])

fmv2x2 = Parameter(name = 'fmv2x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmv2x2}',
                   lhablock = 'FRBlock2',
                   lhacode = [ 2, 2 ])

fmv2x3 = Parameter(name = 'fmv2x3',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmv2x3}',
                   lhablock = 'FRBlock2',
                   lhacode = [ 2, 3 ])

fmv3x1 = Parameter(name = 'fmv3x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmv3x1}',
                   lhablock = 'FRBlock2',
                   lhacode = [ 3, 1 ])

fmv3x2 = Parameter(name = 'fmv3x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmv3x2}',
                   lhablock = 'FRBlock2',
                   lhacode = [ 3, 2 ])

fmv3x3 = Parameter(name = 'fmv3x3',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmv3x3}',
                   lhablock = 'FRBlock2',
                   lhacode = [ 3, 3 ])

fml1x1 = Parameter(name = 'fml1x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0.00051099895,
                   texname = '\\text{fml1x1}',
                   lhablock = 'FRBlock3',
                   lhacode = [ 1, 1 ])

fml1x2 = Parameter(name = 'fml1x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fml1x2}',
                   lhablock = 'FRBlock3',
                   lhacode = [ 1, 2 ])

fml1x3 = Parameter(name = 'fml1x3',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fml1x3}',
                   lhablock = 'FRBlock3',
                   lhacode = [ 1, 3 ])

fml2x1 = Parameter(name = 'fml2x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fml2x1}',
                   lhablock = 'FRBlock3',
                   lhacode = [ 2, 1 ])

fml2x2 = Parameter(name = 'fml2x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0.1056583755,
                   texname = '\\text{fml2x2}',
                   lhablock = 'FRBlock3',
                   lhacode = [ 2, 2 ])

fml2x3 = Parameter(name = 'fml2x3',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fml2x3}',
                   lhablock = 'FRBlock3',
                   lhacode = [ 2, 3 ])

fml3x1 = Parameter(name = 'fml3x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fml3x1}',
                   lhablock = 'FRBlock3',
                   lhacode = [ 3, 1 ])

fml3x2 = Parameter(name = 'fml3x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fml3x2}',
                   lhablock = 'FRBlock3',
                   lhacode = [ 3, 2 ])

fml3x3 = Parameter(name = 'fml3x3',
                   nature = 'external',
                   type = 'complex',
                   value = 1.77686,
                   texname = '\\text{fml3x3}',
                   lhablock = 'FRBlock3',
                   lhacode = [ 3, 3 ])

fmd1x1 = Parameter(name = 'fmd1x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0.00504,
                   texname = '\\text{fmd1x1}',
                   lhablock = 'FRBlock4',
                   lhacode = [ 1, 1 ])

fmd1x2 = Parameter(name = 'fmd1x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmd1x2}',
                   lhablock = 'FRBlock4',
                   lhacode = [ 1, 2 ])

fmd1x3 = Parameter(name = 'fmd1x3',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmd1x3}',
                   lhablock = 'FRBlock4',
                   lhacode = [ 1, 3 ])

fmd2x1 = Parameter(name = 'fmd2x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmd2x1}',
                   lhablock = 'FRBlock4',
                   lhacode = [ 2, 1 ])

fmd2x2 = Parameter(name = 'fmd2x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0.101,
                   texname = '\\text{fmd2x2}',
                   lhablock = 'FRBlock4',
                   lhacode = [ 2, 2 ])

fmd2x3 = Parameter(name = 'fmd2x3',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmd2x3}',
                   lhablock = 'FRBlock4',
                   lhacode = [ 2, 3 ])

fmd3x1 = Parameter(name = 'fmd3x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmd3x1}',
                   lhablock = 'FRBlock4',
                   lhacode = [ 3, 1 ])

fmd3x2 = Parameter(name = 'fmd3x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmd3x2}',
                   lhablock = 'FRBlock4',
                   lhacode = [ 3, 2 ])

fmd3x3 = Parameter(name = 'fmd3x3',
                   nature = 'external',
                   type = 'complex',
                   value = 4.7,
                   texname = '\\text{fmd3x3}',
                   lhablock = 'FRBlock4',
                   lhacode = [ 3, 3 ])

fmu1x1 = Parameter(name = 'fmu1x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0.0025499999999999997,
                   texname = '\\text{fmu1x1}',
                   lhablock = 'FRBlock5',
                   lhacode = [ 1, 1 ])

fmu1x2 = Parameter(name = 'fmu1x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmu1x2}',
                   lhablock = 'FRBlock5',
                   lhacode = [ 1, 2 ])

fmu1x3 = Parameter(name = 'fmu1x3',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmu1x3}',
                   lhablock = 'FRBlock5',
                   lhacode = [ 1, 3 ])

fmu2x1 = Parameter(name = 'fmu2x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmu2x1}',
                   lhablock = 'FRBlock5',
                   lhacode = [ 2, 1 ])

fmu2x2 = Parameter(name = 'fmu2x2',
                   nature = 'external',
                   type = 'complex',
                   value = 1.27,
                   texname = '\\text{fmu2x2}',
                   lhablock = 'FRBlock5',
                   lhacode = [ 2, 2 ])

fmu2x3 = Parameter(name = 'fmu2x3',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmu2x3}',
                   lhablock = 'FRBlock5',
                   lhacode = [ 2, 3 ])

fmu3x1 = Parameter(name = 'fmu3x1',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmu3x1}',
                   lhablock = 'FRBlock5',
                   lhacode = [ 3, 1 ])

fmu3x2 = Parameter(name = 'fmu3x2',
                   nature = 'external',
                   type = 'complex',
                   value = 0,
                   texname = '\\text{fmu3x2}',
                   lhablock = 'FRBlock5',
                   lhacode = [ 3, 2 ])

fmu3x3 = Parameter(name = 'fmu3x3',
                   nature = 'external',
                   type = 'complex',
                   value = 172.76,
                   texname = '\\text{fmu3x3}',
                   lhablock = 'FRBlock5',
                   lhacode = [ 3, 3 ])

Kq1x1 = Parameter(name = 'Kq1x1',
                  nature = 'external',
                  type = 'complex',
                  value = 0.9742817077526272,
                  texname = '\\text{Kq1x1}',
                  lhablock = 'FRBlock6',
                  lhacode = [ 1, 1 ])

Kq1x2 = Parameter(name = 'Kq1x2',
                  nature = 'external',
                  type = 'complex',
                  value = 0.22537,
                  texname = '\\text{Kq1x2}',
                  lhablock = 'FRBlock6',
                  lhacode = [ 1, 2 ])

Kq1x3 = Parameter(name = 'Kq1x3',
                  nature = 'external',
                  type = 'complex',
                  value = 0.0018854365545299529,
                  texname = '\\text{Kq1x3}',
                  lhablock = 'FRBlock6',
                  lhacode = [ 1, 3 ])

Kq2x1 = Parameter(name = 'Kq2x1',
                  nature = 'external',
                  type = 'complex',
                  value = -0.22524802730392973,
                  texname = '\\text{Kq2x1}',
                  lhablock = 'FRBlock6',
                  lhacode = [ 2, 1 ])

Kq2x2 = Parameter(name = 'Kq2x2',
                  nature = 'external',
                  type = 'complex',
                  value = 0.9733973762490354,
                  texname = '\\text{Kq2x2}',
                  lhablock = 'FRBlock6',
                  lhacode = [ 2, 2 ])

Kq2x3 = Parameter(name = 'Kq2x3',
                  nature = 'external',
                  type = 'complex',
                  value = 0.04205547535319999,
                  texname = '\\text{Kq2x3}',
                  lhablock = 'FRBlock6',
                  lhacode = [ 2, 3 ])

Kq3x1 = Parameter(name = 'Kq3x1',
                  nature = 'external',
                  type = 'complex',
                  value = 0.00763930223916265,
                  texname = '\\text{Kq3x1}',
                  lhablock = 'FRBlock6',
                  lhacode = [ 3, 1 ])

Kq3x2 = Parameter(name = 'Kq3x2',
                  nature = 'external',
                  type = 'complex',
                  value = -0.04140183902445822,
                  texname = '\\text{Kq3x2}',
                  lhablock = 'FRBlock6',
                  lhacode = [ 3, 2 ])

Kq3x3 = Parameter(name = 'Kq3x3',
                  nature = 'external',
                  type = 'complex',
                  value = 0.9991156684964082,
                  texname = '\\text{Kq3x3}',
                  lhablock = 'FRBlock6',
                  lhacode = [ 3, 3 ])

Ul1x1 = Parameter(name = 'Ul1x1',
                  nature = 'external',
                  type = 'complex',
                  value = 0.8251461863096498,
                  texname = '\\text{Ul1x1}',
                  lhablock = 'FRBlock7',
                  lhacode = [ 1, 1 ])

Ul1x2 = Parameter(name = 'Ul1x2',
                  nature = 'external',
                  type = 'complex',
                  value = 0.544910563973106,
                  texname = '\\text{Ul1x2}',
                  lhablock = 'FRBlock7',
                  lhacode = [ 1, 2 ])

Ul1x3 = Parameter(name = 'Ul1x3',
                  nature = 'external',
                  type = 'complex',
                  value = -0.14459052219790333,
                  texname = '\\text{Ul1x3}',
                  lhablock = 'FRBlock7',
                  lhacode = [ 1, 3 ])

Ul2x1 = Parameter(name = 'Ul2x1',
                  nature = 'external',
                  type = 'complex',
                  value = -0.2687405948486291,
                  texname = '\\text{Ul2x1}',
                  lhablock = 'FRBlock7',
                  lhacode = [ 2, 1 ])

Ul2x2 = Parameter(name = 'Ul2x2',
                  nature = 'external',
                  type = 'complex',
                  value = 0.6055718161076215,
                  texname = '\\text{Ul2x2}',
                  lhablock = 'FRBlock7',
                  lhacode = [ 2, 2 ])

Ul2x3 = Parameter(name = 'Ul2x3',
                  nature = 'external',
                  type = 'complex',
                  value = 0.7485428591740605,
                  texname = '\\text{Ul2x3}',
                  lhablock = 'FRBlock7',
                  lhacode = [ 2, 3 ])

Ul3x1 = Parameter(name = 'Ul3x1',
                  nature = 'external',
                  type = 'complex',
                  value = 0.49599108348417026,
                  texname = '\\text{Ul3x1}',
                  lhablock = 'FRBlock7',
                  lhacode = [ 3, 1 ])

Ul3x2 = Parameter(name = 'Ul3x2',
                  nature = 'external',
                  type = 'complex',
                  value = -0.5796210248344889,
                  texname = '\\text{Ul3x2}',
                  lhablock = 'FRBlock7',
                  lhacode = [ 3, 2 ])

Ul3x3 = Parameter(name = 'Ul3x3',
                  nature = 'external',
                  type = 'complex',
                  value = 0.6461248636992469,
                  texname = '\\text{Ul3x3}',
                  lhablock = 'FRBlock7',
                  lhacode = [ 3, 3 ])

AZnorm1x1 = Parameter(name = 'AZnorm1x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.8814685330022941,
                      texname = '\\text{AZnorm1x1}',
                      lhablock = 'FRBlock8',
                      lhacode = [ 1, 1 ])

Cdphi1x1 = Parameter(name = 'Cdphi1x1',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cdphi1x1}',
                     lhablock = 'FRBlock9',
                     lhacode = [ 1, 1 ])

Cdphi1x2 = Parameter(name = 'Cdphi1x2',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cdphi1x2}',
                     lhablock = 'FRBlock9',
                     lhacode = [ 1, 2 ])

Cdphi1x3 = Parameter(name = 'Cdphi1x3',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cdphi1x3}',
                     lhablock = 'FRBlock9',
                     lhacode = [ 1, 3 ])

Cdphi2x1 = Parameter(name = 'Cdphi2x1',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cdphi2x1}',
                     lhablock = 'FRBlock9',
                     lhacode = [ 2, 1 ])

Cdphi2x2 = Parameter(name = 'Cdphi2x2',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cdphi2x2}',
                     lhablock = 'FRBlock9',
                     lhacode = [ 2, 2 ])

Cdphi2x3 = Parameter(name = 'Cdphi2x3',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cdphi2x3}',
                     lhablock = 'FRBlock9',
                     lhacode = [ 2, 3 ])

Cdphi3x1 = Parameter(name = 'Cdphi3x1',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cdphi3x1}',
                     lhablock = 'FRBlock9',
                     lhacode = [ 3, 1 ])

Cdphi3x2 = Parameter(name = 'Cdphi3x2',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cdphi3x2}',
                     lhablock = 'FRBlock9',
                     lhacode = [ 3, 2 ])

Cdphi3x3 = Parameter(name = 'Cdphi3x3',
                     nature = 'external',
                     type = 'complex',
                     value = 0,
                     texname = '\\text{Cdphi3x3}',
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

MLE = Parameter(name = 'MLE',
                nature = 'external',
                type = 'real',
                value = 0.00051099895,
                texname = '\\text{MLE}',
                lhablock = 'MASS',
                lhacode = [ 11 ])

MLM = Parameter(name = 'MLM',
                nature = 'external',
                type = 'real',
                value = 0.1056583755,
                texname = '\\text{MLM}',
                lhablock = 'MASS',
                lhacode = [ 13 ])

MLT = Parameter(name = 'MLT',
                nature = 'external',
                type = 'real',
                value = 1.77686,
                texname = '\\text{MLT}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

MQU = Parameter(name = 'MQU',
                nature = 'external',
                type = 'real',
                value = 0.0025499999999999997,
                texname = '\\text{MQU}',
                lhablock = 'MASS',
                lhacode = [ 2 ])

MQC = Parameter(name = 'MQC',
                nature = 'external',
                type = 'real',
                value = 1.27,
                texname = '\\text{MQC}',
                lhablock = 'MASS',
                lhacode = [ 4 ])

MQT = Parameter(name = 'MQT',
                nature = 'external',
                type = 'real',
                value = 172.76,
                texname = '\\text{MQT}',
                lhablock = 'MASS',
                lhacode = [ 6 ])

MQD = Parameter(name = 'MQD',
                nature = 'external',
                type = 'real',
                value = 0.00504,
                texname = '\\text{MQD}',
                lhablock = 'MASS',
                lhacode = [ 1 ])

MQS = Parameter(name = 'MQS',
                nature = 'external',
                type = 'real',
                value = 0.101,
                texname = '\\text{MQS}',
                lhablock = 'MASS',
                lhacode = [ 3 ])

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

