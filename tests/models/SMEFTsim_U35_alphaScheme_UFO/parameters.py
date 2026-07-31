# This file was automatically created by FeynRules 2.3.35
# Mathematica version: 12.1.0 for Linux x86 (64-bit) (March 18, 2020)
# Date: Wed 6 Jan 2021 16:20:38



from .object_library import all_parameters, Parameter


from .function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
CKMlambda = Parameter(name = 'CKMlambda',
                      nature = 'external',
                      type = 'real',
                      value = 0.2265,
                      texname = '\\text{CKMlambda}',
                      lhablock = 'CKMBLOCK',
                      lhacode = [ 2 ])

CKMA = Parameter(name = 'CKMA',
                 nature = 'external',
                 type = 'real',
                 value = 0.79,
                 texname = '\\text{CKMA}',
                 lhablock = 'CKMBLOCK',
                 lhacode = [ 3 ])

CKMrho = Parameter(name = 'CKMrho',
                   nature = 'external',
                   type = 'real',
                   value = 0.141,
                   texname = '\\text{CKMrho}',
                   lhablock = 'CKMBLOCK',
                   lhacode = [ 4 ])

CKMeta = Parameter(name = 'CKMeta',
                   nature = 'external',
                   type = 'real',
                   value = 0.357,
                   texname = '\\text{CKMeta}',
                   lhablock = 'CKMBLOCK',
                   lhacode = [ 5 ])

cG = Parameter(name = 'cG',
               nature = 'external',
               type = 'real',
               value = 0,
               texname = 'c_G',
               lhablock = 'SMEFT',
               lhacode = [ 1 ])

cW = Parameter(name = 'cW',
               nature = 'external',
               type = 'real',
               value = 0,
               texname = 'c_W',
               lhablock = 'SMEFT',
               lhacode = [ 2 ])

cH = Parameter(name = 'cH',
               nature = 'external',
               type = 'real',
               value = 0,
               texname = 'c_H',
               lhablock = 'SMEFT',
               lhacode = [ 3 ])

cHbox = Parameter(name = 'cHbox',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = 'c_{H \\square }',
                  lhablock = 'SMEFT',
                  lhacode = [ 4 ])

cHDD = Parameter(name = 'cHDD',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{HD}}',
                 lhablock = 'SMEFT',
                 lhacode = [ 5 ])

cHG = Parameter(name = 'cHG',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{HG}}',
                lhablock = 'SMEFT',
                lhacode = [ 6 ])

cHW = Parameter(name = 'cHW',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{HW}}',
                lhablock = 'SMEFT',
                lhacode = [ 7 ])

cHB = Parameter(name = 'cHB',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{HB}}',
                lhablock = 'SMEFT',
                lhacode = [ 8 ])

cHWB = Parameter(name = 'cHWB',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{HWB}}',
                 lhablock = 'SMEFT',
                 lhacode = [ 9 ])

ceHRe = Parameter(name = 'ceHRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{ceHRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 10 ])

cuHRe = Parameter(name = 'cuHRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cuHRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 11 ])

cdHRe = Parameter(name = 'cdHRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cdHRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 12 ])

ceWRe = Parameter(name = 'ceWRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{ceWRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 13 ])

ceBRe = Parameter(name = 'ceBRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{ceBRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 14 ])

cuGRe = Parameter(name = 'cuGRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cuGRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 15 ])

cuWRe = Parameter(name = 'cuWRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cuWRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 16 ])

cuBRe = Parameter(name = 'cuBRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cuBRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 17 ])

cdGRe = Parameter(name = 'cdGRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cdGRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 18 ])

cdWRe = Parameter(name = 'cdWRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cdWRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 19 ])

cdBRe = Parameter(name = 'cdBRe',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cdBRe}',
                  lhablock = 'SMEFT',
                  lhacode = [ 20 ])

cHl1 = Parameter(name = 'cHl1',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{Hl}}^1',
                 lhablock = 'SMEFT',
                 lhacode = [ 21 ])

cHl3 = Parameter(name = 'cHl3',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{Hl}}^3',
                 lhablock = 'SMEFT',
                 lhacode = [ 22 ])

cHe = Parameter(name = 'cHe',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{He}}',
                lhablock = 'SMEFT',
                lhacode = [ 23 ])

cHq1 = Parameter(name = 'cHq1',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{Hq}}^1',
                 lhablock = 'SMEFT',
                 lhacode = [ 24 ])

cHq3 = Parameter(name = 'cHq3',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{Hq}}^3',
                 lhablock = 'SMEFT',
                 lhacode = [ 25 ])

cHu = Parameter(name = 'cHu',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{Hu}}',
                lhablock = 'SMEFT',
                lhacode = [ 26 ])

cHd = Parameter(name = 'cHd',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{Hd}}',
                lhablock = 'SMEFT',
                lhacode = [ 27 ])

cHudRe = Parameter(name = 'cHudRe',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{cHudRe}',
                   lhablock = 'SMEFT',
                   lhacode = [ 28 ])

cll = Parameter(name = 'cll',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{ll}}',
                lhablock = 'SMEFT',
                lhacode = [ 29 ])

cll1 = Parameter(name = 'cll1',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{ll}}^{\\text{Prime}}',
                 lhablock = 'SMEFT',
                 lhacode = [ 30 ])

cqq1 = Parameter(name = 'cqq1',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{qq}}^1',
                 lhablock = 'SMEFT',
                 lhacode = [ 31 ])

cqq11 = Parameter(name = 'cqq11',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = 'c_{\\text{qq}}^{\\text{Prime}}',
                  lhablock = 'SMEFT',
                  lhacode = [ 32 ])

cqq3 = Parameter(name = 'cqq3',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{qq}}^3',
                 lhablock = 'SMEFT',
                 lhacode = [ 33 ])

cqq31 = Parameter(name = 'cqq31',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = 'c_{\\text{qq}}^{3 \\text{Prime}}',
                  lhablock = 'SMEFT',
                  lhacode = [ 34 ])

clq1 = Parameter(name = 'clq1',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{lq}}^1',
                 lhablock = 'SMEFT',
                 lhacode = [ 35 ])

clq3 = Parameter(name = 'clq3',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{lq}}^3',
                 lhablock = 'SMEFT',
                 lhacode = [ 36 ])

cee = Parameter(name = 'cee',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{ee}}',
                lhablock = 'SMEFT',
                lhacode = [ 37 ])

cuu = Parameter(name = 'cuu',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{uu}}',
                lhablock = 'SMEFT',
                lhacode = [ 38 ])

cuu1 = Parameter(name = 'cuu1',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{uu}}^{\\text{Prime}}',
                 lhablock = 'SMEFT',
                 lhacode = [ 39 ])

cdd = Parameter(name = 'cdd',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{dd}}',
                lhablock = 'SMEFT',
                lhacode = [ 40 ])

cdd1 = Parameter(name = 'cdd1',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{dd}}^{\\text{Prime}}',
                 lhablock = 'SMEFT',
                 lhacode = [ 41 ])

ceu = Parameter(name = 'ceu',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{eu}}',
                lhablock = 'SMEFT',
                lhacode = [ 42 ])

ced = Parameter(name = 'ced',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{ed}}',
                lhablock = 'SMEFT',
                lhacode = [ 43 ])

cud1 = Parameter(name = 'cud1',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{ud}}^1',
                 lhablock = 'SMEFT',
                 lhacode = [ 44 ])

cud8 = Parameter(name = 'cud8',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{ud}}^8',
                 lhablock = 'SMEFT',
                 lhacode = [ 45 ])

cle = Parameter(name = 'cle',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{le}}',
                lhablock = 'SMEFT',
                lhacode = [ 46 ])

clu = Parameter(name = 'clu',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{lu}}',
                lhablock = 'SMEFT',
                lhacode = [ 47 ])

cld = Parameter(name = 'cld',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{ld}}',
                lhablock = 'SMEFT',
                lhacode = [ 48 ])

cqe = Parameter(name = 'cqe',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = 'c_{\\text{qe}}',
                lhablock = 'SMEFT',
                lhacode = [ 49 ])

cqu1 = Parameter(name = 'cqu1',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{qu}}^1',
                 lhablock = 'SMEFT',
                 lhacode = [ 50 ])

cqu8 = Parameter(name = 'cqu8',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{qu}}^8',
                 lhablock = 'SMEFT',
                 lhacode = [ 51 ])

cqd1 = Parameter(name = 'cqd1',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{qd}}^1',
                 lhablock = 'SMEFT',
                 lhacode = [ 52 ])

cqd8 = Parameter(name = 'cqd8',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = 'c_{\\text{qd}}^8',
                 lhablock = 'SMEFT',
                 lhacode = [ 53 ])

cledqRe = Parameter(name = 'cledqRe',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{cledqRe}',
                    lhablock = 'SMEFT',
                    lhacode = [ 54 ])

cquqd1Re = Parameter(name = 'cquqd1Re',
                     nature = 'external',
                     type = 'real',
                     value = 0,
                     texname = '\\text{cquqd1Re}',
                     lhablock = 'SMEFT',
                     lhacode = [ 55 ])

cquqd11Re = Parameter(name = 'cquqd11Re',
                      nature = 'external',
                      type = 'real',
                      value = 0,
                      texname = '\\text{cquqd11Re}',
                      lhablock = 'SMEFT',
                      lhacode = [ 56 ])

cquqd8Re = Parameter(name = 'cquqd8Re',
                     nature = 'external',
                     type = 'real',
                     value = 0,
                     texname = '\\text{cquqd8Re}',
                     lhablock = 'SMEFT',
                     lhacode = [ 57 ])

cquqd81Re = Parameter(name = 'cquqd81Re',
                      nature = 'external',
                      type = 'real',
                      value = 0,
                      texname = '\\text{cquqd81Re}',
                      lhablock = 'SMEFT',
                      lhacode = [ 58 ])

clequ1Re = Parameter(name = 'clequ1Re',
                     nature = 'external',
                     type = 'real',
                     value = 0,
                     texname = '\\text{clequ1Re}',
                     lhablock = 'SMEFT',
                     lhacode = [ 59 ])

clequ3Re = Parameter(name = 'clequ3Re',
                     nature = 'external',
                     type = 'real',
                     value = 0,
                     texname = '\\text{clequ3Re}',
                     lhablock = 'SMEFT',
                     lhacode = [ 60 ])

cGtil = Parameter(name = 'cGtil',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = 'c_{\\tilde{G}}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 1 ])

cWtil = Parameter(name = 'cWtil',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = 'c_{\\tilde{W}}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 2 ])

cHGtil = Parameter(name = 'cHGtil',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = 'c_{H \\tilde{G}}',
                   lhablock = 'SMEFTcpv',
                   lhacode = [ 3 ])

cHWtil = Parameter(name = 'cHWtil',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = 'c_{H \\tilde{W}}',
                   lhablock = 'SMEFTcpv',
                   lhacode = [ 4 ])

cHBtil = Parameter(name = 'cHBtil',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = 'c_{H \\tilde{B}}',
                   lhablock = 'SMEFTcpv',
                   lhacode = [ 5 ])

cHWBtil = Parameter(name = 'cHWBtil',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = 'c_{B H \\tilde{W}}',
                    lhablock = 'SMEFTcpv',
                    lhacode = [ 6 ])

ceWIm = Parameter(name = 'ceWIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{ceWIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 7 ])

ceBIm = Parameter(name = 'ceBIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{ceBIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 8 ])

cuGIm = Parameter(name = 'cuGIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cuGIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 9 ])

cuWIm = Parameter(name = 'cuWIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cuWIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 10 ])

cuBIm = Parameter(name = 'cuBIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cuBIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 11 ])

cdGIm = Parameter(name = 'cdGIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cdGIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 12 ])

cdWIm = Parameter(name = 'cdWIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cdWIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 13 ])

cdBIm = Parameter(name = 'cdBIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cdBIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 14 ])

cHudIm = Parameter(name = 'cHudIm',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{cHudIm}',
                   lhablock = 'SMEFTcpv',
                   lhacode = [ 15 ])

ceHIm = Parameter(name = 'ceHIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{ceHIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 16 ])

cuHIm = Parameter(name = 'cuHIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cuHIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 17 ])

cdHIm = Parameter(name = 'cdHIm',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\text{cdHIm}',
                  lhablock = 'SMEFTcpv',
                  lhacode = [ 18 ])

cledqIm = Parameter(name = 'cledqIm',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{cledqIm}',
                    lhablock = 'SMEFTcpv',
                    lhacode = [ 19 ])

cquqd1Im = Parameter(name = 'cquqd1Im',
                     nature = 'external',
                     type = 'real',
                     value = 0,
                     texname = '\\text{cquqd1Im}',
                     lhablock = 'SMEFTcpv',
                     lhacode = [ 20 ])

cquqd8Im = Parameter(name = 'cquqd8Im',
                     nature = 'external',
                     type = 'real',
                     value = 0,
                     texname = '\\text{cquqd8Im}',
                     lhablock = 'SMEFTcpv',
                     lhacode = [ 21 ])

cquqd11Im = Parameter(name = 'cquqd11Im',
                      nature = 'external',
                      type = 'real',
                      value = 0,
                      texname = '\\text{cquqd11Im}',
                      lhablock = 'SMEFTcpv',
                      lhacode = [ 22 ])

cquqd81Im = Parameter(name = 'cquqd81Im',
                      nature = 'external',
                      type = 'real',
                      value = 0,
                      texname = '\\text{cquqd81Im}',
                      lhablock = 'SMEFTcpv',
                      lhacode = [ 23 ])

clequ1Im = Parameter(name = 'clequ1Im',
                     nature = 'external',
                     type = 'real',
                     value = 0,
                     texname = '\\text{clequ1Im}',
                     lhablock = 'SMEFTcpv',
                     lhacode = [ 24 ])

clequ3Im = Parameter(name = 'clequ3Im',
                     nature = 'external',
                     type = 'real',
                     value = 0,
                     texname = '\\text{clequ3Im}',
                     lhablock = 'SMEFTcpv',
                     lhacode = [ 25 ])

LambdaSMEFT = Parameter(name = 'LambdaSMEFT',
                        nature = 'external',
                        type = 'real',
                        value = 1000,
                        texname = '\\Lambda',
                        lhablock = 'SMEFTcutoff',
                        lhacode = [ 1 ])

aEW = Parameter(name = 'aEW',
                nature = 'external',
                type = 'real',
                value = 0.007815552950371238,
                texname = '\\alpha _{\\text{EW}}',
                lhablock = 'SMINPUTS',
                lhacode = [ 1 ])

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.000011663787,
               texname = 'G_f',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1179,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

linearPropCorrections = Parameter(name = 'linearPropCorrections',
                                  nature = 'external',
                                  type = 'real',
                                  value = 0,
                                  texname = '\\text{linearPropCorrections}',
                                  lhablock = 'SWITCHES',
                                  lhacode = [ 1 ])

ymdo = Parameter(name = 'ymdo',
                 nature = 'external',
                 type = 'real',
                 value = 0.00467,
                 texname = '\\text{ymdo}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 1 ])

ymup = Parameter(name = 'ymup',
                 nature = 'external',
                 type = 'real',
                 value = 0.00216,
                 texname = '\\text{ymup}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 2 ])

yms = Parameter(name = 'yms',
                nature = 'external',
                type = 'real',
                value = 0.093,
                texname = '\\text{yms}',
                lhablock = 'YUKAWA',
                lhacode = [ 3 ])

ymc = Parameter(name = 'ymc',
                nature = 'external',
                type = 'real',
                value = 1.27,
                texname = '\\text{ymc}',
                lhablock = 'YUKAWA',
                lhacode = [ 4 ])

ymb = Parameter(name = 'ymb',
                nature = 'external',
                type = 'real',
                value = 4.18,
                texname = '\\text{ymb}',
                lhablock = 'YUKAWA',
                lhacode = [ 5 ])

ymt = Parameter(name = 'ymt',
                nature = 'external',
                type = 'real',
                value = 172.76,
                texname = '\\text{ymt}',
                lhablock = 'YUKAWA',
                lhacode = [ 6 ])

yme = Parameter(name = 'yme',
                nature = 'external',
                type = 'real',
                value = 0.000511,
                texname = '\\text{yme}',
                lhablock = 'YUKAWA',
                lhacode = [ 11 ])

ymm = Parameter(name = 'ymm',
                nature = 'external',
                type = 'real',
                value = 0.10566,
                texname = '\\text{ymm}',
                lhablock = 'YUKAWA',
                lhacode = [ 13 ])

ymtau = Parameter(name = 'ymtau',
                  nature = 'external',
                  type = 'real',
                  value = 1.777,
                  texname = '\\text{ymtau}',
                  lhablock = 'YUKAWA',
                  lhacode = [ 15 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

Me = Parameter(name = 'Me',
               nature = 'external',
               type = 'real',
               value = 0.000511,
               texname = '\\text{Me}',
               lhablock = 'MASS',
               lhacode = [ 11 ])

MMU = Parameter(name = 'MMU',
                nature = 'external',
                type = 'real',
                value = 0.10566,
                texname = '\\text{MMU}',
                lhablock = 'MASS',
                lhacode = [ 13 ])

MTA = Parameter(name = 'MTA',
                nature = 'external',
                type = 'real',
                value = 1.777,
                texname = '\\text{MTA}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

MU = Parameter(name = 'MU',
               nature = 'external',
               type = 'real',
               value = 0.00216,
               texname = 'M',
               lhablock = 'MASS',
               lhacode = [ 2 ])

MC = Parameter(name = 'MC',
               nature = 'external',
               type = 'real',
               value = 1.27,
               texname = '\\text{MC}',
               lhablock = 'MASS',
               lhacode = [ 4 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 172.76,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MD = Parameter(name = 'MD',
               nature = 'external',
               type = 'real',
               value = 0.00467,
               texname = '\\text{MD}',
               lhablock = 'MASS',
               lhacode = [ 1 ])

MS = Parameter(name = 'MS',
               nature = 'external',
               type = 'real',
               value = 0.093,
               texname = '\\text{MS}',
               lhablock = 'MASS',
               lhacode = [ 3 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.18,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 5 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 125.09,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.4952,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.085,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.33,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.00407,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])

ceH = Parameter(name = 'ceH',
                nature = 'internal',
                type = 'complex',
                value = 'ceHRe + ceHIm*complex(0,1)',
                texname = 'c_{\\text{eH}}')

cuH = Parameter(name = 'cuH',
                nature = 'internal',
                type = 'complex',
                value = 'cuHRe + cuHIm*complex(0,1)',
                texname = 'c_{\\text{uH}}')

cdH = Parameter(name = 'cdH',
                nature = 'internal',
                type = 'complex',
                value = 'cdHRe + cdHIm*complex(0,1)',
                texname = 'c_{\\text{dH}}')

ceW = Parameter(name = 'ceW',
                nature = 'internal',
                type = 'complex',
                value = 'ceWRe + ceWIm*complex(0,1)',
                texname = 'c_{\\text{eW}}')

ceB = Parameter(name = 'ceB',
                nature = 'internal',
                type = 'complex',
                value = 'ceBRe + ceBIm*complex(0,1)',
                texname = 'c_{\\text{eB}}')

cuG = Parameter(name = 'cuG',
                nature = 'internal',
                type = 'complex',
                value = 'cuGRe + cuGIm*complex(0,1)',
                texname = 'c_{\\text{uG}}')

cuW = Parameter(name = 'cuW',
                nature = 'internal',
                type = 'complex',
                value = 'cuWRe + cuWIm*complex(0,1)',
                texname = 'c_{\\text{uW}}')

cuB = Parameter(name = 'cuB',
                nature = 'internal',
                type = 'complex',
                value = 'cuBRe + cuBIm*complex(0,1)',
                texname = 'c_{\\text{uB}}')

cdG = Parameter(name = 'cdG',
                nature = 'internal',
                type = 'complex',
                value = 'cdGRe + cdGIm*complex(0,1)',
                texname = 'c_{\\text{dG}}')

cdW = Parameter(name = 'cdW',
                nature = 'internal',
                type = 'complex',
                value = 'cdWRe + cdWIm*complex(0,1)',
                texname = 'c_{\\text{dW}}')

cdB = Parameter(name = 'cdB',
                nature = 'internal',
                type = 'complex',
                value = 'cdBRe + cdBIm*complex(0,1)',
                texname = 'c_{\\text{dB}}')

cHud = Parameter(name = 'cHud',
                 nature = 'internal',
                 type = 'complex',
                 value = 'cHudRe + cHudIm*complex(0,1)',
                 texname = 'c_{\\text{Hud}}')

cledq = Parameter(name = 'cledq',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cledqRe + cledqIm*complex(0,1)',
                  texname = 'c_{\\text{ledq}}')

cquqd1 = Parameter(name = 'cquqd1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cquqd1Re + cquqd1Im*complex(0,1)',
                   texname = 'c_{\\text{quqd}}^1')

cquqd11 = Parameter(name = 'cquqd11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'cquqd11Re + cquqd11Im*complex(0,1)',
                    texname = 'c_{\\text{quqd}}^{\\text{Prime}}')

cquqd8 = Parameter(name = 'cquqd8',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cquqd8Re + cquqd8Im*complex(0,1)',
                   texname = 'c_{\\text{quqd}}^8')

cquqd81 = Parameter(name = 'cquqd81',
                    nature = 'internal',
                    type = 'complex',
                    value = 'cquqd81Re + cquqd81Im*complex(0,1)',
                    texname = 'c_{\\text{quqd}}^{8 \\text{Prime}}')

clequ1 = Parameter(name = 'clequ1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'clequ1Re + clequ1Im*complex(0,1)',
                   texname = 'c_{\\text{lequ}}^1')

clequ3 = Parameter(name = 'clequ3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'clequ3Re + clequ3Im*complex(0,1)',
                   texname = 'c_{\\text{lequ}}^3')

vevhat = Parameter(name = 'vevhat',
                   nature = 'internal',
                   type = 'real',
                   value = '1/(2**0.25*cmath.sqrt(Gf))',
                   texname = '\\hat{v}')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = '(Gf*MH**2)/cmath.sqrt(2)',
                texname = '\\text{lam}')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)',
               texname = 'e')

sth2 = Parameter(name = 'sth2',
                 nature = 'internal',
                 type = 'real',
                 value = '0.5 - cmath.sqrt(1 - (2*aEW*cmath.pi*cmath.sqrt(2))/(Gf*MZ**2))/2.',
                 texname = '\\text{sth2}')

CKM1x1 = Parameter(name = 'CKM1x1',
                   nature = 'internal',
                   type = 'complex',
                   value = '1 - CKMlambda**2/2.',
                   texname = '\\text{CKM1x1}')

CKM1x2 = Parameter(name = 'CKM1x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKMlambda',
                   texname = '\\text{CKM1x2}')

CKM1x3 = Parameter(name = 'CKM1x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKMA*CKMlambda**3*(CKMrho - CKMeta*complex(0,1))',
                   texname = '\\text{CKM1x3}')

CKM2x1 = Parameter(name = 'CKM2x1',
                   nature = 'internal',
                   type = 'complex',
                   value = '-CKMlambda',
                   texname = '\\text{CKM2x1}')

CKM2x2 = Parameter(name = 'CKM2x2',
                   nature = 'internal',
                   type = 'complex',
                   value = '1 - CKMlambda**2/2.',
                   texname = '\\text{CKM2x2}')

CKM2x3 = Parameter(name = 'CKM2x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKMA*CKMlambda**2',
                   texname = '\\text{CKM2x3}')

CKM3x1 = Parameter(name = 'CKM3x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKMA*CKMlambda**3*(1 - CKMrho - CKMeta*complex(0,1))',
                   texname = '\\text{CKM3x1}')

CKM3x2 = Parameter(name = 'CKM3x2',
                   nature = 'internal',
                   type = 'complex',
                   value = '-(CKMA*CKMlambda**2)',
                   texname = '\\text{CKM3x2}')

CKM3x3 = Parameter(name = 'CKM3x3',
                   nature = 'internal',
                   type = 'complex',
                   value = '1',
                   texname = '\\text{CKM3x3}')

propCorr = Parameter(name = 'propCorr',
                     nature = 'internal',
                     type = 'real',
                     value = 'abs(linearPropCorrections)/(abs(linearPropCorrections) + 10**-40)',
                     texname = '\\Delta _{\\text{prop.}}')

MZ1 = Parameter(name = 'MZ1',
                nature = 'internal',
                type = 'real',
                value = 'MZ',
                texname = '\\text{MZ}\'')

MH1 = Parameter(name = 'MH1',
                nature = 'internal',
                type = 'real',
                value = 'MH',
                texname = '\\text{MH}\'')

MT1 = Parameter(name = 'MT1',
                nature = 'internal',
                type = 'real',
                value = 'MT',
                texname = '\\text{MT}\'')

WZ1 = Parameter(name = 'WZ1',
                nature = 'internal',
                type = 'real',
                value = 'WZ',
                texname = '\\text{WZ}\'')

WW1 = Parameter(name = 'WW1',
                nature = 'internal',
                type = 'real',
                value = 'WW',
                texname = '\\text{WW}\'')

WH1 = Parameter(name = 'WH1',
                nature = 'internal',
                type = 'real',
                value = 'WH',
                texname = '\\text{WH}\'')

WT1 = Parameter(name = 'WT1',
                nature = 'internal',
                type = 'real',
                value = 'WT',
                texname = '\\text{WT}\'')

gHgg2 = Parameter(name = 'gHgg2',
                  nature = 'internal',
                  type = 'real',
                  value = '(-7*aS)/(720.*cmath.pi)',
                  texname = 'g_{\\text{HGG}}^2')

gHgg4 = Parameter(name = 'gHgg4',
                  nature = 'internal',
                  type = 'real',
                  value = 'aS/(360.*cmath.pi)',
                  texname = 'g_{\\text{HGG}}^4')

gHgg5 = Parameter(name = 'gHgg5',
                  nature = 'internal',
                  type = 'real',
                  value = 'aS/(20.*cmath.pi)',
                  texname = 'g_{\\text{HGG}}^5')

cth = Parameter(name = 'cth',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(1 - sth2)',
                texname = 'c_{\\theta }')

sth = Parameter(name = 'sth',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(sth2)',
                texname = 's_{\\theta }')

yb = Parameter(name = 'yb',
               nature = 'internal',
               type = 'real',
               value = '(ymb*cmath.sqrt(2))/vevhat',
               texname = '\\text{yb}')

yc = Parameter(name = 'yc',
               nature = 'internal',
               type = 'real',
               value = '(ymc*cmath.sqrt(2))/vevhat',
               texname = '\\text{yc}')

ydo = Parameter(name = 'ydo',
                nature = 'internal',
                type = 'real',
                value = '(ymdo*cmath.sqrt(2))/vevhat',
                texname = '\\text{ydo}')

ye = Parameter(name = 'ye',
               nature = 'internal',
               type = 'real',
               value = '(yme*cmath.sqrt(2))/vevhat',
               texname = '\\text{ye}')

ym = Parameter(name = 'ym',
               nature = 'internal',
               type = 'real',
               value = '(ymm*cmath.sqrt(2))/vevhat',
               texname = '\\text{ym}')

ys = Parameter(name = 'ys',
               nature = 'internal',
               type = 'real',
               value = '(yms*cmath.sqrt(2))/vevhat',
               texname = '\\text{ys}')

yt = Parameter(name = 'yt',
               nature = 'internal',
               type = 'real',
               value = '(ymt*cmath.sqrt(2))/vevhat',
               texname = '\\text{yt}')

ytau = Parameter(name = 'ytau',
                 nature = 'internal',
                 type = 'real',
                 value = '(ymtau*cmath.sqrt(2))/vevhat',
                 texname = '\\text{ytau}')

yup = Parameter(name = 'yup',
                nature = 'internal',
                type = 'real',
                value = '(ymup*cmath.sqrt(2))/vevhat',
                texname = '\\text{yup}')

gHgg1 = Parameter(name = 'gHgg1',
                  nature = 'internal',
                  type = 'real',
                  value = 'G**2/(48.*cmath.pi**2)',
                  texname = 'g_{\\text{HGG}}^1')

gHgg3 = Parameter(name = 'gHgg3',
                  nature = 'internal',
                  type = 'real',
                  value = '(aS*G)/(60.*cmath.pi)',
                  texname = 'g_{\\text{HGG}}^3')

dGf = Parameter(name = 'dGf',
                nature = 'internal',
                type = 'real',
                value = '((2*cHl3 - cll1)*vevhat**2)/LambdaSMEFT**2',
                texname = '\\text{dGf}')

dkH = Parameter(name = 'dkH',
                nature = 'internal',
                type = 'real',
                value = '((cHbox - cHDD/4.)*vevhat**2)/LambdaSMEFT**2',
                texname = '\\text{dkH}')

MWsm = Parameter(name = 'MWsm',
                 nature = 'internal',
                 type = 'real',
                 value = 'cth*MZ',
                 texname = '\\text{MWsm}')

vevT = Parameter(name = 'vevT',
                 nature = 'internal',
                 type = 'real',
                 value = '(1 + dGf/2.)*vevhat',
                 texname = '\\text{vevT}')

g1 = Parameter(name = 'g1',
               nature = 'internal',
               type = 'real',
               value = 'ee/cth',
               texname = 'g_1')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = 'ee/sth',
               texname = 'g_w')

dMZ2 = Parameter(name = 'dMZ2',
                 nature = 'internal',
                 type = 'real',
                 value = '((cHDD/2. + 2*cHWB*cth*sth)*vevhat**2)/LambdaSMEFT**2',
                 texname = '\\text{dMZ2}')

dMH2 = Parameter(name = 'dMH2',
                 nature = 'internal',
                 type = 'real',
                 value = '2*dkH - (3*cH*vevhat**2)/(2.*lam*LambdaSMEFT**2)',
                 texname = '\\text{dMH2}')

barlam = Parameter(name = 'barlam',
                   nature = 'internal',
                   type = 'real',
                   value = '(1 - dGf - dMH2)*lam',
                   texname = '\\text{barlam}')

MW1 = Parameter(name = 'MW1',
                nature = 'internal',
                type = 'real',
                value = 'MWsm',
                texname = '\\text{MW}\'')

vev = Parameter(name = 'vev',
                nature = 'internal',
                type = 'real',
                value = '(1 - (3*cH*vevhat**2)/(8.*lam*LambdaSMEFT**2))*vevT',
                texname = '\\text{vev}')

gHaa = Parameter(name = 'gHaa',
                 nature = 'internal',
                 type = 'real',
                 value = '(ee**2*(-1.75 + (4*(0.3333333333333333 + (7*MH**2)/(360.*MT**2)))/3. - (29*MH**6)/(16800.*MWsm**6) - (19*MH**4)/(1680.*MWsm**4) - (11*MH**2)/(120.*MWsm**2)))/(8.*cmath.pi**2)',
                 texname = 'g_{\\text{H$\\gamma \\gamma $}}')

gHza = Parameter(name = 'gHza',
                 nature = 'internal',
                 type = 'real',
                 value = '(ee**2*(((0.4583333333333333 + (29*MH**6)/(100800.*MWsm**6) + (19*MH**4)/(10080.*MWsm**4) + (11*MH**2)/(720.*MWsm**2) + (MH**4*MZ**2)/(2100.*MWsm**6) + (MH**2*MZ**2)/(280.*MWsm**4) + (7*MZ**2)/(180.*MWsm**2) + (67*MH**2*MZ**4)/(100800.*MWsm**6) + (53*MZ**4)/(10080.*MWsm**4) + (43*MZ**6)/(50400.*MWsm**6) - (31*cth**2)/(24.*sth**2) - (29*cth**2*MH**6)/(20160.*MWsm**6*sth**2) - (19*cth**2*MH**4)/(2016.*MWsm**4*sth**2) - (11*cth**2*MH**2)/(144.*MWsm**2*sth**2) - (cth**2*MH**4*MZ**2)/(560.*MWsm**6*sth**2) - (31*cth**2*MH**2*MZ**2)/(2520.*MWsm**4*sth**2) - (cth**2*MZ**2)/(9.*MWsm**2*sth**2) - (43*cth**2*MH**2*MZ**4)/(20160.*MWsm**6*sth**2) - (17*cth**2*MZ**4)/(1120.*MWsm**4*sth**2) - (5*cth**2*MZ**6)/(2016.*MWsm**6*sth**2))*sth)/cth + ((0.3333333333333333 + (7*MH**2)/(360.*MT**2) + (11*MZ**2)/(360.*MT**2))*(0.5 - (4*sth**2)/3.))/(cth*sth)))/(4.*cmath.pi**2)',
                 texname = 'g_{\\text{HZ$\\gamma $}}')

dg1 = Parameter(name = 'dg1',
                nature = 'internal',
                type = 'real',
                value = '(sth**2*(dGf + dMZ2 + (2*cHWB*cth**3*vevhat**2)/(LambdaSMEFT**2*sth)))/(2.*(1 - 2*sth**2))',
                texname = '\\text{dg1}')

dgw = Parameter(name = 'dgw',
                nature = 'internal',
                type = 'real',
                value = '-(cth**2*(dGf + dMZ2 + (2*cHWB*sth**3*vevhat**2)/(cth*LambdaSMEFT**2)))/(2.*(1 - 2*sth**2))',
                texname = '\\text{dgw}')

dWZ = Parameter(name = 'dWZ',
                nature = 'internal',
                type = 'real',
                value = 'WZ*(-1 - (-36*cth*MB*MZ**2*sth*(cdWRe*cth + cdBRe*sth)*(-3 + 4*sth**2)*vevhat*yb*cmath.sqrt(2)*cmath.sqrt(-4*MB**2 + MZ**2) - ee*LambdaSMEFT**2*(2*MZ**3*(27 + 54*dgw - 54*(1 + dg1 + dgw)*sth**2 + 76*(1 + 4*dg1 - 2*dgw)*sth**4 + 152*(-dg1 + dgw)*sth**6) + MZ**2*(9 + 18*dgw - 6*(2 + dg1 + 3*dgw)*sth**2 + 8*(1 + 4*dg1 - 2*dgw)*sth**4 + 16*(-dg1 + dgw)*sth**6)*cmath.sqrt(-4*MB**2 + MZ**2) + MB**2*(-9 - 18*dgw - 6*(4 + 11*dg1 - 3*dgw)*sth**2 + 16*(1 + 4*dg1 - 2*dgw)*sth**4 + 32*(-dg1 + dgw)*sth**6)*cmath.sqrt(-4*MB**2 + MZ**2)) + 2*ee*vevhat**2*(-18*cHl3*MZ**3 - 36*cHq3*MZ**3 + 9*(-3*cHd + cHq1)*MB**2*cmath.sqrt(-4*MB**2 + MZ**2) + 9*cHq3*MB**2*cmath.sqrt(-4*MB**2 + MZ**2) - 9*cHq1*MZ**2*cmath.sqrt(-4*MB**2 + MZ**2) - 9*cHq3*MZ**2*cmath.sqrt(-4*MB**2 + MZ**2) + 3*cHWB*cth*(7*MB**2 - MZ**2)*sth*cmath.sqrt(-4*MB**2 + MZ**2) + 6*sth**2*((2*cHd + 3*cHe + 3*cHl1 + 3*cHl3 - 2*cHq1 + 6*cHq3 - 4*cHu)*MZ**3 + 2*cHd*MB**2*cmath.sqrt(-4*MB**2 + MZ**2) + 2*cHq1*MB**2*cmath.sqrt(-4*MB**2 + MZ**2) + 2*cHq3*MB**2*cmath.sqrt(-4*MB**2 + MZ**2) + cHd*MZ**2*cmath.sqrt(-4*MB**2 + MZ**2) + cHq1*MZ**2*cmath.sqrt(-4*MB**2 + MZ**2) + cHq3*MZ**2*cmath.sqrt(-4*MB**2 + MZ**2)) - 8*cHWB*cth*sth**3*(2*MB**2*cmath.sqrt(-4*MB**2 + MZ**2) + MZ**2*(19*MZ + cmath.sqrt(-4*MB**2 + MZ**2))) + 8*cHWB*cth*sth**5*(2*MB**2*cmath.sqrt(-4*MB**2 + MZ**2) + MZ**2*(19*MZ + cmath.sqrt(-4*MB**2 + MZ**2)))))/(ee*LambdaSMEFT**2*(2*MZ**3*(27 - 54*sth**2 + 76*sth**4) + MZ**2*(9 - 12*sth**2 + 8*sth**4)*cmath.sqrt(-4*MB**2 + MZ**2) + MB**2*(-9 - 24*sth**2 + 16*sth**4)*cmath.sqrt(-4*MB**2 + MZ**2))))',
                texname = '\\text{dWZ}')

g1sh = Parameter(name = 'g1sh',
                 nature = 'internal',
                 type = 'real',
                 value = '(ee*(1 + dg1 - (cHB*vevhat**2)/LambdaSMEFT**2))/cth',
                 texname = 'g_1')

gwsh = Parameter(name = 'gwsh',
                 nature = 'internal',
                 type = 'real',
                 value = '(ee*(1 + dgw - (cHW*vevhat**2)/LambdaSMEFT**2))/sth',
                 texname = 'g_w')

dMW = Parameter(name = 'dMW',
                nature = 'internal',
                type = 'real',
                value = '(dGf/2. + dgw)*MWsm',
                texname = '\\text{dMW}')

dWT = Parameter(name = 'dWT',
                nature = 'internal',
                type = 'real',
                value = 'WT*((dMW*(-(1/(MB + MT - MWsm)) - 4/MWsm + 1/(MB - MT + MWsm) + 1/(-MB + MT + MWsm) + 1/(MB + MT + MWsm) + (4*MWsm*(MB**2 + MT**2 - 4*MWsm**2))/((MB**2 - MT**2)**2 + (MB**2 + MT**2)*MWsm**2 - 2*MWsm**4)))/2. + 2*(dgw + (vevhat*(cHq3*vevhat + (3*MWsm**2*(cHudRe*ee*MB*MT*vevhat*yb*yt + 2*sth*(cdWRe*MB*(-MB**2 + MT**2 + MWsm**2)*yb + cuWRe*MT*(MB**2 - MT**2 + MWsm**2)*yt)*cmath.sqrt(2)))/(ee*((MB**2 - MT**2)**2 + (MB**2 + MT**2)*MWsm**2 - 2*MWsm**4))))/LambdaSMEFT**2))',
                texname = '\\text{dWT}')

dWW = Parameter(name = 'dWW',
                nature = 'internal',
                type = 'real',
                value = '(dMW*WW)/MWsm + (-1 + (3 + 6*dgw + (2*cHl3*vevhat**2)/LambdaSMEFT**2 + (4*cHq3*vevhat**2)/LambdaSMEFT**2)/3.)*WW',
                texname = '\\text{dWW}')

MW = Parameter(name = 'MW',
               nature = 'internal',
               type = 'real',
               value = 'dMW*(1 - linearPropCorrections) + MWsm',
               texname = 'M_W')

dWHc = Parameter(name = 'dWHc',
                 nature = 'internal',
                 type = 'real',
                 value = 'yc/(yc + 10**-40)*(-0.02884*(dGf - 2.*dkH + (2.*cuHRe*vevhat**2)/LambdaSMEFT**2))',
                 texname='\\text{dWHc}')

dWHb = Parameter(name = 'dWHb',
                 nature = 'internal',
                 type = 'real',
                 value = 'yb/(yb + 10**-40)*(-0.5809*(dGf - 2.*dkH + (2.*cdHRe*vevhat**2)/LambdaSMEFT**2))',
                 texname='\\text{dWHb}')

dWHta = Parameter(name = 'dWHta',
                 nature = 'internal',
                 type = 'real',
                 value = 'ytau/(ytau + 10**-40)*(-0.06256*(dGf - 2.*dkH + (2.*ceHRe*vevhat**2)/LambdaSMEFT**2))',
                 texname='\\text{dWHta}')

dWH = Parameter(name = 'dWH',
                nature = 'internal',
                type = 'real',
                value = 'WH*(-0.24161*dGf + 0.96644*dgw + 0.48322*dkH - 0.11186509426655465*dWW - (2.3297576233946*dMW)/MWsm + (0.17608307708657758*cHl3*vevhat**2)/LambdaSMEFT**2 + (0.3641037844923821*cHq3*vevhat**2)/LambdaSMEFT**2 + (0.1636*cHG*MT**2*vevhat**2)/(LambdaSMEFT**2*(-0.5*gHgg2*MH**2 + gHgg1*MT**2)) + (cHW*(-0.3593778511706701*gHaa*gHza + 0.006164*cth*gHaa*sth + 0.00454*gHza*sth**2)*vevhat**2)/(gHaa*gHza*LambdaSMEFT**2) + (cHWB*(-0.00454*cth*gHza*sth + gHaa*(-0.0030819999999999997 + 0.006163999999999999*sth**2))*vevhat**2)/(gHaa*gHza*LambdaSMEFT**2) + (cHB*(-0.006163999999999999*cth*gHaa*sth - 0.00454*gHza*(-1. + sth**2))*vevhat**2)/(gHaa*gHza*LambdaSMEFT**2) + dWHc + dWHb + dWHta)',
                texname = '\\text{dWH}')

