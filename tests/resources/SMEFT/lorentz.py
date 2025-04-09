# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 13.2.1 for Linux x86 (64-bit) (January 27, 2023)
# Date: Fri 28 Apr 2023 20:57:26


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot
try:
   import form_factors as ForFac 
except ImportError:
   pass


UUS1 = Lorentz(name = 'UUS1',
               spins = [ -1, -1, 1 ],
               structure = '1')

UUV1 = Lorentz(name = 'UUV1',
               spins = [ -1, -1, 3 ],
               structure = 'P(3,1)')

SSS1 = Lorentz(name = 'SSS1',
               spins = [ 1, 1, 1 ],
               structure = '1')

SSS2 = Lorentz(name = 'SSS2',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2)')

SSS3 = Lorentz(name = 'SSS3',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2) - P(-1,1)*P(-1,3)')

SSS4 = Lorentz(name = 'SSS4',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3)')

SSS5 = Lorentz(name = 'SSS5',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2) + P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3)')

SSS6 = Lorentz(name = 'SSS6',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)**2 + P(-1,2)**2 + P(-1,3)**2')

SSS7 = Lorentz(name = 'SSS7',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)**2 + (2*P(-1,1)*P(-1,2))/3. + P(-1,2)**2 + P(-1,3)**2')

SSS8 = Lorentz(name = 'SSS8',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)**2 + 2*P(-1,1)*P(-1,2) + P(-1,2)**2 + P(-1,3)**2')

SSS9 = Lorentz(name = 'SSS9',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)**2 + 2*P(-1,1)*P(-1,2) + P(-1,2)**2 - 2*P(-1,1)*P(-1,3) - 2*P(-1,2)*P(-1,3) + P(-1,3)**2')

SSS10 = Lorentz(name = 'SSS10',
                spins = [ 1, 1, 1 ],
                structure = 'P(-1,1)**2 - (2*P(-1,1)*P(-1,2))/3. + P(-1,2)**2 - (2*P(-1,1)*P(-1,3))/3. - (2*P(-1,2)*P(-1,3))/3. + P(-1,3)**2')

SSS11 = Lorentz(name = 'SSS11',
                spins = [ 1, 1, 1 ],
                structure = 'P(-1,1)**2 + (2*P(-1,1)*P(-1,2))/3. + P(-1,2)**2 + (2*P(-1,1)*P(-1,3))/3. + (2*P(-1,2)*P(-1,3))/3. + P(-1,3)**2')

FFS1 = Lorentz(name = 'FFS1',
               spins = [ 2, 2, 1 ],
               structure = 'Gamma5(2,1)')

FFS2 = Lorentz(name = 'FFS2',
               spins = [ 2, 2, 1 ],
               structure = 'Identity(2,1)')

FFS3 = Lorentz(name = 'FFS3',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1)')

FFS4 = Lorentz(name = 'FFS4',
               spins = [ 2, 2, 1 ],
               structure = 'ProjP(2,1)')

FFS5 = Lorentz(name = 'FFS5',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1) + ProjP(2,1)')

FFV1 = Lorentz(name = 'FFV1',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,1)')

FFV2 = Lorentz(name = 'FFV2',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFV3 = Lorentz(name = 'FFV3',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)')

FFV4 = Lorentz(name = 'FFV4',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV5 = Lorentz(name = 'FFV5',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFV6 = Lorentz(name = 'FFV6',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFV7 = Lorentz(name = 'FFV7',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

VSS1 = Lorentz(name = 'VSS1',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2)')

VSS2 = Lorentz(name = 'VSS2',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) - P(1,3)')

VSS3 = Lorentz(name = 'VSS3',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,3)')

VVS1 = Lorentz(name = 'VVS1',
               spins = [ 3, 3, 1 ],
               structure = 'Metric(1,2)')

VVS2 = Lorentz(name = 'VVS2',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVV1 = Lorentz(name = 'VVV1',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2)')

VVV2 = Lorentz(name = 'VVV2',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,2)*Metric(1,2)')

VVV3 = Lorentz(name = 'VVV3',
               spins = [ 3, 3, 3 ],
               structure = 'P(2,1)*Metric(1,3)')

VVV4 = Lorentz(name = 'VVV4',
               spins = [ 3, 3, 3 ],
               structure = 'P(2,3)*Metric(1,3)')

VVV5 = Lorentz(name = 'VVV5',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,2)*Metric(2,3)')

VVV6 = Lorentz(name = 'VVV6',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,3)*Metric(2,3)')

VVV7 = Lorentz(name = 'VVV7',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

SSSS1 = Lorentz(name = 'SSSS1',
                spins = [ 1, 1, 1, 1 ],
                structure = '1')

SSSS2 = Lorentz(name = 'SSSS2',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3) + P(-1,1)*P(-1,4) + P(-1,2)*P(-1,4)')

SSSS3 = Lorentz(name = 'SSSS3',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2) + P(-1,3)*P(-1,4)')

SSSS4 = Lorentz(name = 'SSSS4',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2) - P(-1,1)*P(-1,3) - P(-1,2)*P(-1,4) + P(-1,3)*P(-1,4)')

SSSS5 = Lorentz(name = 'SSSS5',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2) + P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3) + P(-1,1)*P(-1,4) + P(-1,2)*P(-1,4) + P(-1,3)*P(-1,4)')

SSSS6 = Lorentz(name = 'SSSS6',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)**2 + P(-1,2)**2 + P(-1,3)**2 + P(-1,4)**2')

SSSS7 = Lorentz(name = 'SSSS7',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)**2 + P(-1,2)**2 + P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3) + P(-1,3)**2 + P(-1,1)*P(-1,4) + P(-1,2)*P(-1,4) + P(-1,4)**2')

SSSS8 = Lorentz(name = 'SSSS8',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)**2 + (2*P(-1,1)*P(-1,2))/3. + P(-1,2)**2 + (2*P(-1,1)*P(-1,3))/3. + (2*P(-1,2)*P(-1,3))/3. + P(-1,3)**2 + (2*P(-1,1)*P(-1,4))/3. + (2*P(-1,2)*P(-1,4))/3. + (2*P(-1,3)*P(-1,4))/3. + P(-1,4)**2')

SSSS9 = Lorentz(name = 'SSSS9',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)**2 + 2*P(-1,1)*P(-1,2) + P(-1,2)**2 + P(-1,3)**2 + 2*P(-1,3)*P(-1,4) + P(-1,4)**2')

SSSS10 = Lorentz(name = 'SSSS10',
                 spins = [ 1, 1, 1, 1 ],
                 structure = 'P(-1,1)**2 + 2*P(-1,1)*P(-1,2) + P(-1,2)**2 - 2*P(-1,1)*P(-1,3) - 2*P(-1,2)*P(-1,3) + P(-1,3)**2 - 2*P(-1,1)*P(-1,4) - 2*P(-1,2)*P(-1,4) + 2*P(-1,3)*P(-1,4) + P(-1,4)**2')

SSSS11 = Lorentz(name = 'SSSS11',
                 spins = [ 1, 1, 1, 1 ],
                 structure = 'P(-2,3)*P(-2,4)*P(-1,1)*P(-1,2)')

SSSS12 = Lorentz(name = 'SSSS12',
                 spins = [ 1, 1, 1, 1 ],
                 structure = 'P(-2,2)*P(-2,4)*P(-1,1)*P(-1,3) + P(-2,2)*P(-2,3)*P(-1,1)*P(-1,4)')

SSSS13 = Lorentz(name = 'SSSS13',
                 spins = [ 1, 1, 1, 1 ],
                 structure = '-(P(-2,3)*P(-2,4)*P(-1,1)*P(-1,2)) + P(-2,2)*P(-2,4)*P(-1,1)*P(-1,3)')

SSSS14 = Lorentz(name = 'SSSS14',
                 spins = [ 1, 1, 1, 1 ],
                 structure = 'P(-2,3)*P(-2,4)*P(-1,1)*P(-1,2) + P(-2,2)*P(-2,4)*P(-1,1)*P(-1,3) + P(-2,2)*P(-2,3)*P(-1,1)*P(-1,4)')

FFSS1 = Lorentz(name = 'FFSS1',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjM(2,1)')

FFSS2 = Lorentz(name = 'FFSS2',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjP(2,1)')

FFFF1 = Lorentz(name = 'FFFF1',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Gamma(-1,2,-2)*Gamma(-1,4,-3)*ProjM(-3,1)*ProjM(-2,3)')

FFFF2 = Lorentz(name = 'FFFF2',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Gamma(-1,2,-2)*Gamma(-1,4,-3)*ProjM(-3,3)*ProjM(-2,1)')

FFVS1 = Lorentz(name = 'FFVS1',
                spins = [ 2, 2, 3, 1 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS2 = Lorentz(name = 'FFVS2',
                spins = [ 2, 2, 3, 1 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVV1 = Lorentz(name = 'FFVV1',
                spins = [ 2, 2, 3, 3 ],
                structure = '-(Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVV2 = Lorentz(name = 'FFVV2',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVV3 = Lorentz(name = 'FFVV3',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

VSSS1 = Lorentz(name = 'VSSS1',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2)')

VSSS2 = Lorentz(name = 'VSSS2',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,3)')

VSSS3 = Lorentz(name = 'VSSS3',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,4)')

VSSS4 = Lorentz(name = 'VSSS4',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(-1,2)*P(-1,3)*P(1,4)')

VSSS5 = Lorentz(name = 'VSSS5',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(-1,2)*P(-1,4)*P(1,3)')

VSSS6 = Lorentz(name = 'VSSS6',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(-1,3)*P(-1,4)*P(1,2)')

VVSS1 = Lorentz(name = 'VVSS1',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(1,4)*P(2,3)')

VVSS2 = Lorentz(name = 'VVSS2',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(1,3)*P(2,4)')

VVSS3 = Lorentz(name = 'VVSS3',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(1,4)*P(2,3) + P(1,3)*P(2,4)')

VVSS4 = Lorentz(name = 'VVSS4',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Metric(1,2)')

VVSS5 = Lorentz(name = 'VVSS5',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(-1,3)*P(-1,4)*Metric(1,2)')

VVSS6 = Lorentz(name = 'VVSS6',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVSS7 = Lorentz(name = 'VVSS7',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(1,4)*P(2,3) - P(-1,3)*P(-1,4)*Metric(1,2)')

VVSS8 = Lorentz(name = 'VVSS8',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(1,4)*P(2,3) + P(-1,3)*P(-1,4)*Metric(1,2)')

VVSS9 = Lorentz(name = 'VVSS9',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(1,3)*P(2,4) + P(-1,3)*P(-1,4)*Metric(1,2)')

VVSS10 = Lorentz(name = 'VVSS10',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,4)*P(2,3) + P(1,3)*P(2,4) + P(-1,3)*P(-1,4)*Metric(1,2)')

VVVS1 = Lorentz(name = 'VVVS1',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,1)*Metric(1,2)')

VVVS2 = Lorentz(name = 'VVVS2',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,2)*Metric(1,2)')

VVVS3 = Lorentz(name = 'VVVS3',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,4)*Metric(1,2)')

VVVS4 = Lorentz(name = 'VVVS4',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(2,1)*Metric(1,3)')

VVVS5 = Lorentz(name = 'VVVS5',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(2,3)*Metric(1,3)')

VVVS6 = Lorentz(name = 'VVVS6',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(2,4)*Metric(1,3)')

VVVS7 = Lorentz(name = 'VVVS7',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(1,2)*Metric(2,3)')

VVVS8 = Lorentz(name = 'VVVS8',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(1,3)*Metric(2,3)')

VVVS9 = Lorentz(name = 'VVVS9',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(1,4)*Metric(2,3)')

VVVV1 = Lorentz(name = 'VVVV1',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3)')

VVVV2 = Lorentz(name = 'VVVV2',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,3)*Metric(2,4)')

VVVV3 = Lorentz(name = 'VVVV3',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV4 = Lorentz(name = 'VVVV4',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,2)*Metric(3,4)')

VVVV5 = Lorentz(name = 'VVVV5',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV6 = Lorentz(name = 'VVVV6',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

SSSSS1 = Lorentz(name = 'SSSSS1',
                 spins = [ 1, 1, 1, 1, 1 ],
                 structure = '1')

FFSSS1 = Lorentz(name = 'FFSSS1',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjM(2,1)')

FFSSS2 = Lorentz(name = 'FFSSS2',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjP(2,1)')

FFVVS1 = Lorentz(name = 'FFVVS1',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = '-(Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVVS2 = Lorentz(name = 'FFVVS2',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVVS3 = Lorentz(name = 'FFVVS3',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

VSSSS1 = Lorentz(name = 'VSSSS1',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2)')

VSSSS2 = Lorentz(name = 'VSSSS2',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,3)')

VSSSS3 = Lorentz(name = 'VSSSS3',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,4)')

VSSSS4 = Lorentz(name = 'VSSSS4',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(-1,2)*P(-1,3)*P(1,4)')

VSSSS5 = Lorentz(name = 'VSSSS5',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(-1,2)*P(-1,4)*P(1,3)')

VSSSS6 = Lorentz(name = 'VSSSS6',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(-1,3)*P(-1,4)*P(1,2)')

VSSSS7 = Lorentz(name = 'VSSSS7',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) - P(1,5)')

VSSSS8 = Lorentz(name = 'VSSSS8',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3) - P(1,4) - P(1,5)')

VSSSS9 = Lorentz(name = 'VSSSS9',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,5)')

VSSSS10 = Lorentz(name = 'VSSSS10',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,2)*P(-1,3)*P(1,5)')

VSSSS11 = Lorentz(name = 'VSSSS11',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,2)*P(-1,4)*P(1,5)')

VSSSS12 = Lorentz(name = 'VSSSS12',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,3)*P(-1,4)*P(1,5)')

VSSSS13 = Lorentz(name = 'VSSSS13',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,2)*P(-1,3)*P(1,4) - P(-1,2)*P(-1,3)*P(1,5)')

VSSSS14 = Lorentz(name = 'VSSSS14',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,2)*P(-1,5)*P(1,3)')

VSSSS15 = Lorentz(name = 'VSSSS15',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,3)*P(-1,5)*P(1,2)')

VSSSS16 = Lorentz(name = 'VSSSS16',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,2)*P(-1,5)*P(1,4)')

VSSSS17 = Lorentz(name = 'VSSSS17',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,3)*P(-1,5)*P(1,4)')

VSSSS18 = Lorentz(name = 'VSSSS18',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,4)*P(-1,5)*P(1,2)')

VSSSS19 = Lorentz(name = 'VSSSS19',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,4)*P(-1,5)*P(1,3)')

VSSSS20 = Lorentz(name = 'VSSSS20',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,3)*P(-1,4)*P(1,2) - P(-1,3)*P(-1,5)*P(1,2) + P(-1,2)*P(-1,4)*P(1,3) - P(-1,2)*P(-1,5)*P(1,3)')

VSSSS21 = Lorentz(name = 'VSSSS21',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,3)*P(-1,4)*P(1,2) + P(-1,3)*P(-1,5)*P(1,2) + P(-1,2)*P(-1,4)*P(1,3) + P(-1,2)*P(-1,5)*P(1,3) - P(-1,2)*P(-1,5)*P(1,4) - P(-1,3)*P(-1,5)*P(1,4) - P(-1,2)*P(-1,4)*P(1,5) - P(-1,3)*P(-1,4)*P(1,5)')

VSSSS22 = Lorentz(name = 'VSSSS22',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,2)*P(-1,5)*P(1,4) - P(-1,3)*P(-1,5)*P(1,4) + P(-1,2)*P(-1,4)*P(1,5) - P(-1,3)*P(-1,4)*P(1,5)')

VSSSS23 = Lorentz(name = 'VSSSS23',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = '-(P(-1,3)*P(-1,5)*P(1,2)) - P(-1,4)*P(-1,5)*P(1,2) + P(-1,2)*P(-1,3)*P(1,5) + P(-1,2)*P(-1,4)*P(1,5)')

VSSSS24 = Lorentz(name = 'VSSSS24',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(-1,4)*P(-1,5)*P(1,2) - P(-1,4)*P(-1,5)*P(1,3)')

VVSSS1 = Lorentz(name = 'VVSSS1',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'P(1,4)*P(2,3)')

VVSSS2 = Lorentz(name = 'VVSSS2',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'P(1,3)*P(2,4)')

VVSSS3 = Lorentz(name = 'VVSSS3',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'P(1,5)*P(2,3)')

VVSSS4 = Lorentz(name = 'VVSSS4',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'P(1,5)*P(2,4)')

VVSSS5 = Lorentz(name = 'VVSSS5',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'P(1,3)*P(2,5)')

VVSSS6 = Lorentz(name = 'VVSSS6',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'P(1,4)*P(2,5)')

VVSSS7 = Lorentz(name = 'VVSSS7',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'Metric(1,2)')

VVSSS8 = Lorentz(name = 'VVSSS8',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'P(-1,3)*P(-1,4)*Metric(1,2)')

VVSSS9 = Lorentz(name = 'VVSSS9',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'P(-1,3)*P(-1,5)*Metric(1,2)')

VVSSS10 = Lorentz(name = 'VVSSS10',
                  spins = [ 3, 3, 1, 1, 1 ],
                  structure = 'P(-1,4)*P(-1,5)*Metric(1,2)')

VVVSS1 = Lorentz(name = 'VVVSS1',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,1)*Metric(1,2)')

VVVSS2 = Lorentz(name = 'VVVSS2',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,2)*Metric(1,2)')

VVVSS3 = Lorentz(name = 'VVVSS3',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,4)*Metric(1,2)')

VVVSS4 = Lorentz(name = 'VVVSS4',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,5)*Metric(1,2)')

VVVSS5 = Lorentz(name = 'VVVSS5',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(2,1)*Metric(1,3)')

VVVSS6 = Lorentz(name = 'VVVSS6',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(2,3)*Metric(1,3)')

VVVSS7 = Lorentz(name = 'VVVSS7',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(2,4)*Metric(1,3)')

VVVSS8 = Lorentz(name = 'VVVSS8',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(2,5)*Metric(1,3)')

VVVSS9 = Lorentz(name = 'VVVSS9',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(1,2)*Metric(2,3)')

VVVSS10 = Lorentz(name = 'VVVSS10',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(1,3)*Metric(2,3)')

VVVSS11 = Lorentz(name = 'VVVSS11',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(1,4)*Metric(2,3)')

VVVSS12 = Lorentz(name = 'VVVSS12',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(1,5)*Metric(2,3)')

VVVVS1 = Lorentz(name = 'VVVVS1',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3)')

VVVVS2 = Lorentz(name = 'VVVVS2',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,3)*Metric(2,4)')

VVVVS3 = Lorentz(name = 'VVVVS3',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,2)*Metric(3,4)')

SSSSSS1 = Lorentz(name = 'SSSSSS1',
                  spins = [ 1, 1, 1, 1, 1, 1 ],
                  structure = '1')

VVSSSS1 = Lorentz(name = 'VVSSSS1',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'P(1,4)*P(2,3)')

VVSSSS2 = Lorentz(name = 'VVSSSS2',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'P(1,3)*P(2,4)')

VVSSSS3 = Lorentz(name = 'VVSSSS3',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'P(1,4)*P(2,3) + P(1,3)*P(2,4)')

VVSSSS4 = Lorentz(name = 'VVSSSS4',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'P(1,5)*P(2,3)')

VVSSSS5 = Lorentz(name = 'VVSSSS5',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'P(1,5)*P(2,4)')

VVSSSS6 = Lorentz(name = 'VVSSSS6',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'P(1,3)*P(2,5)')

VVSSSS7 = Lorentz(name = 'VVSSSS7',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'P(1,4)*P(2,5)')

VVSSSS8 = Lorentz(name = 'VVSSSS8',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'P(1,6)*P(2,3)')

VVSSSS9 = Lorentz(name = 'VVSSSS9',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'P(1,6)*P(2,4)')

VVSSSS10 = Lorentz(name = 'VVSSSS10',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(1,6)*P(2,5)')

VVSSSS11 = Lorentz(name = 'VVSSSS11',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(1,3)*P(2,6)')

VVSSSS12 = Lorentz(name = 'VVSSSS12',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(1,4)*P(2,6)')

VVSSSS13 = Lorentz(name = 'VVSSSS13',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(1,5)*P(2,6)')

VVSSSS14 = Lorentz(name = 'VVSSSS14',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(1,6)*P(2,5) + P(1,5)*P(2,6)')

VVSSSS15 = Lorentz(name = 'VVSSSS15',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'Metric(1,2)')

VVSSSS16 = Lorentz(name = 'VVSSSS16',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(-1,3)*P(-1,4)*Metric(1,2)')

VVSSSS17 = Lorentz(name = 'VVSSSS17',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(-1,3)*P(-1,5)*Metric(1,2)')

VVSSSS18 = Lorentz(name = 'VVSSSS18',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(-1,4)*P(-1,5)*Metric(1,2)')

VVSSSS19 = Lorentz(name = 'VVSSSS19',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(-1,3)*P(-1,6)*Metric(1,2)')

VVSSSS20 = Lorentz(name = 'VVSSSS20',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(-1,4)*P(-1,6)*Metric(1,2)')

VVSSSS21 = Lorentz(name = 'VVSSSS21',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(-1,5)*P(-1,6)*Metric(1,2)')

VVSSSS22 = Lorentz(name = 'VVSSSS22',
                   spins = [ 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(1,4)*P(2,3) - (P(1,5)*P(2,3))/2. - (P(1,6)*P(2,3))/2. + P(1,3)*P(2,4) - (P(1,5)*P(2,4))/2. - (P(1,6)*P(2,4))/2. - (P(1,3)*P(2,5))/2. - (P(1,4)*P(2,5))/2. + P(1,6)*P(2,5) - (P(1,3)*P(2,6))/2. - (P(1,4)*P(2,6))/2. + P(1,5)*P(2,6) - P(-1,3)*P(-1,5)*Metric(1,2) - P(-1,4)*P(-1,5)*Metric(1,2) - P(-1,3)*P(-1,6)*Metric(1,2) - P(-1,4)*P(-1,6)*Metric(1,2)')

VVVSSS1 = Lorentz(name = 'VVVSSS1',
                  spins = [ 3, 3, 3, 1, 1, 1 ],
                  structure = 'P(3,4)*Metric(1,2)')

VVVSSS2 = Lorentz(name = 'VVVSSS2',
                  spins = [ 3, 3, 3, 1, 1, 1 ],
                  structure = 'P(3,5)*Metric(1,2)')

VVVSSS3 = Lorentz(name = 'VVVSSS3',
                  spins = [ 3, 3, 3, 1, 1, 1 ],
                  structure = 'P(3,6)*Metric(1,2)')

VVVSSS4 = Lorentz(name = 'VVVSSS4',
                  spins = [ 3, 3, 3, 1, 1, 1 ],
                  structure = 'P(2,4)*Metric(1,3)')

VVVSSS5 = Lorentz(name = 'VVVSSS5',
                  spins = [ 3, 3, 3, 1, 1, 1 ],
                  structure = 'P(2,5)*Metric(1,3)')

VVVSSS6 = Lorentz(name = 'VVVSSS6',
                  spins = [ 3, 3, 3, 1, 1, 1 ],
                  structure = 'P(2,6)*Metric(1,3)')

VVVSSS7 = Lorentz(name = 'VVVSSS7',
                  spins = [ 3, 3, 3, 1, 1, 1 ],
                  structure = 'P(1,4)*Metric(2,3)')

VVVSSS8 = Lorentz(name = 'VVVSSS8',
                  spins = [ 3, 3, 3, 1, 1, 1 ],
                  structure = 'P(1,5)*Metric(2,3)')

VVVSSS9 = Lorentz(name = 'VVVSSS9',
                  spins = [ 3, 3, 3, 1, 1, 1 ],
                  structure = 'P(1,6)*Metric(2,3)')

VVVVSS1 = Lorentz(name = 'VVVVSS1',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3)')

VVVVSS2 = Lorentz(name = 'VVVVSS2',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,3)*Metric(2,4)')

VVVVSS3 = Lorentz(name = 'VVVVSS3',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,2)*Metric(3,4)')

SSSSSSS1 = Lorentz(name = 'SSSSSSS1',
                   spins = [ 1, 1, 1, 1, 1, 1, 1 ],
                   structure = '1')

VVVSSSS1 = Lorentz(name = 'VVVSSSS1',
                   spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(3,4)*Metric(1,2)')

VVVSSSS2 = Lorentz(name = 'VVVSSSS2',
                   spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(3,5)*Metric(1,2)')

VVVSSSS3 = Lorentz(name = 'VVVSSSS3',
                   spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(3,6)*Metric(1,2)')

VVVSSSS4 = Lorentz(name = 'VVVSSSS4',
                   spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(3,7)*Metric(1,2)')

VVVSSSS5 = Lorentz(name = 'VVVSSSS5',
                   spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(2,4)*Metric(1,3)')

VVVSSSS6 = Lorentz(name = 'VVVSSSS6',
                   spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(2,5)*Metric(1,3)')

VVVSSSS7 = Lorentz(name = 'VVVSSSS7',
                   spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(2,6)*Metric(1,3)')

VVVSSSS8 = Lorentz(name = 'VVVSSSS8',
                   spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(2,7)*Metric(1,3)')

VVVSSSS9 = Lorentz(name = 'VVVSSSS9',
                   spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                   structure = 'P(1,4)*Metric(2,3)')

VVVSSSS10 = Lorentz(name = 'VVVSSSS10',
                    spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                    structure = 'P(1,5)*Metric(2,3)')

VVVSSSS11 = Lorentz(name = 'VVVSSSS11',
                    spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                    structure = 'P(1,6)*Metric(2,3)')

VVVSSSS12 = Lorentz(name = 'VVVSSSS12',
                    spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                    structure = 'P(1,7)*Metric(2,3)')

VVVSSSS13 = Lorentz(name = 'VVVSSSS13',
                    spins = [ 3, 3, 3, 1, 1, 1, 1 ],
                    structure = 'P(3,4)*Metric(1,2) + P(3,5)*Metric(1,2) - P(3,6)*Metric(1,2) - P(3,7)*Metric(1,2) + P(2,4)*Metric(1,3) + P(2,5)*Metric(1,3) - P(2,6)*Metric(1,3) - P(2,7)*Metric(1,3) + P(1,4)*Metric(2,3) + P(1,5)*Metric(2,3) - P(1,6)*Metric(2,3) - P(1,7)*Metric(2,3)')

VVVVSSS1 = Lorentz(name = 'VVVVSSS1',
                   spins = [ 3, 3, 3, 3, 1, 1, 1 ],
                   structure = 'Metric(1,4)*Metric(2,3)')

VVVVSSS2 = Lorentz(name = 'VVVVSSS2',
                   spins = [ 3, 3, 3, 3, 1, 1, 1 ],
                   structure = 'Metric(1,3)*Metric(2,4)')

VVVVSSS3 = Lorentz(name = 'VVVVSSS3',
                   spins = [ 3, 3, 3, 3, 1, 1, 1 ],
                   structure = 'Metric(1,2)*Metric(3,4)')

SSSSSSSS1 = Lorentz(name = 'SSSSSSSS1',
                    spins = [ 1, 1, 1, 1, 1, 1, 1, 1 ],
                    structure = '1')

VVVVSSSS1 = Lorentz(name = 'VVVVSSSS1',
                    spins = [ 3, 3, 3, 3, 1, 1, 1, 1 ],
                    structure = 'Metric(1,4)*Metric(2,3)')

VVVVSSSS2 = Lorentz(name = 'VVVVSSSS2',
                    spins = [ 3, 3, 3, 3, 1, 1, 1, 1 ],
                    structure = 'Metric(1,3)*Metric(2,4)')

VVVVSSSS3 = Lorentz(name = 'VVVVSSSS3',
                    spins = [ 3, 3, 3, 3, 1, 1, 1, 1 ],
                    structure = 'Metric(1,2)*Metric(3,4)')

VVVVSSSS4 = Lorentz(name = 'VVVVSSSS4',
                    spins = [ 3, 3, 3, 3, 1, 1, 1, 1 ],
                    structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) + Metric(1,2)*Metric(3,4)')

