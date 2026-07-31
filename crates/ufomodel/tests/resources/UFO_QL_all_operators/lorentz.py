# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.0.0 for Mac OS X ARM (64-bit) (December 13, 2023)
# Date: Mon 26 Jan 2026 15:01:52


from .object_library import all_lorentz, Lorentz

from .function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot
try:
   from . import form_factors as ForFac
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
               structure = 'P(-1,1)**2')

SSS3 = Lorentz(name = 'SSS3',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2)')

SSS4 = Lorentz(name = 'SSS4',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,2)**2')

SSS5 = Lorentz(name = 'SSS5',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,3)')

SSS6 = Lorentz(name = 'SSS6',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,2)*P(-1,3)')

SSS7 = Lorentz(name = 'SSS7',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,3)**2')

FFS1 = Lorentz(name = 'FFS1',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1)')

FFS2 = Lorentz(name = 'FFS2',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS3 = Lorentz(name = 'FFS3',
               spins = [ 2, 2, 1 ],
               structure = 'ProjP(2,1)')

FFS4 = Lorentz(name = 'FFS4',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1)')

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
               structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV6 = Lorentz(name = 'FFV6',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,-2,-3)*Gamma(3,2,-2)*ProjM(-3,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)')

FFV7 = Lorentz(name = 'FFV7',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFV8 = Lorentz(name = 'FFV8',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1) + Gamma(3,2,-1)*ProjP(-1,1)')

FFV9 = Lorentz(name = 'FFV9',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFV10 = Lorentz(name = 'FFV10',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV11 = Lorentz(name = 'FFV11',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV12 = Lorentz(name = 'FFV12',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,-3)*ProjP(-3,1)')

FFV13 = Lorentz(name = 'FFV13',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,-2,-3)*Gamma(3,2,-2)*ProjP(-3,1)')

FFV14 = Lorentz(name = 'FFV14',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,-2,-3)*Gamma(3,2,-2)*ProjP(-3,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

VSS1 = Lorentz(name = 'VSS1',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2)')

VSS2 = Lorentz(name = 'VSS2',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) - P(1,3)')

VSS3 = Lorentz(name = 'VSS3',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) + P(1,3)/7.')

VSS4 = Lorentz(name = 'VSS4',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) + P(1,3)/3.')

VSS5 = Lorentz(name = 'VSS5',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) + (3*P(1,3))/5.')

VSS6 = Lorentz(name = 'VSS6',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,3)')

VSS7 = Lorentz(name = 'VSS7',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) + 3*P(1,3)')

VVS1 = Lorentz(name = 'VVS1',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,2)*P(2,1)')

VVS2 = Lorentz(name = 'VVS2',
               spins = [ 3, 3, 1 ],
               structure = 'Metric(1,2)')

VVS3 = Lorentz(name = 'VVS3',
               spins = [ 3, 3, 1 ],
               structure = 'P(-1,1)*P(-1,2)*Metric(1,2)')

VVV1 = Lorentz(name = 'VVV1',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,3)*P(2,1)*P(3,2)')

VVV2 = Lorentz(name = 'VVV2',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,2)*P(2,3)*P(3,1)')

VVV3 = Lorentz(name = 'VVV3',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2)')

VVV4 = Lorentz(name = 'VVV4',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,2)*Metric(1,2)')

VVV5 = Lorentz(name = 'VVV5',
               spins = [ 3, 3, 3 ],
               structure = 'P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2)')

VVV6 = Lorentz(name = 'VVV6',
               spins = [ 3, 3, 3 ],
               structure = 'P(-1,2)*P(-1,3)*P(3,1)*Metric(1,2)')

VVV7 = Lorentz(name = 'VVV7',
               spins = [ 3, 3, 3 ],
               structure = 'P(2,1)*Metric(1,3)')

VVV8 = Lorentz(name = 'VVV8',
               spins = [ 3, 3, 3 ],
               structure = 'P(2,3)*Metric(1,3)')

VVV9 = Lorentz(name = 'VVV9',
               spins = [ 3, 3, 3 ],
               structure = 'P(-1,1)*P(-1,2)*P(2,3)*Metric(1,3)')

VVV10 = Lorentz(name = 'VVV10',
                spins = [ 3, 3, 3 ],
                structure = 'P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3)')

VVV11 = Lorentz(name = 'VVV11',
                spins = [ 3, 3, 3 ],
                structure = 'P(1,2)*Metric(2,3)')

VVV12 = Lorentz(name = 'VVV12',
                spins = [ 3, 3, 3 ],
                structure = 'P(1,3)*Metric(2,3)')

VVV13 = Lorentz(name = 'VVV13',
                spins = [ 3, 3, 3 ],
                structure = 'P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3)')

VVV14 = Lorentz(name = 'VVV14',
                spins = [ 3, 3, 3 ],
                structure = 'P(-1,1)*P(-1,3)*P(1,2)*Metric(2,3)')

SSSS1 = Lorentz(name = 'SSSS1',
                spins = [ 1, 1, 1, 1 ],
                structure = '1')

SSSS2 = Lorentz(name = 'SSSS2',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)**2')

SSSS3 = Lorentz(name = 'SSSS3',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2)')

SSSS4 = Lorentz(name = 'SSSS4',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,2)**2')

SSSS5 = Lorentz(name = 'SSSS5',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,3)')

SSSS6 = Lorentz(name = 'SSSS6',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,2)*P(-1,3)')

SSSS7 = Lorentz(name = 'SSSS7',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,3)**2')

SSSS8 = Lorentz(name = 'SSSS8',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,4)')

SSSS9 = Lorentz(name = 'SSSS9',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,2)*P(-1,4)')

SSSS10 = Lorentz(name = 'SSSS10',
                 spins = [ 1, 1, 1, 1 ],
                 structure = 'P(-1,3)*P(-1,4)')

SSSS11 = Lorentz(name = 'SSSS11',
                 spins = [ 1, 1, 1, 1 ],
                 structure = 'P(-1,4)**2')

FFSS1 = Lorentz(name = 'FFSS1',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjM(2,1)')

FFSS2 = Lorentz(name = 'FFSS2',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS3 = Lorentz(name = 'FFSS3',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS4 = Lorentz(name = 'FFSS4',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS5 = Lorentz(name = 'FFSS5',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjP(2,1)')

FFSS6 = Lorentz(name = 'FFSS6',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFSS7 = Lorentz(name = 'FFSS7',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,4)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFSS8 = Lorentz(name = 'FFSS8',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFVS1 = Lorentz(name = 'FFVS1',
                spins = [ 2, 2, 3, 1 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFVS2 = Lorentz(name = 'FFVS2',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)')

FFVS3 = Lorentz(name = 'FFVS3',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS4 = Lorentz(name = 'FFVS4',
                spins = [ 2, 2, 3, 1 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS5 = Lorentz(name = 'FFVS5',
                spins = [ 2, 2, 3, 1 ],
                structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFVS6 = Lorentz(name = 'FFVS6',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFVS7 = Lorentz(name = 'FFVS7',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS8 = Lorentz(name = 'FFVS8',
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
                structure = '-(Gamma(3,-1,-2)*Gamma(4,2,-1)*ProjM(-2,1)) + Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)')

FFVV4 = Lorentz(name = 'FFVV4',
                spins = [ 2, 2, 3, 3 ],
                structure = '-(Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV5 = Lorentz(name = 'FFVV5',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

VSSS1 = Lorentz(name = 'VSSS1',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2)')

VSSS2 = Lorentz(name = 'VSSS2',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) - P(1,3)')

VSSS3 = Lorentz(name = 'VSSS3',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,3)')

VSSS4 = Lorentz(name = 'VSSS4',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) + P(1,3)')

VSSS5 = Lorentz(name = 'VSSS5',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) + P(1,3) - 2*P(1,4)')

VSSS6 = Lorentz(name = 'VSSS6',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) - P(1,4)/2.')

VSSS7 = Lorentz(name = 'VSSS7',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) - P(1,3)/3. - P(1,4)/3.')

VSSS8 = Lorentz(name = 'VSSS8',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,4)')

VSSS9 = Lorentz(name = 'VSSS9',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2) + P(1,3) + P(1,4)')

VVSS1 = Lorentz(name = 'VVSS1',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(1,2)*P(2,1)')

VVSS2 = Lorentz(name = 'VVSS2',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Metric(1,2)')

VVSS3 = Lorentz(name = 'VVSS3',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2)*Metric(1,2)')

VVVS1 = Lorentz(name = 'VVVS1',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,1)*Metric(1,2)')

VVVS2 = Lorentz(name = 'VVVS2',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,2)*Metric(1,2)')

VVVS3 = Lorentz(name = 'VVVS3',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(2,1)*Metric(1,3)')

VVVS4 = Lorentz(name = 'VVVS4',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(2,3)*Metric(1,3)')

VVVS5 = Lorentz(name = 'VVVS5',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3)')

VVVS6 = Lorentz(name = 'VVVS6',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(1,2)*Metric(2,3)')

VVVS7 = Lorentz(name = 'VVVS7',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(1,3)*Metric(2,3)')

VVVS8 = Lorentz(name = 'VVVS8',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) - P(1,2)*Metric(2,3)')

VVVS9 = Lorentz(name = 'VVVS9',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + (3*P(2,3)*Metric(1,3))/2. + P(1,2)*Metric(2,3) - (3*P(1,3)*Metric(2,3))/2.')

VVVS10 = Lorentz(name = 'VVVS10',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(2,3)*Metric(1,3) - P(1,3)*Metric(2,3)')

VVVS11 = Lorentz(name = 'VVVS11',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,2)*Metric(1,2) + P(2,3)*Metric(1,3) - P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVS12 = Lorentz(name = 'VVVS12',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVS13 = Lorentz(name = 'VVVS13',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - (2*P(3,2)*Metric(1,2))/3. - P(2,1)*Metric(1,3) + (2*P(2,3)*Metric(1,3))/3. + (2*P(1,2)*Metric(2,3))/3. - (2*P(1,3)*Metric(2,3))/3.')

VVVS14 = Lorentz(name = 'VVVS14',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3) - P(2,3)*Metric(1,3) + P(1,3)*Metric(2,3)')

VVVV1 = Lorentz(name = 'VVVV1',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3)')

VVVV2 = Lorentz(name = 'VVVV2',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,3)*Metric(2,4)')

VVVV3 = Lorentz(name = 'VVVV3',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,2)*Metric(3,4)')

VVVV4 = Lorentz(name = 'VVVV4',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVV5 = Lorentz(name = 'VVVV5',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVV6 = Lorentz(name = 'VVVV6',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,2)*P(4,1)*Metric(1,2) - P(3,1)*P(4,2)*Metric(1,2) + P(3,4)*P(4,2)*Metric(1,2) + P(3,1)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) + P(2,1)*P(4,2)*Metric(1,3) - P(2,1)*P(4,3)*Metric(1,3) + P(2,4)*P(4,3)*Metric(1,3) - P(2,3)*P(3,1)*Metric(1,4) - P(2,1)*P(3,2)*Metric(1,4) - P(2,4)*P(3,2)*Metric(1,4) - P(2,3)*P(3,4)*Metric(1,4) - P(1,2)*P(4,1)*Metric(2,3) - P(1,3)*P(4,1)*Metric(2,3) - P(1,4)*P(4,2)*Metric(2,3) - P(1,4)*P(4,3)*Metric(2,3) + P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(1,2)*P(3,1)*Metric(2,4) + P(1,4)*P(3,2)*Metric(2,4) - P(1,2)*P(3,4)*Metric(2,4) + P(1,3)*P(3,4)*Metric(2,4) - P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4) - P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,3)*P(2,1)*Metric(3,4) + P(1,4)*P(2,3)*Metric(3,4) + P(1,2)*P(2,4)*Metric(3,4) - P(1,3)*P(2,4)*Metric(3,4) - P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV7 = Lorentz(name = 'VVVV7',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,4)*P(4,1)*Metric(1,2) + P(3,4)*P(4,2)*Metric(1,2) + P(3,1)*P(4,3)*Metric(1,2) + P(3,2)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) - P(2,4)*P(4,1)*Metric(1,3) - P(2,3)*P(4,2)*Metric(1,3) - P(2,1)*P(4,3)*Metric(1,3) - P(2,3)*P(3,1)*Metric(1,4) + P(2,4)*P(3,1)*Metric(1,4) - P(2,4)*P(3,2)*Metric(1,4) - P(2,1)*P(3,4)*Metric(1,4) - P(1,3)*P(4,1)*Metric(2,3) + P(1,3)*P(4,2)*Metric(2,3) - P(1,4)*P(4,2)*Metric(2,3) - P(1,2)*P(4,3)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3) - P(1,4)*P(3,1)*Metric(2,4) - P(1,3)*P(3,2)*Metric(2,4) + P(1,4)*P(3,2)*Metric(2,4) - P(1,2)*P(3,4)*Metric(2,4) + P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4) + P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,3)*P(2,1)*Metric(3,4) + P(1,4)*P(2,1)*Metric(3,4) + P(1,2)*P(2,3)*Metric(3,4) + P(1,2)*P(2,4)*Metric(3,4) - P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

SSSSS1 = Lorentz(name = 'SSSSS1',
                 spins = [ 1, 1, 1, 1, 1 ],
                 structure = '1')

FFSSS1 = Lorentz(name = 'FFSSS1',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjM(2,1)')

FFSSS2 = Lorentz(name = 'FFSSS2',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjP(2,1)')

FFVSS1 = Lorentz(name = 'FFVSS1',
                 spins = [ 2, 2, 3, 1, 1 ],
                 structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFVSS2 = Lorentz(name = 'FFVSS2',
                 spins = [ 2, 2, 3, 1, 1 ],
                 structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFVVS1 = Lorentz(name = 'FFVVS1',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = '-(Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVVS2 = Lorentz(name = 'FFVVS2',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVVS3 = Lorentz(name = 'FFVVS3',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = '-(Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVVS4 = Lorentz(name = 'FFVVS4',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVVS5 = Lorentz(name = 'FFVVS5',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = '-(Gamma(3,2,-1)*Gamma(4,-1,-2)*ProjP(-2,1)) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

VSSSS1 = Lorentz(name = 'VSSSS1',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2)')

VSSSS2 = Lorentz(name = 'VSSSS2',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) - P(1,3)')

VSSSS3 = Lorentz(name = 'VSSSS3',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,3)')

VSSSS4 = Lorentz(name = 'VSSSS4',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3)')

VSSSS5 = Lorentz(name = 'VSSSS5',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3) - 2*P(1,4)')

VSSSS6 = Lorentz(name = 'VSSSS6',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,4)')

VSSSS7 = Lorentz(name = 'VSSSS7',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3) + P(1,4) - 3*P(1,5)')

VSSSS8 = Lorentz(name = 'VSSSS8',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3) - 2*P(1,5)')

VSSSS9 = Lorentz(name = 'VSSSS9',
                 spins = [ 3, 1, 1, 1, 1 ],
                 structure = 'P(1,3) + P(1,4) - 2*P(1,5)')

VSSSS10 = Lorentz(name = 'VSSSS10',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) - P(1,5)')

VSSSS11 = Lorentz(name = 'VSSSS11',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) + P(1,3) - P(1,4) - P(1,5)')

VSSSS12 = Lorentz(name = 'VSSSS12',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,4) - P(1,5)')

VSSSS13 = Lorentz(name = 'VSSSS13',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) - P(1,4)/2. - P(1,5)/2.')

VSSSS14 = Lorentz(name = 'VSSSS14',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) - P(1,3)/3. - P(1,4)/3. - P(1,5)/3.')

VSSSS15 = Lorentz(name = 'VSSSS15',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,5)')

VVSSS1 = Lorentz(name = 'VVSSS1',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'Metric(1,2)')

VVVSS1 = Lorentz(name = 'VVVSS1',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,1)*Metric(1,2)')

VVVSS2 = Lorentz(name = 'VVVSS2',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,2)*Metric(1,2)')

VVVSS3 = Lorentz(name = 'VVVSS3',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(2,1)*Metric(1,3)')

VVVSS4 = Lorentz(name = 'VVVSS4',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(2,3)*Metric(1,3)')

VVVSS5 = Lorentz(name = 'VVVSS5',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3)')

VVVSS6 = Lorentz(name = 'VVVSS6',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(1,2)*Metric(2,3)')

VVVSS7 = Lorentz(name = 'VVVSS7',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(1,3)*Metric(2,3)')

VVVSS8 = Lorentz(name = 'VVVSS8',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) - P(1,2)*Metric(2,3)')

VVVSS9 = Lorentz(name = 'VVVSS9',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - 2*P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + 2*P(2,3)*Metric(1,3) + 2*P(1,2)*Metric(2,3) - 2*P(1,3)*Metric(2,3)')

VVVSS10 = Lorentz(name = 'VVVSS10',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + (3*P(2,3)*Metric(1,3))/2. + P(1,2)*Metric(2,3) - (3*P(1,3)*Metric(2,3))/2.')

VVVSS11 = Lorentz(name = 'VVVSS11',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(2,3)*Metric(1,3) - P(1,3)*Metric(2,3)')

VVVSS12 = Lorentz(name = 'VVVSS12',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,2)*Metric(1,2) + P(2,3)*Metric(1,3) - P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVSS13 = Lorentz(name = 'VVVSS13',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVSS14 = Lorentz(name = 'VVVSS14',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - (2*P(3,2)*Metric(1,2))/3. - P(2,1)*Metric(1,3) + (2*P(2,3)*Metric(1,3))/3. + (2*P(1,2)*Metric(2,3))/3. - (2*P(1,3)*Metric(2,3))/3.')

VVVSS15 = Lorentz(name = 'VVVSS15',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + (P(2,3)*Metric(1,3))/2. + P(1,2)*Metric(2,3) - (P(1,3)*Metric(2,3))/2.')

VVVSS16 = Lorentz(name = 'VVVSS16',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3) - P(2,3)*Metric(1,3) + P(1,3)*Metric(2,3)')

VVVVS1 = Lorentz(name = 'VVVVS1',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3)')

VVVVS2 = Lorentz(name = 'VVVVS2',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,3)*Metric(2,4)')

VVVVS3 = Lorentz(name = 'VVVVS3',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,2)*Metric(3,4)')

VVVVS4 = Lorentz(name = 'VVVVS4',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVVS5 = Lorentz(name = 'VVVVS5',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVVV1 = Lorentz(name = 'VVVVV1',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - (P(5,2)*Metric(1,4)*Metric(2,3))/2. - (P(5,3)*Metric(1,4)*Metric(2,3))/2. - P(4,1)*Metric(1,5)*Metric(2,3) + (P(4,2)*Metric(1,5)*Metric(2,3))/2. + (P(4,3)*Metric(1,5)*Metric(2,3))/2. - (P(5,1)*Metric(1,3)*Metric(2,4))/2. + P(5,2)*Metric(1,3)*Metric(2,4) - (P(5,3)*Metric(1,3)*Metric(2,4))/2. + (P(3,1)*Metric(1,5)*Metric(2,4))/2. - (P(3,2)*Metric(1,5)*Metric(2,4))/2. + (P(4,1)*Metric(1,3)*Metric(2,5))/2. - P(4,2)*Metric(1,3)*Metric(2,5) + (P(4,3)*Metric(1,3)*Metric(2,5))/2. - (P(3,1)*Metric(1,4)*Metric(2,5))/2. + (P(3,2)*Metric(1,4)*Metric(2,5))/2. - (P(5,1)*Metric(1,2)*Metric(3,4))/2. - (P(5,2)*Metric(1,2)*Metric(3,4))/2. + P(5,3)*Metric(1,2)*Metric(3,4) + (P(2,1)*Metric(1,5)*Metric(3,4))/2. - (P(2,3)*Metric(1,5)*Metric(3,4))/2. + (P(1,2)*Metric(2,5)*Metric(3,4))/2. - (P(1,3)*Metric(2,5)*Metric(3,4))/2. + (P(4,1)*Metric(1,2)*Metric(3,5))/2. + (P(4,2)*Metric(1,2)*Metric(3,5))/2. - P(4,3)*Metric(1,2)*Metric(3,5) - (P(2,1)*Metric(1,4)*Metric(3,5))/2. + (P(2,3)*Metric(1,4)*Metric(3,5))/2. - (P(1,2)*Metric(2,4)*Metric(3,5))/2. + (P(1,3)*Metric(2,4)*Metric(3,5))/2.')

VVVVV2 = Lorentz(name = 'VVVVV2',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) + 2*P(3,4)*Metric(1,5)*Metric(2,4) - P(3,5)*Metric(1,5)*Metric(2,4) + P(4,1)*Metric(1,3)*Metric(2,5) - P(4,5)*Metric(1,3)*Metric(2,5) - P(3,1)*Metric(1,4)*Metric(2,5) - P(3,4)*Metric(1,4)*Metric(2,5) + 2*P(3,5)*Metric(1,4)*Metric(2,5) - P(5,1)*Metric(1,2)*Metric(3,4) + P(5,4)*Metric(1,2)*Metric(3,4) + P(2,1)*Metric(1,5)*Metric(3,4) - 2*P(2,4)*Metric(1,5)*Metric(3,4) + P(2,5)*Metric(1,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) - P(1,5)*Metric(2,5)*Metric(3,4) - P(4,1)*Metric(1,2)*Metric(3,5) + P(4,5)*Metric(1,2)*Metric(3,5) + P(2,1)*Metric(1,4)*Metric(3,5) + P(2,4)*Metric(1,4)*Metric(3,5) - 2*P(2,5)*Metric(1,4)*Metric(3,5) - P(1,4)*Metric(2,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(3,1)*Metric(1,2)*Metric(4,5) - P(3,4)*Metric(1,2)*Metric(4,5) - P(3,5)*Metric(1,2)*Metric(4,5) - 2*P(2,1)*Metric(1,3)*Metric(4,5) + P(2,4)*Metric(1,3)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV3 = Lorentz(name = 'VVVVV3',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,1)*Metric(1,4)*Metric(2,3) + P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) + P(5,1)*Metric(1,3)*Metric(2,4) + P(5,2)*Metric(1,3)*Metric(2,4) - P(5,3)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(4,2)*Metric(1,3)*Metric(2,5) + P(4,3)*Metric(1,3)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) + P(3,4)*Metric(1,4)*Metric(2,5) - 2*P(5,1)*Metric(1,2)*Metric(3,4) - 2*P(5,2)*Metric(1,2)*Metric(3,4) + 2*P(5,3)*Metric(1,2)*Metric(3,4) + 2*P(5,4)*Metric(1,2)*Metric(3,4) + 2*P(2,1)*Metric(1,5)*Metric(3,4) - P(2,3)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,5)*Metric(3,4) + 2*P(1,2)*Metric(2,5)*Metric(3,4) - P(1,3)*Metric(2,5)*Metric(3,4) - P(1,4)*Metric(2,5)*Metric(3,4) + P(4,1)*Metric(1,2)*Metric(3,5) + P(4,2)*Metric(1,2)*Metric(3,5) - 2*P(4,3)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5) + P(2,3)*Metric(1,4)*Metric(3,5) - P(1,2)*Metric(2,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) + P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) - 2*P(3,4)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,4)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV4 = Lorentz(name = 'VVVVV4',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,2)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + 2*P(4,2)*Metric(1,5)*Metric(2,3) - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,1)*Metric(1,3)*Metric(2,4) + P(5,2)*Metric(1,3)*Metric(2,4) + P(3,1)*Metric(1,5)*Metric(2,4) - 2*P(3,2)*Metric(1,5)*Metric(2,4) + P(3,5)*Metric(1,5)*Metric(2,4) + 2*P(4,1)*Metric(1,3)*Metric(2,5) - P(4,2)*Metric(1,3)*Metric(2,5) - P(4,5)*Metric(1,3)*Metric(2,5) - 2*P(3,1)*Metric(1,4)*Metric(2,5) + P(3,2)*Metric(1,4)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(4,1)*Metric(1,2)*Metric(3,5) - P(4,2)*Metric(1,2)*Metric(3,5) + 2*P(4,5)*Metric(1,2)*Metric(3,5) + P(2,1)*Metric(1,4)*Metric(3,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) + P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) - 2*P(3,5)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV5 = Lorentz(name = 'VVVVV5',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,3)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,4)*Metric(1,3)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(3,5)*Metric(1,5)*Metric(2,4) - P(4,3)*Metric(1,3)*Metric(2,5) + P(4,5)*Metric(1,3)*Metric(2,5) - P(3,4)*Metric(1,4)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(2,3)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,5)*Metric(3,4) + 2*P(2,5)*Metric(1,5)*Metric(3,4) + P(1,3)*Metric(2,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) - 2*P(1,5)*Metric(2,5)*Metric(3,4) - P(2,3)*Metric(1,4)*Metric(3,5) + 2*P(2,4)*Metric(1,4)*Metric(3,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) - 2*P(1,4)*Metric(2,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(2,4)*Metric(1,3)*Metric(4,5) - P(2,5)*Metric(1,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5) + P(1,4)*Metric(2,3)*Metric(4,5) + P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV6 = Lorentz(name = 'VVVVV6',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,2)*Metric(1,4)*Metric(2,3) + P(5,3)*Metric(1,4)*Metric(2,3) - 2*P(5,4)*Metric(1,4)*Metric(2,3) + P(4,2)*Metric(1,5)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) - 2*P(4,5)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,4)*Metric(1,3)*Metric(2,4) - P(3,2)*Metric(1,5)*Metric(2,4) + P(3,5)*Metric(1,5)*Metric(2,4) - P(4,3)*Metric(1,3)*Metric(2,5) + P(4,5)*Metric(1,3)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) + P(3,4)*Metric(1,4)*Metric(2,5) - P(5,2)*Metric(1,2)*Metric(3,4) + P(5,4)*Metric(1,2)*Metric(3,4) - P(2,3)*Metric(1,5)*Metric(3,4) + P(2,5)*Metric(1,5)*Metric(3,4) + P(1,2)*Metric(2,5)*Metric(3,4) + P(1,3)*Metric(2,5)*Metric(3,4) - P(1,4)*Metric(2,5)*Metric(3,4) - P(1,5)*Metric(2,5)*Metric(3,4) - P(4,2)*Metric(1,2)*Metric(3,5) + P(4,5)*Metric(1,2)*Metric(3,5) - P(2,3)*Metric(1,4)*Metric(3,5) + P(2,4)*Metric(1,4)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) - P(1,4)*Metric(2,4)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(3,2)*Metric(1,2)*Metric(4,5) - P(3,4)*Metric(1,2)*Metric(4,5) - P(3,5)*Metric(1,2)*Metric(4,5) + 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(2,4)*Metric(1,3)*Metric(4,5) - P(2,5)*Metric(1,3)*Metric(4,5) - 2*P(1,2)*Metric(2,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5) + 2*P(1,4)*Metric(2,3)*Metric(4,5) + 2*P(1,5)*Metric(2,3)*Metric(4,5)')

SSSSSS1 = Lorentz(name = 'SSSSSS1',
                  spins = [ 1, 1, 1, 1, 1, 1 ],
                  structure = '1')

VVSSSS1 = Lorentz(name = 'VVSSSS1',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'Metric(1,2)')

VVVVSS1 = Lorentz(name = 'VVVVSS1',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3)')

VVVVSS2 = Lorentz(name = 'VVVVSS2',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,3)*Metric(2,4)')

VVVVSS3 = Lorentz(name = 'VVVVSS3',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,2)*Metric(3,4)')

VVVVSS4 = Lorentz(name = 'VVVVSS4',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVVSS5 = Lorentz(name = 'VVVVSS5',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVVVV1 = Lorentz(name = 'VVVVVV1',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) - (Metric(1,6)*Metric(2,4)*Metric(3,5))/2. - (Metric(1,4)*Metric(2,6)*Metric(3,5))/2. - (Metric(1,5)*Metric(2,4)*Metric(3,6))/2. - (Metric(1,4)*Metric(2,5)*Metric(3,6))/2. - (Metric(1,6)*Metric(2,3)*Metric(4,5))/2. - (Metric(1,3)*Metric(2,6)*Metric(4,5))/2. + Metric(1,2)*Metric(3,6)*Metric(4,5) - (Metric(1,5)*Metric(2,3)*Metric(4,6))/2. - (Metric(1,3)*Metric(2,5)*Metric(4,6))/2. + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - 2*Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV2 = Lorentz(name = 'VVVVVV2',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - (Metric(1,5)*Metric(2,6)*Metric(3,4))/2. + Metric(1,6)*Metric(2,4)*Metric(3,5) - (Metric(1,4)*Metric(2,6)*Metric(3,5))/2. - (Metric(1,5)*Metric(2,4)*Metric(3,6))/2. - (Metric(1,4)*Metric(2,5)*Metric(3,6))/2. - 2*Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - (Metric(1,3)*Metric(2,5)*Metric(4,6))/2. - (Metric(1,2)*Metric(3,5)*Metric(4,6))/2. + Metric(1,4)*Metric(2,3)*Metric(5,6) - (Metric(1,3)*Metric(2,4)*Metric(5,6))/2. - (Metric(1,2)*Metric(3,4)*Metric(5,6))/2.')
