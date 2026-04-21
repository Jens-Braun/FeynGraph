# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 13.2.1 for Linux x86 (64-bit) (January 27, 2023)
# Date: Mon 8 Dec 2025 20:55:52


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot
try:
   import form_factors as ForFac 
except ImportError:
   pass


SSS1 = Lorentz(name = 'SSS1',
               spins = [ 1, 1, 1 ],
               structure = '1')

SSS2 = Lorentz(name = 'SSS2',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2) + P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3)')

SSS3 = Lorentz(name = 'SSS3',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)**2 + P(-1,2)**2 + P(-1,3)**2')

FFS1 = Lorentz(name = 'FFS1',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1)')

FFS2 = Lorentz(name = 'FFS2',
               spins = [ 2, 2, 1 ],
               structure = 'ProjP(2,1)')

FFS3 = Lorentz(name = 'FFS3',
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
               structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

VVS1 = Lorentz(name = 'VVS1',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1)')

VVS2 = Lorentz(name = 'VVS2',
               spins = [ 3, 3, 1 ],
               structure = '-(Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1)) + Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,2)')

VVS3 = Lorentz(name = 'VVS3',
               spins = [ 3, 3, 1 ],
               structure = 'Metric(1,2)')

VVS4 = Lorentz(name = 'VVS4',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVV1 = Lorentz(name = 'VVV1',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,1))')

VVV2 = Lorentz(name = 'VVV2',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,3))')

VVV3 = Lorentz(name = 'VVV3',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,3)')

VVV4 = Lorentz(name = 'VVV4',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(1,2)*Metric(2,3)')

VVV5 = Lorentz(name = 'VVV5',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVV6 = Lorentz(name = 'VVV6',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,2)*Metric(1,2) - P(2,3)*Metric(1,3) - P(1,2)*Metric(2,3) + P(1,3)*Metric(2,3)')

VVV7 = Lorentz(name = 'VVV7',
               spins = [ 3, 3, 3 ],
               structure = '-(P(1,2)*P(2,3)*P(3,1)) + P(1,3)*P(2,1)*P(3,2) + P(-1,2)*P(-1,3)*P(3,1)*Metric(1,2) - P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2) - P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3) + P(-1,1)*P(-1,2)*P(2,3)*Metric(1,3) + P(-1,1)*P(-1,3)*P(1,2)*Metric(2,3) - P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3)')

VVV8 = Lorentz(name = 'VVV8',
               spins = [ 3, 3, 3 ],
               structure = 'Epsilon(1,2,3,-2)*P(-2,3)*P(-1,1)*P(-1,2) + Epsilon(1,2,3,-2)*P(-2,2)*P(-1,1)*P(-1,3) + Epsilon(1,2,3,-2)*P(-2,1)*P(-1,2)*P(-1,3) - Epsilon(2,3,-1,-2)*P(-2,3)*P(-1,1)*P(1,2) - Epsilon(2,3,-1,-2)*P(-2,2)*P(-1,1)*P(1,3) + Epsilon(1,3,-1,-2)*P(-2,3)*P(-1,2)*P(2,1) - Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,1)*P(2,3) + Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2)*P(3,1) + Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)*P(3,2) + Epsilon(3,-1,-2,-3)*P(-3,3)*P(-2,2)*P(-1,1)*Metric(1,2) + Epsilon(2,-1,-2,-3)*P(-3,3)*P(-2,2)*P(-1,1)*Metric(1,3) + Epsilon(1,-1,-2,-3)*P(-3,3)*P(-2,2)*P(-1,1)*Metric(2,3)')

VVV9 = Lorentz(name = 'VVV9',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-2)*P(-2,3)*P(-1,1)*P(-1,2)) - Epsilon(1,2,3,-2)*P(-2,2)*P(-1,1)*P(-1,3) - Epsilon(1,2,3,-2)*P(-2,1)*P(-1,2)*P(-1,3) - Epsilon(2,3,-1,-2)*P(-2,1)*P(-1,3)*P(1,2) - Epsilon(2,3,-1,-2)*P(-2,1)*P(-1,2)*P(1,3) + Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,3)*P(2,1) + Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,1)*P(2,3) - Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2)*P(3,1) - Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)*P(3,2) - (Epsilon(3,-1,-2,-3)*P(-3,2)*P(-2,1)*P(-1,3)*Metric(1,2))/2. + (Epsilon(3,-1,-2,-3)*P(-3,1)*P(-2,2)*P(-1,3)*Metric(1,2))/2. + (Epsilon(2,-1,-2,-3)*P(-3,3)*P(-2,1)*P(-1,2)*Metric(1,3))/2. - (Epsilon(2,-1,-2,-3)*P(-3,1)*P(-2,3)*P(-1,2)*Metric(1,3))/2. - (Epsilon(1,-1,-2,-3)*P(-3,3)*P(-2,2)*P(-1,1)*Metric(2,3))/2. + (Epsilon(1,-1,-2,-3)*P(-3,2)*P(-2,3)*P(-1,1)*Metric(2,3))/2.')

SSSS1 = Lorentz(name = 'SSSS1',
                spins = [ 1, 1, 1, 1 ],
                structure = '1')

SSSS2 = Lorentz(name = 'SSSS2',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2) + P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3) + P(-1,1)*P(-1,4) + P(-1,2)*P(-1,4) + P(-1,3)*P(-1,4)')

SSSS3 = Lorentz(name = 'SSSS3',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)**2 + (2*P(-1,1)*P(-1,2))/3. + P(-1,2)**2 + (2*P(-1,1)*P(-1,3))/3. + (2*P(-1,2)*P(-1,3))/3. + P(-1,3)**2 + (2*P(-1,1)*P(-1,4))/3. + (2*P(-1,2)*P(-1,4))/3. + (2*P(-1,3)*P(-1,4))/3. + P(-1,4)**2')

FFSS1 = Lorentz(name = 'FFSS1',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjM(2,1)')

FFSS2 = Lorentz(name = 'FFSS2',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjP(2,1)')

VVSS1 = Lorentz(name = 'VVSS1',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1)')

VVSS2 = Lorentz(name = 'VVSS2',
                spins = [ 3, 3, 1, 1 ],
                structure = '-(Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1)) + Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,2)')

VVSS3 = Lorentz(name = 'VVSS3',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Metric(1,2)')

VVSS4 = Lorentz(name = 'VVSS4',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVVS1 = Lorentz(name = 'VVVS1',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,1))')

VVVS2 = Lorentz(name = 'VVVS2',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,3))')

VVVS3 = Lorentz(name = 'VVVS3',
                spins = [ 3, 3, 3, 1 ],
                structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,3)')

VVVS4 = Lorentz(name = 'VVVS4',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3)')

VVVS5 = Lorentz(name = 'VVVS5',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(2,3)*Metric(1,3) - P(1,3)*Metric(2,3)')

VVVS6 = Lorentz(name = 'VVVS6',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVV1 = Lorentz(name = 'VVVV1',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV2 = Lorentz(name = 'VVVV2',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,2)*P(4,1)*Metric(1,2) - P(3,1)*P(4,2)*Metric(1,2) + P(2,1)*P(4,2)*Metric(1,3) + P(2,4)*P(4,3)*Metric(1,3) - P(2,1)*P(3,2)*Metric(1,4) - P(2,3)*P(3,4)*Metric(1,4) - P(1,2)*P(4,1)*Metric(2,3) - P(1,4)*P(4,3)*Metric(2,3) + P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3) + P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(1,2)*P(3,1)*Metric(2,4) + P(1,3)*P(3,4)*Metric(2,4) - P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4) - P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,4)*P(2,3)*Metric(3,4) - P(1,3)*P(2,4)*Metric(3,4)')

VVVV3 = Lorentz(name = 'VVVV3',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Epsilon(1,2,3,4)*P(-1,2)*P(-1,3) + Epsilon(1,2,3,4)*P(-1,1)*P(-1,4) - Epsilon(2,3,4,-1)*P(-1,3)*P(1,2) - Epsilon(2,3,4,-1)*P(-1,2)*P(1,3) + Epsilon(2,3,4,-1)*P(-1,1)*P(1,4) + Epsilon(1,3,4,-1)*P(-1,4)*P(2,1) - Epsilon(1,3,4,-1)*P(-1,2)*P(2,3) + Epsilon(1,3,4,-1)*P(-1,1)*P(2,4) - Epsilon(1,2,4,-1)*P(-1,4)*P(3,1) + Epsilon(1,2,4,-1)*P(-1,3)*P(3,2) - Epsilon(1,2,4,-1)*P(-1,1)*P(3,4) - Epsilon(1,2,3,-1)*P(-1,4)*P(4,1) + Epsilon(1,2,3,-1)*P(-1,3)*P(4,2) + Epsilon(1,2,3,-1)*P(-1,2)*P(4,3) - Epsilon(3,4,-1,-2)*P(-2,2)*P(-1,3)*Metric(1,2) + Epsilon(3,4,-1,-2)*P(-2,1)*P(-1,4)*Metric(1,2) + Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(1,3) - Epsilon(2,4,-1,-2)*P(-2,1)*P(-1,4)*Metric(1,3) + (Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,1)*Metric(1,4))/2. - (Epsilon(2,3,-1,-2)*P(-2,1)*P(-1,4)*Metric(1,4))/2. + (Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(2,3))/2. - (Epsilon(1,4,-1,-2)*P(-2,2)*P(-1,3)*Metric(2,3))/2. + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,1)*Metric(2,4) - Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,3)*Metric(2,4) - Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,1)*Metric(3,4) + Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2)*Metric(3,4)')

VVVV4 = Lorentz(name = 'VVVV4',
                spins = [ 3, 3, 3, 3 ],
                structure = '-(Epsilon(1,2,3,4)*P(-1,1)*P(-1,3)) - Epsilon(1,2,3,4)*P(-1,2)*P(-1,4) + Epsilon(2,3,4,-1)*P(-1,4)*P(1,2) - Epsilon(2,3,4,-1)*P(-1,1)*P(1,3) + Epsilon(2,3,4,-1)*P(-1,2)*P(1,4) - Epsilon(1,3,4,-1)*P(-1,3)*P(2,1) - Epsilon(1,3,4,-1)*P(-1,1)*P(2,3) + Epsilon(1,3,4,-1)*P(-1,2)*P(2,4) - Epsilon(1,2,4,-1)*P(-1,3)*P(3,1) + Epsilon(1,2,4,-1)*P(-1,4)*P(3,2) + Epsilon(1,2,4,-1)*P(-1,2)*P(3,4) - Epsilon(1,2,3,-1)*P(-1,3)*P(4,1) + Epsilon(1,2,3,-1)*P(-1,4)*P(4,2) - Epsilon(1,2,3,-1)*P(-1,1)*P(4,3) - Epsilon(3,4,-1,-2)*P(-2,1)*P(-1,3)*Metric(1,2) + Epsilon(3,4,-1,-2)*P(-2,2)*P(-1,4)*Metric(1,2) + (Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,3))/2. - (Epsilon(2,4,-1,-2)*P(-2,1)*P(-1,3)*Metric(1,3))/2. + Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,4) - Epsilon(2,3,-1,-2)*P(-2,1)*P(-1,3)*Metric(1,4) + Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(2,3) - Epsilon(1,4,-1,-2)*P(-2,2)*P(-1,4)*Metric(2,3) + (Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,4))/2. - (Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,4)*Metric(2,4))/2. - Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)*Metric(3,4) + Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,2)*Metric(3,4)')

VVVV5 = Lorentz(name = 'VVVV5',
                spins = [ 3, 3, 3, 3 ],
                structure = '-(Epsilon(1,2,3,4)*P(-1,1)*P(-1,3)) + Epsilon(1,2,3,4)*P(-1,2)*P(-1,3) + Epsilon(1,2,3,4)*P(-1,1)*P(-1,4) - Epsilon(1,2,3,4)*P(-1,2)*P(-1,4) - Epsilon(2,3,4,-1)*P(-1,3)*P(1,2) + Epsilon(2,3,4,-1)*P(-1,4)*P(1,2) - Epsilon(2,3,4,-1)*P(-1,1)*P(1,3) - Epsilon(2,3,4,-1)*P(-1,2)*P(1,3) + Epsilon(2,3,4,-1)*P(-1,1)*P(1,4) + Epsilon(2,3,4,-1)*P(-1,2)*P(1,4) - Epsilon(1,3,4,-1)*P(-1,3)*P(2,1) + Epsilon(1,3,4,-1)*P(-1,4)*P(2,1) - Epsilon(1,3,4,-1)*P(-1,1)*P(2,3) - Epsilon(1,3,4,-1)*P(-1,2)*P(2,3) + Epsilon(1,3,4,-1)*P(-1,1)*P(2,4) + Epsilon(1,3,4,-1)*P(-1,2)*P(2,4) - Epsilon(1,2,4,-1)*P(-1,3)*P(3,1) - Epsilon(1,2,4,-1)*P(-1,4)*P(3,1) + Epsilon(1,2,4,-1)*P(-1,3)*P(3,2) + Epsilon(1,2,4,-1)*P(-1,4)*P(3,2) - Epsilon(1,2,4,-1)*P(-1,1)*P(3,4) + Epsilon(1,2,4,-1)*P(-1,2)*P(3,4) - Epsilon(1,2,3,-1)*P(-1,3)*P(4,1) - Epsilon(1,2,3,-1)*P(-1,4)*P(4,1) + Epsilon(1,2,3,-1)*P(-1,3)*P(4,2) + Epsilon(1,2,3,-1)*P(-1,4)*P(4,2) - Epsilon(1,2,3,-1)*P(-1,1)*P(4,3) + Epsilon(1,2,3,-1)*P(-1,2)*P(4,3) + Epsilon(3,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,2) - Epsilon(3,4,-1,-2)*P(-2,4)*P(-1,1)*Metric(1,2) + Epsilon(3,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(1,2) - Epsilon(3,4,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,2) + Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,3) + Epsilon(2,4,-1,-2)*P(-2,4)*P(-1,1)*Metric(1,3) + Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(1,3) + Epsilon(2,3,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,4) + Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,1)*Metric(1,4) + Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,4) + Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,2)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,3) + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,1)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,3)*P(-1,2)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,4) - Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)*Metric(3,4) - Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,1)*Metric(3,4) + Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2)*Metric(3,4) + Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,2)*Metric(3,4)')

VVVV6 = Lorentz(name = 'VVVV6',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Epsilon(1,2,3,4)*P(-1,1)*P(-1,2) + Epsilon(1,2,3,4)*P(-1,3)*P(-1,4) + Epsilon(2,3,4,-1)*P(-1,1)*P(1,2) - Epsilon(2,3,4,-1)*P(-1,4)*P(1,3) - Epsilon(2,3,4,-1)*P(-1,3)*P(1,4) - Epsilon(1,3,4,-1)*P(-1,2)*P(2,1) + Epsilon(1,3,4,-1)*P(-1,4)*P(2,3) + Epsilon(1,3,4,-1)*P(-1,3)*P(2,4) - Epsilon(1,2,4,-1)*P(-1,2)*P(3,1) - Epsilon(1,2,4,-1)*P(-1,1)*P(3,2) + Epsilon(1,2,4,-1)*P(-1,3)*P(3,4) + Epsilon(1,2,3,-1)*P(-1,2)*P(4,1) + Epsilon(1,2,3,-1)*P(-1,1)*P(4,2) - Epsilon(1,2,3,-1)*P(-1,4)*P(4,3) + (Epsilon(3,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(1,2))/2. - (Epsilon(3,4,-1,-2)*P(-2,1)*P(-1,2)*Metric(1,2))/2. - Epsilon(2,4,-1,-2)*P(-2,1)*P(-1,2)*Metric(1,3) + Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,4)*Metric(1,3) + Epsilon(2,3,-1,-2)*P(-2,1)*P(-1,2)*Metric(1,4) - Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,3)*Metric(1,4) + Epsilon(1,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(2,3) - Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,4)*Metric(2,3) - Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,1)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,3)*Metric(2,4) + (Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,3)*Metric(3,4))/2. - (Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,4)*Metric(3,4))/2.')

VVVV7 = Lorentz(name = 'VVVV7',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Epsilon(1,2,3,4)*P(-1,1)*P(-1,2) - Epsilon(1,2,3,4)*P(-1,1)*P(-1,3) - Epsilon(1,2,3,4)*P(-1,2)*P(-1,4) + Epsilon(1,2,3,4)*P(-1,3)*P(-1,4) + Epsilon(2,3,4,-1)*P(-1,1)*P(1,2) + Epsilon(2,3,4,-1)*P(-1,4)*P(1,2) - Epsilon(2,3,4,-1)*P(-1,1)*P(1,3) - Epsilon(2,3,4,-1)*P(-1,4)*P(1,3) + Epsilon(2,3,4,-1)*P(-1,2)*P(1,4) - Epsilon(2,3,4,-1)*P(-1,3)*P(1,4) - Epsilon(1,3,4,-1)*P(-1,2)*P(2,1) - Epsilon(1,3,4,-1)*P(-1,3)*P(2,1) - Epsilon(1,3,4,-1)*P(-1,1)*P(2,3) + Epsilon(1,3,4,-1)*P(-1,4)*P(2,3) + Epsilon(1,3,4,-1)*P(-1,2)*P(2,4) + Epsilon(1,3,4,-1)*P(-1,3)*P(2,4) - Epsilon(1,2,4,-1)*P(-1,2)*P(3,1) - Epsilon(1,2,4,-1)*P(-1,3)*P(3,1) - Epsilon(1,2,4,-1)*P(-1,1)*P(3,2) + Epsilon(1,2,4,-1)*P(-1,4)*P(3,2) + Epsilon(1,2,4,-1)*P(-1,2)*P(3,4) + Epsilon(1,2,4,-1)*P(-1,3)*P(3,4) + Epsilon(1,2,3,-1)*P(-1,2)*P(4,1) - Epsilon(1,2,3,-1)*P(-1,3)*P(4,1) + Epsilon(1,2,3,-1)*P(-1,1)*P(4,2) + Epsilon(1,2,3,-1)*P(-1,4)*P(4,2) - Epsilon(1,2,3,-1)*P(-1,1)*P(4,3) - Epsilon(1,2,3,-1)*P(-1,4)*P(4,3) + Epsilon(3,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(1,2) + Epsilon(3,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,2) - Epsilon(3,4,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,2) + Epsilon(2,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(1,3) + Epsilon(2,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,3) - Epsilon(2,4,-1,-2)*P(-2,4)*P(-1,3)*Metric(1,3) - Epsilon(2,3,-1,-2)*P(-2,2)*P(-1,1)*Metric(1,4) + Epsilon(2,3,-1,-2)*P(-2,3)*P(-1,1)*Metric(1,4) + Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(1,4) - Epsilon(2,3,-1,-2)*P(-2,4)*P(-1,3)*Metric(1,4) + Epsilon(1,4,-1,-2)*P(-2,2)*P(-1,1)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,3)*P(-1,1)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,3) + Epsilon(1,4,-1,-2)*P(-2,4)*P(-1,3)*Metric(2,3) - Epsilon(1,3,-1,-2)*P(-2,2)*P(-1,1)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,2)*Metric(2,4) + Epsilon(1,3,-1,-2)*P(-2,4)*P(-1,3)*Metric(2,4) - Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)*Metric(3,4) + Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,2)*Metric(3,4) + Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,3)*Metric(3,4)')

VVVV8 = Lorentz(name = 'VVVV8',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVV9 = Lorentz(name = 'VVVV9',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV10 = Lorentz(name = 'VVVV10',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVV11 = Lorentz(name = 'VVVV11',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVV12 = Lorentz(name = 'VVVV12',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,4)*P(4,1)*Metric(1,2) + P(3,2)*P(4,3)*Metric(1,2) - P(2,4)*P(4,1)*Metric(1,3) - P(2,3)*P(4,2)*Metric(1,3) + P(2,4)*P(3,1)*Metric(1,4) - P(2,1)*P(3,4)*Metric(1,4) + P(1,3)*P(4,2)*Metric(2,3) - P(1,2)*P(4,3)*Metric(2,3) - P(1,4)*P(3,1)*Metric(2,4) - P(1,3)*P(3,2)*Metric(2,4) + P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4) + P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,4)*P(2,1)*Metric(3,4) + P(1,2)*P(2,3)*Metric(3,4) - P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV13 = Lorentz(name = 'VVVV13',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,4)*P(4,2)*Metric(1,2) + P(3,1)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) - P(2,1)*P(4,3)*Metric(1,3) - P(2,3)*P(3,1)*Metric(1,4) - P(2,4)*P(3,2)*Metric(1,4) - P(1,3)*P(4,1)*Metric(2,3) - P(1,4)*P(4,2)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(1,4)*P(3,2)*Metric(2,4) - P(1,2)*P(3,4)*Metric(2,4) + P(1,3)*P(2,1)*Metric(3,4) + P(1,2)*P(2,4)*Metric(3,4) - P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV14 = Lorentz(name = 'VVVV14',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,2)*P(4,1)*Metric(1,2) - P(3,1)*P(4,2)*Metric(1,2) + P(3,4)*P(4,2)*Metric(1,2) + P(3,1)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) + P(2,1)*P(4,2)*Metric(1,3) - P(2,1)*P(4,3)*Metric(1,3) + P(2,4)*P(4,3)*Metric(1,3) - P(2,3)*P(3,1)*Metric(1,4) - P(2,1)*P(3,2)*Metric(1,4) - P(2,4)*P(3,2)*Metric(1,4) - P(2,3)*P(3,4)*Metric(1,4) - P(1,2)*P(4,1)*Metric(2,3) - P(1,3)*P(4,1)*Metric(2,3) - P(1,4)*P(4,2)*Metric(2,3) - P(1,4)*P(4,3)*Metric(2,3) + P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(1,2)*P(3,1)*Metric(2,4) + P(1,4)*P(3,2)*Metric(2,4) - P(1,2)*P(3,4)*Metric(2,4) + P(1,3)*P(3,4)*Metric(2,4) - P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4) - P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,3)*P(2,1)*Metric(3,4) + P(1,4)*P(2,3)*Metric(3,4) + P(1,2)*P(2,4)*Metric(3,4) - P(1,3)*P(2,4)*Metric(3,4) - P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV15 = Lorentz(name = 'VVVV15',
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

VVSSS1 = Lorentz(name = 'VVSSS1',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'Metric(1,2)')

VVVSS1 = Lorentz(name = 'VVVSS1',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,1))')

VVVSS2 = Lorentz(name = 'VVVSS2',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,3))')

VVVSS3 = Lorentz(name = 'VVVSS3',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,3)')

VVVSS4 = Lorentz(name = 'VVVSS4',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3)')

VVVSS5 = Lorentz(name = 'VVVSS5',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(2,3)*Metric(1,3) - P(1,3)*Metric(2,3)')

VVVSS6 = Lorentz(name = 'VVVSS6',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVVS1 = Lorentz(name = 'VVVVS1',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVVS2 = Lorentz(name = 'VVVVS2',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVVS3 = Lorentz(name = 'VVVVS3',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVVS4 = Lorentz(name = 'VVVVS4',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVVS5 = Lorentz(name = 'VVVVS5',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVVV1 = Lorentz(name = 'VVVVV1',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(2,3,4,5)*P(1,3) - Epsilon(1,3,4,5)*P(2,3) - Epsilon(2,4,5,-1)*P(-1,3)*Metric(1,3) - (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. + Epsilon(1,4,5,-1)*P(-1,3)*Metric(2,3) + (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2.')

VVVVV2 = Lorentz(name = 'VVVVV2',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(2,3,4,5)*P(1,4) - Epsilon(1,3,4,5)*P(2,4) + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. + Epsilon(2,3,5,-1)*P(-1,4)*Metric(1,4) + (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. - (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. - Epsilon(1,3,5,-1)*P(-1,4)*Metric(2,4) - (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2.')

VVVVV3 = Lorentz(name = 'VVVVV3',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(2,3,4,5)*P(1,5) - Epsilon(1,3,4,5)*P(2,5) + (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. - Epsilon(2,3,4,-1)*P(-1,5)*Metric(1,5) - (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. + (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. + Epsilon(1,3,4,-1)*P(-1,5)*Metric(2,5)')

VVVVV4 = Lorentz(name = 'VVVVV4',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(1,2,3,5)*P(4,1) - Epsilon(1,2,3,4)*P(5,1) + Epsilon(2,3,5,-1)*P(-1,1)*Metric(1,4) - Epsilon(2,3,4,-1)*P(-1,1)*Metric(1,5) + (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. - (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. - (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. + (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2.')

VVVVV5 = Lorentz(name = 'VVVVV5',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(1,3,4,5)*P(2,1) - Epsilon(1,2,4,5)*P(3,1) + Epsilon(3,4,5,-1)*P(-1,1)*Metric(1,2) - Epsilon(2,4,5,-1)*P(-1,1)*Metric(1,3) - (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. + (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. - (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2.')

VVVVV6 = Lorentz(name = 'VVVVV6',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(2,3,4,5)*P(1,2) + Epsilon(1,2,4,5)*P(3,2) + Epsilon(3,4,5,-1)*P(-1,2)*Metric(1,2) - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. - Epsilon(1,4,5,-1)*P(-1,2)*Metric(2,3) - (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. + (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2.')

VVVVV7 = Lorentz(name = 'VVVVV7',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(1,2,3,5)*P(4,2) - Epsilon(1,2,3,4)*P(5,2) - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. - Epsilon(1,3,5,-1)*P(-1,2)*Metric(2,4) + Epsilon(1,3,4,-1)*P(-1,2)*Metric(2,5) - (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. + (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2.')

VVVVV8 = Lorentz(name = 'VVVVV8',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(2,3,4,5)*P(1,2) - Epsilon(2,3,4,5)*P(1,3) + Epsilon(1,3,4,5)*P(2,1) - Epsilon(1,3,4,5)*P(2,3) + Epsilon(1,2,4,5)*P(3,1) - Epsilon(1,2,4,5)*P(3,2) + Epsilon(3,4,5,-1)*P(-1,1)*Metric(1,2) + Epsilon(3,4,5,-1)*P(-1,2)*Metric(1,2) + Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2) + Epsilon(2,4,5,-1)*P(-1,1)*Metric(1,3) + Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3) + Epsilon(2,4,5,-1)*P(-1,3)*Metric(1,3) - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. - (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. + Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3) + Epsilon(1,4,5,-1)*P(-1,2)*Metric(2,3) + Epsilon(1,4,5,-1)*P(-1,3)*Metric(2,3) - (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. + (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. - (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. - (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. + (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. + (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. - (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2.')

VVVVV9 = Lorentz(name = 'VVVVV9',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'Epsilon(1,2,3,5)*P(4,3) - Epsilon(1,2,3,4)*P(5,3) - (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. + (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. + Epsilon(1,2,5,-1)*P(-1,3)*Metric(3,4) - Epsilon(1,2,4,-1)*P(-1,3)*Metric(3,5)')

VVVVV10 = Lorentz(name = 'VVVVV10',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,4) + Epsilon(1,2,4,5)*P(3,4) - (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. + Epsilon(2,3,5,-1)*P(-1,4)*Metric(1,4) + (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. + (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. + Epsilon(1,2,5,-1)*P(-1,4)*Metric(3,4) + (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2.')

VVVVV11 = Lorentz(name = 'VVVVV11',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,4) - Epsilon(1,2,4,5)*P(3,4) - (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. + Epsilon(1,3,5,-1)*P(-1,4)*Metric(2,4) + (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. - Epsilon(1,2,5,-1)*P(-1,4)*Metric(3,4) - (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2.')

VVVVV12 = Lorentz(name = 'VVVVV12',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,5) - Epsilon(1,2,4,5)*P(3,5) - (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. - Epsilon(1,3,4,-1)*P(-1,5)*Metric(2,5) + (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. + Epsilon(1,2,4,-1)*P(-1,5)*Metric(3,5)')

VVVVV13 = Lorentz(name = 'VVVVV13',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,5) + Epsilon(1,2,4,5)*P(3,5) - (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. - (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. - Epsilon(2,3,4,-1)*P(-1,5)*Metric(1,5) + (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. - (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. - Epsilon(1,2,4,-1)*P(-1,5)*Metric(3,5)')

VVVVV14 = Lorentz(name = 'VVVVV14',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) - P(4,1)*Metric(1,3)*Metric(2,5) + P(3,1)*Metric(1,4)*Metric(2,5) - P(5,1)*Metric(1,2)*Metric(3,4) + P(2,1)*Metric(1,5)*Metric(3,4) + P(4,1)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5)')

VVVVV15 = Lorentz(name = 'VVVVV15',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(4,2)*Metric(1,5)*Metric(2,3) + P(3,2)*Metric(1,5)*Metric(2,4) - P(3,2)*Metric(1,4)*Metric(2,5) - P(5,2)*Metric(1,2)*Metric(3,4) + P(1,2)*Metric(2,5)*Metric(3,4) + P(4,2)*Metric(1,2)*Metric(3,5) - P(1,2)*Metric(2,4)*Metric(3,5)')

VVVVV16 = Lorentz(name = 'VVVVV16',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - (P(5,2)*Metric(1,4)*Metric(2,3))/2. - (P(5,3)*Metric(1,4)*Metric(2,3))/2. - P(4,1)*Metric(1,5)*Metric(2,3) + (P(4,2)*Metric(1,5)*Metric(2,3))/2. + (P(4,3)*Metric(1,5)*Metric(2,3))/2. - (P(5,1)*Metric(1,3)*Metric(2,4))/2. + P(5,2)*Metric(1,3)*Metric(2,4) - (P(5,3)*Metric(1,3)*Metric(2,4))/2. + (P(3,1)*Metric(1,5)*Metric(2,4))/2. - (P(3,2)*Metric(1,5)*Metric(2,4))/2. + (P(4,1)*Metric(1,3)*Metric(2,5))/2. - P(4,2)*Metric(1,3)*Metric(2,5) + (P(4,3)*Metric(1,3)*Metric(2,5))/2. - (P(3,1)*Metric(1,4)*Metric(2,5))/2. + (P(3,2)*Metric(1,4)*Metric(2,5))/2. - (P(5,1)*Metric(1,2)*Metric(3,4))/2. - (P(5,2)*Metric(1,2)*Metric(3,4))/2. + P(5,3)*Metric(1,2)*Metric(3,4) + (P(2,1)*Metric(1,5)*Metric(3,4))/2. - (P(2,3)*Metric(1,5)*Metric(3,4))/2. + (P(1,2)*Metric(2,5)*Metric(3,4))/2. - (P(1,3)*Metric(2,5)*Metric(3,4))/2. + (P(4,1)*Metric(1,2)*Metric(3,5))/2. + (P(4,2)*Metric(1,2)*Metric(3,5))/2. - P(4,3)*Metric(1,2)*Metric(3,5) - (P(2,1)*Metric(1,4)*Metric(3,5))/2. + (P(2,3)*Metric(1,4)*Metric(3,5))/2. - (P(1,2)*Metric(2,4)*Metric(3,5))/2. + (P(1,3)*Metric(2,4)*Metric(3,5))/2.')

VVVVV17 = Lorentz(name = 'VVVVV17',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) - P(4,3)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(4,3)*Metric(1,3)*Metric(2,5) + P(2,3)*Metric(1,5)*Metric(3,4) - P(1,3)*Metric(2,5)*Metric(3,4) - P(2,3)*Metric(1,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5)')

VVVVV18 = Lorentz(name = 'VVVVV18',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,1) + Epsilon(1,2,3,5)*P(4,1) + Epsilon(3,4,5,-1)*P(-1,1)*Metric(1,2) + Epsilon(2,3,5,-1)*P(-1,1)*Metric(1,4) + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. + (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. - (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. + (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2.')

VVVVV19 = Lorentz(name = 'VVVVV19',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,1) + Epsilon(1,2,3,4)*P(5,1) + Epsilon(2,4,5,-1)*P(-1,1)*Metric(1,3) + Epsilon(2,3,4,-1)*P(-1,1)*Metric(1,5) + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. + (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. - (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. + (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2.')

VVVVV20 = Lorentz(name = 'VVVVV20',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,1) - Epsilon(1,2,3,5)*P(4,1) + Epsilon(2,4,5,-1)*P(-1,1)*Metric(1,3) - Epsilon(2,3,5,-1)*P(-1,1)*Metric(1,4) + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. - (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. - (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2.')

VVVVV21 = Lorentz(name = 'VVVVV21',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,1) - Epsilon(1,2,3,4)*P(5,1) + Epsilon(3,4,5,-1)*P(-1,1)*Metric(1,2) - Epsilon(2,3,4,-1)*P(-1,1)*Metric(1,5) + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. - (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. - (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2.')

VVVVV22 = Lorentz(name = 'VVVVV22',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,2) + Epsilon(1,2,3,4)*P(5,2) - (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. - (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. - Epsilon(1,4,5,-1)*P(-1,2)*Metric(2,3) - Epsilon(1,3,4,-1)*P(-1,2)*Metric(2,5) - (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. + (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV23 = Lorentz(name = 'VVVVV23',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,2) + Epsilon(1,2,3,4)*P(5,2) + Epsilon(3,4,5,-1)*P(-1,2)*Metric(1,2) + (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. - Epsilon(1,3,4,-1)*P(-1,2)*Metric(2,5) - (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2. + (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV24 = Lorentz(name = 'VVVVV24',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,2) - Epsilon(1,2,3,5)*P(4,2) + Epsilon(3,4,5,-1)*P(-1,2)*Metric(1,2) + (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. + (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. + Epsilon(1,3,5,-1)*P(-1,2)*Metric(2,4) + (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. - (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV25 = Lorentz(name = 'VVVVV25',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,2) - Epsilon(1,2,3,5)*P(4,2) - (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. + (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. - Epsilon(1,4,5,-1)*P(-1,2)*Metric(2,3) + Epsilon(1,3,5,-1)*P(-1,2)*Metric(2,4) + (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2. - (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV26 = Lorentz(name = 'VVVVV26',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,2) - Epsilon(2,3,4,5)*P(1,5) + Epsilon(1,3,4,5)*P(2,1) - Epsilon(1,3,4,5)*P(2,5) + Epsilon(1,2,3,4)*P(5,1) - Epsilon(1,2,3,4)*P(5,2) + Epsilon(3,4,5,-1)*P(-1,1)*Metric(1,2) + Epsilon(3,4,5,-1)*P(-1,2)*Metric(1,2) + Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2) + (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. - (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. + Epsilon(2,3,4,-1)*P(-1,1)*Metric(1,5) + Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5) + Epsilon(2,3,4,-1)*P(-1,5)*Metric(1,5) + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. - (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. - (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. + Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5) + Epsilon(1,3,4,-1)*P(-1,2)*Metric(2,5) + Epsilon(1,3,4,-1)*P(-1,5)*Metric(2,5) - (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. + (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2. + (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2. - (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV27 = Lorentz(name = 'VVVVV27',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = '-(Epsilon(2,3,4,5)*P(1,3)) + Epsilon(2,3,4,5)*P(1,4) - Epsilon(1,3,4,5)*P(2,3) + Epsilon(1,3,4,5)*P(2,4) + Epsilon(1,2,4,5)*P(3,1) - Epsilon(1,2,4,5)*P(3,2) + Epsilon(1,2,3,5)*P(4,1) - Epsilon(1,2,3,5)*P(4,2) + Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2) - Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2) + Epsilon(2,4,5,-1)*P(-1,1)*Metric(1,3) + (Epsilon(2,4,5,-1)*P(-1,2)*Metric(1,3))/2. + Epsilon(2,4,5,-1)*P(-1,3)*Metric(1,3) + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. + Epsilon(2,3,5,-1)*P(-1,1)*Metric(1,4) + (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + Epsilon(2,3,5,-1)*P(-1,4)*Metric(1,4) - (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. + (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. + (Epsilon(1,4,5,-1)*P(-1,1)*Metric(2,3))/2. + Epsilon(1,4,5,-1)*P(-1,2)*Metric(2,3) + Epsilon(1,4,5,-1)*P(-1,3)*Metric(2,3) + (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. + (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. + Epsilon(1,3,5,-1)*P(-1,2)*Metric(2,4) + (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. + Epsilon(1,3,5,-1)*P(-1,4)*Metric(2,4) - (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. + (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. - Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4) + Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4) + (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. - (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2. + (Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5))/2. - (Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5))/2.')

VVVVV28 = Lorentz(name = 'VVVVV28',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,3) + Epsilon(1,2,3,5)*P(4,3) - (Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2))/2. - (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. - Epsilon(1,4,5,-1)*P(-1,3)*Metric(2,3) + (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. + Epsilon(1,2,5,-1)*P(-1,3)*Metric(3,4) + (Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5))/2.')

VVVVV29 = Lorentz(name = 'VVVVV29',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,3) + Epsilon(1,2,3,4)*P(5,3) - (Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2))/2. - Epsilon(2,4,5,-1)*P(-1,3)*Metric(1,3) - (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. + Epsilon(1,2,4,-1)*P(-1,3)*Metric(3,5) + (Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5))/2.')

VVVVV30 = Lorentz(name = 'VVVVV30',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,3) - Epsilon(1,2,3,5)*P(4,3) - (Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2))/2. - Epsilon(2,4,5,-1)*P(-1,3)*Metric(1,3) + (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. - (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - Epsilon(1,2,5,-1)*P(-1,3)*Metric(3,4) - (Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5))/2.')

VVVVV31 = Lorentz(name = 'VVVVV31',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,3) - Epsilon(1,2,3,4)*P(5,3) - (Epsilon(3,4,5,-1)*P(-1,3)*Metric(1,2))/2. + (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. - Epsilon(1,4,5,-1)*P(-1,3)*Metric(2,3) - (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - Epsilon(1,2,4,-1)*P(-1,3)*Metric(3,5) - (Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5))/2.')

VVVVV32 = Lorentz(name = 'VVVVV32',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = '-(Epsilon(1,3,4,5)*P(2,4)) + Epsilon(1,3,4,5)*P(2,5) - Epsilon(1,2,4,5)*P(3,4) + Epsilon(1,2,4,5)*P(3,5) + Epsilon(1,2,3,5)*P(4,2) - Epsilon(1,2,3,5)*P(4,3) + Epsilon(1,2,3,4)*P(5,2) - Epsilon(1,2,3,4)*P(5,3) + (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. - (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. - (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - (Epsilon(2,3,5,-1)*P(-1,2)*Metric(1,4))/2. + (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. - (Epsilon(2,3,4,-1)*P(-1,2)*Metric(1,5))/2. + (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. - Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3) + Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3) - Epsilon(1,3,5,-1)*P(-1,2)*Metric(2,4) - (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - Epsilon(1,3,5,-1)*P(-1,4)*Metric(2,4) - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. - Epsilon(1,3,4,-1)*P(-1,2)*Metric(2,5) - (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. - (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. - Epsilon(1,3,4,-1)*P(-1,5)*Metric(2,5) - (Epsilon(1,2,5,-1)*P(-1,2)*Metric(3,4))/2. - Epsilon(1,2,5,-1)*P(-1,3)*Metric(3,4) - Epsilon(1,2,5,-1)*P(-1,4)*Metric(3,4) - (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. - (Epsilon(1,2,4,-1)*P(-1,2)*Metric(3,5))/2. - Epsilon(1,2,4,-1)*P(-1,3)*Metric(3,5) - (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2. - Epsilon(1,2,4,-1)*P(-1,5)*Metric(3,5) + Epsilon(1,2,3,-1)*P(-1,2)*Metric(4,5) - Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5)')

VVVVV33 = Lorentz(name = 'VVVVV33',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,4) - Epsilon(1,2,3,4)*P(5,4) - (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. + (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. + (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. + Epsilon(1,3,5,-1)*P(-1,4)*Metric(2,4) + (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2. + Epsilon(1,2,3,-1)*P(-1,4)*Metric(4,5)')

VVVVV34 = Lorentz(name = 'VVVVV34',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,4) + Epsilon(1,2,3,4)*P(5,4) - (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. - (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. + (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. + (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. + Epsilon(1,2,5,-1)*P(-1,4)*Metric(3,4) - Epsilon(1,2,3,-1)*P(-1,4)*Metric(4,5)')

VVVVV35 = Lorentz(name = 'VVVVV35',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,4) + Epsilon(1,2,3,4)*P(5,4) - (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. + Epsilon(2,3,5,-1)*P(-1,4)*Metric(1,4) + (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. - (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2. - Epsilon(1,2,3,-1)*P(-1,4)*Metric(4,5)')

VVVVV36 = Lorentz(name = 'VVVVV36',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,5) - Epsilon(1,2,3,5)*P(4,5) - (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - Epsilon(2,3,4,-1)*P(-1,5)*Metric(1,5) - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. + (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. + Epsilon(1,2,3,-1)*P(-1,5)*Metric(4,5)')

VVVVV37 = Lorentz(name = 'VVVVV37',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,5) - Epsilon(1,2,3,5)*P(4,5) - (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. + (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. + (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. - Epsilon(1,2,4,-1)*P(-1,5)*Metric(3,5) + Epsilon(1,2,3,-1)*P(-1,5)*Metric(4,5)')

VVVVV38 = Lorentz(name = 'VVVVV38',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,5)*P(1,4) - Epsilon(2,3,4,5)*P(1,5) + Epsilon(1,2,3,5)*P(4,1) - Epsilon(1,2,3,5)*P(4,5) + Epsilon(1,2,3,4)*P(5,1) - Epsilon(1,2,3,4)*P(5,4) - (Epsilon(3,4,5,-1)*P(-1,4)*Metric(1,2))/2. + (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. + (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. - (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. + Epsilon(2,3,5,-1)*P(-1,1)*Metric(1,4) + Epsilon(2,3,5,-1)*P(-1,4)*Metric(1,4) + Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4) + Epsilon(2,3,4,-1)*P(-1,1)*Metric(1,5) + Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5) + Epsilon(2,3,4,-1)*P(-1,5)*Metric(1,5) + (Epsilon(1,3,5,-1)*P(-1,1)*Metric(2,4))/2. - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. + (Epsilon(1,3,4,-1)*P(-1,1)*Metric(2,5))/2. - (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. - (Epsilon(1,2,5,-1)*P(-1,1)*Metric(3,4))/2. + (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. - (Epsilon(1,2,4,-1)*P(-1,1)*Metric(3,5))/2. + (Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5))/2. + Epsilon(1,2,3,-1)*P(-1,1)*Metric(4,5) + Epsilon(1,2,3,-1)*P(-1,4)*Metric(4,5) + Epsilon(1,2,3,-1)*P(-1,5)*Metric(4,5)')

VVVVV39 = Lorentz(name = 'VVVVV39',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,2,4,5)*P(3,4) - Epsilon(1,2,4,5)*P(3,5) + Epsilon(1,2,3,5)*P(4,3) - Epsilon(1,2,3,5)*P(4,5) + Epsilon(1,2,3,4)*P(5,3) - Epsilon(1,2,3,4)*P(5,4) - (Epsilon(2,4,5,-1)*P(-1,4)*Metric(1,3))/2. + (Epsilon(2,4,5,-1)*P(-1,5)*Metric(1,3))/2. - (Epsilon(2,3,5,-1)*P(-1,3)*Metric(1,4))/2. + (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. - (Epsilon(2,3,4,-1)*P(-1,3)*Metric(1,5))/2. + (Epsilon(2,3,4,-1)*P(-1,4)*Metric(1,5))/2. + (Epsilon(1,4,5,-1)*P(-1,4)*Metric(2,3))/2. - (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. + (Epsilon(1,3,5,-1)*P(-1,3)*Metric(2,4))/2. - (Epsilon(1,3,5,-1)*P(-1,5)*Metric(2,4))/2. + (Epsilon(1,3,4,-1)*P(-1,3)*Metric(2,5))/2. - (Epsilon(1,3,4,-1)*P(-1,4)*Metric(2,5))/2. + Epsilon(1,2,5,-1)*P(-1,3)*Metric(3,4) + Epsilon(1,2,5,-1)*P(-1,4)*Metric(3,4) + Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4) + Epsilon(1,2,4,-1)*P(-1,3)*Metric(3,5) + Epsilon(1,2,4,-1)*P(-1,4)*Metric(3,5) + Epsilon(1,2,4,-1)*P(-1,5)*Metric(3,5) + Epsilon(1,2,3,-1)*P(-1,3)*Metric(4,5) + Epsilon(1,2,3,-1)*P(-1,4)*Metric(4,5) + Epsilon(1,2,3,-1)*P(-1,5)*Metric(4,5)')

VVVVV40 = Lorentz(name = 'VVVVV40',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(1,3,4,5)*P(2,5) + Epsilon(1,2,3,5)*P(4,5) - (Epsilon(3,4,5,-1)*P(-1,5)*Metric(1,2))/2. - (Epsilon(2,3,5,-1)*P(-1,5)*Metric(1,4))/2. + (Epsilon(1,4,5,-1)*P(-1,5)*Metric(2,3))/2. - Epsilon(1,3,4,-1)*P(-1,5)*Metric(2,5) - (Epsilon(1,2,5,-1)*P(-1,5)*Metric(3,4))/2. - Epsilon(1,2,3,-1)*P(-1,5)*Metric(4,5)')

VVVVV41 = Lorentz(name = 'VVVVV41',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + P(4,1)*Metric(1,3)*Metric(2,5) - P(3,1)*Metric(1,4)*Metric(2,5) - P(5,1)*Metric(1,2)*Metric(3,4) + P(2,1)*Metric(1,5)*Metric(3,4) + P(3,1)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5)')

VVVVV42 = Lorentz(name = 'VVVVV42',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) - P(5,1)*Metric(1,3)*Metric(2,4) + P(3,1)*Metric(1,5)*Metric(2,4) + P(4,1)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5) - P(3,1)*Metric(1,2)*Metric(4,5) + P(2,1)*Metric(1,3)*Metric(4,5)')

VVVVV43 = Lorentz(name = 'VVVVV43',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,3)*Metric(2,4) - P(3,4)*Metric(1,4)*Metric(2,5) - P(5,4)*Metric(1,2)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) + P(2,4)*Metric(1,4)*Metric(3,5) - P(1,4)*Metric(2,4)*Metric(3,5) + P(3,4)*Metric(1,2)*Metric(4,5) - P(2,4)*Metric(1,3)*Metric(4,5)')

VVVVV44 = Lorentz(name = 'VVVVV44',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(3,5)*Metric(1,5)*Metric(2,4) - P(4,5)*Metric(1,3)*Metric(2,5) - P(2,5)*Metric(1,5)*Metric(3,4) + P(1,5)*Metric(2,5)*Metric(3,4) + P(4,5)*Metric(1,2)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) - P(3,5)*Metric(1,2)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV45 = Lorentz(name = 'VVVVV45',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) + 2*P(3,4)*Metric(1,5)*Metric(2,4) - P(3,5)*Metric(1,5)*Metric(2,4) + P(4,1)*Metric(1,3)*Metric(2,5) - P(4,5)*Metric(1,3)*Metric(2,5) - P(3,1)*Metric(1,4)*Metric(2,5) - P(3,4)*Metric(1,4)*Metric(2,5) + 2*P(3,5)*Metric(1,4)*Metric(2,5) - P(5,1)*Metric(1,2)*Metric(3,4) + P(5,4)*Metric(1,2)*Metric(3,4) + P(2,1)*Metric(1,5)*Metric(3,4) - 2*P(2,4)*Metric(1,5)*Metric(3,4) + P(2,5)*Metric(1,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) - P(1,5)*Metric(2,5)*Metric(3,4) - P(4,1)*Metric(1,2)*Metric(3,5) + P(4,5)*Metric(1,2)*Metric(3,5) + P(2,1)*Metric(1,4)*Metric(3,5) + P(2,4)*Metric(1,4)*Metric(3,5) - 2*P(2,5)*Metric(1,4)*Metric(3,5) - P(1,4)*Metric(2,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(3,1)*Metric(1,2)*Metric(4,5) - P(3,4)*Metric(1,2)*Metric(4,5) - P(3,5)*Metric(1,2)*Metric(4,5) - 2*P(2,1)*Metric(1,3)*Metric(4,5) + P(2,4)*Metric(1,3)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV46 = Lorentz(name = 'VVVVV46',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,2)*Metric(1,5)*Metric(2,3) + P(5,2)*Metric(1,3)*Metric(2,4) - P(3,2)*Metric(1,5)*Metric(2,4) - P(4,2)*Metric(1,3)*Metric(2,5) - P(5,2)*Metric(1,2)*Metric(3,4) + P(1,2)*Metric(2,5)*Metric(3,4) + P(3,2)*Metric(1,2)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5)')

VVVVV47 = Lorentz(name = 'VVVVV47',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(5,2)*Metric(1,3)*Metric(2,4) + P(4,2)*Metric(1,3)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) - P(4,2)*Metric(1,2)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) + P(3,2)*Metric(1,2)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5)')

VVVVV48 = Lorentz(name = 'VVVVV48',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) - P(4,3)*Metric(1,3)*Metric(2,5) - P(5,3)*Metric(1,2)*Metric(3,4) + P(1,3)*Metric(2,5)*Metric(3,4) + P(4,3)*Metric(1,2)*Metric(3,5) - P(2,3)*Metric(1,4)*Metric(3,5) + P(2,3)*Metric(1,3)*Metric(4,5) - P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV49 = Lorentz(name = 'VVVVV49',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,3)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,3)*Metric(1,2)*Metric(3,4) - P(2,3)*Metric(1,5)*Metric(3,4) - P(4,3)*Metric(1,2)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) + P(2,3)*Metric(1,3)*Metric(4,5) - P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV50 = Lorentz(name = 'VVVVV50',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,4)*Metric(2,3) - P(3,4)*Metric(1,5)*Metric(2,4) - P(5,4)*Metric(1,2)*Metric(3,4) + P(2,4)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,4)*Metric(3,5) + P(1,4)*Metric(2,4)*Metric(3,5) + P(3,4)*Metric(1,2)*Metric(4,5) - P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV51 = Lorentz(name = 'VVVVV51',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,3)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(3,4)*Metric(1,4)*Metric(2,5) - P(2,4)*Metric(1,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) + P(2,4)*Metric(1,3)*Metric(4,5) - P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV52 = Lorentz(name = 'VVVVV52',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) + P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) + P(5,1)*Metric(1,3)*Metric(2,4) + P(5,2)*Metric(1,3)*Metric(2,4) - P(5,3)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(4,2)*Metric(1,3)*Metric(2,5) + P(4,3)*Metric(1,3)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) + P(3,4)*Metric(1,4)*Metric(2,5) - 2*P(5,1)*Metric(1,2)*Metric(3,4) - 2*P(5,2)*Metric(1,2)*Metric(3,4) + 2*P(5,3)*Metric(1,2)*Metric(3,4) + 2*P(5,4)*Metric(1,2)*Metric(3,4) + 2*P(2,1)*Metric(1,5)*Metric(3,4) - P(2,3)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,5)*Metric(3,4) + 2*P(1,2)*Metric(2,5)*Metric(3,4) - P(1,3)*Metric(2,5)*Metric(3,4) - P(1,4)*Metric(2,5)*Metric(3,4) + P(4,1)*Metric(1,2)*Metric(3,5) + P(4,2)*Metric(1,2)*Metric(3,5) - 2*P(4,3)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5) + P(2,3)*Metric(1,4)*Metric(3,5) - P(1,2)*Metric(2,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) + P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) - 2*P(3,4)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,4)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV53 = Lorentz(name = 'VVVVV53',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,5)*Metric(1,5)*Metric(2,3) - P(3,5)*Metric(1,4)*Metric(2,5) - P(2,5)*Metric(1,5)*Metric(3,4) + P(1,5)*Metric(2,5)*Metric(3,4) - P(4,5)*Metric(1,2)*Metric(3,5) + P(2,5)*Metric(1,4)*Metric(3,5) + P(3,5)*Metric(1,2)*Metric(4,5) - P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV54 = Lorentz(name = 'VVVVV54',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,5)*Metric(1,5)*Metric(2,3) - P(3,5)*Metric(1,5)*Metric(2,4) - P(4,5)*Metric(1,3)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + P(2,5)*Metric(1,3)*Metric(4,5) - P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV55 = Lorentz(name = 'VVVVV55',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,2)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + 2*P(4,2)*Metric(1,5)*Metric(2,3) - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,1)*Metric(1,3)*Metric(2,4) + P(5,2)*Metric(1,3)*Metric(2,4) + P(3,1)*Metric(1,5)*Metric(2,4) - 2*P(3,2)*Metric(1,5)*Metric(2,4) + P(3,5)*Metric(1,5)*Metric(2,4) + 2*P(4,1)*Metric(1,3)*Metric(2,5) - P(4,2)*Metric(1,3)*Metric(2,5) - P(4,5)*Metric(1,3)*Metric(2,5) - 2*P(3,1)*Metric(1,4)*Metric(2,5) + P(3,2)*Metric(1,4)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(4,1)*Metric(1,2)*Metric(3,5) - P(4,2)*Metric(1,2)*Metric(3,5) + 2*P(4,5)*Metric(1,2)*Metric(3,5) + P(2,1)*Metric(1,4)*Metric(3,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) + P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) - 2*P(3,5)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV56 = Lorentz(name = 'VVVVV56',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,4)*Metric(1,3)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(3,5)*Metric(1,5)*Metric(2,4) - P(4,3)*Metric(1,3)*Metric(2,5) + P(4,5)*Metric(1,3)*Metric(2,5) - P(3,4)*Metric(1,4)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(2,3)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,5)*Metric(3,4) + 2*P(2,5)*Metric(1,5)*Metric(3,4) + P(1,3)*Metric(2,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) - 2*P(1,5)*Metric(2,5)*Metric(3,4) - P(2,3)*Metric(1,4)*Metric(3,5) + 2*P(2,4)*Metric(1,4)*Metric(3,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) - 2*P(1,4)*Metric(2,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(2,4)*Metric(1,3)*Metric(4,5) - P(2,5)*Metric(1,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5) + P(1,4)*Metric(2,3)*Metric(4,5) + P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV57 = Lorentz(name = 'VVVVV57',
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
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVVSS2 = Lorentz(name = 'VVVVSS2',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVVSS3 = Lorentz(name = 'VVVVSS3',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVVSS4 = Lorentz(name = 'VVVVSS4',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVVSS5 = Lorentz(name = 'VVVVSS5',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVVVV1 = Lorentz(name = 'VVVVVV1',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,3,5,6)*Metric(2,4)')

VVVVVV2 = Lorentz(name = 'VVVVVV2',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,4,5,6)*Metric(1,3) + Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,3,4,6)*Metric(2,5)')

VVVVVV3 = Lorentz(name = 'VVVVVV3',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(1,3,5,6)*Metric(2,4) + Epsilon(1,3,4,6)*Metric(2,5)')

VVVVVV4 = Lorentz(name = 'VVVVVV4',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,5,6)*Metric(1,4) + Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,3,4,5)*Metric(2,6)')

VVVVVV5 = Lorentz(name = 'VVVVVV5',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,3,4,5)*Metric(2,6)')

VVVVVV6 = Lorentz(name = 'VVVVVV6',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,3,4,6)*Metric(2,5) + Epsilon(1,3,4,5)*Metric(2,6)')

VVVVVV7 = Lorentz(name = 'VVVVVV7',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,4,5,6)*Metric(1,3) + Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,2,5,6)*Metric(3,4)')

VVVVVV8 = Lorentz(name = 'VVVVVV8',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,2,5,6)*Metric(3,4)')

VVVVVV9 = Lorentz(name = 'VVVVVV9',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,2,4,6)*Metric(3,5)')

VVVVVV10 = Lorentz(name = 'VVVVVV10',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,2,5,6)*Metric(3,4) - Epsilon(1,2,4,6)*Metric(3,5)')

VVVVVV11 = Lorentz(name = 'VVVVVV11',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(1,3,4,6)*Metric(2,5) + Epsilon(1,2,4,6)*Metric(3,5)')

VVVVVV12 = Lorentz(name = 'VVVVVV12',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,2,5,6)*Metric(3,4) + Epsilon(1,2,4,6)*Metric(3,5)')

VVVVVV13 = Lorentz(name = 'VVVVVV13',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,4,5,6)*Metric(1,3) + Epsilon(1,3,4,5)*Metric(2,6) - Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV14 = Lorentz(name = 'VVVVVV14',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,5,6)*Metric(2,4) + Epsilon(1,3,4,5)*Metric(2,6) - Epsilon(1,2,5,6)*Metric(3,4) - Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV15 = Lorentz(name = 'VVVVVV15',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(2,3,4,5)*Metric(1,6) + Epsilon(1,2,4,6)*Metric(3,5) - Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV16 = Lorentz(name = 'VVVVVV16',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV17 = Lorentz(name = 'VVVVVV17',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,5,6)*Metric(1,4) + Epsilon(2,3,4,5)*Metric(1,6) + Epsilon(1,2,5,6)*Metric(3,4) + Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV18 = Lorentz(name = 'VVVVVV18',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,3,4,5)*Metric(2,6) - Epsilon(1,2,4,6)*Metric(3,5) + Epsilon(1,2,4,5)*Metric(3,6)')

VVVVVV19 = Lorentz(name = 'VVVVVV19',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV20 = Lorentz(name = 'VVVVVV20',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) + Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,2,5,6)*Metric(3,4) - Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV21 = Lorentz(name = 'VVVVVV21',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,3,5,6)*Metric(2,4) + Epsilon(1,2,4,6)*Metric(3,5) - Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV22 = Lorentz(name = 'VVVVVV22',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,3,5,6)*Metric(2,4) + Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV23 = Lorentz(name = 'VVVVVV23',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,2,5,6)*Metric(3,4) + Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV24 = Lorentz(name = 'VVVVVV24',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(1,2,4,6)*Metric(3,5) + Epsilon(1,2,3,6)*Metric(4,5)')

VVVVVV25 = Lorentz(name = 'VVVVVV25',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,4,5)*Metric(1,6) + Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV26 = Lorentz(name = 'VVVVVV26',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,3,4,5)*Metric(2,6) - Epsilon(1,2,5,6)*Metric(3,4) - Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV27 = Lorentz(name = 'VVVVVV27',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,5,6)*Metric(1,4) + Epsilon(1,2,4,5)*Metric(3,6) - Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV28 = Lorentz(name = 'VVVVVV28',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,3,4,5)*Metric(2,6) + Epsilon(1,2,3,6)*Metric(4,5) - Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV29 = Lorentz(name = 'VVVVVV29',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,5,6)*Metric(1,4) + Epsilon(1,3,4,5)*Metric(2,6) + Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV30 = Lorentz(name = 'VVVVVV30',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,4,5)*Metric(1,6) + Epsilon(1,2,5,6)*Metric(3,4) + Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV31 = Lorentz(name = 'VVVVVV31',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,2,4,5)*Metric(3,6) + Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV32 = Lorentz(name = 'VVVVVV32',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,4,6)*Metric(1,5) - Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,2,3,6)*Metric(4,5) + Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV33 = Lorentz(name = 'VVVVVV33',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,2,4,6)*Metric(3,5) - Epsilon(1,2,4,5)*Metric(3,6) - Epsilon(1,2,3,6)*Metric(4,5) + Epsilon(1,2,3,5)*Metric(4,6)')

VVVVVV34 = Lorentz(name = 'VVVVVV34',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV35 = Lorentz(name = 'VVVVVV35',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV36 = Lorentz(name = 'VVVVVV36',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV37 = Lorentz(name = 'VVVVVV37',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) - Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,3,4,5)*Metric(2,6) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV38 = Lorentz(name = 'VVVVVV38',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) - Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,2,4,6)*Metric(3,5) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV39 = Lorentz(name = 'VVVVVV39',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) + Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,2,4,5)*Metric(3,6) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV40 = Lorentz(name = 'VVVVVV40',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,5,6)*Metric(2,4) + Epsilon(1,3,4,5)*Metric(2,6) + Epsilon(1,2,3,6)*Metric(4,5) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV41 = Lorentz(name = 'VVVVVV41',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,5,6)*Metric(1,4) - Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,2,3,5)*Metric(4,6) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV42 = Lorentz(name = 'VVVVVV42',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,2,5,6)*Metric(3,4) - Epsilon(1,2,4,6)*Metric(3,5) + Epsilon(1,2,3,5)*Metric(4,6) - Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV43 = Lorentz(name = 'VVVVVV43',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(3,4,5,6)*Metric(1,2) + Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,3,4,6)*Metric(2,5) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV44 = Lorentz(name = 'VVVVVV44',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,4,5,6)*Metric(2,3) - Epsilon(1,3,4,5)*Metric(2,6) + Epsilon(1,2,4,6)*Metric(3,5) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV45 = Lorentz(name = 'VVVVVV45',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,4,5,6)*Metric(1,3) + Epsilon(2,3,4,6)*Metric(1,5) + Epsilon(1,2,4,5)*Metric(3,6) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV46 = Lorentz(name = 'VVVVVV46',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(2,3,5,6)*Metric(1,4) + Epsilon(2,3,4,5)*Metric(1,6) - Epsilon(1,2,3,6)*Metric(4,5) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV47 = Lorentz(name = 'VVVVVV47',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,2,5,6)*Metric(3,4) + Epsilon(1,2,4,5)*Metric(3,6) - Epsilon(1,2,3,6)*Metric(4,5) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV48 = Lorentz(name = 'VVVVVV48',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Epsilon(1,3,5,6)*Metric(2,4) - Epsilon(1,3,4,6)*Metric(2,5) - Epsilon(1,2,3,5)*Metric(4,6) + Epsilon(1,2,3,4)*Metric(5,6)')

VVVVVV49 = Lorentz(name = 'VVVVVV49',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV50 = Lorentz(name = 'VVVVVV50',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV51 = Lorentz(name = 'VVVVVV51',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV52 = Lorentz(name = 'VVVVVV52',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV53 = Lorentz(name = 'VVVVVV53',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) - (Metric(1,6)*Metric(2,4)*Metric(3,5))/2. - (Metric(1,4)*Metric(2,6)*Metric(3,5))/2. - (Metric(1,5)*Metric(2,4)*Metric(3,6))/2. - (Metric(1,4)*Metric(2,5)*Metric(3,6))/2. - (Metric(1,6)*Metric(2,3)*Metric(4,5))/2. - (Metric(1,3)*Metric(2,6)*Metric(4,5))/2. + Metric(1,2)*Metric(3,6)*Metric(4,5) - (Metric(1,5)*Metric(2,3)*Metric(4,6))/2. - (Metric(1,3)*Metric(2,5)*Metric(4,6))/2. + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - 2*Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV54 = Lorentz(name = 'VVVVVV54',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV55 = Lorentz(name = 'VVVVVV55',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV56 = Lorentz(name = 'VVVVVV56',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV57 = Lorentz(name = 'VVVVVV57',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV58 = Lorentz(name = 'VVVVVV58',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV59 = Lorentz(name = 'VVVVVV59',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV60 = Lorentz(name = 'VVVVVV60',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV61 = Lorentz(name = 'VVVVVV61',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV62 = Lorentz(name = 'VVVVVV62',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - (Metric(1,5)*Metric(2,6)*Metric(3,4))/2. + Metric(1,6)*Metric(2,4)*Metric(3,5) - (Metric(1,4)*Metric(2,6)*Metric(3,5))/2. - (Metric(1,5)*Metric(2,4)*Metric(3,6))/2. - (Metric(1,4)*Metric(2,5)*Metric(3,6))/2. - 2*Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - (Metric(1,3)*Metric(2,5)*Metric(4,6))/2. - (Metric(1,2)*Metric(3,5)*Metric(4,6))/2. + Metric(1,4)*Metric(2,3)*Metric(5,6) - (Metric(1,3)*Metric(2,4)*Metric(5,6))/2. - (Metric(1,2)*Metric(3,4)*Metric(5,6))/2.')

