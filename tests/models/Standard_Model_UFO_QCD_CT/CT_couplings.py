#
#
#


from .object_library import all_couplings, Coupling

from .function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# strong coupling renormalisation

UVGC_10 = Coupling(name = 'UVGC_10',
                 value = '-G*(G_CT_l+G_CT_b+G_CT_t)',
                 order = {'QCD':3})

UVGC_11 = Coupling(name = 'UVGC_11',
                 value = 'complex(0,1)*G*(G_CT_l+G_CT_b+G_CT_t)',
                 order = {'QCD':3})

UVGC_12 = Coupling(name = 'UVGC_12',
                 value = 'complex(0,1)*G**2*2*(G_CT_l+G_CT_b+G_CT_t)',
                 order = {'QCD':4})

# yukawa renormalisation

UVGC_76 = Coupling(name = 'UVGC_76',
                 value = '-(yb_CT/cmath.sqrt(2))',
                 order = {'QED':1,'QCD':2})

UVGC_77 = Coupling(name = 'UVGC_77',
                 value = '-((complex(0,1)*yb_CT)/cmath.sqrt(2))',
                 order = {'QED':1,'QCD':2})

UVGC_78 = Coupling(name = 'UVGC_78',
                 value = '-((complex(0,1)*yt_CT)/cmath.sqrt(2))',
                 order = {'QED':1,'QCD':2})

UVGC_79 = Coupling(name = 'UVGC_79',
                 value = 'yt_CT/cmath.sqrt(2)',
                 order = {'QED':1,'QCD':2})

# quark mass renormalisation

UVGC_MB = Coupling(name = 'UVGC_MB',
                   value = '-complex(0,1)*MB_CT',
                   order = {'QCD':2})

UVGC_MT = Coupling(name = 'UVGC_MT',
                   value = '-complex(0,1)*MT_CT',
                   order = {'QCD':2})

# finite renormalisation for axial-vector current

# W^- couplings:

UVaxialGC_35 = Coupling(name = 'UVaxialGC_35',
                 value = '(CKM1x1*ee*complex(0,1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_36 = Coupling(name = 'UVaxialGC_36',
                 value = '(CKM1x2*ee*complex(0,1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_37 = Coupling(name = 'UVaxialGC_37',
                 value = '(CKM1x3*ee*complex(0,1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_38 = Coupling(name = 'UVaxialGC_38',
                 value = '(CKM2x1*ee*complex(0,1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_39 = Coupling(name = 'UVaxialGC_39',
                 value = '(CKM2x2*ee*complex(0,1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_40 = Coupling(name = 'UVaxialGC_40',
                 value = '(CKM2x3*ee*complex(0,1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_41 = Coupling(name = 'UVaxialGC_41',
                 value = '(CKM3x1*ee*complex(0,1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_42 = Coupling(name = 'UVaxialGC_42',
                 value = '(CKM3x2*ee*complex(0,1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_43 = Coupling(name = 'UVaxialGC_43',
                 value = '(CKM3x3*ee*complex(0,1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

# W^+ couplings:

UVaxialGC_84 = Coupling(name = 'UVaxialGC_84',
                 value = '(ee*complex(0,1)*complexconjugate(CKM1x1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_85 = Coupling(name = 'UVaxialGC_85',
                 value = '(ee*complex(0,1)*complexconjugate(CKM1x2))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_86 = Coupling(name = 'UVaxialGC_86',
                 value = '(ee*complex(0,1)*complexconjugate(CKM1x3))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_87 = Coupling(name = 'UVaxialGC_87',
                 value = '(ee*complex(0,1)*complexconjugate(CKM2x1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_88 = Coupling(name = 'UVaxialGC_88',
                 value = '(ee*complex(0,1)*complexconjugate(CKM2x2))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_89 = Coupling(name = 'UVaxialGC_89',
                 value = '(ee*complex(0,1)*complexconjugate(CKM2x3))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_90 = Coupling(name = 'UVaxialGC_90',
                 value = '(ee*complex(0,1)*complexconjugate(CKM3x1))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_91 = Coupling(name = 'UVaxialGC_91',
                 value = '(ee*complex(0,1)*complexconjugate(CKM3x2))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_92 = Coupling(name = 'UVaxialGC_92',
                 value = '(ee*complex(0,1)*complexconjugate(CKM3x3))/(sw*cmath.sqrt(2))*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

# Z couplings:

UVaxialGC_Zd = Coupling(name = 'UVaxialGC_Zd',
                 value = '(ee*complex(0,1))/sw/cw*(-1./2.)*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})

UVaxialGC_Zu = Coupling(name = 'UVaxialGC_Zu',
                 value = '(ee*complex(0,1))/sw/cw*(1./2.)*(-axial_CT/4.)',
                 order = {'QED':1,'QCD':2})
