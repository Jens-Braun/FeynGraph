#
#
#



from .object_library import all_parameters, Parameter, all_CTparameters, CTParameter


from .function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

CA = Parameter(name = 'CA',
                     nature = 'internal',
                     type = 'real',
                     value = '3',
                     texname = 'C_A')

CF = Parameter(name = 'CF',
                     nature = 'internal',
                     type = 'real',
                     value = '4/3',
                     texname = 'C_F')

Nfl = Parameter(name = 'Nfl',
                     nature = 'external',
                     type = 'real',
                     value = '5',
                     texname = 'N_f',
                     lhablock = 'FRBlock',
                     lhacode = [ 702 ])

G_CT_l = CTParameter(name = 'G_CT_l',
                  type = 'real',
                  value = {
                     -1: 'aS/4/cmath.pi*((-11/3*CA+2/3*Nfl)/2)',
                      0: 'aS/4/cmath.pi*(dred*CA/6)'
                  },
                     texname = '\\delta\\text{G\\_CT^{light}}')

G_CT_b = CTParameter(name = 'G_CT_b',
                  type = 'real',
                  value = {
                     -1: 'aS/4/cmath.pi*(1/3)',
                      0: '0'
                  },
                     texname = '\\delta\\text{G\\_CT^b}',
                     logarg = ['MB','0'])

G_CT_t = CTParameter(name = 'G_CT_t',
                  type = 'real',
                  value = {
                     -1: 'aS/4/cmath.pi*(1/3)',
                      0: '0'
                  },
                     texname = '\\delta\\text{G\\_CT^t}',
                     logarg = ['MT','WT'])

yb_CT = CTParameter(name = 'yb_CT',
                  type = 'real',
                  value = {
                     -1: 'yb*aS/4/cmath.pi*(-3)*CF',
                      0: 'yb*aS/4/cmath.pi*(-4-dred)*CF'
                  },
                  texname = '\\delta\\text{yb\\_CT}',
                     logarg = ['MB','0'])

yt_CT = CTParameter(name = 'yt_CT',
                  type = 'real',
                  value = {
                     -1: 'yt*aS/4/cmath.pi*(-3)*CF',
                      0: 'yt*aS/4/cmath.pi*(-4-dred)*CF'
                  },
                  texname = '\\delta\\text{yt\\_CT}',
                     logarg = ['MT','0'])

gluonWF_CT_b = CTParameter(name = 'gluonWF_CT_b',
                         type = 'real',
                         value = {
                             -1:'aS/4/cmath.pi*(-2/3)',
                             0:'0'
                         },
                         texname = '\\delta Z_{g}^b',
                         CTtype = 'WF_CT',
                         field =  'g',
                         logarg = ['MB','0'])

gluonWF_CT_t = CTParameter(name = 'gluonWF_CT_t',
                         type = 'real',
                         value = {
                             -1:'aS/4/cmath.pi*(-2/3)',
                             0:'0'
                         },
                         texname = '\\delta Z_{g}^t',
                         CTtype = 'WF_CT',
                         field =  'g',
                         logarg = ['MT','0'])

bWF_CT = CTParameter(name = 'bWF_CT',
                       type = 'real',
                       value = {
                           -1: 'aS/4/cmath.pi*(-3)*CF',
                           0: 'aS/4/cmath.pi*(-4-dred)*CF'
                       },
                       texname = '\\delta Z_{b}',
                       CTtype = 'WF_CT',
                       field =  'b',
                         logarg = ['MB','0'])

tWF_CT = CTParameter(name = 'tWF_CT',
                       type = 'real',
                       value = {
                           -1: 'aS/4/cmath.pi*(-3)*CF',
                           0: 'aS/4/cmath.pi*(-4-dred)*CF'
                       },
                       texname = '\\delta Z_{t}',
                       CTtype = 'WF_CT',
                       field =  't',
                         logarg = ['MT','WT'])

MB_CT = CTParameter(name = 'MB_CT',
                  type = 'real',
                  value = {
                     -1: 'MB*aS/4/cmath.pi*(-3)*CF',
                      0: 'MB*aS/4/cmath.pi*(-4-dred)*CF'
                  },
                  texname = '\\delta\\text{MB\\_CT}',
                    logarg = ['MB','0'])

MT_CT = CTParameter(name = 'MT_CT',
                  type = 'real',
                  value = {
                     -1: 'MT*aS/4/cmath.pi*(-3)*CF',
                      0: 'MT*aS/4/cmath.pi*(-4-dred)*CF'
                  },
                  texname = '\\delta\\text{MT\\_CT}',
                    logarg = ['MT','WT'])

axial_CT = CTParameter(name = 'axial_CT',
                  type = 'real',
                  value = {
                     -1: '0',
                      0: 'aS/4/cmath.pi*(-4*CF)*(1-dred)'
                  },
                  texname = '\\delta\\text{axial}')
