# This file was automatically created by FeynRules 2.3.31
# Mathematica version: 11.2.0 for Mac OS X x86 (64-bit) (September 11, 2017)
# Date: Wed 9 May 2018 16:32:46


from .object_library import all_orders, CouplingOrder


QCD = CouplingOrder(name = 'QCD',
                    expansion_order = 99,
                    hierarchy = 1,
                    perturbative_expansion = 1)

QED = CouplingOrder(name = 'QED',
                    expansion_order = 99,
                    hierarchy = 2)

