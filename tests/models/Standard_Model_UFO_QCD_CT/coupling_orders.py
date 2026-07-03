# This file was automatically created by FeynRules 2.0.23
# Mathematica version: 9.0 for Linux x86 (64-bit) (February 7, 2013)
# Date: Mon 7 Apr 2014 16:44:54


from .object_library import all_orders, CouplingOrder


QCD = CouplingOrder(name = 'QCD',
                    expansion_order = 99,
                    hierarchy = 1)

QED = CouplingOrder(name = 'QED',
                    expansion_order = 99,
                    hierarchy = 2)

