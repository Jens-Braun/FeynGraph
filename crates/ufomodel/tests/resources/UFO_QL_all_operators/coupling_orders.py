# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 14.0.0 for Mac OS X ARM (64-bit) (December 13, 2023)
# Date: Mon 26 Jan 2026 15:02:06


from .object_library import all_orders, CouplingOrder


QCD = CouplingOrder(name = 'QCD',
                    expansion_order = 99,
                    hierarchy = 4)

NP = CouplingOrder(name = 'NP',
                   expansion_order = 1,
                   hierarchy = 2)

QED = CouplingOrder(name = 'QED',
                    expansion_order = 99,
                    hierarchy = 3)

QL = CouplingOrder(name = 'QL',
                   expansion_order = 1,
                   hierarchy = 1)

