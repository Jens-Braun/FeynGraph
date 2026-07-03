# This file was automatically created by FeynRules 2.3.35
# Mathematica version: 12.1.0 for Linux x86 (64-bit) (March 18, 2020)
# Date: Wed 6 Jan 2021 16:20:39


from function_library import acsc, asec, complexconjugate, cot, csc, im, re, sec
from object_library import Coupling, all_couplings

GC_1 = Coupling(name="GC_1", value="(ee*complex(0,1))/3.", order={"QED": 1})

GC_2 = Coupling(name="GC_2", value="(-2*ee*complex(0,1))/3.", order={"QED": 1})

GC_3 = Coupling(name="GC_3", value="-(ee*complex(0,1))", order={"QED": 1})

GC_4 = Coupling(name="GC_4", value="ee*complex(0,1)", order={"QED": 1})

GC_5 = Coupling(name="GC_5", value="ee**2*complex(0,1)", order={"QED": 2})

GC_6 = Coupling(name="GC_6", value="-(complex(0,1)*G)", order={"QCD": 1})

GC_7 = Coupling(name="GC_7", value="G", order={"QCD": 1})

GC_8 = Coupling(name="GC_8", value="complex(0,1)*G**2", order={"QCD": 2})

GC_9 = Coupling(name="GC_9", value="-6*complex(0,1)*lam", order={"QED": 2})

GC_10 = Coupling(
    name="GC_10", value="(2*cdd*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_11 = Coupling(
    name="GC_11",
    value="(2*cdd1*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_12 = Coupling(
    name="GC_12", value="(ced*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_13 = Coupling(
    name="GC_13", value="(2*cee*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_14 = Coupling(
    name="GC_14", value="(ceu*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_15 = Coupling(
    name="GC_15", value="(-6*cG)/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_16 = Coupling(name="GC_16", value="cGtil/LambdaSMEFT**2", order={"NP": 1, "QED": 2})

GC_17 = Coupling(
    name="GC_17", value="(90*cH*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 3}
)

GC_18 = Coupling(
    name="GC_18",
    value="(-3*cHbox*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_19 = Coupling(
    name="GC_19",
    value="-((cHDD*complex(0,1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_20 = Coupling(
    name="GC_20", value="(4*cHG*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_21 = Coupling(
    name="GC_21",
    value="(-2*cHGtil*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_22 = Coupling(
    name="GC_22", value="(4*cHW*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_23 = Coupling(
    name="GC_23",
    value="(4*cHWtil*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_24 = Coupling(
    name="GC_24", value="(cld*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_25 = Coupling(
    name="GC_25", value="(cle*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_26 = Coupling(
    name="GC_26", value="(2*cll*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_27 = Coupling(
    name="GC_27",
    value="(2*cll1*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_28 = Coupling(
    name="GC_28",
    value="(clq1*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_29 = Coupling(
    name="GC_29",
    value="-((clq3*complex(0,1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_30 = Coupling(
    name="GC_30",
    value="(clq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_31 = Coupling(
    name="GC_31",
    value="(2*CKM1x1*clq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_32 = Coupling(
    name="GC_32",
    value="(2*CKM1x2*clq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_33 = Coupling(
    name="GC_33",
    value="(2*CKM1x3*clq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_34 = Coupling(
    name="GC_34",
    value="(2*CKM2x1*clq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_35 = Coupling(
    name="GC_35",
    value="(2*CKM2x2*clq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_36 = Coupling(
    name="GC_36",
    value="(2*CKM2x3*clq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_37 = Coupling(
    name="GC_37",
    value="(2*CKM3x1*clq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_38 = Coupling(
    name="GC_38",
    value="(2*CKM3x2*clq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_39 = Coupling(
    name="GC_39",
    value="(2*CKM3x3*clq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_40 = Coupling(
    name="GC_40", value="(clu*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_41 = Coupling(
    name="GC_41",
    value="(cqd1*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_42 = Coupling(
    name="GC_42",
    value="(cqd8*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_43 = Coupling(
    name="GC_43", value="(cqe*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_44 = Coupling(
    name="GC_44",
    value="(2*cqq1*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_45 = Coupling(
    name="GC_45",
    value="(2*cqq11*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_46 = Coupling(
    name="GC_46",
    value="(-2*cqq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_47 = Coupling(
    name="GC_47",
    value="(2*cqq3*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_48 = Coupling(
    name="GC_48",
    value="(2*cqq31*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_49 = Coupling(
    name="GC_49",
    value="(4*cqq31*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_50 = Coupling(
    name="GC_50",
    value="(cqu1*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_51 = Coupling(
    name="GC_51",
    value="(cqu8*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_52 = Coupling(
    name="GC_52",
    value="(4*cHB*cth**2*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_53 = Coupling(
    name="GC_53",
    value="(-2*cHBtil*cth**2*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_54 = Coupling(
    name="GC_54",
    value="(4*cHW*cth**2*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_55 = Coupling(
    name="GC_55",
    value="(-2*cHWtil*cth**2*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_56 = Coupling(
    name="GC_56",
    value="(cud1*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_57 = Coupling(
    name="GC_57",
    value="(cud8*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_58 = Coupling(
    name="GC_58", value="(2*cuu*complex(0,1))/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_59 = Coupling(
    name="GC_59",
    value="(2*cuu1*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_60 = Coupling(
    name="GC_60",
    value="(6*cth*cW*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_61 = Coupling(
    name="GC_61",
    value="(2*cth*cWtil*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_62 = Coupling(
    name="GC_62",
    value="(4*cHW*ee*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_63 = Coupling(
    name="GC_63",
    value="(2*cHWB*ee*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_64 = Coupling(
    name="GC_64",
    value="(-2*cHWBtil*ee*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_65 = Coupling(
    name="GC_65",
    value="(-4*cHWtil*ee*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_66 = Coupling(
    name="GC_66",
    value="(6*cth*cW*ee*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_67 = Coupling(
    name="GC_67",
    value="(-2*cth*cWtil*ee*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_68 = Coupling(
    name="GC_68",
    value="(-4*cHW*ee**2*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_69 = Coupling(
    name="GC_69",
    value="(-6*cth*cW*ee**2*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_70 = Coupling(
    name="GC_70",
    value="(-4*cth*cWtil*ee**2*complex(0,1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_71 = Coupling(
    name="GC_71",
    value="(-6*cG*complex(0,1)*G)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_72 = Coupling(
    name="GC_72",
    value="(cGtil*complex(0,1)*G)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_73 = Coupling(
    name="GC_73", value="(-4*cHG*G)/LambdaSMEFT**2", order={"NP": 1, "QCD": 1, "QED": 2}
)

GC_74 = Coupling(
    name="GC_74",
    value="(4*cHGtil*G)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_75 = Coupling(
    name="GC_75",
    value="(-3*cG*G**2)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 2, "QED": 2},
)

GC_76 = Coupling(
    name="GC_76",
    value="(3*cG*G**2)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 2, "QED": 2},
)

GC_77 = Coupling(
    name="GC_77",
    value="(-2*cGtil*G**2)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 2, "QED": 2},
)

GC_78 = Coupling(
    name="GC_78",
    value="(2*cGtil*G**2)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 2, "QED": 2},
)

GC_79 = Coupling(
    name="GC_79",
    value="(-4*cHG*complex(0,1)*G**2)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 2, "QED": 2},
)

GC_80 = Coupling(
    name="GC_80",
    value="-((cG*complex(0,1)*G**3)/LambdaSMEFT**2)",
    order={"NP": 1, "QCD": 3, "QED": 2},
)

GC_81 = Coupling(
    name="GC_81",
    value="(cG*complex(0,1)*G**3)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 3, "QED": 2},
)

GC_82 = Coupling(
    name="GC_82",
    value="-((cGtil*complex(0,1)*G**3)/LambdaSMEFT**2)",
    order={"NP": 1, "QCD": 3, "QED": 2},
)

GC_83 = Coupling(
    name="GC_83",
    value="(cGtil*complex(0,1)*G**3)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 3, "QED": 2},
)

GC_84 = Coupling(
    name="GC_84", value="(-2*ee*complex(0,1)*propCorr)/3.", order={"QED": 1}
)

GC_85 = Coupling(name="GC_85", value="-(ee*complex(0,1)*propCorr)", order={"QED": 1})

GC_86 = Coupling(name="GC_86", value="ee**2*complex(0,1)*propCorr", order={"QED": 2})

GC_87 = Coupling(name="GC_87", value="-(complex(0,1)*G*propCorr)", order={"QCD": 1})

GC_88 = Coupling(name="GC_88", value="-6*complex(0,1)*lam*propCorr", order={"QED": 2})

GC_89 = Coupling(
    name="GC_89", value="(-2*ee*complex(0,1)*propCorr**2)/3.", order={"QED": 1}
)

GC_90 = Coupling(name="GC_90", value="-(ee*complex(0,1)*propCorr**2)", order={"QED": 1})

GC_91 = Coupling(name="GC_91", value="ee**2*complex(0,1)*propCorr**2", order={"QED": 2})

GC_92 = Coupling(name="GC_92", value="-(complex(0,1)*G*propCorr**2)", order={"QCD": 1})

GC_93 = Coupling(
    name="GC_93", value="-6*complex(0,1)*lam*propCorr**2", order={"QED": 2}
)

GC_94 = Coupling(
    name="GC_94", value="-6*complex(0,1)*lam*propCorr**3", order={"QED": 2}
)

GC_95 = Coupling(
    name="GC_95", value="-6*complex(0,1)*lam*propCorr**4", order={"QED": 2}
)

GC_96 = Coupling(
    name="GC_96",
    value="(-24*cth**2*cW*ee**3*complex(0,1))/(LambdaSMEFT**2*sth**3)",
    order={"NP": 1, "QED": 5},
)

GC_97 = Coupling(
    name="GC_97", value="(ee**2*complex(0,1))/(2.*sth**2)", order={"QED": 2}
)

GC_98 = Coupling(name="GC_98", value="-((ee**2*complex(0,1))/sth**2)", order={"QED": 2})

GC_99 = Coupling(
    name="GC_99", value="(ee**2*complex(0,1))/(2.*cth**2*sth**2)", order={"QED": 2}
)

GC_100 = Coupling(
    name="GC_100", value="(cth**2*ee**2*complex(0,1))/sth**2", order={"QED": 2}
)

GC_101 = Coupling(
    name="GC_101",
    value="(4*cHW*ee**2*complex(0,1))/(LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 4},
)

GC_102 = Coupling(
    name="GC_102",
    value="(3*cHDD*ee**2*complex(0,1))/(cth**2*LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 4},
)

GC_103 = Coupling(
    name="GC_103",
    value="(-4*cHW*cth**2*ee**2*complex(0,1))/(LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 4},
)

GC_104 = Coupling(
    name="GC_104",
    value="(-6*cth*cW*ee**2*complex(0,1))/(LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 4},
)

GC_105 = Coupling(
    name="GC_105",
    value="(-6*cth**3*cW*ee**2*complex(0,1))/(LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 4},
)

GC_106 = Coupling(
    name="GC_106",
    value="(-4*cth*cWtil*ee**2*complex(0,1))/(LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 4},
)

GC_107 = Coupling(
    name="GC_107",
    value="(-4*cth**3*cWtil*ee**2*complex(0,1))/(LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 4},
)

GC_108 = Coupling(
    name="GC_108",
    value="(-24*cth*cW*ee**3*complex(0,1))/(LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 5},
)

GC_109 = Coupling(
    name="GC_109", value="(ee**2*complex(0,1)*propCorr)/(2.*sth**2)", order={"QED": 2}
)

GC_110 = Coupling(
    name="GC_110", value="-((ee**2*complex(0,1)*propCorr)/sth**2)", order={"QED": 2}
)

GC_111 = Coupling(
    name="GC_111",
    value="(ee**2*complex(0,1)*propCorr)/(2.*cth**2*sth**2)",
    order={"QED": 2},
)

GC_112 = Coupling(
    name="GC_112", value="(cth**2*ee**2*complex(0,1)*propCorr)/sth**2", order={"QED": 2}
)

GC_113 = Coupling(
    name="GC_113",
    value="(ee**2*complex(0,1)*propCorr**2)/(2.*sth**2)",
    order={"QED": 2},
)

GC_114 = Coupling(
    name="GC_114", value="-((ee**2*complex(0,1)*propCorr**2)/sth**2)", order={"QED": 2}
)

GC_115 = Coupling(
    name="GC_115",
    value="(ee**2*complex(0,1)*propCorr**2)/(2.*cth**2*sth**2)",
    order={"QED": 2},
)

GC_116 = Coupling(
    name="GC_116",
    value="(cth**2*ee**2*complex(0,1)*propCorr**2)/sth**2",
    order={"QED": 2},
)

GC_117 = Coupling(
    name="GC_117",
    value="(ee**2*complex(0,1)*propCorr**3)/(2.*sth**2)",
    order={"QED": 2},
)

GC_118 = Coupling(
    name="GC_118", value="-((ee**2*complex(0,1)*propCorr**3)/sth**2)", order={"QED": 2}
)

GC_119 = Coupling(
    name="GC_119",
    value="(ee**2*complex(0,1)*propCorr**3)/(2.*cth**2*sth**2)",
    order={"QED": 2},
)

GC_120 = Coupling(
    name="GC_120",
    value="(cth**2*ee**2*complex(0,1)*propCorr**3)/sth**2",
    order={"QED": 2},
)

GC_121 = Coupling(
    name="GC_121",
    value="(ee**2*complex(0,1)*propCorr**4)/(2.*sth**2)",
    order={"QED": 2},
)

GC_122 = Coupling(
    name="GC_122", value="-((ee**2*complex(0,1)*propCorr**4)/sth**2)", order={"QED": 2}
)

GC_123 = Coupling(
    name="GC_123",
    value="(ee**2*complex(0,1)*propCorr**4)/(2.*cth**2*sth**2)",
    order={"QED": 2},
)

GC_124 = Coupling(
    name="GC_124",
    value="(cth**2*ee**2*complex(0,1)*propCorr**4)/sth**2",
    order={"QED": 2},
)

GC_125 = Coupling(
    name="GC_125", value="-((ee*complex(0,1))/(sth*cmath.sqrt(2)))", order={"QED": 1}
)

GC_126 = Coupling(
    name="GC_126",
    value="-((CKM1x1*ee*complex(0,1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_127 = Coupling(
    name="GC_127",
    value="-((CKM1x2*ee*complex(0,1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_128 = Coupling(
    name="GC_128",
    value="-((CKM1x3*ee*complex(0,1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_129 = Coupling(
    name="GC_129",
    value="-((CKM2x1*ee*complex(0,1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_130 = Coupling(
    name="GC_130",
    value="-((CKM2x2*ee*complex(0,1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_131 = Coupling(
    name="GC_131",
    value="-((CKM2x3*ee*complex(0,1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_132 = Coupling(
    name="GC_132",
    value="-((CKM3x1*ee*complex(0,1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_133 = Coupling(
    name="GC_133",
    value="-((CKM3x2*ee*complex(0,1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_134 = Coupling(
    name="GC_134",
    value="-((CKM3x3*ee*complex(0,1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_135 = Coupling(
    name="GC_135", value="-(ee*complex(0,1))/(2.*cth*sth)", order={"QED": 1}
)

GC_136 = Coupling(
    name="GC_136", value="(ee*complex(0,1))/(2.*cth*sth)", order={"QED": 1}
)

GC_137 = Coupling(name="GC_137", value="-((cth*ee*complex(0,1))/sth)", order={"QED": 1})

GC_138 = Coupling(
    name="GC_138", value="(-2*cth*ee**2*complex(0,1))/sth", order={"QED": 2}
)

GC_139 = Coupling(
    name="GC_139",
    value="-((cHl3*ee*complex(0,1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_140 = Coupling(
    name="GC_140",
    value="-((cHq3*CKM1x1*ee*complex(0,1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_141 = Coupling(
    name="GC_141",
    value="-((cHq3*CKM1x2*ee*complex(0,1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_142 = Coupling(
    name="GC_142",
    value="-((cHq3*CKM1x3*ee*complex(0,1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_143 = Coupling(
    name="GC_143",
    value="-((cHq3*CKM2x1*ee*complex(0,1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_144 = Coupling(
    name="GC_144",
    value="-((cHq3*CKM2x2*ee*complex(0,1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_145 = Coupling(
    name="GC_145",
    value="-((cHq3*CKM2x3*ee*complex(0,1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_146 = Coupling(
    name="GC_146",
    value="-((cHq3*CKM3x1*ee*complex(0,1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_147 = Coupling(
    name="GC_147",
    value="-((cHq3*CKM3x2*ee*complex(0,1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_148 = Coupling(
    name="GC_148",
    value="-((cHq3*CKM3x3*ee*complex(0,1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_149 = Coupling(
    name="GC_149",
    value="(cHd*ee*complex(0,1))/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_150 = Coupling(
    name="GC_150",
    value="(cHe*ee*complex(0,1))/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_151 = Coupling(
    name="GC_151",
    value="(cHl1*ee*complex(0,1))/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_152 = Coupling(
    name="GC_152",
    value="-((cHl3*ee*complex(0,1))/(cth*LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_153 = Coupling(
    name="GC_153",
    value="(cHl3*ee*complex(0,1))/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_154 = Coupling(
    name="GC_154",
    value="(cHq1*ee*complex(0,1))/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_155 = Coupling(
    name="GC_155",
    value="-((cHq3*ee*complex(0,1))/(cth*LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_156 = Coupling(
    name="GC_156",
    value="(cHq3*ee*complex(0,1))/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_157 = Coupling(
    name="GC_157",
    value="(cHu*ee*complex(0,1))/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_158 = Coupling(
    name="GC_158",
    value="(4*cHW*cth*ee*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_159 = Coupling(
    name="GC_159",
    value="(-2*cHWB*cth*ee*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_160 = Coupling(
    name="GC_160",
    value="(2*cHWBtil*cth*ee*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_161 = Coupling(
    name="GC_161",
    value="(-4*cHWtil*cth*ee*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_162 = Coupling(
    name="GC_162",
    value="(6*cW*ee*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_163 = Coupling(
    name="GC_163",
    value="(-6*cth**2*cW*ee*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_164 = Coupling(
    name="GC_164",
    value="(-2*cWtil*ee*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_165 = Coupling(
    name="GC_165",
    value="(2*cth**2*cWtil*ee*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_166 = Coupling(
    name="GC_166",
    value="(8*cHW*cth*ee**2*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_167 = Coupling(
    name="GC_167",
    value="(-6*cW*ee**2*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_168 = Coupling(
    name="GC_168",
    value="(-6*cth**2*cW*ee**2*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_169 = Coupling(
    name="GC_169",
    value="(4*cWtil*ee**2*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_170 = Coupling(
    name="GC_170",
    value="(-4*cth**2*cWtil*ee**2*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_171 = Coupling(
    name="GC_171",
    value="(-24*cW*ee**3*complex(0,1))/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 5},
)

GC_172 = Coupling(
    name="GC_172",
    value="-((ee*complex(0,1)*propCorr)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_173 = Coupling(
    name="GC_173",
    value="-((CKM1x1*ee*complex(0,1)*propCorr)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_174 = Coupling(
    name="GC_174",
    value="-((CKM1x2*ee*complex(0,1)*propCorr)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_175 = Coupling(
    name="GC_175",
    value="-((CKM1x3*ee*complex(0,1)*propCorr)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_176 = Coupling(
    name="GC_176",
    value="-((CKM2x1*ee*complex(0,1)*propCorr)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_177 = Coupling(
    name="GC_177",
    value="-((CKM2x2*ee*complex(0,1)*propCorr)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_178 = Coupling(
    name="GC_178",
    value="-((CKM2x3*ee*complex(0,1)*propCorr)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_179 = Coupling(
    name="GC_179",
    value="-((CKM3x1*ee*complex(0,1)*propCorr)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_180 = Coupling(
    name="GC_180",
    value="-((CKM3x2*ee*complex(0,1)*propCorr)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_181 = Coupling(
    name="GC_181",
    value="-((CKM3x3*ee*complex(0,1)*propCorr)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_182 = Coupling(
    name="GC_182", value="-(ee*complex(0,1)*propCorr)/(2.*cth*sth)", order={"QED": 1}
)

GC_183 = Coupling(
    name="GC_183", value="(ee*complex(0,1)*propCorr)/(2.*cth*sth)", order={"QED": 1}
)

GC_184 = Coupling(
    name="GC_184", value="-((cth*ee*complex(0,1)*propCorr)/sth)", order={"QED": 1}
)

GC_185 = Coupling(
    name="GC_185", value="(-2*cth*ee**2*complex(0,1)*propCorr)/sth", order={"QED": 2}
)

GC_186 = Coupling(
    name="GC_186",
    value="-((CKM3x1*ee*complex(0,1)*propCorr**2)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_187 = Coupling(
    name="GC_187",
    value="-((CKM3x2*ee*complex(0,1)*propCorr**2)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_188 = Coupling(
    name="GC_188",
    value="-((CKM3x3*ee*complex(0,1)*propCorr**2)/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_189 = Coupling(
    name="GC_189", value="-(ee*complex(0,1)*propCorr**2)/(2.*cth*sth)", order={"QED": 1}
)

GC_190 = Coupling(
    name="GC_190", value="-((cth*ee*complex(0,1)*propCorr**2)/sth)", order={"QED": 1}
)

GC_191 = Coupling(
    name="GC_191", value="(-2*cth*ee**2*complex(0,1)*propCorr**2)/sth", order={"QED": 2}
)

GC_192 = Coupling(
    name="GC_192", value="-(ee*complex(0,1)*propCorr**3)/(2.*cth*sth)", order={"QED": 1}
)

GC_193 = Coupling(
    name="GC_193", value="-((cth*ee*complex(0,1)*propCorr**3)/sth)", order={"QED": 1}
)

GC_194 = Coupling(
    name="GC_194", value="(-2*cth*ee**2*complex(0,1)*propCorr**3)/sth", order={"QED": 2}
)

GC_195 = Coupling(
    name="GC_195", value="-(ee*complex(0,1)*sth)/(3.*cth)", order={"QED": 1}
)

GC_196 = Coupling(
    name="GC_196", value="(2*ee*complex(0,1)*sth)/(3.*cth)", order={"QED": 1}
)

GC_197 = Coupling(name="GC_197", value="-((ee*complex(0,1)*sth)/cth)", order={"QED": 1})

GC_198 = Coupling(
    name="GC_198",
    value="(-4*cHB*cth*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_199 = Coupling(
    name="GC_199",
    value="(-4*cHBtil*cth*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_200 = Coupling(
    name="GC_200",
    value="(4*cHW*cth*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_201 = Coupling(
    name="GC_201",
    value="(-4*cHWB*cth*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_202 = Coupling(
    name="GC_202",
    value="(4*cHWB*cth*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_203 = Coupling(
    name="GC_203",
    value="(-2*cHWBtil*cth*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_204 = Coupling(
    name="GC_204",
    value="(2*cHWBtil*cth*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_205 = Coupling(
    name="GC_205",
    value="(4*cHWtil*cth*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_206 = Coupling(
    name="GC_206",
    value="(6*cW*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_207 = Coupling(
    name="GC_207",
    value="(2*cWtil*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_208 = Coupling(
    name="GC_208",
    value="(-6*cW*ee*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_209 = Coupling(
    name="GC_209",
    value="(2*cWtil*ee*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_210 = Coupling(
    name="GC_210",
    value="(-12*cW*ee**2*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_211 = Coupling(
    name="GC_211",
    value="(-4*cWtil*ee**2*complex(0,1)*sth)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_212 = Coupling(
    name="GC_212", value="-(ee*complex(0,1)*propCorr*sth)/(3.*cth)", order={"QED": 1}
)

GC_213 = Coupling(
    name="GC_213", value="(2*ee*complex(0,1)*propCorr*sth)/(3.*cth)", order={"QED": 1}
)

GC_214 = Coupling(
    name="GC_214", value="-((ee*complex(0,1)*propCorr*sth)/cth)", order={"QED": 1}
)

GC_215 = Coupling(
    name="GC_215",
    value="(2*ee*complex(0,1)*propCorr**2*sth)/(3.*cth)",
    order={"QED": 1},
)

GC_216 = Coupling(
    name="GC_216",
    value="(2*ee*complex(0,1)*propCorr**3*sth)/(3.*cth)",
    order={"QED": 1},
)

GC_217 = Coupling(
    name="GC_217",
    value="(4*cHB*complex(0,1)*sth**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_218 = Coupling(
    name="GC_218",
    value="(-2*cHBtil*complex(0,1)*sth**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_219 = Coupling(
    name="GC_219",
    value="(4*cHW*complex(0,1)*sth**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_220 = Coupling(
    name="GC_220",
    value="(-2*cHWtil*complex(0,1)*sth**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_221 = Coupling(
    name="GC_221",
    value="(-2*cHWB*complex(0,1))/LambdaSMEFT**2 + (4*cHWB*complex(0,1)*sth**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_222 = Coupling(
    name="GC_222",
    value="(-2*cHWBtil*complex(0,1))/LambdaSMEFT**2 + (4*cHWBtil*complex(0,1)*sth**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_223 = Coupling(
    name="GC_223",
    value="(4*cHWB*cth*complex(0,1)*sth)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (8*cHWB*cth*complex(0,1)*sth**3)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_224 = Coupling(
    name="GC_224", value="(4*complex(0,1)*gHaa)/vevhat", order={"QED": 3, "SMHLOOP": 1}
)

GC_225 = Coupling(
    name="GC_225",
    value="(4*complex(0,1)*gHgg1)/vevhat",
    order={"QCD": 2, "QED": 1, "SMHLOOP": 1},
)

GC_226 = Coupling(
    name="GC_226", value="(-4*G*gHgg1)/vevhat", order={"QCD": 3, "QED": 1, "SMHLOOP": 1}
)

GC_227 = Coupling(
    name="GC_227",
    value="(-4*complex(0,1)*G**2*gHgg1)/vevhat",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_228 = Coupling(
    name="GC_228", value="(2*complex(0,1)*gHza)/vevhat", order={"QED": 3, "SMHLOOP": 1}
)

GC_229 = Coupling(
    name="GC_229",
    value="(-4*complex(0,1)*gHgg2)/(MT**2*vevhat)",
    order={"QCD": 2, "QED": 1, "SMHLOOP": 1},
)

GC_230 = Coupling(
    name="GC_230",
    value="(-4*G*gHgg2)/(MT**2*vevhat)",
    order={"QCD": 3, "QED": 1, "SMHLOOP": 1},
)

GC_231 = Coupling(
    name="GC_231",
    value="(4*complex(0,1)*G**2*gHgg2)/(MT**2*vevhat)",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_232 = Coupling(
    name="GC_232",
    value="(-6*gHgg3)/(MT**2*vevhat)",
    order={"QCD": 2, "QED": 1, "SMHLOOP": 1},
)

GC_233 = Coupling(
    name="GC_233",
    value="(-6*complex(0,1)*G*gHgg3)/(MT**2*vevhat)",
    order={"QCD": 3, "QED": 1, "SMHLOOP": 1},
)

GC_234 = Coupling(
    name="GC_234",
    value="(-2*complex(0,1)*gHgg4)/(MT**2*vevhat)",
    order={"QCD": 2, "QED": 1, "SMHLOOP": 1},
)

GC_235 = Coupling(
    name="GC_235",
    value="(2*G*gHgg4)/(MT**2*vevhat)",
    order={"QCD": 3, "QED": 1, "SMHLOOP": 1},
)

GC_236 = Coupling(
    name="GC_236",
    value="(-2*complex(0,1)*G**2*gHgg4)/(MT**2*vevhat)",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_237 = Coupling(
    name="GC_237",
    value="(-8*complex(0,1)*G**2*gHgg4)/(MT**2*vevhat)",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_238 = Coupling(
    name="GC_238",
    value="(complex(0,1)*gHgg5)/(MT**2*vevhat)",
    order={"QCD": 2, "QED": 1, "SMHLOOP": 1},
)

GC_239 = Coupling(
    name="GC_239",
    value="(-2*G*gHgg5)/(MT**2*vevhat)",
    order={"QCD": 3, "QED": 1, "SMHLOOP": 1},
)

GC_240 = Coupling(
    name="GC_240",
    value="(2*complex(0,1)*G**2*gHgg5)/(MT**2*vevhat)",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_241 = Coupling(
    name="GC_241",
    value="(6*complex(0,1)*G**2*gHgg5)/(MT**2*vevhat)",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_242 = Coupling(
    name="GC_242",
    value="(4*complex(0,1)*gHaa*propCorr)/vevhat",
    order={"QED": 3, "SMHLOOP": 1},
)

GC_243 = Coupling(
    name="GC_243",
    value="(4*complex(0,1)*gHgg1*propCorr)/vevhat",
    order={"QCD": 2, "QED": 1, "SMHLOOP": 1},
)

GC_244 = Coupling(
    name="GC_244",
    value="(-4*G*gHgg1*propCorr)/vevhat",
    order={"QCD": 3, "QED": 1, "SMHLOOP": 1},
)

GC_245 = Coupling(
    name="GC_245",
    value="(-4*complex(0,1)*G**2*gHgg1*propCorr)/vevhat",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_246 = Coupling(
    name="GC_246",
    value="(2*complex(0,1)*gHza*propCorr)/vevhat",
    order={"QED": 3, "SMHLOOP": 1},
)

GC_247 = Coupling(
    name="GC_247",
    value="(-4*complex(0,1)*gHgg2*propCorr)/(MT**2*vevhat)",
    order={"QCD": 2, "QED": 1, "SMHLOOP": 1},
)

GC_248 = Coupling(
    name="GC_248",
    value="(-4*G*gHgg2*propCorr)/(MT**2*vevhat)",
    order={"QCD": 3, "QED": 1, "SMHLOOP": 1},
)

GC_249 = Coupling(
    name="GC_249",
    value="(4*complex(0,1)*G**2*gHgg2*propCorr)/(MT**2*vevhat)",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_250 = Coupling(
    name="GC_250",
    value="(-6*gHgg3*propCorr)/(MT**2*vevhat)",
    order={"QCD": 2, "QED": 1, "SMHLOOP": 1},
)

GC_251 = Coupling(
    name="GC_251",
    value="(-6*complex(0,1)*G*gHgg3*propCorr)/(MT**2*vevhat)",
    order={"QCD": 3, "QED": 1, "SMHLOOP": 1},
)

GC_252 = Coupling(
    name="GC_252",
    value="(-2*complex(0,1)*gHgg4*propCorr)/(MT**2*vevhat)",
    order={"QCD": 2, "QED": 1, "SMHLOOP": 1},
)

GC_253 = Coupling(
    name="GC_253",
    value="(2*G*gHgg4*propCorr)/(MT**2*vevhat)",
    order={"QCD": 3, "QED": 1, "SMHLOOP": 1},
)

GC_254 = Coupling(
    name="GC_254",
    value="(-2*complex(0,1)*G**2*gHgg4*propCorr)/(MT**2*vevhat)",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_255 = Coupling(
    name="GC_255",
    value="(-8*complex(0,1)*G**2*gHgg4*propCorr)/(MT**2*vevhat)",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_256 = Coupling(
    name="GC_256",
    value="(complex(0,1)*gHgg5*propCorr)/(MT**2*vevhat)",
    order={"QCD": 2, "QED": 1, "SMHLOOP": 1},
)

GC_257 = Coupling(
    name="GC_257",
    value="(-2*G*gHgg5*propCorr)/(MT**2*vevhat)",
    order={"QCD": 3, "QED": 1, "SMHLOOP": 1},
)

GC_258 = Coupling(
    name="GC_258",
    value="(2*complex(0,1)*G**2*gHgg5*propCorr)/(MT**2*vevhat)",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_259 = Coupling(
    name="GC_259",
    value="(6*complex(0,1)*G**2*gHgg5*propCorr)/(MT**2*vevhat)",
    order={"QCD": 4, "QED": 1, "SMHLOOP": 1},
)

GC_260 = Coupling(
    name="GC_260",
    value="(2*complex(0,1)*gHza*propCorr**2)/vevhat",
    order={"QED": 3, "SMHLOOP": 1},
)

GC_261 = Coupling(name="GC_261", value="-6*complex(0,1)*lam*vevhat", order={"QED": 1})

GC_262 = Coupling(
    name="GC_262",
    value="(90*cH*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_263 = Coupling(
    name="GC_263",
    value="(-3*cHbox*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_264 = Coupling(
    name="GC_264",
    value="-((cHDD*complex(0,1)*vevhat)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 1},
)

GC_265 = Coupling(
    name="GC_265",
    value="(4*cHG*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_266 = Coupling(
    name="GC_266",
    value="(-2*cHGtil*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_267 = Coupling(
    name="GC_267",
    value="(4*cHW*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_268 = Coupling(
    name="GC_268",
    value="(4*cHWtil*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_269 = Coupling(
    name="GC_269",
    value="(4*cHB*cth**2*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_270 = Coupling(
    name="GC_270",
    value="(-2*cHBtil*cth**2*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_271 = Coupling(
    name="GC_271",
    value="(4*cHW*cth**2*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_272 = Coupling(
    name="GC_272",
    value="(-2*cHWtil*cth**2*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_273 = Coupling(
    name="GC_273",
    value="(4*cHW*ee*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_274 = Coupling(
    name="GC_274",
    value="(2*cHWB*ee*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_275 = Coupling(
    name="GC_275",
    value="(-2*cHWBtil*ee*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_276 = Coupling(
    name="GC_276",
    value="(-4*cHWtil*ee*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_277 = Coupling(
    name="GC_277",
    value="(-4*cHW*ee**2*complex(0,1)*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_278 = Coupling(
    name="GC_278",
    value="(-4*cHG*G*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 1, "QED": 1},
)

GC_279 = Coupling(
    name="GC_279",
    value="(4*cHGtil*G*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 1, "QED": 1},
)

GC_280 = Coupling(
    name="GC_280",
    value="(-4*cHG*complex(0,1)*G**2*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 2, "QED": 1},
)

GC_281 = Coupling(
    name="GC_281", value="-6*complex(0,1)*lam*propCorr*vevhat", order={"QED": 1}
)

GC_282 = Coupling(
    name="GC_282", value="-6*complex(0,1)*lam*propCorr**2*vevhat", order={"QED": 1}
)

GC_283 = Coupling(
    name="GC_283", value="-6*complex(0,1)*lam*propCorr**3*vevhat", order={"QED": 1}
)

GC_284 = Coupling(
    name="GC_284", value="(ee**2*complex(0,1)*vevhat)/(2.*sth**2)", order={"QED": 1}
)

GC_285 = Coupling(
    name="GC_285",
    value="(ee**2*complex(0,1)*vevhat)/(2.*cth**2*sth**2)",
    order={"QED": 1},
)

GC_286 = Coupling(
    name="GC_286",
    value="(4*cHW*ee**2*complex(0,1)*vevhat)/(LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 3},
)

GC_287 = Coupling(
    name="GC_287",
    value="(3*cHDD*ee**2*complex(0,1)*vevhat)/(cth**2*LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 3},
)

GC_288 = Coupling(
    name="GC_288",
    value="(-4*cHW*cth**2*ee**2*complex(0,1)*vevhat)/(LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 3},
)

GC_289 = Coupling(
    name="GC_289",
    value="(ee**2*complex(0,1)*propCorr*vevhat)/(2.*sth**2)",
    order={"QED": 1},
)

GC_290 = Coupling(
    name="GC_290",
    value="(ee**2*complex(0,1)*propCorr*vevhat)/(2.*cth**2*sth**2)",
    order={"QED": 1},
)

GC_291 = Coupling(
    name="GC_291",
    value="(ee**2*complex(0,1)*propCorr**2*vevhat)/(2.*sth**2)",
    order={"QED": 1},
)

GC_292 = Coupling(
    name="GC_292",
    value="(ee**2*complex(0,1)*propCorr**2*vevhat)/(2.*cth**2*sth**2)",
    order={"QED": 1},
)

GC_293 = Coupling(
    name="GC_293",
    value="(ee**2*complex(0,1)*propCorr**3*vevhat)/(2.*sth**2)",
    order={"QED": 1},
)

GC_294 = Coupling(
    name="GC_294",
    value="(ee**2*complex(0,1)*propCorr**3*vevhat)/(2.*cth**2*sth**2)",
    order={"QED": 1},
)

GC_295 = Coupling(
    name="GC_295",
    value="-((cHl3*ee*complex(0,1)*vevhat*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_296 = Coupling(
    name="GC_296",
    value="-((cHq3*CKM1x1*ee*complex(0,1)*vevhat*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_297 = Coupling(
    name="GC_297",
    value="-((cHq3*CKM1x2*ee*complex(0,1)*vevhat*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_298 = Coupling(
    name="GC_298",
    value="-((cHq3*CKM1x3*ee*complex(0,1)*vevhat*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_299 = Coupling(
    name="GC_299",
    value="-((cHq3*CKM2x1*ee*complex(0,1)*vevhat*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_300 = Coupling(
    name="GC_300",
    value="-((cHq3*CKM2x2*ee*complex(0,1)*vevhat*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_301 = Coupling(
    name="GC_301",
    value="-((cHq3*CKM2x3*ee*complex(0,1)*vevhat*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_302 = Coupling(
    name="GC_302",
    value="-((cHq3*CKM3x1*ee*complex(0,1)*vevhat*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_303 = Coupling(
    name="GC_303",
    value="-((cHq3*CKM3x2*ee*complex(0,1)*vevhat*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_304 = Coupling(
    name="GC_304",
    value="-((cHq3*CKM3x3*ee*complex(0,1)*vevhat*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_305 = Coupling(
    name="GC_305",
    value="(cHd*ee*complex(0,1)*vevhat)/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_306 = Coupling(
    name="GC_306",
    value="(cHe*ee*complex(0,1)*vevhat)/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_307 = Coupling(
    name="GC_307",
    value="(cHl1*ee*complex(0,1)*vevhat)/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_308 = Coupling(
    name="GC_308",
    value="-((cHl3*ee*complex(0,1)*vevhat)/(cth*LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_309 = Coupling(
    name="GC_309",
    value="(cHl3*ee*complex(0,1)*vevhat)/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_310 = Coupling(
    name="GC_310",
    value="(cHq1*ee*complex(0,1)*vevhat)/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_311 = Coupling(
    name="GC_311",
    value="-((cHq3*ee*complex(0,1)*vevhat)/(cth*LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_312 = Coupling(
    name="GC_312",
    value="(cHq3*ee*complex(0,1)*vevhat)/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_313 = Coupling(
    name="GC_313",
    value="(cHu*ee*complex(0,1)*vevhat)/(cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_314 = Coupling(
    name="GC_314",
    value="(4*cHW*cth*ee*complex(0,1)*vevhat)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_315 = Coupling(
    name="GC_315",
    value="(-2*cHWB*cth*ee*complex(0,1)*vevhat)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_316 = Coupling(
    name="GC_316",
    value="(2*cHWBtil*cth*ee*complex(0,1)*vevhat)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_317 = Coupling(
    name="GC_317",
    value="(-4*cHWtil*cth*ee*complex(0,1)*vevhat)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 2},
)

GC_318 = Coupling(
    name="GC_318",
    value="(8*cHW*cth*ee**2*complex(0,1)*vevhat)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_319 = Coupling(
    name="GC_319",
    value="(-4*cHB*cth*complex(0,1)*sth*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_320 = Coupling(
    name="GC_320",
    value="(-4*cHBtil*cth*complex(0,1)*sth*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_321 = Coupling(
    name="GC_321",
    value="(4*cHW*cth*complex(0,1)*sth*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_322 = Coupling(
    name="GC_322",
    value="(-4*cHWB*cth*complex(0,1)*sth*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_323 = Coupling(
    name="GC_323",
    value="(4*cHWB*cth*complex(0,1)*sth*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_324 = Coupling(
    name="GC_324",
    value="(-2*cHWBtil*cth*complex(0,1)*sth*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_325 = Coupling(
    name="GC_325",
    value="(2*cHWBtil*cth*complex(0,1)*sth*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_326 = Coupling(
    name="GC_326",
    value="(4*cHWtil*cth*complex(0,1)*sth*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_327 = Coupling(
    name="GC_327",
    value="(4*cHB*complex(0,1)*sth**2*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_328 = Coupling(
    name="GC_328",
    value="(-2*cHBtil*complex(0,1)*sth**2*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_329 = Coupling(
    name="GC_329",
    value="(4*cHW*complex(0,1)*sth**2*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_330 = Coupling(
    name="GC_330",
    value="(-2*cHWtil*complex(0,1)*sth**2*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_331 = Coupling(
    name="GC_331",
    value="(36*cH*complex(0,1)*vevhat**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_332 = Coupling(
    name="GC_332",
    value="-((cHWBtil*ee*complex(0,1)*vevhat**2)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 1},
)

GC_333 = Coupling(
    name="GC_333",
    value="(-2*cHWtil*ee*complex(0,1)*vevhat**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_334 = Coupling(
    name="GC_334",
    value="(2*cHGtil*G*vevhat**2)/LambdaSMEFT**2",
    order={"NP": 1, "QCD": 1},
)

GC_335 = Coupling(
    name="GC_335",
    value="(-12*cHbox*complex(0,1)*lam*vevhat**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_336 = Coupling(
    name="GC_336",
    value="(3*cHDD*complex(0,1)*lam*vevhat**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_337 = Coupling(
    name="GC_337",
    value="(12*cHl3*complex(0,1)*lam*vevhat**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_338 = Coupling(
    name="GC_338",
    value="(-6*cll1*complex(0,1)*lam*vevhat**2)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_339 = Coupling(
    name="GC_339",
    value="(cHbox*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 2},
)

GC_340 = Coupling(
    name="GC_340",
    value="(cHbox*ee**2*complex(0,1)*vevhat**2)/(cth**2*LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 2},
)

GC_341 = Coupling(
    name="GC_341",
    value="-((cHq3*CKM1x1*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_342 = Coupling(
    name="GC_342",
    value="-((cHq3*CKM1x2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_343 = Coupling(
    name="GC_343",
    value="-((cHq3*CKM1x3*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_344 = Coupling(
    name="GC_344",
    value="-((cHq3*CKM2x1*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_345 = Coupling(
    name="GC_345",
    value="-((cHq3*CKM2x2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_346 = Coupling(
    name="GC_346",
    value="-((cHq3*CKM2x3*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_347 = Coupling(
    name="GC_347",
    value="-((cHq3*CKM3x1*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_348 = Coupling(
    name="GC_348",
    value="-((cHq3*CKM3x2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_349 = Coupling(
    name="GC_349",
    value="-((cHq3*CKM3x3*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_350 = Coupling(
    name="GC_350",
    value="(cHd*ee*complex(0,1)*vevhat**2)/(2.*cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_351 = Coupling(
    name="GC_351",
    value="(cHDD*ee*complex(0,1)*vevhat**2)/(8.*cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_352 = Coupling(
    name="GC_352",
    value="(cHe*ee*complex(0,1)*vevhat**2)/(2.*cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_353 = Coupling(
    name="GC_353",
    value="(cHl1*ee*complex(0,1)*vevhat**2)/(2.*cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_354 = Coupling(
    name="GC_354",
    value="(cHq1*ee*complex(0,1)*vevhat**2)/(2.*cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_355 = Coupling(
    name="GC_355",
    value="-(cHq3*ee*complex(0,1)*vevhat**2)/(2.*cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_356 = Coupling(
    name="GC_356",
    value="(cHq3*ee*complex(0,1)*vevhat**2)/(2.*cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_357 = Coupling(
    name="GC_357",
    value="(cHu*ee*complex(0,1)*vevhat**2)/(2.*cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_358 = Coupling(
    name="GC_358",
    value="-(cll1*ee*complex(0,1)*vevhat**2)/(4.*cth*LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_359 = Coupling(
    name="GC_359",
    value="(cHWBtil*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_360 = Coupling(
    name="GC_360",
    value="(-2*cHWtil*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 1},
)

GC_361 = Coupling(
    name="GC_361",
    value="(2*cHWB*ee*complex(0,1)*vevhat**2)/(3.*LambdaSMEFT**2*(1 - 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_362 = Coupling(
    name="GC_362",
    value="(cll1*cth**2*ee**2*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*sth**2*(1 - 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_363 = Coupling(
    name="GC_363",
    value="(cHWB*ee*complex(0,1)*vevhat**2)/(3.*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_364 = Coupling(
    name="GC_364",
    value="(cHWB*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_365 = Coupling(
    name="GC_365",
    value="-((cHWB*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_366 = Coupling(
    name="GC_366",
    value="-((cHWB*CKM1x1*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_367 = Coupling(
    name="GC_367",
    value="-((cHWB*CKM1x2*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_368 = Coupling(
    name="GC_368",
    value="-((cHWB*CKM1x3*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_369 = Coupling(
    name="GC_369",
    value="-((cHWB*CKM2x1*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_370 = Coupling(
    name="GC_370",
    value="-((cHWB*CKM2x2*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_371 = Coupling(
    name="GC_371",
    value="-((cHWB*CKM2x3*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_372 = Coupling(
    name="GC_372",
    value="-((cHWB*CKM3x1*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_373 = Coupling(
    name="GC_373",
    value="-((cHWB*CKM3x2*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_374 = Coupling(
    name="GC_374",
    value="-((cHWB*CKM3x3*cth*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_375 = Coupling(
    name="GC_375",
    value="-((cHWB*cth**2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)))",
    order={"NP": 1, "QED": 1},
)

GC_376 = Coupling(
    name="GC_376",
    value="-(cHDD*cth**2*ee**2*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_377 = Coupling(
    name="GC_377",
    value="(cHDD*cth**2*ee**2*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_378 = Coupling(
    name="GC_378",
    value="(cHl3*cth**2*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_379 = Coupling(
    name="GC_379",
    value="(-2*cHl3*cth**2*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_380 = Coupling(
    name="GC_380",
    value="(2*cHl3*cth**2*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_381 = Coupling(
    name="GC_381",
    value="-((cll1*cth**2*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)))",
    order={"NP": 1, "QED": 2},
)

GC_382 = Coupling(
    name="GC_382",
    value="(cll1*cth**2*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_383 = Coupling(
    name="GC_383",
    value="-(cHDD*ee*complex(0,1)*vevhat**2)/(8.*cth*LambdaSMEFT**2*sth*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_384 = Coupling(
    name="GC_384",
    value="(cHDD*ee*complex(0,1)*vevhat**2)/(8.*cth*LambdaSMEFT**2*sth*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_385 = Coupling(
    name="GC_385",
    value="-(cHl3*ee*complex(0,1)*vevhat**2)/(2.*cth*LambdaSMEFT**2*sth*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_386 = Coupling(
    name="GC_386",
    value="(cHl3*ee*complex(0,1)*vevhat**2)/(2.*cth*LambdaSMEFT**2*sth*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_387 = Coupling(
    name="GC_387",
    value="-(cll1*ee*complex(0,1)*vevhat**2)/(4.*cth*LambdaSMEFT**2*sth*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_388 = Coupling(
    name="GC_388",
    value="(cll1*ee*complex(0,1)*vevhat**2)/(4.*cth*LambdaSMEFT**2*sth*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_389 = Coupling(
    name="GC_389",
    value="-((cHl3*ee*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_390 = Coupling(
    name="GC_390",
    value="(cHDD*ee*complex(0,1)*sth*vevhat**2)/(12.*cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_391 = Coupling(
    name="GC_391",
    value="-(cHDD*ee*complex(0,1)*sth*vevhat**2)/(6.*cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_392 = Coupling(
    name="GC_392",
    value="(cHDD*ee*complex(0,1)*sth*vevhat**2)/(4.*cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_393 = Coupling(
    name="GC_393",
    value="(cHl3*ee*complex(0,1)*sth*vevhat**2)/(3.*cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_394 = Coupling(
    name="GC_394",
    value="(-2*cHl3*ee*complex(0,1)*sth*vevhat**2)/(3.*cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_395 = Coupling(
    name="GC_395",
    value="(cHl3*ee*complex(0,1)*sth*vevhat**2)/(cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_396 = Coupling(
    name="GC_396",
    value="-(cll1*ee*complex(0,1)*sth*vevhat**2)/(6.*cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_397 = Coupling(
    name="GC_397",
    value="(cll1*ee*complex(0,1)*sth*vevhat**2)/(3.*cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_398 = Coupling(
    name="GC_398",
    value="-(cll1*ee*complex(0,1)*sth*vevhat**2)/(2.*cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_399 = Coupling(
    name="GC_399",
    value="-((cHWB*cth**2*ee*complex(0,1)*sth**2*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)))",
    order={"NP": 1, "QED": 1},
)

GC_400 = Coupling(
    name="GC_400",
    value="-((cHl3*CKM1x1*cth**2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_401 = Coupling(
    name="GC_401",
    value="-((cHl3*CKM1x2*cth**2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_402 = Coupling(
    name="GC_402",
    value="-((cHl3*CKM1x3*cth**2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_403 = Coupling(
    name="GC_403",
    value="-((cHl3*CKM2x1*cth**2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_404 = Coupling(
    name="GC_404",
    value="-((cHl3*CKM2x2*cth**2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_405 = Coupling(
    name="GC_405",
    value="-((cHl3*CKM2x3*cth**2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_406 = Coupling(
    name="GC_406",
    value="-((cHl3*CKM3x1*cth**2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_407 = Coupling(
    name="GC_407",
    value="-((cHl3*CKM3x2*cth**2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_408 = Coupling(
    name="GC_408",
    value="-((cHl3*CKM3x3*cth**2*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_409 = Coupling(
    name="GC_409",
    value="(cll1*cth**2*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_410 = Coupling(
    name="GC_410",
    value="(CKM1x1*cll1*cth**2*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_411 = Coupling(
    name="GC_411",
    value="(CKM1x2*cll1*cth**2*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_412 = Coupling(
    name="GC_412",
    value="(CKM1x3*cll1*cth**2*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_413 = Coupling(
    name="GC_413",
    value="(CKM2x1*cll1*cth**2*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_414 = Coupling(
    name="GC_414",
    value="(CKM2x2*cll1*cth**2*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_415 = Coupling(
    name="GC_415",
    value="(CKM2x3*cll1*cth**2*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_416 = Coupling(
    name="GC_416",
    value="(CKM3x1*cll1*cth**2*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_417 = Coupling(
    name="GC_417",
    value="(CKM3x2*cll1*cth**2*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_418 = Coupling(
    name="GC_418",
    value="(CKM3x3*cll1*cth**2*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_419 = Coupling(
    name="GC_419",
    value="(cHDD*cth**2*ee*complex(0,1)*vevhat**2)/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_420 = Coupling(
    name="GC_420",
    value="(cHDD*CKM1x1*cth**2*ee*complex(0,1)*vevhat**2)/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_421 = Coupling(
    name="GC_421",
    value="(cHDD*CKM1x2*cth**2*ee*complex(0,1)*vevhat**2)/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_422 = Coupling(
    name="GC_422",
    value="(cHDD*CKM1x3*cth**2*ee*complex(0,1)*vevhat**2)/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_423 = Coupling(
    name="GC_423",
    value="(cHDD*CKM2x1*cth**2*ee*complex(0,1)*vevhat**2)/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_424 = Coupling(
    name="GC_424",
    value="(cHDD*CKM2x2*cth**2*ee*complex(0,1)*vevhat**2)/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_425 = Coupling(
    name="GC_425",
    value="(cHDD*CKM2x3*cth**2*ee*complex(0,1)*vevhat**2)/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_426 = Coupling(
    name="GC_426",
    value="(cHDD*CKM3x1*cth**2*ee*complex(0,1)*vevhat**2)/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_427 = Coupling(
    name="GC_427",
    value="(cHDD*CKM3x2*cth**2*ee*complex(0,1)*vevhat**2)/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_428 = Coupling(
    name="GC_428",
    value="(cHDD*CKM3x3*cth**2*ee*complex(0,1)*vevhat**2)/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_429 = Coupling(
    name="GC_429",
    value="(6*cH*complex(0,1)*vevhat**3)/LambdaSMEFT**2",
    order={
        "NP": 1,
    },
)

GC_430 = Coupling(
    name="GC_430",
    value="(-6*cHbox*complex(0,1)*lam*vevhat**3)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_431 = Coupling(
    name="GC_431",
    value="(3*cHDD*complex(0,1)*lam*vevhat**3)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 1},
)

GC_432 = Coupling(
    name="GC_432",
    value="(6*cHl3*complex(0,1)*lam*vevhat**3)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_433 = Coupling(
    name="GC_433",
    value="(-3*cll1*complex(0,1)*lam*vevhat**3)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_434 = Coupling(
    name="GC_434",
    value="(cHbox*ee**2*complex(0,1)*vevhat**3)/(2.*LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 1},
)

GC_435 = Coupling(
    name="GC_435",
    value="(cHbox*ee**2*complex(0,1)*vevhat**3)/(2.*cth**2*LambdaSMEFT**2*sth**2)",
    order={"NP": 1, "QED": 1},
)

GC_436 = Coupling(
    name="GC_436",
    value="(-2*cHWB*complex(0,1)*vevhat)/LambdaSMEFT**2 + (4*cHWB*complex(0,1)*sth**2*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_437 = Coupling(
    name="GC_437",
    value="(-2*cHWBtil*complex(0,1)*vevhat)/LambdaSMEFT**2 + (4*cHWBtil*complex(0,1)*sth**2*vevhat)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 1},
)

GC_438 = Coupling(
    name="GC_438",
    value="(4*cHWB*cth*complex(0,1)*sth*vevhat)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (8*cHWB*cth*complex(0,1)*sth**3*vevhat)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_439 = Coupling(
    name="GC_439",
    value="-(cHDD*ee**2*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*(-1 + 2*sth**2)) + (cHDD*ee**2*complex(0,1)*vevhat**2)/(4.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) + (cHDD*cth**2*ee**2*complex(0,1)*vevhat**2)/(4.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_440 = Coupling(
    name="GC_440",
    value="(cHWB*cth**3*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*(-1 + 2*sth**2)) - (2*cHWB*cth**3*ee*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_441 = Coupling(
    name="GC_441",
    value="(cHWB*cth**3*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*(-1 + 2*sth**2)) + (cHWB*cth*ee**2*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_442 = Coupling(
    name="GC_442",
    value="(-2*cHWB*cth**3*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*(-1 + 2*sth**2)) - (2*cHWB*cth*ee**2*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_443 = Coupling(
    name="GC_443",
    value="(5*cHDD*ee**2*complex(0,1)*vevhat**2)/(2.*cth**2*LambdaSMEFT**2*(-1 + 2*sth**2)) - (5*cHDD*ee**2*complex(0,1)*vevhat**2)/(4.*cth**2*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) + (cHDD*cth**2*ee**2*complex(0,1)*vevhat**2)/(4.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) - (cHDD*ee**2*complex(0,1)*sth**2*vevhat**2)/(4.*cth**2*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_444 = Coupling(
    name="GC_444",
    value="(cHl3*cth**2*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) - (cHl3*ee**2*complex(0,1)*sth**2*vevhat**2)/(cth**2*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_445 = Coupling(
    name="GC_445",
    value="-(cll1*cth**2*ee**2*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) + (cll1*ee**2*complex(0,1)*sth**2*vevhat**2)/(2.*cth**2*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_446 = Coupling(
    name="GC_446",
    value="-(cHDD*cth**3*ee*complex(0,1)*vevhat**2)/(4.*LambdaSMEFT**2*sth*(-1 + 2*sth**2)) - (cHDD*cth**3*ee*complex(0,1)*sth*vevhat**2)/(4.*LambdaSMEFT**2*(-1 + 2*sth**2)) - (cHDD*cth*ee*complex(0,1)*sth**3*vevhat**2)/(4.*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_447 = Coupling(
    name="GC_447",
    value="-((cHl3*cth**3*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*(-1 + 2*sth**2))) - (cHl3*cth**3*ee*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (cHl3*cth*ee*complex(0,1)*sth**3*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_448 = Coupling(
    name="GC_448",
    value="(cHWB*cth*ee*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (2*cHWB*cth*ee*complex(0,1)*sth**3*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_449 = Coupling(
    name="GC_449",
    value="(cll1*cth**3*ee*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*sth*(-1 + 2*sth**2)) + (cll1*cth**3*ee*complex(0,1)*sth*vevhat**2)/(2.*LambdaSMEFT**2*(-1 + 2*sth**2)) + (cll1*cth*ee*complex(0,1)*sth**3*vevhat**2)/(2.*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_450 = Coupling(
    name="GC_450",
    value="-((cHWB*ee**2*complex(0,1)*vevhat**2)/(cth*LambdaSMEFT**2*sth*(-1 + 2*sth**2))) + (cHWB*cth**3*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*(-1 + 2*sth**2)) + (2*cHWB*ee**2*complex(0,1)*sth*vevhat**2)/(cth*LambdaSMEFT**2*(-1 + 2*sth**2)) - (cHWB*ee**2*complex(0,1)*sth**3*vevhat**2)/(cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_451 = Coupling(
    name="GC_451",
    value="-(cHDD*cth**3*ee**2*complex(0,1)*vevhat**2)/(2.*LambdaSMEFT**2*sth*(-1 + 2*sth**2)) + (cHDD*cth*ee**2*complex(0,1)*sth*vevhat**2)/(2.*LambdaSMEFT**2*(-1 + 2*sth**2)) - (cHDD*cth**3*ee**2*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (cHDD*cth*ee**2*complex(0,1)*sth**3*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_452 = Coupling(
    name="GC_452",
    value="(-2*cHl3*cth**3*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*(-1 + 2*sth**2)) + (2*cHl3*cth*ee**2*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (4*cHl3*cth**3*ee**2*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (4*cHl3*cth*ee**2*complex(0,1)*sth**3*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_453 = Coupling(
    name="GC_453",
    value="(cll1*cth**3*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*(-1 + 2*sth**2)) - (cll1*cth*ee**2*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) + (2*cll1*cth**3*ee**2*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) + (2*cll1*cth*ee**2*complex(0,1)*sth**3*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_454 = Coupling(
    name="GC_454",
    value="(cHWB*ee*complex(0,1)*sth**2*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (2*cHWB*ee*complex(0,1)*sth**4*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_455 = Coupling(
    name="GC_455",
    value="-((cHWB*cth**4*ee*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))) - (2*cHWB*cth**4*ee*complex(0,1)*sth**2*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (2*cHWB*cth**2*ee*complex(0,1)*sth**4*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_456 = Coupling(
    name="GC_456",
    value="(-2*cHWB*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) + (8*cHWB*ee**2*complex(0,1)*sth**2*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (8*cHWB*cth**4*ee**2*complex(0,1)*sth**2*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (8*cHWB*ee**2*complex(0,1)*sth**4*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (8*cHWB*cth**2*ee**2*complex(0,1)*sth**4*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_457 = Coupling(
    name="GC_457",
    value="(cHWB*cth**5*ee*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (cHWB*cth*ee*complex(0,1)*sth**5*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_458 = Coupling(
    name="GC_458",
    value="(2*cHWB*cth**3*ee**2*complex(0,1)*vevhat**2)/(LambdaSMEFT**2*sth*(-1 + 2*sth**2)) + (2*cHWB*cth**5*ee**2*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) + (4*cHWB*cth*ee**2*complex(0,1)*sth**3*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (2*cHWB*cth*ee**2*complex(0,1)*sth**5*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_459 = Coupling(
    name="GC_459",
    value="(2*cHWB*cth*ee**2*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (2*cHWB*cth**5*ee**2*complex(0,1)*sth*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (4*cHWB*cth*ee**2*complex(0,1)*sth**3*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2)) + (2*cHWB*cth*ee**2*complex(0,1)*sth**5*vevhat**2)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 2},
)

GC_460 = Coupling(
    name="GC_460",
    value="-(cHDD*ee**2*complex(0,1)*vevhat**3)/(4.*LambdaSMEFT**2*(-1 + 2*sth**2)) + (cHDD*ee**2*complex(0,1)*vevhat**3)/(8.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) + (cHDD*cth**2*ee**2*complex(0,1)*vevhat**3)/(4.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_461 = Coupling(
    name="GC_461",
    value="(cHl3*ee**2*complex(0,1)*vevhat**3)/(LambdaSMEFT**2*(-1 + 2*sth**2)) - (cHl3*ee**2*complex(0,1)*vevhat**3)/(2.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) + (cHl3*cth**2*ee**2*complex(0,1)*vevhat**3)/(LambdaSMEFT**2*sth**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_462 = Coupling(
    name="GC_462",
    value="-(cll1*ee**2*complex(0,1)*vevhat**3)/(2.*LambdaSMEFT**2*(-1 + 2*sth**2)) + (cll1*ee**2*complex(0,1)*vevhat**3)/(4.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) - (cll1*cth**2*ee**2*complex(0,1)*vevhat**3)/(2.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_463 = Coupling(
    name="GC_463",
    value="(cHWB*cth**3*ee**2*complex(0,1)*vevhat**3)/(LambdaSMEFT**2*sth*(-1 + 2*sth**2)) + (cHWB*cth*ee**2*complex(0,1)*sth*vevhat**3)/(LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_464 = Coupling(
    name="GC_464",
    value="(3*cHDD*ee**2*complex(0,1)*vevhat**3)/(4.*cth**2*LambdaSMEFT**2*(-1 + 2*sth**2)) - (3*cHDD*ee**2*complex(0,1)*vevhat**3)/(8.*cth**2*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) + (cHDD*cth**2*ee**2*complex(0,1)*vevhat**3)/(4.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) - (cHDD*ee**2*complex(0,1)*sth**2*vevhat**3)/(4.*cth**2*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_465 = Coupling(
    name="GC_465",
    value="(cHl3*ee**2*complex(0,1)*vevhat**3)/(cth**2*LambdaSMEFT**2*(-1 + 2*sth**2)) - (cHl3*ee**2*complex(0,1)*vevhat**3)/(2.*cth**2*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) + (cHl3*cth**2*ee**2*complex(0,1)*vevhat**3)/(LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) - (cHl3*ee**2*complex(0,1)*sth**2*vevhat**3)/(cth**2*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_466 = Coupling(
    name="GC_466",
    value="-(cll1*ee**2*complex(0,1)*vevhat**3)/(2.*cth**2*LambdaSMEFT**2*(-1 + 2*sth**2)) + (cll1*ee**2*complex(0,1)*vevhat**3)/(4.*cth**2*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) - (cll1*cth**2*ee**2*complex(0,1)*vevhat**3)/(2.*LambdaSMEFT**2*sth**2*(-1 + 2*sth**2)) + (cll1*ee**2*complex(0,1)*sth**2*vevhat**3)/(2.*cth**2*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_467 = Coupling(
    name="GC_467",
    value="-((cHWB*ee**2*complex(0,1)*vevhat**3)/(cth*LambdaSMEFT**2*sth*(-1 + 2*sth**2))) + (cHWB*cth**3*ee**2*complex(0,1)*vevhat**3)/(LambdaSMEFT**2*sth*(-1 + 2*sth**2)) + (2*cHWB*ee**2*complex(0,1)*sth*vevhat**3)/(cth*LambdaSMEFT**2*(-1 + 2*sth**2)) - (cHWB*ee**2*complex(0,1)*sth**3*vevhat**3)/(cth*LambdaSMEFT**2*(-1 + 2*sth**2))",
    order={"NP": 1, "QED": 1},
)

GC_468 = Coupling(
    name="GC_468", value="-((complex(0,1)*yb)/cmath.sqrt(2))", order={"QED": 1}
)

GC_469 = Coupling(
    name="GC_469",
    value="-((cdGIm*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_470 = Coupling(
    name="GC_470",
    value="(cdGRe*complex(0,1)*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_471 = Coupling(
    name="GC_471",
    value="(-3*cdHIm*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_472 = Coupling(
    name="GC_472",
    value="(3*cdHRe*complex(0,1)*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_473 = Coupling(
    name="GC_473", value="-((cdWIm*yb)/LambdaSMEFT**2)", order={"NP": 1, "QED": 3}
)

GC_474 = Coupling(
    name="GC_474", value="(cdWIm*yb)/LambdaSMEFT**2", order={"NP": 1, "QED": 3}
)

GC_475 = Coupling(
    name="GC_475",
    value="(cdWRe*complex(0,1)*yb)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_476 = Coupling(
    name="GC_476",
    value="-((cdBIm*cth*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_477 = Coupling(
    name="GC_477",
    value="(cdBRe*cth*complex(0,1)*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_478 = Coupling(
    name="GC_478",
    value="(cdWIm*cth*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_479 = Coupling(
    name="GC_479",
    value="-((cdWRe*cth*complex(0,1)*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_480 = Coupling(
    name="GC_480", value="(cdWIm*ee*yb)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_481 = Coupling(
    name="GC_481",
    value="-((cdWRe*ee*complex(0,1)*yb)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_482 = Coupling(
    name="GC_482",
    value="(cdWRe*ee*complex(0,1)*yb)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_483 = Coupling(
    name="GC_483",
    value="(cdGIm*complex(0,1)*G*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_484 = Coupling(
    name="GC_484",
    value="(cdGRe*G*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_485 = Coupling(
    name="GC_485", value="-((complex(0,1)*propCorr*yb)/cmath.sqrt(2))", order={"QED": 1}
)

GC_486 = Coupling(
    name="GC_486",
    value="-((cdWIm*ee*yb)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_487 = Coupling(
    name="GC_487",
    value="(cdWRe*ee*complex(0,1)*yb)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_488 = Coupling(
    name="GC_488",
    value="-((cdWIm*cth*ee*yb)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_489 = Coupling(
    name="GC_489",
    value="-((cdWRe*cth*ee*complex(0,1)*yb)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_490 = Coupling(
    name="GC_490",
    value="(cdWRe*cth*ee*complex(0,1)*yb)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_491 = Coupling(
    name="GC_491",
    value="(cdBIm*sth*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_492 = Coupling(
    name="GC_492",
    value="-((cdBRe*complex(0,1)*sth*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_493 = Coupling(
    name="GC_493",
    value="(cdWIm*sth*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_494 = Coupling(
    name="GC_494",
    value="-((cdWRe*complex(0,1)*sth*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_495 = Coupling(
    name="GC_495",
    value="-((cdGIm*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_496 = Coupling(
    name="GC_496",
    value="(cdGRe*complex(0,1)*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_497 = Coupling(
    name="GC_497",
    value="(-3*cdHIm*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_498 = Coupling(
    name="GC_498",
    value="(3*cdHRe*complex(0,1)*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_499 = Coupling(
    name="GC_499",
    value="-((cdWIm*vevhat*yb)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_500 = Coupling(
    name="GC_500", value="(cdWIm*vevhat*yb)/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_501 = Coupling(
    name="GC_501",
    value="(cdWRe*complex(0,1)*vevhat*yb)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_502 = Coupling(
    name="GC_502",
    value="-((cdBIm*cth*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_503 = Coupling(
    name="GC_503",
    value="(cdBRe*cth*complex(0,1)*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_504 = Coupling(
    name="GC_504",
    value="(cdWIm*cth*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_505 = Coupling(
    name="GC_505",
    value="-((cdWRe*cth*complex(0,1)*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_506 = Coupling(
    name="GC_506",
    value="(cdWIm*ee*vevhat*yb)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_507 = Coupling(
    name="GC_507",
    value="-((cdWRe*ee*complex(0,1)*vevhat*yb)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_508 = Coupling(
    name="GC_508",
    value="(cdWRe*ee*complex(0,1)*vevhat*yb)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_509 = Coupling(
    name="GC_509",
    value="(cdGIm*complex(0,1)*G*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_510 = Coupling(
    name="GC_510",
    value="(cdGRe*G*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_511 = Coupling(
    name="GC_511",
    value="-((cdWIm*ee*vevhat*yb)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_512 = Coupling(
    name="GC_512",
    value="(cdWRe*ee*complex(0,1)*vevhat*yb)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_513 = Coupling(
    name="GC_513",
    value="-((cdWIm*cth*ee*vevhat*yb)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_514 = Coupling(
    name="GC_514",
    value="-((cdWRe*cth*ee*complex(0,1)*vevhat*yb)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_515 = Coupling(
    name="GC_515",
    value="(cdWRe*cth*ee*complex(0,1)*vevhat*yb)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_516 = Coupling(
    name="GC_516",
    value="(cdBIm*sth*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_517 = Coupling(
    name="GC_517",
    value="-((cdBRe*complex(0,1)*sth*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_518 = Coupling(
    name="GC_518",
    value="(cdWIm*sth*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_519 = Coupling(
    name="GC_519",
    value="-((cdWRe*complex(0,1)*sth*vevhat*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_520 = Coupling(
    name="GC_520",
    value="-((cdHIm*vevhat**2*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_521 = Coupling(
    name="GC_521",
    value="(cdHRe*complex(0,1)*vevhat**2*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_522 = Coupling(
    name="GC_522",
    value="-((cHbox*complex(0,1)*vevhat**2*yb)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_523 = Coupling(
    name="GC_523",
    value="(cHDD*complex(0,1)*vevhat**2*yb)/(4.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_524 = Coupling(
    name="GC_524",
    value="(cHl3*complex(0,1)*vevhat**2*yb)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_525 = Coupling(
    name="GC_525",
    value="-(cll1*complex(0,1)*vevhat**2*yb)/(2.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_526 = Coupling(
    name="GC_526", value="-((complex(0,1)*yc)/cmath.sqrt(2))", order={"QED": 1}
)

GC_527 = Coupling(
    name="GC_527",
    value="-((cth*cuBIm*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_528 = Coupling(
    name="GC_528",
    value="(cth*cuBRe*complex(0,1)*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_529 = Coupling(
    name="GC_529",
    value="-((cuGIm*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_530 = Coupling(
    name="GC_530",
    value="(cuGRe*complex(0,1)*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_531 = Coupling(
    name="GC_531",
    value="(-3*cuHIm*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_532 = Coupling(
    name="GC_532",
    value="(3*cuHRe*complex(0,1)*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_533 = Coupling(
    name="GC_533", value="-((cuWIm*yc)/LambdaSMEFT**2)", order={"NP": 1, "QED": 3}
)

GC_534 = Coupling(
    name="GC_534", value="(cuWIm*yc)/LambdaSMEFT**2", order={"NP": 1, "QED": 3}
)

GC_535 = Coupling(
    name="GC_535",
    value="-((cth*cuWIm*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_536 = Coupling(
    name="GC_536",
    value="(cuWRe*complex(0,1)*yc)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_537 = Coupling(
    name="GC_537",
    value="(cth*cuWRe*complex(0,1)*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_538 = Coupling(
    name="GC_538", value="-((cuWIm*ee*yc)/LambdaSMEFT**2)", order={"NP": 1, "QED": 4}
)

GC_539 = Coupling(
    name="GC_539",
    value="-((cuWRe*ee*complex(0,1)*yc)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_540 = Coupling(
    name="GC_540",
    value="(cuWRe*ee*complex(0,1)*yc)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_541 = Coupling(
    name="GC_541",
    value="(cuGIm*complex(0,1)*G*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_542 = Coupling(
    name="GC_542",
    value="(cuGRe*G*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_543 = Coupling(
    name="GC_543", value="-((complex(0,1)*propCorr*yc)/cmath.sqrt(2))", order={"QED": 1}
)

GC_544 = Coupling(
    name="GC_544",
    value="(cuWIm*ee*yc)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_545 = Coupling(
    name="GC_545",
    value="(cth*cuWIm*ee*yc)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_546 = Coupling(
    name="GC_546",
    value="-((cuWRe*ee*complex(0,1)*yc)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_547 = Coupling(
    name="GC_547",
    value="-((cth*cuWRe*ee*complex(0,1)*yc)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_548 = Coupling(
    name="GC_548",
    value="(cth*cuWRe*ee*complex(0,1)*yc)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_549 = Coupling(
    name="GC_549",
    value="(cuBIm*sth*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_550 = Coupling(
    name="GC_550",
    value="-((cuBRe*complex(0,1)*sth*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_551 = Coupling(
    name="GC_551",
    value="-((cuWIm*sth*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_552 = Coupling(
    name="GC_552",
    value="(cuWRe*complex(0,1)*sth*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_553 = Coupling(
    name="GC_553",
    value="-((cth*cuBIm*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_554 = Coupling(
    name="GC_554",
    value="(cth*cuBRe*complex(0,1)*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_555 = Coupling(
    name="GC_555",
    value="-((cuGIm*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_556 = Coupling(
    name="GC_556",
    value="(cuGRe*complex(0,1)*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_557 = Coupling(
    name="GC_557",
    value="(-3*cuHIm*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_558 = Coupling(
    name="GC_558",
    value="(3*cuHRe*complex(0,1)*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_559 = Coupling(
    name="GC_559",
    value="-((cuWIm*vevhat*yc)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_560 = Coupling(
    name="GC_560", value="(cuWIm*vevhat*yc)/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_561 = Coupling(
    name="GC_561",
    value="-((cth*cuWIm*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_562 = Coupling(
    name="GC_562",
    value="(cuWRe*complex(0,1)*vevhat*yc)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_563 = Coupling(
    name="GC_563",
    value="(cth*cuWRe*complex(0,1)*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_564 = Coupling(
    name="GC_564",
    value="-((cuWIm*ee*vevhat*yc)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_565 = Coupling(
    name="GC_565",
    value="-((cuWRe*ee*complex(0,1)*vevhat*yc)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_566 = Coupling(
    name="GC_566",
    value="(cuWRe*ee*complex(0,1)*vevhat*yc)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_567 = Coupling(
    name="GC_567",
    value="(cuGIm*complex(0,1)*G*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_568 = Coupling(
    name="GC_568",
    value="(cuGRe*G*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_569 = Coupling(
    name="GC_569",
    value="(cuWIm*ee*vevhat*yc)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_570 = Coupling(
    name="GC_570",
    value="(cth*cuWIm*ee*vevhat*yc)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_571 = Coupling(
    name="GC_571",
    value="-((cuWRe*ee*complex(0,1)*vevhat*yc)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_572 = Coupling(
    name="GC_572",
    value="-((cth*cuWRe*ee*complex(0,1)*vevhat*yc)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_573 = Coupling(
    name="GC_573",
    value="(cth*cuWRe*ee*complex(0,1)*vevhat*yc)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_574 = Coupling(
    name="GC_574",
    value="(cuBIm*sth*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_575 = Coupling(
    name="GC_575",
    value="-((cuBRe*complex(0,1)*sth*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_576 = Coupling(
    name="GC_576",
    value="-((cuWIm*sth*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_577 = Coupling(
    name="GC_577",
    value="(cuWRe*complex(0,1)*sth*vevhat*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_578 = Coupling(
    name="GC_578",
    value="-((cHbox*complex(0,1)*vevhat**2*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_579 = Coupling(
    name="GC_579",
    value="(cHDD*complex(0,1)*vevhat**2*yc)/(4.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_580 = Coupling(
    name="GC_580",
    value="(cHl3*complex(0,1)*vevhat**2*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_581 = Coupling(
    name="GC_581",
    value="-(cll1*complex(0,1)*vevhat**2*yc)/(2.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_582 = Coupling(
    name="GC_582",
    value="-((cuHIm*vevhat**2*yc)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_583 = Coupling(
    name="GC_583",
    value="(cuHRe*complex(0,1)*vevhat**2*yc)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_584 = Coupling(
    name="GC_584",
    value="-((cquqd11Im*yb*yc)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_585 = Coupling(
    name="GC_585",
    value="(cquqd11Im*yb*yc)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_586 = Coupling(
    name="GC_586",
    value="-((cquqd11Re*complex(0,1)*yb*yc)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_587 = Coupling(
    name="GC_587",
    value="-((cquqd1Im*yb*yc)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_588 = Coupling(
    name="GC_588",
    value="(cquqd1Im*yb*yc)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_589 = Coupling(
    name="GC_589",
    value="(cquqd1Re*complex(0,1)*yb*yc)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_590 = Coupling(
    name="GC_590",
    value="-((cquqd81Im*yb*yc)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_591 = Coupling(
    name="GC_591",
    value="(cquqd81Im*yb*yc)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_592 = Coupling(
    name="GC_592",
    value="-((cquqd81Re*complex(0,1)*yb*yc)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_593 = Coupling(
    name="GC_593",
    value="-((cquqd8Im*yb*yc)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_594 = Coupling(
    name="GC_594",
    value="(cquqd8Im*yb*yc)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_595 = Coupling(
    name="GC_595",
    value="(cquqd8Re*complex(0,1)*yb*yc)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_596 = Coupling(
    name="GC_596",
    value="-((cHudIm*CKM2x3*ee*yb*yc)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 5},
)

GC_597 = Coupling(
    name="GC_597",
    value="(cHudRe*CKM2x3*ee*complex(0,1)*yb*yc)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_598 = Coupling(
    name="GC_598",
    value="-((cHudIm*CKM2x3*ee*vevhat*yb*yc)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_599 = Coupling(
    name="GC_599",
    value="(cHudRe*CKM2x3*ee*complex(0,1)*vevhat*yb*yc)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_600 = Coupling(
    name="GC_600",
    value="-(cHudIm*CKM2x3*ee*vevhat**2*yb*yc)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_601 = Coupling(
    name="GC_601",
    value="(cHudRe*CKM2x3*ee*complex(0,1)*vevhat**2*yb*yc)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_602 = Coupling(
    name="GC_602", value="-((complex(0,1)*ydo)/cmath.sqrt(2))", order={"QED": 1}
)

GC_603 = Coupling(
    name="GC_603",
    value="-((cdGIm*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_604 = Coupling(
    name="GC_604",
    value="(cdGRe*complex(0,1)*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_605 = Coupling(
    name="GC_605",
    value="(-3*cdHIm*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_606 = Coupling(
    name="GC_606",
    value="(3*cdHRe*complex(0,1)*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_607 = Coupling(
    name="GC_607", value="-((cdWIm*ydo)/LambdaSMEFT**2)", order={"NP": 1, "QED": 3}
)

GC_608 = Coupling(
    name="GC_608", value="(cdWIm*ydo)/LambdaSMEFT**2", order={"NP": 1, "QED": 3}
)

GC_609 = Coupling(
    name="GC_609",
    value="(cdWRe*complex(0,1)*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_610 = Coupling(
    name="GC_610",
    value="-((cdBIm*cth*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_611 = Coupling(
    name="GC_611",
    value="(cdBRe*cth*complex(0,1)*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_612 = Coupling(
    name="GC_612",
    value="(cdWIm*cth*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_613 = Coupling(
    name="GC_613",
    value="-((cdWRe*cth*complex(0,1)*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_614 = Coupling(
    name="GC_614", value="(cdWIm*ee*ydo)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_615 = Coupling(
    name="GC_615",
    value="-((cdWRe*ee*complex(0,1)*ydo)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_616 = Coupling(
    name="GC_616",
    value="(cdWRe*ee*complex(0,1)*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_617 = Coupling(
    name="GC_617",
    value="(cdGIm*complex(0,1)*G*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_618 = Coupling(
    name="GC_618",
    value="(cdGRe*G*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_619 = Coupling(
    name="GC_619",
    value="-((complex(0,1)*propCorr*ydo)/cmath.sqrt(2))",
    order={"QED": 1},
)

GC_620 = Coupling(
    name="GC_620",
    value="-((cdWIm*ee*ydo)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_621 = Coupling(
    name="GC_621",
    value="(cdWRe*ee*complex(0,1)*ydo)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_622 = Coupling(
    name="GC_622",
    value="-((cdWIm*cth*ee*ydo)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_623 = Coupling(
    name="GC_623",
    value="-((cdWRe*cth*ee*complex(0,1)*ydo)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_624 = Coupling(
    name="GC_624",
    value="(cdWRe*cth*ee*complex(0,1)*ydo)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_625 = Coupling(
    name="GC_625",
    value="(cdBIm*sth*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_626 = Coupling(
    name="GC_626",
    value="-((cdBRe*complex(0,1)*sth*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_627 = Coupling(
    name="GC_627",
    value="(cdWIm*sth*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_628 = Coupling(
    name="GC_628",
    value="-((cdWRe*complex(0,1)*sth*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_629 = Coupling(
    name="GC_629",
    value="-((cdGIm*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_630 = Coupling(
    name="GC_630",
    value="(cdGRe*complex(0,1)*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_631 = Coupling(
    name="GC_631",
    value="(-3*cdHIm*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_632 = Coupling(
    name="GC_632",
    value="(3*cdHRe*complex(0,1)*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_633 = Coupling(
    name="GC_633",
    value="-((cdWIm*vevhat*ydo)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_634 = Coupling(
    name="GC_634", value="(cdWIm*vevhat*ydo)/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_635 = Coupling(
    name="GC_635",
    value="(cdWRe*complex(0,1)*vevhat*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_636 = Coupling(
    name="GC_636",
    value="-((cdBIm*cth*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_637 = Coupling(
    name="GC_637",
    value="(cdBRe*cth*complex(0,1)*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_638 = Coupling(
    name="GC_638",
    value="(cdWIm*cth*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_639 = Coupling(
    name="GC_639",
    value="-((cdWRe*cth*complex(0,1)*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_640 = Coupling(
    name="GC_640",
    value="(cdWIm*ee*vevhat*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_641 = Coupling(
    name="GC_641",
    value="-((cdWRe*ee*complex(0,1)*vevhat*ydo)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_642 = Coupling(
    name="GC_642",
    value="(cdWRe*ee*complex(0,1)*vevhat*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_643 = Coupling(
    name="GC_643",
    value="(cdGIm*complex(0,1)*G*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_644 = Coupling(
    name="GC_644",
    value="(cdGRe*G*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_645 = Coupling(
    name="GC_645",
    value="-((cdWIm*ee*vevhat*ydo)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_646 = Coupling(
    name="GC_646",
    value="(cdWRe*ee*complex(0,1)*vevhat*ydo)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_647 = Coupling(
    name="GC_647",
    value="-((cdWIm*cth*ee*vevhat*ydo)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_648 = Coupling(
    name="GC_648",
    value="-((cdWRe*cth*ee*complex(0,1)*vevhat*ydo)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_649 = Coupling(
    name="GC_649",
    value="(cdWRe*cth*ee*complex(0,1)*vevhat*ydo)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_650 = Coupling(
    name="GC_650",
    value="(cdBIm*sth*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_651 = Coupling(
    name="GC_651",
    value="-((cdBRe*complex(0,1)*sth*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_652 = Coupling(
    name="GC_652",
    value="(cdWIm*sth*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_653 = Coupling(
    name="GC_653",
    value="-((cdWRe*complex(0,1)*sth*vevhat*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_654 = Coupling(
    name="GC_654",
    value="-((cdHIm*vevhat**2*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_655 = Coupling(
    name="GC_655",
    value="(cdHRe*complex(0,1)*vevhat**2*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_656 = Coupling(
    name="GC_656",
    value="-((cHbox*complex(0,1)*vevhat**2*ydo)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_657 = Coupling(
    name="GC_657",
    value="(cHDD*complex(0,1)*vevhat**2*ydo)/(4.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_658 = Coupling(
    name="GC_658",
    value="(cHl3*complex(0,1)*vevhat**2*ydo)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_659 = Coupling(
    name="GC_659",
    value="-(cll1*complex(0,1)*vevhat**2*ydo)/(2.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_660 = Coupling(
    name="GC_660",
    value="-((cquqd11Im*yc*ydo)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_661 = Coupling(
    name="GC_661",
    value="(cquqd11Im*yc*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_662 = Coupling(
    name="GC_662",
    value="-((cquqd11Re*complex(0,1)*yc*ydo)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_663 = Coupling(
    name="GC_663",
    value="-((cquqd1Im*yc*ydo)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_664 = Coupling(
    name="GC_664",
    value="(cquqd1Im*yc*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_665 = Coupling(
    name="GC_665",
    value="(cquqd1Re*complex(0,1)*yc*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_666 = Coupling(
    name="GC_666",
    value="-((cquqd81Im*yc*ydo)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_667 = Coupling(
    name="GC_667",
    value="(cquqd81Im*yc*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_668 = Coupling(
    name="GC_668",
    value="-((cquqd81Re*complex(0,1)*yc*ydo)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_669 = Coupling(
    name="GC_669",
    value="-((cquqd8Im*yc*ydo)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_670 = Coupling(
    name="GC_670",
    value="(cquqd8Im*yc*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_671 = Coupling(
    name="GC_671",
    value="(cquqd8Re*complex(0,1)*yc*ydo)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_672 = Coupling(
    name="GC_672",
    value="-((cHudIm*CKM2x1*ee*yc*ydo)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 5},
)

GC_673 = Coupling(
    name="GC_673",
    value="(cHudRe*CKM2x1*ee*complex(0,1)*yc*ydo)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_674 = Coupling(
    name="GC_674",
    value="-((cHudIm*CKM2x1*ee*vevhat*yc*ydo)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_675 = Coupling(
    name="GC_675",
    value="(cHudRe*CKM2x1*ee*complex(0,1)*vevhat*yc*ydo)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_676 = Coupling(
    name="GC_676",
    value="-(cHudIm*CKM2x1*ee*vevhat**2*yc*ydo)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_677 = Coupling(
    name="GC_677",
    value="(cHudRe*CKM2x1*ee*complex(0,1)*vevhat**2*yc*ydo)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_678 = Coupling(
    name="GC_678", value="-((complex(0,1)*ye)/cmath.sqrt(2))", order={"QED": 1}
)

GC_679 = Coupling(
    name="GC_679",
    value="(-3*ceHIm*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_680 = Coupling(
    name="GC_680",
    value="(3*ceHRe*complex(0,1)*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_681 = Coupling(
    name="GC_681", value="-((ceWIm*ye)/LambdaSMEFT**2)", order={"NP": 1, "QED": 3}
)

GC_682 = Coupling(
    name="GC_682", value="(ceWIm*ye)/LambdaSMEFT**2", order={"NP": 1, "QED": 3}
)

GC_683 = Coupling(
    name="GC_683",
    value="(ceWRe*complex(0,1)*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_684 = Coupling(
    name="GC_684",
    value="-((ceBIm*cth*ye)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_685 = Coupling(
    name="GC_685",
    value="(ceBRe*cth*complex(0,1)*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_686 = Coupling(
    name="GC_686",
    value="(ceWIm*cth*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_687 = Coupling(
    name="GC_687",
    value="-((ceWRe*cth*complex(0,1)*ye)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_688 = Coupling(
    name="GC_688", value="(ceWIm*ee*ye)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_689 = Coupling(
    name="GC_689",
    value="-((ceWRe*ee*complex(0,1)*ye)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_690 = Coupling(
    name="GC_690",
    value="(ceWRe*ee*complex(0,1)*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_691 = Coupling(
    name="GC_691", value="-((complex(0,1)*propCorr*ye)/cmath.sqrt(2))", order={"QED": 1}
)

GC_692 = Coupling(
    name="GC_692",
    value="-((ceWIm*ee*ye)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_693 = Coupling(
    name="GC_693",
    value="(ceWRe*ee*complex(0,1)*ye)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_694 = Coupling(
    name="GC_694",
    value="-((ceWIm*cth*ee*ye)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_695 = Coupling(
    name="GC_695",
    value="-((ceWRe*cth*ee*complex(0,1)*ye)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_696 = Coupling(
    name="GC_696",
    value="(ceWRe*cth*ee*complex(0,1)*ye)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_697 = Coupling(
    name="GC_697",
    value="(ceBIm*sth*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_698 = Coupling(
    name="GC_698",
    value="-((ceBRe*complex(0,1)*sth*ye)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_699 = Coupling(
    name="GC_699",
    value="(ceWIm*sth*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_700 = Coupling(
    name="GC_700",
    value="-((ceWRe*complex(0,1)*sth*ye)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_701 = Coupling(
    name="GC_701",
    value="(-3*ceHIm*vevhat*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_702 = Coupling(
    name="GC_702",
    value="(3*ceHRe*complex(0,1)*vevhat*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_703 = Coupling(
    name="GC_703",
    value="-((ceWIm*vevhat*ye)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_704 = Coupling(
    name="GC_704", value="(ceWIm*vevhat*ye)/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_705 = Coupling(
    name="GC_705",
    value="(ceWRe*complex(0,1)*vevhat*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_706 = Coupling(
    name="GC_706",
    value="-((ceBIm*cth*vevhat*ye)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_707 = Coupling(
    name="GC_707",
    value="(ceBRe*cth*complex(0,1)*vevhat*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_708 = Coupling(
    name="GC_708",
    value="(ceWIm*cth*vevhat*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_709 = Coupling(
    name="GC_709",
    value="-((ceWRe*cth*complex(0,1)*vevhat*ye)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_710 = Coupling(
    name="GC_710",
    value="(ceWIm*ee*vevhat*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_711 = Coupling(
    name="GC_711",
    value="-((ceWRe*ee*complex(0,1)*vevhat*ye)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_712 = Coupling(
    name="GC_712",
    value="(ceWRe*ee*complex(0,1)*vevhat*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_713 = Coupling(
    name="GC_713",
    value="-((ceWIm*ee*vevhat*ye)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_714 = Coupling(
    name="GC_714",
    value="(ceWRe*ee*complex(0,1)*vevhat*ye)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_715 = Coupling(
    name="GC_715",
    value="-((ceWIm*cth*ee*vevhat*ye)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_716 = Coupling(
    name="GC_716",
    value="-((ceWRe*cth*ee*complex(0,1)*vevhat*ye)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_717 = Coupling(
    name="GC_717",
    value="(ceWRe*cth*ee*complex(0,1)*vevhat*ye)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_718 = Coupling(
    name="GC_718",
    value="(ceBIm*sth*vevhat*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_719 = Coupling(
    name="GC_719",
    value="-((ceBRe*complex(0,1)*sth*vevhat*ye)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_720 = Coupling(
    name="GC_720",
    value="(ceWIm*sth*vevhat*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_721 = Coupling(
    name="GC_721",
    value="-((ceWRe*complex(0,1)*sth*vevhat*ye)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_722 = Coupling(
    name="GC_722",
    value="-((ceHIm*vevhat**2*ye)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_723 = Coupling(
    name="GC_723",
    value="(ceHRe*complex(0,1)*vevhat**2*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_724 = Coupling(
    name="GC_724",
    value="-((cHbox*complex(0,1)*vevhat**2*ye)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_725 = Coupling(
    name="GC_725",
    value="(cHDD*complex(0,1)*vevhat**2*ye)/(4.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_726 = Coupling(
    name="GC_726",
    value="(cHl3*complex(0,1)*vevhat**2*ye)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_727 = Coupling(
    name="GC_727",
    value="-(cll1*complex(0,1)*vevhat**2*ye)/(2.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_728 = Coupling(
    name="GC_728", value="-((cledqIm*yb*ye)/LambdaSMEFT**2)", order={"NP": 1, "QED": 4}
)

GC_729 = Coupling(
    name="GC_729", value="(cledqIm*yb*ye)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_730 = Coupling(
    name="GC_730",
    value="(CKM1x3*cledqIm*yb*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_731 = Coupling(
    name="GC_731",
    value="(CKM2x3*cledqIm*yb*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_732 = Coupling(
    name="GC_732",
    value="(CKM3x3*cledqIm*yb*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_733 = Coupling(
    name="GC_733",
    value="(cledqRe*complex(0,1)*yb*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_734 = Coupling(
    name="GC_734",
    value="(CKM1x3*cledqRe*complex(0,1)*yb*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_735 = Coupling(
    name="GC_735",
    value="(CKM2x3*cledqRe*complex(0,1)*yb*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_736 = Coupling(
    name="GC_736",
    value="(CKM3x3*cledqRe*complex(0,1)*yb*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_737 = Coupling(
    name="GC_737",
    value="-((clequ1Im*yc*ye)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_738 = Coupling(
    name="GC_738",
    value="(clequ1Im*yc*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_739 = Coupling(
    name="GC_739",
    value="(CKM2x1*clequ1Im*yc*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_740 = Coupling(
    name="GC_740",
    value="(CKM2x2*clequ1Im*yc*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_741 = Coupling(
    name="GC_741",
    value="(CKM2x3*clequ1Im*yc*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_742 = Coupling(
    name="GC_742",
    value="-((clequ1Re*complex(0,1)*yc*ye)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_743 = Coupling(
    name="GC_743",
    value="(CKM2x1*clequ1Re*complex(0,1)*yc*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_744 = Coupling(
    name="GC_744",
    value="(CKM2x2*clequ1Re*complex(0,1)*yc*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_745 = Coupling(
    name="GC_745",
    value="(CKM2x3*clequ1Re*complex(0,1)*yc*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_746 = Coupling(
    name="GC_746",
    value="-(clequ3Im*yc*ye)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_747 = Coupling(
    name="GC_747",
    value="-(clequ3Im*yc*ye)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_748 = Coupling(
    name="GC_748",
    value="(clequ3Im*yc*ye)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_749 = Coupling(
    name="GC_749",
    value="(clequ3Im*yc*ye)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_750 = Coupling(
    name="GC_750",
    value="-(CKM2x1*clequ3Im*yc*ye)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_751 = Coupling(
    name="GC_751",
    value="(CKM2x1*clequ3Im*yc*ye)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_752 = Coupling(
    name="GC_752",
    value="-(CKM2x2*clequ3Im*yc*ye)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_753 = Coupling(
    name="GC_753",
    value="(CKM2x2*clequ3Im*yc*ye)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_754 = Coupling(
    name="GC_754",
    value="-(CKM2x3*clequ3Im*yc*ye)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_755 = Coupling(
    name="GC_755",
    value="(CKM2x3*clequ3Im*yc*ye)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_756 = Coupling(
    name="GC_756",
    value="-(clequ3Re*complex(0,1)*yc*ye)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_757 = Coupling(
    name="GC_757",
    value="(clequ3Re*complex(0,1)*yc*ye)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_758 = Coupling(
    name="GC_758",
    value="(CKM2x1*clequ3Re*complex(0,1)*yc*ye)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_759 = Coupling(
    name="GC_759",
    value="-(CKM2x1*clequ3Re*complex(0,1)*yc*ye)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_760 = Coupling(
    name="GC_760",
    value="(CKM2x2*clequ3Re*complex(0,1)*yc*ye)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_761 = Coupling(
    name="GC_761",
    value="-(CKM2x2*clequ3Re*complex(0,1)*yc*ye)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_762 = Coupling(
    name="GC_762",
    value="(CKM2x3*clequ3Re*complex(0,1)*yc*ye)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_763 = Coupling(
    name="GC_763",
    value="-(CKM2x3*clequ3Re*complex(0,1)*yc*ye)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_764 = Coupling(
    name="GC_764", value="-((cledqIm*ydo*ye)/LambdaSMEFT**2)", order={"NP": 1, "QED": 4}
)

GC_765 = Coupling(
    name="GC_765", value="(cledqIm*ydo*ye)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_766 = Coupling(
    name="GC_766",
    value="(CKM1x1*cledqIm*ydo*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_767 = Coupling(
    name="GC_767",
    value="(CKM2x1*cledqIm*ydo*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_768 = Coupling(
    name="GC_768",
    value="(CKM3x1*cledqIm*ydo*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_769 = Coupling(
    name="GC_769",
    value="(cledqRe*complex(0,1)*ydo*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_770 = Coupling(
    name="GC_770",
    value="(CKM1x1*cledqRe*complex(0,1)*ydo*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_771 = Coupling(
    name="GC_771",
    value="(CKM2x1*cledqRe*complex(0,1)*ydo*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_772 = Coupling(
    name="GC_772",
    value="(CKM3x1*cledqRe*complex(0,1)*ydo*ye)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_773 = Coupling(
    name="GC_773", value="-((complex(0,1)*ym)/cmath.sqrt(2))", order={"QED": 1}
)

GC_774 = Coupling(
    name="GC_774",
    value="(-3*ceHIm*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_775 = Coupling(
    name="GC_775",
    value="(3*ceHRe*complex(0,1)*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_776 = Coupling(
    name="GC_776", value="-((ceWIm*ym)/LambdaSMEFT**2)", order={"NP": 1, "QED": 3}
)

GC_777 = Coupling(
    name="GC_777", value="(ceWIm*ym)/LambdaSMEFT**2", order={"NP": 1, "QED": 3}
)

GC_778 = Coupling(
    name="GC_778",
    value="(ceWRe*complex(0,1)*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_779 = Coupling(
    name="GC_779",
    value="-((ceBIm*cth*ym)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_780 = Coupling(
    name="GC_780",
    value="(ceBRe*cth*complex(0,1)*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_781 = Coupling(
    name="GC_781",
    value="(ceWIm*cth*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_782 = Coupling(
    name="GC_782",
    value="-((ceWRe*cth*complex(0,1)*ym)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_783 = Coupling(
    name="GC_783", value="(ceWIm*ee*ym)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_784 = Coupling(
    name="GC_784",
    value="-((ceWRe*ee*complex(0,1)*ym)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_785 = Coupling(
    name="GC_785",
    value="(ceWRe*ee*complex(0,1)*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_786 = Coupling(
    name="GC_786", value="-((complex(0,1)*propCorr*ym)/cmath.sqrt(2))", order={"QED": 1}
)

GC_787 = Coupling(
    name="GC_787",
    value="-((ceWIm*ee*ym)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_788 = Coupling(
    name="GC_788",
    value="(ceWRe*ee*complex(0,1)*ym)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_789 = Coupling(
    name="GC_789",
    value="-((ceWIm*cth*ee*ym)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_790 = Coupling(
    name="GC_790",
    value="-((ceWRe*cth*ee*complex(0,1)*ym)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_791 = Coupling(
    name="GC_791",
    value="(ceWRe*cth*ee*complex(0,1)*ym)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_792 = Coupling(
    name="GC_792",
    value="(ceBIm*sth*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_793 = Coupling(
    name="GC_793",
    value="-((ceBRe*complex(0,1)*sth*ym)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_794 = Coupling(
    name="GC_794",
    value="(ceWIm*sth*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_795 = Coupling(
    name="GC_795",
    value="-((ceWRe*complex(0,1)*sth*ym)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_796 = Coupling(
    name="GC_796",
    value="(-3*ceHIm*vevhat*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_797 = Coupling(
    name="GC_797",
    value="(3*ceHRe*complex(0,1)*vevhat*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_798 = Coupling(
    name="GC_798",
    value="-((ceWIm*vevhat*ym)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_799 = Coupling(
    name="GC_799", value="(ceWIm*vevhat*ym)/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_800 = Coupling(
    name="GC_800",
    value="(ceWRe*complex(0,1)*vevhat*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_801 = Coupling(
    name="GC_801",
    value="-((ceBIm*cth*vevhat*ym)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_802 = Coupling(
    name="GC_802",
    value="(ceBRe*cth*complex(0,1)*vevhat*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_803 = Coupling(
    name="GC_803",
    value="(ceWIm*cth*vevhat*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_804 = Coupling(
    name="GC_804",
    value="-((ceWRe*cth*complex(0,1)*vevhat*ym)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_805 = Coupling(
    name="GC_805",
    value="(ceWIm*ee*vevhat*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_806 = Coupling(
    name="GC_806",
    value="-((ceWRe*ee*complex(0,1)*vevhat*ym)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_807 = Coupling(
    name="GC_807",
    value="(ceWRe*ee*complex(0,1)*vevhat*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_808 = Coupling(
    name="GC_808",
    value="-((ceWIm*ee*vevhat*ym)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_809 = Coupling(
    name="GC_809",
    value="(ceWRe*ee*complex(0,1)*vevhat*ym)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_810 = Coupling(
    name="GC_810",
    value="-((ceWIm*cth*ee*vevhat*ym)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_811 = Coupling(
    name="GC_811",
    value="-((ceWRe*cth*ee*complex(0,1)*vevhat*ym)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_812 = Coupling(
    name="GC_812",
    value="(ceWRe*cth*ee*complex(0,1)*vevhat*ym)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_813 = Coupling(
    name="GC_813",
    value="(ceBIm*sth*vevhat*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_814 = Coupling(
    name="GC_814",
    value="-((ceBRe*complex(0,1)*sth*vevhat*ym)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_815 = Coupling(
    name="GC_815",
    value="(ceWIm*sth*vevhat*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_816 = Coupling(
    name="GC_816",
    value="-((ceWRe*complex(0,1)*sth*vevhat*ym)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_817 = Coupling(
    name="GC_817",
    value="-((ceHIm*vevhat**2*ym)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_818 = Coupling(
    name="GC_818",
    value="(ceHRe*complex(0,1)*vevhat**2*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_819 = Coupling(
    name="GC_819",
    value="-((cHbox*complex(0,1)*vevhat**2*ym)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_820 = Coupling(
    name="GC_820",
    value="(cHDD*complex(0,1)*vevhat**2*ym)/(4.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_821 = Coupling(
    name="GC_821",
    value="(cHl3*complex(0,1)*vevhat**2*ym)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_822 = Coupling(
    name="GC_822",
    value="-(cll1*complex(0,1)*vevhat**2*ym)/(2.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_823 = Coupling(
    name="GC_823", value="-((cledqIm*yb*ym)/LambdaSMEFT**2)", order={"NP": 1, "QED": 4}
)

GC_824 = Coupling(
    name="GC_824", value="(cledqIm*yb*ym)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_825 = Coupling(
    name="GC_825",
    value="(CKM1x3*cledqIm*yb*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_826 = Coupling(
    name="GC_826",
    value="(CKM2x3*cledqIm*yb*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_827 = Coupling(
    name="GC_827",
    value="(CKM3x3*cledqIm*yb*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_828 = Coupling(
    name="GC_828",
    value="(cledqRe*complex(0,1)*yb*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_829 = Coupling(
    name="GC_829",
    value="(CKM1x3*cledqRe*complex(0,1)*yb*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_830 = Coupling(
    name="GC_830",
    value="(CKM2x3*cledqRe*complex(0,1)*yb*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_831 = Coupling(
    name="GC_831",
    value="(CKM3x3*cledqRe*complex(0,1)*yb*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_832 = Coupling(
    name="GC_832",
    value="-((clequ1Im*yc*ym)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_833 = Coupling(
    name="GC_833",
    value="(clequ1Im*yc*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_834 = Coupling(
    name="GC_834",
    value="(CKM2x1*clequ1Im*yc*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_835 = Coupling(
    name="GC_835",
    value="(CKM2x2*clequ1Im*yc*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_836 = Coupling(
    name="GC_836",
    value="(CKM2x3*clequ1Im*yc*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_837 = Coupling(
    name="GC_837",
    value="-((clequ1Re*complex(0,1)*yc*ym)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_838 = Coupling(
    name="GC_838",
    value="(CKM2x1*clequ1Re*complex(0,1)*yc*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_839 = Coupling(
    name="GC_839",
    value="(CKM2x2*clequ1Re*complex(0,1)*yc*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_840 = Coupling(
    name="GC_840",
    value="(CKM2x3*clequ1Re*complex(0,1)*yc*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_841 = Coupling(
    name="GC_841",
    value="-(clequ3Im*yc*ym)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_842 = Coupling(
    name="GC_842",
    value="-(clequ3Im*yc*ym)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_843 = Coupling(
    name="GC_843",
    value="(clequ3Im*yc*ym)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_844 = Coupling(
    name="GC_844",
    value="(clequ3Im*yc*ym)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_845 = Coupling(
    name="GC_845",
    value="-(CKM2x1*clequ3Im*yc*ym)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_846 = Coupling(
    name="GC_846",
    value="(CKM2x1*clequ3Im*yc*ym)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_847 = Coupling(
    name="GC_847",
    value="-(CKM2x2*clequ3Im*yc*ym)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_848 = Coupling(
    name="GC_848",
    value="(CKM2x2*clequ3Im*yc*ym)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_849 = Coupling(
    name="GC_849",
    value="-(CKM2x3*clequ3Im*yc*ym)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_850 = Coupling(
    name="GC_850",
    value="(CKM2x3*clequ3Im*yc*ym)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_851 = Coupling(
    name="GC_851",
    value="-(clequ3Re*complex(0,1)*yc*ym)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_852 = Coupling(
    name="GC_852",
    value="(clequ3Re*complex(0,1)*yc*ym)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_853 = Coupling(
    name="GC_853",
    value="(CKM2x1*clequ3Re*complex(0,1)*yc*ym)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_854 = Coupling(
    name="GC_854",
    value="-(CKM2x1*clequ3Re*complex(0,1)*yc*ym)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_855 = Coupling(
    name="GC_855",
    value="(CKM2x2*clequ3Re*complex(0,1)*yc*ym)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_856 = Coupling(
    name="GC_856",
    value="-(CKM2x2*clequ3Re*complex(0,1)*yc*ym)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_857 = Coupling(
    name="GC_857",
    value="(CKM2x3*clequ3Re*complex(0,1)*yc*ym)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_858 = Coupling(
    name="GC_858",
    value="-(CKM2x3*clequ3Re*complex(0,1)*yc*ym)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_859 = Coupling(
    name="GC_859", value="-((cledqIm*ydo*ym)/LambdaSMEFT**2)", order={"NP": 1, "QED": 4}
)

GC_860 = Coupling(
    name="GC_860", value="(cledqIm*ydo*ym)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_861 = Coupling(
    name="GC_861",
    value="(CKM1x1*cledqIm*ydo*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_862 = Coupling(
    name="GC_862",
    value="(CKM2x1*cledqIm*ydo*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_863 = Coupling(
    name="GC_863",
    value="(CKM3x1*cledqIm*ydo*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_864 = Coupling(
    name="GC_864",
    value="(cledqRe*complex(0,1)*ydo*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_865 = Coupling(
    name="GC_865",
    value="(CKM1x1*cledqRe*complex(0,1)*ydo*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_866 = Coupling(
    name="GC_866",
    value="(CKM2x1*cledqRe*complex(0,1)*ydo*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_867 = Coupling(
    name="GC_867",
    value="(CKM3x1*cledqRe*complex(0,1)*ydo*ym)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_868 = Coupling(
    name="GC_868", value="-((complex(0,1)*ys)/cmath.sqrt(2))", order={"QED": 1}
)

GC_869 = Coupling(
    name="GC_869",
    value="-((cdGIm*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_870 = Coupling(
    name="GC_870",
    value="(cdGRe*complex(0,1)*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_871 = Coupling(
    name="GC_871",
    value="(-3*cdHIm*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_872 = Coupling(
    name="GC_872",
    value="(3*cdHRe*complex(0,1)*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_873 = Coupling(
    name="GC_873", value="-((cdWIm*ys)/LambdaSMEFT**2)", order={"NP": 1, "QED": 3}
)

GC_874 = Coupling(
    name="GC_874", value="(cdWIm*ys)/LambdaSMEFT**2", order={"NP": 1, "QED": 3}
)

GC_875 = Coupling(
    name="GC_875",
    value="(cdWRe*complex(0,1)*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_876 = Coupling(
    name="GC_876",
    value="-((cdBIm*cth*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_877 = Coupling(
    name="GC_877",
    value="(cdBRe*cth*complex(0,1)*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_878 = Coupling(
    name="GC_878",
    value="(cdWIm*cth*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_879 = Coupling(
    name="GC_879",
    value="-((cdWRe*cth*complex(0,1)*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_880 = Coupling(
    name="GC_880", value="(cdWIm*ee*ys)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_881 = Coupling(
    name="GC_881",
    value="-((cdWRe*ee*complex(0,1)*ys)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_882 = Coupling(
    name="GC_882",
    value="(cdWRe*ee*complex(0,1)*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_883 = Coupling(
    name="GC_883",
    value="(cdGIm*complex(0,1)*G*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_884 = Coupling(
    name="GC_884",
    value="(cdGRe*G*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_885 = Coupling(
    name="GC_885", value="-((complex(0,1)*propCorr*ys)/cmath.sqrt(2))", order={"QED": 1}
)

GC_886 = Coupling(
    name="GC_886",
    value="-((cdWIm*ee*ys)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_887 = Coupling(
    name="GC_887",
    value="(cdWRe*ee*complex(0,1)*ys)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_888 = Coupling(
    name="GC_888",
    value="-((cdWIm*cth*ee*ys)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_889 = Coupling(
    name="GC_889",
    value="-((cdWRe*cth*ee*complex(0,1)*ys)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_890 = Coupling(
    name="GC_890",
    value="(cdWRe*cth*ee*complex(0,1)*ys)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_891 = Coupling(
    name="GC_891",
    value="(cdBIm*sth*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_892 = Coupling(
    name="GC_892",
    value="-((cdBRe*complex(0,1)*sth*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_893 = Coupling(
    name="GC_893",
    value="(cdWIm*sth*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_894 = Coupling(
    name="GC_894",
    value="-((cdWRe*complex(0,1)*sth*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_895 = Coupling(
    name="GC_895",
    value="-((cdGIm*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_896 = Coupling(
    name="GC_896",
    value="(cdGRe*complex(0,1)*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_897 = Coupling(
    name="GC_897",
    value="(-3*cdHIm*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_898 = Coupling(
    name="GC_898",
    value="(3*cdHRe*complex(0,1)*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_899 = Coupling(
    name="GC_899",
    value="-((cdWIm*vevhat*ys)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_900 = Coupling(
    name="GC_900", value="(cdWIm*vevhat*ys)/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_901 = Coupling(
    name="GC_901",
    value="(cdWRe*complex(0,1)*vevhat*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_902 = Coupling(
    name="GC_902",
    value="-((cdBIm*cth*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_903 = Coupling(
    name="GC_903",
    value="(cdBRe*cth*complex(0,1)*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_904 = Coupling(
    name="GC_904",
    value="(cdWIm*cth*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_905 = Coupling(
    name="GC_905",
    value="-((cdWRe*cth*complex(0,1)*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_906 = Coupling(
    name="GC_906",
    value="(cdWIm*ee*vevhat*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_907 = Coupling(
    name="GC_907",
    value="-((cdWRe*ee*complex(0,1)*vevhat*ys)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_908 = Coupling(
    name="GC_908",
    value="(cdWRe*ee*complex(0,1)*vevhat*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_909 = Coupling(
    name="GC_909",
    value="(cdGIm*complex(0,1)*G*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_910 = Coupling(
    name="GC_910",
    value="(cdGRe*G*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_911 = Coupling(
    name="GC_911",
    value="-((cdWIm*ee*vevhat*ys)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_912 = Coupling(
    name="GC_912",
    value="(cdWRe*ee*complex(0,1)*vevhat*ys)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_913 = Coupling(
    name="GC_913",
    value="-((cdWIm*cth*ee*vevhat*ys)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_914 = Coupling(
    name="GC_914",
    value="-((cdWRe*cth*ee*complex(0,1)*vevhat*ys)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_915 = Coupling(
    name="GC_915",
    value="(cdWRe*cth*ee*complex(0,1)*vevhat*ys)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_916 = Coupling(
    name="GC_916",
    value="(cdBIm*sth*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_917 = Coupling(
    name="GC_917",
    value="-((cdBRe*complex(0,1)*sth*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_918 = Coupling(
    name="GC_918",
    value="(cdWIm*sth*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_919 = Coupling(
    name="GC_919",
    value="-((cdWRe*complex(0,1)*sth*vevhat*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_920 = Coupling(
    name="GC_920",
    value="-((cdHIm*vevhat**2*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_921 = Coupling(
    name="GC_921",
    value="(cdHRe*complex(0,1)*vevhat**2*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_922 = Coupling(
    name="GC_922",
    value="-((cHbox*complex(0,1)*vevhat**2*ys)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_923 = Coupling(
    name="GC_923",
    value="(cHDD*complex(0,1)*vevhat**2*ys)/(4.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_924 = Coupling(
    name="GC_924",
    value="(cHl3*complex(0,1)*vevhat**2*ys)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_925 = Coupling(
    name="GC_925",
    value="-(cll1*complex(0,1)*vevhat**2*ys)/(2.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_926 = Coupling(
    name="GC_926",
    value="-((cquqd11Im*yc*ys)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_927 = Coupling(
    name="GC_927",
    value="(cquqd11Im*yc*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_928 = Coupling(
    name="GC_928",
    value="-((cquqd11Re*complex(0,1)*yc*ys)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_929 = Coupling(
    name="GC_929",
    value="-((cquqd1Im*yc*ys)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_930 = Coupling(
    name="GC_930",
    value="(cquqd1Im*yc*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_931 = Coupling(
    name="GC_931",
    value="(cquqd1Re*complex(0,1)*yc*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_932 = Coupling(
    name="GC_932",
    value="-((cquqd81Im*yc*ys)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_933 = Coupling(
    name="GC_933",
    value="(cquqd81Im*yc*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_934 = Coupling(
    name="GC_934",
    value="-((cquqd81Re*complex(0,1)*yc*ys)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_935 = Coupling(
    name="GC_935",
    value="-((cquqd8Im*yc*ys)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_936 = Coupling(
    name="GC_936",
    value="(cquqd8Im*yc*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_937 = Coupling(
    name="GC_937",
    value="(cquqd8Re*complex(0,1)*yc*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_938 = Coupling(
    name="GC_938",
    value="-((cHudIm*CKM2x2*ee*yc*ys)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 5},
)

GC_939 = Coupling(
    name="GC_939",
    value="(cHudRe*CKM2x2*ee*complex(0,1)*yc*ys)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_940 = Coupling(
    name="GC_940",
    value="-((cHudIm*CKM2x2*ee*vevhat*yc*ys)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_941 = Coupling(
    name="GC_941",
    value="(cHudRe*CKM2x2*ee*complex(0,1)*vevhat*yc*ys)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_942 = Coupling(
    name="GC_942",
    value="-(cHudIm*CKM2x2*ee*vevhat**2*yc*ys)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_943 = Coupling(
    name="GC_943",
    value="(cHudRe*CKM2x2*ee*complex(0,1)*vevhat**2*yc*ys)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_944 = Coupling(
    name="GC_944", value="-((cledqIm*ye*ys)/LambdaSMEFT**2)", order={"NP": 1, "QED": 4}
)

GC_945 = Coupling(
    name="GC_945", value="(cledqIm*ye*ys)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_946 = Coupling(
    name="GC_946",
    value="(CKM1x2*cledqIm*ye*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_947 = Coupling(
    name="GC_947",
    value="(CKM2x2*cledqIm*ye*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_948 = Coupling(
    name="GC_948",
    value="(CKM3x2*cledqIm*ye*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_949 = Coupling(
    name="GC_949",
    value="(cledqRe*complex(0,1)*ye*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_950 = Coupling(
    name="GC_950",
    value="(CKM1x2*cledqRe*complex(0,1)*ye*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_951 = Coupling(
    name="GC_951",
    value="(CKM2x2*cledqRe*complex(0,1)*ye*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_952 = Coupling(
    name="GC_952",
    value="(CKM3x2*cledqRe*complex(0,1)*ye*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_953 = Coupling(
    name="GC_953", value="-((cledqIm*ym*ys)/LambdaSMEFT**2)", order={"NP": 1, "QED": 4}
)

GC_954 = Coupling(
    name="GC_954", value="(cledqIm*ym*ys)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_955 = Coupling(
    name="GC_955",
    value="(CKM1x2*cledqIm*ym*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_956 = Coupling(
    name="GC_956",
    value="(CKM2x2*cledqIm*ym*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_957 = Coupling(
    name="GC_957",
    value="(CKM3x2*cledqIm*ym*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_958 = Coupling(
    name="GC_958",
    value="(cledqRe*complex(0,1)*ym*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_959 = Coupling(
    name="GC_959",
    value="(CKM1x2*cledqRe*complex(0,1)*ym*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_960 = Coupling(
    name="GC_960",
    value="(CKM2x2*cledqRe*complex(0,1)*ym*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_961 = Coupling(
    name="GC_961",
    value="(CKM3x2*cledqRe*complex(0,1)*ym*ys)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_962 = Coupling(
    name="GC_962", value="-((complex(0,1)*yt)/cmath.sqrt(2))", order={"QED": 1}
)

GC_963 = Coupling(
    name="GC_963",
    value="-((cth*cuBIm*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_964 = Coupling(
    name="GC_964",
    value="(cth*cuBRe*complex(0,1)*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_965 = Coupling(
    name="GC_965",
    value="-((cuGIm*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_966 = Coupling(
    name="GC_966",
    value="(cuGRe*complex(0,1)*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_967 = Coupling(
    name="GC_967",
    value="(-3*cuHIm*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_968 = Coupling(
    name="GC_968",
    value="(3*cuHRe*complex(0,1)*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_969 = Coupling(
    name="GC_969", value="-((cuWIm*yt)/LambdaSMEFT**2)", order={"NP": 1, "QED": 3}
)

GC_970 = Coupling(
    name="GC_970", value="(cuWIm*yt)/LambdaSMEFT**2", order={"NP": 1, "QED": 3}
)

GC_971 = Coupling(
    name="GC_971",
    value="-((cth*cuWIm*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_972 = Coupling(
    name="GC_972",
    value="(cuWRe*complex(0,1)*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_973 = Coupling(
    name="GC_973",
    value="(cth*cuWRe*complex(0,1)*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_974 = Coupling(
    name="GC_974", value="-((cuWIm*ee*yt)/LambdaSMEFT**2)", order={"NP": 1, "QED": 4}
)

GC_975 = Coupling(
    name="GC_975",
    value="-((cuWRe*ee*complex(0,1)*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_976 = Coupling(
    name="GC_976",
    value="(cuWRe*ee*complex(0,1)*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_977 = Coupling(
    name="GC_977",
    value="(cuGIm*complex(0,1)*G*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_978 = Coupling(
    name="GC_978",
    value="(cuGRe*G*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_979 = Coupling(
    name="GC_979", value="-((complex(0,1)*propCorr*yt)/cmath.sqrt(2))", order={"QED": 1}
)

GC_980 = Coupling(
    name="GC_980",
    value="-((complex(0,1)*propCorr**2*yt)/cmath.sqrt(2))",
    order={"QED": 1},
)

GC_981 = Coupling(
    name="GC_981",
    value="-((complex(0,1)*propCorr**3*yt)/cmath.sqrt(2))",
    order={"QED": 1},
)

GC_982 = Coupling(
    name="GC_982",
    value="(cuWIm*ee*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_983 = Coupling(
    name="GC_983",
    value="(cth*cuWIm*ee*yt)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_984 = Coupling(
    name="GC_984",
    value="-((cuWRe*ee*complex(0,1)*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_985 = Coupling(
    name="GC_985",
    value="-((cth*cuWRe*ee*complex(0,1)*yt)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_986 = Coupling(
    name="GC_986",
    value="(cth*cuWRe*ee*complex(0,1)*yt)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_987 = Coupling(
    name="GC_987",
    value="(cuBIm*sth*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_988 = Coupling(
    name="GC_988",
    value="-((cuBRe*complex(0,1)*sth*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_989 = Coupling(
    name="GC_989",
    value="-((cuWIm*sth*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_990 = Coupling(
    name="GC_990",
    value="(cuWRe*complex(0,1)*sth*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_991 = Coupling(
    name="GC_991",
    value="-((cth*cuBIm*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_992 = Coupling(
    name="GC_992",
    value="(cth*cuBRe*complex(0,1)*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_993 = Coupling(
    name="GC_993",
    value="-((cuGIm*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_994 = Coupling(
    name="GC_994",
    value="(cuGRe*complex(0,1)*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_995 = Coupling(
    name="GC_995",
    value="(-3*cuHIm*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_996 = Coupling(
    name="GC_996",
    value="(3*cuHRe*complex(0,1)*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_997 = Coupling(
    name="GC_997",
    value="-((cuWIm*vevhat*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_998 = Coupling(
    name="GC_998", value="(cuWIm*vevhat*yt)/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_999 = Coupling(
    name="GC_999",
    value="-((cth*cuWIm*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1000 = Coupling(
    name="GC_1000",
    value="(cuWRe*complex(0,1)*vevhat*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1001 = Coupling(
    name="GC_1001",
    value="(cth*cuWRe*complex(0,1)*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1002 = Coupling(
    name="GC_1002",
    value="-((cuWIm*ee*vevhat*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_1003 = Coupling(
    name="GC_1003",
    value="-((cuWRe*ee*complex(0,1)*vevhat*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_1004 = Coupling(
    name="GC_1004",
    value="(cuWRe*ee*complex(0,1)*vevhat*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_1005 = Coupling(
    name="GC_1005",
    value="(cuGIm*complex(0,1)*G*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_1006 = Coupling(
    name="GC_1006",
    value="(cuGRe*G*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_1007 = Coupling(
    name="GC_1007",
    value="(cuWIm*ee*vevhat*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1008 = Coupling(
    name="GC_1008",
    value="(cth*cuWIm*ee*vevhat*yt)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_1009 = Coupling(
    name="GC_1009",
    value="-((cuWRe*ee*complex(0,1)*vevhat*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1010 = Coupling(
    name="GC_1010",
    value="-((cth*cuWRe*ee*complex(0,1)*vevhat*yt)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_1011 = Coupling(
    name="GC_1011",
    value="(cth*cuWRe*ee*complex(0,1)*vevhat*yt)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_1012 = Coupling(
    name="GC_1012",
    value="(cuBIm*sth*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1013 = Coupling(
    name="GC_1013",
    value="-((cuBRe*complex(0,1)*sth*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1014 = Coupling(
    name="GC_1014",
    value="-((cuWIm*sth*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1015 = Coupling(
    name="GC_1015",
    value="(cuWRe*complex(0,1)*sth*vevhat*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1016 = Coupling(
    name="GC_1016",
    value="-((cHbox*complex(0,1)*vevhat**2*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1017 = Coupling(
    name="GC_1017",
    value="(cHDD*complex(0,1)*vevhat**2*yt)/(4.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1018 = Coupling(
    name="GC_1018",
    value="(cHl3*complex(0,1)*vevhat**2*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1019 = Coupling(
    name="GC_1019",
    value="-(cll1*complex(0,1)*vevhat**2*yt)/(2.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1020 = Coupling(
    name="GC_1020",
    value="-((cuHIm*vevhat**2*yt)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1021 = Coupling(
    name="GC_1021",
    value="(cuHRe*complex(0,1)*vevhat**2*yt)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1022 = Coupling(
    name="GC_1022",
    value="-((cquqd11Im*yb*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1023 = Coupling(
    name="GC_1023",
    value="(cquqd11Im*yb*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1024 = Coupling(
    name="GC_1024",
    value="-((cquqd11Re*complex(0,1)*yb*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1025 = Coupling(
    name="GC_1025",
    value="-((cquqd1Im*yb*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1026 = Coupling(
    name="GC_1026",
    value="(cquqd1Im*yb*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1027 = Coupling(
    name="GC_1027",
    value="(cquqd1Re*complex(0,1)*yb*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1028 = Coupling(
    name="GC_1028",
    value="-((cquqd81Im*yb*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1029 = Coupling(
    name="GC_1029",
    value="(cquqd81Im*yb*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1030 = Coupling(
    name="GC_1030",
    value="-((cquqd81Re*complex(0,1)*yb*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1031 = Coupling(
    name="GC_1031",
    value="-((cquqd8Im*yb*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1032 = Coupling(
    name="GC_1032",
    value="(cquqd8Im*yb*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1033 = Coupling(
    name="GC_1033",
    value="(cquqd8Re*complex(0,1)*yb*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1034 = Coupling(
    name="GC_1034",
    value="-((cHudIm*CKM3x3*ee*yb*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 5},
)

GC_1035 = Coupling(
    name="GC_1035",
    value="(cHudRe*CKM3x3*ee*complex(0,1)*yb*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_1036 = Coupling(
    name="GC_1036",
    value="-((cHudIm*CKM3x3*ee*vevhat*yb*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_1037 = Coupling(
    name="GC_1037",
    value="(cHudRe*CKM3x3*ee*complex(0,1)*vevhat*yb*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1038 = Coupling(
    name="GC_1038",
    value="-(cHudIm*CKM3x3*ee*vevhat**2*yb*yt)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1039 = Coupling(
    name="GC_1039",
    value="(cHudRe*CKM3x3*ee*complex(0,1)*vevhat**2*yb*yt)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1040 = Coupling(
    name="GC_1040",
    value="-((cquqd11Im*ydo*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1041 = Coupling(
    name="GC_1041",
    value="(cquqd11Im*ydo*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1042 = Coupling(
    name="GC_1042",
    value="-((cquqd11Re*complex(0,1)*ydo*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1043 = Coupling(
    name="GC_1043",
    value="-((cquqd1Im*ydo*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1044 = Coupling(
    name="GC_1044",
    value="(cquqd1Im*ydo*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1045 = Coupling(
    name="GC_1045",
    value="(cquqd1Re*complex(0,1)*ydo*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1046 = Coupling(
    name="GC_1046",
    value="-((cquqd81Im*ydo*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1047 = Coupling(
    name="GC_1047",
    value="(cquqd81Im*ydo*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1048 = Coupling(
    name="GC_1048",
    value="-((cquqd81Re*complex(0,1)*ydo*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1049 = Coupling(
    name="GC_1049",
    value="-((cquqd8Im*ydo*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1050 = Coupling(
    name="GC_1050",
    value="(cquqd8Im*ydo*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1051 = Coupling(
    name="GC_1051",
    value="(cquqd8Re*complex(0,1)*ydo*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1052 = Coupling(
    name="GC_1052",
    value="-((cHudIm*CKM3x1*ee*ydo*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 5},
)

GC_1053 = Coupling(
    name="GC_1053",
    value="(cHudRe*CKM3x1*ee*complex(0,1)*ydo*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_1054 = Coupling(
    name="GC_1054",
    value="-((cHudIm*CKM3x1*ee*vevhat*ydo*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_1055 = Coupling(
    name="GC_1055",
    value="(cHudRe*CKM3x1*ee*complex(0,1)*vevhat*ydo*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1056 = Coupling(
    name="GC_1056",
    value="-(cHudIm*CKM3x1*ee*vevhat**2*ydo*yt)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1057 = Coupling(
    name="GC_1057",
    value="(cHudRe*CKM3x1*ee*complex(0,1)*vevhat**2*ydo*yt)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1058 = Coupling(
    name="GC_1058",
    value="-((clequ1Im*ye*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1059 = Coupling(
    name="GC_1059",
    value="(clequ1Im*ye*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1060 = Coupling(
    name="GC_1060",
    value="(CKM3x1*clequ1Im*ye*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1061 = Coupling(
    name="GC_1061",
    value="(CKM3x2*clequ1Im*ye*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1062 = Coupling(
    name="GC_1062",
    value="(CKM3x3*clequ1Im*ye*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1063 = Coupling(
    name="GC_1063",
    value="-((clequ1Re*complex(0,1)*ye*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1064 = Coupling(
    name="GC_1064",
    value="(CKM3x1*clequ1Re*complex(0,1)*ye*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1065 = Coupling(
    name="GC_1065",
    value="(CKM3x2*clequ1Re*complex(0,1)*ye*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1066 = Coupling(
    name="GC_1066",
    value="(CKM3x3*clequ1Re*complex(0,1)*ye*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1067 = Coupling(
    name="GC_1067",
    value="-(clequ3Im*ye*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1068 = Coupling(
    name="GC_1068",
    value="-(clequ3Im*ye*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1069 = Coupling(
    name="GC_1069",
    value="(clequ3Im*ye*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1070 = Coupling(
    name="GC_1070",
    value="(clequ3Im*ye*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1071 = Coupling(
    name="GC_1071",
    value="-(CKM3x1*clequ3Im*ye*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1072 = Coupling(
    name="GC_1072",
    value="(CKM3x1*clequ3Im*ye*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1073 = Coupling(
    name="GC_1073",
    value="-(CKM3x2*clequ3Im*ye*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1074 = Coupling(
    name="GC_1074",
    value="(CKM3x2*clequ3Im*ye*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1075 = Coupling(
    name="GC_1075",
    value="-(CKM3x3*clequ3Im*ye*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1076 = Coupling(
    name="GC_1076",
    value="(CKM3x3*clequ3Im*ye*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1077 = Coupling(
    name="GC_1077",
    value="-(clequ3Re*complex(0,1)*ye*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1078 = Coupling(
    name="GC_1078",
    value="(clequ3Re*complex(0,1)*ye*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1079 = Coupling(
    name="GC_1079",
    value="(CKM3x1*clequ3Re*complex(0,1)*ye*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1080 = Coupling(
    name="GC_1080",
    value="-(CKM3x1*clequ3Re*complex(0,1)*ye*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1081 = Coupling(
    name="GC_1081",
    value="(CKM3x2*clequ3Re*complex(0,1)*ye*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1082 = Coupling(
    name="GC_1082",
    value="-(CKM3x2*clequ3Re*complex(0,1)*ye*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1083 = Coupling(
    name="GC_1083",
    value="(CKM3x3*clequ3Re*complex(0,1)*ye*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1084 = Coupling(
    name="GC_1084",
    value="-(CKM3x3*clequ3Re*complex(0,1)*ye*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1085 = Coupling(
    name="GC_1085",
    value="-((clequ1Im*ym*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1086 = Coupling(
    name="GC_1086",
    value="(clequ1Im*ym*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1087 = Coupling(
    name="GC_1087",
    value="(CKM3x1*clequ1Im*ym*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1088 = Coupling(
    name="GC_1088",
    value="(CKM3x2*clequ1Im*ym*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1089 = Coupling(
    name="GC_1089",
    value="(CKM3x3*clequ1Im*ym*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1090 = Coupling(
    name="GC_1090",
    value="-((clequ1Re*complex(0,1)*ym*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1091 = Coupling(
    name="GC_1091",
    value="(CKM3x1*clequ1Re*complex(0,1)*ym*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1092 = Coupling(
    name="GC_1092",
    value="(CKM3x2*clequ1Re*complex(0,1)*ym*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1093 = Coupling(
    name="GC_1093",
    value="(CKM3x3*clequ1Re*complex(0,1)*ym*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1094 = Coupling(
    name="GC_1094",
    value="-(clequ3Im*ym*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1095 = Coupling(
    name="GC_1095",
    value="-(clequ3Im*ym*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1096 = Coupling(
    name="GC_1096",
    value="(clequ3Im*ym*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1097 = Coupling(
    name="GC_1097",
    value="(clequ3Im*ym*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1098 = Coupling(
    name="GC_1098",
    value="-(CKM3x1*clequ3Im*ym*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1099 = Coupling(
    name="GC_1099",
    value="(CKM3x1*clequ3Im*ym*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1100 = Coupling(
    name="GC_1100",
    value="-(CKM3x2*clequ3Im*ym*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1101 = Coupling(
    name="GC_1101",
    value="(CKM3x2*clequ3Im*ym*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1102 = Coupling(
    name="GC_1102",
    value="-(CKM3x3*clequ3Im*ym*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1103 = Coupling(
    name="GC_1103",
    value="(CKM3x3*clequ3Im*ym*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1104 = Coupling(
    name="GC_1104",
    value="-(clequ3Re*complex(0,1)*ym*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1105 = Coupling(
    name="GC_1105",
    value="(clequ3Re*complex(0,1)*ym*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1106 = Coupling(
    name="GC_1106",
    value="(CKM3x1*clequ3Re*complex(0,1)*ym*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1107 = Coupling(
    name="GC_1107",
    value="-(CKM3x1*clequ3Re*complex(0,1)*ym*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1108 = Coupling(
    name="GC_1108",
    value="(CKM3x2*clequ3Re*complex(0,1)*ym*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1109 = Coupling(
    name="GC_1109",
    value="-(CKM3x2*clequ3Re*complex(0,1)*ym*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1110 = Coupling(
    name="GC_1110",
    value="(CKM3x3*clequ3Re*complex(0,1)*ym*yt)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1111 = Coupling(
    name="GC_1111",
    value="-(CKM3x3*clequ3Re*complex(0,1)*ym*yt)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1112 = Coupling(
    name="GC_1112",
    value="-((cquqd11Im*ys*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1113 = Coupling(
    name="GC_1113",
    value="(cquqd11Im*ys*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1114 = Coupling(
    name="GC_1114",
    value="-((cquqd11Re*complex(0,1)*ys*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1115 = Coupling(
    name="GC_1115",
    value="-((cquqd1Im*ys*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1116 = Coupling(
    name="GC_1116",
    value="(cquqd1Im*ys*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1117 = Coupling(
    name="GC_1117",
    value="(cquqd1Re*complex(0,1)*ys*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1118 = Coupling(
    name="GC_1118",
    value="-((cquqd81Im*ys*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1119 = Coupling(
    name="GC_1119",
    value="(cquqd81Im*ys*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1120 = Coupling(
    name="GC_1120",
    value="-((cquqd81Re*complex(0,1)*ys*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1121 = Coupling(
    name="GC_1121",
    value="-((cquqd8Im*ys*yt)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1122 = Coupling(
    name="GC_1122",
    value="(cquqd8Im*ys*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1123 = Coupling(
    name="GC_1123",
    value="(cquqd8Re*complex(0,1)*ys*yt)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1124 = Coupling(
    name="GC_1124",
    value="-((cHudIm*CKM3x2*ee*ys*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 5},
)

GC_1125 = Coupling(
    name="GC_1125",
    value="(cHudRe*CKM3x2*ee*complex(0,1)*ys*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_1126 = Coupling(
    name="GC_1126",
    value="-((cHudIm*CKM3x2*ee*vevhat*ys*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_1127 = Coupling(
    name="GC_1127",
    value="(cHudRe*CKM3x2*ee*complex(0,1)*vevhat*ys*yt)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1128 = Coupling(
    name="GC_1128",
    value="-(cHudIm*CKM3x2*ee*vevhat**2*ys*yt)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1129 = Coupling(
    name="GC_1129",
    value="(cHudRe*CKM3x2*ee*complex(0,1)*vevhat**2*ys*yt)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1130 = Coupling(
    name="GC_1130", value="-((complex(0,1)*ytau)/cmath.sqrt(2))", order={"QED": 1}
)

GC_1131 = Coupling(
    name="GC_1131",
    value="(-3*ceHIm*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1132 = Coupling(
    name="GC_1132",
    value="(3*ceHRe*complex(0,1)*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1133 = Coupling(
    name="GC_1133", value="-((ceWIm*ytau)/LambdaSMEFT**2)", order={"NP": 1, "QED": 3}
)

GC_1134 = Coupling(
    name="GC_1134", value="(ceWIm*ytau)/LambdaSMEFT**2", order={"NP": 1, "QED": 3}
)

GC_1135 = Coupling(
    name="GC_1135",
    value="(ceWRe*complex(0,1)*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_1136 = Coupling(
    name="GC_1136",
    value="-((ceBIm*cth*ytau)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1137 = Coupling(
    name="GC_1137",
    value="(ceBRe*cth*complex(0,1)*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1138 = Coupling(
    name="GC_1138",
    value="(ceWIm*cth*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1139 = Coupling(
    name="GC_1139",
    value="-((ceWRe*cth*complex(0,1)*ytau)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1140 = Coupling(
    name="GC_1140", value="(ceWIm*ee*ytau)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_1141 = Coupling(
    name="GC_1141",
    value="-((ceWRe*ee*complex(0,1)*ytau)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1142 = Coupling(
    name="GC_1142",
    value="(ceWRe*ee*complex(0,1)*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1143 = Coupling(
    name="GC_1143",
    value="-((complex(0,1)*propCorr*ytau)/cmath.sqrt(2))",
    order={"QED": 1},
)

GC_1144 = Coupling(
    name="GC_1144",
    value="-((ceWIm*ee*ytau)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_1145 = Coupling(
    name="GC_1145",
    value="(ceWRe*ee*complex(0,1)*ytau)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1146 = Coupling(
    name="GC_1146",
    value="-((ceWIm*cth*ee*ytau)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_1147 = Coupling(
    name="GC_1147",
    value="-((ceWRe*cth*ee*complex(0,1)*ytau)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_1148 = Coupling(
    name="GC_1148",
    value="(ceWRe*cth*ee*complex(0,1)*ytau)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_1149 = Coupling(
    name="GC_1149",
    value="(ceBIm*sth*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1150 = Coupling(
    name="GC_1150",
    value="-((ceBRe*complex(0,1)*sth*ytau)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1151 = Coupling(
    name="GC_1151",
    value="(ceWIm*sth*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1152 = Coupling(
    name="GC_1152",
    value="-((ceWRe*complex(0,1)*sth*ytau)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1153 = Coupling(
    name="GC_1153",
    value="(-3*ceHIm*vevhat*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1154 = Coupling(
    name="GC_1154",
    value="(3*ceHRe*complex(0,1)*vevhat*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1155 = Coupling(
    name="GC_1155",
    value="-((ceWIm*vevhat*ytau)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_1156 = Coupling(
    name="GC_1156",
    value="(ceWIm*vevhat*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1157 = Coupling(
    name="GC_1157",
    value="(ceWRe*complex(0,1)*vevhat*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1158 = Coupling(
    name="GC_1158",
    value="-((ceBIm*cth*vevhat*ytau)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1159 = Coupling(
    name="GC_1159",
    value="(ceBRe*cth*complex(0,1)*vevhat*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1160 = Coupling(
    name="GC_1160",
    value="(ceWIm*cth*vevhat*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1161 = Coupling(
    name="GC_1161",
    value="-((ceWRe*cth*complex(0,1)*vevhat*ytau)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1162 = Coupling(
    name="GC_1162",
    value="(ceWIm*ee*vevhat*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_1163 = Coupling(
    name="GC_1163",
    value="-((ceWRe*ee*complex(0,1)*vevhat*ytau)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_1164 = Coupling(
    name="GC_1164",
    value="(ceWRe*ee*complex(0,1)*vevhat*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_1165 = Coupling(
    name="GC_1165",
    value="-((ceWIm*ee*vevhat*ytau)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1166 = Coupling(
    name="GC_1166",
    value="(ceWRe*ee*complex(0,1)*vevhat*ytau)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1167 = Coupling(
    name="GC_1167",
    value="-((ceWIm*cth*ee*vevhat*ytau)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_1168 = Coupling(
    name="GC_1168",
    value="-((ceWRe*cth*ee*complex(0,1)*vevhat*ytau)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_1169 = Coupling(
    name="GC_1169",
    value="(ceWRe*cth*ee*complex(0,1)*vevhat*ytau)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_1170 = Coupling(
    name="GC_1170",
    value="(ceBIm*sth*vevhat*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1171 = Coupling(
    name="GC_1171",
    value="-((ceBRe*complex(0,1)*sth*vevhat*ytau)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1172 = Coupling(
    name="GC_1172",
    value="(ceWIm*sth*vevhat*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1173 = Coupling(
    name="GC_1173",
    value="-((ceWRe*complex(0,1)*sth*vevhat*ytau)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1174 = Coupling(
    name="GC_1174",
    value="-((ceHIm*vevhat**2*ytau)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1175 = Coupling(
    name="GC_1175",
    value="(ceHRe*complex(0,1)*vevhat**2*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1176 = Coupling(
    name="GC_1176",
    value="-((cHbox*complex(0,1)*vevhat**2*ytau)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1177 = Coupling(
    name="GC_1177",
    value="(cHDD*complex(0,1)*vevhat**2*ytau)/(4.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1178 = Coupling(
    name="GC_1178",
    value="(cHl3*complex(0,1)*vevhat**2*ytau)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1179 = Coupling(
    name="GC_1179",
    value="-(cll1*complex(0,1)*vevhat**2*ytau)/(2.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1180 = Coupling(
    name="GC_1180",
    value="-((cledqIm*yb*ytau)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1181 = Coupling(
    name="GC_1181", value="(cledqIm*yb*ytau)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_1182 = Coupling(
    name="GC_1182",
    value="(CKM1x3*cledqIm*yb*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1183 = Coupling(
    name="GC_1183",
    value="(CKM2x3*cledqIm*yb*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1184 = Coupling(
    name="GC_1184",
    value="(CKM3x3*cledqIm*yb*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1185 = Coupling(
    name="GC_1185",
    value="(cledqRe*complex(0,1)*yb*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1186 = Coupling(
    name="GC_1186",
    value="(CKM1x3*cledqRe*complex(0,1)*yb*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1187 = Coupling(
    name="GC_1187",
    value="(CKM2x3*cledqRe*complex(0,1)*yb*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1188 = Coupling(
    name="GC_1188",
    value="(CKM3x3*cledqRe*complex(0,1)*yb*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1189 = Coupling(
    name="GC_1189",
    value="-((clequ1Im*yc*ytau)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1190 = Coupling(
    name="GC_1190",
    value="(clequ1Im*yc*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1191 = Coupling(
    name="GC_1191",
    value="(CKM2x1*clequ1Im*yc*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1192 = Coupling(
    name="GC_1192",
    value="(CKM2x2*clequ1Im*yc*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1193 = Coupling(
    name="GC_1193",
    value="(CKM2x3*clequ1Im*yc*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1194 = Coupling(
    name="GC_1194",
    value="-((clequ1Re*complex(0,1)*yc*ytau)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1195 = Coupling(
    name="GC_1195",
    value="(CKM2x1*clequ1Re*complex(0,1)*yc*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1196 = Coupling(
    name="GC_1196",
    value="(CKM2x2*clequ1Re*complex(0,1)*yc*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1197 = Coupling(
    name="GC_1197",
    value="(CKM2x3*clequ1Re*complex(0,1)*yc*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1198 = Coupling(
    name="GC_1198",
    value="-(clequ3Im*yc*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1199 = Coupling(
    name="GC_1199",
    value="-(clequ3Im*yc*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1200 = Coupling(
    name="GC_1200",
    value="(clequ3Im*yc*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1201 = Coupling(
    name="GC_1201",
    value="(clequ3Im*yc*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1202 = Coupling(
    name="GC_1202",
    value="-(CKM2x1*clequ3Im*yc*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1203 = Coupling(
    name="GC_1203",
    value="(CKM2x1*clequ3Im*yc*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1204 = Coupling(
    name="GC_1204",
    value="-(CKM2x2*clequ3Im*yc*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1205 = Coupling(
    name="GC_1205",
    value="(CKM2x2*clequ3Im*yc*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1206 = Coupling(
    name="GC_1206",
    value="-(CKM2x3*clequ3Im*yc*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1207 = Coupling(
    name="GC_1207",
    value="(CKM2x3*clequ3Im*yc*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1208 = Coupling(
    name="GC_1208",
    value="-(clequ3Re*complex(0,1)*yc*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1209 = Coupling(
    name="GC_1209",
    value="(clequ3Re*complex(0,1)*yc*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1210 = Coupling(
    name="GC_1210",
    value="(CKM2x1*clequ3Re*complex(0,1)*yc*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1211 = Coupling(
    name="GC_1211",
    value="-(CKM2x1*clequ3Re*complex(0,1)*yc*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1212 = Coupling(
    name="GC_1212",
    value="(CKM2x2*clequ3Re*complex(0,1)*yc*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1213 = Coupling(
    name="GC_1213",
    value="-(CKM2x2*clequ3Re*complex(0,1)*yc*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1214 = Coupling(
    name="GC_1214",
    value="(CKM2x3*clequ3Re*complex(0,1)*yc*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1215 = Coupling(
    name="GC_1215",
    value="-(CKM2x3*clequ3Re*complex(0,1)*yc*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1216 = Coupling(
    name="GC_1216",
    value="-((cledqIm*ydo*ytau)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1217 = Coupling(
    name="GC_1217", value="(cledqIm*ydo*ytau)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_1218 = Coupling(
    name="GC_1218",
    value="(CKM1x1*cledqIm*ydo*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1219 = Coupling(
    name="GC_1219",
    value="(CKM2x1*cledqIm*ydo*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1220 = Coupling(
    name="GC_1220",
    value="(CKM3x1*cledqIm*ydo*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1221 = Coupling(
    name="GC_1221",
    value="(cledqRe*complex(0,1)*ydo*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1222 = Coupling(
    name="GC_1222",
    value="(CKM1x1*cledqRe*complex(0,1)*ydo*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1223 = Coupling(
    name="GC_1223",
    value="(CKM2x1*cledqRe*complex(0,1)*ydo*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1224 = Coupling(
    name="GC_1224",
    value="(CKM3x1*cledqRe*complex(0,1)*ydo*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1225 = Coupling(
    name="GC_1225",
    value="-((cledqIm*ys*ytau)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1226 = Coupling(
    name="GC_1226", value="(cledqIm*ys*ytau)/LambdaSMEFT**2", order={"NP": 1, "QED": 4}
)

GC_1227 = Coupling(
    name="GC_1227",
    value="(CKM1x2*cledqIm*ys*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1228 = Coupling(
    name="GC_1228",
    value="(CKM2x2*cledqIm*ys*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1229 = Coupling(
    name="GC_1229",
    value="(CKM3x2*cledqIm*ys*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1230 = Coupling(
    name="GC_1230",
    value="(cledqRe*complex(0,1)*ys*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1231 = Coupling(
    name="GC_1231",
    value="(CKM1x2*cledqRe*complex(0,1)*ys*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1232 = Coupling(
    name="GC_1232",
    value="(CKM2x2*cledqRe*complex(0,1)*ys*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1233 = Coupling(
    name="GC_1233",
    value="(CKM3x2*cledqRe*complex(0,1)*ys*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1234 = Coupling(
    name="GC_1234",
    value="-((clequ1Im*yt*ytau)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1235 = Coupling(
    name="GC_1235",
    value="(clequ1Im*yt*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1236 = Coupling(
    name="GC_1236",
    value="(CKM3x1*clequ1Im*yt*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1237 = Coupling(
    name="GC_1237",
    value="(CKM3x2*clequ1Im*yt*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1238 = Coupling(
    name="GC_1238",
    value="(CKM3x3*clequ1Im*yt*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1239 = Coupling(
    name="GC_1239",
    value="-((clequ1Re*complex(0,1)*yt*ytau)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1240 = Coupling(
    name="GC_1240",
    value="(CKM3x1*clequ1Re*complex(0,1)*yt*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1241 = Coupling(
    name="GC_1241",
    value="(CKM3x2*clequ1Re*complex(0,1)*yt*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1242 = Coupling(
    name="GC_1242",
    value="(CKM3x3*clequ1Re*complex(0,1)*yt*ytau)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1243 = Coupling(
    name="GC_1243",
    value="-(clequ3Im*yt*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1244 = Coupling(
    name="GC_1244",
    value="-(clequ3Im*yt*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1245 = Coupling(
    name="GC_1245",
    value="(clequ3Im*yt*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1246 = Coupling(
    name="GC_1246",
    value="(clequ3Im*yt*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1247 = Coupling(
    name="GC_1247",
    value="-(CKM3x1*clequ3Im*yt*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1248 = Coupling(
    name="GC_1248",
    value="(CKM3x1*clequ3Im*yt*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1249 = Coupling(
    name="GC_1249",
    value="-(CKM3x2*clequ3Im*yt*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1250 = Coupling(
    name="GC_1250",
    value="(CKM3x2*clequ3Im*yt*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1251 = Coupling(
    name="GC_1251",
    value="-(CKM3x3*clequ3Im*yt*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1252 = Coupling(
    name="GC_1252",
    value="(CKM3x3*clequ3Im*yt*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1253 = Coupling(
    name="GC_1253",
    value="-(clequ3Re*complex(0,1)*yt*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1254 = Coupling(
    name="GC_1254",
    value="(clequ3Re*complex(0,1)*yt*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1255 = Coupling(
    name="GC_1255",
    value="(CKM3x1*clequ3Re*complex(0,1)*yt*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1256 = Coupling(
    name="GC_1256",
    value="-(CKM3x1*clequ3Re*complex(0,1)*yt*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1257 = Coupling(
    name="GC_1257",
    value="(CKM3x2*clequ3Re*complex(0,1)*yt*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1258 = Coupling(
    name="GC_1258",
    value="-(CKM3x2*clequ3Re*complex(0,1)*yt*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1259 = Coupling(
    name="GC_1259",
    value="(CKM3x3*clequ3Re*complex(0,1)*yt*ytau)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1260 = Coupling(
    name="GC_1260",
    value="-(CKM3x3*clequ3Re*complex(0,1)*yt*ytau)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1261 = Coupling(
    name="GC_1261", value="-((complex(0,1)*yup)/cmath.sqrt(2))", order={"QED": 1}
)

GC_1262 = Coupling(
    name="GC_1262",
    value="-((cth*cuBIm*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1263 = Coupling(
    name="GC_1263",
    value="(cth*cuBRe*complex(0,1)*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1264 = Coupling(
    name="GC_1264",
    value="-((cuGIm*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1265 = Coupling(
    name="GC_1265",
    value="(cuGRe*complex(0,1)*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1266 = Coupling(
    name="GC_1266",
    value="(-3*cuHIm*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1267 = Coupling(
    name="GC_1267",
    value="(3*cuHRe*complex(0,1)*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1268 = Coupling(
    name="GC_1268", value="-((cuWIm*yup)/LambdaSMEFT**2)", order={"NP": 1, "QED": 3}
)

GC_1269 = Coupling(
    name="GC_1269", value="(cuWIm*yup)/LambdaSMEFT**2", order={"NP": 1, "QED": 3}
)

GC_1270 = Coupling(
    name="GC_1270",
    value="-((cth*cuWIm*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1271 = Coupling(
    name="GC_1271",
    value="(cuWRe*complex(0,1)*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_1272 = Coupling(
    name="GC_1272",
    value="(cth*cuWRe*complex(0,1)*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1273 = Coupling(
    name="GC_1273", value="-((cuWIm*ee*yup)/LambdaSMEFT**2)", order={"NP": 1, "QED": 4}
)

GC_1274 = Coupling(
    name="GC_1274",
    value="-((cuWRe*ee*complex(0,1)*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1275 = Coupling(
    name="GC_1275",
    value="(cuWRe*ee*complex(0,1)*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1276 = Coupling(
    name="GC_1276",
    value="(cuGIm*complex(0,1)*G*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_1277 = Coupling(
    name="GC_1277",
    value="(cuGRe*G*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 3},
)

GC_1278 = Coupling(
    name="GC_1278",
    value="-((complex(0,1)*propCorr*yup)/cmath.sqrt(2))",
    order={"QED": 1},
)

GC_1279 = Coupling(
    name="GC_1279",
    value="(cuWIm*ee*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1280 = Coupling(
    name="GC_1280",
    value="(cth*cuWIm*ee*yup)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_1281 = Coupling(
    name="GC_1281",
    value="-((cuWRe*ee*complex(0,1)*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_1282 = Coupling(
    name="GC_1282",
    value="-((cth*cuWRe*ee*complex(0,1)*yup)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 4},
)

GC_1283 = Coupling(
    name="GC_1283",
    value="(cth*cuWRe*ee*complex(0,1)*yup)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 4},
)

GC_1284 = Coupling(
    name="GC_1284",
    value="(cuBIm*sth*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1285 = Coupling(
    name="GC_1285",
    value="-((cuBRe*complex(0,1)*sth*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1286 = Coupling(
    name="GC_1286",
    value="-((cuWIm*sth*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1287 = Coupling(
    name="GC_1287",
    value="(cuWRe*complex(0,1)*sth*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1288 = Coupling(
    name="GC_1288",
    value="-((cth*cuBIm*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1289 = Coupling(
    name="GC_1289",
    value="(cth*cuBRe*complex(0,1)*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1290 = Coupling(
    name="GC_1290",
    value="-((cuGIm*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1291 = Coupling(
    name="GC_1291",
    value="(cuGRe*complex(0,1)*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1292 = Coupling(
    name="GC_1292",
    value="(-3*cuHIm*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1293 = Coupling(
    name="GC_1293",
    value="(3*cuHRe*complex(0,1)*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1294 = Coupling(
    name="GC_1294",
    value="-((cuWIm*vevhat*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 2},
)

GC_1295 = Coupling(
    name="GC_1295", value="(cuWIm*vevhat*yup)/LambdaSMEFT**2", order={"NP": 1, "QED": 2}
)

GC_1296 = Coupling(
    name="GC_1296",
    value="-((cth*cuWIm*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1297 = Coupling(
    name="GC_1297",
    value="(cuWRe*complex(0,1)*vevhat*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1298 = Coupling(
    name="GC_1298",
    value="(cth*cuWRe*complex(0,1)*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1299 = Coupling(
    name="GC_1299",
    value="-((cuWIm*ee*vevhat*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_1300 = Coupling(
    name="GC_1300",
    value="-((cuWRe*ee*complex(0,1)*vevhat*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 3},
)

GC_1301 = Coupling(
    name="GC_1301",
    value="(cuWRe*ee*complex(0,1)*vevhat*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 3},
)

GC_1302 = Coupling(
    name="GC_1302",
    value="(cuGIm*complex(0,1)*G*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_1303 = Coupling(
    name="GC_1303",
    value="(cuGRe*G*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QCD": 1, "QED": 2},
)

GC_1304 = Coupling(
    name="GC_1304",
    value="(cuWIm*ee*vevhat*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1305 = Coupling(
    name="GC_1305",
    value="(cth*cuWIm*ee*vevhat*yup)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_1306 = Coupling(
    name="GC_1306",
    value="-((cuWRe*ee*complex(0,1)*vevhat*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 3},
)

GC_1307 = Coupling(
    name="GC_1307",
    value="-((cth*cuWRe*ee*complex(0,1)*vevhat*yup)/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_1308 = Coupling(
    name="GC_1308",
    value="(cth*cuWRe*ee*complex(0,1)*vevhat*yup)/(LambdaSMEFT**2*sth)",
    order={"NP": 1, "QED": 3},
)

GC_1309 = Coupling(
    name="GC_1309",
    value="(cuBIm*sth*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1310 = Coupling(
    name="GC_1310",
    value="-((cuBRe*complex(0,1)*sth*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1311 = Coupling(
    name="GC_1311",
    value="-((cuWIm*sth*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 2},
)

GC_1312 = Coupling(
    name="GC_1312",
    value="(cuWRe*complex(0,1)*sth*vevhat*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 2},
)

GC_1313 = Coupling(
    name="GC_1313",
    value="-((cHbox*complex(0,1)*vevhat**2*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1314 = Coupling(
    name="GC_1314",
    value="(cHDD*complex(0,1)*vevhat**2*yup)/(4.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1315 = Coupling(
    name="GC_1315",
    value="(cHl3*complex(0,1)*vevhat**2*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1316 = Coupling(
    name="GC_1316",
    value="-(cll1*complex(0,1)*vevhat**2*yup)/(2.*LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1317 = Coupling(
    name="GC_1317",
    value="-((cuHIm*vevhat**2*yup)/(LambdaSMEFT**2*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1318 = Coupling(
    name="GC_1318",
    value="(cuHRe*complex(0,1)*vevhat**2*yup)/(LambdaSMEFT**2*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1319 = Coupling(
    name="GC_1319",
    value="-((cquqd11Im*yb*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1320 = Coupling(
    name="GC_1320",
    value="(cquqd11Im*yb*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1321 = Coupling(
    name="GC_1321",
    value="-((cquqd11Re*complex(0,1)*yb*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1322 = Coupling(
    name="GC_1322",
    value="-((cquqd1Im*yb*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1323 = Coupling(
    name="GC_1323",
    value="(cquqd1Im*yb*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1324 = Coupling(
    name="GC_1324",
    value="(cquqd1Re*complex(0,1)*yb*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1325 = Coupling(
    name="GC_1325",
    value="-((cquqd81Im*yb*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1326 = Coupling(
    name="GC_1326",
    value="(cquqd81Im*yb*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1327 = Coupling(
    name="GC_1327",
    value="-((cquqd81Re*complex(0,1)*yb*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1328 = Coupling(
    name="GC_1328",
    value="-((cquqd8Im*yb*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1329 = Coupling(
    name="GC_1329",
    value="(cquqd8Im*yb*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1330 = Coupling(
    name="GC_1330",
    value="(cquqd8Re*complex(0,1)*yb*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1331 = Coupling(
    name="GC_1331",
    value="-((cHudIm*CKM1x3*ee*yb*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 5},
)

GC_1332 = Coupling(
    name="GC_1332",
    value="(cHudRe*CKM1x3*ee*complex(0,1)*yb*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_1333 = Coupling(
    name="GC_1333",
    value="-((cHudIm*CKM1x3*ee*vevhat*yb*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_1334 = Coupling(
    name="GC_1334",
    value="(cHudRe*CKM1x3*ee*complex(0,1)*vevhat*yb*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1335 = Coupling(
    name="GC_1335",
    value="-(cHudIm*CKM1x3*ee*vevhat**2*yb*yup)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1336 = Coupling(
    name="GC_1336",
    value="(cHudRe*CKM1x3*ee*complex(0,1)*vevhat**2*yb*yup)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1337 = Coupling(
    name="GC_1337",
    value="-((cquqd11Im*ydo*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1338 = Coupling(
    name="GC_1338",
    value="(cquqd11Im*ydo*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1339 = Coupling(
    name="GC_1339",
    value="-((cquqd11Re*complex(0,1)*ydo*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1340 = Coupling(
    name="GC_1340",
    value="-((cquqd1Im*ydo*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1341 = Coupling(
    name="GC_1341",
    value="(cquqd1Im*ydo*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1342 = Coupling(
    name="GC_1342",
    value="(cquqd1Re*complex(0,1)*ydo*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1343 = Coupling(
    name="GC_1343",
    value="-((cquqd81Im*ydo*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1344 = Coupling(
    name="GC_1344",
    value="(cquqd81Im*ydo*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1345 = Coupling(
    name="GC_1345",
    value="-((cquqd81Re*complex(0,1)*ydo*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1346 = Coupling(
    name="GC_1346",
    value="-((cquqd8Im*ydo*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1347 = Coupling(
    name="GC_1347",
    value="(cquqd8Im*ydo*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1348 = Coupling(
    name="GC_1348",
    value="(cquqd8Re*complex(0,1)*ydo*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1349 = Coupling(
    name="GC_1349",
    value="-((cHudIm*CKM1x1*ee*ydo*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 5},
)

GC_1350 = Coupling(
    name="GC_1350",
    value="(cHudRe*CKM1x1*ee*complex(0,1)*ydo*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_1351 = Coupling(
    name="GC_1351",
    value="-((cHudIm*CKM1x1*ee*vevhat*ydo*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_1352 = Coupling(
    name="GC_1352",
    value="(cHudRe*CKM1x1*ee*complex(0,1)*vevhat*ydo*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1353 = Coupling(
    name="GC_1353",
    value="-(cHudIm*CKM1x1*ee*vevhat**2*ydo*yup)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1354 = Coupling(
    name="GC_1354",
    value="(cHudRe*CKM1x1*ee*complex(0,1)*vevhat**2*ydo*yup)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1355 = Coupling(
    name="GC_1355",
    value="-((clequ1Im*ye*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1356 = Coupling(
    name="GC_1356",
    value="(clequ1Im*ye*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1357 = Coupling(
    name="GC_1357",
    value="(CKM1x1*clequ1Im*ye*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1358 = Coupling(
    name="GC_1358",
    value="(CKM1x2*clequ1Im*ye*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1359 = Coupling(
    name="GC_1359",
    value="(CKM1x3*clequ1Im*ye*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1360 = Coupling(
    name="GC_1360",
    value="-((clequ1Re*complex(0,1)*ye*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1361 = Coupling(
    name="GC_1361",
    value="(CKM1x1*clequ1Re*complex(0,1)*ye*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1362 = Coupling(
    name="GC_1362",
    value="(CKM1x2*clequ1Re*complex(0,1)*ye*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1363 = Coupling(
    name="GC_1363",
    value="(CKM1x3*clequ1Re*complex(0,1)*ye*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1364 = Coupling(
    name="GC_1364",
    value="-(clequ3Im*ye*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1365 = Coupling(
    name="GC_1365",
    value="-(clequ3Im*ye*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1366 = Coupling(
    name="GC_1366",
    value="(clequ3Im*ye*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1367 = Coupling(
    name="GC_1367",
    value="(clequ3Im*ye*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1368 = Coupling(
    name="GC_1368",
    value="-(CKM1x1*clequ3Im*ye*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1369 = Coupling(
    name="GC_1369",
    value="(CKM1x1*clequ3Im*ye*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1370 = Coupling(
    name="GC_1370",
    value="-(CKM1x2*clequ3Im*ye*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1371 = Coupling(
    name="GC_1371",
    value="(CKM1x2*clequ3Im*ye*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1372 = Coupling(
    name="GC_1372",
    value="-(CKM1x3*clequ3Im*ye*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1373 = Coupling(
    name="GC_1373",
    value="(CKM1x3*clequ3Im*ye*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1374 = Coupling(
    name="GC_1374",
    value="-(clequ3Re*complex(0,1)*ye*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1375 = Coupling(
    name="GC_1375",
    value="(clequ3Re*complex(0,1)*ye*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1376 = Coupling(
    name="GC_1376",
    value="(CKM1x1*clequ3Re*complex(0,1)*ye*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1377 = Coupling(
    name="GC_1377",
    value="-(CKM1x1*clequ3Re*complex(0,1)*ye*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1378 = Coupling(
    name="GC_1378",
    value="(CKM1x2*clequ3Re*complex(0,1)*ye*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1379 = Coupling(
    name="GC_1379",
    value="-(CKM1x2*clequ3Re*complex(0,1)*ye*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1380 = Coupling(
    name="GC_1380",
    value="(CKM1x3*clequ3Re*complex(0,1)*ye*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1381 = Coupling(
    name="GC_1381",
    value="-(CKM1x3*clequ3Re*complex(0,1)*ye*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1382 = Coupling(
    name="GC_1382",
    value="-((clequ1Im*ym*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1383 = Coupling(
    name="GC_1383",
    value="(clequ1Im*ym*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1384 = Coupling(
    name="GC_1384",
    value="(CKM1x1*clequ1Im*ym*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1385 = Coupling(
    name="GC_1385",
    value="(CKM1x2*clequ1Im*ym*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1386 = Coupling(
    name="GC_1386",
    value="(CKM1x3*clequ1Im*ym*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1387 = Coupling(
    name="GC_1387",
    value="-((clequ1Re*complex(0,1)*ym*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1388 = Coupling(
    name="GC_1388",
    value="(CKM1x1*clequ1Re*complex(0,1)*ym*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1389 = Coupling(
    name="GC_1389",
    value="(CKM1x2*clequ1Re*complex(0,1)*ym*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1390 = Coupling(
    name="GC_1390",
    value="(CKM1x3*clequ1Re*complex(0,1)*ym*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1391 = Coupling(
    name="GC_1391",
    value="-(clequ3Im*ym*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1392 = Coupling(
    name="GC_1392",
    value="-(clequ3Im*ym*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1393 = Coupling(
    name="GC_1393",
    value="(clequ3Im*ym*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1394 = Coupling(
    name="GC_1394",
    value="(clequ3Im*ym*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1395 = Coupling(
    name="GC_1395",
    value="-(CKM1x1*clequ3Im*ym*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1396 = Coupling(
    name="GC_1396",
    value="(CKM1x1*clequ3Im*ym*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1397 = Coupling(
    name="GC_1397",
    value="-(CKM1x2*clequ3Im*ym*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1398 = Coupling(
    name="GC_1398",
    value="(CKM1x2*clequ3Im*ym*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1399 = Coupling(
    name="GC_1399",
    value="-(CKM1x3*clequ3Im*ym*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1400 = Coupling(
    name="GC_1400",
    value="(CKM1x3*clequ3Im*ym*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1401 = Coupling(
    name="GC_1401",
    value="-(clequ3Re*complex(0,1)*ym*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1402 = Coupling(
    name="GC_1402",
    value="(clequ3Re*complex(0,1)*ym*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1403 = Coupling(
    name="GC_1403",
    value="(CKM1x1*clequ3Re*complex(0,1)*ym*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1404 = Coupling(
    name="GC_1404",
    value="-(CKM1x1*clequ3Re*complex(0,1)*ym*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1405 = Coupling(
    name="GC_1405",
    value="(CKM1x2*clequ3Re*complex(0,1)*ym*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1406 = Coupling(
    name="GC_1406",
    value="-(CKM1x2*clequ3Re*complex(0,1)*ym*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1407 = Coupling(
    name="GC_1407",
    value="(CKM1x3*clequ3Re*complex(0,1)*ym*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1408 = Coupling(
    name="GC_1408",
    value="-(CKM1x3*clequ3Re*complex(0,1)*ym*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1409 = Coupling(
    name="GC_1409",
    value="-((cquqd11Im*ys*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1410 = Coupling(
    name="GC_1410",
    value="(cquqd11Im*ys*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1411 = Coupling(
    name="GC_1411",
    value="-((cquqd11Re*complex(0,1)*ys*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1412 = Coupling(
    name="GC_1412",
    value="-((cquqd1Im*ys*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1413 = Coupling(
    name="GC_1413",
    value="(cquqd1Im*ys*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1414 = Coupling(
    name="GC_1414",
    value="(cquqd1Re*complex(0,1)*ys*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1415 = Coupling(
    name="GC_1415",
    value="-((cquqd81Im*ys*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1416 = Coupling(
    name="GC_1416",
    value="(cquqd81Im*ys*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1417 = Coupling(
    name="GC_1417",
    value="-((cquqd81Re*complex(0,1)*ys*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1418 = Coupling(
    name="GC_1418",
    value="-((cquqd8Im*ys*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1419 = Coupling(
    name="GC_1419",
    value="(cquqd8Im*ys*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1420 = Coupling(
    name="GC_1420",
    value="(cquqd8Re*complex(0,1)*ys*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1421 = Coupling(
    name="GC_1421",
    value="-((cHudIm*CKM1x2*ee*ys*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 5},
)

GC_1422 = Coupling(
    name="GC_1422",
    value="(cHudRe*CKM1x2*ee*complex(0,1)*ys*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_1423 = Coupling(
    name="GC_1423",
    value="-((cHudIm*CKM1x2*ee*vevhat*ys*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 4},
)

GC_1424 = Coupling(
    name="GC_1424",
    value="(cHudRe*CKM1x2*ee*complex(0,1)*vevhat*ys*yup)/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1425 = Coupling(
    name="GC_1425",
    value="-(cHudIm*CKM1x2*ee*vevhat**2*ys*yup)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1426 = Coupling(
    name="GC_1426",
    value="(cHudRe*CKM1x2*ee*complex(0,1)*vevhat**2*ys*yup)/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1427 = Coupling(
    name="GC_1427",
    value="-((clequ1Im*ytau*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1428 = Coupling(
    name="GC_1428",
    value="(clequ1Im*ytau*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1429 = Coupling(
    name="GC_1429",
    value="(CKM1x1*clequ1Im*ytau*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1430 = Coupling(
    name="GC_1430",
    value="(CKM1x2*clequ1Im*ytau*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1431 = Coupling(
    name="GC_1431",
    value="(CKM1x3*clequ1Im*ytau*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1432 = Coupling(
    name="GC_1432",
    value="-((clequ1Re*complex(0,1)*ytau*yup)/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1433 = Coupling(
    name="GC_1433",
    value="(CKM1x1*clequ1Re*complex(0,1)*ytau*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1434 = Coupling(
    name="GC_1434",
    value="(CKM1x2*clequ1Re*complex(0,1)*ytau*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1435 = Coupling(
    name="GC_1435",
    value="(CKM1x3*clequ1Re*complex(0,1)*ytau*yup)/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1436 = Coupling(
    name="GC_1436",
    value="-(clequ3Im*ytau*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1437 = Coupling(
    name="GC_1437",
    value="-(clequ3Im*ytau*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1438 = Coupling(
    name="GC_1438",
    value="(clequ3Im*ytau*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1439 = Coupling(
    name="GC_1439",
    value="(clequ3Im*ytau*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1440 = Coupling(
    name="GC_1440",
    value="-(CKM1x1*clequ3Im*ytau*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1441 = Coupling(
    name="GC_1441",
    value="(CKM1x1*clequ3Im*ytau*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1442 = Coupling(
    name="GC_1442",
    value="-(CKM1x2*clequ3Im*ytau*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1443 = Coupling(
    name="GC_1443",
    value="(CKM1x2*clequ3Im*ytau*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1444 = Coupling(
    name="GC_1444",
    value="-(CKM1x3*clequ3Im*ytau*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1445 = Coupling(
    name="GC_1445",
    value="(CKM1x3*clequ3Im*ytau*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1446 = Coupling(
    name="GC_1446",
    value="-(clequ3Re*complex(0,1)*ytau*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1447 = Coupling(
    name="GC_1447",
    value="(clequ3Re*complex(0,1)*ytau*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1448 = Coupling(
    name="GC_1448",
    value="(CKM1x1*clequ3Re*complex(0,1)*ytau*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1449 = Coupling(
    name="GC_1449",
    value="-(CKM1x1*clequ3Re*complex(0,1)*ytau*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1450 = Coupling(
    name="GC_1450",
    value="(CKM1x2*clequ3Re*complex(0,1)*ytau*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1451 = Coupling(
    name="GC_1451",
    value="-(CKM1x2*clequ3Re*complex(0,1)*ytau*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1452 = Coupling(
    name="GC_1452",
    value="(CKM1x3*clequ3Re*complex(0,1)*ytau*yup)/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1453 = Coupling(
    name="GC_1453",
    value="-(CKM1x3*clequ3Re*complex(0,1)*ytau*yup)/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1454 = Coupling(
    name="GC_1454",
    value="(2*clq3*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1455 = Coupling(
    name="GC_1455",
    value="(2*CKM1x1*cqq11*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1456 = Coupling(
    name="GC_1456",
    value="(2*CKM1x2*cqq11*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1457 = Coupling(
    name="GC_1457",
    value="(2*CKM1x3*cqq11*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1458 = Coupling(
    name="GC_1458",
    value="(2*CKM2x1*cqq11*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1459 = Coupling(
    name="GC_1459",
    value="(2*CKM2x2*cqq11*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1460 = Coupling(
    name="GC_1460",
    value="(2*CKM2x3*cqq11*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1461 = Coupling(
    name="GC_1461",
    value="(2*CKM3x1*cqq11*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1462 = Coupling(
    name="GC_1462",
    value="(2*CKM3x2*cqq11*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1463 = Coupling(
    name="GC_1463",
    value="(2*CKM3x3*cqq11*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1464 = Coupling(
    name="GC_1464",
    value="(4*CKM1x1*cqq3*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1465 = Coupling(
    name="GC_1465",
    value="(4*CKM1x2*cqq3*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1466 = Coupling(
    name="GC_1466",
    value="(4*CKM1x3*cqq3*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1467 = Coupling(
    name="GC_1467",
    value="(4*CKM2x1*cqq3*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1468 = Coupling(
    name="GC_1468",
    value="(4*CKM2x2*cqq3*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1469 = Coupling(
    name="GC_1469",
    value="(4*CKM2x3*cqq3*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1470 = Coupling(
    name="GC_1470",
    value="(4*CKM3x1*cqq3*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1471 = Coupling(
    name="GC_1471",
    value="(4*CKM3x2*cqq3*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1472 = Coupling(
    name="GC_1472",
    value="(4*CKM3x3*cqq3*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1473 = Coupling(
    name="GC_1473",
    value="(-2*CKM1x1*cqq31*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1474 = Coupling(
    name="GC_1474",
    value="(-2*CKM1x2*cqq31*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1475 = Coupling(
    name="GC_1475",
    value="(-2*CKM1x3*cqq31*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1476 = Coupling(
    name="GC_1476",
    value="(-2*CKM2x1*cqq31*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1477 = Coupling(
    name="GC_1477",
    value="(-2*CKM2x2*cqq31*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1478 = Coupling(
    name="GC_1478",
    value="(-2*CKM2x3*cqq31*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1479 = Coupling(
    name="GC_1479",
    value="(-2*CKM3x1*cqq31*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1480 = Coupling(
    name="GC_1480",
    value="(-2*CKM3x2*cqq31*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1481 = Coupling(
    name="GC_1481",
    value="(-2*CKM3x3*cqq31*complex(0,1)*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1482 = Coupling(
    name="GC_1482",
    value="-((ee*complex(0,1)*complexconjugate(CKM1x1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_1483 = Coupling(
    name="GC_1483",
    value="-((cHq3*ee*complex(0,1)*complexconjugate(CKM1x1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_1484 = Coupling(
    name="GC_1484",
    value="-((ee*complex(0,1)*propCorr*complexconjugate(CKM1x1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_1485 = Coupling(
    name="GC_1485",
    value="-((cHq3*ee*complex(0,1)*vevhat*complexconjugate(CKM1x1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_1486 = Coupling(
    name="GC_1486",
    value="-((cHq3*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1487 = Coupling(
    name="GC_1487",
    value="-((cHWB*cth*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x1))/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1488 = Coupling(
    name="GC_1488",
    value="-((cHl3*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x1))/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1489 = Coupling(
    name="GC_1489",
    value="(cll1*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x1))/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1490 = Coupling(
    name="GC_1490",
    value="(cHDD*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x1))/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1491 = Coupling(
    name="GC_1491",
    value="(CKM2x1*cquqd11Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1492 = Coupling(
    name="GC_1492",
    value="(CKM2x2*cquqd11Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1493 = Coupling(
    name="GC_1493",
    value="(CKM2x3*cquqd11Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1494 = Coupling(
    name="GC_1494",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1495 = Coupling(
    name="GC_1495",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1496 = Coupling(
    name="GC_1496",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1497 = Coupling(
    name="GC_1497",
    value="-((CKM2x1*cquqd1Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1498 = Coupling(
    name="GC_1498",
    value="-((CKM2x2*cquqd1Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1499 = Coupling(
    name="GC_1499",
    value="-((CKM2x3*cquqd1Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1500 = Coupling(
    name="GC_1500",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1501 = Coupling(
    name="GC_1501",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1502 = Coupling(
    name="GC_1502",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1503 = Coupling(
    name="GC_1503",
    value="(CKM2x1*cquqd81Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1504 = Coupling(
    name="GC_1504",
    value="(CKM2x2*cquqd81Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1505 = Coupling(
    name="GC_1505",
    value="(CKM2x3*cquqd81Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1506 = Coupling(
    name="GC_1506",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1507 = Coupling(
    name="GC_1507",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1508 = Coupling(
    name="GC_1508",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1509 = Coupling(
    name="GC_1509",
    value="-((CKM2x1*cquqd8Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1510 = Coupling(
    name="GC_1510",
    value="-((CKM2x2*cquqd8Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1511 = Coupling(
    name="GC_1511",
    value="-((CKM2x3*cquqd8Im*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1512 = Coupling(
    name="GC_1512",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1513 = Coupling(
    name="GC_1513",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1514 = Coupling(
    name="GC_1514",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1515 = Coupling(
    name="GC_1515",
    value="-((cledqIm*ydo*ye*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1516 = Coupling(
    name="GC_1516",
    value="(cledqRe*complex(0,1)*ydo*ye*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1517 = Coupling(
    name="GC_1517",
    value="-((cledqIm*ydo*ym*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1518 = Coupling(
    name="GC_1518",
    value="(cledqRe*complex(0,1)*ydo*ym*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1519 = Coupling(
    name="GC_1519",
    value="(CKM3x1*cquqd11Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1520 = Coupling(
    name="GC_1520",
    value="(CKM3x2*cquqd11Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1521 = Coupling(
    name="GC_1521",
    value="(CKM3x3*cquqd11Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1522 = Coupling(
    name="GC_1522",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1523 = Coupling(
    name="GC_1523",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1524 = Coupling(
    name="GC_1524",
    value="(CKM3x3*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1525 = Coupling(
    name="GC_1525",
    value="-((CKM3x1*cquqd1Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1526 = Coupling(
    name="GC_1526",
    value="-((CKM3x2*cquqd1Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1527 = Coupling(
    name="GC_1527",
    value="-((CKM3x3*cquqd1Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1528 = Coupling(
    name="GC_1528",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1529 = Coupling(
    name="GC_1529",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1530 = Coupling(
    name="GC_1530",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1531 = Coupling(
    name="GC_1531",
    value="(CKM3x1*cquqd81Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1532 = Coupling(
    name="GC_1532",
    value="(CKM3x2*cquqd81Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1533 = Coupling(
    name="GC_1533",
    value="(CKM3x3*cquqd81Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1534 = Coupling(
    name="GC_1534",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1535 = Coupling(
    name="GC_1535",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1536 = Coupling(
    name="GC_1536",
    value="(CKM3x3*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1537 = Coupling(
    name="GC_1537",
    value="-((CKM3x1*cquqd8Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1538 = Coupling(
    name="GC_1538",
    value="-((CKM3x2*cquqd8Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1539 = Coupling(
    name="GC_1539",
    value="-((CKM3x3*cquqd8Im*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1540 = Coupling(
    name="GC_1540",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1541 = Coupling(
    name="GC_1541",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1542 = Coupling(
    name="GC_1542",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1543 = Coupling(
    name="GC_1543",
    value="-((cledqIm*ydo*ytau*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1544 = Coupling(
    name="GC_1544",
    value="(cledqRe*complex(0,1)*ydo*ytau*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1545 = Coupling(
    name="GC_1545",
    value="-((CKM1x3*cquqd11Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1546 = Coupling(
    name="GC_1546",
    value="-((CKM2x3*cquqd11Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1547 = Coupling(
    name="GC_1547",
    value="-((CKM3x3*cquqd11Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1548 = Coupling(
    name="GC_1548",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1549 = Coupling(
    name="GC_1549",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1550 = Coupling(
    name="GC_1550",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1551 = Coupling(
    name="GC_1551",
    value="(CKM1x3*cquqd1Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1552 = Coupling(
    name="GC_1552",
    value="(CKM2x3*cquqd1Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1553 = Coupling(
    name="GC_1553",
    value="(CKM3x3*cquqd1Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1554 = Coupling(
    name="GC_1554",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1555 = Coupling(
    name="GC_1555",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1556 = Coupling(
    name="GC_1556",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1557 = Coupling(
    name="GC_1557",
    value="-((CKM1x3*cquqd81Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1558 = Coupling(
    name="GC_1558",
    value="-((CKM2x3*cquqd81Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1559 = Coupling(
    name="GC_1559",
    value="-((CKM3x3*cquqd81Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1560 = Coupling(
    name="GC_1560",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1561 = Coupling(
    name="GC_1561",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1562 = Coupling(
    name="GC_1562",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1563 = Coupling(
    name="GC_1563",
    value="(CKM1x3*cquqd8Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1564 = Coupling(
    name="GC_1564",
    value="(CKM2x3*cquqd8Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1565 = Coupling(
    name="GC_1565",
    value="(CKM3x3*cquqd8Im*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1566 = Coupling(
    name="GC_1566",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1567 = Coupling(
    name="GC_1567",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1568 = Coupling(
    name="GC_1568",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1569 = Coupling(
    name="GC_1569",
    value="-((CKM1x1*cquqd11Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1570 = Coupling(
    name="GC_1570",
    value="(CKM1x1*cquqd11Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1571 = Coupling(
    name="GC_1571",
    value="(CKM1x2*cquqd11Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1572 = Coupling(
    name="GC_1572",
    value="(CKM1x3*cquqd11Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1573 = Coupling(
    name="GC_1573",
    value="-((CKM2x1*cquqd11Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1574 = Coupling(
    name="GC_1574",
    value="-((CKM3x1*cquqd11Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1575 = Coupling(
    name="GC_1575",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1576 = Coupling(
    name="GC_1576",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1577 = Coupling(
    name="GC_1577",
    value="(CKM1x3*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1578 = Coupling(
    name="GC_1578",
    value="(CKM2x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1579 = Coupling(
    name="GC_1579",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1580 = Coupling(
    name="GC_1580",
    value="-((CKM1x1*cquqd1Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1581 = Coupling(
    name="GC_1581",
    value="(CKM1x1*cquqd1Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1582 = Coupling(
    name="GC_1582",
    value="-((CKM1x2*cquqd1Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1583 = Coupling(
    name="GC_1583",
    value="-((CKM1x3*cquqd1Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1584 = Coupling(
    name="GC_1584",
    value="(CKM2x1*cquqd1Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1585 = Coupling(
    name="GC_1585",
    value="(CKM3x1*cquqd1Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1586 = Coupling(
    name="GC_1586",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1587 = Coupling(
    name="GC_1587",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1588 = Coupling(
    name="GC_1588",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1589 = Coupling(
    name="GC_1589",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1590 = Coupling(
    name="GC_1590",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1591 = Coupling(
    name="GC_1591",
    value="-((CKM1x1*cquqd81Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1592 = Coupling(
    name="GC_1592",
    value="(CKM1x1*cquqd81Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1593 = Coupling(
    name="GC_1593",
    value="(CKM1x2*cquqd81Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1594 = Coupling(
    name="GC_1594",
    value="(CKM1x3*cquqd81Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1595 = Coupling(
    name="GC_1595",
    value="-((CKM2x1*cquqd81Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1596 = Coupling(
    name="GC_1596",
    value="-((CKM3x1*cquqd81Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1597 = Coupling(
    name="GC_1597",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1598 = Coupling(
    name="GC_1598",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1599 = Coupling(
    name="GC_1599",
    value="(CKM1x3*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1600 = Coupling(
    name="GC_1600",
    value="(CKM2x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1601 = Coupling(
    name="GC_1601",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1602 = Coupling(
    name="GC_1602",
    value="-((CKM1x1*cquqd8Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1603 = Coupling(
    name="GC_1603",
    value="(CKM1x1*cquqd8Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1604 = Coupling(
    name="GC_1604",
    value="-((CKM1x2*cquqd8Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1605 = Coupling(
    name="GC_1605",
    value="-((CKM1x3*cquqd8Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1606 = Coupling(
    name="GC_1606",
    value="(CKM2x1*cquqd8Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1607 = Coupling(
    name="GC_1607",
    value="(CKM3x1*cquqd8Im*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1608 = Coupling(
    name="GC_1608",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1609 = Coupling(
    name="GC_1609",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1610 = Coupling(
    name="GC_1610",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1611 = Coupling(
    name="GC_1611",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1612 = Coupling(
    name="GC_1612",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1613 = Coupling(
    name="GC_1613",
    value="(cHudIm*ee*ydo*yup*complexconjugate(CKM1x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_1614 = Coupling(
    name="GC_1614",
    value="(cHudRe*ee*complex(0,1)*ydo*yup*complexconjugate(CKM1x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_1615 = Coupling(
    name="GC_1615",
    value="(cHudIm*ee*vevhat*ydo*yup*complexconjugate(CKM1x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1616 = Coupling(
    name="GC_1616",
    value="(cHudRe*ee*complex(0,1)*vevhat*ydo*yup*complexconjugate(CKM1x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1617 = Coupling(
    name="GC_1617",
    value="(cHudIm*ee*vevhat**2*ydo*yup*complexconjugate(CKM1x1))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1618 = Coupling(
    name="GC_1618",
    value="(cHudRe*ee*complex(0,1)*vevhat**2*ydo*yup*complexconjugate(CKM1x1))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1619 = Coupling(
    name="GC_1619",
    value="-((clequ1Im*ye*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1620 = Coupling(
    name="GC_1620",
    value="(clequ1Re*complex(0,1)*ye*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1621 = Coupling(
    name="GC_1621",
    value="-(clequ3Im*ye*yup*complexconjugate(CKM1x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1622 = Coupling(
    name="GC_1622",
    value="(clequ3Im*ye*yup*complexconjugate(CKM1x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1623 = Coupling(
    name="GC_1623",
    value="(clequ3Re*complex(0,1)*ye*yup*complexconjugate(CKM1x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1624 = Coupling(
    name="GC_1624",
    value="-(clequ3Re*complex(0,1)*ye*yup*complexconjugate(CKM1x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1625 = Coupling(
    name="GC_1625",
    value="-((clequ1Im*ym*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1626 = Coupling(
    name="GC_1626",
    value="(clequ1Re*complex(0,1)*ym*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1627 = Coupling(
    name="GC_1627",
    value="-(clequ3Im*ym*yup*complexconjugate(CKM1x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1628 = Coupling(
    name="GC_1628",
    value="(clequ3Im*ym*yup*complexconjugate(CKM1x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1629 = Coupling(
    name="GC_1629",
    value="(clequ3Re*complex(0,1)*ym*yup*complexconjugate(CKM1x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1630 = Coupling(
    name="GC_1630",
    value="-(clequ3Re*complex(0,1)*ym*yup*complexconjugate(CKM1x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1631 = Coupling(
    name="GC_1631",
    value="-((CKM1x2*cquqd11Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1632 = Coupling(
    name="GC_1632",
    value="-((CKM2x2*cquqd11Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1633 = Coupling(
    name="GC_1633",
    value="-((CKM3x2*cquqd11Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1634 = Coupling(
    name="GC_1634",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1635 = Coupling(
    name="GC_1635",
    value="(CKM2x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1636 = Coupling(
    name="GC_1636",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1637 = Coupling(
    name="GC_1637",
    value="(CKM1x2*cquqd1Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1638 = Coupling(
    name="GC_1638",
    value="(CKM2x2*cquqd1Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1639 = Coupling(
    name="GC_1639",
    value="(CKM3x2*cquqd1Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1640 = Coupling(
    name="GC_1640",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1641 = Coupling(
    name="GC_1641",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1642 = Coupling(
    name="GC_1642",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1643 = Coupling(
    name="GC_1643",
    value="-((CKM1x2*cquqd81Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1644 = Coupling(
    name="GC_1644",
    value="-((CKM2x2*cquqd81Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1645 = Coupling(
    name="GC_1645",
    value="-((CKM3x2*cquqd81Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1646 = Coupling(
    name="GC_1646",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1647 = Coupling(
    name="GC_1647",
    value="(CKM2x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1648 = Coupling(
    name="GC_1648",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1649 = Coupling(
    name="GC_1649",
    value="(CKM1x2*cquqd8Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1650 = Coupling(
    name="GC_1650",
    value="(CKM2x2*cquqd8Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1651 = Coupling(
    name="GC_1651",
    value="(CKM3x2*cquqd8Im*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1652 = Coupling(
    name="GC_1652",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1653 = Coupling(
    name="GC_1653",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1654 = Coupling(
    name="GC_1654",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1655 = Coupling(
    name="GC_1655",
    value="-((clequ1Im*ytau*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1656 = Coupling(
    name="GC_1656",
    value="(clequ1Re*complex(0,1)*ytau*yup*complexconjugate(CKM1x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1657 = Coupling(
    name="GC_1657",
    value="-(clequ3Im*ytau*yup*complexconjugate(CKM1x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1658 = Coupling(
    name="GC_1658",
    value="(clequ3Im*ytau*yup*complexconjugate(CKM1x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1659 = Coupling(
    name="GC_1659",
    value="(clequ3Re*complex(0,1)*ytau*yup*complexconjugate(CKM1x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1660 = Coupling(
    name="GC_1660",
    value="-(clequ3Re*complex(0,1)*ytau*yup*complexconjugate(CKM1x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1661 = Coupling(
    name="GC_1661",
    value="(2*clq3*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1662 = Coupling(
    name="GC_1662",
    value="(2*CKM1x1*cqq11*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1663 = Coupling(
    name="GC_1663",
    value="(2*CKM1x2*cqq11*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1664 = Coupling(
    name="GC_1664",
    value="(2*CKM1x3*cqq11*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1665 = Coupling(
    name="GC_1665",
    value="(2*CKM2x1*cqq11*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1666 = Coupling(
    name="GC_1666",
    value="(2*CKM2x2*cqq11*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1667 = Coupling(
    name="GC_1667",
    value="(2*CKM2x3*cqq11*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1668 = Coupling(
    name="GC_1668",
    value="(2*CKM3x1*cqq11*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1669 = Coupling(
    name="GC_1669",
    value="(2*CKM3x2*cqq11*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1670 = Coupling(
    name="GC_1670",
    value="(2*CKM3x3*cqq11*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1671 = Coupling(
    name="GC_1671",
    value="(4*CKM1x1*cqq3*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1672 = Coupling(
    name="GC_1672",
    value="(4*CKM1x2*cqq3*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1673 = Coupling(
    name="GC_1673",
    value="(4*CKM1x3*cqq3*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1674 = Coupling(
    name="GC_1674",
    value="(4*CKM2x1*cqq3*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1675 = Coupling(
    name="GC_1675",
    value="(4*CKM2x2*cqq3*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1676 = Coupling(
    name="GC_1676",
    value="(4*CKM2x3*cqq3*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1677 = Coupling(
    name="GC_1677",
    value="(4*CKM3x1*cqq3*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1678 = Coupling(
    name="GC_1678",
    value="(4*CKM3x2*cqq3*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1679 = Coupling(
    name="GC_1679",
    value="(4*CKM3x3*cqq3*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1680 = Coupling(
    name="GC_1680",
    value="(-2*CKM1x1*cqq31*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1681 = Coupling(
    name="GC_1681",
    value="(-2*CKM1x2*cqq31*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1682 = Coupling(
    name="GC_1682",
    value="(-2*CKM1x3*cqq31*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1683 = Coupling(
    name="GC_1683",
    value="(-2*CKM2x1*cqq31*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1684 = Coupling(
    name="GC_1684",
    value="(-2*CKM2x2*cqq31*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1685 = Coupling(
    name="GC_1685",
    value="(-2*CKM2x3*cqq31*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1686 = Coupling(
    name="GC_1686",
    value="(-2*CKM3x1*cqq31*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1687 = Coupling(
    name="GC_1687",
    value="(-2*CKM3x2*cqq31*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1688 = Coupling(
    name="GC_1688",
    value="(-2*CKM3x3*cqq31*complex(0,1)*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1689 = Coupling(
    name="GC_1689",
    value="-((ee*complex(0,1)*complexconjugate(CKM1x2))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_1690 = Coupling(
    name="GC_1690",
    value="-((cHq3*ee*complex(0,1)*complexconjugate(CKM1x2)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_1691 = Coupling(
    name="GC_1691",
    value="-((ee*complex(0,1)*propCorr*complexconjugate(CKM1x2))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_1692 = Coupling(
    name="GC_1692",
    value="-((cHq3*ee*complex(0,1)*vevhat*complexconjugate(CKM1x2)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_1693 = Coupling(
    name="GC_1693",
    value="-((cHq3*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1694 = Coupling(
    name="GC_1694",
    value="-((cHWB*cth*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x2))/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1695 = Coupling(
    name="GC_1695",
    value="-((cHl3*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x2))/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1696 = Coupling(
    name="GC_1696",
    value="(cll1*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x2))/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1697 = Coupling(
    name="GC_1697",
    value="(cHDD*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x2))/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1698 = Coupling(
    name="GC_1698",
    value="(CKM2x1*cquqd11Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1699 = Coupling(
    name="GC_1699",
    value="(CKM2x2*cquqd11Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1700 = Coupling(
    name="GC_1700",
    value="(CKM2x3*cquqd11Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1701 = Coupling(
    name="GC_1701",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1702 = Coupling(
    name="GC_1702",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1703 = Coupling(
    name="GC_1703",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1704 = Coupling(
    name="GC_1704",
    value="-((CKM2x1*cquqd1Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1705 = Coupling(
    name="GC_1705",
    value="-((CKM2x2*cquqd1Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1706 = Coupling(
    name="GC_1706",
    value="-((CKM2x3*cquqd1Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1707 = Coupling(
    name="GC_1707",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1708 = Coupling(
    name="GC_1708",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1709 = Coupling(
    name="GC_1709",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1710 = Coupling(
    name="GC_1710",
    value="(CKM2x1*cquqd81Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1711 = Coupling(
    name="GC_1711",
    value="(CKM2x2*cquqd81Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1712 = Coupling(
    name="GC_1712",
    value="(CKM2x3*cquqd81Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1713 = Coupling(
    name="GC_1713",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1714 = Coupling(
    name="GC_1714",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1715 = Coupling(
    name="GC_1715",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1716 = Coupling(
    name="GC_1716",
    value="-((CKM2x1*cquqd8Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1717 = Coupling(
    name="GC_1717",
    value="-((CKM2x2*cquqd8Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1718 = Coupling(
    name="GC_1718",
    value="-((CKM2x3*cquqd8Im*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1719 = Coupling(
    name="GC_1719",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1720 = Coupling(
    name="GC_1720",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1721 = Coupling(
    name="GC_1721",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1722 = Coupling(
    name="GC_1722",
    value="-((cledqIm*ye*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1723 = Coupling(
    name="GC_1723",
    value="(cledqRe*complex(0,1)*ye*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1724 = Coupling(
    name="GC_1724",
    value="-((cledqIm*ym*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1725 = Coupling(
    name="GC_1725",
    value="(cledqRe*complex(0,1)*ym*ys*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1726 = Coupling(
    name="GC_1726",
    value="(CKM3x1*cquqd11Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1727 = Coupling(
    name="GC_1727",
    value="(CKM3x2*cquqd11Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1728 = Coupling(
    name="GC_1728",
    value="(CKM3x3*cquqd11Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1729 = Coupling(
    name="GC_1729",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1730 = Coupling(
    name="GC_1730",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1731 = Coupling(
    name="GC_1731",
    value="(CKM3x3*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1732 = Coupling(
    name="GC_1732",
    value="-((CKM3x1*cquqd1Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1733 = Coupling(
    name="GC_1733",
    value="-((CKM3x2*cquqd1Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1734 = Coupling(
    name="GC_1734",
    value="-((CKM3x3*cquqd1Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1735 = Coupling(
    name="GC_1735",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1736 = Coupling(
    name="GC_1736",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1737 = Coupling(
    name="GC_1737",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1738 = Coupling(
    name="GC_1738",
    value="(CKM3x1*cquqd81Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1739 = Coupling(
    name="GC_1739",
    value="(CKM3x2*cquqd81Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1740 = Coupling(
    name="GC_1740",
    value="(CKM3x3*cquqd81Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1741 = Coupling(
    name="GC_1741",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1742 = Coupling(
    name="GC_1742",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1743 = Coupling(
    name="GC_1743",
    value="(CKM3x3*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1744 = Coupling(
    name="GC_1744",
    value="-((CKM3x1*cquqd8Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1745 = Coupling(
    name="GC_1745",
    value="-((CKM3x2*cquqd8Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1746 = Coupling(
    name="GC_1746",
    value="-((CKM3x3*cquqd8Im*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1747 = Coupling(
    name="GC_1747",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1748 = Coupling(
    name="GC_1748",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1749 = Coupling(
    name="GC_1749",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1750 = Coupling(
    name="GC_1750",
    value="-((cledqIm*ys*ytau*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1751 = Coupling(
    name="GC_1751",
    value="(cledqRe*complex(0,1)*ys*ytau*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1752 = Coupling(
    name="GC_1752",
    value="-((CKM1x3*cquqd11Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1753 = Coupling(
    name="GC_1753",
    value="-((CKM2x3*cquqd11Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1754 = Coupling(
    name="GC_1754",
    value="-((CKM3x3*cquqd11Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1755 = Coupling(
    name="GC_1755",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1756 = Coupling(
    name="GC_1756",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1757 = Coupling(
    name="GC_1757",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1758 = Coupling(
    name="GC_1758",
    value="(CKM1x3*cquqd1Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1759 = Coupling(
    name="GC_1759",
    value="(CKM2x3*cquqd1Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1760 = Coupling(
    name="GC_1760",
    value="(CKM3x3*cquqd1Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1761 = Coupling(
    name="GC_1761",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1762 = Coupling(
    name="GC_1762",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1763 = Coupling(
    name="GC_1763",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1764 = Coupling(
    name="GC_1764",
    value="-((CKM1x3*cquqd81Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1765 = Coupling(
    name="GC_1765",
    value="-((CKM2x3*cquqd81Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1766 = Coupling(
    name="GC_1766",
    value="-((CKM3x3*cquqd81Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1767 = Coupling(
    name="GC_1767",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1768 = Coupling(
    name="GC_1768",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1769 = Coupling(
    name="GC_1769",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1770 = Coupling(
    name="GC_1770",
    value="(CKM1x3*cquqd8Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1771 = Coupling(
    name="GC_1771",
    value="(CKM2x3*cquqd8Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1772 = Coupling(
    name="GC_1772",
    value="(CKM3x3*cquqd8Im*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1773 = Coupling(
    name="GC_1773",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1774 = Coupling(
    name="GC_1774",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1775 = Coupling(
    name="GC_1775",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1776 = Coupling(
    name="GC_1776",
    value="-((CKM1x1*cquqd11Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1777 = Coupling(
    name="GC_1777",
    value="-((CKM2x1*cquqd11Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1778 = Coupling(
    name="GC_1778",
    value="-((CKM3x1*cquqd11Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1779 = Coupling(
    name="GC_1779",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1780 = Coupling(
    name="GC_1780",
    value="(CKM2x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1781 = Coupling(
    name="GC_1781",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1782 = Coupling(
    name="GC_1782",
    value="(CKM1x1*cquqd1Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1783 = Coupling(
    name="GC_1783",
    value="(CKM2x1*cquqd1Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1784 = Coupling(
    name="GC_1784",
    value="(CKM3x1*cquqd1Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1785 = Coupling(
    name="GC_1785",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1786 = Coupling(
    name="GC_1786",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1787 = Coupling(
    name="GC_1787",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1788 = Coupling(
    name="GC_1788",
    value="-((CKM1x1*cquqd81Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1789 = Coupling(
    name="GC_1789",
    value="-((CKM2x1*cquqd81Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1790 = Coupling(
    name="GC_1790",
    value="-((CKM3x1*cquqd81Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1791 = Coupling(
    name="GC_1791",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1792 = Coupling(
    name="GC_1792",
    value="(CKM2x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1793 = Coupling(
    name="GC_1793",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1794 = Coupling(
    name="GC_1794",
    value="(CKM1x1*cquqd8Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1795 = Coupling(
    name="GC_1795",
    value="(CKM2x1*cquqd8Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1796 = Coupling(
    name="GC_1796",
    value="(CKM3x1*cquqd8Im*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1797 = Coupling(
    name="GC_1797",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1798 = Coupling(
    name="GC_1798",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1799 = Coupling(
    name="GC_1799",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1800 = Coupling(
    name="GC_1800",
    value="-((clequ1Im*ye*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1801 = Coupling(
    name="GC_1801",
    value="(clequ1Re*complex(0,1)*ye*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1802 = Coupling(
    name="GC_1802",
    value="-(clequ3Im*ye*yup*complexconjugate(CKM1x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1803 = Coupling(
    name="GC_1803",
    value="(clequ3Im*ye*yup*complexconjugate(CKM1x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1804 = Coupling(
    name="GC_1804",
    value="(clequ3Re*complex(0,1)*ye*yup*complexconjugate(CKM1x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1805 = Coupling(
    name="GC_1805",
    value="-(clequ3Re*complex(0,1)*ye*yup*complexconjugate(CKM1x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1806 = Coupling(
    name="GC_1806",
    value="-((clequ1Im*ym*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1807 = Coupling(
    name="GC_1807",
    value="(clequ1Re*complex(0,1)*ym*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1808 = Coupling(
    name="GC_1808",
    value="-(clequ3Im*ym*yup*complexconjugate(CKM1x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1809 = Coupling(
    name="GC_1809",
    value="(clequ3Im*ym*yup*complexconjugate(CKM1x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1810 = Coupling(
    name="GC_1810",
    value="(clequ3Re*complex(0,1)*ym*yup*complexconjugate(CKM1x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1811 = Coupling(
    name="GC_1811",
    value="-(clequ3Re*complex(0,1)*ym*yup*complexconjugate(CKM1x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1812 = Coupling(
    name="GC_1812",
    value="(CKM1x1*cquqd11Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1813 = Coupling(
    name="GC_1813",
    value="-((CKM1x2*cquqd11Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1814 = Coupling(
    name="GC_1814",
    value="(CKM1x2*cquqd11Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1815 = Coupling(
    name="GC_1815",
    value="(CKM1x3*cquqd11Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1816 = Coupling(
    name="GC_1816",
    value="-((CKM2x2*cquqd11Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1817 = Coupling(
    name="GC_1817",
    value="-((CKM3x2*cquqd11Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1818 = Coupling(
    name="GC_1818",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1819 = Coupling(
    name="GC_1819",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1820 = Coupling(
    name="GC_1820",
    value="(CKM1x3*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1821 = Coupling(
    name="GC_1821",
    value="(CKM2x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1822 = Coupling(
    name="GC_1822",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1823 = Coupling(
    name="GC_1823",
    value="-((CKM1x1*cquqd1Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1824 = Coupling(
    name="GC_1824",
    value="-((CKM1x2*cquqd1Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1825 = Coupling(
    name="GC_1825",
    value="(CKM1x2*cquqd1Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1826 = Coupling(
    name="GC_1826",
    value="-((CKM1x3*cquqd1Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1827 = Coupling(
    name="GC_1827",
    value="(CKM2x2*cquqd1Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1828 = Coupling(
    name="GC_1828",
    value="(CKM3x2*cquqd1Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1829 = Coupling(
    name="GC_1829",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1830 = Coupling(
    name="GC_1830",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1831 = Coupling(
    name="GC_1831",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1832 = Coupling(
    name="GC_1832",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1833 = Coupling(
    name="GC_1833",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1834 = Coupling(
    name="GC_1834",
    value="(CKM1x1*cquqd81Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1835 = Coupling(
    name="GC_1835",
    value="-((CKM1x2*cquqd81Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1836 = Coupling(
    name="GC_1836",
    value="(CKM1x2*cquqd81Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1837 = Coupling(
    name="GC_1837",
    value="(CKM1x3*cquqd81Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1838 = Coupling(
    name="GC_1838",
    value="-((CKM2x2*cquqd81Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1839 = Coupling(
    name="GC_1839",
    value="-((CKM3x2*cquqd81Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1840 = Coupling(
    name="GC_1840",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1841 = Coupling(
    name="GC_1841",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1842 = Coupling(
    name="GC_1842",
    value="(CKM1x3*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1843 = Coupling(
    name="GC_1843",
    value="(CKM2x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1844 = Coupling(
    name="GC_1844",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1845 = Coupling(
    name="GC_1845",
    value="-((CKM1x1*cquqd8Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1846 = Coupling(
    name="GC_1846",
    value="-((CKM1x2*cquqd8Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1847 = Coupling(
    name="GC_1847",
    value="(CKM1x2*cquqd8Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1848 = Coupling(
    name="GC_1848",
    value="-((CKM1x3*cquqd8Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1849 = Coupling(
    name="GC_1849",
    value="(CKM2x2*cquqd8Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1850 = Coupling(
    name="GC_1850",
    value="(CKM3x2*cquqd8Im*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1851 = Coupling(
    name="GC_1851",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1852 = Coupling(
    name="GC_1852",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1853 = Coupling(
    name="GC_1853",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1854 = Coupling(
    name="GC_1854",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1855 = Coupling(
    name="GC_1855",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1856 = Coupling(
    name="GC_1856",
    value="(cHudIm*ee*ys*yup*complexconjugate(CKM1x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_1857 = Coupling(
    name="GC_1857",
    value="(cHudRe*ee*complex(0,1)*ys*yup*complexconjugate(CKM1x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_1858 = Coupling(
    name="GC_1858",
    value="(cHudIm*ee*vevhat*ys*yup*complexconjugate(CKM1x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1859 = Coupling(
    name="GC_1859",
    value="(cHudRe*ee*complex(0,1)*vevhat*ys*yup*complexconjugate(CKM1x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_1860 = Coupling(
    name="GC_1860",
    value="(cHudIm*ee*vevhat**2*ys*yup*complexconjugate(CKM1x2))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1861 = Coupling(
    name="GC_1861",
    value="(cHudRe*ee*complex(0,1)*vevhat**2*ys*yup*complexconjugate(CKM1x2))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_1862 = Coupling(
    name="GC_1862",
    value="-((clequ1Im*ytau*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1863 = Coupling(
    name="GC_1863",
    value="(clequ1Re*complex(0,1)*ytau*yup*complexconjugate(CKM1x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1864 = Coupling(
    name="GC_1864",
    value="-(clequ3Im*ytau*yup*complexconjugate(CKM1x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1865 = Coupling(
    name="GC_1865",
    value="(clequ3Im*ytau*yup*complexconjugate(CKM1x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1866 = Coupling(
    name="GC_1866",
    value="(clequ3Re*complex(0,1)*ytau*yup*complexconjugate(CKM1x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1867 = Coupling(
    name="GC_1867",
    value="-(clequ3Re*complex(0,1)*ytau*yup*complexconjugate(CKM1x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1868 = Coupling(
    name="GC_1868",
    value="(2*clq3*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1869 = Coupling(
    name="GC_1869",
    value="(2*CKM1x1*cqq11*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1870 = Coupling(
    name="GC_1870",
    value="(2*CKM1x2*cqq11*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1871 = Coupling(
    name="GC_1871",
    value="(2*CKM1x3*cqq11*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1872 = Coupling(
    name="GC_1872",
    value="(2*CKM2x1*cqq11*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1873 = Coupling(
    name="GC_1873",
    value="(2*CKM2x2*cqq11*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1874 = Coupling(
    name="GC_1874",
    value="(2*CKM2x3*cqq11*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1875 = Coupling(
    name="GC_1875",
    value="(2*CKM3x1*cqq11*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1876 = Coupling(
    name="GC_1876",
    value="(2*CKM3x2*cqq11*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1877 = Coupling(
    name="GC_1877",
    value="(2*CKM3x3*cqq11*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1878 = Coupling(
    name="GC_1878",
    value="(4*CKM1x1*cqq3*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1879 = Coupling(
    name="GC_1879",
    value="(4*CKM1x2*cqq3*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1880 = Coupling(
    name="GC_1880",
    value="(4*CKM1x3*cqq3*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1881 = Coupling(
    name="GC_1881",
    value="(4*CKM2x1*cqq3*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1882 = Coupling(
    name="GC_1882",
    value="(4*CKM2x2*cqq3*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1883 = Coupling(
    name="GC_1883",
    value="(4*CKM2x3*cqq3*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1884 = Coupling(
    name="GC_1884",
    value="(4*CKM3x1*cqq3*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1885 = Coupling(
    name="GC_1885",
    value="(4*CKM3x2*cqq3*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1886 = Coupling(
    name="GC_1886",
    value="(4*CKM3x3*cqq3*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1887 = Coupling(
    name="GC_1887",
    value="(-2*CKM1x1*cqq31*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1888 = Coupling(
    name="GC_1888",
    value="(-2*CKM1x2*cqq31*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1889 = Coupling(
    name="GC_1889",
    value="(-2*CKM1x3*cqq31*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1890 = Coupling(
    name="GC_1890",
    value="(-2*CKM2x1*cqq31*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1891 = Coupling(
    name="GC_1891",
    value="(-2*CKM2x2*cqq31*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1892 = Coupling(
    name="GC_1892",
    value="(-2*CKM2x3*cqq31*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1893 = Coupling(
    name="GC_1893",
    value="(-2*CKM3x1*cqq31*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1894 = Coupling(
    name="GC_1894",
    value="(-2*CKM3x2*cqq31*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1895 = Coupling(
    name="GC_1895",
    value="(-2*CKM3x3*cqq31*complex(0,1)*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_1896 = Coupling(
    name="GC_1896",
    value="-((ee*complex(0,1)*complexconjugate(CKM1x3))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_1897 = Coupling(
    name="GC_1897",
    value="-((cHq3*ee*complex(0,1)*complexconjugate(CKM1x3)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_1898 = Coupling(
    name="GC_1898",
    value="-((ee*complex(0,1)*propCorr*complexconjugate(CKM1x3))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_1899 = Coupling(
    name="GC_1899",
    value="-((cHq3*ee*complex(0,1)*vevhat*complexconjugate(CKM1x3)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_1900 = Coupling(
    name="GC_1900",
    value="-((cHq3*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1901 = Coupling(
    name="GC_1901",
    value="-((cHWB*cth*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x3))/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1902 = Coupling(
    name="GC_1902",
    value="-((cHl3*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x3))/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_1903 = Coupling(
    name="GC_1903",
    value="(cll1*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x3))/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1904 = Coupling(
    name="GC_1904",
    value="(cHDD*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM1x3))/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_1905 = Coupling(
    name="GC_1905",
    value="(CKM2x1*cquqd11Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1906 = Coupling(
    name="GC_1906",
    value="(CKM2x2*cquqd11Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1907 = Coupling(
    name="GC_1907",
    value="(CKM2x3*cquqd11Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1908 = Coupling(
    name="GC_1908",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1909 = Coupling(
    name="GC_1909",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1910 = Coupling(
    name="GC_1910",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1911 = Coupling(
    name="GC_1911",
    value="-((CKM2x1*cquqd1Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1912 = Coupling(
    name="GC_1912",
    value="-((CKM2x2*cquqd1Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1913 = Coupling(
    name="GC_1913",
    value="-((CKM2x3*cquqd1Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1914 = Coupling(
    name="GC_1914",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1915 = Coupling(
    name="GC_1915",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1916 = Coupling(
    name="GC_1916",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1917 = Coupling(
    name="GC_1917",
    value="(CKM2x1*cquqd81Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1918 = Coupling(
    name="GC_1918",
    value="(CKM2x2*cquqd81Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1919 = Coupling(
    name="GC_1919",
    value="(CKM2x3*cquqd81Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1920 = Coupling(
    name="GC_1920",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1921 = Coupling(
    name="GC_1921",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1922 = Coupling(
    name="GC_1922",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1923 = Coupling(
    name="GC_1923",
    value="-((CKM2x1*cquqd8Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1924 = Coupling(
    name="GC_1924",
    value="-((CKM2x2*cquqd8Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1925 = Coupling(
    name="GC_1925",
    value="-((CKM2x3*cquqd8Im*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1926 = Coupling(
    name="GC_1926",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1927 = Coupling(
    name="GC_1927",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1928 = Coupling(
    name="GC_1928",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1929 = Coupling(
    name="GC_1929",
    value="-((cledqIm*yb*ye*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1930 = Coupling(
    name="GC_1930",
    value="(cledqRe*complex(0,1)*yb*ye*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1931 = Coupling(
    name="GC_1931",
    value="-((cledqIm*yb*ym*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1932 = Coupling(
    name="GC_1932",
    value="(cledqRe*complex(0,1)*yb*ym*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1933 = Coupling(
    name="GC_1933",
    value="(CKM3x1*cquqd11Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1934 = Coupling(
    name="GC_1934",
    value="(CKM3x2*cquqd11Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1935 = Coupling(
    name="GC_1935",
    value="(CKM3x3*cquqd11Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1936 = Coupling(
    name="GC_1936",
    value="(CKM3x1*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1937 = Coupling(
    name="GC_1937",
    value="(CKM3x2*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1938 = Coupling(
    name="GC_1938",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1939 = Coupling(
    name="GC_1939",
    value="-((CKM3x1*cquqd1Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1940 = Coupling(
    name="GC_1940",
    value="-((CKM3x2*cquqd1Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1941 = Coupling(
    name="GC_1941",
    value="-((CKM3x3*cquqd1Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1942 = Coupling(
    name="GC_1942",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1943 = Coupling(
    name="GC_1943",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1944 = Coupling(
    name="GC_1944",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1945 = Coupling(
    name="GC_1945",
    value="(CKM3x1*cquqd81Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1946 = Coupling(
    name="GC_1946",
    value="(CKM3x2*cquqd81Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1947 = Coupling(
    name="GC_1947",
    value="(CKM3x3*cquqd81Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1948 = Coupling(
    name="GC_1948",
    value="(CKM3x1*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1949 = Coupling(
    name="GC_1949",
    value="(CKM3x2*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1950 = Coupling(
    name="GC_1950",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1951 = Coupling(
    name="GC_1951",
    value="-((CKM3x1*cquqd8Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1952 = Coupling(
    name="GC_1952",
    value="-((CKM3x2*cquqd8Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1953 = Coupling(
    name="GC_1953",
    value="-((CKM3x3*cquqd8Im*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1954 = Coupling(
    name="GC_1954",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1955 = Coupling(
    name="GC_1955",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1956 = Coupling(
    name="GC_1956",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1957 = Coupling(
    name="GC_1957",
    value="-((cledqIm*yb*ytau*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1958 = Coupling(
    name="GC_1958",
    value="(cledqRe*complex(0,1)*yb*ytau*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1959 = Coupling(
    name="GC_1959",
    value="(CKM1x1*cquqd11Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1960 = Coupling(
    name="GC_1960",
    value="(CKM1x2*cquqd11Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1961 = Coupling(
    name="GC_1961",
    value="-((CKM1x3*cquqd11Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1962 = Coupling(
    name="GC_1962",
    value="(CKM1x3*cquqd11Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1963 = Coupling(
    name="GC_1963",
    value="-((CKM2x3*cquqd11Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1964 = Coupling(
    name="GC_1964",
    value="-((CKM3x3*cquqd11Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1965 = Coupling(
    name="GC_1965",
    value="(CKM1x1*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1966 = Coupling(
    name="GC_1966",
    value="(CKM1x2*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1967 = Coupling(
    name="GC_1967",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1968 = Coupling(
    name="GC_1968",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1969 = Coupling(
    name="GC_1969",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1970 = Coupling(
    name="GC_1970",
    value="-((CKM1x1*cquqd1Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1971 = Coupling(
    name="GC_1971",
    value="-((CKM1x2*cquqd1Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1972 = Coupling(
    name="GC_1972",
    value="-((CKM1x3*cquqd1Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1973 = Coupling(
    name="GC_1973",
    value="(CKM1x3*cquqd1Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1974 = Coupling(
    name="GC_1974",
    value="(CKM2x3*cquqd1Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1975 = Coupling(
    name="GC_1975",
    value="(CKM3x3*cquqd1Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1976 = Coupling(
    name="GC_1976",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1977 = Coupling(
    name="GC_1977",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1978 = Coupling(
    name="GC_1978",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1979 = Coupling(
    name="GC_1979",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1980 = Coupling(
    name="GC_1980",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1981 = Coupling(
    name="GC_1981",
    value="(CKM1x1*cquqd81Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1982 = Coupling(
    name="GC_1982",
    value="(CKM1x2*cquqd81Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1983 = Coupling(
    name="GC_1983",
    value="-((CKM1x3*cquqd81Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1984 = Coupling(
    name="GC_1984",
    value="(CKM1x3*cquqd81Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1985 = Coupling(
    name="GC_1985",
    value="-((CKM2x3*cquqd81Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1986 = Coupling(
    name="GC_1986",
    value="-((CKM3x3*cquqd81Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1987 = Coupling(
    name="GC_1987",
    value="(CKM1x1*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1988 = Coupling(
    name="GC_1988",
    value="(CKM1x2*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1989 = Coupling(
    name="GC_1989",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1990 = Coupling(
    name="GC_1990",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1991 = Coupling(
    name="GC_1991",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1992 = Coupling(
    name="GC_1992",
    value="-((CKM1x1*cquqd8Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1993 = Coupling(
    name="GC_1993",
    value="-((CKM1x2*cquqd8Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1994 = Coupling(
    name="GC_1994",
    value="-((CKM1x3*cquqd8Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1995 = Coupling(
    name="GC_1995",
    value="(CKM1x3*cquqd8Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1996 = Coupling(
    name="GC_1996",
    value="(CKM2x3*cquqd8Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1997 = Coupling(
    name="GC_1997",
    value="(CKM3x3*cquqd8Im*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_1998 = Coupling(
    name="GC_1998",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_1999 = Coupling(
    name="GC_1999",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2000 = Coupling(
    name="GC_2000",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2001 = Coupling(
    name="GC_2001",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2002 = Coupling(
    name="GC_2002",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2003 = Coupling(
    name="GC_2003",
    value="(cHudIm*ee*yb*yup*complexconjugate(CKM1x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_2004 = Coupling(
    name="GC_2004",
    value="(cHudRe*ee*complex(0,1)*yb*yup*complexconjugate(CKM1x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_2005 = Coupling(
    name="GC_2005",
    value="(cHudIm*ee*vevhat*yb*yup*complexconjugate(CKM1x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_2006 = Coupling(
    name="GC_2006",
    value="(cHudRe*ee*complex(0,1)*vevhat*yb*yup*complexconjugate(CKM1x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_2007 = Coupling(
    name="GC_2007",
    value="(cHudIm*ee*vevhat**2*yb*yup*complexconjugate(CKM1x3))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_2008 = Coupling(
    name="GC_2008",
    value="(cHudRe*ee*complex(0,1)*vevhat**2*yb*yup*complexconjugate(CKM1x3))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_2009 = Coupling(
    name="GC_2009",
    value="-((CKM1x1*cquqd11Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2010 = Coupling(
    name="GC_2010",
    value="-((CKM2x1*cquqd11Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2011 = Coupling(
    name="GC_2011",
    value="-((CKM3x1*cquqd11Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2012 = Coupling(
    name="GC_2012",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2013 = Coupling(
    name="GC_2013",
    value="(CKM2x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2014 = Coupling(
    name="GC_2014",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2015 = Coupling(
    name="GC_2015",
    value="(CKM1x1*cquqd1Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2016 = Coupling(
    name="GC_2016",
    value="(CKM2x1*cquqd1Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2017 = Coupling(
    name="GC_2017",
    value="(CKM3x1*cquqd1Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2018 = Coupling(
    name="GC_2018",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2019 = Coupling(
    name="GC_2019",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2020 = Coupling(
    name="GC_2020",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2021 = Coupling(
    name="GC_2021",
    value="-((CKM1x1*cquqd81Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2022 = Coupling(
    name="GC_2022",
    value="-((CKM2x1*cquqd81Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2023 = Coupling(
    name="GC_2023",
    value="-((CKM3x1*cquqd81Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2024 = Coupling(
    name="GC_2024",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2025 = Coupling(
    name="GC_2025",
    value="(CKM2x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2026 = Coupling(
    name="GC_2026",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2027 = Coupling(
    name="GC_2027",
    value="(CKM1x1*cquqd8Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2028 = Coupling(
    name="GC_2028",
    value="(CKM2x1*cquqd8Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2029 = Coupling(
    name="GC_2029",
    value="(CKM3x1*cquqd8Im*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2030 = Coupling(
    name="GC_2030",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2031 = Coupling(
    name="GC_2031",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2032 = Coupling(
    name="GC_2032",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2033 = Coupling(
    name="GC_2033",
    value="-((clequ1Im*ye*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2034 = Coupling(
    name="GC_2034",
    value="(clequ1Re*complex(0,1)*ye*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2035 = Coupling(
    name="GC_2035",
    value="-(clequ3Im*ye*yup*complexconjugate(CKM1x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2036 = Coupling(
    name="GC_2036",
    value="(clequ3Im*ye*yup*complexconjugate(CKM1x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2037 = Coupling(
    name="GC_2037",
    value="(clequ3Re*complex(0,1)*ye*yup*complexconjugate(CKM1x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2038 = Coupling(
    name="GC_2038",
    value="-(clequ3Re*complex(0,1)*ye*yup*complexconjugate(CKM1x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2039 = Coupling(
    name="GC_2039",
    value="-((clequ1Im*ym*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2040 = Coupling(
    name="GC_2040",
    value="(clequ1Re*complex(0,1)*ym*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2041 = Coupling(
    name="GC_2041",
    value="-(clequ3Im*ym*yup*complexconjugate(CKM1x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2042 = Coupling(
    name="GC_2042",
    value="(clequ3Im*ym*yup*complexconjugate(CKM1x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2043 = Coupling(
    name="GC_2043",
    value="(clequ3Re*complex(0,1)*ym*yup*complexconjugate(CKM1x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2044 = Coupling(
    name="GC_2044",
    value="-(clequ3Re*complex(0,1)*ym*yup*complexconjugate(CKM1x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2045 = Coupling(
    name="GC_2045",
    value="-((CKM1x2*cquqd11Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2046 = Coupling(
    name="GC_2046",
    value="-((CKM2x2*cquqd11Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2047 = Coupling(
    name="GC_2047",
    value="-((CKM3x2*cquqd11Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2048 = Coupling(
    name="GC_2048",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2049 = Coupling(
    name="GC_2049",
    value="(CKM2x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2050 = Coupling(
    name="GC_2050",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2051 = Coupling(
    name="GC_2051",
    value="(CKM1x2*cquqd1Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2052 = Coupling(
    name="GC_2052",
    value="(CKM2x2*cquqd1Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2053 = Coupling(
    name="GC_2053",
    value="(CKM3x2*cquqd1Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2054 = Coupling(
    name="GC_2054",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2055 = Coupling(
    name="GC_2055",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2056 = Coupling(
    name="GC_2056",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2057 = Coupling(
    name="GC_2057",
    value="-((CKM1x2*cquqd81Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2058 = Coupling(
    name="GC_2058",
    value="-((CKM2x2*cquqd81Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2059 = Coupling(
    name="GC_2059",
    value="-((CKM3x2*cquqd81Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2060 = Coupling(
    name="GC_2060",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2061 = Coupling(
    name="GC_2061",
    value="(CKM2x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2062 = Coupling(
    name="GC_2062",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2063 = Coupling(
    name="GC_2063",
    value="(CKM1x2*cquqd8Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2064 = Coupling(
    name="GC_2064",
    value="(CKM2x2*cquqd8Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2065 = Coupling(
    name="GC_2065",
    value="(CKM3x2*cquqd8Im*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2066 = Coupling(
    name="GC_2066",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2067 = Coupling(
    name="GC_2067",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2068 = Coupling(
    name="GC_2068",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2069 = Coupling(
    name="GC_2069",
    value="-((clequ1Im*ytau*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2070 = Coupling(
    name="GC_2070",
    value="(clequ1Re*complex(0,1)*ytau*yup*complexconjugate(CKM1x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2071 = Coupling(
    name="GC_2071",
    value="-(clequ3Im*ytau*yup*complexconjugate(CKM1x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2072 = Coupling(
    name="GC_2072",
    value="(clequ3Im*ytau*yup*complexconjugate(CKM1x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2073 = Coupling(
    name="GC_2073",
    value="(clequ3Re*complex(0,1)*ytau*yup*complexconjugate(CKM1x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2074 = Coupling(
    name="GC_2074",
    value="-(clequ3Re*complex(0,1)*ytau*yup*complexconjugate(CKM1x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2075 = Coupling(
    name="GC_2075",
    value="(2*clq3*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2076 = Coupling(
    name="GC_2076",
    value="(2*CKM1x1*cqq11*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2077 = Coupling(
    name="GC_2077",
    value="(2*CKM1x2*cqq11*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2078 = Coupling(
    name="GC_2078",
    value="(2*CKM1x3*cqq11*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2079 = Coupling(
    name="GC_2079",
    value="(2*CKM2x1*cqq11*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2080 = Coupling(
    name="GC_2080",
    value="(2*CKM2x2*cqq11*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2081 = Coupling(
    name="GC_2081",
    value="(2*CKM2x3*cqq11*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2082 = Coupling(
    name="GC_2082",
    value="(2*CKM3x1*cqq11*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2083 = Coupling(
    name="GC_2083",
    value="(2*CKM3x2*cqq11*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2084 = Coupling(
    name="GC_2084",
    value="(2*CKM3x3*cqq11*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2085 = Coupling(
    name="GC_2085",
    value="(4*CKM1x1*cqq3*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2086 = Coupling(
    name="GC_2086",
    value="(4*CKM1x2*cqq3*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2087 = Coupling(
    name="GC_2087",
    value="(4*CKM1x3*cqq3*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2088 = Coupling(
    name="GC_2088",
    value="(4*CKM2x1*cqq3*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2089 = Coupling(
    name="GC_2089",
    value="(4*CKM2x2*cqq3*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2090 = Coupling(
    name="GC_2090",
    value="(4*CKM2x3*cqq3*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2091 = Coupling(
    name="GC_2091",
    value="(4*CKM3x1*cqq3*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2092 = Coupling(
    name="GC_2092",
    value="(4*CKM3x2*cqq3*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2093 = Coupling(
    name="GC_2093",
    value="(4*CKM3x3*cqq3*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2094 = Coupling(
    name="GC_2094",
    value="(-2*CKM1x1*cqq31*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2095 = Coupling(
    name="GC_2095",
    value="(-2*CKM1x2*cqq31*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2096 = Coupling(
    name="GC_2096",
    value="(-2*CKM1x3*cqq31*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2097 = Coupling(
    name="GC_2097",
    value="(-2*CKM2x1*cqq31*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2098 = Coupling(
    name="GC_2098",
    value="(-2*CKM2x2*cqq31*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2099 = Coupling(
    name="GC_2099",
    value="(-2*CKM2x3*cqq31*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2100 = Coupling(
    name="GC_2100",
    value="(-2*CKM3x1*cqq31*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2101 = Coupling(
    name="GC_2101",
    value="(-2*CKM3x2*cqq31*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2102 = Coupling(
    name="GC_2102",
    value="(-2*CKM3x3*cqq31*complex(0,1)*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2103 = Coupling(
    name="GC_2103",
    value="-((ee*complex(0,1)*complexconjugate(CKM2x1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2104 = Coupling(
    name="GC_2104",
    value="-((cHq3*ee*complex(0,1)*complexconjugate(CKM2x1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_2105 = Coupling(
    name="GC_2105",
    value="-((ee*complex(0,1)*propCorr*complexconjugate(CKM2x1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2106 = Coupling(
    name="GC_2106",
    value="-((cHq3*ee*complex(0,1)*vevhat*complexconjugate(CKM2x1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_2107 = Coupling(
    name="GC_2107",
    value="-((cHq3*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2108 = Coupling(
    name="GC_2108",
    value="-((cHWB*cth*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x1))/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2109 = Coupling(
    name="GC_2109",
    value="-((cHl3*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x1))/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2110 = Coupling(
    name="GC_2110",
    value="(cll1*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x1))/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_2111 = Coupling(
    name="GC_2111",
    value="(cHDD*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x1))/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_2112 = Coupling(
    name="GC_2112",
    value="-((CKM1x3*cquqd11Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2113 = Coupling(
    name="GC_2113",
    value="-((CKM2x3*cquqd11Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2114 = Coupling(
    name="GC_2114",
    value="-((CKM3x3*cquqd11Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2115 = Coupling(
    name="GC_2115",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2116 = Coupling(
    name="GC_2116",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2117 = Coupling(
    name="GC_2117",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2118 = Coupling(
    name="GC_2118",
    value="(CKM1x3*cquqd1Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2119 = Coupling(
    name="GC_2119",
    value="(CKM2x3*cquqd1Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2120 = Coupling(
    name="GC_2120",
    value="(CKM3x3*cquqd1Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2121 = Coupling(
    name="GC_2121",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2122 = Coupling(
    name="GC_2122",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2123 = Coupling(
    name="GC_2123",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2124 = Coupling(
    name="GC_2124",
    value="-((CKM1x3*cquqd81Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2125 = Coupling(
    name="GC_2125",
    value="-((CKM2x3*cquqd81Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2126 = Coupling(
    name="GC_2126",
    value="-((CKM3x3*cquqd81Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2127 = Coupling(
    name="GC_2127",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2128 = Coupling(
    name="GC_2128",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2129 = Coupling(
    name="GC_2129",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2130 = Coupling(
    name="GC_2130",
    value="(CKM1x3*cquqd8Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2131 = Coupling(
    name="GC_2131",
    value="(CKM2x3*cquqd8Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2132 = Coupling(
    name="GC_2132",
    value="(CKM3x3*cquqd8Im*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2133 = Coupling(
    name="GC_2133",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2134 = Coupling(
    name="GC_2134",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2135 = Coupling(
    name="GC_2135",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2136 = Coupling(
    name="GC_2136",
    value="-((CKM1x1*cquqd11Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2137 = Coupling(
    name="GC_2137",
    value="-((CKM2x1*cquqd11Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2138 = Coupling(
    name="GC_2138",
    value="(CKM2x1*cquqd11Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2139 = Coupling(
    name="GC_2139",
    value="(CKM2x2*cquqd11Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2140 = Coupling(
    name="GC_2140",
    value="(CKM2x3*cquqd11Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2141 = Coupling(
    name="GC_2141",
    value="-((CKM3x1*cquqd11Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2142 = Coupling(
    name="GC_2142",
    value="(CKM1x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2143 = Coupling(
    name="GC_2143",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2144 = Coupling(
    name="GC_2144",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2145 = Coupling(
    name="GC_2145",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2146 = Coupling(
    name="GC_2146",
    value="(CKM3x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2147 = Coupling(
    name="GC_2147",
    value="(CKM1x1*cquqd1Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2148 = Coupling(
    name="GC_2148",
    value="-((CKM2x1*cquqd1Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2149 = Coupling(
    name="GC_2149",
    value="(CKM2x1*cquqd1Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2150 = Coupling(
    name="GC_2150",
    value="-((CKM2x2*cquqd1Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2151 = Coupling(
    name="GC_2151",
    value="-((CKM2x3*cquqd1Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2152 = Coupling(
    name="GC_2152",
    value="(CKM3x1*cquqd1Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2153 = Coupling(
    name="GC_2153",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2154 = Coupling(
    name="GC_2154",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2155 = Coupling(
    name="GC_2155",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2156 = Coupling(
    name="GC_2156",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2157 = Coupling(
    name="GC_2157",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2158 = Coupling(
    name="GC_2158",
    value="-((CKM1x1*cquqd81Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2159 = Coupling(
    name="GC_2159",
    value="-((CKM2x1*cquqd81Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2160 = Coupling(
    name="GC_2160",
    value="(CKM2x1*cquqd81Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2161 = Coupling(
    name="GC_2161",
    value="(CKM2x2*cquqd81Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2162 = Coupling(
    name="GC_2162",
    value="(CKM2x3*cquqd81Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2163 = Coupling(
    name="GC_2163",
    value="-((CKM3x1*cquqd81Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2164 = Coupling(
    name="GC_2164",
    value="(CKM1x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2165 = Coupling(
    name="GC_2165",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2166 = Coupling(
    name="GC_2166",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2167 = Coupling(
    name="GC_2167",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2168 = Coupling(
    name="GC_2168",
    value="(CKM3x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2169 = Coupling(
    name="GC_2169",
    value="(CKM1x1*cquqd8Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2170 = Coupling(
    name="GC_2170",
    value="-((CKM2x1*cquqd8Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2171 = Coupling(
    name="GC_2171",
    value="(CKM2x1*cquqd8Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2172 = Coupling(
    name="GC_2172",
    value="-((CKM2x2*cquqd8Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2173 = Coupling(
    name="GC_2173",
    value="-((CKM2x3*cquqd8Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2174 = Coupling(
    name="GC_2174",
    value="(CKM3x1*cquqd8Im*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2175 = Coupling(
    name="GC_2175",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2176 = Coupling(
    name="GC_2176",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2177 = Coupling(
    name="GC_2177",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2178 = Coupling(
    name="GC_2178",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2179 = Coupling(
    name="GC_2179",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2180 = Coupling(
    name="GC_2180",
    value="(cHudIm*ee*yc*ydo*complexconjugate(CKM2x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_2181 = Coupling(
    name="GC_2181",
    value="(cHudRe*ee*complex(0,1)*yc*ydo*complexconjugate(CKM2x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_2182 = Coupling(
    name="GC_2182",
    value="(cHudIm*ee*vevhat*yc*ydo*complexconjugate(CKM2x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_2183 = Coupling(
    name="GC_2183",
    value="(cHudRe*ee*complex(0,1)*vevhat*yc*ydo*complexconjugate(CKM2x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_2184 = Coupling(
    name="GC_2184",
    value="(cHudIm*ee*vevhat**2*yc*ydo*complexconjugate(CKM2x1))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_2185 = Coupling(
    name="GC_2185",
    value="(cHudRe*ee*complex(0,1)*vevhat**2*yc*ydo*complexconjugate(CKM2x1))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_2186 = Coupling(
    name="GC_2186",
    value="-((clequ1Im*yc*ye*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2187 = Coupling(
    name="GC_2187",
    value="(clequ1Re*complex(0,1)*yc*ye*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2188 = Coupling(
    name="GC_2188",
    value="-(clequ3Im*yc*ye*complexconjugate(CKM2x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2189 = Coupling(
    name="GC_2189",
    value="(clequ3Im*yc*ye*complexconjugate(CKM2x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2190 = Coupling(
    name="GC_2190",
    value="(clequ3Re*complex(0,1)*yc*ye*complexconjugate(CKM2x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2191 = Coupling(
    name="GC_2191",
    value="-(clequ3Re*complex(0,1)*yc*ye*complexconjugate(CKM2x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2192 = Coupling(
    name="GC_2192",
    value="-((cledqIm*ydo*ye*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2193 = Coupling(
    name="GC_2193",
    value="(cledqRe*complex(0,1)*ydo*ye*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2194 = Coupling(
    name="GC_2194",
    value="-((clequ1Im*yc*ym*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2195 = Coupling(
    name="GC_2195",
    value="(clequ1Re*complex(0,1)*yc*ym*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2196 = Coupling(
    name="GC_2196",
    value="-(clequ3Im*yc*ym*complexconjugate(CKM2x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2197 = Coupling(
    name="GC_2197",
    value="(clequ3Im*yc*ym*complexconjugate(CKM2x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2198 = Coupling(
    name="GC_2198",
    value="(clequ3Re*complex(0,1)*yc*ym*complexconjugate(CKM2x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2199 = Coupling(
    name="GC_2199",
    value="-(clequ3Re*complex(0,1)*yc*ym*complexconjugate(CKM2x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2200 = Coupling(
    name="GC_2200",
    value="-((cledqIm*ydo*ym*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2201 = Coupling(
    name="GC_2201",
    value="(cledqRe*complex(0,1)*ydo*ym*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2202 = Coupling(
    name="GC_2202",
    value="-((CKM1x2*cquqd11Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2203 = Coupling(
    name="GC_2203",
    value="-((CKM2x2*cquqd11Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2204 = Coupling(
    name="GC_2204",
    value="-((CKM3x2*cquqd11Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2205 = Coupling(
    name="GC_2205",
    value="(CKM1x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2206 = Coupling(
    name="GC_2206",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2207 = Coupling(
    name="GC_2207",
    value="(CKM3x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2208 = Coupling(
    name="GC_2208",
    value="(CKM1x2*cquqd1Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2209 = Coupling(
    name="GC_2209",
    value="(CKM2x2*cquqd1Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2210 = Coupling(
    name="GC_2210",
    value="(CKM3x2*cquqd1Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2211 = Coupling(
    name="GC_2211",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2212 = Coupling(
    name="GC_2212",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2213 = Coupling(
    name="GC_2213",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2214 = Coupling(
    name="GC_2214",
    value="-((CKM1x2*cquqd81Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2215 = Coupling(
    name="GC_2215",
    value="-((CKM2x2*cquqd81Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2216 = Coupling(
    name="GC_2216",
    value="-((CKM3x2*cquqd81Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2217 = Coupling(
    name="GC_2217",
    value="(CKM1x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2218 = Coupling(
    name="GC_2218",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2219 = Coupling(
    name="GC_2219",
    value="(CKM3x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2220 = Coupling(
    name="GC_2220",
    value="(CKM1x2*cquqd8Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2221 = Coupling(
    name="GC_2221",
    value="(CKM2x2*cquqd8Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2222 = Coupling(
    name="GC_2222",
    value="(CKM3x2*cquqd8Im*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2223 = Coupling(
    name="GC_2223",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2224 = Coupling(
    name="GC_2224",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2225 = Coupling(
    name="GC_2225",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2226 = Coupling(
    name="GC_2226",
    value="(CKM3x1*cquqd11Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2227 = Coupling(
    name="GC_2227",
    value="(CKM3x2*cquqd11Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2228 = Coupling(
    name="GC_2228",
    value="(CKM3x3*cquqd11Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2229 = Coupling(
    name="GC_2229",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2230 = Coupling(
    name="GC_2230",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2231 = Coupling(
    name="GC_2231",
    value="(CKM3x3*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2232 = Coupling(
    name="GC_2232",
    value="-((CKM3x1*cquqd1Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2233 = Coupling(
    name="GC_2233",
    value="-((CKM3x2*cquqd1Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2234 = Coupling(
    name="GC_2234",
    value="-((CKM3x3*cquqd1Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2235 = Coupling(
    name="GC_2235",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2236 = Coupling(
    name="GC_2236",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2237 = Coupling(
    name="GC_2237",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2238 = Coupling(
    name="GC_2238",
    value="(CKM3x1*cquqd81Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2239 = Coupling(
    name="GC_2239",
    value="(CKM3x2*cquqd81Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2240 = Coupling(
    name="GC_2240",
    value="(CKM3x3*cquqd81Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2241 = Coupling(
    name="GC_2241",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2242 = Coupling(
    name="GC_2242",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2243 = Coupling(
    name="GC_2243",
    value="(CKM3x3*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2244 = Coupling(
    name="GC_2244",
    value="-((CKM3x1*cquqd8Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2245 = Coupling(
    name="GC_2245",
    value="-((CKM3x2*cquqd8Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2246 = Coupling(
    name="GC_2246",
    value="-((CKM3x3*cquqd8Im*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2247 = Coupling(
    name="GC_2247",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2248 = Coupling(
    name="GC_2248",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2249 = Coupling(
    name="GC_2249",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2250 = Coupling(
    name="GC_2250",
    value="-((clequ1Im*yc*ytau*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2251 = Coupling(
    name="GC_2251",
    value="(clequ1Re*complex(0,1)*yc*ytau*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2252 = Coupling(
    name="GC_2252",
    value="-(clequ3Im*yc*ytau*complexconjugate(CKM2x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2253 = Coupling(
    name="GC_2253",
    value="(clequ3Im*yc*ytau*complexconjugate(CKM2x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2254 = Coupling(
    name="GC_2254",
    value="(clequ3Re*complex(0,1)*yc*ytau*complexconjugate(CKM2x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2255 = Coupling(
    name="GC_2255",
    value="-(clequ3Re*complex(0,1)*yc*ytau*complexconjugate(CKM2x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2256 = Coupling(
    name="GC_2256",
    value="-((cledqIm*ydo*ytau*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2257 = Coupling(
    name="GC_2257",
    value="(cledqRe*complex(0,1)*ydo*ytau*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2258 = Coupling(
    name="GC_2258",
    value="(CKM1x1*cquqd11Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2259 = Coupling(
    name="GC_2259",
    value="(CKM1x2*cquqd11Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2260 = Coupling(
    name="GC_2260",
    value="(CKM1x3*cquqd11Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2261 = Coupling(
    name="GC_2261",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2262 = Coupling(
    name="GC_2262",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2263 = Coupling(
    name="GC_2263",
    value="(CKM1x3*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2264 = Coupling(
    name="GC_2264",
    value="-((CKM1x1*cquqd1Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2265 = Coupling(
    name="GC_2265",
    value="-((CKM1x2*cquqd1Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2266 = Coupling(
    name="GC_2266",
    value="-((CKM1x3*cquqd1Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2267 = Coupling(
    name="GC_2267",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2268 = Coupling(
    name="GC_2268",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2269 = Coupling(
    name="GC_2269",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2270 = Coupling(
    name="GC_2270",
    value="(CKM1x1*cquqd81Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2271 = Coupling(
    name="GC_2271",
    value="(CKM1x2*cquqd81Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2272 = Coupling(
    name="GC_2272",
    value="(CKM1x3*cquqd81Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2273 = Coupling(
    name="GC_2273",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2274 = Coupling(
    name="GC_2274",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2275 = Coupling(
    name="GC_2275",
    value="(CKM1x3*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2276 = Coupling(
    name="GC_2276",
    value="-((CKM1x1*cquqd8Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2277 = Coupling(
    name="GC_2277",
    value="-((CKM1x2*cquqd8Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2278 = Coupling(
    name="GC_2278",
    value="-((CKM1x3*cquqd8Im*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2279 = Coupling(
    name="GC_2279",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2280 = Coupling(
    name="GC_2280",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2281 = Coupling(
    name="GC_2281",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM2x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2282 = Coupling(
    name="GC_2282",
    value="(2*clq3*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2283 = Coupling(
    name="GC_2283",
    value="(2*CKM1x1*cqq11*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2284 = Coupling(
    name="GC_2284",
    value="(2*CKM1x2*cqq11*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2285 = Coupling(
    name="GC_2285",
    value="(2*CKM1x3*cqq11*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2286 = Coupling(
    name="GC_2286",
    value="(2*CKM2x1*cqq11*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2287 = Coupling(
    name="GC_2287",
    value="(2*CKM2x2*cqq11*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2288 = Coupling(
    name="GC_2288",
    value="(2*CKM2x3*cqq11*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2289 = Coupling(
    name="GC_2289",
    value="(2*CKM3x1*cqq11*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2290 = Coupling(
    name="GC_2290",
    value="(2*CKM3x2*cqq11*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2291 = Coupling(
    name="GC_2291",
    value="(2*CKM3x3*cqq11*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2292 = Coupling(
    name="GC_2292",
    value="(4*CKM1x1*cqq3*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2293 = Coupling(
    name="GC_2293",
    value="(4*CKM1x2*cqq3*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2294 = Coupling(
    name="GC_2294",
    value="(4*CKM1x3*cqq3*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2295 = Coupling(
    name="GC_2295",
    value="(4*CKM2x1*cqq3*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2296 = Coupling(
    name="GC_2296",
    value="(4*CKM2x2*cqq3*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2297 = Coupling(
    name="GC_2297",
    value="(4*CKM2x3*cqq3*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2298 = Coupling(
    name="GC_2298",
    value="(4*CKM3x1*cqq3*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2299 = Coupling(
    name="GC_2299",
    value="(4*CKM3x2*cqq3*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2300 = Coupling(
    name="GC_2300",
    value="(4*CKM3x3*cqq3*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2301 = Coupling(
    name="GC_2301",
    value="(-2*CKM1x1*cqq31*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2302 = Coupling(
    name="GC_2302",
    value="(-2*CKM1x2*cqq31*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2303 = Coupling(
    name="GC_2303",
    value="(-2*CKM1x3*cqq31*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2304 = Coupling(
    name="GC_2304",
    value="(-2*CKM2x1*cqq31*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2305 = Coupling(
    name="GC_2305",
    value="(-2*CKM2x2*cqq31*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2306 = Coupling(
    name="GC_2306",
    value="(-2*CKM2x3*cqq31*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2307 = Coupling(
    name="GC_2307",
    value="(-2*CKM3x1*cqq31*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2308 = Coupling(
    name="GC_2308",
    value="(-2*CKM3x2*cqq31*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2309 = Coupling(
    name="GC_2309",
    value="(-2*CKM3x3*cqq31*complex(0,1)*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2310 = Coupling(
    name="GC_2310",
    value="-((ee*complex(0,1)*complexconjugate(CKM2x2))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2311 = Coupling(
    name="GC_2311",
    value="-((cHq3*ee*complex(0,1)*complexconjugate(CKM2x2)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_2312 = Coupling(
    name="GC_2312",
    value="-((ee*complex(0,1)*propCorr*complexconjugate(CKM2x2))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2313 = Coupling(
    name="GC_2313",
    value="-((cHq3*ee*complex(0,1)*vevhat*complexconjugate(CKM2x2)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_2314 = Coupling(
    name="GC_2314",
    value="-((cHq3*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2315 = Coupling(
    name="GC_2315",
    value="-((cHWB*cth*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x2))/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2316 = Coupling(
    name="GC_2316",
    value="-((cHl3*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x2))/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2317 = Coupling(
    name="GC_2317",
    value="(cll1*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x2))/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_2318 = Coupling(
    name="GC_2318",
    value="(cHDD*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x2))/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_2319 = Coupling(
    name="GC_2319",
    value="-((CKM1x3*cquqd11Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2320 = Coupling(
    name="GC_2320",
    value="-((CKM2x3*cquqd11Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2321 = Coupling(
    name="GC_2321",
    value="-((CKM3x3*cquqd11Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2322 = Coupling(
    name="GC_2322",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2323 = Coupling(
    name="GC_2323",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2324 = Coupling(
    name="GC_2324",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2325 = Coupling(
    name="GC_2325",
    value="(CKM1x3*cquqd1Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2326 = Coupling(
    name="GC_2326",
    value="(CKM2x3*cquqd1Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2327 = Coupling(
    name="GC_2327",
    value="(CKM3x3*cquqd1Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2328 = Coupling(
    name="GC_2328",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2329 = Coupling(
    name="GC_2329",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2330 = Coupling(
    name="GC_2330",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2331 = Coupling(
    name="GC_2331",
    value="-((CKM1x3*cquqd81Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2332 = Coupling(
    name="GC_2332",
    value="-((CKM2x3*cquqd81Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2333 = Coupling(
    name="GC_2333",
    value="-((CKM3x3*cquqd81Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2334 = Coupling(
    name="GC_2334",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2335 = Coupling(
    name="GC_2335",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2336 = Coupling(
    name="GC_2336",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2337 = Coupling(
    name="GC_2337",
    value="(CKM1x3*cquqd8Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2338 = Coupling(
    name="GC_2338",
    value="(CKM2x3*cquqd8Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2339 = Coupling(
    name="GC_2339",
    value="(CKM3x3*cquqd8Im*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2340 = Coupling(
    name="GC_2340",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2341 = Coupling(
    name="GC_2341",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2342 = Coupling(
    name="GC_2342",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2343 = Coupling(
    name="GC_2343",
    value="-((CKM1x1*cquqd11Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2344 = Coupling(
    name="GC_2344",
    value="-((CKM2x1*cquqd11Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2345 = Coupling(
    name="GC_2345",
    value="-((CKM3x1*cquqd11Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2346 = Coupling(
    name="GC_2346",
    value="(CKM1x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2347 = Coupling(
    name="GC_2347",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2348 = Coupling(
    name="GC_2348",
    value="(CKM3x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2349 = Coupling(
    name="GC_2349",
    value="(CKM1x1*cquqd1Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2350 = Coupling(
    name="GC_2350",
    value="(CKM2x1*cquqd1Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2351 = Coupling(
    name="GC_2351",
    value="(CKM3x1*cquqd1Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2352 = Coupling(
    name="GC_2352",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2353 = Coupling(
    name="GC_2353",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2354 = Coupling(
    name="GC_2354",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2355 = Coupling(
    name="GC_2355",
    value="-((CKM1x1*cquqd81Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2356 = Coupling(
    name="GC_2356",
    value="-((CKM2x1*cquqd81Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2357 = Coupling(
    name="GC_2357",
    value="-((CKM3x1*cquqd81Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2358 = Coupling(
    name="GC_2358",
    value="(CKM1x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2359 = Coupling(
    name="GC_2359",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2360 = Coupling(
    name="GC_2360",
    value="(CKM3x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2361 = Coupling(
    name="GC_2361",
    value="(CKM1x1*cquqd8Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2362 = Coupling(
    name="GC_2362",
    value="(CKM2x1*cquqd8Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2363 = Coupling(
    name="GC_2363",
    value="(CKM3x1*cquqd8Im*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2364 = Coupling(
    name="GC_2364",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2365 = Coupling(
    name="GC_2365",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2366 = Coupling(
    name="GC_2366",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2367 = Coupling(
    name="GC_2367",
    value="-((clequ1Im*yc*ye*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2368 = Coupling(
    name="GC_2368",
    value="(clequ1Re*complex(0,1)*yc*ye*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2369 = Coupling(
    name="GC_2369",
    value="-(clequ3Im*yc*ye*complexconjugate(CKM2x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2370 = Coupling(
    name="GC_2370",
    value="(clequ3Im*yc*ye*complexconjugate(CKM2x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2371 = Coupling(
    name="GC_2371",
    value="(clequ3Re*complex(0,1)*yc*ye*complexconjugate(CKM2x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2372 = Coupling(
    name="GC_2372",
    value="-(clequ3Re*complex(0,1)*yc*ye*complexconjugate(CKM2x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2373 = Coupling(
    name="GC_2373",
    value="-((clequ1Im*yc*ym*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2374 = Coupling(
    name="GC_2374",
    value="(clequ1Re*complex(0,1)*yc*ym*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2375 = Coupling(
    name="GC_2375",
    value="-(clequ3Im*yc*ym*complexconjugate(CKM2x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2376 = Coupling(
    name="GC_2376",
    value="(clequ3Im*yc*ym*complexconjugate(CKM2x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2377 = Coupling(
    name="GC_2377",
    value="(clequ3Re*complex(0,1)*yc*ym*complexconjugate(CKM2x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2378 = Coupling(
    name="GC_2378",
    value="-(clequ3Re*complex(0,1)*yc*ym*complexconjugate(CKM2x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2379 = Coupling(
    name="GC_2379",
    value="-((CKM1x2*cquqd11Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2380 = Coupling(
    name="GC_2380",
    value="(CKM2x1*cquqd11Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2381 = Coupling(
    name="GC_2381",
    value="-((CKM2x2*cquqd11Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2382 = Coupling(
    name="GC_2382",
    value="(CKM2x2*cquqd11Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2383 = Coupling(
    name="GC_2383",
    value="(CKM2x3*cquqd11Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2384 = Coupling(
    name="GC_2384",
    value="-((CKM3x2*cquqd11Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2385 = Coupling(
    name="GC_2385",
    value="(CKM1x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2386 = Coupling(
    name="GC_2386",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2387 = Coupling(
    name="GC_2387",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2388 = Coupling(
    name="GC_2388",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2389 = Coupling(
    name="GC_2389",
    value="(CKM3x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2390 = Coupling(
    name="GC_2390",
    value="(CKM1x2*cquqd1Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2391 = Coupling(
    name="GC_2391",
    value="-((CKM2x1*cquqd1Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2392 = Coupling(
    name="GC_2392",
    value="-((CKM2x2*cquqd1Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2393 = Coupling(
    name="GC_2393",
    value="(CKM2x2*cquqd1Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2394 = Coupling(
    name="GC_2394",
    value="-((CKM2x3*cquqd1Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2395 = Coupling(
    name="GC_2395",
    value="(CKM3x2*cquqd1Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2396 = Coupling(
    name="GC_2396",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2397 = Coupling(
    name="GC_2397",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2398 = Coupling(
    name="GC_2398",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2399 = Coupling(
    name="GC_2399",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2400 = Coupling(
    name="GC_2400",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2401 = Coupling(
    name="GC_2401",
    value="-((CKM1x2*cquqd81Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2402 = Coupling(
    name="GC_2402",
    value="(CKM2x1*cquqd81Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2403 = Coupling(
    name="GC_2403",
    value="-((CKM2x2*cquqd81Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2404 = Coupling(
    name="GC_2404",
    value="(CKM2x2*cquqd81Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2405 = Coupling(
    name="GC_2405",
    value="(CKM2x3*cquqd81Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2406 = Coupling(
    name="GC_2406",
    value="-((CKM3x2*cquqd81Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2407 = Coupling(
    name="GC_2407",
    value="(CKM1x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2408 = Coupling(
    name="GC_2408",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2409 = Coupling(
    name="GC_2409",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2410 = Coupling(
    name="GC_2410",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2411 = Coupling(
    name="GC_2411",
    value="(CKM3x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2412 = Coupling(
    name="GC_2412",
    value="(CKM1x2*cquqd8Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2413 = Coupling(
    name="GC_2413",
    value="-((CKM2x1*cquqd8Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2414 = Coupling(
    name="GC_2414",
    value="-((CKM2x2*cquqd8Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2415 = Coupling(
    name="GC_2415",
    value="(CKM2x2*cquqd8Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2416 = Coupling(
    name="GC_2416",
    value="-((CKM2x3*cquqd8Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2417 = Coupling(
    name="GC_2417",
    value="(CKM3x2*cquqd8Im*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2418 = Coupling(
    name="GC_2418",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2419 = Coupling(
    name="GC_2419",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2420 = Coupling(
    name="GC_2420",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2421 = Coupling(
    name="GC_2421",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2422 = Coupling(
    name="GC_2422",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2423 = Coupling(
    name="GC_2423",
    value="(cHudIm*ee*yc*ys*complexconjugate(CKM2x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_2424 = Coupling(
    name="GC_2424",
    value="(cHudRe*ee*complex(0,1)*yc*ys*complexconjugate(CKM2x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_2425 = Coupling(
    name="GC_2425",
    value="(cHudIm*ee*vevhat*yc*ys*complexconjugate(CKM2x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_2426 = Coupling(
    name="GC_2426",
    value="(cHudRe*ee*complex(0,1)*vevhat*yc*ys*complexconjugate(CKM2x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_2427 = Coupling(
    name="GC_2427",
    value="(cHudIm*ee*vevhat**2*yc*ys*complexconjugate(CKM2x2))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_2428 = Coupling(
    name="GC_2428",
    value="(cHudRe*ee*complex(0,1)*vevhat**2*yc*ys*complexconjugate(CKM2x2))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_2429 = Coupling(
    name="GC_2429",
    value="-((cledqIm*ye*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2430 = Coupling(
    name="GC_2430",
    value="(cledqRe*complex(0,1)*ye*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2431 = Coupling(
    name="GC_2431",
    value="-((cledqIm*ym*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2432 = Coupling(
    name="GC_2432",
    value="(cledqRe*complex(0,1)*ym*ys*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2433 = Coupling(
    name="GC_2433",
    value="(CKM3x1*cquqd11Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2434 = Coupling(
    name="GC_2434",
    value="(CKM3x2*cquqd11Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2435 = Coupling(
    name="GC_2435",
    value="(CKM3x3*cquqd11Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2436 = Coupling(
    name="GC_2436",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2437 = Coupling(
    name="GC_2437",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2438 = Coupling(
    name="GC_2438",
    value="(CKM3x3*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2439 = Coupling(
    name="GC_2439",
    value="-((CKM3x1*cquqd1Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2440 = Coupling(
    name="GC_2440",
    value="-((CKM3x2*cquqd1Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2441 = Coupling(
    name="GC_2441",
    value="-((CKM3x3*cquqd1Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2442 = Coupling(
    name="GC_2442",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2443 = Coupling(
    name="GC_2443",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2444 = Coupling(
    name="GC_2444",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2445 = Coupling(
    name="GC_2445",
    value="(CKM3x1*cquqd81Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2446 = Coupling(
    name="GC_2446",
    value="(CKM3x2*cquqd81Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2447 = Coupling(
    name="GC_2447",
    value="(CKM3x3*cquqd81Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2448 = Coupling(
    name="GC_2448",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2449 = Coupling(
    name="GC_2449",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2450 = Coupling(
    name="GC_2450",
    value="(CKM3x3*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2451 = Coupling(
    name="GC_2451",
    value="-((CKM3x1*cquqd8Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2452 = Coupling(
    name="GC_2452",
    value="-((CKM3x2*cquqd8Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2453 = Coupling(
    name="GC_2453",
    value="-((CKM3x3*cquqd8Im*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2454 = Coupling(
    name="GC_2454",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2455 = Coupling(
    name="GC_2455",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2456 = Coupling(
    name="GC_2456",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2457 = Coupling(
    name="GC_2457",
    value="-((clequ1Im*yc*ytau*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2458 = Coupling(
    name="GC_2458",
    value="(clequ1Re*complex(0,1)*yc*ytau*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2459 = Coupling(
    name="GC_2459",
    value="-(clequ3Im*yc*ytau*complexconjugate(CKM2x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2460 = Coupling(
    name="GC_2460",
    value="(clequ3Im*yc*ytau*complexconjugate(CKM2x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2461 = Coupling(
    name="GC_2461",
    value="(clequ3Re*complex(0,1)*yc*ytau*complexconjugate(CKM2x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2462 = Coupling(
    name="GC_2462",
    value="-(clequ3Re*complex(0,1)*yc*ytau*complexconjugate(CKM2x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2463 = Coupling(
    name="GC_2463",
    value="-((cledqIm*ys*ytau*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2464 = Coupling(
    name="GC_2464",
    value="(cledqRe*complex(0,1)*ys*ytau*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2465 = Coupling(
    name="GC_2465",
    value="(CKM1x1*cquqd11Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2466 = Coupling(
    name="GC_2466",
    value="(CKM1x2*cquqd11Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2467 = Coupling(
    name="GC_2467",
    value="(CKM1x3*cquqd11Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2468 = Coupling(
    name="GC_2468",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2469 = Coupling(
    name="GC_2469",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2470 = Coupling(
    name="GC_2470",
    value="(CKM1x3*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2471 = Coupling(
    name="GC_2471",
    value="-((CKM1x1*cquqd1Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2472 = Coupling(
    name="GC_2472",
    value="-((CKM1x2*cquqd1Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2473 = Coupling(
    name="GC_2473",
    value="-((CKM1x3*cquqd1Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2474 = Coupling(
    name="GC_2474",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2475 = Coupling(
    name="GC_2475",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2476 = Coupling(
    name="GC_2476",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2477 = Coupling(
    name="GC_2477",
    value="(CKM1x1*cquqd81Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2478 = Coupling(
    name="GC_2478",
    value="(CKM1x2*cquqd81Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2479 = Coupling(
    name="GC_2479",
    value="(CKM1x3*cquqd81Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2480 = Coupling(
    name="GC_2480",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2481 = Coupling(
    name="GC_2481",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2482 = Coupling(
    name="GC_2482",
    value="(CKM1x3*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2483 = Coupling(
    name="GC_2483",
    value="-((CKM1x1*cquqd8Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2484 = Coupling(
    name="GC_2484",
    value="-((CKM1x2*cquqd8Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2485 = Coupling(
    name="GC_2485",
    value="-((CKM1x3*cquqd8Im*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2486 = Coupling(
    name="GC_2486",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2487 = Coupling(
    name="GC_2487",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2488 = Coupling(
    name="GC_2488",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM2x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2489 = Coupling(
    name="GC_2489",
    value="(2*clq3*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2490 = Coupling(
    name="GC_2490",
    value="(2*CKM1x1*cqq11*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2491 = Coupling(
    name="GC_2491",
    value="(2*CKM1x2*cqq11*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2492 = Coupling(
    name="GC_2492",
    value="(2*CKM1x3*cqq11*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2493 = Coupling(
    name="GC_2493",
    value="(2*CKM2x1*cqq11*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2494 = Coupling(
    name="GC_2494",
    value="(2*CKM2x2*cqq11*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2495 = Coupling(
    name="GC_2495",
    value="(2*CKM2x3*cqq11*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2496 = Coupling(
    name="GC_2496",
    value="(2*CKM3x1*cqq11*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2497 = Coupling(
    name="GC_2497",
    value="(2*CKM3x2*cqq11*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2498 = Coupling(
    name="GC_2498",
    value="(2*CKM3x3*cqq11*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2499 = Coupling(
    name="GC_2499",
    value="(4*CKM1x1*cqq3*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2500 = Coupling(
    name="GC_2500",
    value="(4*CKM1x2*cqq3*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2501 = Coupling(
    name="GC_2501",
    value="(4*CKM1x3*cqq3*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2502 = Coupling(
    name="GC_2502",
    value="(4*CKM2x1*cqq3*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2503 = Coupling(
    name="GC_2503",
    value="(4*CKM2x2*cqq3*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2504 = Coupling(
    name="GC_2504",
    value="(4*CKM2x3*cqq3*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2505 = Coupling(
    name="GC_2505",
    value="(4*CKM3x1*cqq3*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2506 = Coupling(
    name="GC_2506",
    value="(4*CKM3x2*cqq3*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2507 = Coupling(
    name="GC_2507",
    value="(4*CKM3x3*cqq3*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2508 = Coupling(
    name="GC_2508",
    value="(-2*CKM1x1*cqq31*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2509 = Coupling(
    name="GC_2509",
    value="(-2*CKM1x2*cqq31*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2510 = Coupling(
    name="GC_2510",
    value="(-2*CKM1x3*cqq31*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2511 = Coupling(
    name="GC_2511",
    value="(-2*CKM2x1*cqq31*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2512 = Coupling(
    name="GC_2512",
    value="(-2*CKM2x2*cqq31*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2513 = Coupling(
    name="GC_2513",
    value="(-2*CKM2x3*cqq31*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2514 = Coupling(
    name="GC_2514",
    value="(-2*CKM3x1*cqq31*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2515 = Coupling(
    name="GC_2515",
    value="(-2*CKM3x2*cqq31*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2516 = Coupling(
    name="GC_2516",
    value="(-2*CKM3x3*cqq31*complex(0,1)*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2517 = Coupling(
    name="GC_2517",
    value="-((ee*complex(0,1)*complexconjugate(CKM2x3))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2518 = Coupling(
    name="GC_2518",
    value="-((cHq3*ee*complex(0,1)*complexconjugate(CKM2x3)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_2519 = Coupling(
    name="GC_2519",
    value="-((ee*complex(0,1)*propCorr*complexconjugate(CKM2x3))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2520 = Coupling(
    name="GC_2520",
    value="-((cHq3*ee*complex(0,1)*vevhat*complexconjugate(CKM2x3)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_2521 = Coupling(
    name="GC_2521",
    value="-((cHq3*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2522 = Coupling(
    name="GC_2522",
    value="-((cHWB*cth*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x3))/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2523 = Coupling(
    name="GC_2523",
    value="-((cHl3*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x3))/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2524 = Coupling(
    name="GC_2524",
    value="(cll1*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x3))/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_2525 = Coupling(
    name="GC_2525",
    value="(cHDD*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM2x3))/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_2526 = Coupling(
    name="GC_2526",
    value="-((CKM1x3*cquqd11Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2527 = Coupling(
    name="GC_2527",
    value="(CKM2x1*cquqd11Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2528 = Coupling(
    name="GC_2528",
    value="(CKM2x2*cquqd11Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2529 = Coupling(
    name="GC_2529",
    value="-((CKM2x3*cquqd11Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2530 = Coupling(
    name="GC_2530",
    value="(CKM2x3*cquqd11Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2531 = Coupling(
    name="GC_2531",
    value="-((CKM3x3*cquqd11Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2532 = Coupling(
    name="GC_2532",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2533 = Coupling(
    name="GC_2533",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2534 = Coupling(
    name="GC_2534",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2535 = Coupling(
    name="GC_2535",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2536 = Coupling(
    name="GC_2536",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2537 = Coupling(
    name="GC_2537",
    value="(CKM1x3*cquqd1Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2538 = Coupling(
    name="GC_2538",
    value="-((CKM2x1*cquqd1Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2539 = Coupling(
    name="GC_2539",
    value="-((CKM2x2*cquqd1Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2540 = Coupling(
    name="GC_2540",
    value="-((CKM2x3*cquqd1Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2541 = Coupling(
    name="GC_2541",
    value="(CKM2x3*cquqd1Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2542 = Coupling(
    name="GC_2542",
    value="(CKM3x3*cquqd1Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2543 = Coupling(
    name="GC_2543",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2544 = Coupling(
    name="GC_2544",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2545 = Coupling(
    name="GC_2545",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2546 = Coupling(
    name="GC_2546",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2547 = Coupling(
    name="GC_2547",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2548 = Coupling(
    name="GC_2548",
    value="-((CKM1x3*cquqd81Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2549 = Coupling(
    name="GC_2549",
    value="(CKM2x1*cquqd81Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2550 = Coupling(
    name="GC_2550",
    value="(CKM2x2*cquqd81Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2551 = Coupling(
    name="GC_2551",
    value="-((CKM2x3*cquqd81Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2552 = Coupling(
    name="GC_2552",
    value="(CKM2x3*cquqd81Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2553 = Coupling(
    name="GC_2553",
    value="-((CKM3x3*cquqd81Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2554 = Coupling(
    name="GC_2554",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2555 = Coupling(
    name="GC_2555",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2556 = Coupling(
    name="GC_2556",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2557 = Coupling(
    name="GC_2557",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2558 = Coupling(
    name="GC_2558",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2559 = Coupling(
    name="GC_2559",
    value="(CKM1x3*cquqd8Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2560 = Coupling(
    name="GC_2560",
    value="-((CKM2x1*cquqd8Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2561 = Coupling(
    name="GC_2561",
    value="-((CKM2x2*cquqd8Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2562 = Coupling(
    name="GC_2562",
    value="-((CKM2x3*cquqd8Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2563 = Coupling(
    name="GC_2563",
    value="(CKM2x3*cquqd8Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2564 = Coupling(
    name="GC_2564",
    value="(CKM3x3*cquqd8Im*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2565 = Coupling(
    name="GC_2565",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2566 = Coupling(
    name="GC_2566",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2567 = Coupling(
    name="GC_2567",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2568 = Coupling(
    name="GC_2568",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2569 = Coupling(
    name="GC_2569",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2570 = Coupling(
    name="GC_2570",
    value="(cHudIm*ee*yb*yc*complexconjugate(CKM2x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_2571 = Coupling(
    name="GC_2571",
    value="(cHudRe*ee*complex(0,1)*yb*yc*complexconjugate(CKM2x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_2572 = Coupling(
    name="GC_2572",
    value="(cHudIm*ee*vevhat*yb*yc*complexconjugate(CKM2x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_2573 = Coupling(
    name="GC_2573",
    value="(cHudRe*ee*complex(0,1)*vevhat*yb*yc*complexconjugate(CKM2x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_2574 = Coupling(
    name="GC_2574",
    value="(cHudIm*ee*vevhat**2*yb*yc*complexconjugate(CKM2x3))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_2575 = Coupling(
    name="GC_2575",
    value="(cHudRe*ee*complex(0,1)*vevhat**2*yb*yc*complexconjugate(CKM2x3))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_2576 = Coupling(
    name="GC_2576",
    value="-((CKM1x1*cquqd11Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2577 = Coupling(
    name="GC_2577",
    value="-((CKM2x1*cquqd11Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2578 = Coupling(
    name="GC_2578",
    value="-((CKM3x1*cquqd11Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2579 = Coupling(
    name="GC_2579",
    value="(CKM1x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2580 = Coupling(
    name="GC_2580",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2581 = Coupling(
    name="GC_2581",
    value="(CKM3x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2582 = Coupling(
    name="GC_2582",
    value="(CKM1x1*cquqd1Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2583 = Coupling(
    name="GC_2583",
    value="(CKM2x1*cquqd1Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2584 = Coupling(
    name="GC_2584",
    value="(CKM3x1*cquqd1Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2585 = Coupling(
    name="GC_2585",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2586 = Coupling(
    name="GC_2586",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2587 = Coupling(
    name="GC_2587",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2588 = Coupling(
    name="GC_2588",
    value="-((CKM1x1*cquqd81Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2589 = Coupling(
    name="GC_2589",
    value="-((CKM2x1*cquqd81Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2590 = Coupling(
    name="GC_2590",
    value="-((CKM3x1*cquqd81Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2591 = Coupling(
    name="GC_2591",
    value="(CKM1x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2592 = Coupling(
    name="GC_2592",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2593 = Coupling(
    name="GC_2593",
    value="(CKM3x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2594 = Coupling(
    name="GC_2594",
    value="(CKM1x1*cquqd8Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2595 = Coupling(
    name="GC_2595",
    value="(CKM2x1*cquqd8Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2596 = Coupling(
    name="GC_2596",
    value="(CKM3x1*cquqd8Im*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2597 = Coupling(
    name="GC_2597",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2598 = Coupling(
    name="GC_2598",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2599 = Coupling(
    name="GC_2599",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2600 = Coupling(
    name="GC_2600",
    value="-((cledqIm*yb*ye*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2601 = Coupling(
    name="GC_2601",
    value="(cledqRe*complex(0,1)*yb*ye*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2602 = Coupling(
    name="GC_2602",
    value="-((clequ1Im*yc*ye*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2603 = Coupling(
    name="GC_2603",
    value="(clequ1Re*complex(0,1)*yc*ye*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2604 = Coupling(
    name="GC_2604",
    value="-(clequ3Im*yc*ye*complexconjugate(CKM2x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2605 = Coupling(
    name="GC_2605",
    value="(clequ3Im*yc*ye*complexconjugate(CKM2x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2606 = Coupling(
    name="GC_2606",
    value="(clequ3Re*complex(0,1)*yc*ye*complexconjugate(CKM2x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2607 = Coupling(
    name="GC_2607",
    value="-(clequ3Re*complex(0,1)*yc*ye*complexconjugate(CKM2x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2608 = Coupling(
    name="GC_2608",
    value="-((cledqIm*yb*ym*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2609 = Coupling(
    name="GC_2609",
    value="(cledqRe*complex(0,1)*yb*ym*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2610 = Coupling(
    name="GC_2610",
    value="-((clequ1Im*yc*ym*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2611 = Coupling(
    name="GC_2611",
    value="(clequ1Re*complex(0,1)*yc*ym*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2612 = Coupling(
    name="GC_2612",
    value="-(clequ3Im*yc*ym*complexconjugate(CKM2x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2613 = Coupling(
    name="GC_2613",
    value="(clequ3Im*yc*ym*complexconjugate(CKM2x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2614 = Coupling(
    name="GC_2614",
    value="(clequ3Re*complex(0,1)*yc*ym*complexconjugate(CKM2x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2615 = Coupling(
    name="GC_2615",
    value="-(clequ3Re*complex(0,1)*yc*ym*complexconjugate(CKM2x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2616 = Coupling(
    name="GC_2616",
    value="-((CKM1x2*cquqd11Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2617 = Coupling(
    name="GC_2617",
    value="-((CKM2x2*cquqd11Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2618 = Coupling(
    name="GC_2618",
    value="-((CKM3x2*cquqd11Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2619 = Coupling(
    name="GC_2619",
    value="(CKM1x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2620 = Coupling(
    name="GC_2620",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2621 = Coupling(
    name="GC_2621",
    value="(CKM3x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2622 = Coupling(
    name="GC_2622",
    value="(CKM1x2*cquqd1Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2623 = Coupling(
    name="GC_2623",
    value="(CKM2x2*cquqd1Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2624 = Coupling(
    name="GC_2624",
    value="(CKM3x2*cquqd1Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2625 = Coupling(
    name="GC_2625",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2626 = Coupling(
    name="GC_2626",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2627 = Coupling(
    name="GC_2627",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2628 = Coupling(
    name="GC_2628",
    value="-((CKM1x2*cquqd81Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2629 = Coupling(
    name="GC_2629",
    value="-((CKM2x2*cquqd81Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2630 = Coupling(
    name="GC_2630",
    value="-((CKM3x2*cquqd81Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2631 = Coupling(
    name="GC_2631",
    value="(CKM1x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2632 = Coupling(
    name="GC_2632",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2633 = Coupling(
    name="GC_2633",
    value="(CKM3x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2634 = Coupling(
    name="GC_2634",
    value="(CKM1x2*cquqd8Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2635 = Coupling(
    name="GC_2635",
    value="(CKM2x2*cquqd8Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2636 = Coupling(
    name="GC_2636",
    value="(CKM3x2*cquqd8Im*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2637 = Coupling(
    name="GC_2637",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2638 = Coupling(
    name="GC_2638",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2639 = Coupling(
    name="GC_2639",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2640 = Coupling(
    name="GC_2640",
    value="(CKM3x1*cquqd11Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2641 = Coupling(
    name="GC_2641",
    value="(CKM3x2*cquqd11Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2642 = Coupling(
    name="GC_2642",
    value="(CKM3x3*cquqd11Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2643 = Coupling(
    name="GC_2643",
    value="(CKM3x1*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2644 = Coupling(
    name="GC_2644",
    value="(CKM3x2*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2645 = Coupling(
    name="GC_2645",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2646 = Coupling(
    name="GC_2646",
    value="-((CKM3x1*cquqd1Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2647 = Coupling(
    name="GC_2647",
    value="-((CKM3x2*cquqd1Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2648 = Coupling(
    name="GC_2648",
    value="-((CKM3x3*cquqd1Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2649 = Coupling(
    name="GC_2649",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2650 = Coupling(
    name="GC_2650",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2651 = Coupling(
    name="GC_2651",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2652 = Coupling(
    name="GC_2652",
    value="(CKM3x1*cquqd81Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2653 = Coupling(
    name="GC_2653",
    value="(CKM3x2*cquqd81Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2654 = Coupling(
    name="GC_2654",
    value="(CKM3x3*cquqd81Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2655 = Coupling(
    name="GC_2655",
    value="(CKM3x1*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2656 = Coupling(
    name="GC_2656",
    value="(CKM3x2*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2657 = Coupling(
    name="GC_2657",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2658 = Coupling(
    name="GC_2658",
    value="-((CKM3x1*cquqd8Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2659 = Coupling(
    name="GC_2659",
    value="-((CKM3x2*cquqd8Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2660 = Coupling(
    name="GC_2660",
    value="-((CKM3x3*cquqd8Im*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2661 = Coupling(
    name="GC_2661",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2662 = Coupling(
    name="GC_2662",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2663 = Coupling(
    name="GC_2663",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2664 = Coupling(
    name="GC_2664",
    value="-((cledqIm*yb*ytau*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2665 = Coupling(
    name="GC_2665",
    value="(cledqRe*complex(0,1)*yb*ytau*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2666 = Coupling(
    name="GC_2666",
    value="-((clequ1Im*yc*ytau*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2667 = Coupling(
    name="GC_2667",
    value="(clequ1Re*complex(0,1)*yc*ytau*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2668 = Coupling(
    name="GC_2668",
    value="-(clequ3Im*yc*ytau*complexconjugate(CKM2x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2669 = Coupling(
    name="GC_2669",
    value="(clequ3Im*yc*ytau*complexconjugate(CKM2x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2670 = Coupling(
    name="GC_2670",
    value="(clequ3Re*complex(0,1)*yc*ytau*complexconjugate(CKM2x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2671 = Coupling(
    name="GC_2671",
    value="-(clequ3Re*complex(0,1)*yc*ytau*complexconjugate(CKM2x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2672 = Coupling(
    name="GC_2672",
    value="(CKM1x1*cquqd11Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2673 = Coupling(
    name="GC_2673",
    value="(CKM1x2*cquqd11Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2674 = Coupling(
    name="GC_2674",
    value="(CKM1x3*cquqd11Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2675 = Coupling(
    name="GC_2675",
    value="(CKM1x1*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2676 = Coupling(
    name="GC_2676",
    value="(CKM1x2*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2677 = Coupling(
    name="GC_2677",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2678 = Coupling(
    name="GC_2678",
    value="-((CKM1x1*cquqd1Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2679 = Coupling(
    name="GC_2679",
    value="-((CKM1x2*cquqd1Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2680 = Coupling(
    name="GC_2680",
    value="-((CKM1x3*cquqd1Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2681 = Coupling(
    name="GC_2681",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2682 = Coupling(
    name="GC_2682",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2683 = Coupling(
    name="GC_2683",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2684 = Coupling(
    name="GC_2684",
    value="(CKM1x1*cquqd81Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2685 = Coupling(
    name="GC_2685",
    value="(CKM1x2*cquqd81Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2686 = Coupling(
    name="GC_2686",
    value="(CKM1x3*cquqd81Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2687 = Coupling(
    name="GC_2687",
    value="(CKM1x1*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2688 = Coupling(
    name="GC_2688",
    value="(CKM1x2*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2689 = Coupling(
    name="GC_2689",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2690 = Coupling(
    name="GC_2690",
    value="-((CKM1x1*cquqd8Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2691 = Coupling(
    name="GC_2691",
    value="-((CKM1x2*cquqd8Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2692 = Coupling(
    name="GC_2692",
    value="-((CKM1x3*cquqd8Im*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2693 = Coupling(
    name="GC_2693",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2694 = Coupling(
    name="GC_2694",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2695 = Coupling(
    name="GC_2695",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM2x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2696 = Coupling(
    name="GC_2696",
    value="(2*clq3*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2697 = Coupling(
    name="GC_2697",
    value="(2*CKM1x1*cqq11*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2698 = Coupling(
    name="GC_2698",
    value="(2*CKM1x2*cqq11*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2699 = Coupling(
    name="GC_2699",
    value="(2*CKM1x3*cqq11*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2700 = Coupling(
    name="GC_2700",
    value="(2*CKM2x1*cqq11*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2701 = Coupling(
    name="GC_2701",
    value="(2*CKM2x2*cqq11*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2702 = Coupling(
    name="GC_2702",
    value="(2*CKM2x3*cqq11*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2703 = Coupling(
    name="GC_2703",
    value="(2*CKM3x1*cqq11*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2704 = Coupling(
    name="GC_2704",
    value="(2*CKM3x2*cqq11*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2705 = Coupling(
    name="GC_2705",
    value="(2*CKM3x3*cqq11*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2706 = Coupling(
    name="GC_2706",
    value="(4*CKM1x1*cqq3*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2707 = Coupling(
    name="GC_2707",
    value="(4*CKM1x2*cqq3*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2708 = Coupling(
    name="GC_2708",
    value="(4*CKM1x3*cqq3*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2709 = Coupling(
    name="GC_2709",
    value="(4*CKM2x1*cqq3*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2710 = Coupling(
    name="GC_2710",
    value="(4*CKM2x2*cqq3*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2711 = Coupling(
    name="GC_2711",
    value="(4*CKM2x3*cqq3*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2712 = Coupling(
    name="GC_2712",
    value="(4*CKM3x1*cqq3*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2713 = Coupling(
    name="GC_2713",
    value="(4*CKM3x2*cqq3*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2714 = Coupling(
    name="GC_2714",
    value="(4*CKM3x3*cqq3*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2715 = Coupling(
    name="GC_2715",
    value="(-2*CKM1x1*cqq31*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2716 = Coupling(
    name="GC_2716",
    value="(-2*CKM1x2*cqq31*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2717 = Coupling(
    name="GC_2717",
    value="(-2*CKM1x3*cqq31*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2718 = Coupling(
    name="GC_2718",
    value="(-2*CKM2x1*cqq31*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2719 = Coupling(
    name="GC_2719",
    value="(-2*CKM2x2*cqq31*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2720 = Coupling(
    name="GC_2720",
    value="(-2*CKM2x3*cqq31*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2721 = Coupling(
    name="GC_2721",
    value="(-2*CKM3x1*cqq31*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2722 = Coupling(
    name="GC_2722",
    value="(-2*CKM3x2*cqq31*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2723 = Coupling(
    name="GC_2723",
    value="(-2*CKM3x3*cqq31*complex(0,1)*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2724 = Coupling(
    name="GC_2724",
    value="-((ee*complex(0,1)*complexconjugate(CKM3x1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2725 = Coupling(
    name="GC_2725",
    value="-((cHq3*ee*complex(0,1)*complexconjugate(CKM3x1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_2726 = Coupling(
    name="GC_2726",
    value="-((ee*complex(0,1)*propCorr*complexconjugate(CKM3x1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2727 = Coupling(
    name="GC_2727",
    value="-((ee*complex(0,1)*propCorr**2*complexconjugate(CKM3x1))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2728 = Coupling(
    name="GC_2728",
    value="-((cHq3*ee*complex(0,1)*vevhat*complexconjugate(CKM3x1)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_2729 = Coupling(
    name="GC_2729",
    value="-((cHq3*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2730 = Coupling(
    name="GC_2730",
    value="-((cHWB*cth*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x1))/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2731 = Coupling(
    name="GC_2731",
    value="-((cHl3*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x1))/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2732 = Coupling(
    name="GC_2732",
    value="(cll1*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x1))/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_2733 = Coupling(
    name="GC_2733",
    value="(cHDD*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x1))/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_2734 = Coupling(
    name="GC_2734",
    value="(CKM2x1*cquqd11Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2735 = Coupling(
    name="GC_2735",
    value="(CKM2x2*cquqd11Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2736 = Coupling(
    name="GC_2736",
    value="(CKM2x3*cquqd11Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2737 = Coupling(
    name="GC_2737",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2738 = Coupling(
    name="GC_2738",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2739 = Coupling(
    name="GC_2739",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2740 = Coupling(
    name="GC_2740",
    value="-((CKM2x1*cquqd1Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2741 = Coupling(
    name="GC_2741",
    value="-((CKM2x2*cquqd1Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2742 = Coupling(
    name="GC_2742",
    value="-((CKM2x3*cquqd1Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2743 = Coupling(
    name="GC_2743",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2744 = Coupling(
    name="GC_2744",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2745 = Coupling(
    name="GC_2745",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2746 = Coupling(
    name="GC_2746",
    value="(CKM2x1*cquqd81Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2747 = Coupling(
    name="GC_2747",
    value="(CKM2x2*cquqd81Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2748 = Coupling(
    name="GC_2748",
    value="(CKM2x3*cquqd81Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2749 = Coupling(
    name="GC_2749",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2750 = Coupling(
    name="GC_2750",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2751 = Coupling(
    name="GC_2751",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2752 = Coupling(
    name="GC_2752",
    value="-((CKM2x1*cquqd8Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2753 = Coupling(
    name="GC_2753",
    value="-((CKM2x2*cquqd8Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2754 = Coupling(
    name="GC_2754",
    value="-((CKM2x3*cquqd8Im*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2755 = Coupling(
    name="GC_2755",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2756 = Coupling(
    name="GC_2756",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2757 = Coupling(
    name="GC_2757",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yc*ydo*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2758 = Coupling(
    name="GC_2758",
    value="-((cledqIm*ydo*ye*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2759 = Coupling(
    name="GC_2759",
    value="(cledqRe*complex(0,1)*ydo*ye*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2760 = Coupling(
    name="GC_2760",
    value="-((cledqIm*ydo*ym*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2761 = Coupling(
    name="GC_2761",
    value="(cledqRe*complex(0,1)*ydo*ym*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2762 = Coupling(
    name="GC_2762",
    value="-((CKM1x3*cquqd11Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2763 = Coupling(
    name="GC_2763",
    value="-((CKM2x3*cquqd11Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2764 = Coupling(
    name="GC_2764",
    value="-((CKM3x3*cquqd11Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2765 = Coupling(
    name="GC_2765",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2766 = Coupling(
    name="GC_2766",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2767 = Coupling(
    name="GC_2767",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2768 = Coupling(
    name="GC_2768",
    value="(CKM1x3*cquqd1Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2769 = Coupling(
    name="GC_2769",
    value="(CKM2x3*cquqd1Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2770 = Coupling(
    name="GC_2770",
    value="(CKM3x3*cquqd1Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2771 = Coupling(
    name="GC_2771",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2772 = Coupling(
    name="GC_2772",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2773 = Coupling(
    name="GC_2773",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2774 = Coupling(
    name="GC_2774",
    value="-((CKM1x3*cquqd81Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2775 = Coupling(
    name="GC_2775",
    value="-((CKM2x3*cquqd81Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2776 = Coupling(
    name="GC_2776",
    value="-((CKM3x3*cquqd81Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2777 = Coupling(
    name="GC_2777",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2778 = Coupling(
    name="GC_2778",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2779 = Coupling(
    name="GC_2779",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2780 = Coupling(
    name="GC_2780",
    value="(CKM1x3*cquqd8Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2781 = Coupling(
    name="GC_2781",
    value="(CKM2x3*cquqd8Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2782 = Coupling(
    name="GC_2782",
    value="(CKM3x3*cquqd8Im*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2783 = Coupling(
    name="GC_2783",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2784 = Coupling(
    name="GC_2784",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2785 = Coupling(
    name="GC_2785",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2786 = Coupling(
    name="GC_2786",
    value="-((CKM1x1*cquqd11Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2787 = Coupling(
    name="GC_2787",
    value="-((CKM2x1*cquqd11Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2788 = Coupling(
    name="GC_2788",
    value="-((CKM3x1*cquqd11Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2789 = Coupling(
    name="GC_2789",
    value="(CKM3x1*cquqd11Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2790 = Coupling(
    name="GC_2790",
    value="(CKM3x2*cquqd11Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2791 = Coupling(
    name="GC_2791",
    value="(CKM3x3*cquqd11Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2792 = Coupling(
    name="GC_2792",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2793 = Coupling(
    name="GC_2793",
    value="(CKM2x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2794 = Coupling(
    name="GC_2794",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2795 = Coupling(
    name="GC_2795",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2796 = Coupling(
    name="GC_2796",
    value="(CKM3x3*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2797 = Coupling(
    name="GC_2797",
    value="(CKM1x1*cquqd1Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2798 = Coupling(
    name="GC_2798",
    value="(CKM2x1*cquqd1Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2799 = Coupling(
    name="GC_2799",
    value="-((CKM3x1*cquqd1Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2800 = Coupling(
    name="GC_2800",
    value="(CKM3x1*cquqd1Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2801 = Coupling(
    name="GC_2801",
    value="-((CKM3x2*cquqd1Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2802 = Coupling(
    name="GC_2802",
    value="-((CKM3x3*cquqd1Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2803 = Coupling(
    name="GC_2803",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2804 = Coupling(
    name="GC_2804",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2805 = Coupling(
    name="GC_2805",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2806 = Coupling(
    name="GC_2806",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2807 = Coupling(
    name="GC_2807",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2808 = Coupling(
    name="GC_2808",
    value="-((CKM1x1*cquqd81Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2809 = Coupling(
    name="GC_2809",
    value="-((CKM2x1*cquqd81Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2810 = Coupling(
    name="GC_2810",
    value="-((CKM3x1*cquqd81Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2811 = Coupling(
    name="GC_2811",
    value="(CKM3x1*cquqd81Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2812 = Coupling(
    name="GC_2812",
    value="(CKM3x2*cquqd81Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2813 = Coupling(
    name="GC_2813",
    value="(CKM3x3*cquqd81Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2814 = Coupling(
    name="GC_2814",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2815 = Coupling(
    name="GC_2815",
    value="(CKM2x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2816 = Coupling(
    name="GC_2816",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2817 = Coupling(
    name="GC_2817",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2818 = Coupling(
    name="GC_2818",
    value="(CKM3x3*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2819 = Coupling(
    name="GC_2819",
    value="(CKM1x1*cquqd8Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2820 = Coupling(
    name="GC_2820",
    value="(CKM2x1*cquqd8Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2821 = Coupling(
    name="GC_2821",
    value="-((CKM3x1*cquqd8Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2822 = Coupling(
    name="GC_2822",
    value="(CKM3x1*cquqd8Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2823 = Coupling(
    name="GC_2823",
    value="-((CKM3x2*cquqd8Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2824 = Coupling(
    name="GC_2824",
    value="-((CKM3x3*cquqd8Im*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2825 = Coupling(
    name="GC_2825",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2826 = Coupling(
    name="GC_2826",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2827 = Coupling(
    name="GC_2827",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2828 = Coupling(
    name="GC_2828",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2829 = Coupling(
    name="GC_2829",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2830 = Coupling(
    name="GC_2830",
    value="(cHudIm*ee*ydo*yt*complexconjugate(CKM3x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_2831 = Coupling(
    name="GC_2831",
    value="(cHudRe*ee*complex(0,1)*ydo*yt*complexconjugate(CKM3x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_2832 = Coupling(
    name="GC_2832",
    value="(cHudIm*ee*vevhat*ydo*yt*complexconjugate(CKM3x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_2833 = Coupling(
    name="GC_2833",
    value="(cHudRe*ee*complex(0,1)*vevhat*ydo*yt*complexconjugate(CKM3x1))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_2834 = Coupling(
    name="GC_2834",
    value="(cHudIm*ee*vevhat**2*ydo*yt*complexconjugate(CKM3x1))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_2835 = Coupling(
    name="GC_2835",
    value="(cHudRe*ee*complex(0,1)*vevhat**2*ydo*yt*complexconjugate(CKM3x1))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_2836 = Coupling(
    name="GC_2836",
    value="-((clequ1Im*ye*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2837 = Coupling(
    name="GC_2837",
    value="(clequ1Re*complex(0,1)*ye*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2838 = Coupling(
    name="GC_2838",
    value="-(clequ3Im*ye*yt*complexconjugate(CKM3x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2839 = Coupling(
    name="GC_2839",
    value="(clequ3Im*ye*yt*complexconjugate(CKM3x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2840 = Coupling(
    name="GC_2840",
    value="(clequ3Re*complex(0,1)*ye*yt*complexconjugate(CKM3x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2841 = Coupling(
    name="GC_2841",
    value="-(clequ3Re*complex(0,1)*ye*yt*complexconjugate(CKM3x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2842 = Coupling(
    name="GC_2842",
    value="-((clequ1Im*ym*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2843 = Coupling(
    name="GC_2843",
    value="(clequ1Re*complex(0,1)*ym*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2844 = Coupling(
    name="GC_2844",
    value="-(clequ3Im*ym*yt*complexconjugate(CKM3x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2845 = Coupling(
    name="GC_2845",
    value="(clequ3Im*ym*yt*complexconjugate(CKM3x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2846 = Coupling(
    name="GC_2846",
    value="(clequ3Re*complex(0,1)*ym*yt*complexconjugate(CKM3x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2847 = Coupling(
    name="GC_2847",
    value="-(clequ3Re*complex(0,1)*ym*yt*complexconjugate(CKM3x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2848 = Coupling(
    name="GC_2848",
    value="-((CKM1x2*cquqd11Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2849 = Coupling(
    name="GC_2849",
    value="-((CKM2x2*cquqd11Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2850 = Coupling(
    name="GC_2850",
    value="-((CKM3x2*cquqd11Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2851 = Coupling(
    name="GC_2851",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2852 = Coupling(
    name="GC_2852",
    value="(CKM2x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2853 = Coupling(
    name="GC_2853",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2854 = Coupling(
    name="GC_2854",
    value="(CKM1x2*cquqd1Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2855 = Coupling(
    name="GC_2855",
    value="(CKM2x2*cquqd1Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2856 = Coupling(
    name="GC_2856",
    value="(CKM3x2*cquqd1Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2857 = Coupling(
    name="GC_2857",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2858 = Coupling(
    name="GC_2858",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2859 = Coupling(
    name="GC_2859",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2860 = Coupling(
    name="GC_2860",
    value="-((CKM1x2*cquqd81Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2861 = Coupling(
    name="GC_2861",
    value="-((CKM2x2*cquqd81Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2862 = Coupling(
    name="GC_2862",
    value="-((CKM3x2*cquqd81Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2863 = Coupling(
    name="GC_2863",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2864 = Coupling(
    name="GC_2864",
    value="(CKM2x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2865 = Coupling(
    name="GC_2865",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2866 = Coupling(
    name="GC_2866",
    value="(CKM1x2*cquqd8Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2867 = Coupling(
    name="GC_2867",
    value="(CKM2x2*cquqd8Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2868 = Coupling(
    name="GC_2868",
    value="(CKM3x2*cquqd8Im*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2869 = Coupling(
    name="GC_2869",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2870 = Coupling(
    name="GC_2870",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2871 = Coupling(
    name="GC_2871",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2872 = Coupling(
    name="GC_2872",
    value="-((cledqIm*ydo*ytau*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2873 = Coupling(
    name="GC_2873",
    value="(cledqRe*complex(0,1)*ydo*ytau*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2874 = Coupling(
    name="GC_2874",
    value="-((clequ1Im*yt*ytau*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2875 = Coupling(
    name="GC_2875",
    value="(clequ1Re*complex(0,1)*yt*ytau*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2876 = Coupling(
    name="GC_2876",
    value="-(clequ3Im*yt*ytau*complexconjugate(CKM3x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2877 = Coupling(
    name="GC_2877",
    value="(clequ3Im*yt*ytau*complexconjugate(CKM3x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2878 = Coupling(
    name="GC_2878",
    value="(clequ3Re*complex(0,1)*yt*ytau*complexconjugate(CKM3x1))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2879 = Coupling(
    name="GC_2879",
    value="-(clequ3Re*complex(0,1)*yt*ytau*complexconjugate(CKM3x1))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2880 = Coupling(
    name="GC_2880",
    value="(CKM1x1*cquqd11Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2881 = Coupling(
    name="GC_2881",
    value="(CKM1x2*cquqd11Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2882 = Coupling(
    name="GC_2882",
    value="(CKM1x3*cquqd11Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2883 = Coupling(
    name="GC_2883",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2884 = Coupling(
    name="GC_2884",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2885 = Coupling(
    name="GC_2885",
    value="(CKM1x3*cquqd11Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2886 = Coupling(
    name="GC_2886",
    value="-((CKM1x1*cquqd1Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2887 = Coupling(
    name="GC_2887",
    value="-((CKM1x2*cquqd1Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2888 = Coupling(
    name="GC_2888",
    value="-((CKM1x3*cquqd1Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2889 = Coupling(
    name="GC_2889",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2890 = Coupling(
    name="GC_2890",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2891 = Coupling(
    name="GC_2891",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2892 = Coupling(
    name="GC_2892",
    value="(CKM1x1*cquqd81Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2893 = Coupling(
    name="GC_2893",
    value="(CKM1x2*cquqd81Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2894 = Coupling(
    name="GC_2894",
    value="(CKM1x3*cquqd81Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2895 = Coupling(
    name="GC_2895",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2896 = Coupling(
    name="GC_2896",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2897 = Coupling(
    name="GC_2897",
    value="(CKM1x3*cquqd81Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2898 = Coupling(
    name="GC_2898",
    value="-((CKM1x1*cquqd8Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2899 = Coupling(
    name="GC_2899",
    value="-((CKM1x2*cquqd8Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2900 = Coupling(
    name="GC_2900",
    value="-((CKM1x3*cquqd8Im*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2901 = Coupling(
    name="GC_2901",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2902 = Coupling(
    name="GC_2902",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2903 = Coupling(
    name="GC_2903",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*ydo*yup*complexconjugate(CKM3x1))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2904 = Coupling(
    name="GC_2904",
    value="(2*clq3*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2905 = Coupling(
    name="GC_2905",
    value="(2*CKM1x1*cqq11*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2906 = Coupling(
    name="GC_2906",
    value="(2*CKM1x2*cqq11*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2907 = Coupling(
    name="GC_2907",
    value="(2*CKM1x3*cqq11*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2908 = Coupling(
    name="GC_2908",
    value="(2*CKM2x1*cqq11*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2909 = Coupling(
    name="GC_2909",
    value="(2*CKM2x2*cqq11*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2910 = Coupling(
    name="GC_2910",
    value="(2*CKM2x3*cqq11*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2911 = Coupling(
    name="GC_2911",
    value="(2*CKM3x1*cqq11*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2912 = Coupling(
    name="GC_2912",
    value="(2*CKM3x2*cqq11*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2913 = Coupling(
    name="GC_2913",
    value="(2*CKM3x3*cqq11*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2914 = Coupling(
    name="GC_2914",
    value="(4*CKM1x1*cqq3*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2915 = Coupling(
    name="GC_2915",
    value="(4*CKM1x2*cqq3*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2916 = Coupling(
    name="GC_2916",
    value="(4*CKM1x3*cqq3*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2917 = Coupling(
    name="GC_2917",
    value="(4*CKM2x1*cqq3*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2918 = Coupling(
    name="GC_2918",
    value="(4*CKM2x2*cqq3*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2919 = Coupling(
    name="GC_2919",
    value="(4*CKM2x3*cqq3*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2920 = Coupling(
    name="GC_2920",
    value="(4*CKM3x1*cqq3*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2921 = Coupling(
    name="GC_2921",
    value="(4*CKM3x2*cqq3*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2922 = Coupling(
    name="GC_2922",
    value="(4*CKM3x3*cqq3*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2923 = Coupling(
    name="GC_2923",
    value="(-2*CKM1x1*cqq31*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2924 = Coupling(
    name="GC_2924",
    value="(-2*CKM1x2*cqq31*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2925 = Coupling(
    name="GC_2925",
    value="(-2*CKM1x3*cqq31*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2926 = Coupling(
    name="GC_2926",
    value="(-2*CKM2x1*cqq31*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2927 = Coupling(
    name="GC_2927",
    value="(-2*CKM2x2*cqq31*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2928 = Coupling(
    name="GC_2928",
    value="(-2*CKM2x3*cqq31*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2929 = Coupling(
    name="GC_2929",
    value="(-2*CKM3x1*cqq31*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2930 = Coupling(
    name="GC_2930",
    value="(-2*CKM3x2*cqq31*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2931 = Coupling(
    name="GC_2931",
    value="(-2*CKM3x3*cqq31*complex(0,1)*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_2932 = Coupling(
    name="GC_2932",
    value="-((ee*complex(0,1)*complexconjugate(CKM3x2))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2933 = Coupling(
    name="GC_2933",
    value="-((cHq3*ee*complex(0,1)*complexconjugate(CKM3x2)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_2934 = Coupling(
    name="GC_2934",
    value="-((ee*complex(0,1)*propCorr*complexconjugate(CKM3x2))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2935 = Coupling(
    name="GC_2935",
    value="-((ee*complex(0,1)*propCorr**2*complexconjugate(CKM3x2))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_2936 = Coupling(
    name="GC_2936",
    value="-((cHq3*ee*complex(0,1)*vevhat*complexconjugate(CKM3x2)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_2937 = Coupling(
    name="GC_2937",
    value="-((cHq3*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2938 = Coupling(
    name="GC_2938",
    value="-((cHWB*cth*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x2))/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2939 = Coupling(
    name="GC_2939",
    value="-((cHl3*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x2))/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_2940 = Coupling(
    name="GC_2940",
    value="(cll1*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x2))/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_2941 = Coupling(
    name="GC_2941",
    value="(cHDD*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x2))/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_2942 = Coupling(
    name="GC_2942",
    value="(CKM2x1*cquqd11Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2943 = Coupling(
    name="GC_2943",
    value="(CKM2x2*cquqd11Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2944 = Coupling(
    name="GC_2944",
    value="(CKM2x3*cquqd11Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2945 = Coupling(
    name="GC_2945",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2946 = Coupling(
    name="GC_2946",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2947 = Coupling(
    name="GC_2947",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2948 = Coupling(
    name="GC_2948",
    value="-((CKM2x1*cquqd1Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2949 = Coupling(
    name="GC_2949",
    value="-((CKM2x2*cquqd1Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2950 = Coupling(
    name="GC_2950",
    value="-((CKM2x3*cquqd1Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2951 = Coupling(
    name="GC_2951",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2952 = Coupling(
    name="GC_2952",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2953 = Coupling(
    name="GC_2953",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2954 = Coupling(
    name="GC_2954",
    value="(CKM2x1*cquqd81Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2955 = Coupling(
    name="GC_2955",
    value="(CKM2x2*cquqd81Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2956 = Coupling(
    name="GC_2956",
    value="(CKM2x3*cquqd81Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2957 = Coupling(
    name="GC_2957",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2958 = Coupling(
    name="GC_2958",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2959 = Coupling(
    name="GC_2959",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2960 = Coupling(
    name="GC_2960",
    value="-((CKM2x1*cquqd8Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2961 = Coupling(
    name="GC_2961",
    value="-((CKM2x2*cquqd8Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2962 = Coupling(
    name="GC_2962",
    value="-((CKM2x3*cquqd8Im*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2963 = Coupling(
    name="GC_2963",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2964 = Coupling(
    name="GC_2964",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2965 = Coupling(
    name="GC_2965",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yc*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2966 = Coupling(
    name="GC_2966",
    value="-((cledqIm*ye*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2967 = Coupling(
    name="GC_2967",
    value="(cledqRe*complex(0,1)*ye*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2968 = Coupling(
    name="GC_2968",
    value="-((cledqIm*ym*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2969 = Coupling(
    name="GC_2969",
    value="(cledqRe*complex(0,1)*ym*ys*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2970 = Coupling(
    name="GC_2970",
    value="-((CKM1x3*cquqd11Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2971 = Coupling(
    name="GC_2971",
    value="-((CKM2x3*cquqd11Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2972 = Coupling(
    name="GC_2972",
    value="-((CKM3x3*cquqd11Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2973 = Coupling(
    name="GC_2973",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2974 = Coupling(
    name="GC_2974",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2975 = Coupling(
    name="GC_2975",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2976 = Coupling(
    name="GC_2976",
    value="(CKM1x3*cquqd1Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2977 = Coupling(
    name="GC_2977",
    value="(CKM2x3*cquqd1Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2978 = Coupling(
    name="GC_2978",
    value="(CKM3x3*cquqd1Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2979 = Coupling(
    name="GC_2979",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2980 = Coupling(
    name="GC_2980",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2981 = Coupling(
    name="GC_2981",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2982 = Coupling(
    name="GC_2982",
    value="-((CKM1x3*cquqd81Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2983 = Coupling(
    name="GC_2983",
    value="-((CKM2x3*cquqd81Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2984 = Coupling(
    name="GC_2984",
    value="-((CKM3x3*cquqd81Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2985 = Coupling(
    name="GC_2985",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2986 = Coupling(
    name="GC_2986",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2987 = Coupling(
    name="GC_2987",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2988 = Coupling(
    name="GC_2988",
    value="(CKM1x3*cquqd8Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2989 = Coupling(
    name="GC_2989",
    value="(CKM2x3*cquqd8Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2990 = Coupling(
    name="GC_2990",
    value="(CKM3x3*cquqd8Im*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2991 = Coupling(
    name="GC_2991",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2992 = Coupling(
    name="GC_2992",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2993 = Coupling(
    name="GC_2993",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2994 = Coupling(
    name="GC_2994",
    value="-((CKM1x1*cquqd11Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2995 = Coupling(
    name="GC_2995",
    value="-((CKM2x1*cquqd11Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2996 = Coupling(
    name="GC_2996",
    value="-((CKM3x1*cquqd11Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_2997 = Coupling(
    name="GC_2997",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2998 = Coupling(
    name="GC_2998",
    value="(CKM2x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_2999 = Coupling(
    name="GC_2999",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3000 = Coupling(
    name="GC_3000",
    value="(CKM1x1*cquqd1Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3001 = Coupling(
    name="GC_3001",
    value="(CKM2x1*cquqd1Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3002 = Coupling(
    name="GC_3002",
    value="(CKM3x1*cquqd1Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3003 = Coupling(
    name="GC_3003",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3004 = Coupling(
    name="GC_3004",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3005 = Coupling(
    name="GC_3005",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3006 = Coupling(
    name="GC_3006",
    value="-((CKM1x1*cquqd81Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3007 = Coupling(
    name="GC_3007",
    value="-((CKM2x1*cquqd81Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3008 = Coupling(
    name="GC_3008",
    value="-((CKM3x1*cquqd81Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3009 = Coupling(
    name="GC_3009",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3010 = Coupling(
    name="GC_3010",
    value="(CKM2x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3011 = Coupling(
    name="GC_3011",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3012 = Coupling(
    name="GC_3012",
    value="(CKM1x1*cquqd8Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3013 = Coupling(
    name="GC_3013",
    value="(CKM2x1*cquqd8Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3014 = Coupling(
    name="GC_3014",
    value="(CKM3x1*cquqd8Im*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3015 = Coupling(
    name="GC_3015",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3016 = Coupling(
    name="GC_3016",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3017 = Coupling(
    name="GC_3017",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3018 = Coupling(
    name="GC_3018",
    value="-((clequ1Im*ye*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3019 = Coupling(
    name="GC_3019",
    value="(clequ1Re*complex(0,1)*ye*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3020 = Coupling(
    name="GC_3020",
    value="-(clequ3Im*ye*yt*complexconjugate(CKM3x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3021 = Coupling(
    name="GC_3021",
    value="(clequ3Im*ye*yt*complexconjugate(CKM3x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3022 = Coupling(
    name="GC_3022",
    value="(clequ3Re*complex(0,1)*ye*yt*complexconjugate(CKM3x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3023 = Coupling(
    name="GC_3023",
    value="-(clequ3Re*complex(0,1)*ye*yt*complexconjugate(CKM3x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3024 = Coupling(
    name="GC_3024",
    value="-((clequ1Im*ym*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3025 = Coupling(
    name="GC_3025",
    value="(clequ1Re*complex(0,1)*ym*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3026 = Coupling(
    name="GC_3026",
    value="-(clequ3Im*ym*yt*complexconjugate(CKM3x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3027 = Coupling(
    name="GC_3027",
    value="(clequ3Im*ym*yt*complexconjugate(CKM3x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3028 = Coupling(
    name="GC_3028",
    value="(clequ3Re*complex(0,1)*ym*yt*complexconjugate(CKM3x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3029 = Coupling(
    name="GC_3029",
    value="-(clequ3Re*complex(0,1)*ym*yt*complexconjugate(CKM3x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3030 = Coupling(
    name="GC_3030",
    value="-((CKM1x2*cquqd11Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3031 = Coupling(
    name="GC_3031",
    value="-((CKM2x2*cquqd11Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3032 = Coupling(
    name="GC_3032",
    value="(CKM3x1*cquqd11Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3033 = Coupling(
    name="GC_3033",
    value="-((CKM3x2*cquqd11Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3034 = Coupling(
    name="GC_3034",
    value="(CKM3x2*cquqd11Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3035 = Coupling(
    name="GC_3035",
    value="(CKM3x3*cquqd11Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3036 = Coupling(
    name="GC_3036",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3037 = Coupling(
    name="GC_3037",
    value="(CKM2x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3038 = Coupling(
    name="GC_3038",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3039 = Coupling(
    name="GC_3039",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3040 = Coupling(
    name="GC_3040",
    value="(CKM3x3*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3041 = Coupling(
    name="GC_3041",
    value="(CKM1x2*cquqd1Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3042 = Coupling(
    name="GC_3042",
    value="(CKM2x2*cquqd1Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3043 = Coupling(
    name="GC_3043",
    value="-((CKM3x1*cquqd1Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3044 = Coupling(
    name="GC_3044",
    value="-((CKM3x2*cquqd1Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3045 = Coupling(
    name="GC_3045",
    value="(CKM3x2*cquqd1Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3046 = Coupling(
    name="GC_3046",
    value="-((CKM3x3*cquqd1Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3047 = Coupling(
    name="GC_3047",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3048 = Coupling(
    name="GC_3048",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3049 = Coupling(
    name="GC_3049",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3050 = Coupling(
    name="GC_3050",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3051 = Coupling(
    name="GC_3051",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3052 = Coupling(
    name="GC_3052",
    value="-((CKM1x2*cquqd81Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3053 = Coupling(
    name="GC_3053",
    value="-((CKM2x2*cquqd81Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3054 = Coupling(
    name="GC_3054",
    value="(CKM3x1*cquqd81Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3055 = Coupling(
    name="GC_3055",
    value="-((CKM3x2*cquqd81Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3056 = Coupling(
    name="GC_3056",
    value="(CKM3x2*cquqd81Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3057 = Coupling(
    name="GC_3057",
    value="(CKM3x3*cquqd81Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3058 = Coupling(
    name="GC_3058",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3059 = Coupling(
    name="GC_3059",
    value="(CKM2x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3060 = Coupling(
    name="GC_3060",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3061 = Coupling(
    name="GC_3061",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3062 = Coupling(
    name="GC_3062",
    value="(CKM3x3*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3063 = Coupling(
    name="GC_3063",
    value="(CKM1x2*cquqd8Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3064 = Coupling(
    name="GC_3064",
    value="(CKM2x2*cquqd8Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3065 = Coupling(
    name="GC_3065",
    value="-((CKM3x1*cquqd8Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3066 = Coupling(
    name="GC_3066",
    value="-((CKM3x2*cquqd8Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3067 = Coupling(
    name="GC_3067",
    value="(CKM3x2*cquqd8Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3068 = Coupling(
    name="GC_3068",
    value="-((CKM3x3*cquqd8Im*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3069 = Coupling(
    name="GC_3069",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3070 = Coupling(
    name="GC_3070",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3071 = Coupling(
    name="GC_3071",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3072 = Coupling(
    name="GC_3072",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3073 = Coupling(
    name="GC_3073",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3074 = Coupling(
    name="GC_3074",
    value="(cHudIm*ee*ys*yt*complexconjugate(CKM3x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_3075 = Coupling(
    name="GC_3075",
    value="(cHudRe*ee*complex(0,1)*ys*yt*complexconjugate(CKM3x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_3076 = Coupling(
    name="GC_3076",
    value="(cHudIm*ee*vevhat*ys*yt*complexconjugate(CKM3x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_3077 = Coupling(
    name="GC_3077",
    value="(cHudRe*ee*complex(0,1)*vevhat*ys*yt*complexconjugate(CKM3x2))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_3078 = Coupling(
    name="GC_3078",
    value="(cHudIm*ee*vevhat**2*ys*yt*complexconjugate(CKM3x2))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_3079 = Coupling(
    name="GC_3079",
    value="(cHudRe*ee*complex(0,1)*vevhat**2*ys*yt*complexconjugate(CKM3x2))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_3080 = Coupling(
    name="GC_3080",
    value="-((cledqIm*ys*ytau*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3081 = Coupling(
    name="GC_3081",
    value="(cledqRe*complex(0,1)*ys*ytau*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3082 = Coupling(
    name="GC_3082",
    value="-((clequ1Im*yt*ytau*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3083 = Coupling(
    name="GC_3083",
    value="(clequ1Re*complex(0,1)*yt*ytau*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3084 = Coupling(
    name="GC_3084",
    value="-(clequ3Im*yt*ytau*complexconjugate(CKM3x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3085 = Coupling(
    name="GC_3085",
    value="(clequ3Im*yt*ytau*complexconjugate(CKM3x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3086 = Coupling(
    name="GC_3086",
    value="(clequ3Re*complex(0,1)*yt*ytau*complexconjugate(CKM3x2))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3087 = Coupling(
    name="GC_3087",
    value="-(clequ3Re*complex(0,1)*yt*ytau*complexconjugate(CKM3x2))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3088 = Coupling(
    name="GC_3088",
    value="(CKM1x1*cquqd11Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3089 = Coupling(
    name="GC_3089",
    value="(CKM1x2*cquqd11Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3090 = Coupling(
    name="GC_3090",
    value="(CKM1x3*cquqd11Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3091 = Coupling(
    name="GC_3091",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3092 = Coupling(
    name="GC_3092",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3093 = Coupling(
    name="GC_3093",
    value="(CKM1x3*cquqd11Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3094 = Coupling(
    name="GC_3094",
    value="-((CKM1x1*cquqd1Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3095 = Coupling(
    name="GC_3095",
    value="-((CKM1x2*cquqd1Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3096 = Coupling(
    name="GC_3096",
    value="-((CKM1x3*cquqd1Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3097 = Coupling(
    name="GC_3097",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3098 = Coupling(
    name="GC_3098",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3099 = Coupling(
    name="GC_3099",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3100 = Coupling(
    name="GC_3100",
    value="(CKM1x1*cquqd81Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3101 = Coupling(
    name="GC_3101",
    value="(CKM1x2*cquqd81Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3102 = Coupling(
    name="GC_3102",
    value="(CKM1x3*cquqd81Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3103 = Coupling(
    name="GC_3103",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3104 = Coupling(
    name="GC_3104",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3105 = Coupling(
    name="GC_3105",
    value="(CKM1x3*cquqd81Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3106 = Coupling(
    name="GC_3106",
    value="-((CKM1x1*cquqd8Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3107 = Coupling(
    name="GC_3107",
    value="-((CKM1x2*cquqd8Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3108 = Coupling(
    name="GC_3108",
    value="-((CKM1x3*cquqd8Im*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3109 = Coupling(
    name="GC_3109",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3110 = Coupling(
    name="GC_3110",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3111 = Coupling(
    name="GC_3111",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*ys*yup*complexconjugate(CKM3x2))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3112 = Coupling(
    name="GC_3112",
    value="(2*clq3*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3113 = Coupling(
    name="GC_3113",
    value="(2*CKM1x1*cqq11*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3114 = Coupling(
    name="GC_3114",
    value="(2*CKM1x2*cqq11*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3115 = Coupling(
    name="GC_3115",
    value="(2*CKM1x3*cqq11*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3116 = Coupling(
    name="GC_3116",
    value="(2*CKM2x1*cqq11*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3117 = Coupling(
    name="GC_3117",
    value="(2*CKM2x2*cqq11*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3118 = Coupling(
    name="GC_3118",
    value="(2*CKM2x3*cqq11*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3119 = Coupling(
    name="GC_3119",
    value="(2*CKM3x1*cqq11*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3120 = Coupling(
    name="GC_3120",
    value="(2*CKM3x2*cqq11*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3121 = Coupling(
    name="GC_3121",
    value="(2*CKM3x3*cqq11*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3122 = Coupling(
    name="GC_3122",
    value="(4*CKM1x1*cqq3*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3123 = Coupling(
    name="GC_3123",
    value="(4*CKM1x2*cqq3*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3124 = Coupling(
    name="GC_3124",
    value="(4*CKM1x3*cqq3*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3125 = Coupling(
    name="GC_3125",
    value="(4*CKM2x1*cqq3*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3126 = Coupling(
    name="GC_3126",
    value="(4*CKM2x2*cqq3*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3127 = Coupling(
    name="GC_3127",
    value="(4*CKM2x3*cqq3*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3128 = Coupling(
    name="GC_3128",
    value="(4*CKM3x1*cqq3*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3129 = Coupling(
    name="GC_3129",
    value="(4*CKM3x2*cqq3*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3130 = Coupling(
    name="GC_3130",
    value="(4*CKM3x3*cqq3*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3131 = Coupling(
    name="GC_3131",
    value="(-2*CKM1x1*cqq31*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3132 = Coupling(
    name="GC_3132",
    value="(-2*CKM1x2*cqq31*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3133 = Coupling(
    name="GC_3133",
    value="(-2*CKM1x3*cqq31*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3134 = Coupling(
    name="GC_3134",
    value="(-2*CKM2x1*cqq31*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3135 = Coupling(
    name="GC_3135",
    value="(-2*CKM2x2*cqq31*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3136 = Coupling(
    name="GC_3136",
    value="(-2*CKM2x3*cqq31*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3137 = Coupling(
    name="GC_3137",
    value="(-2*CKM3x1*cqq31*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3138 = Coupling(
    name="GC_3138",
    value="(-2*CKM3x2*cqq31*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3139 = Coupling(
    name="GC_3139",
    value="(-2*CKM3x3*cqq31*complex(0,1)*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 2},
)

GC_3140 = Coupling(
    name="GC_3140",
    value="-((ee*complex(0,1)*complexconjugate(CKM3x3))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_3141 = Coupling(
    name="GC_3141",
    value="-((cHq3*ee*complex(0,1)*complexconjugate(CKM3x3)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 3},
)

GC_3142 = Coupling(
    name="GC_3142",
    value="-((ee*complex(0,1)*propCorr*complexconjugate(CKM3x3))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_3143 = Coupling(
    name="GC_3143",
    value="-((ee*complex(0,1)*propCorr**2*complexconjugate(CKM3x3))/(sth*cmath.sqrt(2)))",
    order={"QED": 1},
)

GC_3144 = Coupling(
    name="GC_3144",
    value="-((cHq3*ee*complex(0,1)*vevhat*complexconjugate(CKM3x3)*cmath.sqrt(2))/(LambdaSMEFT**2*sth))",
    order={"NP": 1, "QED": 2},
)

GC_3145 = Coupling(
    name="GC_3145",
    value="-((cHq3*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_3146 = Coupling(
    name="GC_3146",
    value="-((cHWB*cth*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x3))/(LambdaSMEFT**2*(-1 + 2*sth**2)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_3147 = Coupling(
    name="GC_3147",
    value="-((cHl3*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x3))/(LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2)))",
    order={"NP": 1, "QED": 1},
)

GC_3148 = Coupling(
    name="GC_3148",
    value="(cll1*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x3))/(2.*LambdaSMEFT**2*(-sth + 2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_3149 = Coupling(
    name="GC_3149",
    value="(cHDD*cth**2*ee*complex(0,1)*vevhat**2*complexconjugate(CKM3x3))/((4*LambdaSMEFT**2*sth - 8*LambdaSMEFT**2*sth**3)*cmath.sqrt(2))",
    order={"NP": 1, "QED": 1},
)

GC_3150 = Coupling(
    name="GC_3150",
    value="(CKM2x1*cquqd11Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3151 = Coupling(
    name="GC_3151",
    value="(CKM2x2*cquqd11Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3152 = Coupling(
    name="GC_3152",
    value="(CKM2x3*cquqd11Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3153 = Coupling(
    name="GC_3153",
    value="(CKM2x1*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3154 = Coupling(
    name="GC_3154",
    value="(CKM2x2*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3155 = Coupling(
    name="GC_3155",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3156 = Coupling(
    name="GC_3156",
    value="-((CKM2x1*cquqd1Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3157 = Coupling(
    name="GC_3157",
    value="-((CKM2x2*cquqd1Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3158 = Coupling(
    name="GC_3158",
    value="-((CKM2x3*cquqd1Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3159 = Coupling(
    name="GC_3159",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3160 = Coupling(
    name="GC_3160",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3161 = Coupling(
    name="GC_3161",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3162 = Coupling(
    name="GC_3162",
    value="(CKM2x1*cquqd81Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3163 = Coupling(
    name="GC_3163",
    value="(CKM2x2*cquqd81Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3164 = Coupling(
    name="GC_3164",
    value="(CKM2x3*cquqd81Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3165 = Coupling(
    name="GC_3165",
    value="(CKM2x1*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3166 = Coupling(
    name="GC_3166",
    value="(CKM2x2*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3167 = Coupling(
    name="GC_3167",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3168 = Coupling(
    name="GC_3168",
    value="-((CKM2x1*cquqd8Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3169 = Coupling(
    name="GC_3169",
    value="-((CKM2x2*cquqd8Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3170 = Coupling(
    name="GC_3170",
    value="-((CKM2x3*cquqd8Im*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3171 = Coupling(
    name="GC_3171",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3172 = Coupling(
    name="GC_3172",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3173 = Coupling(
    name="GC_3173",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yc*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3174 = Coupling(
    name="GC_3174",
    value="-((cledqIm*yb*ye*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3175 = Coupling(
    name="GC_3175",
    value="(cledqRe*complex(0,1)*yb*ye*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3176 = Coupling(
    name="GC_3176",
    value="-((cledqIm*yb*ym*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3177 = Coupling(
    name="GC_3177",
    value="(cledqRe*complex(0,1)*yb*ym*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3178 = Coupling(
    name="GC_3178",
    value="-((CKM1x3*cquqd11Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3179 = Coupling(
    name="GC_3179",
    value="-((CKM2x3*cquqd11Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3180 = Coupling(
    name="GC_3180",
    value="(CKM3x1*cquqd11Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3181 = Coupling(
    name="GC_3181",
    value="(CKM3x2*cquqd11Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3182 = Coupling(
    name="GC_3182",
    value="-((CKM3x3*cquqd11Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3183 = Coupling(
    name="GC_3183",
    value="(CKM3x3*cquqd11Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3184 = Coupling(
    name="GC_3184",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3185 = Coupling(
    name="GC_3185",
    value="(CKM2x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3186 = Coupling(
    name="GC_3186",
    value="(CKM3x1*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3187 = Coupling(
    name="GC_3187",
    value="(CKM3x2*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3188 = Coupling(
    name="GC_3188",
    value="(CKM3x3*cquqd11Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3189 = Coupling(
    name="GC_3189",
    value="(CKM1x3*cquqd1Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3190 = Coupling(
    name="GC_3190",
    value="(CKM2x3*cquqd1Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3191 = Coupling(
    name="GC_3191",
    value="-((CKM3x1*cquqd1Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3192 = Coupling(
    name="GC_3192",
    value="-((CKM3x2*cquqd1Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3193 = Coupling(
    name="GC_3193",
    value="-((CKM3x3*cquqd1Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3194 = Coupling(
    name="GC_3194",
    value="(CKM3x3*cquqd1Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3195 = Coupling(
    name="GC_3195",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3196 = Coupling(
    name="GC_3196",
    value="-((CKM2x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3197 = Coupling(
    name="GC_3197",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3198 = Coupling(
    name="GC_3198",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3199 = Coupling(
    name="GC_3199",
    value="-((CKM3x3*cquqd1Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3200 = Coupling(
    name="GC_3200",
    value="-((CKM1x3*cquqd81Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3201 = Coupling(
    name="GC_3201",
    value="-((CKM2x3*cquqd81Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3202 = Coupling(
    name="GC_3202",
    value="(CKM3x1*cquqd81Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3203 = Coupling(
    name="GC_3203",
    value="(CKM3x2*cquqd81Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3204 = Coupling(
    name="GC_3204",
    value="-((CKM3x3*cquqd81Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3205 = Coupling(
    name="GC_3205",
    value="(CKM3x3*cquqd81Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3206 = Coupling(
    name="GC_3206",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3207 = Coupling(
    name="GC_3207",
    value="(CKM2x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3208 = Coupling(
    name="GC_3208",
    value="(CKM3x1*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3209 = Coupling(
    name="GC_3209",
    value="(CKM3x2*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3210 = Coupling(
    name="GC_3210",
    value="(CKM3x3*cquqd81Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3211 = Coupling(
    name="GC_3211",
    value="(CKM1x3*cquqd8Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3212 = Coupling(
    name="GC_3212",
    value="(CKM2x3*cquqd8Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3213 = Coupling(
    name="GC_3213",
    value="-((CKM3x1*cquqd8Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3214 = Coupling(
    name="GC_3214",
    value="-((CKM3x2*cquqd8Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3215 = Coupling(
    name="GC_3215",
    value="-((CKM3x3*cquqd8Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3216 = Coupling(
    name="GC_3216",
    value="(CKM3x3*cquqd8Im*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3217 = Coupling(
    name="GC_3217",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3218 = Coupling(
    name="GC_3218",
    value="-((CKM2x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3219 = Coupling(
    name="GC_3219",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3220 = Coupling(
    name="GC_3220",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3221 = Coupling(
    name="GC_3221",
    value="-((CKM3x3*cquqd8Re*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3222 = Coupling(
    name="GC_3222",
    value="(cHudIm*ee*yb*yt*complexconjugate(CKM3x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_3223 = Coupling(
    name="GC_3223",
    value="(cHudRe*ee*complex(0,1)*yb*yt*complexconjugate(CKM3x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 5},
)

GC_3224 = Coupling(
    name="GC_3224",
    value="(cHudIm*ee*vevhat*yb*yt*complexconjugate(CKM3x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_3225 = Coupling(
    name="GC_3225",
    value="(cHudRe*ee*complex(0,1)*vevhat*yb*yt*complexconjugate(CKM3x3))/(LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 4},
)

GC_3226 = Coupling(
    name="GC_3226",
    value="(cHudIm*ee*vevhat**2*yb*yt*complexconjugate(CKM3x3))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_3227 = Coupling(
    name="GC_3227",
    value="(cHudRe*ee*complex(0,1)*vevhat**2*yb*yt*complexconjugate(CKM3x3))/(2.*LambdaSMEFT**2*sth*cmath.sqrt(2))",
    order={"NP": 1, "QED": 3},
)

GC_3228 = Coupling(
    name="GC_3228",
    value="-((CKM1x1*cquqd11Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3229 = Coupling(
    name="GC_3229",
    value="-((CKM2x1*cquqd11Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3230 = Coupling(
    name="GC_3230",
    value="-((CKM3x1*cquqd11Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3231 = Coupling(
    name="GC_3231",
    value="(CKM1x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3232 = Coupling(
    name="GC_3232",
    value="(CKM2x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3233 = Coupling(
    name="GC_3233",
    value="(CKM3x1*cquqd11Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3234 = Coupling(
    name="GC_3234",
    value="(CKM1x1*cquqd1Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3235 = Coupling(
    name="GC_3235",
    value="(CKM2x1*cquqd1Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3236 = Coupling(
    name="GC_3236",
    value="(CKM3x1*cquqd1Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3237 = Coupling(
    name="GC_3237",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3238 = Coupling(
    name="GC_3238",
    value="-((CKM2x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3239 = Coupling(
    name="GC_3239",
    value="-((CKM3x1*cquqd1Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3240 = Coupling(
    name="GC_3240",
    value="-((CKM1x1*cquqd81Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3241 = Coupling(
    name="GC_3241",
    value="-((CKM2x1*cquqd81Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3242 = Coupling(
    name="GC_3242",
    value="-((CKM3x1*cquqd81Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3243 = Coupling(
    name="GC_3243",
    value="(CKM1x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3244 = Coupling(
    name="GC_3244",
    value="(CKM2x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3245 = Coupling(
    name="GC_3245",
    value="(CKM3x1*cquqd81Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3246 = Coupling(
    name="GC_3246",
    value="(CKM1x1*cquqd8Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3247 = Coupling(
    name="GC_3247",
    value="(CKM2x1*cquqd8Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3248 = Coupling(
    name="GC_3248",
    value="(CKM3x1*cquqd8Im*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3249 = Coupling(
    name="GC_3249",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3250 = Coupling(
    name="GC_3250",
    value="-((CKM2x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3251 = Coupling(
    name="GC_3251",
    value="-((CKM3x1*cquqd8Re*complex(0,1)*ydo*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3252 = Coupling(
    name="GC_3252",
    value="-((clequ1Im*ye*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3253 = Coupling(
    name="GC_3253",
    value="(clequ1Re*complex(0,1)*ye*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3254 = Coupling(
    name="GC_3254",
    value="-(clequ3Im*ye*yt*complexconjugate(CKM3x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3255 = Coupling(
    name="GC_3255",
    value="(clequ3Im*ye*yt*complexconjugate(CKM3x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3256 = Coupling(
    name="GC_3256",
    value="(clequ3Re*complex(0,1)*ye*yt*complexconjugate(CKM3x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3257 = Coupling(
    name="GC_3257",
    value="-(clequ3Re*complex(0,1)*ye*yt*complexconjugate(CKM3x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3258 = Coupling(
    name="GC_3258",
    value="-((clequ1Im*ym*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3259 = Coupling(
    name="GC_3259",
    value="(clequ1Re*complex(0,1)*ym*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3260 = Coupling(
    name="GC_3260",
    value="-(clequ3Im*ym*yt*complexconjugate(CKM3x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3261 = Coupling(
    name="GC_3261",
    value="(clequ3Im*ym*yt*complexconjugate(CKM3x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3262 = Coupling(
    name="GC_3262",
    value="(clequ3Re*complex(0,1)*ym*yt*complexconjugate(CKM3x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3263 = Coupling(
    name="GC_3263",
    value="-(clequ3Re*complex(0,1)*ym*yt*complexconjugate(CKM3x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3264 = Coupling(
    name="GC_3264",
    value="-((CKM1x2*cquqd11Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3265 = Coupling(
    name="GC_3265",
    value="-((CKM2x2*cquqd11Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3266 = Coupling(
    name="GC_3266",
    value="-((CKM3x2*cquqd11Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3267 = Coupling(
    name="GC_3267",
    value="(CKM1x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3268 = Coupling(
    name="GC_3268",
    value="(CKM2x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3269 = Coupling(
    name="GC_3269",
    value="(CKM3x2*cquqd11Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3270 = Coupling(
    name="GC_3270",
    value="(CKM1x2*cquqd1Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3271 = Coupling(
    name="GC_3271",
    value="(CKM2x2*cquqd1Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3272 = Coupling(
    name="GC_3272",
    value="(CKM3x2*cquqd1Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3273 = Coupling(
    name="GC_3273",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3274 = Coupling(
    name="GC_3274",
    value="-((CKM2x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3275 = Coupling(
    name="GC_3275",
    value="-((CKM3x2*cquqd1Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3276 = Coupling(
    name="GC_3276",
    value="-((CKM1x2*cquqd81Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3277 = Coupling(
    name="GC_3277",
    value="-((CKM2x2*cquqd81Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3278 = Coupling(
    name="GC_3278",
    value="-((CKM3x2*cquqd81Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3279 = Coupling(
    name="GC_3279",
    value="(CKM1x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3280 = Coupling(
    name="GC_3280",
    value="(CKM2x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3281 = Coupling(
    name="GC_3281",
    value="(CKM3x2*cquqd81Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3282 = Coupling(
    name="GC_3282",
    value="(CKM1x2*cquqd8Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3283 = Coupling(
    name="GC_3283",
    value="(CKM2x2*cquqd8Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3284 = Coupling(
    name="GC_3284",
    value="(CKM3x2*cquqd8Im*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3285 = Coupling(
    name="GC_3285",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3286 = Coupling(
    name="GC_3286",
    value="-((CKM2x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3287 = Coupling(
    name="GC_3287",
    value="-((CKM3x2*cquqd8Re*complex(0,1)*ys*yt*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3288 = Coupling(
    name="GC_3288",
    value="-((cledqIm*yb*ytau*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3289 = Coupling(
    name="GC_3289",
    value="(cledqRe*complex(0,1)*yb*ytau*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3290 = Coupling(
    name="GC_3290",
    value="-((clequ1Im*yt*ytau*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3291 = Coupling(
    name="GC_3291",
    value="(clequ1Re*complex(0,1)*yt*ytau*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3292 = Coupling(
    name="GC_3292",
    value="-(clequ3Im*yt*ytau*complexconjugate(CKM3x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3293 = Coupling(
    name="GC_3293",
    value="(clequ3Im*yt*ytau*complexconjugate(CKM3x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3294 = Coupling(
    name="GC_3294",
    value="(clequ3Re*complex(0,1)*yt*ytau*complexconjugate(CKM3x3))/(4.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3295 = Coupling(
    name="GC_3295",
    value="-(clequ3Re*complex(0,1)*yt*ytau*complexconjugate(CKM3x3))/(2.*LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3296 = Coupling(
    name="GC_3296",
    value="(CKM1x1*cquqd11Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3297 = Coupling(
    name="GC_3297",
    value="(CKM1x2*cquqd11Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3298 = Coupling(
    name="GC_3298",
    value="(CKM1x3*cquqd11Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3299 = Coupling(
    name="GC_3299",
    value="(CKM1x1*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3300 = Coupling(
    name="GC_3300",
    value="(CKM1x2*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3301 = Coupling(
    name="GC_3301",
    value="(CKM1x3*cquqd11Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3302 = Coupling(
    name="GC_3302",
    value="-((CKM1x1*cquqd1Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3303 = Coupling(
    name="GC_3303",
    value="-((CKM1x2*cquqd1Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3304 = Coupling(
    name="GC_3304",
    value="-((CKM1x3*cquqd1Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3305 = Coupling(
    name="GC_3305",
    value="-((CKM1x1*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3306 = Coupling(
    name="GC_3306",
    value="-((CKM1x2*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3307 = Coupling(
    name="GC_3307",
    value="-((CKM1x3*cquqd1Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3308 = Coupling(
    name="GC_3308",
    value="(CKM1x1*cquqd81Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3309 = Coupling(
    name="GC_3309",
    value="(CKM1x2*cquqd81Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3310 = Coupling(
    name="GC_3310",
    value="(CKM1x3*cquqd81Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3311 = Coupling(
    name="GC_3311",
    value="(CKM1x1*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3312 = Coupling(
    name="GC_3312",
    value="(CKM1x2*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3313 = Coupling(
    name="GC_3313",
    value="(CKM1x3*cquqd81Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2",
    order={"NP": 1, "QED": 4},
)

GC_3314 = Coupling(
    name="GC_3314",
    value="-((CKM1x1*cquqd8Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3315 = Coupling(
    name="GC_3315",
    value="-((CKM1x2*cquqd8Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3316 = Coupling(
    name="GC_3316",
    value="-((CKM1x3*cquqd8Im*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3317 = Coupling(
    name="GC_3317",
    value="-((CKM1x1*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3318 = Coupling(
    name="GC_3318",
    value="-((CKM1x2*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)

GC_3319 = Coupling(
    name="GC_3319",
    value="-((CKM1x3*cquqd8Re*complex(0,1)*yb*yup*complexconjugate(CKM3x3))/LambdaSMEFT**2)",
    order={"NP": 1, "QED": 4},
)
