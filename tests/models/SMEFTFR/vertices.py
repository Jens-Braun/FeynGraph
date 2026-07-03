# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 13.2.1 for Linux x86 (64-bit) (January 27, 2023)
# Date: Mon 8 Dec 2025 20:55:51


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1, L.SSS2, L.SSS3 ],
             couplings = {(0,0):C.GC_28,(0,2):C.GC_114,(0,1):C.GC_128})

V_2 = Vertex(name = 'V_2',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_66})

V_3 = Vertex(name = 'V_3',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_157})

V_4 = Vertex(name = 'V_4',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1, L.SSSS2, L.SSSS3 ],
             couplings = {(0,0):C.GC_29,(0,2):C.GC_51,(0,1):C.GC_52})

V_5 = Vertex(name = 'V_5',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_85})

V_6 = Vertex(name = 'V_6',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_156})

V_7 = Vertex(name = 'V_7',
             particles = [ P.H, P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSS1 ],
             couplings = {(0,0):C.GC_113})

V_8 = Vertex(name = 'V_8',
             particles = [ P.H, P.H, P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSSSS1 ],
             couplings = {(0,0):C.GC_50})

V_9 = Vertex(name = 'V_9',
             particles = [ P.A, P.A, P.H ],
             color = [ '1' ],
             lorentz = [ L.VVS2, L.VVS4 ],
             couplings = {(0,0):C.GC_281,(0,1):C.GC_277})

V_10 = Vertex(name = 'V_10',
              particles = [ P.A, P.A, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS2, L.VVSS4 ],
              couplings = {(0,0):C.GC_275,(0,1):C.GC_271})

V_11 = Vertex(name = 'V_11',
              particles = [ P.g, P.g, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVS2, L.VVS4 ],
              couplings = {(0,0):C.GC_116,(0,1):C.GC_115})

V_12 = Vertex(name = 'V_12',
              particles = [ P.g, P.g, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVSS2, L.VVSS4 ],
              couplings = {(0,0):C.GC_54,(0,1):C.GC_53})

V_13 = Vertex(name = 'V_13',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1, L.VVS3, L.VVS4 ],
              couplings = {(0,1):C.GC_198,(0,0):C.GC_118,(0,2):C.GC_117})

V_14 = Vertex(name = 'V_14',
              particles = [ P.W__minus__, P.W__plus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS3 ],
              couplings = {(0,0):C.GC_204})

V_15 = Vertex(name = 'V_15',
              particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1, L.VVSS3, L.VVSS4 ],
              couplings = {(0,0):C.GC_56,(0,2):C.GC_55,(0,1):C.GC_199})

V_16 = Vertex(name = 'V_16',
              particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS3 ],
              couplings = {(0,0):C.GC_203})

V_17 = Vertex(name = 'V_17',
              particles = [ P.A, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV1, L.VVV3, L.VVV5, L.VVV6, L.VVV7, L.VVV8 ],
              couplings = {(0,0):C.GC_269,(0,3):C.GC_268,(0,5):C.GC_300,(0,4):C.GC_299,(0,2):C.GC_304,(0,1):C.GC_305})

V_18 = Vertex(name = 'V_18',
              particles = [ P.A, P.W__minus__, P.W__plus__ ],
              color = [ '1' ],
              lorentz = [ L.VVV5 ],
              couplings = {(0,0):C.GC_265})

V_19 = Vertex(name = 'V_19',
              particles = [ P.A, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1, L.VVS4 ],
              couplings = {(0,0):C.GC_279,(0,1):C.GC_276})

V_20 = Vertex(name = 'V_20',
              particles = [ P.A, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS1, L.VVSS4 ],
              couplings = {(0,0):C.GC_273,(0,1):C.GC_270})

V_21 = Vertex(name = 'V_21',
              particles = [ P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS2, L.VVS3, L.VVS4 ],
              couplings = {(0,1):C.GC_238,(0,0):C.GC_280,(0,2):C.GC_278})

V_22 = Vertex(name = 'V_22',
              particles = [ P.Z, P.Z, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS3 ],
              couplings = {(0,0):C.GC_267})

V_23 = Vertex(name = 'V_23',
              particles = [ P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS2, L.VVSS3, L.VVSS4 ],
              couplings = {(0,0):C.GC_274,(0,2):C.GC_272,(0,1):C.GC_239})

V_24 = Vertex(name = 'V_24',
              particles = [ P.Z, P.Z, P.H, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVSS3 ],
              couplings = {(0,0):C.GC_266})

V_25 = Vertex(name = 'V_25',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV2, L.VVV3, L.VVV4, L.VVV5, L.VVV7, L.VVV8 ],
              couplings = {(0,5):C.GC_222,(0,4):C.GC_221,(0,3):C.GC_223,(0,1):C.GC_229,(0,0):C.GC_258,(0,2):C.GC_257})

V_26 = Vertex(name = 'V_26',
              particles = [ P.W__minus__, P.W__plus__, P.Z ],
              color = [ '1' ],
              lorentz = [ L.VVV5 ],
              couplings = {(0,0):C.GC_228})

V_27 = Vertex(name = 'V_27',
              particles = [ P.g, P.g, P.g ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVV3, L.VVV5, L.VVV7, L.VVV9 ],
              couplings = {(0,3):C.GC_49,(0,2):C.GC_48,(0,1):C.GC_159,(0,0):C.GC_177})

V_28 = Vertex(name = 'V_28',
              particles = [ P.g, P.g, P.g, P.H ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVS3, L.VVVS6 ],
              couplings = {(0,0):C.GC_165,(0,1):C.GC_164})

V_29 = Vertex(name = 'V_29',
              particles = [ P.g, P.g, P.g, P.H, P.H ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVSS3, L.VVVSS6 ],
              couplings = {(0,0):C.GC_163,(0,1):C.GC_162})

V_30 = Vertex(name = 'V_30',
              particles = [ P.g, P.g, P.g, P.g ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVV1, L.VVVV10, L.VVVV12, L.VVVV13, L.VVVV2, L.VVVV3, L.VVVV4, L.VVVV6, L.VVVV9 ],
              couplings = {(1,8):C.GC_166,(0,0):C.GC_166,(2,1):C.GC_166,(0,7):C.GC_161,(1,6):C.GC_161,(2,5):C.GC_161,(0,4):C.GC_160,(1,3):C.GC_160,(2,2):C.GC_160})

V_31 = Vertex(name = 'V_31',
              particles = [ P.g, P.g, P.g, P.g, P.H ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVS1, L.VVVVS3, L.VVVVS4 ],
              couplings = {(1,1):C.GC_172,(0,0):C.GC_172,(2,2):C.GC_172})

V_32 = Vertex(name = 'V_32',
              particles = [ P.g, P.g, P.g, P.g, P.H, P.H ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVSS1, L.VVVVSS3, L.VVVVSS4 ],
              couplings = {(1,1):C.GC_171,(0,0):C.GC_171,(2,2):C.GC_171})

V_33 = Vertex(name = 'V_33',
              particles = [ P.g, P.g, P.g, P.g, P.g ],
              color = [ 'f(-2,1,2)*f(-1,-2,3)*f(4,5,-1)', 'f(-2,1,2)*f(-1,-2,4)*f(3,5,-1)', 'f(-2,1,2)*f(-1,-2,5)*f(3,4,-1)', 'f(-2,1,3)*f(-1,-2,2)*f(4,5,-1)', 'f(-2,1,3)*f(-1,-2,4)*f(2,5,-1)', 'f(-2,1,3)*f(-1,-2,5)*f(2,4,-1)', 'f(-2,1,4)*f(-1,-2,2)*f(3,5,-1)', 'f(-2,1,4)*f(-1,-2,3)*f(2,5,-1)', 'f(-2,1,4)*f(-1,-2,5)*f(2,3,-1)', 'f(-2,1,5)*f(-1,-2,2)*f(3,4,-1)', 'f(-2,1,5)*f(-1,-2,3)*f(2,4,-1)', 'f(-2,1,5)*f(-1,-2,4)*f(2,3,-1)', 'f(-2,2,3)*f(-1,-2,1)*f(4,5,-1)', 'f(-2,2,3)*f(-1,-2,4)*f(1,5,-1)', 'f(-2,2,3)*f(-1,-2,5)*f(1,4,-1)', 'f(-2,2,4)*f(-1,-2,1)*f(3,5,-1)', 'f(-2,2,4)*f(-1,-2,3)*f(1,5,-1)', 'f(-2,2,4)*f(-1,-2,5)*f(1,3,-1)', 'f(-2,2,5)*f(-1,-2,1)*f(3,4,-1)', 'f(-2,2,5)*f(-1,-2,3)*f(1,4,-1)', 'f(-2,2,5)*f(-1,-2,4)*f(1,3,-1)', 'f(-2,3,4)*f(-1,-2,1)*f(2,5,-1)', 'f(-2,3,4)*f(-1,-2,2)*f(1,5,-1)', 'f(-2,3,4)*f(-1,-2,5)*f(1,2,-1)', 'f(-2,3,5)*f(-1,-2,1)*f(2,4,-1)', 'f(-2,3,5)*f(-1,-2,2)*f(1,4,-1)', 'f(-2,3,5)*f(-1,-2,4)*f(1,2,-1)', 'f(-2,4,5)*f(-1,-2,1)*f(2,3,-1)', 'f(-2,4,5)*f(-1,-2,2)*f(1,3,-1)', 'f(-2,4,5)*f(-1,-2,3)*f(1,2,-1)' ],
              lorentz = [ L.VVVVV1, L.VVVVV10, L.VVVVV11, L.VVVVV12, L.VVVVV13, L.VVVVV14, L.VVVVV15, L.VVVVV17, L.VVVVV18, L.VVVVV19, L.VVVVV2, L.VVVVV20, L.VVVVV21, L.VVVVV22, L.VVVVV23, L.VVVVV24, L.VVVVV25, L.VVVVV28, L.VVVVV29, L.VVVVV3, L.VVVVV30, L.VVVVV31, L.VVVVV33, L.VVVVV34, L.VVVVV35, L.VVVVV36, L.VVVVV37, L.VVVVV4, L.VVVVV40, L.VVVVV41, L.VVVVV42, L.VVVVV43, L.VVVVV44, L.VVVVV46, L.VVVVV47, L.VVVVV48, L.VVVVV49, L.VVVVV5, L.VVVVV50, L.VVVVV51, L.VVVVV53, L.VVVVV54, L.VVVVV6, L.VVVVV7, L.VVVVV9 ],
              couplings = {(27,37):C.GC_169,(24,8):C.GC_170,(21,12):C.GC_169,(18,11):C.GC_170,(15,9):C.GC_169,(12,27):C.GC_169,(28,42):C.GC_169,(25,15):C.GC_170,(22,14):C.GC_169,(9,16):C.GC_169,(6,13):C.GC_170,(3,43):C.GC_170,(29,0):C.GC_170,(19,20):C.GC_169,(16,18):C.GC_170,(10,17):C.GC_169,(7,21):C.GC_170,(0,44):C.GC_169,(26,10):C.GC_169,(20,1):C.GC_170,(13,24):C.GC_169,(11,2):C.GC_170,(4,22):C.GC_169,(1,23):C.GC_169,(23,19):C.GC_170,(17,4):C.GC_169,(14,25):C.GC_170,(8,3):C.GC_169,(5,28):C.GC_170,(2,26):C.GC_170,(24,29):C.GC_168,(21,30):C.GC_167,(18,30):C.GC_168,(15,29):C.GC_167,(28,6):C.GC_168,(22,34):C.GC_168,(9,34):C.GC_167,(3,6):C.GC_167,(29,7):C.GC_168,(16,35):C.GC_168,(10,35):C.GC_167,(0,7):C.GC_167,(26,39):C.GC_167,(20,38):C.GC_167,(4,38):C.GC_168,(1,39):C.GC_168,(25,33):C.GC_168,(6,33):C.GC_167,(19,36):C.GC_168,(7,36):C.GC_167,(23,41):C.GC_167,(17,40):C.GC_167,(5,40):C.GC_168,(2,41):C.GC_168,(27,5):C.GC_168,(12,5):C.GC_167,(13,31):C.GC_168,(11,31):C.GC_167,(14,32):C.GC_167,(8,32):C.GC_168})

V_34 = Vertex(name = 'V_34',
              particles = [ P.g, P.g, P.g, P.g, P.g, P.g ],
              color = [ 'f(-3,1,2)*f(-2,3,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,2)*f(-2,3,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,2)*f(-2,3,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,2)*f(-2,4,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,2)*f(-2,4,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,2)*f(-2,5,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,3)*f(-2,2,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,3)*f(-2,2,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,3)*f(-2,2,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,3)*f(-2,4,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,3)*f(-2,4,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,3)*f(-2,5,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,4)*f(-2,2,3)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,1,4)*f(-2,2,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,4)*f(-2,2,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,4)*f(-2,3,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,4)*f(-2,3,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,4)*f(-2,5,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,1,5)*f(-2,2,3)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,1,5)*f(-2,2,4)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,1,5)*f(-2,2,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,5)*f(-2,3,4)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,1,5)*f(-2,3,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,5)*f(-2,4,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,1,6)*f(-2,2,3)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,1,6)*f(-2,2,4)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,1,6)*f(-2,2,5)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,1,6)*f(-2,3,4)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,1,6)*f(-2,3,5)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,1,6)*f(-2,4,5)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,2,3)*f(-2,1,4)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,2,3)*f(-2,1,5)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,2,3)*f(-2,1,6)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,2,3)*f(-2,4,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,3)*f(-2,4,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,3)*f(-2,5,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,4)*f(-2,1,3)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,2,4)*f(-2,1,5)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,2,4)*f(-2,1,6)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,2,4)*f(-2,3,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,4)*f(-2,3,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,4)*f(-2,5,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,2,5)*f(-2,1,3)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,2,5)*f(-2,1,4)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,2,5)*f(-2,1,6)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,2,5)*f(-2,3,4)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,2,5)*f(-2,3,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,5)*f(-2,4,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,2,6)*f(-2,1,3)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,2,6)*f(-2,1,4)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,2,6)*f(-2,1,5)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,2,6)*f(-2,3,4)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,2,6)*f(-2,3,5)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,2,6)*f(-2,4,5)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,3,4)*f(-2,1,2)*f(-1,-2,-3)*f(5,6,-1)', 'f(-3,3,4)*f(-2,1,5)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,3,4)*f(-2,1,6)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,3,4)*f(-2,2,5)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,3,4)*f(-2,2,6)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,3,4)*f(-2,5,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,3,5)*f(-2,1,2)*f(-1,-2,-3)*f(4,6,-1)', 'f(-3,3,5)*f(-2,1,4)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,3,5)*f(-2,1,6)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,3,5)*f(-2,2,4)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,3,5)*f(-2,2,6)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,3,5)*f(-2,4,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,3,6)*f(-2,1,2)*f(-1,-2,-3)*f(4,5,-1)', 'f(-3,3,6)*f(-2,1,4)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,3,6)*f(-2,1,5)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,3,6)*f(-2,2,4)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,3,6)*f(-2,2,5)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,3,6)*f(-2,4,5)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,4,5)*f(-2,1,2)*f(-1,-2,-3)*f(3,6,-1)', 'f(-3,4,5)*f(-2,1,3)*f(-1,-2,-3)*f(2,6,-1)', 'f(-3,4,5)*f(-2,1,6)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,4,5)*f(-2,2,3)*f(-1,-2,-3)*f(1,6,-1)', 'f(-3,4,5)*f(-2,2,6)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,4,5)*f(-2,3,6)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,4,6)*f(-2,1,2)*f(-1,-2,-3)*f(3,5,-1)', 'f(-3,4,6)*f(-2,1,3)*f(-1,-2,-3)*f(2,5,-1)', 'f(-3,4,6)*f(-2,1,5)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,4,6)*f(-2,2,3)*f(-1,-2,-3)*f(1,5,-1)', 'f(-3,4,6)*f(-2,2,5)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,4,6)*f(-2,3,5)*f(-1,-2,-3)*f(1,2,-1)', 'f(-3,5,6)*f(-2,1,2)*f(-1,-2,-3)*f(3,4,-1)', 'f(-3,5,6)*f(-2,1,3)*f(-1,-2,-3)*f(2,4,-1)', 'f(-3,5,6)*f(-2,1,4)*f(-1,-2,-3)*f(2,3,-1)', 'f(-3,5,6)*f(-2,2,3)*f(-1,-2,-3)*f(1,4,-1)', 'f(-3,5,6)*f(-2,2,4)*f(-1,-2,-3)*f(1,3,-1)', 'f(-3,5,6)*f(-2,3,4)*f(-1,-2,-3)*f(1,2,-1)' ],
              lorentz = [ L.VVVVVV1, L.VVVVVV10, L.VVVVVV11, L.VVVVVV12, L.VVVVVV13, L.VVVVVV14, L.VVVVVV15, L.VVVVVV16, L.VVVVVV17, L.VVVVVV18, L.VVVVVV19, L.VVVVVV2, L.VVVVVV20, L.VVVVVV21, L.VVVVVV22, L.VVVVVV23, L.VVVVVV24, L.VVVVVV25, L.VVVVVV26, L.VVVVVV27, L.VVVVVV28, L.VVVVVV29, L.VVVVVV3, L.VVVVVV30, L.VVVVVV31, L.VVVVVV32, L.VVVVVV33, L.VVVVVV34, L.VVVVVV35, L.VVVVVV36, L.VVVVVV37, L.VVVVVV38, L.VVVVVV39, L.VVVVVV4, L.VVVVVV40, L.VVVVVV41, L.VVVVVV42, L.VVVVVV43, L.VVVVVV44, L.VVVVVV45, L.VVVVVV46, L.VVVVVV47, L.VVVVVV48, L.VVVVVV49, L.VVVVVV5, L.VVVVVV50, L.VVVVVV51, L.VVVVVV52, L.VVVVVV54, L.VVVVVV55, L.VVVVVV56, L.VVVVVV57, L.VVVVVV58, L.VVVVVV59, L.VVVVVV6, L.VVVVVV60, L.VVVVVV61, L.VVVVVV7, L.VVVVVV8, L.VVVVVV9 ],
              couplings = {(41,58):C.GC_176,(47,59):C.GC_175,(53,7):C.GC_176,(35,57):C.GC_175,(46,14):C.GC_176,(52,17):C.GC_175,(34,2):C.GC_176,(40,10):C.GC_175,(51,37):C.GC_176,(33,4):C.GC_175,(39,21):C.GC_176,(45,30):C.GC_175,(17,57):C.GC_176,(23,2):C.GC_175,(29,4):C.GC_176,(11,58):C.GC_175,(22,10):C.GC_176,(28,21):C.GC_175,(10,59):C.GC_176,(16,14):C.GC_175,(27,30):C.GC_176,(9,7):C.GC_175,(15,17):C.GC_176,(21,37):C.GC_175,(59,0):C.GC_176,(65,11):C.GC_175,(71,44):C.GC_176,(64,12):C.GC_176,(70,23):C.GC_175,(58,16):C.GC_175,(69,31):C.GC_176,(57,19):C.GC_176,(63,39):C.GC_175,(5,0):C.GC_175,(20,16):C.GC_176,(26,19):C.GC_175,(4,11):C.GC_176,(14,12):C.GC_175,(25,39):C.GC_176,(3,44):C.GC_175,(13,23):C.GC_176,(19,31):C.GC_175,(77,22):C.GC_175,(83,33):C.GC_176,(76,1):C.GC_176,(82,8):C.GC_175,(81,40):C.GC_176,(75,35):C.GC_175,(2,22):C.GC_176,(8,1):C.GC_175,(24,35):C.GC_176,(1,33):C.GC_175,(7,8):C.GC_176,(18,40):C.GC_175,(89,54):C.GC_176,(88,6):C.GC_175,(87,25):C.GC_176,(0,54):C.GC_175,(6,6):C.GC_176,(12,25):C.GC_175,(62,15):C.GC_176,(68,18):C.GC_175,(56,13):C.GC_175,(67,38):C.GC_176,(55,24):C.GC_176,(61,32):C.GC_175,(44,13):C.GC_176,(50,24):C.GC_175,(38,15):C.GC_175,(49,32):C.GC_176,(37,18):C.GC_176,(43,38):C.GC_175,(74,3):C.GC_176,(80,5):C.GC_175,(79,34):C.GC_176,(73,42):C.GC_175,(32,3):C.GC_175,(48,42):C.GC_176,(31,5):C.GC_176,(42,34):C.GC_175,(86,9):C.GC_175,(85,20):C.GC_176,(30,9):C.GC_176,(36,20):C.GC_175,(78,41):C.GC_176,(72,36):C.GC_175,(66,36):C.GC_176,(60,41):C.GC_175,(65,43):C.GC_173,(71,46):C.GC_174,(77,46):C.GC_173,(83,43):C.GC_174,(41,28):C.GC_173,(53,50):C.GC_173,(76,50):C.GC_174,(88,28):C.GC_174,(35,29):C.GC_173,(52,53):C.GC_173,(64,53):C.GC_174,(87,29):C.GC_174,(34,52):C.GC_174,(40,51):C.GC_174,(69,51):C.GC_173,(81,52):C.GC_173,(17,29):C.GC_174,(23,52):C.GC_173,(80,52):C.GC_174,(86,29):C.GC_173,(11,28):C.GC_174,(22,51):C.GC_173,(68,51):C.GC_174,(85,28):C.GC_173,(9,50):C.GC_174,(15,53):C.GC_174,(61,53):C.GC_173,(73,50):C.GC_173,(4,43):C.GC_174,(14,53):C.GC_173,(49,53):C.GC_174,(78,43):C.GC_173,(3,46):C.GC_173,(19,51):C.GC_174,(37,51):C.GC_173,(72,46):C.GC_174,(2,46):C.GC_174,(8,50):C.GC_173,(48,50):C.GC_174,(66,46):C.GC_173,(1,43):C.GC_173,(18,52):C.GC_174,(31,52):C.GC_173,(60,43):C.GC_174,(6,28):C.GC_173,(12,29):C.GC_173,(30,29):C.GC_174,(36,28):C.GC_174,(47,48):C.GC_173,(82,48):C.GC_174,(46,55):C.GC_173,(70,55):C.GC_174,(33,56):C.GC_174,(39,49):C.GC_174,(63,49):C.GC_173,(75,56):C.GC_173,(29,56):C.GC_173,(74,56):C.GC_174,(28,49):C.GC_173,(62,49):C.GC_174,(10,48):C.GC_174,(16,55):C.GC_174,(67,55):C.GC_173,(79,48):C.GC_173,(25,49):C.GC_174,(38,49):C.GC_173,(13,55):C.GC_173,(43,55):C.GC_174,(24,56):C.GC_174,(32,56):C.GC_173,(7,48):C.GC_173,(42,48):C.GC_174,(84,26):C.GC_176,(54,26):C.GC_175,(59,27):C.GC_173,(89,27):C.GC_174,(51,45):C.GC_173,(58,45):C.GC_174,(21,45):C.GC_174,(55,45):C.GC_173,(5,27):C.GC_174,(20,45):C.GC_173,(50,45):C.GC_174,(84,27):C.GC_173,(0,27):C.GC_173,(54,27):C.GC_174,(45,47):C.GC_174,(57,47):C.GC_173,(27,47):C.GC_173,(56,47):C.GC_174,(26,47):C.GC_174,(44,47):C.GC_173})

V_35 = Vertex(name = 'V_35',
              particles = [ P.d__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2, L.FFS3 ],
              couplings = {(0,0):C.GC_307,(0,1):C.GC_67,(0,2):C.GC_1})

V_36 = Vertex(name = 'V_36',
              particles = [ P.d__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS3 ],
              couplings = {(0,0):C.GC_129})

V_37 = Vertex(name = 'V_37',
              particles = [ P.s__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_2,(0,1):C.GC_2})

V_38 = Vertex(name = 'V_38',
              particles = [ P.s__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_130,(0,1):C.GC_68})

V_39 = Vertex(name = 'V_39',
              particles = [ P.s__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_316,(0,1):C.GC_130})

V_40 = Vertex(name = 'V_40',
              particles = [ P.b__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_3,(0,1):C.GC_3})

V_41 = Vertex(name = 'V_41',
              particles = [ P.b__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_131,(0,1):C.GC_69})

V_42 = Vertex(name = 'V_42',
              particles = [ P.b__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_325,(0,1):C.GC_131})

V_43 = Vertex(name = 'V_43',
              particles = [ P.d__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_4,(0,1):C.GC_4})

V_44 = Vertex(name = 'V_44',
              particles = [ P.d__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_132,(0,1):C.GC_70})

V_45 = Vertex(name = 'V_45',
              particles = [ P.d__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_310,(0,1):C.GC_132})

V_46 = Vertex(name = 'V_46',
              particles = [ P.s__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_5,(0,1):C.GC_5})

V_47 = Vertex(name = 'V_47',
              particles = [ P.s__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_133,(0,1):C.GC_71})

V_48 = Vertex(name = 'V_48',
              particles = [ P.s__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_319,(0,1):C.GC_133})

V_49 = Vertex(name = 'V_49',
              particles = [ P.b__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_6,(0,1):C.GC_6})

V_50 = Vertex(name = 'V_50',
              particles = [ P.b__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_134,(0,1):C.GC_72})

V_51 = Vertex(name = 'V_51',
              particles = [ P.b__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_328,(0,1):C.GC_134})

V_52 = Vertex(name = 'V_52',
              particles = [ P.d__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_7,(0,1):C.GC_7})

V_53 = Vertex(name = 'V_53',
              particles = [ P.d__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_135,(0,1):C.GC_73})

V_54 = Vertex(name = 'V_54',
              particles = [ P.d__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_313,(0,1):C.GC_135})

V_55 = Vertex(name = 'V_55',
              particles = [ P.s__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_8,(0,1):C.GC_8})

V_56 = Vertex(name = 'V_56',
              particles = [ P.s__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_136,(0,1):C.GC_74})

V_57 = Vertex(name = 'V_57',
              particles = [ P.s__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_322,(0,1):C.GC_136})

V_58 = Vertex(name = 'V_58',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_9,(0,1):C.GC_9})

V_59 = Vertex(name = 'V_59',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_137,(0,1):C.GC_75})

V_60 = Vertex(name = 'V_60',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_331,(0,1):C.GC_137})

V_61 = Vertex(name = 'V_61',
              particles = [ P.d__tilde__, P.d, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_308,(0,1):C.GC_95})

V_62 = Vertex(name = 'V_62',
              particles = [ P.s__tilde__, P.d, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_317,(0,1):C.GC_96})

V_63 = Vertex(name = 'V_63',
              particles = [ P.b__tilde__, P.d, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_326,(0,1):C.GC_97})

V_64 = Vertex(name = 'V_64',
              particles = [ P.d__tilde__, P.s, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_311,(0,1):C.GC_98})

V_65 = Vertex(name = 'V_65',
              particles = [ P.s__tilde__, P.s, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_320,(0,1):C.GC_99})

V_66 = Vertex(name = 'V_66',
              particles = [ P.b__tilde__, P.s, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_329,(0,1):C.GC_100})

V_67 = Vertex(name = 'V_67',
              particles = [ P.d__tilde__, P.b, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_314,(0,1):C.GC_101})

V_68 = Vertex(name = 'V_68',
              particles = [ P.s__tilde__, P.b, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_323,(0,1):C.GC_102})

V_69 = Vertex(name = 'V_69',
              particles = [ P.b__tilde__, P.b, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSS1, L.FFSS2 ],
              couplings = {(0,0):C.GC_332,(0,1):C.GC_103})

V_70 = Vertex(name = 'V_70',
              particles = [ P.d__tilde__, P.d, P.H, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSSS1, L.FFSSS2 ],
              couplings = {(0,0):C.GC_306,(0,1):C.GC_30})

V_71 = Vertex(name = 'V_71',
              particles = [ P.s__tilde__, P.d, P.H, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSSS1, L.FFSSS2 ],
              couplings = {(0,0):C.GC_315,(0,1):C.GC_31})

V_72 = Vertex(name = 'V_72',
              particles = [ P.b__tilde__, P.d, P.H, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSSS1, L.FFSSS2 ],
              couplings = {(0,0):C.GC_324,(0,1):C.GC_32})

V_73 = Vertex(name = 'V_73',
              particles = [ P.d__tilde__, P.s, P.H, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSSS1, L.FFSSS2 ],
              couplings = {(0,0):C.GC_309,(0,1):C.GC_33})

V_74 = Vertex(name = 'V_74',
              particles = [ P.s__tilde__, P.s, P.H, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSSS1, L.FFSSS2 ],
              couplings = {(0,0):C.GC_318,(0,1):C.GC_34})

V_75 = Vertex(name = 'V_75',
              particles = [ P.b__tilde__, P.s, P.H, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSSS1, L.FFSSS2 ],
              couplings = {(0,0):C.GC_327,(0,1):C.GC_35})

V_76 = Vertex(name = 'V_76',
              particles = [ P.d__tilde__, P.b, P.H, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSSS1, L.FFSSS2 ],
              couplings = {(0,0):C.GC_312,(0,1):C.GC_36})

V_77 = Vertex(name = 'V_77',
              particles = [ P.s__tilde__, P.b, P.H, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSSS1, L.FFSSS2 ],
              couplings = {(0,0):C.GC_321,(0,1):C.GC_37})

V_78 = Vertex(name = 'V_78',
              particles = [ P.b__tilde__, P.b, P.H, P.H, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFSSS1, L.FFSSS2 ],
              couplings = {(0,0):C.GC_330,(0,1):C.GC_38})

V_79 = Vertex(name = 'V_79',
              particles = [ P.e__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_10,(0,1):C.GC_10})

V_80 = Vertex(name = 'V_80',
              particles = [ P.e__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_138,(0,1):C.GC_76})

V_81 = Vertex(name = 'V_81',
              particles = [ P.e__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_334,(0,1):C.GC_138})

V_82 = Vertex(name = 'V_82',
              particles = [ P.mu__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_11,(0,1):C.GC_11})

V_83 = Vertex(name = 'V_83',
              particles = [ P.mu__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_139,(0,1):C.GC_77})

V_84 = Vertex(name = 'V_84',
              particles = [ P.mu__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_343,(0,1):C.GC_139})

V_85 = Vertex(name = 'V_85',
              particles = [ P.ta__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_12,(0,1):C.GC_12})

V_86 = Vertex(name = 'V_86',
              particles = [ P.ta__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_140,(0,1):C.GC_78})

V_87 = Vertex(name = 'V_87',
              particles = [ P.ta__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_352,(0,1):C.GC_140})

V_88 = Vertex(name = 'V_88',
              particles = [ P.e__plus__, P.mu__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_13,(0,1):C.GC_13})

V_89 = Vertex(name = 'V_89',
              particles = [ P.e__plus__, P.mu__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_141,(0,1):C.GC_79})

V_90 = Vertex(name = 'V_90',
              particles = [ P.e__plus__, P.mu__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_337,(0,1):C.GC_141})

V_91 = Vertex(name = 'V_91',
              particles = [ P.mu__plus__, P.mu__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_14,(0,1):C.GC_14})

V_92 = Vertex(name = 'V_92',
              particles = [ P.mu__plus__, P.mu__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_142,(0,1):C.GC_80})

V_93 = Vertex(name = 'V_93',
              particles = [ P.mu__plus__, P.mu__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_346,(0,1):C.GC_142})

V_94 = Vertex(name = 'V_94',
              particles = [ P.ta__plus__, P.mu__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_15,(0,1):C.GC_15})

V_95 = Vertex(name = 'V_95',
              particles = [ P.ta__plus__, P.mu__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_143,(0,1):C.GC_81})

V_96 = Vertex(name = 'V_96',
              particles = [ P.ta__plus__, P.mu__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_355,(0,1):C.GC_143})

V_97 = Vertex(name = 'V_97',
              particles = [ P.e__plus__, P.ta__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_16,(0,1):C.GC_16})

V_98 = Vertex(name = 'V_98',
              particles = [ P.e__plus__, P.ta__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_144,(0,1):C.GC_82})

V_99 = Vertex(name = 'V_99',
              particles = [ P.e__plus__, P.ta__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1, L.FFS2 ],
              couplings = {(0,0):C.GC_340,(0,1):C.GC_144})

V_100 = Vertex(name = 'V_100',
               particles = [ P.mu__plus__, P.ta__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_17,(0,1):C.GC_17})

V_101 = Vertex(name = 'V_101',
               particles = [ P.mu__plus__, P.ta__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_145,(0,1):C.GC_83})

V_102 = Vertex(name = 'V_102',
               particles = [ P.mu__plus__, P.ta__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_349,(0,1):C.GC_145})

V_103 = Vertex(name = 'V_103',
               particles = [ P.ta__plus__, P.ta__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_18,(0,1):C.GC_18})

V_104 = Vertex(name = 'V_104',
               particles = [ P.ta__plus__, P.ta__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_146,(0,1):C.GC_84})

V_105 = Vertex(name = 'V_105',
               particles = [ P.ta__plus__, P.ta__minus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_358,(0,1):C.GC_146})

V_106 = Vertex(name = 'V_106',
               particles = [ P.e__plus__, P.e__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_335,(0,1):C.GC_104})

V_107 = Vertex(name = 'V_107',
               particles = [ P.mu__plus__, P.e__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_344,(0,1):C.GC_105})

V_108 = Vertex(name = 'V_108',
               particles = [ P.ta__plus__, P.e__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_353,(0,1):C.GC_106})

V_109 = Vertex(name = 'V_109',
               particles = [ P.e__plus__, P.mu__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_338,(0,1):C.GC_107})

V_110 = Vertex(name = 'V_110',
               particles = [ P.mu__plus__, P.mu__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_347,(0,1):C.GC_108})

V_111 = Vertex(name = 'V_111',
               particles = [ P.ta__plus__, P.mu__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_356,(0,1):C.GC_109})

V_112 = Vertex(name = 'V_112',
               particles = [ P.e__plus__, P.ta__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_341,(0,1):C.GC_110})

V_113 = Vertex(name = 'V_113',
               particles = [ P.mu__plus__, P.ta__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_350,(0,1):C.GC_111})

V_114 = Vertex(name = 'V_114',
               particles = [ P.ta__plus__, P.ta__minus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_359,(0,1):C.GC_112})

V_115 = Vertex(name = 'V_115',
               particles = [ P.e__plus__, P.e__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_333,(0,1):C.GC_39})

V_116 = Vertex(name = 'V_116',
               particles = [ P.mu__plus__, P.e__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_342,(0,1):C.GC_40})

V_117 = Vertex(name = 'V_117',
               particles = [ P.ta__plus__, P.e__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_351,(0,1):C.GC_41})

V_118 = Vertex(name = 'V_118',
               particles = [ P.e__plus__, P.mu__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_336,(0,1):C.GC_42})

V_119 = Vertex(name = 'V_119',
               particles = [ P.mu__plus__, P.mu__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_345,(0,1):C.GC_43})

V_120 = Vertex(name = 'V_120',
               particles = [ P.ta__plus__, P.mu__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_354,(0,1):C.GC_44})

V_121 = Vertex(name = 'V_121',
               particles = [ P.e__plus__, P.ta__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_339,(0,1):C.GC_45})

V_122 = Vertex(name = 'V_122',
               particles = [ P.mu__plus__, P.ta__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_348,(0,1):C.GC_46})

V_123 = Vertex(name = 'V_123',
               particles = [ P.ta__plus__, P.ta__minus__, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_357,(0,1):C.GC_47})

V_124 = Vertex(name = 'V_124',
               particles = [ P.u__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_19,(0,1):C.GC_19})

V_125 = Vertex(name = 'V_125',
               particles = [ P.u__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_147,(0,1):C.GC_86})

V_126 = Vertex(name = 'V_126',
               particles = [ P.u__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_361,(0,1):C.GC_147})

V_127 = Vertex(name = 'V_127',
               particles = [ P.c__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_20,(0,1):C.GC_20})

V_128 = Vertex(name = 'V_128',
               particles = [ P.c__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_148,(0,1):C.GC_87})

V_129 = Vertex(name = 'V_129',
               particles = [ P.c__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_370,(0,1):C.GC_148})

V_130 = Vertex(name = 'V_130',
               particles = [ P.t__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_21,(0,1):C.GC_21})

V_131 = Vertex(name = 'V_131',
               particles = [ P.t__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_149,(0,1):C.GC_88})

V_132 = Vertex(name = 'V_132',
               particles = [ P.t__tilde__, P.u, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_379,(0,1):C.GC_149})

V_133 = Vertex(name = 'V_133',
               particles = [ P.u__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_22,(0,1):C.GC_22})

V_134 = Vertex(name = 'V_134',
               particles = [ P.u__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_150,(0,1):C.GC_89})

V_135 = Vertex(name = 'V_135',
               particles = [ P.u__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_364,(0,1):C.GC_150})

V_136 = Vertex(name = 'V_136',
               particles = [ P.c__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_23,(0,1):C.GC_23})

V_137 = Vertex(name = 'V_137',
               particles = [ P.c__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_151,(0,1):C.GC_90})

V_138 = Vertex(name = 'V_138',
               particles = [ P.c__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_373,(0,1):C.GC_151})

V_139 = Vertex(name = 'V_139',
               particles = [ P.t__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_24,(0,1):C.GC_24})

V_140 = Vertex(name = 'V_140',
               particles = [ P.t__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_152,(0,1):C.GC_91})

V_141 = Vertex(name = 'V_141',
               particles = [ P.t__tilde__, P.c, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_382,(0,1):C.GC_152})

V_142 = Vertex(name = 'V_142',
               particles = [ P.u__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_25,(0,1):C.GC_25})

V_143 = Vertex(name = 'V_143',
               particles = [ P.u__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_153,(0,1):C.GC_92})

V_144 = Vertex(name = 'V_144',
               particles = [ P.u__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_367,(0,1):C.GC_153})

V_145 = Vertex(name = 'V_145',
               particles = [ P.c__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_26,(0,1):C.GC_26})

V_146 = Vertex(name = 'V_146',
               particles = [ P.c__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_154,(0,1):C.GC_93})

V_147 = Vertex(name = 'V_147',
               particles = [ P.c__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_376,(0,1):C.GC_154})

V_148 = Vertex(name = 'V_148',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_27,(0,1):C.GC_27})

V_149 = Vertex(name = 'V_149',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_155,(0,1):C.GC_94})

V_150 = Vertex(name = 'V_150',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1, L.FFS2 ],
               couplings = {(0,0):C.GC_385,(0,1):C.GC_155})

V_151 = Vertex(name = 'V_151',
               particles = [ P.u__tilde__, P.u, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_362,(0,1):C.GC_119})

V_152 = Vertex(name = 'V_152',
               particles = [ P.c__tilde__, P.u, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_371,(0,1):C.GC_120})

V_153 = Vertex(name = 'V_153',
               particles = [ P.t__tilde__, P.u, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_380,(0,1):C.GC_121})

V_154 = Vertex(name = 'V_154',
               particles = [ P.u__tilde__, P.c, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_365,(0,1):C.GC_122})

V_155 = Vertex(name = 'V_155',
               particles = [ P.c__tilde__, P.c, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_374,(0,1):C.GC_123})

V_156 = Vertex(name = 'V_156',
               particles = [ P.t__tilde__, P.c, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_383,(0,1):C.GC_124})

V_157 = Vertex(name = 'V_157',
               particles = [ P.u__tilde__, P.t, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_368,(0,1):C.GC_125})

V_158 = Vertex(name = 'V_158',
               particles = [ P.c__tilde__, P.t, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_377,(0,1):C.GC_126})

V_159 = Vertex(name = 'V_159',
               particles = [ P.t__tilde__, P.t, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSS1, L.FFSS2 ],
               couplings = {(0,0):C.GC_386,(0,1):C.GC_127})

V_160 = Vertex(name = 'V_160',
               particles = [ P.u__tilde__, P.u, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_360,(0,1):C.GC_57})

V_161 = Vertex(name = 'V_161',
               particles = [ P.c__tilde__, P.u, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_369,(0,1):C.GC_58})

V_162 = Vertex(name = 'V_162',
               particles = [ P.t__tilde__, P.u, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_378,(0,1):C.GC_59})

V_163 = Vertex(name = 'V_163',
               particles = [ P.u__tilde__, P.c, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_363,(0,1):C.GC_60})

V_164 = Vertex(name = 'V_164',
               particles = [ P.c__tilde__, P.c, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_372,(0,1):C.GC_61})

V_165 = Vertex(name = 'V_165',
               particles = [ P.t__tilde__, P.c, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_381,(0,1):C.GC_62})

V_166 = Vertex(name = 'V_166',
               particles = [ P.u__tilde__, P.t, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_366,(0,1):C.GC_63})

V_167 = Vertex(name = 'V_167',
               particles = [ P.c__tilde__, P.t, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_375,(0,1):C.GC_64})

V_168 = Vertex(name = 'V_168',
               particles = [ P.t__tilde__, P.t, P.H, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFSSS1, L.FFSSS2 ],
               couplings = {(0,0):C.GC_384,(0,1):C.GC_65})

V_169 = Vertex(name = 'V_169',
               particles = [ P.A, P.W__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVS1, L.VVVS3, L.VVVS4, L.VVVS6 ],
               couplings = {(0,0):C.GC_227,(0,2):C.GC_225,(0,1):C.GC_255,(0,3):C.GC_253})

V_170 = Vertex(name = 'V_170',
               particles = [ P.A, P.W__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS1, L.VVVSS3, L.VVVSS4, L.VVVSS6 ],
               couplings = {(0,0):C.GC_233,(0,2):C.GC_231,(0,1):C.GC_261,(0,3):C.GC_259})

V_171 = Vertex(name = 'V_171',
               particles = [ P.A, P.A, P.W__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVV15, L.VVVV5, L.VVVV8 ],
               couplings = {(0,2):C.GC_207,(0,1):C.GC_206,(0,0):C.GC_205})

V_172 = Vertex(name = 'V_172',
               particles = [ P.A, P.A, P.W__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVV8 ],
               couplings = {(0,0):C.GC_282})

V_173 = Vertex(name = 'V_173',
               particles = [ P.A, P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV11, L.VVVV14, L.VVVV7 ],
               couplings = {(0,0):C.GC_295,(0,2):C.GC_293,(0,1):C.GC_292})

V_174 = Vertex(name = 'V_174',
               particles = [ P.A, P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV11 ],
               couplings = {(0,0):C.GC_294})

V_175 = Vertex(name = 'V_175',
               particles = [ P.A, P.A, P.W__minus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_208})

V_176 = Vertex(name = 'V_176',
               particles = [ P.A, P.A, P.W__minus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_209})

V_177 = Vertex(name = 'V_177',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVS2, L.VVVS3, L.VVVS5, L.VVVS6 ],
               couplings = {(0,1):C.GC_226,(0,3):C.GC_224,(0,0):C.GC_256,(0,2):C.GC_254})

V_178 = Vertex(name = 'V_178',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVSS2, L.VVVSS3, L.VVVSS5, L.VVVSS6 ],
               couplings = {(0,1):C.GC_232,(0,3):C.GC_230,(0,0):C.GC_262,(0,2):C.GC_260})

V_179 = Vertex(name = 'V_179',
               particles = [ P.A, P.A, P.A, P.W__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVV16, L.VVVVV8 ],
               couplings = {(0,1):C.GC_291,(0,0):C.GC_290})

V_180 = Vertex(name = 'V_180',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVV15, L.VVVV5, L.VVVV8 ],
               couplings = {(0,2):C.GC_200,(0,1):C.GC_188,(0,0):C.GC_187})

V_181 = Vertex(name = 'V_181',
               particles = [ P.A, P.A, P.W__minus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV26, L.VVVVV55 ],
               couplings = {(0,0):C.GC_212,(0,1):C.GC_211})

V_182 = Vertex(name = 'V_182',
               particles = [ P.A, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVV32, L.VVVVV57 ],
               couplings = {(0,0):C.GC_264,(0,1):C.GC_263})

V_183 = Vertex(name = 'V_183',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_201})

V_184 = Vertex(name = 'V_184',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_202})

V_185 = Vertex(name = 'V_185',
               particles = [ P.A, P.A, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV53 ],
               couplings = {(0,0):C.GC_210})

V_186 = Vertex(name = 'V_186',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV15, L.VVVV5, L.VVVV8 ],
               couplings = {(0,2):C.GC_217,(0,1):C.GC_216,(0,0):C.GC_215})

V_187 = Vertex(name = 'V_187',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVV8 ],
               couplings = {(0,0):C.GC_283})

V_188 = Vertex(name = 'V_188',
               particles = [ P.A, P.W__minus__, P.W__plus__, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS5 ],
               couplings = {(0,0):C.GC_250})

V_189 = Vertex(name = 'V_189',
               particles = [ P.A, P.W__minus__, P.W__plus__, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS5 ],
               couplings = {(0,0):C.GC_251})

V_190 = Vertex(name = 'V_190',
               particles = [ P.A, P.W__minus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV38, L.VVVVV45 ],
               couplings = {(0,0):C.GC_249,(0,1):C.GC_248})

V_191 = Vertex(name = 'V_191',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV27, L.VVVVV52 ],
               couplings = {(0,0):C.GC_235,(0,1):C.GC_234})

V_192 = Vertex(name = 'V_192',
               particles = [ P.A, P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV62 ],
               couplings = {(0,0):C.GC_252})

V_193 = Vertex(name = 'V_193',
               particles = [ P.Z, P.Z, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSS1 ],
               couplings = {(0,0):C.GC_240})

V_194 = Vertex(name = 'V_194',
               particles = [ P.Z, P.Z, P.H, P.H, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVSSSS1 ],
               couplings = {(0,0):C.GC_241})

V_195 = Vertex(name = 'V_195',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVS2 ],
               couplings = {(0,0):C.GC_218})

V_196 = Vertex(name = 'V_196',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.H, P.H ],
               color = [ '1' ],
               lorentz = [ L.VVVVSS2 ],
               couplings = {(0,0):C.GC_219})

V_197 = Vertex(name = 'V_197',
               particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVVV53 ],
               couplings = {(0,0):C.GC_220})

V_198 = Vertex(name = 'V_198',
               particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z, P.Z ],
               color = [ '1' ],
               lorentz = [ L.VVVVV39, L.VVVVV56 ],
               couplings = {(0,0):C.GC_214,(0,1):C.GC_213})

V_199 = Vertex(name = 'V_199',
               particles = [ P.d__tilde__, P.d, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_296})

V_200 = Vertex(name = 'V_200',
               particles = [ P.d__tilde__, P.d, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_301})

V_201 = Vertex(name = 'V_201',
               particles = [ P.s__tilde__, P.s, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_296})

V_202 = Vertex(name = 'V_202',
               particles = [ P.s__tilde__, P.s, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_301})

V_203 = Vertex(name = 'V_203',
               particles = [ P.b__tilde__, P.b, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_296})

V_204 = Vertex(name = 'V_204',
               particles = [ P.b__tilde__, P.b, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_301})

V_205 = Vertex(name = 'V_205',
               particles = [ P.e__plus__, P.e__minus__, P.A ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_298})

V_206 = Vertex(name = 'V_206',
               particles = [ P.e__plus__, P.e__minus__, P.A ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_303})

V_207 = Vertex(name = 'V_207',
               particles = [ P.mu__plus__, P.mu__minus__, P.A ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_298})

V_208 = Vertex(name = 'V_208',
               particles = [ P.mu__plus__, P.mu__minus__, P.A ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_303})

V_209 = Vertex(name = 'V_209',
               particles = [ P.ta__plus__, P.ta__minus__, P.A ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_298})

V_210 = Vertex(name = 'V_210',
               particles = [ P.ta__plus__, P.ta__minus__, P.A ],
               color = [ '1' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_303})

V_211 = Vertex(name = 'V_211',
               particles = [ P.u__tilde__, P.u, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_297})

V_212 = Vertex(name = 'V_212',
               particles = [ P.u__tilde__, P.u, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_302})

V_213 = Vertex(name = 'V_213',
               particles = [ P.c__tilde__, P.c, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_297})

V_214 = Vertex(name = 'V_214',
               particles = [ P.c__tilde__, P.c, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_302})

V_215 = Vertex(name = 'V_215',
               particles = [ P.t__tilde__, P.t, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_297})

V_216 = Vertex(name = 'V_216',
               particles = [ P.t__tilde__, P.t, P.A ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_302})

V_217 = Vertex(name = 'V_217',
               particles = [ P.d__tilde__, P.d, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_158})

V_218 = Vertex(name = 'V_218',
               particles = [ P.s__tilde__, P.s, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_158})

V_219 = Vertex(name = 'V_219',
               particles = [ P.b__tilde__, P.b, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_158})

V_220 = Vertex(name = 'V_220',
               particles = [ P.u__tilde__, P.u, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_158})

V_221 = Vertex(name = 'V_221',
               particles = [ P.c__tilde__, P.c, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_158})

V_222 = Vertex(name = 'V_222',
               particles = [ P.t__tilde__, P.t, P.g ],
               color = [ 'T(3,2,1)' ],
               lorentz = [ L.FFV1 ],
               couplings = {(0,0):C.GC_158})

V_223 = Vertex(name = 'V_223',
               particles = [ P.e__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_396})

V_224 = Vertex(name = 'V_224',
               particles = [ P.mu__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_399})

V_225 = Vertex(name = 'V_225',
               particles = [ P.ta__plus__, P.ve, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_402})

V_226 = Vertex(name = 'V_226',
               particles = [ P.e__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_397})

V_227 = Vertex(name = 'V_227',
               particles = [ P.mu__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_400})

V_228 = Vertex(name = 'V_228',
               particles = [ P.ta__plus__, P.vm, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_403})

V_229 = Vertex(name = 'V_229',
               particles = [ P.e__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_398})

V_230 = Vertex(name = 'V_230',
               particles = [ P.mu__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_401})

V_231 = Vertex(name = 'V_231',
               particles = [ P.ta__plus__, P.vt, P.W__minus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_404})

V_232 = Vertex(name = 'V_232',
               particles = [ P.d__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_178})

V_233 = Vertex(name = 'V_233',
               particles = [ P.s__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_179})

V_234 = Vertex(name = 'V_234',
               particles = [ P.b__tilde__, P.u, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_180})

V_235 = Vertex(name = 'V_235',
               particles = [ P.d__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_181})

V_236 = Vertex(name = 'V_236',
               particles = [ P.s__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_182})

V_237 = Vertex(name = 'V_237',
               particles = [ P.b__tilde__, P.c, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_183})

V_238 = Vertex(name = 'V_238',
               particles = [ P.d__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_184})

V_239 = Vertex(name = 'V_239',
               particles = [ P.s__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_185})

V_240 = Vertex(name = 'V_240',
               particles = [ P.b__tilde__, P.t, P.W__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_186})

V_241 = Vertex(name = 'V_241',
               particles = [ P.u__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_387})

V_242 = Vertex(name = 'V_242',
               particles = [ P.c__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_390})

V_243 = Vertex(name = 'V_243',
               particles = [ P.t__tilde__, P.d, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_393})

V_244 = Vertex(name = 'V_244',
               particles = [ P.u__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_388})

V_245 = Vertex(name = 'V_245',
               particles = [ P.c__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_391})

V_246 = Vertex(name = 'V_246',
               particles = [ P.t__tilde__, P.s, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_394})

V_247 = Vertex(name = 'V_247',
               particles = [ P.u__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_389})

V_248 = Vertex(name = 'V_248',
               particles = [ P.c__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_392})

V_249 = Vertex(name = 'V_249',
               particles = [ P.t__tilde__, P.b, P.W__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_395})

V_250 = Vertex(name = 'V_250',
               particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_189})

V_251 = Vertex(name = 'V_251',
               particles = [ P.vm__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_190})

V_252 = Vertex(name = 'V_252',
               particles = [ P.vt__tilde__, P.e__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_191})

V_253 = Vertex(name = 'V_253',
               particles = [ P.ve__tilde__, P.mu__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_192})

V_254 = Vertex(name = 'V_254',
               particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_193})

V_255 = Vertex(name = 'V_255',
               particles = [ P.vt__tilde__, P.mu__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_194})

V_256 = Vertex(name = 'V_256',
               particles = [ P.ve__tilde__, P.ta__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_195})

V_257 = Vertex(name = 'V_257',
               particles = [ P.vm__tilde__, P.ta__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_196})

V_258 = Vertex(name = 'V_258',
               particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__ ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_197})

V_259 = Vertex(name = 'V_259',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_242,(0,1):C.GC_244})

V_260 = Vertex(name = 'V_260',
               particles = [ P.d__tilde__, P.d, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_284,(0,1):C.GC_285})

V_261 = Vertex(name = 'V_261',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_242,(0,1):C.GC_244})

V_262 = Vertex(name = 'V_262',
               particles = [ P.s__tilde__, P.s, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_284,(0,1):C.GC_285})

V_263 = Vertex(name = 'V_263',
               particles = [ P.b__tilde__, P.b, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_242,(0,1):C.GC_244})

V_264 = Vertex(name = 'V_264',
               particles = [ P.b__tilde__, P.b, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_284,(0,1):C.GC_285})

V_265 = Vertex(name = 'V_265',
               particles = [ P.e__plus__, P.e__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_245,(0,1):C.GC_247})

V_266 = Vertex(name = 'V_266',
               particles = [ P.e__plus__, P.e__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_288,(0,1):C.GC_289})

V_267 = Vertex(name = 'V_267',
               particles = [ P.mu__plus__, P.mu__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_245,(0,1):C.GC_247})

V_268 = Vertex(name = 'V_268',
               particles = [ P.mu__plus__, P.mu__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_288,(0,1):C.GC_289})

V_269 = Vertex(name = 'V_269',
               particles = [ P.ta__plus__, P.ta__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_245,(0,1):C.GC_247})

V_270 = Vertex(name = 'V_270',
               particles = [ P.ta__plus__, P.ta__minus__, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_288,(0,1):C.GC_289})

V_271 = Vertex(name = 'V_271',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_243,(0,1):C.GC_246})

V_272 = Vertex(name = 'V_272',
               particles = [ P.u__tilde__, P.u, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_286,(0,1):C.GC_287})

V_273 = Vertex(name = 'V_273',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_243,(0,1):C.GC_246})

V_274 = Vertex(name = 'V_274',
               particles = [ P.c__tilde__, P.c, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_286,(0,1):C.GC_287})

V_275 = Vertex(name = 'V_275',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_243,(0,1):C.GC_246})

V_276 = Vertex(name = 'V_276',
               particles = [ P.t__tilde__, P.t, P.Z ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFV2, L.FFV3 ],
               couplings = {(0,0):C.GC_286,(0,1):C.GC_287})

V_277 = Vertex(name = 'V_277',
               particles = [ P.ve__tilde__, P.ve, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_236})

V_278 = Vertex(name = 'V_278',
               particles = [ P.ve__tilde__, P.ve, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_237})

V_279 = Vertex(name = 'V_279',
               particles = [ P.vm__tilde__, P.vm, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_236})

V_280 = Vertex(name = 'V_280',
               particles = [ P.vm__tilde__, P.vm, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_237})

V_281 = Vertex(name = 'V_281',
               particles = [ P.vt__tilde__, P.vt, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_236})

V_282 = Vertex(name = 'V_282',
               particles = [ P.vt__tilde__, P.vt, P.Z ],
               color = [ '1' ],
               lorentz = [ L.FFV2 ],
               couplings = {(0,0):C.GC_237})

