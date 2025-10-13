# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 13.0.1 for Linux x86 (64-bit) (January 29, 2022)
# Date: Mon 8 Jan 2024 12:54:06


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
	particles = [ P.G0, P.G0, P.G0, P.G0 ],
	color = [ '1' ],
	lorentz = [ L.SSSS1 ],
	couplings = {(0, 0):C.GC_59})

V_2 = Vertex(name = 'V_2',
	particles = [ P.G0, P.G0, P.G__minus__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.SSSS1 ],
	couplings = {(0, 0):C.GC_57})

V_3 = Vertex(name = 'V_3',
	particles = [ P.G__minus__, P.G__minus__, P.G__plus__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.SSSS1 ],
	couplings = {(0, 0):C.GC_58})

V_4 = Vertex(name = 'V_4',
	particles = [ P.G0, P.G0, P.H ],
	color = [ '1' ],
	lorentz = [ L.SSS1 ],
	couplings = {(0, 0):C.GC_55})

V_5 = Vertex(name = 'V_5',
	particles = [ P.G__minus__, P.G__plus__, P.H ],
	color = [ '1' ],
	lorentz = [ L.SSS1 ],
	couplings = {(0, 0):C.GC_55})

V_6 = Vertex(name = 'V_6',
	particles = [ P.G0, P.G0, P.H, P.H ],
	color = [ '1' ],
	lorentz = [ L.SSSS1 ],
	couplings = {(0, 0):C.GC_57})

V_7 = Vertex(name = 'V_7',
	particles = [ P.G__minus__, P.G__plus__, P.H, P.H ],
	color = [ '1' ],
	lorentz = [ L.SSSS1 ],
	couplings = {(0, 0):C.GC_57})

V_8 = Vertex(name = 'V_8',
	particles = [ P.H, P.H, P.H ],
	color = [ '1' ],
	lorentz = [ L.SSS1 ],
	couplings = {(0, 0):C.GC_56})

V_9 = Vertex(name = 'V_9',
	particles = [ P.H, P.H, P.H, P.H ],
	color = [ '1' ],
	lorentz = [ L.SSSS1 ],
	couplings = {(0, 0):C.GC_59})

V_10 = Vertex(name = 'V_10',
	particles = [ P.ghWp, P.ghZ__tilde__, P.G__minus__ ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_297_1})

V_11 = Vertex(name = 'V_11',
	particles = [ P.ghWm, P.ghZ__tilde__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_297_1})

V_12 = Vertex(name = 'V_12',
	particles = [ P.ghWm, P.ghWm__tilde__, P.G0 ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_275_1})

V_13 = Vertex(name = 'V_13',
	particles = [ P.ghWp, P.ghWp__tilde__, P.G0 ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_273_1})

V_14 = Vertex(name = 'V_14',
	particles = [ P.ghWm, P.ghWm__tilde__, P.H ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_274_1})

V_15 = Vertex(name = 'V_15',
	particles = [ P.ghWp, P.ghWp__tilde__, P.H ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_274_1})

V_16 = Vertex(name = 'V_16',
	particles = [ P.ghZ, P.ghWm__tilde__, P.G__minus__ ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_283_1})

V_17 = Vertex(name = 'V_17',
	particles = [ P.ghZ, P.ghWp__tilde__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_283_1})

V_18 = Vertex(name = 'V_18',
	particles = [ P.ghZ, P.ghZ__tilde__, P.H ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_286_1})

V_19 = Vertex(name = 'V_19',
	particles = [ P.ghWp, P.ghA__tilde__, P.G__minus__ ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_307_1})

V_20 = Vertex(name = 'V_20',
	particles = [ P.ghWm, P.ghA__tilde__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.UUS1 ],
	couplings = {(0, 0):C.GC_307_1})

V_21 = Vertex(name = 'V_21',
	particles = [ P.A, P.A, P.G__minus__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_277})

V_22 = Vertex(name = 'V_22',
	particles = [ P.ghWm, P.ghWm__tilde__, P.A ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_301_1})

V_23 = Vertex(name = 'V_23',
	particles = [ P.ghWp, P.ghWp__tilde__, P.A ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_302_1})

V_24 = Vertex(name = 'V_24',
	particles = [ P.A, P.G__minus__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VSS1 ],
	couplings = {(0, 0):C.GC_302})

V_29 = Vertex(name = 'V_29',
	particles = [ P.g, P.g, P.g ],
	color = [ 'f(1,2,3)' ],
	lorentz = [ L.VVV1 ],
	couplings = {(0, 0):C.GC_243})

V_34 = Vertex(name = 'V_34',
	particles = [ P.g, P.g, P.g, P.g ],
	color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
	lorentz = [ L.VVVV1, L.VVVV3, L.VVVV4 ],
	couplings = {(1, 1):C.GC_246, (0, 0):C.GC_246, (2, 2):C.GC_246})

V_39 = Vertex(name = 'V_39',
	particles = [ P.ghG, P.ghG__tilde__, P.g ],
	color = [ 'f(1,2,3)' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_243_1})

V_101 = Vertex(name = 'V_101',
	particles = [ P.b__tilde__, P.b, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFS3, L.FFS4 ],
	couplings = {(0, 0):C.GC_17, (0, 1):C.GC_17})

V_192 = Vertex(name = 'V_192',
	particles = [ P.b__tilde__, P.b, P.G0 ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFS1 ],
	couplings = {(0, 0):C.GC_18})

V_201 = Vertex(name = 'V_201',
	particles = [ P.t__tilde__, P.b, P.G__plus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFS3, L.FFS4 ],
	couplings = {(0, 0):C.GC_68, (0, 1):C.GC_77})

V_210 = Vertex(name = 'V_210',
	particles = [ P.ta__plus__, P.ta__minus__, P.G0 ],
	color = [ '1' ],
	lorentz = [ L.FFS1 ],
	couplings = {(0, 0):C.GC_36})

V_219 = Vertex(name = 'V_219',
	particles = [ P.ta__plus__, P.ta__minus__, P.H ],
	color = [ '1' ],
	lorentz = [ L.FFS2 ],
	couplings = {(0, 0):C.GC_35})

V_228 = Vertex(name = 'V_228',
	particles = [ P.t__tilde__, P.t, P.G0 ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFS1 ],
	couplings = {(0, 0):C.GC_53})

V_237 = Vertex(name = 'V_237',
	particles = [ P.t__tilde__, P.t, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFS2 ],
	couplings = {(0, 0):C.GC_54})

V_246 = Vertex(name = 'V_246',
	particles = [ P.b__tilde__, P.t, P.G__minus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFS3, L.FFS4 ],
	couplings = {(0, 0):C.GC_86, (0, 1):C.GC_95})

V_255 = Vertex(name = 'V_255',
	particles = [ P.ta__plus__, P.vt, P.G__minus__ ],
	color = [ '1' ],
	lorentz = [ L.FFS4 ],
	couplings = {(0, 0):C.GC_104})

V_300 = Vertex(name = 'V_300',
	particles = [ P.vt__tilde__, P.ta__minus__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.FFS3 ],
	couplings = {(0, 0):C.GC_113})

V_301 = Vertex(name = 'V_301',
	particles = [ P.A, P.W__minus__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVS1 ],
	couplings = {(0, 0):C.GC_303})

V_302 = Vertex(name = 'V_302',
	particles = [ P.A, P.W__minus__, P.G0, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_304})

V_303 = Vertex(name = 'V_303',
	particles = [ P.A, P.W__minus__, P.G__plus__, P.H ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_305})

V_304 = Vertex(name = 'V_304',
	particles = [ P.W__minus__, P.G0, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VSS1 ],
	couplings = {(0, 0):C.GC_249})

V_305 = Vertex(name = 'V_305',
	particles = [ P.ghA, P.ghWm__tilde__, P.W__minus__ ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_302_1})

V_306 = Vertex(name = 'V_306',
	particles = [ P.ghWp, P.ghZ__tilde__, P.W__minus__ ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_279_1})

V_307 = Vertex(name = 'V_307',
	particles = [ P.ghWp, P.ghA__tilde__, P.W__minus__ ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_301_1})

V_308 = Vertex(name = 'V_308',
	particles = [ P.ghZ, P.ghWm__tilde__, P.W__minus__ ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_280_1})

V_309 = Vertex(name = 'V_309',
	particles = [ P.W__minus__, P.G__plus__, P.H ],
	color = [ '1' ],
	lorentz = [ L.VSS1 ],
	couplings = {(0, 0):C.GC_250})

V_310 = Vertex(name = 'V_310',
	particles = [ P.A, P.W__minus__, P.W__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVV1 ],
	couplings = {(0, 0):C.GC_301})

V_311 = Vertex(name = 'V_311',
	particles = [ P.A, P.W__plus__, P.G__minus__ ],
	color = [ '1' ],
	lorentz = [ L.VVS1 ],
	couplings = {(0, 0):C.GC_303})

V_312 = Vertex(name = 'V_312',
	particles = [ P.A, P.W__plus__, P.G0, P.G__minus__ ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_306})

V_313 = Vertex(name = 'V_313',
	particles = [ P.A, P.W__plus__, P.G__minus__, P.H ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_305})

V_314 = Vertex(name = 'V_314',
	particles = [ P.W__plus__, P.G0, P.G__minus__ ],
	color = [ '1' ],
	lorentz = [ L.VSS1 ],
	couplings = {(0, 0):C.GC_249})

V_315 = Vertex(name = 'V_315',
	particles = [ P.ghA, P.ghWp__tilde__, P.W__plus__ ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_301_1})

V_316 = Vertex(name = 'V_316',
	particles = [ P.ghWm, P.ghZ__tilde__, P.W__plus__ ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_280_1})

V_317 = Vertex(name = 'V_317',
	particles = [ P.ghWm, P.ghA__tilde__, P.W__plus__ ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_302_1})

V_318 = Vertex(name = 'V_318',
	particles = [ P.ghZ, P.ghWp__tilde__, P.W__plus__ ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_279_1})

V_319 = Vertex(name = 'V_319',
	particles = [ P.W__plus__, P.G__minus__, P.H ],
	color = [ '1' ],
	lorentz = [ L.VSS1 ],
	couplings = {(0, 0):C.GC_251})

V_320 = Vertex(name = 'V_320',
	particles = [ P.W__minus__, P.W__plus__, P.G0, P.G0 ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_271})

V_321 = Vertex(name = 'V_321',
	particles = [ P.W__minus__, P.W__plus__, P.G__minus__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_271})

V_322 = Vertex(name = 'V_322',
	particles = [ P.W__minus__, P.W__plus__, P.H ],
	color = [ '1' ],
	lorentz = [ L.VVS1 ],
	couplings = {(0, 0):C.GC_270})

V_323 = Vertex(name = 'V_323',
	particles = [ P.W__minus__, P.W__plus__, P.H, P.H ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_271})

V_324 = Vertex(name = 'V_324',
	particles = [ P.A, P.A, P.W__minus__, P.W__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVVV2 ],
	couplings = {(0, 0):C.GC_276})

V_325 = Vertex(name = 'V_325',
	particles = [ P.W__minus__, P.W__plus__, P.Z ],
	color = [ '1' ],
	lorentz = [ L.VVV1 ],
	couplings = {(0, 0):C.GC_279})

V_326 = Vertex(name = 'V_326',
	particles = [ P.W__minus__, P.W__minus__, P.W__plus__, P.W__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVVV2 ],
	couplings = {(0, 0):C.GC_272})

V_327 = Vertex(name = 'V_327',
	particles = [ P.A, P.Z, P.G__minus__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_309})

V_328 = Vertex(name = 'V_328',
	particles = [ P.Z, P.G0, P.H ],
	color = [ '1' ],
	lorentz = [ L.VSS1 ],
	couplings = {(0, 0):C.GC_282})

V_329 = Vertex(name = 'V_329',
	particles = [ P.ghWm, P.ghWm__tilde__, P.Z ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_279_1})

V_330 = Vertex(name = 'V_330',
	particles = [ P.ghWp, P.ghWp__tilde__, P.Z ],
	color = [ '1' ],
	lorentz = [ L.UUV1 ],
	couplings = {(0, 0):C.GC_280_1})

V_331 = Vertex(name = 'V_331',
	particles = [ P.Z, P.G__minus__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VSS1 ],
	couplings = {(0, 0):C.GC_290})

V_332 = Vertex(name = 'V_332',
	particles = [ P.W__minus__, P.Z, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVS1 ],
	couplings = {(0, 0):C.GC_293})

V_333 = Vertex(name = 'V_333',
	particles = [ P.W__minus__, P.Z, P.G0, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_296})

V_334 = Vertex(name = 'V_334',
	particles = [ P.W__minus__, P.Z, P.G__plus__, P.H ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_295})

V_335 = Vertex(name = 'V_335',
	particles = [ P.W__plus__, P.Z, P.G__minus__ ],
	color = [ '1' ],
	lorentz = [ L.VVS1 ],
	couplings = {(0, 0):C.GC_293})

V_336 = Vertex(name = 'V_336',
	particles = [ P.W__plus__, P.Z, P.G0, P.G__minus__ ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_294})

V_337 = Vertex(name = 'V_337',
	particles = [ P.W__plus__, P.Z, P.G__minus__, P.H ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_295})

V_338 = Vertex(name = 'V_338',
	particles = [ P.A, P.W__minus__, P.W__plus__, P.Z ],
	color = [ '1' ],
	lorentz = [ L.VVVV5 ],
	couplings = {(0, 0):C.GC_298})

V_339 = Vertex(name = 'V_339',
	particles = [ P.Z, P.Z, P.G__minus__, P.G__plus__ ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_308})

V_340 = Vertex(name = 'V_340',
	particles = [ P.Z, P.Z, P.G0, P.G0 ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_285})

V_341 = Vertex(name = 'V_341',
	particles = [ P.Z, P.Z, P.H ],
	color = [ '1' ],
	lorentz = [ L.VVS1 ],
	couplings = {(0, 0):C.GC_284})

V_342 = Vertex(name = 'V_342',
	particles = [ P.Z, P.Z, P.H, P.H ],
	color = [ '1' ],
	lorentz = [ L.VVSS1 ],
	couplings = {(0, 0):C.GC_285})

V_343 = Vertex(name = 'V_343',
	particles = [ P.W__minus__, P.W__plus__, P.Z, P.Z ],
	color = [ '1' ],
	lorentz = [ L.VVVV2 ],
	couplings = {(0, 0):C.GC_278})

V_344 = Vertex(name = 'V_344',
	particles = [ P.d__tilde__, P.d, P.A ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_299})

V_345 = Vertex(name = 'V_345',
	particles = [ P.s__tilde__, P.s, P.A ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_299})

V_346 = Vertex(name = 'V_346',
	particles = [ P.b__tilde__, P.b, P.A ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_299})

V_347 = Vertex(name = 'V_347',
	particles = [ P.e__plus__, P.e__minus__, P.A ],
	color = [ '1' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_302})

V_348 = Vertex(name = 'V_348',
	particles = [ P.mu__plus__, P.mu__minus__, P.A ],
	color = [ '1' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_302})

V_349 = Vertex(name = 'V_349',
	particles = [ P.ta__plus__, P.ta__minus__, P.A ],
	color = [ '1' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_302})

V_350 = Vertex(name = 'V_350',
	particles = [ P.u__tilde__, P.u, P.A ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_300})

V_351 = Vertex(name = 'V_351',
	particles = [ P.c__tilde__, P.c, P.A ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_300})

V_352 = Vertex(name = 'V_352',
	particles = [ P.t__tilde__, P.t, P.A ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_300})

V_353 = Vertex(name = 'V_353',
	particles = [ P.d__tilde__, P.d, P.g ],
	color = [ 'T(3,2,1)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_242})

V_354 = Vertex(name = 'V_354',
	particles = [ P.s__tilde__, P.s, P.g ],
	color = [ 'T(3,2,1)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_242})

V_355 = Vertex(name = 'V_355',
	particles = [ P.b__tilde__, P.b, P.g ],
	color = [ 'T(3,2,1)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_242})

V_356 = Vertex(name = 'V_356',
	particles = [ P.u__tilde__, P.u, P.g ],
	color = [ 'T(3,2,1)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_242})

V_357 = Vertex(name = 'V_357',
	particles = [ P.c__tilde__, P.c, P.g ],
	color = [ 'T(3,2,1)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_242})

V_358 = Vertex(name = 'V_358',
	particles = [ P.t__tilde__, P.t, P.g ],
	color = [ 'T(3,2,1)' ],
	lorentz = [ L.FFV1 ],
	couplings = {(0, 0):C.GC_242})

V_359 = Vertex(name = 'V_359',
	particles = [ P.e__plus__, P.ve, P.W__minus__ ],
	color = [ '1' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_391})

V_363 = Vertex(name = 'V_363',
	particles = [ P.mu__plus__, P.vm, P.W__minus__ ],
	color = [ '1' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_395})

V_367 = Vertex(name = 'V_367',
	particles = [ P.ta__plus__, P.vt, P.W__minus__ ],
	color = [ '1' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_399})

V_368 = Vertex(name = 'V_368',
	particles = [ P.d__tilde__, P.u, P.W__minus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_252})

V_372 = Vertex(name = 'V_372',
	particles = [ P.s__tilde__, P.c, P.W__minus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_256})

V_376 = Vertex(name = 'V_376',
	particles = [ P.b__tilde__, P.t, P.W__minus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_260})

V_377 = Vertex(name = 'V_377',
	particles = [ P.u__tilde__, P.d, P.W__plus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_382})

V_381 = Vertex(name = 'V_381',
	particles = [ P.c__tilde__, P.s, P.W__plus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_386})

V_385 = Vertex(name = 'V_385',
	particles = [ P.t__tilde__, P.b, P.W__plus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_390})

V_386 = Vertex(name = 'V_386',
	particles = [ P.ve__tilde__, P.e__minus__, P.W__plus__ ],
	color = [ '1' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_261})

V_390 = Vertex(name = 'V_390',
	particles = [ P.vm__tilde__, P.mu__minus__, P.W__plus__ ],
	color = [ '1' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_265})

V_394 = Vertex(name = 'V_394',
	particles = [ P.vt__tilde__, P.ta__minus__, P.W__plus__ ],
	color = [ '1' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_269})

V_395 = Vertex(name = 'V_395',
	particles = [ P.d__tilde__, P.d, P.Z ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2, L.FFV3 ],
	couplings = {(0, 0):C.GC_287, (0, 1):C.GC_289})

V_396 = Vertex(name = 'V_396',
	particles = [ P.s__tilde__, P.s, P.Z ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2, L.FFV3 ],
	couplings = {(0, 0):C.GC_287, (0, 1):C.GC_289})

V_397 = Vertex(name = 'V_397',
	particles = [ P.b__tilde__, P.b, P.Z ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2, L.FFV3 ],
	couplings = {(0, 0):C.GC_287, (0, 1):C.GC_289})

V_398 = Vertex(name = 'V_398',
	particles = [ P.e__plus__, P.e__minus__, P.Z ],
	color = [ '1' ],
	lorentz = [ L.FFV2, L.FFV3 ],
	couplings = {(0, 0):C.GC_290, (0, 1):C.GC_292})

V_399 = Vertex(name = 'V_399',
	particles = [ P.mu__plus__, P.mu__minus__, P.Z ],
	color = [ '1' ],
	lorentz = [ L.FFV2, L.FFV3 ],
	couplings = {(0, 0):C.GC_290, (0, 1):C.GC_292})

V_400 = Vertex(name = 'V_400',
	particles = [ P.ta__plus__, P.ta__minus__, P.Z ],
	color = [ '1' ],
	lorentz = [ L.FFV2, L.FFV3 ],
	couplings = {(0, 0):C.GC_290, (0, 1):C.GC_292})

V_401 = Vertex(name = 'V_401',
	particles = [ P.u__tilde__, P.u, P.Z ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2, L.FFV3 ],
	couplings = {(0, 0):C.GC_288, (0, 1):C.GC_291})

V_402 = Vertex(name = 'V_402',
	particles = [ P.c__tilde__, P.c, P.Z ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2, L.FFV3 ],
	couplings = {(0, 0):C.GC_288, (0, 1):C.GC_291})

V_403 = Vertex(name = 'V_403',
	particles = [ P.t__tilde__, P.t, P.Z ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFV2, L.FFV3 ],
	couplings = {(0, 0):C.GC_288, (0, 1):C.GC_291})

V_404 = Vertex(name = 'V_404',
	particles = [ P.ve__tilde__, P.ve, P.Z ],
	color = [ '1' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_281})

V_405 = Vertex(name = 'V_405',
	particles = [ P.vm__tilde__, P.vm, P.Z ],
	color = [ '1' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_281})

V_406 = Vertex(name = 'V_406',
	particles = [ P.vt__tilde__, P.vt, P.Z ],
	color = [ '1' ],
	lorentz = [ L.FFV2 ],
	couplings = {(0, 0):C.GC_281})



#### NP1 vertices ####
V_25 = Vertex(name = 'V_25',
	particles = [ P.g, P.g, P.G0, P.G0 ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.VVSS2 ],
	couplings = {(0, 0):C.GC_150})

V_26 = Vertex(name = 'V_26',
	particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.VVSS2 ],
	couplings = {(0, 0):C.GC_150})

V_27 = Vertex(name = 'V_27',
	particles = [ P.g, P.g, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.VVS2 ],
	couplings = {(0, 0):C.GC_187})

V_28 = Vertex(name = 'V_28',
	particles = [ P.g, P.g, P.H, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.VVSS2 ],
	couplings = {(0, 0):C.GC_150})

V_30 = Vertex(name = 'V_30',
	particles = [ P.g, P.g, P.g, P.G0, P.G0 ],
	color = [ 'f(1,2,3)' ],
	lorentz = [ L.VVVSS1 ],
	couplings = {(0, 0):C.GC_244})

V_31 = Vertex(name = 'V_31',
	particles = [ P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
	color = [ 'f(1,2,3)' ],
	lorentz = [ L.VVVSS1 ],
	couplings = {(0, 0):C.GC_244})

V_32 = Vertex(name = 'V_32',
	particles = [ P.g, P.g, P.g, P.H ],
	color = [ 'f(1,2,3)' ],
	lorentz = [ L.VVVS1 ],
	couplings = {(0, 0):C.GC_245})

V_33 = Vertex(name = 'V_33',
	particles = [ P.g, P.g, P.g, P.H, P.H ],
	color = [ 'f(1,2,3)' ],
	lorentz = [ L.VVVSS1 ],
	couplings = {(0, 0):C.GC_244})

V_35 = Vertex(name = 'V_35',
	particles = [ P.g, P.g, P.g, P.g, P.G0, P.G0 ],
	color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
	lorentz = [ L.VVVVSS1, L.VVVVSS2, L.VVVVSS3 ],
	couplings = {(1, 1):C.GC_247, (0, 0):C.GC_247, (2, 2):C.GC_247})

V_36 = Vertex(name = 'V_36',
	particles = [ P.g, P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
	color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
	lorentz = [ L.VVVVSS1, L.VVVVSS2, L.VVVVSS3 ],
	couplings = {(1, 1):C.GC_247, (0, 0):C.GC_247, (2, 2):C.GC_247})

V_37 = Vertex(name = 'V_37',
	particles = [ P.g, P.g, P.g, P.g, P.H ],
	color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
	lorentz = [ L.VVVVS1, L.VVVVS2, L.VVVVS3 ],
	couplings = {(1, 1):C.GC_248, (0, 0):C.GC_248, (2, 2):C.GC_248})

V_38 = Vertex(name = 'V_38',
	particles = [ P.g, P.g, P.g, P.g, P.H, P.H ],
	color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
	lorentz = [ L.VVVVSS1, L.VVVVSS2, L.VVVVSS3 ],
	couplings = {(1, 1):C.GC_247, (0, 0):C.GC_247, (2, 2):C.GC_247})

V_48 = Vertex(name = 'V_48',
	particles = [ P.b__tilde__, P.b, P.G0, P.G0, P.G0 ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	couplings = {(0, 0):C.GC_377, (0, 1):C.GC_146})

V_57 = Vertex(name = 'V_57',
	particles = [ P.b__tilde__, P.b, P.G0, P.G0 ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSS1, L.FFSS2 ],
	couplings = {(0, 0):C.GC_379, (0, 1):C.GC_185})

V_66 = Vertex(name = 'V_66',
	particles = [ P.b__tilde__, P.b, P.G0, P.G__minus__, P.G__plus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	couplings = {(0, 0):C.GC_376, (0, 1):C.GC_147})

V_75 = Vertex(name = 'V_75',
	particles = [ P.b__tilde__, P.b, P.G__minus__, P.G__plus__ ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSS1, L.FFSS2 ],
	couplings = {(0, 0):C.GC_379, (0, 1):C.GC_185})

V_84 = Vertex(name = 'V_84',
	particles = [ P.b__tilde__, P.b, P.G0, P.G0, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	couplings = {(0, 0):C.GC_374, (0, 1):C.GC_148})

V_102 = Vertex(name = 'V_102',
	particles = [ P.b__tilde__, P.b, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFS3, L.FFS4 ],
	couplings = {(0, 0):C.GC_378, (0, 1):C.GC_159})

V_111 = Vertex(name = 'V_111',
	particles = [ P.b__tilde__, P.b, P.G0, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSS1, L.FFSS2 ],
	couplings = {(0, 0):C.GC_381, (0, 1):C.GC_184})

V_120 = Vertex(name = 'V_120',
	particles = [ P.b__tilde__, P.b, P.G__minus__, P.G__plus__, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	couplings = {(0, 0):C.GC_374, (0, 1):C.GC_148})

V_129 = Vertex(name = 'V_129',
	particles = [ P.b__tilde__, P.b, P.G0, P.H, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	couplings = {(0, 0):C.GC_376, (0, 1):C.GC_147})

V_138 = Vertex(name = 'V_138',
	particles = [ P.b__tilde__, P.b, P.H, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSS1, L.FFSS2 ],
	couplings = {(0, 0):C.GC_380, (0, 1):C.GC_186})

V_147 = Vertex(name = 'V_147',
	particles = [ P.b__tilde__, P.b, P.H, P.H, P.H ],
	color = [ 'Identity(1,2)' ],
	lorentz = [ L.FFSSS1, L.FFSSS2 ],
	couplings = {(0, 0):C.GC_375, (0, 1):C.GC_149})

