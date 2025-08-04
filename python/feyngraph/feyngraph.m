(* ::Package:: *)

BeginPackage["FeynGraph`"];
$FeynGraphVersion = ExternalEvaluate["Python", "from importlib.metadata import version; version(\"feyngraph\")"]
Print["Successfully imported FeynGraph (Version " <> $FeynGraphVersion <> ")."];


FGGenerateTopologies::usage = "Generate all topologies with `loops` loops and `legs` external legs.";
Options[FGGenerateTopologies] = {VertexDegrees -> {3, 4}};
FGGenerateTopologies[loops_Integer, inlegs_Integer -> outlegs_Integer, OptionsPattern[]] := ToExpression[
	ExternalEvaluate["Python",
		"from feyngraph.topology import TopologyGenerator, TopologyModel; TopologyGenerator(" 
		<> ToString[inlegs + outlegs]  <>
		", "
		<> ToString[loops] <>
		", TopologyModel(" <> ExportString[OptionValue[VertexDegrees], "PythonExpression"] <> ")).generate().to_feynarts(n_in="
		<> ToString[inlegs] <>
		")"
	]
];
FGGenerateTopologies[loops_Integer, legs_Integer, OptionsPattern[]] := ToExpression[
	ExternalEvaluate["Python",
		"from feyngraph.topology import TopologyGenerator, TopologyModel; TopologyGenerator(" 
		<> ToString[legs]  <>
		", "
		<> ToString[loops] <>
		", TopologyModel(" <> ExportString[OptionValue[VertexDegrees], "PythonExpression"] <> ")).generate().to_feynarts()"
	]
];


FGGenerateDiagrams::usage "Generate all Feynman diagrams with the given external legs and number of loops in the Standard Model in Feynman gauge."
FGGenerateDiagrams[loops_Integer, in_List -> out_List] := ToExpression[
	ExternalEvaluate["Python",
		"from feyngraph import _diagrams_feynarts; _diagrams_feynarts(" <>
		ExportString[in, "PythonExpression"] <> ", " <> ExportString[out, "PythonExpression"] <> ", " <> ToString[loops] <>
		")"
	]
];


EndPackage[];
