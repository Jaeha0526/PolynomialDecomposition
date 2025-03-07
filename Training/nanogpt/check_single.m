(* ::Package:: *)

Get["MMA2TxtFormat3.m"]

PolynomialDecomposite[raw_,sub_]:=Module[
{coeff,result,reminder},
Clear[s,p];
result=p;
coeff=raw;
While[True,If[coeff==0,Break[]];
{{coeff},reminder}=PolynomialReduce[coeff,sub];
result=result/. p->p s+reminder;];
result=result/. p->0;result]

PolySimplifyWithHint[{expr$unsimplified_,expr$sub_}]:=Module[
{expr$simplifed,result},
result=If[NumericQ[expr$unsimplified],True,False];
expr$simplifed=Expand[PolynomialDecomposite[expr$unsimplified,expr$sub]];
result=If[True&&!FreeQ[expr$simplifed,_a],False,True];
result]

PolySimplifyWithHint[{expr$unsimplified_,False},___]:=False

MMACheck[str1_,str2_]:=Module[
{long,short,pre$result,result},
long=JF$Parser[str1];
short=JF$Parser[str2];

pre$result = PolySimplifyWithHint[{long,short}];
Print[pre$result];
result=Not[pre$result==False];

If[result,Print["MMACheck Succeed"],Print["MMACheck Failed"]];
Return[True]
];
