(* ::Package:: *)

Get["MMA2TxtFormat3.m"]

ReadPredictionString[pred_String]:=Module[
{result,sublist,short,bindices,bindices$max},
result=StringSplit[pred,"&"];
If[Length[result]=!=2,Print["ReadPredictionString failure 1"];Return[False]];
{sublist,short}=result;
sublist=StringSplit[sublist,"$"];
result={sublist,short}/. str_String:>JF$Parser[str];
bindices=DeleteDuplicates@Cases[short,b[n_]:>n,Infinity];
bindices$max=Max[bindices];
If[Length[sublist]<bindices$max+1,Print["ReadPredictionString failure 2"];Return[False]];

result
];

MMASubtituteCheck[long_,sublist_,short_]:=Module[
{sub$result},
sub$result=short/.b[n_]:>sublist[[n+1]]//Expand;

sub$result===Expand[long]
];

MMACheck[str1_,str2_]:=Module[
{long,sublist,short,result},
long=JF$Parser[str1];
{sublist,short}=ReadPredictionString[str2];

Print[long-Expand[short/.b[n_]:>sublist[[n+1]]]];

result=MMASubtituteCheck[long,sublist,short];

If[result,Print["MMACheck Succeed"],Print["MMACheck Failed"]];
Return[True]
];
