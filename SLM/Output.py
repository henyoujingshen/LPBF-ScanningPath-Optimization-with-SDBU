# Auther : REN Huilin
# Date : 2021/12/20 20:34
# File : StrainOutput
# Function:
# IDE : PyCharm
# region
from odbAccess import *
from abaqusConstants import *
import sys
import os
import numpy
import csv

fileOdb_TEMP = 'I:\\SLM\\Job_TEMP.odb'
#fileOdb_TEMP = 'I:\\SLM\\job-5.odb'
# fileOdb_TEMP_cooling = 'I:\\SLM\\Job_TEMP_cooling.odb'

filenamecsvTEMP = 'I:\\SLM\\TEMP'+sys.argv[-1]+'.txt'
# filenamecsvTEMP_cooling = 'I:\\SLM\\TEMP_cooling.txt'


# odb_TEMP_cooling = openOdb(path=fileOdb_TEMP_cooling)
odb_TEMP = openOdb(path=fileOdb_TEMP)


Noderegion1 = odb_TEMP.rootAssembly.elementSets['SET-1']
# Noderegion2 = odb_TEMP_cooling.rootAssembly.elementSets['SET-1']

cpFile = open(filenamecsvTEMP, 'w')
NT = 0
maxElem = 0
maxFrame = 0
i = -1
for frame in odb_TEMP.steps['Step_thermal'].frames:
    fieldOutsValues = frame.fieldOutputs['TEMP'].getSubset(
        region=Noderegion1).values
    i = i + 1
    for ele in fieldOutsValues:
        NT = ele.data
        Elem = ele.elementLabel
        Frame = i
        cpFile.write('%8d\t%8d\t%8.9f\n' %
                     (Frame, Elem, NT))

# cpFile = open(filenamecsvTEMP_cooling, 'w')
# NT_cooling = 0
# maxElem = 0
# maxFrame = 0
# i = -1
# for frame in odb_TEMP_cooling.steps['Step_thermal'].frames:
#     fieldOutsValues = frame.fieldOutputs['TEMP'].getSubset(
#         region=Noderegion2).values
#     i = i + 1
#     for ele in fieldOutsValues:
#         NT_cooling = ele.data
#         Elem = ele.elementLabel
#         Frame = i
#         cpFile.write('%8d\t%8d\t%8.9f\n' %
#                      (Frame, Elem, NT_cooling))