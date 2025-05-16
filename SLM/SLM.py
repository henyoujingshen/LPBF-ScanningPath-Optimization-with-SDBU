# Auther : REN Huilin
# Date : 2022/11
# Function: 用于本科毕业设计机器学习模型，RV for SLM 定步长，算的快，扫完一层铺粉0.02s，再扫一层，最终冷却100s

##########* IMPORT ##########
# region
import amKernelInit
import amModule
import customKernel
from amConstants import *
from abaqus import *
import sys
from amModule import *
from customKernel import *
from connectorBehavior import *
from visualization import *
from sketch import *
from job import *
from optimization import *
from mesh import *
from load import *
from interaction import *
from step import *
from assembly import *
from section import *
from material import *
from part import *
sys.path.append('c:\\SIMULIA\\CAE\\plugins\\2021\\AMModeler')
session.journalOptions.setValues(
    replayGeometry=COORDINATE, recoverGeometry=COORDINATE)


# endregion
Mdb()

# ##########* initialization ##########
# # region
#############* read the total time of laser scanning ############
FileID = 'I:\\SLM\\t_max5.txt'
with open(FileID, 'r') as f:
    for line in f:
        xx = line.strip('\n')
        Time = float(xx)


fileCae = 'I:\\SLM\\SLM_N5.cae'
fileOdb_TEMP = 'I:\\SLM\\Job_TEMP.odb'
fileOdb_TEMP_cooling = 'I:\\SLM\\Job_TEMP_cooling.odb'
fileInp_t = 'Job_TEMP'
fileInp_s = 'Job_PE_EE'
fileInp_tc = 'Job_TEMP_cooling'
fileInp_sc = 'Job_PE_EE_cooling'
filePP = 'I:\\SLM\\PP5.txt'
fileRP = 'I:\\SLM\\RP5.txt'


# endregion

#############* read the number of chessboards ############
FileID = 'I:\\SLM\\n5.txt'
with open(FileID, 'r') as f:
    for line in f:
        xx = line.strip('\n')
        N = float(xx)

##########* AM Process #########
# region
##########* THERMAL #########
##########* PART ###########
# region
mdb.models.changeKey(fromName='Model-1', toName='Thermal')
mdb.models['Thermal'].ConstrainedSketch(name='__profile__', sheetSize=20.0)
mdb.models['Thermal'].sketches['__profile__'].rectangle(point1=((-2.0/1000), (-2.0/1000)),
                                                        point2=((N+2)/1000, (N+2)/1000))
mdb.models['Thermal'].Part(dimensionality=THREE_D,
                           name='Substrate', type=DEFORMABLE_BODY)
mdb.models['Thermal'].parts['Substrate'].BaseSolidExtrude(
    depth=0.2/1000, sketch=mdb.models['Thermal'].sketches['__profile__'])
del mdb.models['Thermal'].sketches['__profile__']

mdb.models['Thermal'].ConstrainedSketch(name='__profile__', sheetSize=20.0)
mdb.models['Thermal'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
                                                        point2=(N/1000, N/1000))
mdb.models['Thermal'].Part(dimensionality=THREE_D,
                           name='Deposition', type=DEFORMABLE_BODY)
mdb.models['Thermal'].parts['Deposition'].BaseSolidExtrude(
    depth=0.1/1000, sketch=mdb.models['Thermal'].sketches['__profile__'])
del mdb.models['Thermal'].sketches['__profile__']
# endregion

##########* MATERIAL ###########
# region
mdb.models['Thermal'].Material(name='Ti-6Al-4V')
mdb.models['Thermal'].materials['Ti-6Al-4V'].SpecificHeat(dependencies=0, law=CONSTANTVOLUME, table=((546.0, 298.0), (606.0, 573.0), (694.0, 973.0), (
    697.0, 1268.0), (696.0, 1573.0), (795.0, 1923.0), (840.0, 1973.0)),
    temperatureDependency=ON)
mdb.models['Thermal'].materials['Ti-6Al-4V'].setValues(materialIdentifier='')
mdb.models['Thermal'].materials['Ti-6Al-4V'].setValues(description='')
mdb.models['Thermal'].materials['Ti-6Al-4V'].Elastic(dependencies=0, moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((104800310000.0, 0.31),
                                                                                                                                ), temperatureDependency=OFF, type=ISOTROPIC)
mdb.models['Thermal'].materials['Ti-6Al-4V'].Density(dependencies=0,
                                                     distributionType=UNIFORM, fieldName='', table=((4420.0, 298.0), (4381.0,
                                                                                                                      573.0), (4324.0, 973.0), (4282.0, 1268.0), (4240.0, 1573.0), (4054.5,
                                                                                                                                                                                    1923.0), (3886.0, 1973.0)), temperatureDependency=ON)
mdb.models['Thermal'].materials['Ti-6Al-4V'].Plastic(dataType=HALF_CYCLE,
                                                     dependencies=0, hardening=ISOTROPIC, numBackstresses=1, rate=OFF,
                                                     scaleStress=None, staticRecovery=OFF, strainRangeDependency=OFF, table=((
                                                         1050000000.0, 0.0), (1050000000.0, 0.01), (1145000000.0, 0.02), (
                                                         1160000000.0, 0.03), (1165000000.0, 0.04)), temperatureDependency=OFF)
mdb.models['Thermal'].materials['Ti-6Al-4V'].Conductivity(dependencies=0,
                                                          table=((7.0, 298.0), (10.15, 573.0), (15.5, 973.0), (21.0, 1268.0), (23.7,
                                                                                                                               1573.0), (30.9, 1923.0), (34.6, 1973.0)), temperatureDependency=ON, type=ISOTROPIC)
mdb.models['Thermal'].materials['Ti-6Al-4V'].Expansion(dependencies=0, table=((
    2.86e-06, 298.0), (3.76e-06, 572.0), (5.06e-06, 977.0), (6.83e-06, 1268.0),
    (7.76e-06, 1577.0), (8.57e-06, 1925.0), (8.99e-06, 1972.0)),
    temperatureDependency=ON, type=ISOTROPIC, userSubroutine=OFF, zero=0.0)
mdb.models['Thermal'].HomogeneousSolidSection(
    material='Ti-6Al-4V', name='Section-1', thickness=None)
mdb.models['Thermal'].parts['Deposition'].Set(cells=mdb.models['Thermal'].parts['Deposition'].cells.getSequenceFromMask((
    '[#1 ]', ), ), name='Set_deposition')
mdb.models['Thermal'].parts['Deposition'].SectionAssignment(offset=0.0,
                                                            offsetField='', offsetType=MIDDLE_SURFACE, region=mdb.models['Thermal'].parts['Deposition'].sets['Set_deposition'],
                                                            sectionName='Section-1', thicknessAssignment=FROM_SECTION)
mdb.models['Thermal'].parts['Substrate'].Set(cells=mdb.models['Thermal'].parts['Substrate'].cells.getSequenceFromMask((
    '[#1 ]', ), ), name='Set_substrate')
mdb.models['Thermal'].parts['Substrate'].SectionAssignment(offset=0.0,
                                                           offsetField='', offsetType=MIDDLE_SURFACE, region=mdb.models['Thermal'].parts['Substrate'].sets['Set_substrate'],
                                                           sectionName='Section-1', thicknessAssignment=FROM_SECTION)
# endregion

##########* ASSEMBLY ###########
# region
mdb.models['Thermal'].rootAssembly.DatumCsysByDefault(CARTESIAN)
mdb.models['Thermal'].rootAssembly.Instance(dependent=ON, name='Deposition-1',
                                            part=mdb.models['Thermal'].parts['Deposition'])
mdb.models['Thermal'].rootAssembly.Instance(dependent=ON, name='Substrate-1',
                                            part=mdb.models['Thermal'].parts['Substrate'])
mdb.models['Thermal'].rootAssembly.translate(instanceList=('Substrate-1', ),
                                             vector=(0.0, 0.0, -0.2/1000))
mdb.models['Thermal'].rootAssembly.InstanceFromBooleanMerge(domain=GEOMETRY,
                                                            instances=(mdb.models['Thermal'].rootAssembly.instances['Deposition-1'],
                                                                       mdb.models['Thermal'].rootAssembly.instances['Substrate-1']), name='Part',
                                                            originalInstances=SUPPRESS)
# endregion

##########* STEP ###########
# region

mdb.models['Thermal'].HeatTransferStep(name='Step_thermal', previous='Initial',
                                       timePeriod=Time, maxNumInc=int(Time/0.0002)+10, timeIncrementationMethod=FIXED,
                                       initialInc=0.0002)
# OUTPUT
mdb.models['Thermal'].fieldOutputRequests['F-Output-1'].setValues(
    variables=('TEMP',))
# mdb.models['Thermal'].HistoryOutputRequest(
#     createStepName='Step_thermal', name='H-Output-1', variables=('HTL',))
# endregion

##########* INTERACTION ###########
# ! 手动建立 a.Surface(side1Faces=side1Faces1, name='Surf-all')

# region
a = mdb.models['Thermal'].rootAssembly
s1 = a.instances['Part-1'].faces
side1Faces1 = s1.findAt(((0.001867, 0.000733, 0.0), ), ((0.0, 0.000733,
                                                         -0.000333), ), ((0.000733, 0.0022, -0.000333), ), ((0.0022, 0.001467,
                                                                                                             -0.000333), ), ((0.001467, 0.0, -0.000333), ), ((0.001467, 0.000733,
                                                                                                                                                              -0.001), ), ((0.0005, 0.0009, 6.7e-05), ), ((0.0009, 0.0017, 6.7e-05), ), (
    (0.0017, 0.0013, 6.7e-05), ), ((0.0013, 0.0005, 6.7e-05), ), ((0.0009,
                                                                   0.0009, 0.0001), ))
a.Surface(side1Faces=side1Faces1, name='Surf-all')
region = a.surfaces['Surf-all']
mdb.models['Thermal'].FilmCondition(name='convenction',
                                    createStepName='Step_thermal', surface=region, definition=EMBEDDED_COEFF,
                                    filmCoeff=22.5, filmCoeffAmplitude='', sinkTemperature=473.0,
                                    sinkAmplitude='', sinkDistributionType=UNIFORM, sinkFieldName='')
mdb.models['Thermal'].RadiationToAmbient(name='radiation',
                                         createStepName='Step_thermal', surface=region, radiationType=AMBIENT,
                                         distributionType=UNIFORM, field='', emissivity=0.7,
                                         ambientTemperature=473.0, ambientTemperatureAmp='')
mdb.models['Thermal'].setValues(absoluteZero=0)
mdb.models['Thermal'].setValues(stefanBoltzmann=5.67e-08)
# endregion

##########* LOAD ###########

# region
a = mdb.models['Thermal'].rootAssembly
region = a.instances['Part-1'].sets['Set_deposition']
mdb.models['Thermal'].Temperature(name='IniTemp',
                                  createStepName='Initial', region=region, distributionType=UNIFORM,
                                  crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(473.0, ))
# endregion

##########* MESH ###########
##########! 手动重新选择 ###########
# region
p = mdb.models['Thermal'].parts['Part']
c = p.cells
pickedCells = c.findAt(((0.004, 0.001, -0.0002), ))
v, e, d = p.vertices, p.edges, p.datums
p.PartitionCellByPlaneThreePoints(point1=v.findAt(coordinates=(0.0, 0.005, 
    0.0)), point2=v.findAt(coordinates=(0.005, 0.005, 0.0)), point3=v.findAt(
    coordinates=(0.0, 0.0, 0.0)), cells=pickedCells)
p = mdb.models['Thermal'].parts['Part']
e = p.edges
pickedEdges = e.findAt(((0.0, 0.00125, 0.0001), ), ((0.0, 0.005, 2.5e-05), ), (
    (0.0, 0.00125, 0.0), ), ((0.0, 0.0, 2.5e-05), ), ((0.00125, 0.005, 0.0001), 
    ), ((0.005, 0.005, 2.5e-05), ), ((0.00125, 0.005, 0.0), ), ((0.005, 
    0.00375, 0.0001), ), ((0.005, 0.0, 2.5e-05), ), ((0.005, 0.00375, 0.0), ), 
    ((0.00375, 0.0, 0.0001), ), ((0.00375, 0.0, 0.0), ))
p.seedEdgeBySize(edges=pickedEdges, size=1e-04, deviationFactor=0.1, 
    constraint=FINER)
p = mdb.models['Thermal'].parts['Part']
p.seedPart(size=0.00045, deviationFactor=0.1, minSizeFactor=0.1)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0192466, 
    farPlane=0.0317291, width=0.00859161, height=0.00401144, 
    viewOffsetX=0.00122781, viewOffsetY=0.000418462)
p = mdb.models['Thermal'].parts['Part']
p.generateMesh()

mdb.models['Thermal'].parts['Part'].setElementType(elemTypes=(ElemType(
    elemCode=DC3D8, elemLibrary=STANDARD), ElemType(elemCode=DC3D6,
                                                    elemLibrary=STANDARD), ElemType(elemCode=DC3D4, elemLibrary=STANDARD)),
    regions=(mdb.models['Thermal'].parts['Part'].cells.findAt(((0.0005, 0.0009, 4e-05),
                                                               ), ((0.002267, 0.000733, -0.0002), ), ), ))
mdb.models['Thermal'].rootAssembly.Set(cells=mdb.models['Thermal'].rootAssembly.instances['Part-1'].cells.findAt(
    ((0.0005, 0.0009, 4e-05), )), name='Set-1')
mdb.models['Thermal'].rootAssembly.Set(cells=mdb.models['Thermal'].rootAssembly.instances['Part-1'].cells.findAt(((
    0.0005, 0.0009, 4e-05), )), name='Set-1')
mdb.models['Thermal'].rootAssembly.Set(elements=mdb.models['Thermal'].rootAssembly.instances['Part-1'].elements[3200:6400],
                                       name='Set-2')
# endregion


##########* STRUCTURAL #########
# region
mdb.Model(name='Structural', objectToCopy=mdb.models['Thermal'])
del mdb.models['Structural'].steps['Step_thermal']
mdb.models['Structural'].StaticStep(initialInc=0.0002, maxNumInc=int(Time/0.0002)+10, name='Step_structural', nlgeom=ON, noStop=OFF, previous='Initial',
                                    timeIncrementationMethod=FIXED, timePeriod=Time)

mdb.models['Structural'].fieldOutputRequests['F-Output-1'].setValues(
    variables=('PE', 'EE'))

mdb.models['Structural'].EncastreBC(createStepName='Initial', localCsys=None,
                                    name='BC-1', region=Region(
                                        faces=mdb.models['Structural'].rootAssembly.instances['Part-1'].faces.findAt(
                                            ((0.001333, -0.000333, -0.0002), ), )))
del mdb.models['Structural'].predefinedFields['IniTemp']
mdb.models['Structural'].Temperature(absoluteExteriorTolerance=0.0,
                                     beginIncrement=1, beginStep=1, createStepName='Step_structural',
                                     distributionType=FROM_FILE, endIncrement=5000, endStep=1,
                                     exteriorTolerance=0.05, fileName=fileOdb_TEMP,
                                     interpolate=OFF, name='Predefined Field-1')
# endregion

##########! 手动重新选择 ###########
mdb.models['Structural'].parts['Part'].setElementType(elemTypes=(ElemType(
    elemCode=C3D8R, elemLibrary=STANDARD, secondOrderAccuracy=OFF,
    kinematicSplit=AVERAGE_STRAIN, hourglassControl=DEFAULT,
    distortionControl=DEFAULT), ElemType(elemCode=C3D6, elemLibrary=STANDARD),
    ElemType(elemCode=C3D4, elemLibrary=STANDARD)), regions=(
    mdb.models['Structural'].parts['Part'].cells.findAt(((0.0005, 0.0009, 4e-05),
                                                         ), ((0.002267, 0.000733, -0.0002), ), ), ))


#########* AM MODELER #########
# region
sys.path.insert(8, r'c:/SIMULIA/CAE/plugins/2021/AMModeler')
amModule.createAMModel(amModelName='AM-Model-1', modelName1='Thermal',
                       stepName1='Step_thermal', analysisType1=HEAT_TRANSFER, isSequential=ON,
                       modelName2='Structural', stepName2='Step_structural',
                       analysisType2=STATIC_GENERAL, processType=AMPROC_ABAQUS_BUILTIN)
mdb.customData.am.amModels['AM-Model-1'].addEventSeries(
    eventSeriesName='PowerPath', eventSeriesTypeName='"ABQ_AM.PowerMagnitude"',
    timeSpan='TOTAL TIME', fileName=filePP,
    isFile=ON)
mdb.customData.am.amModels['AM-Model-1'].addEventSeries(
    eventSeriesName='RollerPath',
    eventSeriesTypeName='"ABQ_AM.MaterialDeposition"', timeSpan='TOTAL TIME',
    fileName=fileRP, isFile=ON)
mdb.customData.am.amModels['AM-Model-1'].addTableCollection(
    tableCollectionName='ABQ_AM_Table Collection-1')
mdb.customData.am.amModels['AM-Model-1'].dataSetup.tableCollections['ABQ_AM_Table Collection-1'].PropertyTable(
    name='_propertyTable_"ABQ_AM.AbsorptionCoeff"_', propertyTableType='"ABQ_AM.AbsorptionCoeff"',
    propertyTableData=((0.35,),), numDependencies=0, temperatureDependency=OFF)
mdb.customData.am.amModels['AM-Model-1'].dataSetup.tableCollections['ABQ_AM_Table Collection-1'].PropertyTable(
    name='_propertyTable_"ABQ_AM.EnclosureAmbientTemp"_', propertyTableType='"ABQ_AM.EnclosureAmbientTemp"',
    propertyTableData=((473,),), numDependencies=0, temperatureDependency=OFF)
mdb.customData.am.amModels['AM-Model-1'].dataSetup.tableCollections['ABQ_AM_Table Collection-1'].ParameterTable(
    name='_parameterTable_"ABQ_AM.MovingHeatSource"_', parameterTabletype='"ABQ_AM.MovingHeatSource"',
    parameterData=(('PowerPath', 'Goldak'),))
mdb.customData.am.amModels['AM-Model-1'].dataSetup.tableCollections['ABQ_AM_Table Collection-1'].ParameterTable(
    name='_parameterTable_"ABQ_AM.MovingHeatSource.Goldak"_', parameterTabletype='"ABQ_AM.MovingHeatSource.Goldak"',
    parameterData=(('4', '4', '2', 0.0001, 0.0001, 0.0001, 0.0001, 0.66667, 1.33333, 1),))
mdb.customData.am.amModels['AM-Model-1'].dataSetup.tableCollections['ABQ_AM_Table Collection-1'].ParameterTable(
    name='_parameterTable_"ABQ_AM.MaterialDeposition"_', parameterTabletype='"ABQ_AM.MaterialDeposition"',
    parameterData=(('RollerPath', 'Roller'),))
mdb.models['Thermal'].rootAssembly.regenerate()
mdb.models['Thermal'].rootAssembly.Set(
    elements=mdb.models['Thermal'].rootAssembly.instances['Part-1'].elements, name='_AM-Model-1__AllBuildParts__')
mdb.models['Structural'].rootAssembly.regenerate()
mdb.models['Structural'].rootAssembly.Set(
    elements=mdb.models['Structural'].rootAssembly.instances['Part-1'].elements, name='_AM-Model-1__AllBuildParts__')
mdb.customData.am.amModels['AM-Model-1'].assignAMPart(amPartsData=(('Part-1',
                                                                    'Build Part'), ('', ''), ('', ''), ('', ''), ('', '')))
mdb.customData.am.amModels['AM-Model-1'].addMaterialArrival(
    materialArrivalName='Material Source -1',
    tableCollection='ABQ_AM_Table Collection-1', followDeformation=OFF,
    useElementSet=OFF, elementSetRegion=())
mdb.customData.am.amModels['AM-Model-1'].addHeatSourceDefinition(
    heatSourceName='Heat Source -1', dfluxDistribution='Moving-UserDefined',
    dfluxMagnitude=1, tableCollection='ABQ_AM_Table Collection-1',
    useElementSet=OFF, elementSetRegion=())
mdb.customData.am.amModels['AM-Model-1'].addCoolingInteractions(
    coolingInteractionName='Cooling Interaction -1', useElementSet=OFF,
    elementSetRegion=(), isConvectionActive=ON, isRadiationActive=OFF,
    filmDefinition='Embedded Coefficient', filmCoefficient=22.5,
    filmcoefficeintamplitude='Instantaneous', sinkDefinition='Uniform',
    sinkTemperature=473, sinkAmplitude='Instantaneous',
    radiationType='toAmbient', emissivityDistribution='Uniform', emissivity=0.7,
    ambientTemperature=473, ambientTemperatureAmplitude='Instanteneous')
mdb.saveAs(pathName=fileCae)
# endregion

##########* Cooling Process #########
# region
##########* THERMAL2 & STRUCTURAL2#########
mdb.Model(name='Thermal_cooling', objectToCopy=mdb.models['Thermal'])
mdb.models['Thermal_cooling'].steps['Step_thermal'].setValues(initialInc=1,
                                                              timePeriod=100)
mdb.models['Thermal_cooling'].predefinedFields['IniTemp'].setValues(beginIncrement=int(
    Time/0.0002), beginStep=1, distributionType=FROM_FILE, fileName=fileOdb_TEMP)
mdb.Model(name='Structural_cooling', objectToCopy=mdb.models['Structural'])
mdb.models['Structural_cooling'].steps['Step_structural'].setValues(initialInc=1,
                                                                    timePeriod=100)
mdb.models['Structural_cooling'].predefinedFields['Predefined Field-1'].setValues(
    endIncrement=100, fileName=fileOdb_TEMP_cooling)
# endregion


# endregion

##########* JOB ###########
# region
mdb.saveAs(pathName=fileCae)
mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
        explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
        memory=90, memoryUnits=PERCENTAGE, model='Thermal', modelPrint=OFF,
        multiprocessingMode=DEFAULT, name=fileInp_t, nodalOutputPrecision=SINGLE, numCpus=12, numDomains=12, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
        explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
        memory=90, memoryUnits=PERCENTAGE, model='Structural', modelPrint=OFF,
        multiprocessingMode=DEFAULT, name=fileInp_s, nodalOutputPrecision=SINGLE, numCpus=12, numDomains=12, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
        explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
        memory=90, memoryUnits=PERCENTAGE, model='Thermal_cooling', modelPrint=OFF,
        multiprocessingMode=DEFAULT, name=fileInp_tc, nodalOutputPrecision=SINGLE, numCpus=12, numDomains=12, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
        explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
        memory=90, memoryUnits=PERCENTAGE, model='Structural_cooling', modelPrint=OFF,
        multiprocessingMode=DEFAULT, name=fileInp_sc, nodalOutputPrecision=SINGLE, numCpus=12, numDomains=12, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
mdb.jobs[fileInp_t].writeInput(consistencyChecking=OFF)
mdb.jobs[fileInp_s].writeInput(consistencyChecking=OFF)
mdb.jobs[fileInp_tc].writeInput(consistencyChecking=OFF)
mdb.jobs[fileInp_sc].writeInput(consistencyChecking=OFF)
mdb.saveAs(pathName=fileCae)

# mdb.jobs[fileInp_t].submit(consistencyChecking=OFF)
# mdb.jobs[fileInp_s].submit(consistencyChecking=OFF)
# mdb.jobs[fileInp_tc].submit(consistencyChecking=OFF)
# mdb.jobs[fileInp_sc].submit(consistencyChecking=OFF)
# endregion

# endregion
