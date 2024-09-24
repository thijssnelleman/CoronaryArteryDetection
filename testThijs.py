import utilsGraphThijs as u
import json
import os
import sys

pathLoadGraph = r'H:\Data\GraphFiles\graphs\rough\high-res'
#os.chdir("H:\Data\graphprepFineGT\graphprepFineGTAMC")
os.chdir(pathLoadGraph)
dirList = [i for i in os.listdir() if i.endswith('.json')]

sys.setrecursionlimit(3000)


#pathSaveGraph = "H:\\Data\\GraphFiles\\graphs\\rough\\high-res\\vis"
pathSaveMevis = "H:\\Data\\GraphFiles\\graphs\\rough\\high-res\\vis\\"

# load graph file into dictionary
for file in dirList:
    print("Doing " + file + "... ")
    with open(file) as f:
        graphprep = json.load(f)

    # create graph class from dictionary
    try:
        graphFine = u.GraphContFC(**graphprep, keysToIntBool=True)
        graphFine.appendOrder(correctOrderBool=False)
        #graphFine.output(pathSaveGraph+file, verbose=True)
        # create graph with segments as nodes (graphCourse) from graph with centerline points as nodes (graphFine)
        #graphCourse = graphFine.createCourseGraph()
        # save graphCourse to json
        #graphCourse.output(pathSaveGraph, verbose=True)

        # save graphCourse into Mevislab readable format
    
    except:
        print("Failed.")
    fname = file[:-5] + ".xml"
    graphFine.saveFullTree(pathOut=pathSaveMevis+fname, modeAppendTreeID='segLabGT', modeAppendTreeName='keys', verbose=True)




