roots = 0
nodes = 0

for p in nodeDict.keys():
    patient = nodeDict[p]
    nodeIDs = set(patient.keys())
    for e in edgeDict[p].keys():
        if str(edgeDict[p][e]["incNodeKey"]) in nodeIDs:
            nodeIDs.remove(str(edgeDict[p][e]["incNodeKey"]))
    roots += len(nodeIDs)
    nodes += len(patient.keys())

print("Average number of roots per patient: " + str(roots / len(nodeDict.keys())))
print("Root ratio over all patients: " + str(roots/nodes))