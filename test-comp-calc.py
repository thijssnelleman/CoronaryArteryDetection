import json
import pickle
import os
from re import A
from tabnanny import check

import numpy as np

test_set = [0, 3, 4, 13, 20, 24, 25, 27, 33, 38, 40, 44, 52, 55, 57, 69, 71, 72, 73, 77, 80, 85, 88, 101, 108, 113, 114, 116, 119, 120, 130, 131, 135, 143, 150, 151, 153, 154, 161, 163, 173, 174, 177, 178, 179, 184, 188, 191, 193, 199]

def distance(A, B):
    x = (A[0] - B[0])**2
    y = (A[1] - B[1])**2
    z = (A[2] - B[2])**2
    d = x + y + z
    return np.sqrt(d)

def check_sphere_intersection(sphereA, sphereB):
    dist = distance(sphereA, sphereB)
    return dist < sphereB[3]

#counts = []

fnr = []
tpr = []
def compareTrees(referenceTree, outputTree):
    TPM, TPR, FN, FP = 0,0,0,0

    for outnode in outputTree:
        match = False
        for refnode in referenceTree:
            if check_sphere_intersection(outnode, refnode):
                match = True
                break
        if match:
            TPM += 1
            
        else:
            FP += 1
    
    
    for refnode in referenceTree:
        match = False
        c = 0
        for outnode in outputTree:
            if check_sphere_intersection(outnode, refnode):
                match = True
                #break
                c += 1

        if match:
            TPR += 1
            tpr.append(refnode[3])
        else:
            fnr.append(refnode[3])
            FN += 1
    
    
    return TPM, TPR, FN, FP

"""def compareTrees(referenceTree, outputTree):
    TPM, TPR, FN, FP = 0,0,0,0
    #a = np.array(referenceTree)
    #print(np.max(a[:,3]))
    to_find = set(x for x in range(len(referenceTree)))
    for outnode in outputTree:
        match = False
        for idx, refnode in enumerate(referenceTree):
            if check_sphere_intersection(outnode, refnode):
                if idx in to_find:
                    to_find.remove(idx)
                    TPR += 1
                
                match = True
        
        if match:
            TPM += 1
        else:
            FP += 1
    #TPR = len(referenceTree) - len(to_find)
    FN = len(to_find)
    #print(FP)

    return TPM, TPR, FN, FP"""

#ofname = "good-outputTrees.npy"
ofname = "outputTrees.npy"

outputTrees = np.load(file=ofname, allow_pickle=True)
referenceTrees = np.load(file="referenceTrees.npy", allow_pickle=True)

TPM, TPR, FN, FP = 0,0,0,0

for i in range(len(outputTrees)):
    print("Doing Tree ", i)

    #print(np.max(outputTrees[i]))

    #    print(np.max(referenceTrees[i]))

    a,b,c, d = compareTrees(referenceTrees[i], outputTrees[i])
    TPM += a
    TPR += b
    FN += c
    FP += d


p = TPM / (TPM + FP) #Precision
r = TPR / (TPR + FN) #Sensitiviy

#OT = (TPM + TPR) / (TPM + TPR + FN + FP) #Doesnt work

f1 = 2*p*r / (p+r)

fnov = (TPR + TPM) / (TPM + TPR + FN +FP)

print("\n\n")
#print("OT: ", OT)
print("Sensitivity / Recall: ", r)
print("Precision: ", p)
#print("Precision: ", p, "Recall: ",  r)
print("F1 / Overlap: ", f1)
print("fnov: ", fnov)

import matplotlib.pyplot as plt

plt.boxplot([tpr, fnr])
plt.xlabel("True positives radius versus false negative radius")
plt.ylabel("Radius size")
plt.show()

"""if False:
    os.chdir(r"H:\Data\GraphFiles\graphs\referenceTrees")

    files = [x for x in os.listdir() if x.endswith(".json")]
    referenceTrees = []

    for e in test_set:
        fname = files[e]
        nodesList = []
        with open(fname) as f:
            dict = json.load(f)
            nodes = dict["nodeDictDict"]

            for n in nodes:
                coords, radius = nodes[n]["coords"], nodes[n]["radius"]
                coords.append(radius)
                #if coords is None or radius is None:
                #    print("errrroooor")
                nodesList.append(coords)

        print(fname, len(nodesList))
        referenceTrees.append(nodesList)

    #print(referenceTrees)
    referenceTrees = np.array(referenceTrees)

    os.chdir(r"H:\Data\GraphFiles")
    
    input()
    np.save(arr=referenceTrees, file="referenceTrees.npy")

#outputTrees = np.load(file="outputTrees.npy", allow_pickle=True)
#referenceTrees = np.load(file="referenceTrees.npy", allow_pickle=True)




os.chdir(r"H:\Data\GraphFiles\results")
save_name = "Graph UNET2- ClusterPool-18-06-2022 17-27-02.pkl"
labels = []
with open(save_name, 'rb') as f:
    loaded_dict = pickle.load(f)
    loaded_dict = loaded_dict["Graph UNET2- ClusterPool"]
    test_vals = loaded_dict["test_data"]
    for folds in test_vals:
        if len(labels) == 0:
            labels = np.array(folds[0])
        else:
            labels += np.array(folds[0])

labels = np.round(labels / 10)


os.chdir(r"H:\Data\GraphFiles\graphs\rough\high-res")
files = [x for x in os.listdir() if x.endswith(".json")]
outputTrees = []

node_index = 0
for e in test_set:
    fname = files[e]
    nodesList = []

    #print(fname)
    with open(fname) as f:
        dict = json.load(f)
        nodeDict = dict["nodeDictDict"]
        #Remove the undirected nodes
        del_nodes = []
        for node in nodeDict.keys():                
            if nodeDict[node]["incEdgeKey"] is None and len(nodeDict[node]["outEdgeKeyList"]) == 0: #No unconnected nodes
                del_nodes.append(node)

        for dn in del_nodes:
            del nodeDict[dn]

        for index, nodeID in enumerate(nodeDict.keys()):
            if labels[node_index] == 1:
                vals = nodeDict[nodeID]['coords']
                vals.append(nodeDict[nodeID]['radius'])
                nodesList.append(vals)
            node_index += 1

    print(fname, len(nodesList))
    outputTrees.append(nodesList)


outputTrees = np.array(outputTrees)
os.chdir(r"H:\Data\GraphFiles")
np.save(arr=outputTrees, file="outputTrees.npy")
#print(node_index, len(labels))

exit()
print("-------------------------------")


os.chdir(r"H:\Data\GraphFiles\graphs\referenceTrees")

files = [x for x in os.listdir() if x.endswith(".json")]
referenceTrees = []

for e in test_set:
    fname = files[e]
    nodesList = []
    with open(fname) as f:
        dict = json.load(f)
        nodes = dict["nodeDictDict"]

        for n in nodes:
            coords, radius = nodes[n]["coords"], nodes[n]["radius"]
            nodesList.append(coords.append(radius))

    print(fname, len(nodesList))
    referenceTrees.append(nodesList)

referenceTrees = np.array(referenceTrees)

os.chdir(r"H:\Data\GraphFiles")

np.save(arr=referenceTrees, file="referenceTrees.npy")"""
