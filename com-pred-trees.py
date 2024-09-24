import os
import numpy as np
import json
import pickle
from sklearn.metrics import precision_recall_curve

test_set = [0, 3, 4, 13, 20, 24, 25, 27, 33, 38, 40, 44, 52, 55, 57, 69, 71, 72, 73, 77, 80, 85, 88, 101, 108, 113, 114, 116, 119, 120, 130, 131, 135, 143, 150, 151, 153, 154, 161, 163, 173, 174, 177, 178, 179, 184, 188, 191, 193, 199]

os.chdir(r"H:\Data\GraphFiles\results")
save_name = "Graph UNET2- ClusterPool-18-06-2022 17-27-02.pkl"
labels = []

bst = 0.0
threshold = 0
mxf1 = 0
with open(save_name, 'rb') as f:
    loaded_dict = pickle.load(f)
    loaded_dict = loaded_dict["Graph UNET2- ClusterPool"]
    test_vals = loaded_dict["test_data"]
    #labels = test_vals[0][0]
    for folds in test_vals:
        prec, rec, t =  precision_recall_curve(test_vals[0][0], folds[1])
        f1s = (2*(prec * rec)) / (prec + rec)
        threshold += t[np.argmax(f1s)]
        print(np.max(f1s))
        mxf1 += np.max(f1s)
        #if np.max(f1s) > bst:
        #    labels = np.array(folds[1])
        if len(labels) == 0:
            labels = np.array((folds[1] > t[np.argmax(f1s)]).astype(int))
        else:
            labels += np.array((folds[1] > t[np.argmax(f1s)]).astype(int))

print("Avg f1:", mxf1 / 10)

ground_truth = test_vals[0][0]
print(labels)
labels = labels / 10
threshold = threshold / 10
#print("thresh og:",threshold)

print(labels == 0.5)
tie = (labels == 0.5).nonzero()
print(tie)
print(len(tie[0]), len(labels))
labels = (labels >= 0.5).astype(int)
#input()

prec, rec, t =  precision_recall_curve(test_vals[0][0], labels)
f1s = (2*(prec * rec)) / (prec + rec)
print(np.max(f1s), "\n\n")

#threshold = t[np.argmax(f1s)]
#vl = 0.84
#dist = 100000
#sel = -1
#for i,e in enumerate(t):
#    if np.abs(vl -e) < dist:
#        sel = i
#        dist = np.abs(vl -e)

"""for i,e in enumerate(t):
    if e >= threshold:
        sel = i
        break

#sel = np.argmax(f1s)

print(f1s[sel])
print(prec[sel], rec[sel])
threshold = t[sel]
print("thresh:",threshold)

labels = (labels > threshold).astype(int)"""



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