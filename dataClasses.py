import os
import time
import json

import numpy as np
import math

import torch

import utilsGraphThijs as u

class dataLoader:
    def __init__(self, loc="H:\Data\GraphFiles\graphs", processed=False, asBinary=True):
        start_time = time.time()
        self.data_loc = loc
        self.nodeDict = {}
        self.edgeDict = {}

        self.node_tensors, self.edge_tensors, self.label_tensors, self.pIDs = None, None, None, None
        self.patient_subset = ["F" + f"{x:03}" + ".json" for x in range(1,51)]

        self.__processed = processed
        self.__binaryCLF = asBinary
        self.__binary_labels = [10, 11 ,12]

        self.test_set_idx = [0, 3, 4, 13, 20, 24, 25, 27, 33, 38, 40, 44, 52, 55, 57, 69, 71, 72, 73, 77, 80, 85, 88, 101, 108, 113, 114, 116, 119, 120, 130, 131, 135, 143, 150, 151, 153, 154, 161, 163, 173, 174, 177, 178, 179, 184, 188, 191, 193, 199] #Randomly selected patients as static test set

        #Actual possible labels. Labels that are NOT present in the data set are removed as class during processing.
        self.__real_labels = ["Right Coronary Artery (RCA)", #(0)
                       "Acute Marginal (AM)", #(1)
                       "Right Posterior Descending Artery (rPDA)", #(2)
                       "Left Main (LM)", #(3)
                       "Left Anterior Descending (LAD)", #(4)
                       "Septal (S)", #(5)
                       "Diagonal (D)", #(6)
                       "Left Circumflex (LCX)", #(7)
                       "Obtuse Marginal (OM)", #(8)
                       "Right Posterior Lateral Branch (rPLB)", #(9)
                       "Func 1", #functional label (10)
                       "Func 2", #functional label (11)
                       "Func 3", #functional label (12)
                       "Left Posterior Lateral Branch (lPLB)", #(13)
                       "Left Posterior Descending Artery (lPDA)"] #(14)

        if self.__processed:
            #Processed Node features
            self.features = ["X-Coordinate",
                             "Y-Coordinate",
                             "Z-Coordinate",
                             "Radius",
                             "Order",
                             "Ground-Truth Label"] #Not really a feature but eh
        else:
            #Unprocessed Node Features
            self.features = ["X-Coordinate",
                             "Y-Coordinate",
                             "Z-Coordinate",
                             "Radius",
                             "Order",
                             "Seed Confidence",
                             "Direction Entropy",
                             "Directional Vector X",
                             "Directional Vector Y",
                             "Directional Vector Z",
                             "Ground-Truth Label"] #Not really a feature but eh

        """"
        Unprocessed graph features:
        "coords", #Three floats
        "Radius", #Float
        "Order", #Int (Might not make sense here due to gaps)
        "connOstBool", #Always false! (Boolean)
        "Directions", #Direction of tracker search: Likely hood of each possible new point still being an artery (List of 500 floats)
        "seedNetVal", #Seed Network Value (Prediction of how likely this point is an artery), comes from tracking procedure (Single number)
        "segLab", #Segmentation label: Has values {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
                Label proportions (143.576 nodes over 34 patients):
                0.014730874240820193
                0.04537666462361398
                0.01325430434055831
                0.00129548113890901
                0.024217139354766812
                0.011728979773778347
                0.027650860868111662
                0.014438346241711707
                0.029364239148604223
                0.01769794394606341
                0.016541761854348917
                0.00036217752270574467
                0.7833412269460077
        "actions", #dictionary inside a dict with keys: ['nodeKeyGT', 'dirUntrackedVec', 'dirTrackedVec', 'farOutKeyGTVec', 'entropy', 'entropyStart', 'dirPred', 'numTrackedAtDirPred']
        "coordsFeat", #This seems to contain the re-centralized coordinates
        "orderFeat", # Single float, all equal to -0.001 in every node in every patient
        "radiusFeat", # Single float -> Normalized variant of Radius on entire set of patients
        "segLabPred"] #Segmentation label prediction? Varies about 21% of the nodes from segLab and has values: {3, 4, 7, 8, 12}
        """
        #
        """self.rough_features = ["X-Coordinate", #Float
                               "Y-Coordinate", #Float
                               "Z-Coordinate", #Float
                               "Radius", #Float
                               "Order", #Int
                               "segLab"], #Segmentation label: Has values {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}"""

        self.__present_labels = set() #Labels that were present before fixing
        self.__load()
        self.labels = [self.__real_labels[i] for i in list(self.__present_labels)]
        if self.__binaryCLF:
            self.labels = ["Negative", "Positive"]
        
        elapsed_time = time.time() - start_time
        elapsed_time = elapsed_time / 60.0
        print(f"Loaded data in: {elapsed_time:.3f} m")

    #Loads the json files
    def __load(self):
        ext = ""
        ext += r"\processed" if self.__processed else r"\rough"
        ext += r"\high-res"
        try: #Windows
            os.chdir(self.data_loc + ext)
        except: #Linux
            os.chdir('/mnt/H/Data/GraphFiles/graphs/rough/high-res')

        if "patientTensors.pt" in os.listdir(): #We have already created a tensor representation of the data
            self.__load_tensors()
            return

        self.pIDs = [i for i in os.listdir() if i.endswith('.json')]
        
        for index, file in enumerate(self.pIDs):
            with open(file) as f:
                dict = json.load(f)
                for node in dict["nodeDictDict"].keys():
                    del dict["nodeDictDict"][node]["directions"] #Reduce the memory load
                self.nodeDict[file] = dict["nodeDictDict"]
                self.edgeDict[file] = dict["edgeDictDict"]    

        self.__process_nodes()
        self.__fix_labels()
        self.__create_torch_data()
        self.__save_tensors()
        del self.nodeDict
        del self.edgeDict

    #If we have already saved a dictionary of tensors, we can load that instead
    def __load_tensors(self):
        dict = torch.load("patientTensors.pt")
        self.node_tensors, self.edge_tensors, self.label_tensors, self.pIDs = [], [], [], []
        for k in dict.keys(): #For each patient
            #if k not in self.patient_subset:
            #    continue
            tSet = dict[k]
            self.node_tensors.append(tSet[0])
            self.edge_tensors.append(tSet[1])
            self.label_tensors.append(tSet[2])
            self.pIDs.append(k)

    def __save_tensors(self):
        dict = {}
        for index, name in enumerate(self.nodeDict.keys()):
            dict[name] = [self.node_tensors[index], self.edge_tensors[index], self.label_tensors[index]]
        torch.save(dict, "patientTensors.pt")

    #Creates a .xml file visualizing the labels
    def create_vis(self, labels, fname: str, mname: str, thresh: float, as_probs=False):
        graph = None
        with open(fname) as f:
            graph = json.load(f)
        
        nodeDict = graph["nodeDictDict"]

        #Remove the undirected nodes
        del_nodes = []
        for node in nodeDict.keys():                
            if nodeDict[node]["incEdgeKey"] is None and len(nodeDict[node]["outEdgeKeyList"]) == 0: #No unconnected nodes
                del_nodes.append(node)

        for dn in del_nodes:
            del nodeDict[dn]

        
        for index, nodeID in enumerate(nodeDict.keys()):

            if as_probs: #Give node a label 1-10 to demonstrate its distribution
                probability = int(round(labels[index], 1) * 10) + 1
                nodeDict[nodeID]["segLabPred"] = probability
            else:
                ground_truth = int(int(nodeDict[nodeID]["segLab"]) not in self.__binary_labels)
                prediction = int(labels[index] > thresh)
                if ground_truth == 1 and prediction == 1:   #True Positive
                    nodeDict[nodeID]["segLabPred"] = 1
                elif ground_truth == 0 and prediction == 0: #True Negative
                    nodeDict[nodeID]["segLabPred"] = 2
                elif ground_truth == 0 and prediction == 1: #False Positive
                    nodeDict[nodeID]["segLabPred"] = 3
                elif ground_truth == 1 and prediction == 0: #False Negative
                    nodeDict[nodeID]["segLabPred"] = 4
   
        graphFine = u.GraphContFC(**graph, keysToIntBool=True)
        graphFine.appendOrder(correctOrderBool=False)

        if as_probs:
            mode = "Probabilities"
        else:
            mode = "Labels"
        fname = "exports/" + fname[:-5] + "-" + mname + "-" + mode + ".xml"
        graphFine.saveFullTree(pathOut=fname, modeAppendTreeID='segLabPred', modeAppendTreeName='segLabGT', verbose=True)

    #Visualizes which nodes participate in a cluster
    def create_cluster_vis(self, cluster_map: list, fname: str, mname: str):

        #labels = np.array(labels)
        graph = None
        with open(fname) as f:
            graph = json.load(f)
        
        nodeDict = graph["nodeDictDict"]
        edgeDict = graph["edgeDictDict"]

        #Remove the undirected nodes
        del_nodes = []
        for node in nodeDict.keys():                
            if nodeDict[node]["incEdgeKey"] is None and len(nodeDict[node]["outEdgeKeyList"]) == 0: #No unconnected nodes
                del_nodes.append(node)

        for dn in del_nodes:
            del nodeDict[dn]

        pairs = []
        patientNodes = list(nodeDict.keys())
        for edgeID in edgeDict.keys():
            edge = edgeDict[edgeID]

            nodeIDLeft = patientNodes.index(str(edge['outNodeKey'])) #Fix node ID to its index in the nodeList instead of its index label (Can have gaps etc)
            nodeIDRight = patientNodes.index(str(edge['incNodeKey'])) #Fix node ID to its index in the nodeList instead of its index label   
            if nodeIDLeft != nodeIDRight: #No self edges 
                pairs.append([nodeIDLeft, nodeIDRight])

        labels = []
        for cluster in cluster_map:
            for n in cluster:
                labels.append(-1)

        #labels = [-1 for _ in nodeDict.keys()]
        #print(i, len(labels))
        for index, cluster in enumerate(cluster_map):
            for node_id in cluster:
                labels[node_id] = index 

        labels = np.array(labels)

        tensors_id = self.pIDs.index(fname)
        nodes = self.node_tensors[tensors_id]
        edges = torch.unique(torch.tensor(pairs, dtype=torch.long), dim=0).T
        #, edges = self.node_tensors[tensors_id], self.edge_tensors[tensors_id]

        roots = (nodes[:,4] == 0).nonzero().flatten().tolist() #Get the roots based on the order feature

        def set_labs(segment, predecessor, labels, clusters, toggle=False):
            
            stop = 0
            for i,e in enumerate(segment):
                if type(e) == list:
                    break
                stop += 1
            
            if stop > 0 and predecessor is not None and labels[segment[0]] != -1: #If our predecessor is our "second visitor"
                if clusters[predecessor] != clusters[segment[0]] and labels[predecessor] == labels[segment[0]]: #We aren't in a cluster together, but our labels would match..
                    labels[predecessor] = 2 #Mark it specially so it stands out
                elif clusters[predecessor] == clusters[segment[0]] and labels[predecessor] != labels[segment[0]]: #We are in a cluster together, but our labels don't match..
                    labels[predecessor] = 3
            
            new_labels = []
            for node in segment[:stop]:
                if clusters[node] != clusters[predecessor]:
                    toggle = not toggle
                new_labels.append(toggle)
                predecessor = node
            
            for seg in segment[stop:]: #Now do the subtrees
                labels = set_labs(seg, predecessor, labels, toggle=toggle)

            #Update yourself as last! To avoid messing around with the labels
            labels[segment[:stop]] = np.array(new_labels)
            return labels

        #print(edges)
        totn = 0
        for root in roots:
            #print(root)
            comp = dataLoader.get_segment(root, None, edges)
            #print(comp)
            labels = set_labs(comp, comp[0], labels, labels)
            totn += len(dataLoader.flatten(comp))
        
        

        #print("------------------------------------------------------------------")
        for index, nodeID in enumerate(nodeDict.keys()):
            #print(labels[index])
            nodeDict[nodeID]["segLabPred"] = int(labels[index])

        #print(len(roots))
        #print(len(nodeDict.keys()), len(labels), totn)
        graphFine = u.GraphContFC(**graph, keysToIntBool=True)
        graphFine.appendOrder(correctOrderBool=False)

        fname = "exports/" + fname[:-5] + "-" + mname + "-" + "Clusterings" + ".xml"
        graphFine.saveFullTree(pathOut=fname, modeAppendTreeID='segLabPred', modeAppendTreeName='segLabGT', verbose=True)
        
        
    #processes the data in the node dict
    def __process_nodes(self):
        for id in self.nodeDict.keys():
            patient = self.nodeDict[id]
            del_nodes = []
            for node in patient.keys():                
                if patient[node]["incEdgeKey"] is None and len(patient[node]["outEdgeKeyList"]) == 0: #No unconnected nodes
                    del_nodes.append(node)
                    continue
                self.__present_labels.add(int(patient[node]['segLab']))

            for dn in del_nodes:
                del patient[dn]

    #Fixes labels to be a continuous range of integers
    def __fix_labels(self):
        mapping = {}
        last = 0
        for index, entry in enumerate(self.__present_labels):
            if index != entry:
                mapping[entry] = last
            last +=1

        for PID in self.nodeDict.keys():
            patient = self.nodeDict[PID]
            for node in patient.keys():
                if int(patient[node]['segLabPred']) in mapping: #The label needs to be fixed
                    patient[node]['segLabPred'] = mapping[int(patient[node]['segLabPred'])]

    #Returns a list of tensors, each corresponding to the edges of a patient
    def get_edge_tensors(self, undirected=True):
        edge_tensors = []

        for index, PID in enumerate(self.edgeDict.keys()):
            patient = self.edgeDict[PID]
            patientNodes = list(self.nodeDict[PID].keys())

            pairs = []
            for edgeID in patient.keys():
                edge = patient[edgeID]

                nodeIDLeft = patientNodes.index(str(edge['outNodeKey'])) #Fix node ID to its index in the nodeList instead of its index label (Can have gaps etc)
                nodeIDRight = patientNodes.index(str(edge['incNodeKey'])) #Fix node ID to its index in the nodeList instead of its index label   
                if nodeIDLeft != nodeIDRight: #No self edges 
                    pairs.append([nodeIDLeft, nodeIDRight])
                    if undirected: pairs.append([nodeIDRight, nodeIDLeft])

            #edg_t = torch.tensor(pairs, dtype=torch.long)
            #edg_t = torch.unique(edg_t, dim=0)
            edge_tensors.append(torch.tensor(pairs, dtype=torch.long))
        return edge_tensors

    #Returns a list of tensors, each corresponding to the nodes of a patient
    def get_node_tensors(self):
        node_tensors = []
        node_labels = []

        max_norm = 0.0
        max_rad = 0.0
        max_order = 0
        max_seed_net = 0

        #process each patient
        for index, PID in enumerate(self.nodeDict.keys()):
            patient = self.nodeDict[PID]
            nodes = [] #These need to be sorted in numerical order of their nodeID
            labels = [] #labels corresponding to each node

            for nodeID in patient.keys(): #Fix all the nodes
                node = patient[nodeID]               
                prev_edge = node["incEdgeKey"]
                
                def calc_dir_vec(prev, next): #Caclulate the direction of the bloodflow in the patch
                    x = prev[0] - next[0]
                    y = prev[1] - next[1]
                    z = prev[2] - next[2]
                    
                    return [x,y,z]
                
                if prev_edge is not None: #We have a predecessor
                    prev_edge = self.edgeDict[PID][str(prev_edge)]
                    prev_n = str(prev_edge["incNodeKey"]) #Extract the predecessor node from the Edge
                    prev_n = patient[prev_n]['coordsFeat']
                    dir_vec = calc_dir_vec(prev_n, node['coordsFeat'])
                elif len(node["outEdgeKeyList"]) == 1: #Exactly one successor
                    next_edge = self.edgeDict[PID][str(node["outEdgeKeyList"][0])]
                    next_n = str(next_edge["outNodeKey"])
                    next_n = patient[next_n]['coordsFeat']
                    dir_vec = calc_dir_vec(next_n, node['coordsFeat'])
                else:
                    dir_vec = [0.0, 0.0, 0.0]

                #Normalize the vector to 1
                if np.sum(dir_vec) != 0.0:
                    dir_vec = np.array(dir_vec)
                    dir_vec = dir_vec / np.linalg.norm(dir_vec)

                features = [node['coordsFeat'][0],              #x
                            node['coordsFeat'][1],              #y
                            node['coordsFeat'][2],              #z
                            node['radiusFeat'],                 #Node Radius
                            node['order'],                      #Number of predecessors
                            node['seedNetVal'][0],              #Trackers' confidence
                            dir_vec[0],
                            dir_vec[1],
                            dir_vec[2],
                            node['actions']['entropy'][0]]         #Something to do with direction

                norm = math.sqrt(features[0]**2 + features[1]**2 + features[2]**2) #||v|| = sqrt(a^2 + b^2 + c^2)
                if norm > max_norm: max_norm = norm
                if features[3] > max_rad: max_rad = features[3]
                if features[4] > max_order: max_order = features[4]
                if features[5] > max_seed_net: max_seed_net = features[5]

                nodes.append(features)

                if self.__processed:
                    labels.append(int(node['segLabPred']))
                else:
                    if self.__binaryCLF:
                        if int(node['segLab']) in self.__binary_labels:
                            labels.append(0) #Noise (Negative)
                        else:
                            labels.append(1) #Valid (Positive)
                    else:
                        labels.append(int(node['segLab']))

            #node_tensors.append(torch.nn.functional.normalize(torch.tensor(nodes, dtype=torch.float), p=1.0, dim=1))
            node_tensors.append(torch.tensor(nodes, dtype=torch.float))
            if self.__binaryCLF:
                node_labels.append(torch.tensor(labels, dtype=torch.float))
            else:
                node_labels.append(torch.tensor(labels, dtype=torch.int32))

        #Normalize xyz using the maximum vector length found and the other features just by max value
        #Usually normalzie using: Set mean to zero and std. dev. to 1
        #Maybe shift to model normalization instead (seperate layer)
        for patient in node_tensors:
            for nodes in patient:
                nodes[0] = nodes[0] / max_norm
                nodes[1] = nodes[1] / max_norm
                nodes[2] = nodes[2] / max_norm
                nodes[3] = nodes[3] / max_rad
                nodes[4] = nodes[4] / max_order
                nodes[5] = nodes[5] / max_seed_net

        return node_tensors, node_labels

    def fix_ordering(self):        
        def fix_succesion(segments, nodes, depth=0):
            stop = 0
            for item in segments:
                if type(item) == list: #new part
                    break
                stop += 1
            section = segments[:stop]

            for i, node_idx in enumerate(section):
                if nodes[node_idx][4] != -1 and nodes[node_idx][4] >= depth: #early stopping
                    return depth-1, section[:i]
                nodes[node_idx][4] = depth
                depth += 1

            #del segments[:stop]
            maxd = depth-1
            new_seg = segments[:stop]
            for lists in segments[stop:]:
                rd, s = fix_succesion(lists, nodes, depth=depth+1)
                new_seg.append(s)
                if rd > maxd: maxd = rd
            return maxd, new_seg

        def fix_noise(segments, labels, seen_green=False):
            stop = 0
            for item in segments:
                if type(item) == list: #new part
                    break
                stop += 1
            section = segments[:stop]            
            length = 0
            
            cur_nodes = []
            for node in section:        
                if labels[node] == 0:
                    length += 1
                    cur_nodes.append(node)
                else:
                    if length == 0: #Skip to the first 0 label
                        continue
                    if seen_green:
                        if len(cur_nodes) < 5:
                            labels[cur_nodes] = 1 #Fix the labels
                        cur_nodes = []
                    else:
                        seen_green = True
                    length = 0
            
            for lists in segments[stop:]:
                fix_noise(lists, seen_green=seen_green)
            return


        for patients in zip(self.node_tensors, self.edge_tensors, self.label_tensors):
            nodes = patients[0]
            nodes[:,4] = -1
            edges = patients[1].T
            labels = patients[2]
            prev_pos = (labels == 1).nonzero().size(0)
            roots = dataLoader.get_roots(nodes, edges)
            cn = 0
            segs = []
            for root_idx in roots:
                segments = dataLoader.get_segment(root_idx, None, edges)
                #fix_noise(segments, labels)
                dpth, segments = fix_succesion(segments, nodes)
                segs.append((dpth, segments))
            
            for depth, segments in segs:
                flat_list = dataLoader.flatten(segments)
                cn += len(flat_list)
                if depth != 0:
                    nodes[flat_list,4] = nodes[flat_list,4] / depth #Normalize
            
            #print("# of changed labels: ", (labels == 1).nonzero().size(0) - prev_pos)

    def __create_torch_data(self):
        self.node_tensors, self.label_tensors = self.get_node_tensors()
        self.edge_tensors = self.get_edge_tensors(undirected=False) #So that we can detect roots
        self.fix_ordering()
        self.edge_tensors = self.get_edge_tensors(undirected=True)
        
        
    """def drop_components_test(self):
        for PID in range(len(self.node_tensors)):
            nodes = self.node_tensors[PID]
            edges = self.edge_tensors[PID]
            labels = self.label_tensors[PID]

            from cluster_pool import calculate_components
            comps = calculate_components(nodes.size(0), edges.T)
            drop_nodes = []
            for c in comps:
                if not torch.any(labels[c] == 1): #This components has no positive labels
                    drop_nodes.extend(c)
            
            keep_nodes = np.sort(list(set([x for x in range(nodes.size(0))]) - set(drop_nodes)))
            drop_nodes = np.sort(drop_nodes)
            node_maps = [-1 for _ in range(nodes.size(0))]
            d_id = 0
            for node in keep_nodes:
                while(d_id+1 < len(drop_nodes) and drop_nodes[d_id+1] < node): #Check if the next one is still smaller than us
                    d_id += 1
                node_maps[node] = node-drop_nodes[d_id]
            
            self.node_tensors[PID] = self.node_tensors[PID][keep_nodes] # Filter nodes
            self.label_tensors[PID] = self.label_tensors[PID][keep_nodes] # Filter nodes

            #remove the dead edges
            edge_msk = torch.tensor([False for _ in range(edges.size(0))])
            keep_nodes = set(keep_nodes)
            for i, val in enumerate(edges.T[0]):
                if val in keep_nodes:
                    edge_msk[i] = True
            
            #Remap the edges
            edges = edges[edge_msk]
            for i, e in enumerate(edges):
                edges[i][0] = node_maps[e[0]]
                edges[i][1] = node_maps[e[1]]
            self.edge_tensors[PID] = edges"""


    """def inject_central_node(self, undirected=True):
        for PID, node_tensor in enumerate(self.node_tensors):
            node = torch.tensor([[0 for _ in range(node_tensor.size(1))]])
            node_id = node_tensor.size(0)
            roots = (node_tensor[:,4] == 0).nonzero().flatten().tolist()
            
            extra_edges = [[],[]]
            for r in roots:
                extra_edges[0].append(node_id)
                extra_edges[1].append(r)
                if undirected:
                    extra_edges[0].append(r)
                    extra_edges[1].append(node_id)
            extra_edges = torch.tensor(extra_edges).T
            self.node_tensors[PID] = torch.cat((self.node_tensors[PID], node))
            self.edge_tensors[PID] = torch.cat((self.edge_tensors[PID], extra_edges), dim=0)
            self.label_tensors[PID]= torch.cat((self.label_tensors[PID], torch.tensor([0])))"""
            


    #Returns a list of lists where each index in the list represents a patient and each patient consists of [tensor(nodes), tensor(edges), tensor(labels), list(patient)]
    def get_torch_data(self):
        return [list(a) for a in zip(self.node_tensors, self.edge_tensors, self.label_tensors, self.pIDs)]

    #Returns a numpy array of all nodes in self.nodeDict per patient (array[patient][node])
    def get_node_list(self):
        res = []
        for p in range(len(self.node_tensors)):
            res.append(np.c_[self.node_tensors[p].numpy(), self.label_tensors[p].numpy()])
        return res

    def flatten(segments):
        stop = 0
        for item in segments:
            if type(item) == list: #new part
                break
            stop += 1
        section = segments[:stop]

        for lists in segments[stop:]:
            section.extend(dataLoader.flatten(lists))
        return section

    #General helper functions
    def get_roots(nodes, edges):
            roots = []
            for node_idx, n in enumerate(nodes):
                incoming = (edges[1] == node_idx).nonzero().tolist()
                if not incoming:
                    roots.append(node_idx)
            return roots

    def get_segment(node, predecessor, edges):
        seg = []
        recipients = True
        while recipients:
            recipients = edges[1][(edges[0] == node)].tolist()
            if node in recipients: #No self loops
                recipients.remove(node)
            if predecessor in recipients: #Don't go backwards
                recipients.remove(predecessor)
            if len(recipients) == 0: #Leaf!
                seg.append(node)
            elif len(recipients) == 1: #Append it
                seg.append(node)
                predecessor = node
                node = recipients[0]
            elif len(recipients) > 1: #Bifurcation
                sub_segs = []
                for r in recipients:
                    res = dataLoader.get_segment(r, node, edges)
                    sub_segs.append(res)
                break
        return seg