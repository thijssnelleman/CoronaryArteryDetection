from dataClasses import dataLoader
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, Linear, GATConv
from torch_geometric.nn.pool import EdgePooling
from torch_geometric.data import Data
import numpy as np
import sklearn.metrics as metrics

#Can we transform this to doing edge labelling instead of node?
class GraphConvNNEdge(torch.nn.Module):
    archName = "Graph Convolutional Neural Network for Edge Prediction"
    def __init__(self, node_features, num_classes):
        super().__init__()
        self.n_epochs = 10
        self.num_classes = num_classes
        if self.num_classes == 2: #binary
            self.num_classes = 1

        hidden_channel = 64
        self.conv1 = GCNConv(node_features, hidden_channel)
        self.conv2 = GCNConv(hidden_channel+node_features, hidden_channel)
        self.conv3 = GCNConv(hidden_channel+node_features, hidden_channel)
        self.conv4 = GCNConv(hidden_channel+node_features, hidden_channel)
        self.lin1 = Linear(hidden_channel+node_features, hidden_channel)
        self.lin2 = Linear(hidden_channel+node_features, out_channels=self.num_classes)

        def BCELoss_class_weighted(weights):
            def loss(input, target):
                input = torch.clamp(input,min=1e-7,max=1-1e-7)
                bce = - weights[1] * target * torch.log(input) - \
                        weights[0] * (1 - target) * torch.log(1 - input)
                return torch.mean(bce)
            return loss

        self.loss_func = BCELoss_class_weighted([500,1])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00025)

    def forward(self, data):
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index

        #print(x.type())
        #print(edge_index.type())

        x_in = x.clone()

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = torch.cat((x_in, x), -1)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = torch.cat((x_in, x), -1)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = torch.cat((x_in, x), -1)
        x = self.conv4(x, edge_index)
        x = F.relu(x)

        x = F.relu(self.lin1(torch.cat((x_in, x), -1)))
        x = self.lin2(torch.cat((x_in, x), -1))

        if self.num_classes == 1: #binary
            return torch.flatten(torch.sigmoid(x))

        return F.log_softmax(x, dim=1)

    def train(self, data):

        for edx in range(self.n_epochs+1):
            pos_f1 = 0.0
            neg_f1 = 0.0
            tot_lss = 0.0
            fpr = 0.0
            selection_rate = 0.0

            for patient in data:
                nodes, edg, class_lbls = patient[0], patient[1], patient[2]
                out = self.forward([nodes, edg])

                loss = self.loss_func(out, class_lbls)
                pos_f1 += metrics.f1_score(class_lbls.cpu().detach().numpy(), torch.round(out).cpu().detach().numpy(), zero_division=0)
                neg_f1 += metrics.f1_score(class_lbls.cpu().detach().numpy(), torch.round(out).cpu().detach().numpy(), zero_division=0, pos_label=0)
                fpr += torch.sum((class_lbls != torch.round(out))[(class_lbls == 0).nonzero()]) / torch.sum(class_lbls == 0)
                selection_rate += torch.sum(torch.round(out)) / out.size(0)
                
                tot_lss += loss.item()
                loss.backward()
                self.optimizer.step()
            
            pos_f1 = pos_f1 / len(data)
            neg_f1 = neg_f1 / len(data)
            fpr = fpr / len(data)
            selection_rate = selection_rate / len(data)

            print(f"Epoch {edx} Positive F1: {pos_f1:.5f}, Negative F1: {neg_f1:.5f}, FPR: {fpr:.5f}, Selection rate: {selection_rate:.4f}")

class test_edgeNN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super().__init__()
        self.n_epochs = 10
        self.num_classes = num_classes
        if self.num_classes == 2: #binary
            self.num_classes = 1

        hidden_channel = 64
        self.lin1 = Linear(node_features*2, hidden_channel)
        self.lin2 = Linear(hidden_channel+node_features*2, hidden_channel)
        self.lin3 = Linear(hidden_channel+node_features*2, hidden_channel)
        self.linout = Linear(hidden_channel+node_features*2, out_channels=self.num_classes)

        def BCELoss_class_weighted(weights):
            def loss(input, target):
                input = torch.clamp(input,min=1e-7,max=1-1e-7)
                bce = - weights[1] * target * torch.log(input) - \
                        weights[0] * (1 - target) * torch.log(1 - input)
                return torch.mean(bce)
            return loss

        self.loss_func = BCELoss_class_weighted([500,1])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00025)

    #Here, x is two nodes that have an edge
    def forward(self, x):
        x_in = x.clone()

        x = F.relu(self.lin1(x))

        x = torch.cat((x, x_in), -1)
        x = F.relu(self.lin2(x))

        x = torch.cat((x, x_in), -1)
        x = F.relu(self.lin3(x))

        x = torch.cat((x, x_in), -1)
        x = self.linout(x)
        return torch.flatten(torch.sigmoid(x))            

    def train(self, data):

        for edx in range(self.n_epochs+1):
            pos_f1 = 0.0
            neg_f1 = 0.0
            tot_lss = 0.0
            fpr = 0.0
            selection_rate = 0.0

            for patient in data:
                nodes, edg, labels = patient[0], patient[1].T, patient[2]
                nodes_left = labels[edg[0]] #Get the nodes' labels 
                nodes_right = labels[edg[1]]

                class_lbls = (nodes_left == nodes_right).type(torch.float)
                e = torch.cat([nodes[edg[0]], nodes[edg[1]]], dim=-1) 
                out = self.forward(e)

                loss = self.loss_func(out, class_lbls)
                pos_f1 += metrics.f1_score(class_lbls.cpu().detach().numpy(), torch.round(out).cpu().detach().numpy(), zero_division=0)
                neg_f1 += metrics.f1_score(class_lbls.cpu().detach().numpy(), torch.round(out).cpu().detach().numpy(), zero_division=0, pos_label=0)
                fpr += torch.sum((class_lbls != torch.round(out))[(class_lbls == 0).nonzero()]) / torch.sum(class_lbls == 0)
                selection_rate += torch.sum(torch.round(out)) / out.size(0)
                
                tot_lss += loss.item()
                loss.backward()
                self.optimizer.step()
            
            pos_f1 = pos_f1 / len(data)
            neg_f1 = neg_f1 / len(data)
            fpr = fpr / len(data)
            selection_rate = selection_rate / len(data)

            print(f"Epoch {edx} Positive F1: {pos_f1:.5f}, Negative F1: {neg_f1:.5f}, FPR: {fpr:.5f}, Selection rate: {selection_rate:.4f}")



dl = dataLoader(processed=False, asBinary=True)

tensors = dl.get_torch_data()

clf = test_edgeNN(tensors[0][0].size(1), 2)

clf.train(tensors)

#Lets try and flip the edges and nodes: each edge becomes the concatenation of the two nodes, each node determining to which edges they connect

new_nodes = []
new_edges = []
new_labels = []

new_tensors = []

for patient in tensors:
    
    nodes, edges, labels, pname = patient
    print(pname)
    edges = edges.T

    nodes_left = labels[edges[0]] #Get the nodes' labels 
    nodes_right = labels[edges[1]]
    class_lbls = (nodes_left == nodes_right).type(torch.float)

    n_nodes = torch.cat([nodes[edges[0]], nodes[edges[1]]], dim=-1) #Create the new nodes as a concatenation of the two participating in an edge
    n_edges = [[], []]

    for edge_id, e in enumerate(edges.T):
        out_node, inc_node = e[0], e[1]
        related_edges_left = torch.cat(((edges[0] == out_node).nonzero().flatten(), (edges[1] == out_node).nonzero().flatten())) #Get the edges attached to the left node
        related_edges_right = torch.cat(((edges[0] == inc_node).nonzero().flatten(), (edges[1] == inc_node).nonzero().flatten())) #Get the eges attached to the right node
        related_edges = torch.unique(torch.cat((related_edges_left, related_edges_right))).tolist()

        n_edges[0].extend(edge_id for _ in range(len(related_edges)))
        n_edges[1].extend(related_edges)
    
    new_tensors.append([n_nodes, torch.tensor(n_edges, dtype=torch.long).T, class_lbls])
    #new_nodes.append(n_nodes)
    #new_edges.append(torch.tensor(n_edges, dtype=torch.long).T)
    #new_labels.append(class_lbls)
    #print(new_nodes[-1].size())
    #print(new_edges[-1].size())
    #print(new_labels[-1].size())

clf = GraphConvNNEdge(new_tensors[0][0].size(1), 2)

#print(new_tensors[0][0].T.type())

clf.train(new_tensors[:-10])

x = new_tensors[-1][0]
e = new_tensors[-1][1]

e = torch.cat([x[e[0]], x[e[1]]], dim=-1) 
example = clf(e)
    
epool = EdgePooling(tensors[0][0].size(1))

batch = torch.tensor(np.zeros(x.shape[0])).long().to(x.get_device())
x, edge_index, batch, unpool1 = epool.__merge_edges_sigmoid__(x, e,  batch, example)

dl.create_cluster_vis(unpool1.cluster_map, tensors[-1][-1], "Learned Edge Pool labels example")

exit()

error_cluster_sizes = [0 for x in range(30)]

def get_roots(nodes, edges):
    roots = []
    for node_idx, n in enumerate(nodes):
        recipients = edges[1][(edges[0] == node_idx)].tolist()
        is_root = True
        if node_idx in recipients: recipients.remove(node_idx)

        if len(recipients) > 1:
            continue
        elif nodes[recipients[0]][5] < n[5]: #If we are connected to a node with a lower predecessor label
            continue
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
        if len(recipients) == 1: #Append it
            seg.append(node)
            predecessor = node
            node = recipients[0]
        elif len(recipients) > 1: #Bifurcation
            sub_segs = []
            for r in recipients:
                res = get_segment(r, node, edges)
                sub_segs.append(res)
            break
    return seg

def fix_noise(segments, seen_green=False):
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
            if length == 0: #Skip to the first 1 label
                continue
            if seen_green:
                labels[cur_nodes] = 1 #Fix the labels
                cur_nodes = []
            else:
                seen_green = True
            length = 0
    
    del segments[:stop]
    for lists in segments:
        fix_noise(lists, seen_green=seen_green)

def check_succesion(segments, nodes):
    r = 0
    stop = 0
    for item in segments:
        if type(item) == list: #new part
            break
        stop += 1
    section = segments[:stop]

    for i, node_idx in enumerate(section[1:]):
        if nodes[node_idx][4] <= nodes[section[i-1]][4]:
            r += 1

    del segments[:stop]
    for lists in segments:
        r += check_succesion(lists)
    return r  

"""for patient in tensors:
    nodes = patient[0]
    edges = patient[1].T
    labels = patient[2]
    roots = get_roots(nodes, edges)
    print(patient[3])

    #print(nodes[roots].T[4:6].T)
    for r in roots:
    #    print(nodes[r])
        if nodes[r][5] != 0:
            print("Root starting at: ", nodes[r][5])
    #break
    #print(len(roots) / len(nodes))
    #print((labels == 1).nonzero().flatten().size())
    for root_idx in roots: #Now get all of the segments
        #print("hi")
        segments = get_segment(root_idx, None, edges)
        
        cnt = check_succesion(segments, nodes)
        if cnt > 0:
            print("Succesion feature noise: ", cnt, ". Proportionality: ", cnt/nodes.size(0))
        #noises = get_noise(segments)
        #for n in noises:
        #    print("Noise of length: ", n)
        #    error_cluster_sizes[n] += 1
    break"""
#print(error_cluster_sizes)

#import matplotlib.pyplot as plt

#plt.bar(list(range(len(error_cluster_sizes))), error_cluster_sizes)
#plt.show()



                



            


        