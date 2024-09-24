import numpy as np
import math
import copy

import sklearn

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, GATConv
from torch_geometric.nn.pool import EdgePooling
from cluster_pool import ClusterPooling

from DAGpool import DAGPool

from torch_geometric.data import Data
from sklearn import metrics

from ThesisModel import ThesisModelInterface

class GraphConvNN(torch.nn.Module):
    archName = "Graph Convolutional Neural Network"
    def __init__(self, node_features, num_classes):
        super().__init__()
        self.n_epochs = 500
        self.num_classes = num_classes
        if self.num_classes == 2: #binary
            self.num_classes = 1

        hidden_channel = 128

        self.conv1 = GCNConv(node_features, hidden_channel)
        self.conv2 = GCNConv(hidden_channel+node_features, hidden_channel)
        self.conv3 = GCNConv(hidden_channel+node_features, hidden_channel)
        self.conv4 = GCNConv(hidden_channel+node_features, hidden_channel)
        self.lin1 = Linear(hidden_channel+node_features, hidden_channel)
        self.lin2 = Linear(hidden_channel+node_features, out_channels=self.num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00020, weight_decay=1e-4)

    def forward(self, data):
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index

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

class GraphConvPoolNN(torch.nn.Module):
    archName = "GCN Pooling"
    def __init__(self, node_features, num_classes):
        super().__init__()
        self.n_epochs = 500
        self.num_classes = num_classes
        if self.num_classes == 2: #binary
            self.num_classes = 1

        hid_channel = 64
        self.dropout = torch.nn.Dropout(p=0.1) #Should've given this to the pooling layers too?

        self.conv1 = GCNConv(node_features, hid_channel)
        
        self.pool1 = EdgePooling(hid_channel+node_features)
        self.pool2 = EdgePooling(hid_channel+node_features)
        self.pool3 = EdgePooling(hid_channel+node_features)
        self.conv2 = GCNConv(hid_channel+node_features, hid_channel)
        self.conv3 = GCNConv(hid_channel*2+node_features, hid_channel)
        self.conv4 = GCNConv(hid_channel*2+node_features, hid_channel)
        self.fc1 = torch.nn.Linear(hid_channel+node_features, hid_channel)
        self.fc2 = torch.nn.Linear(hid_channel+node_features, self.num_classes)

    def forward(self, data):
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        x_in = torch.clone(x)

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)
        
        x = torch.cat((x_in, x), -1) #Skip here from input

        batch = torch.tensor(np.zeros(x.shape[0])).long().to(x.get_device()) #Make a batch tensor of np.zeros of length num nodes
        
        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)
        x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)
        x, edge_index, batch, unpool3 = self.pool3(x, edge_index.long(), batch)
        
        x_pool3 = x.clone()

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)        

        x = self.conv3(torch.cat((x_pool3, x), -1), edge_index) #Skip connection
        x = self.dropout(x)
        x = F.relu(x)

        x = self.conv4(torch.cat((x_pool3, x), -1), edge_index) #Skip connection
        x = self.dropout(x)
        x = F.relu(x)
        
        x, edge_index, batch = self.pool3.unpool(x, unpool3)
        x, edge_index, batch = self.pool2.unpool(x, unpool2)
        x, edge_index, batch = self.pool1.unpool(x, unpool1)

        x = torch.cat((x_in, x), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(torch.cat((x_in, x), -1)) 
        
        if self.num_classes == 1: #binary
            return torch.flatten(torch.sigmoid(x))
        return F.log_softmax(x, dim=1)

class GraphConvDAGPoolNN(torch.nn.Module):
    archName = "GCN DAG Pooling"
    def __init__(self, node_features, num_classes):
        super().__init__()
        self.n_epochs = 500
        self.num_classes = num_classes
        if self.num_classes == 2: #binary
            self.num_classes = 1

        self.hid_channel = 128
        self.nfeatures = node_features
        
        self.cluster_sizes = 3
        self.dropout = torch.nn.Dropout(p=0.5)

        self.conv1 = GCNConv(node_features, self.hid_channel)
        self.conv2 = GCNConv(self.hid_channel+node_features, self.hid_channel)
        self.pool1 = DAGPool(5)
        self.conv3 = GCNConv(self.hid_channel+node_features, self.hid_channel)
        self.fc1 = torch.nn.Linear(self.hid_channel*2+node_features, self.hid_channel)
        #self.conv3 = GCNConv((self.hid_channel+node_features)*self.cluster_sizes, self.hid_channel)
        #self.fc1 = torch.nn.Linear(self.hid_channel + (self.hid_channel+node_features)*self.cluster_sizes, self.hid_channel)
        self.pool2 = DAGPool(self.cluster_sizes)
        self.conv4 = GCNConv(self.hid_channel, self.hid_channel)
        self.fc2 = torch.nn.Linear(self.hid_channel*2, self.hid_channel)
        #self.conv4 = GCNConv(self.hid_channel*self.cluster_sizes, self.hid_channel)
        #self.fc2 = torch.nn.Linear(self.hid_channel + self.hid_channel*self.cluster_sizes, self.hid_channel)
        self.pool3 = DAGPool(self.cluster_sizes)
        self.conv5 = GCNConv(self.hid_channel, self.hid_channel)
        self.fc3 = torch.nn.Linear(self.hid_channel*2, self.hid_channel)
        #self.conv5 = GCNConv(self.hid_channel*self.cluster_sizes, self.hid_channel)
        #self.fc3 = torch.nn.Linear(self.hid_channel + self.hid_channel*self.cluster_sizes, self.hid_channel)
        
        self.fc4 = torch.nn.Linear(self.hid_channel+node_features, self.hid_channel)
        self.fc5 = torch.nn.Linear(self.hid_channel+node_features, self.num_classes)


    def forward(self, data):
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        x_in = torch.clone(x)

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)

        x = torch.cat((x, x_in), -1)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)
        
        x = torch.cat((x_in, x), -1) #Skip here from input

        #batch = torch.tensor(np.zeros(x.shape[0])).long().to(x.get_device()) #Make a batch tensor of np.zeros of length num nodes
        
        x, edge_index, unpool1 = self.pool1(x, edge_index.long())
        x_out = torch.clone(x)
        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)
        
        #print(x_out.size(), x.size())

        x = torch.cat((x_out, x), -1)
        #print(x.size())
        #print("Test: ", (self.hid_channel+self.nfeatures)*self.cluster_sizes+self.hid_channel)
        x = self.fc1(x)
        self.dropout(x)
        x = F.relu(x)
        

        x, edge_index, unpool2 = self.pool2(x, edge_index.long())
        x_out = torch.clone(x)
        x = self.conv4(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)

        x = torch.cat((x_out, x), -1)
        x = self.fc2(x)
        self.dropout(x)
        x = F.relu(x)

        x, edge_index, unpool3 = self.pool3(x, edge_index.long())
        x_out = torch.clone(x)
        x = self.conv5(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)

        x = torch.cat((x_out, x), -1)
        x = self.fc3(x)
        self.dropout(x)
        x = F.relu(x)
        
        x, edge_index = self.pool3.unpool(unpool3)
        x, edge_index = self.pool2.unpool(unpool2)
        x, edge_index = self.pool1.unpool(unpool1)



        #x = torch.cat((x_in, x), -1)
        x = self.fc4(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc5(torch.cat((x_in, x), -1)) 
        
        if self.num_classes == 1: #binary
            return torch.flatten(torch.sigmoid(x))
        return F.log_softmax(x, dim=1)

class GraphGATPoolNN(torch.nn.Module):
    archName = "GAT Pooling"
    def __init__(self, node_features, num_classes):
        super().__init__()
        self.n_epochs = 200
        self.num_classes = num_classes
        if self.num_classes == 2: #binary
            self.num_classes = 1

        hid_channel = 64
        heads = 2

        self.gat1 = GATConv(node_features, hid_channel, heads=heads)         #Replace with GAT, use 2-4 heads MAX (Heads double the amount of output features)
        self.pool1 = EdgePooling(hid_channel*heads+node_features)
        self.pool2 = EdgePooling(hid_channel*heads+node_features)
        self.pool3 = EdgePooling(hid_channel*heads+node_features)
        self.gat2 = GATConv(hid_channel*heads+node_features, hid_channel, heads=heads) #Replace with GAT, use 2-4 heads MAX
        self.gat3 = GATConv(hid_channel*heads*2+node_features, hid_channel, heads=heads) #Replace with GAT, use 2-4 heads MAX
        self.gat4 = GATConv(hid_channel*heads*2+node_features, hid_channel, heads=heads) #Replace with GAT, use 2-4 heads MAX
        self.fc1 = torch.nn.Linear(hid_channel*heads+node_features, hid_channel)
        self.fc2 = torch.nn.Linear(hid_channel+node_features, self.num_classes)


    def forward(self, data):
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        x_in = torch.clone(x)

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        
        x = torch.cat((x_in, x), -1) #Skip here from input

        batch = torch.tensor(np.zeros(x.shape[0])).long().to(x.get_device()) #Make a batch tensor of np.zeros of length num nodes
        
        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)
        x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)
        x, edge_index, batch, unpool3 = self.pool3(x, edge_index.long(), batch)
        
        x_pool3 = x.clone()

        x = self.gat2(x, edge_index)
        x = F.relu(x)        

        x = self.gat3(torch.cat((x_pool3, x), -1), edge_index) #Skip connection
        x = F.relu(x)

        x = self.gat4(torch.cat((x_pool3, x), -1), edge_index) #Skip connection
        x = F.relu(x)
        
        x, edge_index, batch = self.pool3.unpool(x, unpool3)
        x, edge_index, batch = self.pool2.unpool(x, unpool2)
        x, edge_index, batch = self.pool1.unpool(x, unpool1)

        x = F.relu(self.fc1(torch.cat((x_in, x), -1)))
        x = self.fc2(torch.cat((x_in, x), -1)) #Skip here from x_in
        
        if self.num_classes == 1: #binary
            return torch.flatten(torch.sigmoid(x))
        return F.log_softmax(x, dim=1)

class GraphConvNewPoolNN(torch.nn.Module):
    archName = "GCN Cluster Pooling"
    def __init__(self, node_features, num_classes):
        super().__init__()
        self.n_epochs = 500
        
        self.num_classes = num_classes
        if self.num_classes == 2: #binary
            self.num_classes = 1
        self.minsize = 0

        self.hidden_channel = 128
        self.clusmap = None
        self.dropoutp = 0.1

        self.dropout = torch.nn.Dropout(p=self.dropoutp)

        self.conv1 = GCNConv(node_features, self.hidden_channel)              

        self.pool1 = ClusterPooling(self.hidden_channel+node_features, dropout=self.dropoutp)
        self.conv4 = GCNConv(self.hidden_channel+node_features, self.hidden_channel)

        self.pool2 = ClusterPooling(2*self.hidden_channel+node_features, dropout=self.dropoutp)
        self.conv5 = GCNConv(self.hidden_channel*2+node_features, self.hidden_channel)

        self.pool3 = ClusterPooling(3*self.hidden_channel+node_features, dropout=self.dropoutp)
        self.conv6 = GCNConv(self.hidden_channel*3+node_features, self.hidden_channel)

        self.conv7 = GCNConv(self.hidden_channel+node_features, self.hidden_channel)
        self.fc1 = torch.nn.Linear(self.hidden_channel + node_features, self.hidden_channel)
        self.fc2 = torch.nn.Linear(self.hidden_channel + node_features, self.num_classes)

    def forward(self, data):
        data = Data(x=data[0], edge_index=data[1].t().contiguous())

        x, edge_index = data.x, data.edge_index
        x_in = torch.clone(x)

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)

        x = torch.cat((x_in, x), -1) #Skip connection

        batch = torch.tensor(np.zeros(x.shape[0])).long().to(x.get_device()) #Make a batch tensor of np.zeros of length num nodes
        
        x, edge_index, batch, unpool1 = self.pool1(x, edge_index.long(), batch)
        x_pool = x.clone()
        
        self.clusmap = unpool1.cluster_map

        x = self.conv4(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)

        x = torch.cat((x_pool, x), -1) #Skip connection
        x, edge_index, batch, unpool2 = self.pool2(x, edge_index.long(), batch)
        x_pool = x.clone()
    
        x = self.conv5(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)

        x = torch.cat((x_pool, x), -1) #Skip connection
        x, edge_index, batch, unpool3 = self.pool3(x, edge_index.long(), batch)
        x_pool = x.clone()

        x = self.conv6(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)

        #Unpool
        x, edge_index, batch = self.pool3.unpool(x, unpool3)
        x, edge_index, batch = self.pool2.unpool(x, unpool2)
        x, edge_index, batch = self.pool1.unpool(x, unpool1)

        x = torch.cat((x_in, x), -1) #Skip connection

        x = self.conv7(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)

        x = torch.cat((x_in, x), -1) #Skip connection
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = torch.cat((x_in, x), -1) #Skip connection
        x = self.fc2(x)

        if self.num_classes == 1: #binary            
            return torch.flatten(torch.sigmoid(x))
        return F.log_softmax(x, dim=1)

class GUNET(torch.nn.Module):
    archName = "Graph UNET2"
    def __init__(self, features, labels, pType=ClusterPooling):
        super().__init__()

        self.in_channels = features
        self.out_channels = labels
        self.hidden_channels = 128
        if self.out_channels == 2:
            self.out_channels = 1
        self.depth = 3#Try bigger sizes? [1, 10] range makes sense for this problem
        self.n_epochs = 500
        self.num_classes = self.out_channels
        
        self.dropoutval = 0.1
        self.pooldropoutval = 0.05
        self.dropout = torch.nn.Dropout(p=self.dropoutval)

        self.poolingType = pType

        self.show_cluster_plots = True
        self.shown = False
        self.cf1 = [[] for _ in range(self.depth)]
        #self.optim = torch.optim.Adam(self.mdl.parameters(), lr=0.00020)

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(self.in_channels, self.hidden_channels, improved=True))
        for i in range(self.depth):
            self.pools.append(self.poolingType(self.hidden_channels, dropout=self.pooldropoutval))
            self.down_convs.append(GCNConv(self.hidden_channels, self.hidden_channels, improved=True))

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth-1):
            self.up_convs.append(GCNConv(self.hidden_channels*2, self.hidden_channels, improved=True))
        self.up_convs.append(GCNConv(self.hidden_channels*2+self.in_channels, self.out_channels, improved=True)) #+self.in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, data):
        data = Data(x=data[0], edge_index=data[1].t().contiguous(), y=data[2])
        x, edge_index = data.x, data.edge_index

        x_in = torch.clone(x)

        batch = torch.tensor(np.zeros(x.shape[0])).long().to(x.get_device()) #Make a batch tensor of np.zeros of length num nodes
        memory = [] 
        unpool_infos = []
        for i in range(self.depth):
            x = self.down_convs[i](x, edge_index)
            if self.training: x = self.dropout(x)
            x = F.relu(x)
            memory.append(x.clone())
            x, edge_index, batch, unpool = self.pools[i](x, edge_index.long(), batch)
            unpool_infos.append(unpool)            

        memory[0] = torch.cat((memory[0], x_in), -1) #Concatenate the input features to the output of the first convolutional layer
        x = self.down_convs[-1](x, edge_index)

        for i in range(self.depth):
            j = self.depth - 1 - i
            x, edge_index, batch = self.pools[j].unpool(x, unpool_infos.pop())
            x = torch.cat((memory.pop(), x), -1)
            x = self.up_convs[i](x, edge_index)
            if self.training and i < self.depth - 1: x = self.dropout(x)
            x = F.relu(x) if i < self.depth - 1 else x
                    
        return torch.sigmoid(x).flatten()


class GCNModel(ThesisModelInterface):
    def __init__(self, data, labels, test_set_idx, type=GraphConvNN):
        super().__init__(data, labels, test_set_idx)
        self.architecture = type
        
        self.clfName = self.architecture.archName
        if self.architecture == GUNET:
            self.clfName = self.clfName + "- ClusterPool"
        self.n_node_features = len(data[0][0][0])
        self.n_labels = len(labels)

    def train_model(self, replace_model=True):
        "Function to fit the model to the data"
        if self.clf is None or replace_model is True:
            self.clf = self.architecture(self.n_node_features, self.n_labels)
            if self.architecture == GUNET:
                self.clfName = self.clfName + " " + str(self.clf.poolingType)
            self.clf.to(self.device)

        if hasattr(self.clf, "optimizer"):
            optimizer = self.clf.optimizer
        else:
            optimizer = torch.optim.Adam(self.clf.parameters(), lr=0.00005, weight_decay=1e-4)
            #optimizer = torch.optim.Adam(self.clf.parameters(), lr=0.00025, weight_decay=1e-4)
        

        self.clf.train()
        loss_func = F.nll_loss #nll_loss is logloss
        if self.clf.num_classes == 1: #Binary
            
            def BCELoss_class_weighted(weights):
                def loss(input, target):
                    input = torch.clamp(input,min=1e-7,max=1-1e-7)
                    bce = - weights[1] * target * torch.log(input) - \
                            weights[0] * (1 - target) * torch.log(1 - input)
                    return torch.mean(bce)
                return loss
            loss_func = BCELoss_class_weighted([1,4])
            

        tf1 = []
        tloss = []
        vf1 = []
        nr = []

        best_mod = copy.deepcopy(self.clf.state_dict())
        

        for epoch in range(self.clf.n_epochs + 1):
            tot_lss = 0.0   
            train_f1 = 0.0
            
            y_train_probs = []
            for index, data in enumerate(self.train): #For every graph in the data set
                optimizer.zero_grad()
                out = self.clf(data) #Get the labels from all the nodes in one graph
                class_lbls = data[2]
                
                loss = loss_func(out, class_lbls) #Now get the loss based on these outputs and the actual labels of the graph
                
                #print(metrics.f1_score(class_lbls.cpu().detach().numpy(), (out > self.threshold).int().cpu().detach().numpy(), zero_division=0))
                y_train_probs.extend(out.cpu().detach().numpy().tolist())
                tot_lss += loss.item()
                

                if math.isnan(loss.item()):
                    print("\t\tError in loss in Epoch: " + str(epoch+1) + "/" + str(self.clf.n_epochs))
                    if torch.isnan(out).nonzero().size(0) > 0: #We have a nan output
                        print("\t\t Error in output of the network: " + str(torch.isnan(out).nonzero().size(0)), " nan values")
                    return tf1, tloss, vf1

                loss.backward()              
                optimizer.step()

            #test
            prec, rec, threshold =  sklearn.metrics.precision_recall_curve(self.y_train, y_train_probs)
            if not ((prec+rec) == 0).any():
                f1s = (2*(prec * rec)) / (prec + rec)
                train_f1 = np.max(f1s)
                if len(tf1) == 0 or train_f1 > np.max(tf1):
                    best_mod = copy.deepcopy(self.clf.state_dict())
                    self.threshold = threshold[np.argmax(f1s)]

            #train_f1 = train_f1 / len(self.train)
            tf1.append(train_f1)
            tloss.append(tot_lss/ len(self.train))

            """if epoch % 10 == 0 and epoch > 0:
                self.clf.train(mode=False)
                valid_f1 = self.validate_model()
                self.clf.train()
                
                prec, rec, threshold =  sklearn.metrics.precision_recall_curve(self.y_valid, self.y_valid_dist)
                if not ((prec+rec) == 0).any():
                    f1s = (2*(prec * rec)) / (prec + rec)
                    valid_f1 = np.max(f1s)
                vf1.append(valid_f1)
                if valid_f1 >= np.max(vf1): #Best validation score thusfar
                    best_mod = copy.deepcopy(self.clf.state_dict())
                    if not ((prec+rec) == 0).any(): #Can we calculate the best threshold?
                        self.threshold = threshold[np.argmax(f1s)] #Set the threshhold to the most optimal one
                    
                print("\n")
                print("\t\tLoss in Epoch " + str(epoch) + ": " + str(tot_lss))
                print(f"\t\tValid F1 score: {valid_f1:.4f} (Best: {np.max(vf1):.4f}, Thresh: {self.threshold:.4f})")"""

            
            print(f"\t\tEpoch {epoch} Train F1: {train_f1:.4f}, Best: {np.max(tf1):.4f}")

        self.clf.load_state_dict(best_mod)
        self.clf.train(mode=False)
        return tf1, tloss, vf1