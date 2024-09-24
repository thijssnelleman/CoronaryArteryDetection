from cmath import log
import numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from sklearn import datasets
from sklearn.model_selection import cross_val_score

from dataClasses import dataLoader
from rf import RandomForest


import torch
from cluster_pool import ClusterPooling
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn import metrics
import math

class GUNET(torch.nn.Module):
    archName = "Graph UNET2"
    #def __init__(self, features, labels, pType=ClusterPooling):
    def __init__(self, train_data, valid_data, lr=0.00020, wd=0.0, pType=ClusterPooling, hidden_channels=128, depth=3, epochs=100, dropout=0.1, pdropout=0.05, loss_factor=4.0):
        super().__init__()

        self.train_data = train_data
        self.valid_data = valid_data
        self.in_channels = self.train_data[0][0].size(1)
        self.out_channels = 1
        self.hidden_channels = hidden_channels
        #if self.out_channels == 2:
        #    self.out_channels = 1
        self.depth = depth
        self.n_epochs = epochs
        self.num_classes = self.out_channels
        
        self.dropoutval = dropout
        self.pooldropoutval = pdropout
        self.dropout = torch.nn.Dropout(p=self.dropoutval)

        self.poolingType = pType
        self.threshold = 0.5

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(self.in_channels, self.hidden_channels, improved=True))
        for i in range(self.depth):
            self.pools.append(self.poolingType(self.hidden_channels, dropout=self.pooldropoutval))
            self.down_convs.append(GCNConv(self.hidden_channels, self.hidden_channels, improved=True))

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth-1):
            self.up_convs.append(GCNConv(self.hidden_channels*2, self.hidden_channels, improved=True))
        self.up_convs.append(GCNConv(self.hidden_channels*2+self.in_channels, self.out_channels, improved=True))

        self.optimizer = torch.optim.Adam(self.mdl.parameters(), lr=lr, weight_decay=wd)
        
        def BCELoss_class_weighted(weights):
            def loss(input, target):
                input = torch.clamp(input,min=1e-7,max=1-1e-7)
                bce = - weights[1] * target * torch.log(input) - \
                        weights[0] * (1 - target) * torch.log(1 - input)
                return torch.mean(bce)
            return loss
        self.loss_func = BCELoss_class_weighted([1,loss_factor])

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

    def validate_model(self):
        y_valid_pred = []
        
        y_valid = []
        for data in self.valid_data: #For every graph in the data set
            out = self.forward(data) #Get the labels from all the nodes in one graph 

            #if type(out) == tuple: out = out[0]

            labels = ((out > self.threshold).int()).cpu().detach().numpy()
            y_valid_pred.extend(labels.tolist())
            y_valid.extend(data[2].int().cpu().detach().numpy().tolist())
        return metrics.f1_score(y_valid, y_valid_pred)

    def run_folds(self, folds=1, display=False, only_scores=True):
        
        #scores = []
        tf1 = []
        tloss = []
        vf1 = []
        for _ in range(folds):
            for epoch in range(self.n_epochs + 1):
                tot_lss = 0.0   
                train_f1 = 0.0
            
                for index, data in enumerate(self.train_data): #For every graph in the data set
                    self.optimizer.zero_grad()
                    out = self.forward(data) #Get the labels from all the nodes in one graph
                    class_lbls = data[2]
                    
                    loss = self.loss_func(out, class_lbls) #Now get the loss based on these outputs and the actual labels of the graph
                    
                    train_f1 += metrics.f1_score(class_lbls.cpu().detach().numpy(), (out > self.threshold).int().cpu().detach().numpy(), zero_division=0)
                    tot_lss += loss.item()
                   

                    if math.isnan(loss.item()):
                        print("\t\tError in loss in Epoch: " + str(epoch+1) + "/" + str(self.n_epochs))
                        if torch.isnan(out).nonzero().size(0) > 0: #We have a nan output
                            print("\t\t Error in output of the network: " + str(torch.isnan(out).nonzero().size(0)), " nan values")
                        return tf1, tloss, vf1

                    loss.backward()                
                    self.optimizer.step()
                
                train_f1 = train_f1 / len(self.train_data)
                tf1.append(train_f1)
                tloss.append(tot_lss/ len(self.train_data))

                if epoch % 10 == 0 and epoch > 0:
                    self.train(mode=False)
                    valid_f1 = self.validate_model()
                    self.train()
                    
                    prec, rec, threshold =  metrics.precision_recall_curve(self.y_valid, self.y_valid_dist)
                    if not ((prec+rec) == 0).any():
                        f1s = (2*(prec * rec)) / (prec + rec)
                        valid_f1 = np.max(f1s)
                    vf1.append(valid_f1)
                    if valid_f1 >= np.max(vf1): #Best validation score thusfar
                        #best_mod = copy.deepcopy(self.state_dict())
                        if not ((prec+rec) == 0).any(): #Can we calculate the best threshold?
                            self.threshold = threshold[np.argmax(f1s)] #Set the threshhold to the most optimal one

                    if display:  
                        print("\n")
                        print("\t\tLoss in Epoch " + str(epoch) + ": " + str(tot_lss))
                        print(f"\t\tValid F1 score: {valid_f1:.4f} (Best: {np.max(vf1):.4f}, Thresh: {self.threshold:.4f})")
                        print(f"\t\tTrain F1: {train_f1:.4f}")

            #self.load_state_dict(best_mod)
            self.train(mode=False)
            #return tf1, tloss, vf1
            return np.max(vf1)



iris = datasets.load_iris()



dl = dataLoader()
data = dl.get_torch_data()



n_lbls = list(range(len(dl.labels)))


def train_random_forest(config):
    """ 
    Trains a random forest on the given hyperparameters, defined by config, and returns the accuracy
    on the validation data.

    Input:
        config (Configuration): Configuration object derived from ConfigurationSpace.

    Return:
        cost (float): Performance measure on the validation data.
    """
    
    #clf = RandomForest(data, n_lbls, n_estimators=config["trees"], max_depth=config["depth"])
    testclf = GUNET(10, 2)
    #model.fit(X_train, y_train)
    #scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    scores = clf.run_folds(folds=5, display=False, only_scores=True)
    
    #print(config["depth"], model.score(X_val, y_val))
    # define the evaluation metric as return
    print(1 - np.mean(scores))
    #print(config)
    return 1 - np.mean(scores)


if __name__ == "__main__":
    # Define your hyperparameters
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter("depth", 2, 500))
    cs.add_hyperparameter(UniformIntegerHyperparameter("trees", 2, 100))

    hpspace = ConfigurationSpace()
    hpspace.add_hyperparameter(UniformFloatHyperparameter("lr", lower=1e-10, upper=1.0, log=True))
    hpspace.add_hyperparameter(UniformIntegerHyperparameter("depth", 1, 6))
    hpspace.add_hyperparameter(UniformIntegerHyperparameter("width", 16, 1024))
    hpspace.add_hyperparameter(UniformFloatHyperparameter("loss_scale", lower=1.0, upper=8.0))


    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 50,  # Max number of function evaluations (the more the better)
        "cs": cs,
    })

    smac = SMAC4HPO(scenario=scenario, tae_runner=train_random_forest)

    def_value = train_random_forest(cs.get_default_configuration())
    print("Default Value: %.2f" % (def_value))

    best_found_config = smac.optimize()

    inc_value = train_random_forest(best_found_config)
    #for attempt in smac.runhistory.get_all_configs():
    #    print(attempt, train_random_forest(attempt))
    print(best_found_config)
    print("Optimized Value: %.8f" % (inc_value))