from ThesisModel import ThesisModelInterface

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np

from sklearn import metrics

class NeuralNet(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.bnry = False
        if n_classes == 1:
            self.bnry = True

        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(128, n_classes)

    def forward(self, data):
        x = data[0] #Extract node tensors
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        if self.bnry:
            x = torch.sigmoid(x)
            return torch.flatten(x)

        return F.log_softmax(x, dim=1)

class NNModel(ThesisModelInterface):
    def __init__(self, data, labels, test_set_idx):
        super().__init__(data, labels, test_set_idx)
        self.clfName = "Neural Network"
        self.n_node_features = len(data[0][0][0])
        self.n_labels = len(labels)
        if self.n_labels == 2: #Binary classification
            self.n_labels = 1
        self.n_epochs = 150

    def train_model(self, replace_model=True):
        "Function to fit the model to the data"
        if self.clf is None or replace_model is True:
            self.clf = NeuralNet(self.n_node_features, self.n_labels).to(self.device)

        optimizer = torch.optim.Adam(self.clf.parameters(), lr=0.0005, weight_decay=1e-4)
        self.clf.train()
        tf1 = []
        vf1 = []
        tloss = []
        lss_fnc = F.nll_loss
        if self.clf.bnry:
            #lss_fnc = nn.BCELoss()
            def BCELoss_class_weighted(weights):
                def loss(input, target):
                    input = torch.clamp(input,min=1e-7,max=1-1e-7)
                    bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
                    return torch.mean(bce)
                return loss

            lss_fnc = BCELoss_class_weighted([1,4])
            optimizer = torch.optim.Adam(self.clf.parameters(), lr=0.001, weight_decay=1e-6)

        #valid_step = int(self.n_epochs / 10)
        best_mod = copy.deepcopy(self.clf.state_dict())
        for epoch in range(self.n_epochs+1):
            train_f1 = 0.0
            tot_lss = 0.0

            for data in self.train: #For every graph in the data set
                optimizer.zero_grad()
                out = self.clf(data) #Get the labels from all the nodes in one graph (Each node gets 12 outputs: one for each class)

                train_f1 += metrics.f1_score(data[2].cpu().detach().numpy(), (out > self.threshold).int().cpu().detach().numpy()) #Maybe do the PR curve to get the best F1
                loss = lss_fnc(out, data[2])
                tot_lss += loss.item()
                loss.backward()
                optimizer.step()
            
            tloss.append(tot_lss / len(self.train))
            train_f1 = train_f1 / len(self.train)
            tf1.append(train_f1)

            if epoch % 10 == 0 and epoch > 0:
                self.clf.train(mode=False)
                valid_f1 = self.validate_model()
                vf1.append(valid_f1)
                self.clf.train()
                if valid_f1 >= np.max(vf1):
                    best_mod = copy.deepcopy(self.clf.state_dict())
        
        self.clf.load_state_dict(best_mod)
        self.clf.train(mode=False)
        return tf1, tloss, vf1
