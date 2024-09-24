from re import A
import time
import random
from attr import has
import torch

import numpy as np
import math
from sklearn import metrics

"Model Interface, few base definitions that each model needs"
class ThesisModelInterface:
    "Internal data object. Is in general a two dimensional list of self.data[patient][tensor]"
    "Each patient has three tensors and a string:"
    "[0]: Node tensor, containing each node and its features"
    "[1]: Edge tensor"
    "[2]: Node label tensor, containing the class of each node, index pairwise to [0]"
    "[3]: String indicating from which .json it was created"

    def __init__(self, data, labels, test_set_idx):
        "Receives data from controller"
        self.test = [e for i,e in enumerate(data) if i in test_set_idx]
        self.data = [e for i,e in enumerate(data) if i not in test_set_idx]

        self.labels = labels
        self.n_labels = len(labels)
        self.bnry = (self.n_labels == 2)
        
        self.clf = None
        self.clfName = "ModelInterface"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        random.seed()
        self.maxSeed = 4294967295 #Maximum value in 32 bits
        self.threshold = 0.5 #Standard threshold for binary classifications

        self.cross_test_size = -1
        self.train = []
        self.valid = []

    def generate_train_validation(self, split=0.8, validation=False):
        "Creates a train/validation/test split from the internal data object"

        train_index = int(len(self.data) * split)
        self.train = self.data[:train_index]
        if validation:
            #size = (1 - split) / 2
            #valid_index = int(len(self.data) * (split + size))
            self.valid = self.data[train_index:]
            
        else:
            self.train = self.data

        #If the presented data needs some modification, the following function can be overwritten
        self.format_data_values(validation=validation)

    def generate_k_fold(self, folds):
        "Creates a train/validation/test split based on the number of folds. Returns the number of folds possible."       
        self.cross_test_size = math.ceil(len(self.data)/ folds)
        return math.ceil(len(self.data) / self.cross_test_size)

    #Gets the PR_AUC over the current test set
    #def get_pr_auc(self):
    #    prec, rec, _ =  metrics.precision_recall_curve(self.y_test, self.y_test_pred)
    #    return metrics.auc(rec, prec)

    "Finishes data set formatting after generating train/test sets. Overwrite if no Tensors are used."
    def format_data_values(self, validation=False):
        "Send the tensors to the correct device"
        for i in range(len(self.train)):
            self.train[i][0] = self.train[i][0].to(self.device) #Send nodes
            self.train[i][1] = self.train[i][1].to(self.device) #Send edges
            self.train[i][2] = self.train[i][2].to(self.device) #Send node labels
        
        for i in range(len(self.valid)):
            self.valid[i][0] = self.valid[i][0].to(self.device) #Send nodes
            self.valid[i][1] = self.valid[i][1].to(self.device) #Send edges
            self.valid[i][2] = self.valid[i][2].to(self.device) #Send node labels

        for i in range(len(self.test)):
            self.test[i][0] = self.test[i][0].to(self.device) #Send nodes
            self.test[i][1] = self.test[i][1].to(self.device) #Send edges
            self.test[i][2] = self.test[i][2].to(self.device) #Send node labels

        "Extracts the x and/or y from the train/validation/test sets"
        self.y_train = []
        for patient in self.train:
            self.y_train.extend(patient[2].cpu().numpy().tolist())

        self.y_valid = []
        for patient in self.valid:
            self.y_valid.extend(patient[2].cpu().numpy().tolist())

        self.y_test = []
        for patient in self.test:
            self.y_test.extend(patient[2].cpu().numpy().tolist())

    def train_model(self, replace_model=True):
        "Function to fit the model to the train set"
        pass

    def validate_model(self):
        
        self.y_valid_pred = []
        self.y_valid_dist = []
        
        for data in self.valid: #For every graph in the data set
            out = self.clf(data) #Get the labels from all the nodes in one graph 

            if type(out) == tuple: out = out[0]

            labels = ((out > self.threshold).int()).cpu().detach().numpy()

            self.y_valid_dist.extend(out.cpu().detach().numpy().tolist())
            self.y_valid_pred.extend(labels.tolist())
        return metrics.f1_score(self.y_valid, self.y_valid_pred)

    def test_model(self):
        "Function that calculates labelling results on the test set"
        self.y_test_pred = []
        self.y_test_dist = []
        vr = []
        for data in self.test: #For every graph in the data set
            out = self.clf(data) #Get the labels from all the nodes in one graph (Each node gets 12 outputs: one for each class)
            labels = ((out > self.threshold).int()).cpu().detach().numpy()
            vr.extend(self.calculate_prediction_variance(data[0], data[1], out, data[2])) #Move this one to test_model
            self.y_test_dist.extend(out.cpu().detach().numpy().tolist())
            self.y_test_pred.extend(labels.tolist())
        return vr
        
    
    "Determines the threshold on the test set that corresponds to the given recall"
    #def calculate_threshold(self, recall=0.95):
    #    prec, rec, threshold =  metrics.precision_recall_curve(self.y_valid, self.y_valid_dist)
    #    for i, e in enumerate(rec):
    #        if e >= 0.95:
    #            return threshold[i]
    
    #Gives the probabilities and name of test case for visualizations
    def get_test_example(self, idx=0):
        lablen = [len(self.test[i][2]) for i in range(len(self.test))]
        offset = int(np.sum(lablen[:idx]))
        #if hasattr(self.clf, "clusmap"):
        #    self.clf(self.test[0])
        #    return self.y_test_dist[offset:offset+lablen[idx]], self.test[idx][3], self.clf.clusmap
        return np.array(self.y_test_dist[offset:offset+lablen[idx]]), self.test[idx][3]

    def calculate_test_metrics(self):
        #self.correct_test_labels()
        precision, recall, f, _ = metrics.precision_recall_fscore_support(self.y_test, self.y_test_pred, average=None, labels=self.labels)
        brier = metrics.brier_score_loss(self.y_test, self.y_test_dist)
        p,r, t = metrics.precision_recall_curve(self.y_test, self.y_test_dist)
        f1s = (2*(p * r)) / (p + r)
        #if self.bnry: #Binary, return positive F1 score
        return f[1], precision, recall, np.max(f1s)
        #return np.average(f), precision, recall

    def get_valid_preds(self):
        self.validate_model()
        return self.y_valid, self.y_valid_dist

    #Measures the distance in predictions in compariosn to its neighbours
    def calculate_prediction_variance(self, nodes, edges, probabilities, labels):
        from cluster_pool import calculate_components
        edges = edges.T
        comps = calculate_components(nodes.size(0), edges)

        sel_nodes = []

        #We only look at nodes that are in components containing any positive labels
        for c in comps:
            if torch.any(labels[c] == 1):
                sel_nodes.extend(c)        

        tot_var = []
        for node in sel_nodes:
            nb = torch.cat((edges[1][(edges[0] == node)],edges[0][(edges[1] == node)]))
            variance = torch.abs(probabilities[nb] - probabilities[node]).cpu().detach().numpy()

            tot_var.extend(list(variance))
        edges = edges.T
        return tot_var
    
    "(int) folds: Number of folds. If k-cross, then folds <= len(data)"
    "(Boolean) kCross: Use k-fold-cross-validation"
    "(Boolean) display: Print statistics"
    "(Boolean) record_roc: Save and return test probabilities and values for ROC curve"
    def run_folds(self, folds, kCross=True, display=True, record_roc=True, validation=True):
        "Function that runs train and test for new models"
        F1 = []
        p_class = []
        r_class = []
        valid_preds = []
        test_preds = []
        train_info = []
        t_holds = []
        prob_var = []
        test_examples = []
        best_f1s = []

        if not kCross:
            self.generate_train_validation(validation=True)
        elif self.cross_test_size <= 0: #Have to create a cross validation
            if folds > len(self.data):
                folds = len(self.data)
            folds = self.generate_k_fold(folds)

        if display:
            print("\nRunning " + str(folds) + " folds with " + str(self.clfName) + ":")
        
        for i in range(folds):
            if display:
                start_time = time.time()
                print("\tFold " + str(i+1) + "/" + str(folds) + "...", end='')

            if not kCross: #Generate new sets
                self.generate_train_validation(validation=validation)
            else: #Shift the sets one up
                tindex = self.cross_test_size * (i)
                tend = min(tindex + self.cross_test_size, len(self.data))
                self.valid = self.data[tindex:tend]
                self.train = self.data[:tindex] + self.data[tend:]
                if len(self.train) == 0:
                    self.train = self.data[tindex:tend]
                    self.valid = []
                self.format_data_values(validation=True)

            ti = self.train_model()
            train_info.append(ti)
            
            prob_var.extend(self.test_model())
            res, pClass, rClass, b = self.calculate_test_metrics()
            test_preds.append([self.y_test, self.y_test_dist])

            F1.append(res)
            p_class.append(pClass)
            r_class.append(rClass)
            best_f1s.append(b)

            for i in range(len(self.test)):
                value, name = self.get_test_example(idx=i)

                if i >= len(test_examples):                    
                    test_examples.append([value, name])
                else:
                    
                    test_examples[i][0] += value

            t_holds.append(self.threshold)
            if record_roc:
                valid_preds.append(self.get_valid_preds())           

            if display:
                elapsed_time = time.time() - start_time
                time_mes = "s"
                if elapsed_time > 100.0: #Over a hundred seconds
                    elapsed_time = elapsed_time / 60.0
                    time_mes = "m"
                
                print(f" ({elapsed_time:.3f} {time_mes}) [{res:.4f} Achieved F1, {b:.4f} Best F1]")


        print("\tTest set neighbour prediction Variance: ", np.mean(prob_var), " +/- ", np.std(prob_var))
        print("\Best possible Test set score: ", np.mean(best_f1s), " +/- ", np.std(best_f1s))

        for i in range(len(self.test)):
            test_examples[i][0] = test_examples[i][0] / folds
            test_examples[i].append(np.average(t_holds))

        return F1, p_class, r_class, valid_preds, folds, train_info, test_preds, t_holds, prob_var, test_examples
