import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ThesisModel import ThesisModelInterface

class RandomForest(ThesisModelInterface):
    def __init__(self, data, labels, test_set_idx, n_estimators=250, max_depth=10):
        super().__init__(data, labels, test_set_idx)
        self.clfName = "Random Forest Classifier"
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def format_data_values(self, validation):

        def extract_nodes(pdata):
            dset = None
            labels = None
            for patient in pdata:
                #print(patient)
                nodes, edges, classes, name = patient
                if dset is None:
                    dset = nodes.detach().numpy()
                    labels = classes.detach().numpy()
                else:
                    np.append(dset, nodes.detach().numpy(), axis=0)
                    np.append(labels, classes.detach().numpy(), axis=0)
                
            return np.array(dset), np.array(labels)

        #train_set = extract_nodes(self.train)
        #if validation: validation_set = extract_nodes(self.valid)
        #test_set = extract_nodes(self.test)


        self.x_train, self.y_train = extract_nodes(self.train)
        if validation: self.x_valid, self.y_valid = extract_nodes(self.valid)
        self.x_test, self.y_test = extract_nodes(self.test)


        #np.random.shuffle(train_set) #Shuffle nodes
        #if validation: np.random.shuffle(validation_set)
        #np.random.shuffle(test_set) #Shuffle nodes

        #self.x_train = np.delete(train_set, np.s_[-1:], axis=1) #Remove labels
        #self.x_train = np.delete(self.x_train, np.s_[:2], axis=1) #Remove Patient ID and Node ID

        #if validation:
        #    self.x_validation = np.delete(validation_set, np.s_[-1:], axis=1) #Remove labels
            #self.x_validation = np.delete(self.x_validation, np.s_[:2], axis=1) #Remove Patient ID and Node ID

        #self.x_test = np.delete(test_set, np.s_[-1:], axis=1) #Remove labels
        #self.x_test = np.delete(self.x_test, np.s_[:2], axis=1) #Remove Patient ID and Node ID

        #self.y_train = np.delete(train_set, np.s_[:len(train_set[0])-1], axis=1).astype('int32').flatten() #Extract labels
        #if validation: self.y_validation = np.delete(validation_set, np.s_[:len(validation_set[0])-1], axis=1).astype('int32').flatten() #Extract labels
        #self.y_test = np.delete(test_set, np.s_[:len(train_set[0])-1], axis=1).astype('int32').flatten() #Extract labels


    def train_model(self, replace_model=True):
        "Function to fit the model to the data"
        if self.clf is None or replace_model is True:
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, class_weight={0:1, 1:50})
        
        self.clf.fit(self.x_train, self.y_train)

    def test_model(self):
        "Function that calculates test set classifications"
        self.y_test_pred = self.clf.predict(self.x_test)
        self.y_test_dist = self.clf.predict_proba(self.x_test)
        if self.bnry:
            res = []
            for i, e in enumerate(self.y_test_dist):
                res.append(e[1])
            self.y_test_dist = res

    def calculate_feature_importances(self):
        return self.clf.feature_importances_
