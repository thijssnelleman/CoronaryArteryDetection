import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

from dataClasses import dataLoader
from rf import RandomForest
from XGB import XGBoost
from nn import NNModel

import gcn
from gcn import GCNModel

parser = argparse.ArgumentParser(description='Input Parser for Model Handling')

parser.add_argument("--mdls", dest='models', type=str, nargs='+', required=False, help='Select one or multiple possible models.')
parser.add_argument("-f", dest='folds', type=int, required=False, help='Sets number of folds for each model.')
parser.add_argument("--plts", dest='plots', type=str, nargs='+', required=False, help="Select which plots should be produced after the run.")
parser.add_argument("--fimp", dest='fimp', action='store_true', required=False, help="Whether feature importances are calculated if supported")
parser.set_defaults(fimp=False)

def parse_args(args):
    if args.models is None:
        args.models = ["rf", "xgb", "nn", "gcn"]

    if "rf" in args.models:
        RF = RandomForest(dataTorch, list(range(len(dLoader.labels), dLoader.test_set_idx)))
        models.append(RF)
        modelNames.append(RF.clfName)

    if "xgb" in args.models:
        XGB = XGBoost(dataTorch, list(range(len(dLoader.labels))), dLoader.test_set_idx)
        models.append(XGB)
        modelNames.append(XGB.clfName)

    if "nn" in args.models:
        NN = NNModel(dataTorch, list(range(len(dLoader.labels))), dLoader.test_set_idx)
        models.append(NN)
        modelNames.append(NN.clfName)

    if "gcn" in args.models:
        GCN = GCNModel(dataTorch, list(range(len(dLoader.labels))), dLoader.test_set_idx)
        models.append(GCN)
        modelNames.append(GCN.clfName)

    if "gcn-pool" in args.models:
        GCNp = GCNModel(dataTorch, list(range(len(dLoader.labels))), dLoader.test_set_idx, type=gcn.GraphConvPoolNN)
        models.append(GCNp)
        modelNames.append(GCNp.clfName)

    #if "gat-pool" in args.models:
    #    GATp = GCNModel(dataTorch, list(range(len(dLoader.labels))), dLoader.test_set_idx, type=gcn.GraphGATPoolNN)
    #    models.append(GATp)
    #    modelNames.append(GATp.clfName)

    if "gcn-pool-new" in args.models:
        GCNpn = GCNModel(dataTorch, list(range(len(dLoader.labels))), dLoader.test_set_idx, type=gcn.GraphConvNewPoolNN)
        models.append(GCNpn)
        modelNames.append(GCNpn.clfName)

    #if "gcn-pool-DAG" in args.models:
    #    GCNDAG = GCNModel(dataTorch, list(range(len(dLoader.labels))), dLoader.test_set_idx, type=gcn.GraphConvDAGPoolNN)
    #    models.append(GCNDAG)
    #    modelNames.append(GCNDAG.clfName)

    if "unet" in args.models:
        UNET = GCNModel(dataTorch, list(range(len(dLoader.labels))), dLoader.test_set_idx, type=gcn.GUNET)
        models.append(UNET)
        modelNames.append(UNET.clfName)

    if args.folds is not None:
        folds = args.folds
    else:
        folds = 2 #default

    if args.plots is not None:
        plots = args.plots

    return folds, args.fimp


"Responsible for calling models and reporting/visualizing their results"
sns.set_theme(style="ticks", color_codes=True)

"""Gets data in format [model][folds](test_set, predictions)"""
"""def visualize_ROC(roc_data):
    #print(roc_data)
    #calculate scores for 'no skill'
    ns_probs = [0 for _ in range(len(roc_data[0][0][0]))]
    mean_fpr = np.linspace(0, 1, len(ns_probs))
    #ns_auc = roc_auc_score(testy, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(roc_data[0][0][0], ns_probs)

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

    ax = plt.gca()

    #For each model
    for i in range(len(roc_data)):
        m_data = roc_data[i]
        lr_auc = []
        #lr_fpr = []
        lr_tpr = []

        for j in range(len(m_data)): #For each fold
            lr_auc.append(roc_auc_score(m_data[j][0], m_data[j][1]))

            fpr, tpr, _ = roc_curve(m_data[j][0], m_data[j][1])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            lr_tpr.append(interp_tpr)

        lr_auc_mean = np.mean(lr_auc)
        lr_auc_std = np.std(lr_auc)
        name = modelNames[i] + r" (AUC = %0.2f $\pm$ %0.2f)" % (lr_auc_mean, lr_auc_std)

        lr_tpr_mean = np.mean(lr_tpr, axis=0)
        lr_tpr_std = np.std(lr_tpr, axis=0)

        color=next(ax._get_lines.prop_cycler)['color']
        
        plt.plot(mean_fpr, lr_tpr_mean, marker='.', label=name, color=color)
        plt.plot(mean_fpr[int(len(mean_fpr)/2)], lr_tpr_mean[int(len(mean_fpr)/2)], marker='+', color='black')
        tprs_upper = np.minimum(lr_tpr_mean + lr_tpr_std, 1)
        tprs_lower = np.maximum(lr_tpr_mean - lr_tpr_std, 0)

        area = {
            'x': mean_fpr,
            'y1': tprs_upper,
            'y2': tprs_lower
        }
        plt.fill_between(**area, alpha=0.2, color=color)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')    
    plt.legend()
    plt.show()"""


"""Gets data in format [model][folds](test_set, predictions)"""
"""Same data as for ROC, but instead visualize precision vs recall for positive class"""
def visualize_prec_rec(pr_rec_data, avg_f1):
    steps = np.linspace(0, 1, len(pr_rec_data[0][0][0]))
    ax = plt.gca()
    for index, model in enumerate(pr_rec_data):
        prcs = []
        pr_auc = []
        
        #thresh_f1s = [] 
        for i, fold in enumerate(model):
            prec, rec, _ =  precision_recall_curve(fold[0], fold[1])
            
            pr_auc.append(auc(rec, prec))
            interp_tpr = np.interp(steps, prec, rec)
            prcs.append(interp_tpr) 

        #for e in pr_thresholds: print(len(e))

        auc_mean = np.mean(pr_auc)
        auc_std = np.std(pr_auc)
        name = modelNames[index] + r" (AUC = %0.2f $\pm$ %0.2f)" % (auc_mean, auc_std)

        pc_mean = np.mean(prcs, axis=0)
        pc_std = np.std(prcs, axis=0)
        f1s = (2*(pc_mean * steps)) / (pc_mean + steps)

        #Plot the PR curve
        color=next(ax._get_lines.prop_cycler)['color']
        plt.plot(steps, pc_mean, marker='.', label=name, color=color)
        
        #Show the best possible average F1 scores over the folds
        #best = np.argmax(np.flip(f1s)) * -1
        #plt.plot(steps[best], pc_mean[best], marker="x", color="green")
        #ax.annotate("F1: " + str(round(f1s[best],4)), (steps[best], pc_mean[best]), fontsize=8, weight='bold')
        #plt.vlines(steps[best], 0, pc_mean[best], linestyle="dashed", color="green")
        #plt.hlines(pc_mean[best], 0, steps[best], linestyle="dashed", color="green")

        #Show the average F1 score we achieved
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        achieved = find_nearest(f1s, avg_f1)
        plt.plot(steps[achieved], pc_mean[achieved], marker="x", color="green")
       # ax.annotate("F1: " + str(round(f1s[achieved],4)), (steps[achieved], pc_mean[achieved]), fontsize=8, weight='bold')
        plt.vlines(steps[achieved], 0, pc_mean[achieved], linestyle="dashed", color="green", label="Achieved F1: "+ str(round(f1s[achieved],4)))
        plt.hlines(pc_mean[achieved], 0, steps[achieved], linestyle="dashed", color="green")

        #Show the F1 scores at 95% recall
        rec_id = find_nearest(steps, 0.95)
        rec_val = rec_id / len(steps)
        prec_val = pc_mean[rec_id]
        l_f1 = round(f1s[rec_id],4)
        #l_f1 = round((2 * (prec_val * rec_val)) / (prec_val + rec_val), 4)
        
        plt.plot(rec_val, prec_val, marker="x", color="black")
        #ax.annotate("F1: " + str(l_f1), (rec_val, prec_val), fontsize=8, weight='bold')
        plt.vlines(rec_val, 0, prec_val, linestyle="dashed", color="black", label="95% Recall F1: " + str(l_f1))
        plt.hlines(prec_val, 0, rec_val, linestyle="dashed", color="black")
        
        tprs_upper = np.minimum(pc_mean + pc_std, 1)
        tprs_lower = np.maximum(pc_mean - pc_std, 0)

        area = {
            'x': steps,
            'y1': tprs_upper,
            'y2': tprs_lower
        }
        plt.fill_between(**area, alpha=0.2, color=color)

    plt.xlabel('Recall Positive Label')
    plt.ylabel('Precision Positive Label')  
    plt.legend()
    plt.show()

#Compares models over their list of scores
#Takes in a two dimensional list of [Model][Folds]
def visualize_models(data):
    scores = np.array(data).T
    ymin = np.max([0.0, np.min(scores) - 0.1])
    ymax = np.min([1.0, np.max(scores) + 0.1])

    df = pd.DataFrame(scores, columns=modelNames)
    ax = sns.boxplot(data=df)#, palette="Set3")
    ax.set(xlabel="Models", ylabel = "Avg. F1-Score")
    ax.set(ylim=(ymin, ymax))
    ax.set_title("Average F1-scores over " + str(len(modelNames)) + " models with " + str(folds) + " folds")
    plt.show()

#Compares class quality over models
#Takes in a three dimensional list of [Model][Classes][Folds]
"""def visualize_classes(data, mthd):
    scores = np.array(data) #To np
    df = pd.DataFrame(scores.T.reshape(folds, -1), columns=pd.MultiIndex.from_product([modelNames, classes])) #Create three dimensional data frame
    df = df.melt(var_name=["Model", "Class"]) #Melt the three dimensions into features
    ax = sns.boxplot(x="Class", y="value", hue="Model", data=df) #Plot grouped by Classes over the models on the values
    ax.set(xlabel="Classes", ylabel = mthd + " Distribution")
    ax.set(ylim=(0.0, 1.0))
    ax.set_title("Class " + mthd + " over " + str(len(modelNames)) + " models with " + str(folds) + " folds")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)
    plt.show()"""


"""Takes in training data as a list of tuples:"""
"""data[model][(train_f1, loss, validation_f1)]"""
def visualize_training_data(data):
    
    for i,e in enumerate(data): #For each model
        if type(e) is None or None in e:
            continue
        ax = plt.gca()
        train = []
        valid = []
        for t,l,v in e: #Extract values from fold: Training, Loss, Validation scores
            train.append(t)
            valid.append(v)

        color=next(ax._get_lines.prop_cycler)['color']
        train = np.array(train)
        t = np.average(train, axis=0)
        tstd = np.std(train, axis=0)
        plt.plot(range(len(t)), t, '-', color=color, label="Training")
        plt.plot(np.argmax(t), np.max(t), marker="x", color="black")
        plt.fill_between(range(len(t)), t-tstd, t+tstd ,alpha=0.3, facecolor=color)

        color=next(ax._get_lines.prop_cycler)['color']
        valid = np.array(valid)
        v = np.average(valid, axis=0)
        vstd = np.std(valid, axis=0)
        plt.plot([(x+1)*10 for x in range(len(v))], v, "--", color=color, label="Validation")
        plt.fill_between([(x+1)*10 for x in range(len(v))], v-vstd, v+vstd ,alpha=0.3, facecolor=color)
        plt.plot((np.argmax(v)+1)*10, np.max(v), marker="x", color="black")

        plt.xlabel("Epochs")
        plt.ylabel("F1 score")
        plt.ylim(0.0,1.0)
        plt.legend(loc="lower right")
        plt.title(modelNames[i] + " Training Data")
        plt.show()


def plot_prob_var(mdl_idx, var):
    sns.boxplot(y=var)
    plt.ylabel("Neighbour Probability Variance")
    
    plt.title(modelNames[mdl_idx])
    plt.show()

bnry = True

dLoader = dataLoader(processed=False, asBinary=bnry)
#data = dLoader.get_node_list()
dataTorch = dLoader.get_torch_data()

if bnry:
    classes = dLoader.labels
else:
    classes = [x.replace(" ", "\n") for x in dLoader.labels] #Format labels

models = []
modelNames = []

#plots = ["train", "avg", "prc", "xml"]
plots = ["train", "avg", "prc", "xml", "var"]

folds, f_imp = parse_args(parser.parse_args())

modelF1 = []
#modelClassResults = []
#f_importances = []
#roc_data = []
test_data = []
vis_data = [[] for _ in range(len(modelNames))]
#r_roc = True
tdata = []
#model_thresholds = []
mod_var = []

if len(models) == 0:
    print("Error: No models selected. Exiting.")
    exit(-1)

for mod_idx, curMod in enumerate(models):
    f1, p_class, r_class, r, folds, train_inf, test_dists, threshes, variance, test_examples = curMod.run_folds(folds, kCross=True)
    modelF1.append(f1)
    #modelClassResults.append(r_class)
    #f_importances.append(feature_imp)
    tdata.append(train_inf)
    mod_var.append(variance)
    #roc_data.append(r)
    test_data.append(test_dists)
    if bnry:
        print("\nF1 Positive Class " + modelNames[mod_idx] + ":", round(np.average(f1),4), "+/-", round(np.std(f1), 4))
    else:
        print("\nF1 Average " + modelNames[mod_idx] + ":", round(np.average(f1),4))

    print("Precision and Recall Average per class:\n",)
    p_class = np.array(p_class).T
    r_class = np.array(r_class).T
    
    for i,f in enumerate(zip(np.average(p_class, axis=1), np.average(r_class, axis=1))):
        print("\t+ " + dLoader.labels[i] + ":\t Precision " + str(round(f[0], 4)) + ", Recall " + str(round(f[1], 4)))

    #t = []
    #for fold in r:
    #    prec, rec, threshold =  precision_recall_curve(fold[0], fold[1])
    #    f1s = (2*(prec * rec)) / (prec + rec)
    #    t.append(threshold[np.argmax(f1s)])

    #print(threshes)
    #thresh = np.average(threshes) #Get the average threshold over the folds

    #if "xml" in plots:
        
        #if hasattr(curMod[0].clf, "clusmap"):
        #    print(len(curMod[0].clf.clusmap))
        #    vis_data.append([curMod[0].get_test_example(), curMod[1], thresh])
        #    print(len(curMod[0].clf.clusmap))
        #    vis_data[-1].append(curMod[0].clf.clusmap)
        #else:
    vis_data[mod_idx] = test_examples
    #for PID in range(len(curMod.test)): #Run every single one in the test set
    #    vis_data[mod_idx].append([curMod.get_test_example(idx=PID), modelNames[mod_idx], np.average(threshes)])
    
if "train" in plots:
    visualize_training_data(tdata)

if "avg" in plots:
    visualize_models(modelF1)

#if "class" in plots:
#    visualize_classes(modelClassResults, "Recall")

#if "roc" in plots:
#    visualize_ROC(roc_data)

if "var" in plots:
    for i in range(len(modelNames)):
        plot_prob_var(i, mod_var[i])

if "prc" in plots:
    visualize_prec_rec(test_data, np.average(modelF1, axis=1))

if "xml" in plots:   
    for i,models in enumerate(vis_data):
        for probs, pname, t in models:
            
            #if hasattr(models[i].clf, "clusmap"): #Test for new edge pool
            #    #print(models[i].clf.clusmap, ex[0][1], ex[1])
            #    dLoader.create_cluster_vis(ex[0][2], ex[0][1], ex[1])
            #comp.append([x > ex[2] for x in ex[0][0]])
            dLoader.create_vis(probs, pname, modelNames[i], t)
            dLoader.create_vis(probs, pname, modelNames[i], t, as_probs=True)


os.chdir(f"H:\\Data\\GraphFiles\\results")
time_stamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
result = {}
save_name = ""
for i,e in enumerate(modelNames):
    result[e] = {"F1": modelF1[i], "test_data": test_data[i], "train_data": tdata[i], "variance": mod_var[i]}
    save_name += e + "-"

save_name += time_stamp

import pickle
a_file = open(save_name+".pkl", "wb")
pickle.dump(result, a_file)
a_file.close()

#Save the results as .npy

#with open(save_name, 'rb') as f:
#    loaded_dict = pickle.load(f)