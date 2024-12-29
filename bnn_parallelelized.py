import os
import time
import numpy as np
import math
import warnings
import shutil
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, Normalizer, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    multilabel_confusion_matrix, 
    RocCurveDisplay, 
    ConfusionMatrixDisplay, 
    mean_squared_error, 
    r2_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern
from sklearn import preprocessing, metrics

from scipy import linalg, stats
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule

from Bayesneuralnet_regcls import MCMC, Network, scikit_linear_mod
from convergence_test import gelman_rubin

import multiprocess as mp
from multiprocessing import cpu_count


def make_results_folder():
    # Define the base folder name
    base_folder = 'result'
    
    # Initialize the folder number
    folder_number = 1
    
    # Generate the folder name
    folder_name = f"{base_folder}{folder_number}"
    
    # Check if the folder exists
    while os.path.exists(folder_name):
        # If it exists, increment the folder number and generate the new folder name
        folder_number += 1
        folder_name = f"{base_folder}{folder_number}"
    
    # Create the new folder
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
    return folder_name


def resume_training(maxruns):

    
    #res = [int(i[6:8]) if not(i[6:8] == '') else 0 for i in os.listdir() if "result" in i]
    res = [int(i[6:8]) if i[6:8].isdigit() else 0 for i in os.listdir() if "result" in i]
    folder_name = f"result{max(res)}"
    
    if(os.path.exists(folder_name)):
        #print(folder_name)
        ''' 
        checkpoint = f"{folder_name}/checkpoint.txt"
        if (os.path.exists(checkpoint)):
            #print(checkpoint)
            
            problem, run = np.loadtxt(checkpoint)
            problem_num = int(problem)
            run_num = int(run)
            print()
            print()
            print(problem_num,run_num)
            print()
            print()
            if run_num < maxruns-1:
                
                if problem_num >= len(os.listdir("data/")):
                    print('Resuming Training for all problems at run number ', run_num )
                    problem_num = 1
                    
                else:
                    print('Resuming Training at problem ', problem_num, " run number ", run_num )
                    
            else:
                if problem_num >= 4 : #len(os.listdir("data/")):
                    print("Starting Training at problem 1, run 0 in new results folder")
                    problem_num = 1
                    run_num = 0
                    folder_name = make_results_folder()
                    
                else:
                    run_num = 0
                    problem_num += 1
                    print('Resuming Training at next ', problem_num, " run number ", run_num )
                    
            
            ###################################################################################    
            
            return problem_num, run_num, folder_name
            
        else:
            return 1,0, folder_name
        '''
        return 1,0, make_results_folder()
    else:
        return 1,0, make_results_folder()
        
def get_dataset_details(problem):

  
    name = ""
    
    
    
    if problem == 1:
        # problem 1
        data = np.genfromtxt('data/Lazer/Lazer_processed.csv',delimiter=',')
        #traindata = np.loadtxt("data/Lazer/train.txt")
        #testdata = np.loadtxt("data/Lazer/test.txt")  #
        name = "Lazer"
        hidden = 5
        input = 4  
        output = 1
        prob_type = 'regression'
        numSamples = 50000 
        #numSamples = 500
        train_ratio = 0.6
                
        np.random.shuffle(data)
        scaler = MinMaxScaler()
        scaler.fit(data)
        traindata = scaler.transform(data)[: int(train_ratio * data.shape[0])]
        testdata = scaler.transform(data)[int(train_ratio * data.shape[0]): ]
        
    elif problem == 2:
        # problem 2
        data = np.genfromtxt('data/Sunspot/Sunspots.csv',delimiter=',')
        #traindata = np.loadtxt("data/Sunspot/train.txt")
        #testdata = np.loadtxt("data/Sunspot/test.txt")  #
        name = "Sunspot"
        hidden = 5
        input = 4  #
        output = 1
        prob_type = 'regression' 
        numSamples = 50000 
        
        train_ratio = 0.6
                
        np.random.shuffle(data)
        scaler = MinMaxScaler()
        scaler.fit(data)
        traindata = scaler.transform(data)[: int(train_ratio * data.shape[0])]
        testdata = scaler.transform(data)[int(train_ratio * data.shape[0]): ]

    
    
    
    elif problem == 3:
        # problem 3
        data  = np.genfromtxt('data/iris.csv',delimiter=';')
        classes = data[:,4].reshape(data.shape[0],1)-1
        features = data[:,0:4]#Normalizing Data
    
        name = "Iris"
        hidden = 12
        input = 4 #input
        output = 3
    
        for k in range(input):
            mean = np.mean(features[:,k])
            dev = np.std(features[:,k])
            features[:,k] = (features[:,k]-mean)/dev
            train_ratio = 0.6 #choose
            indices = np.random.permutation(features.shape[0])
            traindata = np.hstack([features[indices[:int(train_ratio*features.shape[0])],:],classes[indices[:int(train_ratio*features.shape[0])],:]])
            testdata = np.hstack([features[indices[int(train_ratio*features.shape[0])]:,:],classes[indices[int(train_ratio*features.shape[0])]:,:]])
            prob_type = 'classification'
        numSamples = 50000 
        
    elif problem == 4:
        # problem 4
        data = np.genfromtxt('data/ionesphere/ionosphere.csv',delimiter=',')
        #print(data)
        
        #traindata = np.genfromtxt('data/ionesphere/ftrain.csv',delimiter=',')[:,:-1]
        #testdata = np.genfromtxt('data/ionesphere/ftest.csv',delimiter=',')[:,:-1]
        
        name = "Ionosphere"
        hidden = 50
        input = 34 #input
        output = 2
        prob_type = 'classification'
        numSamples = 50000

        train_ratio = 0.6
        
        np.random.shuffle(data)
        scaler = MinMaxScaler()
        scaler.fit(data)
        traindata = scaler.transform(data)[: int(train_ratio * data.shape[0])]
        testdata = scaler.transform(data)[int(train_ratio * data.shape[0]): ]

    elif problem == 5:
        # abalone
        data = np.genfromtxt('data/abalone/abalone.csv',delimiter=',')
        name = "abalone"
        hidden = 12
        input = 8 #input
        output = 1
        train_ratio = 0.6

        
        
        np.random.shuffle(data)
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        traindata = data[: int(train_ratio * data.shape[0])]
        testdata = data[int(train_ratio * data.shape[0]): ]

        
        prob_type = 'regression' 
        numSamples = 50000
        #numSamples = 200

    elif problem == 6:
        # abalone classification
        data = np.genfromtxt('data/abalone_classification/abalone_classification.csv',delimiter=',')
        name = "abalone_classification"
        hidden = 12
        input = 8 #input
        output = 4
        train_ratio = 0.6

        
        
        np.random.shuffle(data)

        traindata = data[: int(train_ratio * data.shape[0])]
        testdata = data[int(train_ratio * data.shape[0]): ]

        
        prob_type = 'classification' 
        numSamples = 50000
        #numSamples = 200

    elif problem == 7:
        # exp_325
        
        name = "exp_325"
        hidden = 8
        input = 3 #input
        output = 6
        
        
        prob_type = 'classification' 
        numSamples = 50000

        x_train, x_test, y_train, y_test = read_preprocess_data("data/exp_325/Unbalanced_325.csv")
        #numSamples = 200
        traindata = np.concatenate((x_train, y_train[:, np.newaxis]), axis=1)
        testdata = np.concatenate((x_test, y_test[:, np.newaxis]), axis=1)

    elif problem == 8:
        # exp_310
        
        name = "exp_310"
        hidden = 8
        input = 3 #input
        output = 6
        
        
        prob_type = 'classification' 
        numSamples = 50000

        x_train, x_test, y_train, y_test = read_preprocess_data("data/exp_310/Unbalanced_310.csv")
        #numSamples = 200
        traindata = np.concatenate((x_train, y_train[:, np.newaxis]), axis=1)
        testdata = np.concatenate((x_test, y_test[:, np.newaxis]), axis=1)

    return traindata,testdata,numSamples, prob_type, [input,hidden,output], name



def read_preprocess_data(filename):
    
    #df = pd.read_csv("data/exp_325/Unbalanced_325.csv")
    df = pd.read_csv(filename)

    df_filtered = df[["Bulk","Porosity", "Resistivity"]]
    print(df_filtered.describe())

    

    normalize = True

    if normalize:


        #Use minmax normalization to get data in 0-1 rnage for lstm model
        x = df_filtered.values #returns a numpy array of the data frame
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        
        
        # Splitting data in traning and testting set
        all_data = pd.DataFrame(x_scaled).values


    else:
        all_data = df_filtered.values
        


    encoded_labels = df[["Species"]]
    
    le = preprocessing.LabelEncoder()
    
    encoded_labels =  le.fit_transform(encoded_labels)
    labels = list(le.classes_)


    balance_data = False
    if balance_data:
        resample = NeighbourhoodCleaningRule()
        all_data, encoded_labels = resample.fit_resample(all_data, encoded_labels)

    
    desc_data = False
    if desc_data:

        counter = Counter(encoded_labels)

        for k,v in counter.items():
            per = v / len(encoded_labels) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

        #plt.bar(counter.keys(), counter.values())
        plt.bar(labels, counter.values())
        plt.xticks(rotation = 30)
        
        #encoded_labels.value_counts().plot(kind='bar')
        #plt.show()

    
    x_train, x_test, y_train, y_test = train_test_split(all_data, encoded_labels, test_size=0.4, shuffle=True)

    return x_train, x_test, y_train, y_test


# Define the function to handle each experimental run for a given problem
def run_experiment(args):
    problem, run_num, w_limit, tau_limit, save, results_folder,traindata,testdata,numSamples, prob_type, topology,name = args


    use_langevin_gradients = True
    l_prob = 0.5
    learn_rate = 0.01

    timer = time.time()

    print(f'Started sampling for problem {problem} run {run_num} at time {timer}')
    # Placeholder for MCMC class
    mcmc = MCMC(use_langevin_gradients, l_prob, learn_rate, numSamples, traindata, testdata, topology, prob_type)

    [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, p_train, p_test, accept_ratio] = mcmc.sampler(w_limit, tau_limit)

    print(f'Successfully sampled for problem {problem} run {run_num}')

    burnin = 0.5 * numSamples
    timer2 = time.time()
    timetotal = (timer2 - timer) / 60
    print(f"{timetotal} min taken")

    pos_w = pos_w[int(burnin):, ]
    pos_tau = pos_tau[int(burnin):, ]

    fx_mu = fx_test[int(burnin):, ].mean(axis=0)
    fx_high = np.percentile(fx_test[int(burnin):, ], 95, axis=0)
    fx_low = np.percentile(fx_test[int(burnin):, ], 5, axis=0)

    fx_mu_tr = fx_train[int(burnin):, ].mean(axis=0)
    fx_high_tr = np.percentile(fx_train[int(burnin):, ], 95, axis=0)
    fx_low_tr = np.percentile(fx_train[int(burnin):, ], 5, axis=0)

    pos_w_mean = pos_w.mean(axis=0)

    p_tr = np.mean(p_train[int(burnin):])
    ptr_std = np.std(p_train[int(burnin):])
    p_tes = np.mean(p_test[int(burnin):])
    ptest_std = np.std(p_test[int(burnin):])
    print(p_tr, ptr_std, p_tes, ptest_std)

    if save:
        np.savetxt(f'{results_folder}/pos_w_{run_num}.txt', pos_w)
        with open(f'{results_folder}/pos_w_{run_num}_result.txt', "a+") as outres_db:
            outres_db.write(f"{use_langevin_gradients}, {learn_rate}, {p_tr}, {ptr_std}, {p_tes}, {ptest_std}, {accept_ratio}, {timetotal}\n")

        np.savetxt(f'{results_folder}/checkpoint.txt', np.asarray([problem, run_num]).astype(int))

    ytestdata = testdata[:, topology[0]]
    ytraindata = traindata[:, topology[0]]

    if prob_type == 'regression':
        print('---RMSE train---')
        print('mean', mcmc.rmse(ytraindata, fx_mu_tr))
        print('high', mcmc.rmse(ytraindata, fx_high_tr))
        print('low', mcmc.rmse(ytraindata, fx_low_tr))
        print('---RMSE test---')
        print('mean', mcmc.rmse(ytestdata, fx_mu))
        print('high', mcmc.rmse(ytestdata, fx_high))
        print('low', mcmc.rmse(ytestdata, fx_low))
    else:
        print('---accuracy train---')
        print('mean', mcmc.accuracy(ytraindata, fx_mu_tr))
        print('high', mcmc.accuracy(ytraindata, fx_high_tr))
        print('low', mcmc.accuracy(ytraindata, fx_low_tr))
        print('---accuracy test---')
        print('mean', mcmc.accuracy(ytestdata, fx_mu))
        print('high', mcmc.accuracy(ytestdata, fx_high))
        print('low', mcmc.accuracy(ytestdata, fx_low))




# Main function to run experiments in parallel for each run of the model


def main():
    w_limit = 0.025  # step size for w
    tau_limit = 0.2  # step size for eta
    save = True
    maxruns = 30

    #problem_num, run_num, results_folder = resume_training(maxruns)
    
    results_folder = "result_beast"
    
    for problem in range(5, 6):  # Adjust according to the problem range you want to test
        # Prepare arguments for each run to be parallelized
        traindata,testdata,numSamples, prob_type, topology,name = get_dataset_details(problem)
        #numSamples=50
        # Create results directory if it doesn't exist
        if not os.path.exists(results_folder + '/' + name):
            os.makedirs(results_folder + '/' + name)

        args_list = [(problem, run, w_limit, tau_limit, save, results_folder + '/' + name , traindata,testdata,numSamples, prob_type, topology,name)
                     for run in range(0, maxruns)]

        

        print(f"Running problem {problem} with {len(args_list)} runs in parallel")
        
        print(len(args_list))
        
        # Create a pool with as many workers as there are CPU cores
        pool_size = min(len(args_list), cpu_count()- 4)
        with mp.Pool(pool_size) as pool:
            # Apply parallel execution of runs
            
            pool.map(run_experiment, args_list)
    
if __name__ == "__main__":
    main()