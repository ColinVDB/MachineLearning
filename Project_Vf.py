# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 19:33:42 2020

@author: Colin Vanden Bulcke
         Hadrien Cools
"""


import pandas  as pd  
import numpy as np 
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =============================================================================
# 1. Loading of the data                                                         
# =============================================================================

X1 = pd.read_csv("X1.csv")
Y1 = pd.read_csv("Y1.csv",header=None, names=['shares'])
X2 = pd.read_csv("X2.csv")

X1 = X1.values
Y1 = Y1.values
X2 = X2.values

# Booleans
figure = True # boolean to show the figures or not
reg_test = True # boolean to make the regression tests
cl_test = False # boolean to make the classification tests
imp_test = False # boolean to make the improvement tests

# =============================================================================
# 2. Preprocessing 
# =============================================================================
# normalisation
scaler = preprocessing.StandardScaler()
X1_normalised = scaler.fit_transform(X1)

# nb FOlds
k=10
random_state = 12883823
kf = KFold(n_splits=10, random_state=random_state,shuffle=True)
#print("Param kfold:",kf)
#print("Nb split:", kf.get_n_splits(X1_normalised))

trial_matrix_data = np.zeros((17840, 58))
trial_matrix_target = np.zeros((17840, 1))
for train_index, test_index in kf.split(X1_normalised):
#    print("train_index ",train_index , "test_index",test_index)
    training_data = X1_normalised[train_index]
    training_target = Y1[train_index]
    validation_data = X1_normalised[test_index]
    validation_target = Y1[test_index]
#    print(np.shape(training_data))
#    print(np.shape(training_target))
#    print(np.shape(training_data))
#    print(np.shape(validation_target))
    trial_matrix_data = training_data
    trial_matrix_target =training_target

# Covariance matrix
#print(np.shape(trial_matrix_data))
#print(np.shape(trial_matrix_target))
covX1 = np.cov(trial_matrix_data.T)



# Feature selection 
## Apply PCA
pca = PCA(.95)
X1_normalised_copy = np.copy(X1_normalised)
pca.fit(X1_normalised_copy)
#print(pca.explained_variance_ratio_)

PCAed_X1 = pca.transform(X1_normalised_copy)

## Draw cor matrix pca datas

covX1_PCAed = np.cov(PCAed_X1.T)

## remove diagonal  = remove autocovariance
if figure:
    plt.figure()
    x,y = np.shape(covX1_PCAed)
    for i in range(x):
        covX1_PCAed[i,i] = 0
    sn.heatmap(covX1_PCAed, fmt='g')
    plt.title("Covariance matrix  selected features")
    plt.show()

# Draw cor matrix raw datas

# Add output to matrix, last column
gg = np.copy(Y1[:,0])
dd = np.copy(X1)
cc = np.vstack((dd.T,gg))
print(np.shape(X1))

min_max_scaler = preprocessing.MinMaxScaler()
covX1_test_normalised = min_max_scaler.fit_transform(cc.T)
covX1_test = np.cov(covX1_test_normalised.T)

if figure:
    plt.figure()
    #print("drawing")
    #print(np.shape(covX1_test))
    x,y = np.shape(covX1_test)
    for i in range(x):
        covX1_test[i,i] = 0
    sn.heatmap(covX1_test, fmt='g')
    plt.title("Covariance matrix datas")
    plt.show()


# =============================================================================
# 3. Models
# =============================================================================

# 1. Metric
from sklearn.metrics import f1_score

def score_f1(y_true, y_pred, th):
    return f1_score(y_true > th, y_pred > th)

def score_regression(y_true, y_pred):
    scores = [score_f1(y_true,y_pred,th) for th in [500,1400,5000,10000]]
    return np.mean(scores)

k = 4 # Kfold

# 2. Regression
if reg_test:
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
      
    # 2.1. dataset generation
    dataset = PCAed_X1
    dataset_target = Y1
    
    # 2.2. Model Evaluation
    kfold = KFold(k)
    count = 0
    perf_LR = 0
    perf_Lasso = 0
    perf_KNN = 0
    perf_MLP = 0
    perf_DT = 0
    perf_SVM = 0
    for trn_idx, tst_idx in kfold.split(dataset):
        training_data = dataset[trn_idx]
        training_target = dataset_target[trn_idx]
        validation_data = dataset[tst_idx]
        validation_target = dataset_target[tst_idx]
        # Linear Regression
        model_LR = LinearRegression()
        model_LR.fit(training_data,training_target)
        score_LR = score_regression(validation_target, model_LR.predict(validation_data))
        perf_LR = perf_LR + score_LR
        # Lasso
        model_Lasso = Lasso()
        model_Lasso.fit(training_data,training_target)
        score_Lasso = score_regression(validation_target, model_Lasso.predict(validation_data))
        perf_Lasso = perf_Lasso + score_Lasso
        # KNN
        model_KNN = KNeighborsRegressor(n_neighbors=7,weights='distance')
        model_KNN.fit(training_data,np.ravel(training_target))
        score_KNN = score_regression(validation_target, model_KNN.predict(validation_data))
        perf_KNN = perf_KNN + score_KNN
        # MLP
        model_MLP = MLPRegressor()
        model_MLP.fit(training_data,np.ravel(training_target))
        score_MLP = score_regression(validation_target, model_MLP.predict(validation_data))
        perf_MLP = perf_MLP + score_MLP
        # Decision Tree
        model_DT = DecisionTreeRegressor()
        model_DT.fit(training_data, training_target)
        score_DT = score_regression(validation_target, model_DT.predict(validation_data))
        perf_DT = perf_DT + score_DT
        # SVM
        model_SVM = SVR()
        model_SVM.fit(training_data, training_target)
        score_SVM = score_regression(validation_target, model_SVM.predict(validation_data))
        perf_SVM = perf_SVM + score_SVM  
        count = count + 1
    perf_LR = perf_LR/count
    perf_Lasso = perf_Lasso/count
    perf_KNN = perf_KNN/count
    perf_MLP = perf_MLP/count
    perf_DT = perf_DT/count
    perf_SVM = perf_SVM/count
    
    if figure:
        plt.figure()
        models_reg = ['LR','Lasso','KNN','MLP','DT','SVM']
        plt.bar(models_reg,[perf_LR,perf_Lasso,perf_KNN,perf_MLP,perf_DT,perf_SVM])
        plt.title("Accuracy of regression models")
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
    
# 3. Classification
if cl_test:
    # 3.1. dataset generation
    Y1_cl = np.zeros(len(Y1))
    for i in range(len(Y1)):
        if Y1[i] < 500:
            Y1_cl[i] = 0
        elif Y1[i] < 1400:
            Y1_cl[i] = 1
        elif Y1[i] < 5000:
            Y1_cl[i] = 2
        elif Y1[i] < 10000:
            Y1_cl[i] = 3
        else:
            Y1_cl[i] = 4
    
    Y1_cl_num = np.zeros(5)
    for i in range(len(Y1_cl)):
        Y1_cl_num[int(Y1_cl[i])] = Y1_cl_num[int(Y1_cl[i])] + 1
    
    if figure:  
        plt.figure()
        plt.pie(Y1_cl_num)  
        plt.title("Imbalanced classes")
        plt.legend(['class 0','class 1','class 2','class 3','class 4'])
    
    dataset_cl = PCAed_X1
    dataset_target_cl = Y1_cl
    
    # 3.2. Model Evaluation
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    
    kfold = KFold(k)
    count = 0
    perf_KNN_cl = 0
    perf_MLP_cl = 0
    perf_DT_cl = 0
    perf_SVM_cl = 0
    for trn_idx, tst_idx in kfold.split(dataset):
        training_data = dataset_cl[trn_idx]
        training_target = dataset_target_cl[trn_idx]
        validation_data = dataset_cl[tst_idx]
        validation_target = dataset_target_cl[tst_idx]
        # KNN
        model_KNN_cl = KNeighborsClassifier(n_neighbors=30,weights='distance')
        model_KNN_cl.fit(training_data,np.ravel(training_target))
        score_KNN_cl = model_KNN_cl.score(validation_data, validation_target)
        perf_KNN_cl = perf_KNN_cl + score_KNN_cl
        # MLP
        model_MLP_cl = MLPClassifier()
        model_MLP_cl.fit(training_data,np.ravel(training_target))
        score_MLP_cl = model_MLP_cl.score(validation_data, validation_target)
        perf_MLP_cl = perf_MLP_cl + score_MLP_cl
        # Decision Tree
        model_DT_cl = DecisionTreeClassifier()
        model_DT_cl.fit(training_data, training_target)
        score_DT_cl = model_DT_cl.score(validation_data, validation_target)
        perf_DT_cl = perf_DT_cl + score_DT_cl
        #SVM
        model_SVM_cl = SVC()
        model_SVM_cl.fit(training_data, training_target)
        score_SVM_cl = model_SVM_cl.score(validation_data, validation_target)
        perf_SVM_cl = perf_SVM_cl + score_SVM_cl  
        count = count + 1
    perf_KNN_cl = perf_KNN_cl/count
    perf_MLP_cl = perf_MLP_cl/count
    perf_DT_cl = perf_DT_cl/count
    perf_SVM_cl = perf_SVM_cl/count
    
    if figure:
        plt.figure()
        models_cl = ['KNN','MLP','DT','SVM']
        plt.bar(models_cl,[perf_KNN_cl,perf_MLP_cl,perf_DT_cl,perf_SVM_cl])
        plt.title("Accuracy of classification models")
        plt.xlabel('Models')
        plt.ylabel('Accuracy')

# =============================================================================
# 4. Improvements
# =============================================================================

if imp_test:
    # 4.1. Boosting regression
    from sklearn.ensemble import GradientBoostingRegressor
    
    kfold = KFold(k)
    count = 0
    perf_boos = 0
    for trn_idx, tst_idx in kfold.split(dataset):
        training_data = dataset[trn_idx]
        training_target = dataset_target[trn_idx]
        validation_data = dataset[tst_idx]
        validation_target = dataset_target[tst_idx]
        # Linear Regression
        model_boos = GradientBoostingRegressor()
        model_boos.fit(training_data,training_target)
        score_boos = score_regression(validation_target, model_boos.predict(validation_data))
        perf_boos = perf_boos + score_boos
        count = count + 1
    perf_boos = perf_boos/count
    
    # 4.2. Bagging Classification
    from sklearn.ensemble import BaggingClassifier
    
    kfold = KFold(k)
    count = 0
    perf_bag = 0
    for trn_idx, tst_idx in kfold.split(dataset):
        training_data = dataset_cl[trn_idx]
        training_target = dataset_target_cl[trn_idx]
        validation_data = dataset_cl[tst_idx]
        validation_target = dataset_target_cl[tst_idx]
        # Classification
        model_bag = BaggingClassifier()
        model_bag.fit(training_data,np.ravel(training_target))
        score_bag = model_bag.score(validation_data, validation_target)
        perf_bag = perf_bag + score_bag
        count = count + 1
    perf_bag = perf_bag/count
    
    #4.3. Combination of regression and classification
    kfold = KFold(k)
    count = 0
    perf_reg_cl = 0
    n = 0
    for trn_idx, tst_idx in kfold.split(dataset):
        training_data = dataset[trn_idx]
        training_target_reg = dataset_target[trn_idx]
        training_target_cl = dataset_target_cl[trn_idx]
        validation_data = dataset[tst_idx]
        validation_target_reg = dataset_target[tst_idx]
        validation_target_cl = dataset_target_cl[tst_idx]
        # Regression
        model_reg = KNeighborsRegressor(n_neighbors=7, weights='distance')
        model_reg.fit(training_data, np.ravel(training_target_reg))
        predict_reg = model_reg.predict(validation_data)
        # Classification   
        model_cl = SVC()
        model_cl.fit(training_data, np.ravel(training_target_cl))
        predict_cl = model_cl.predict(validation_data)
        # Ensemble
        predict = np.zeros(len(predict_reg))
        for i in range(len(predict_reg)):
            if predict_reg[i] < 500:
                if predict_cl[i] == 0:
                    predict[i] = predict_reg[i]
                elif predict_cl[i] == 1:
                    predict[i] = np.random.random_integers(500,1400)
                    n = n+1
                elif predict_cl[i] == 2:
                    predict[i] = np.random.random_integers(1400,5000)
                    n = n+1
                elif predict_cl[i] == 3:
                    predict[i] = np.random.random_integers(5000,10000)
                    n = n+1
                else:
                    predict[i] = np.random.random_integers(10000,15000)
                    n = n+1
            elif predict_reg[i] < 1400:
                if predict_cl[i] == 0:
                    predict[i] = np.random.random_integers(0,500)
                    n = n+1
                elif predict_cl[i] == 1:
                    predict[i] = predict_reg[i]
                elif predict_cl[i] == 2:
                    predict[i] = np.random.random_integers(1400,5000)
                    n = n+1
                elif predict_cl[i] == 3:
                    predict[i] = np.random.random_integers(5000,10000)
                    n = n+1
                else:
                    predict[i] = np.random.random_integers(10000,15000)
                    n = n+1
            elif predict_reg[i] < 5000:
                if predict_cl[i] == 0:
                    predict[i] = np.random.random_integers(0,500)
                    n = n+1
                elif predict_cl[i] == 1:
                    predict[i] = np.random.random_integers(500,1400)
                    n = n+1
                elif predict_cl[i] == 2:
                    predict[i] = predict_reg[i]
                elif predict_cl[i] == 3:
                    predict[i] = np.random.random_integers(5000,10000)
                    n = n+1
                else:
                    predict[i] = np.random.random_integers(10000,15000)
                    n = n+1
            elif predict_reg[i] < 10000:
                if predict_cl[i] == 0:
                    predict[i] = np.random.random_integers(0,500)
                    n = n+1
                elif predict_cl[i] == 1:
                    predict[i] = np.random.random_integers(500,1400)
                    n = n+1
                elif predict_cl[i] == 2:
                    predict[i] = np.random.random_integers(1400,5000)
                    n = n+1
                elif predict_cl[i] == 3:
                    predict[i] = predict_reg[i]
                else:
                    predict[i] = np.random.random_integers(10000,15000)
                    n = n+1
            else:
                if predict_cl[i] == 0:
                    predict[i] = np.random.random_integers(0,500)
                    n = n+1
                elif predict_cl[i] == 1:
                    predict[i] = np.random.random_integers(500,1400)
                    n = n+1
                elif predict_cl[i] == 2:
                    predict[i] = np.random.random_integers(1400,5000)
                    n = n+1
                elif predict_cl[i] == 3:
                    predict[i] = np.random.random_integers(5000,10000)
                    n = n+1
                else:
                    predict[i] = predict_reg[i]
        # Score evaluation
        score_reg_cl = score_regression(validation_target_reg,predict)
        perf_reg_cl = perf_reg_cl + score_reg_cl
        count = count + 1
    perf_reg_cl = perf_reg_cl/count
    
    if figure:
        plt.figure()
        models_imp = ['Boosting','Bagging','Reg_Cl']
        plt.bar(models_imp,[perf_boos,perf_bag,perf_reg_cl])
        plt.title("Accuracy of improved models")
        plt.xlabel('Models')
        plt.ylabel('Accuracy')

# =============================================================================
# Final model
# =============================================================================
scaler = preprocessing.StandardScaler()
X2_normalised = scaler.fit_transform(X2)

PCAed_X2 = pca.transform(X2_normalised)

final_model = KNeighborsRegressor(n_neighbors=7,weights='distance')
final_model.fit(dataset, dataset_target)
Y2 = final_model.predict(PCAed_X2)

Y2_np = np.array(Y2)

Y2_round = np.zeros(len(Y2_np))
for i in range(len(Y2_np)):
    Y2_round[i] = int(np.round(Y2_np[i]))

Y2_copy = np.copy(Y2_np)

pd.DataFrame(Y2_copy.astype('int32')).to_csv("Y2.csv", index=False, encoding='utf-8')
