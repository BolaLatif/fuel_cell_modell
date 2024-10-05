#%%
# importing modules
import numpy as np
import pandas as pd
import seaborn as sns
import os
import time


# importing sklearn modules
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split

n_cpus = int(np.round(os.cpu_count()*0.75))

#%%

# Prepairing Dataframe

df = pd.read_excel('lastkurven.xlsx', sheet_name = 'lastkurven_adiabat')
df.drop(index=(0), inplace= True)
df.dropna(inplace=True)
df.drop(axis=1, columns=("Unnamed: 0"), inplace= True)

data_in = ['x_H2O', 'x_H2', 'n_fuel_in', 'n_air_in', 'I', 'T_air_in', 'T_fuel_in']

X = df[data_in].to_numpy(dtype = float)
U = df['U_load'].to_numpy(dtype = float)
T = df['T_mean'].to_numpy(dtype = float)

# splitting the Data to 2/3 for Training and Validation and 1/3 for testing

X_train, X_test, U_train, U_test, T_train, T_test =train_test_split(X, U, T, test_size=0.333, random_state=42)

#%%

def TFC(X_train, y_train, nu_range, C_range, degree_range, coef_range, gamma,  kernel):
    
    print("{}{} REGRESSION USING {} CPUs USING {} KERNEL".format(2*"\n", 10*"#", n_cpus, kernel))
    
    # create a dictionary with hyperparameters to investigate
    if kernel == "poly" :
        hp_dict = {"nusvr__nu": nu_range,
                   "nusvr__C": C_range,
                   "nusvr__degree": degree_range,
                   "nusvr__coef0": coef_range,
                   "nusvr__gamma":gamma
                   }
    elif kernel == "rbf" :
        hp_dict = {"nusvr__nu": nu_range,
                   "nusvr__C": C_range,
                   "nusvr__gamma":gamma
                   }
    elif kernel == "sigmoid" :
        hp_dict = {"nusvr__nu": nu_range,
                   "nusvr__C": C_range,
                   "nusvr__coef0": coef_range,
                   "nusvr__gamma":gamma
                   }
    
    # create a grid-search setup
    # optimize hyperparams with respect to maximize R2-score
    model = GridSearchCV(estimator  = make_pipeline(StandardScaler(), NuSVR(kernel=kernel, shrinking=False,
                                                                            verbose=False, cache_size=1000)),
                         param_grid = hp_dict,
                         scoring    = "r2",
                         n_jobs     = n_cpus,
                         cv         = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42),
                         verbose    = 1)
    
    # execute the grid search
    model.fit(X_train, y_train)
    
    # show best hyperparams (hyperparams, that maximized mean R2-score)
    print("\n best hyper-params:")
    for param_name, param_value in model.best_params_.items():
        print("  => {}: {}".format(param_name, param_value))
    
    # show best mean validation-R2-score
    print("\n best mean score of k-fold crossvalidation: \n  => {}".format(model.best_score_.round(3)))
    
    # calculate R2-score for train-data
    score = model.score(X_train, y_train)
    print("\n score on training-data: \n  => {}".format(score.round(3)))
    
    return model

#%%%

# Range for Hyperparameter

nu_range = np.arange(0.1, 1, 0.1)
c_range = np.arange(0, 1, 0.1)
degree_range = [3, 4, 5]
coef_range = np.arange(0, 1, 0.1)
gamma = ['scale', 'auto']
kernels =['rbf', 'sigmoid']


# trying to create a model from data

for kernel in kernels :

# starting counter
    time_start=time.perf_counter()
    SOFC_Model = TFC(X_train, U_train, nu_range, c_range, degree_range, coef_range, gamma, kernel)

    U_predict = SOFC_Model.predict(X_train)

    print("the time took to fit the Model using {} Kernel and predict the outputs is {} sec.".format(kernel, round(time.perf_counter() - time_start, 1)))
# df.to_excel('xxx.xlsx')

# testing the Model 
 
    U_test_predict = SOFC_Model.predict(X_test)

    print("\n score on the Training Data is {} \n score on the test Data is {}".format(r2_score(U_predict, U_train).round(3), 
                                                                                   r2_score(U_test_predict, U_test).round(3)))

#plotting the data

    # sns.set(rc={'figure.figsize':(10,10)})
    df['U_svr_'+ str(kernel)] = SOFC_Model.predict(X)
    sns.lineplot(x='I', y='U_svr_'+str(kernel), hue='T_air_in', data=df, palette="Set2")
    sns.scatterplot(x='I', y='U_load', data=df, hue="T_air_in", palette="Set2")
# g = sns.FacetGrid(df, col=['U_svr_rbf', 'U_svr_sigmoid'],row='T_air_in',hue="T_air_in", margin_titles=True, height=5)
# g.map(sns.lineplot, "I", 'U_svr_sigmoid')
# g.map(sns.lineplot, "I", "U_svr_rbf")
# g.map(sns.lineplot, "I", 'U_svr_poly')
# g.map(sns.scatterplot, "I", "U_load")

"""
Created on Tue Nov 15 13:23:41 2022

@author: Bola Latif
"""

