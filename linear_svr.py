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
from sklearn.svm import LinearSVR
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

def TFC(X_train, y_train, epsilon_range, C_range):
    
    print("{}{} REGRESSION USING {} CPUs".format(2*"\n", 10*"#", n_cpus))
    
    # create a dictionary with hyperparameters to investigate
    hp_dict = {"linearsvr__epsilon": epsilon_range,
               "linearsvr__C": C_range,
               }
    
    # create a grid-search setup
    # optimize hyperparams with respect to maximize R2-score
    model = GridSearchCV(estimator  = make_pipeline(StandardScaler(), LinearSVR(dual=False,max_iter=1000_000, loss='squared_epsilon_insensitive')),
                         param_grid = hp_dict,
                         scoring    = "r2",
                         n_jobs     = n_cpus,
                         cv         = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42),
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

epsilon_range = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
C_range = epsilon_range

# starting counter
time_start=time.perf_counter()

# trying to make a model from inputs & Outputs

SOFC_Model = TFC(X_train, U_train, epsilon_range, C_range)

U_predict = SOFC_Model.predict(X_train)

print("the time took to fit the Model and predict the outputs is {} sec.".format(round(time.perf_counter() - time_start, 1)))
# df.to_excel('xxx.xlsx')

# testing the Model 
 
U_test_predict = SOFC_Model.predict(X_test)

print("\n score on the Training Data is {} \n score on the test Data is {}".format(r2_score(U_predict, U_train).round(3), 
                                                                                   r2_score(U_test_predict, U_test).round(3)))

#plotting the data

sns.set(rc={'figure.figsize':(10,10)})
df['U_ml'] = SOFC_Model.predict(X)
sns.lineplot(x='I', y='U_ml', hue='T_air_in', data=df, palette="Set2")
sns.scatterplot(x='I', y='U_load', data=df, hue="T_air_in", palette="Set2")

"""
Created on Mon Nov 14 18:35:55 2022

@author: Bola Latif Noshy
"""

