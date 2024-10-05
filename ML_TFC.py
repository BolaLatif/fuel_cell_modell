"""
main scrpt
"""
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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split

n_cpus = int(np.round(os.cpu_count()*0.75))

#%%

# Prepairing Dataframe

df = pd.read_excel('E:\Arbeit\Tfc\Daten\lastkurven.xlsx', sheet_name = 'lastkurven_adiabat')
df.drop(index=(0), inplace= True)
df.dropna(inplace=True)
df.drop(axis=1, columns=("Unnamed: 0"), inplace= True)

data_in = ['x_H2O', 'x_H2', 'n_fuel_in', 'n_air_in', 'I', 'T_air_in', 'T_fuel_in']

X = df[data_in].to_numpy(dtype = float)

U = df['U_load'].to_numpy(dtype = float)

T = df['T_mean'].to_numpy(dtype = float)


# splitting the Data

X_train, X_test, U_train, U_test, T_train, T_test =train_test_split(X, U, T, test_size=0.333, random_state=42)



#%%

def SOFC_ML(X_train, y_train, degree_range, l2_range):
    
    print("{}{} REGRESSION USING {} CPUs".format(2*"\n", 10*"#", n_cpus))
    
    # create a dictionary with hyperparameters to investigate
    hp_dict = {"polynomialfeatures__degree": degree_range,
               "ridge__alpha": l2_range,
               }
    
    # create a grid-search setup
    # optimize hyperparams with respect to maximize R2-score
    model = GridSearchCV(estimator  = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge(solver = 'lsqr')),
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

degree_range = np.arange(1, 7)
l2_range =  [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] 

# starting counter
time_start=time.perf_counter()

# trying to make a model from inputs & Outputs

SOFC_Model_U = SOFC_ML(X_train, U_train, degree_range, l2_range)

U_pred = SOFC_Model_U.predict(X_train)

# df.to_excel('xxx.xlsx')

# testing the Model 
 
U_test_pred = SOFC_Model_U.predict(X_test)

print("\n score for U on the Training Data is {} \n score for U on the test Data is {}".format(r2_score(U_train, U_pred).round(3), 
                                                                                               r2_score(U_test, U_test_pred).round(3)))

# Predicting T_mean

SOFC_Model_T = SOFC_ML(X_train, T_train, degree_range, l2_range)

T_pred = SOFC_Model_T.predict(X_train)

T_test_pred = SOFC_Model_T.predict(X_test)

print("\n score for T_mean on the Training Data is {} \n score for T_mean on the test Data is {}".format(r2_score(U_train, U_pred).round(3),
                                                                                                         r2_score(U_test, U_test_pred).round(3)))
print("the time took to fit and predict for both Voltage and T_mean is {} sec".format(round(time.perf_counter() - time_start, 1)))

#plotting the data

# sns.set(rc={'figure.figsize':(10,10)})
df['U_ml'] = SOFC_Model_U.predict(X)
df['T_mean_ml'] = SOFC_Model_T.predict(X)
# sns.lineplot(x='U_load', y='T_mean_ml', data=df)
# sns.scatterplot(x='U_load', y='T_mean', data=df)
g = sns.FacetGrid(df, col="T_air_in", margin_titles=True, height=5, hue="T_air_in" )
g.map(sns.lineplot, "I", "U_ml")
g.map(sns.scatterplot, "I", "U_load")

# fig, axs = plt.subplot()


"""
Created on Fri Aug 19 05:47:58 2022

@author: Bola Latif Noshy
"""

