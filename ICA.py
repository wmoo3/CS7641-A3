

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg, reconstructionError_ICA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
from collections import defaultdict
from itertools import product

out = './ICA/'

np.random.seed(0)
digits = pd.read_hdf('./BASE/datasets.hdf','digits')
digitsX = digits.drop('Class',1).copy().values
digitsY = digits['Class'].copy().values

diamonds = pd.read_hdf('./BASE/datasets.hdf','diamonds')        
diamondsX = diamonds.drop('Class',1).copy().values
diamondsY = diamonds['Class'].copy().values


diamondsX = StandardScaler().fit_transform(diamondsX)
digitsX= StandardScaler().fit_transform(digitsX)

clusters =  [2,5,10,15,20,25,30,35,40,45,50]
dims1 = [2,3,4,5,6,7,8,9]
dims2 = [2,5,10,15,20,25,30,35,40,45,50,55,60]
#raise


#%% task 2

ica = FastICA(random_state=5)
kurt = {}
for dim in dims1:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(diamondsX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()
    print ("diamonds",dim, tmp)
kurt = pd.Series(kurt) 
kurt.to_csv(out+'diamonds scree1.csv')

ica = FastICA(random_state=5, tol=2E-2)
kurt = {}
for dim in dims2:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(digitsX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()
    print ("digits",dim, tmp)
kurt = pd.Series(kurt) 
kurt.to_csv(out+'digits scree1.csv')


ica = FastICA(random_state=5)
SSE = {}
for dim in dims1:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(diamondsX)
    SSE[dim] = reconstructionError_ICA(ica, tmp, diamondsX)
    print ("diamonds",dim, SSE[dim])
SSE = pd.Series(SSE) 
SSE.to_csv(out+'diamonds scree2.csv')

ica = FastICA(random_state=5, tol=2E-2)
SSE = {}
for dim in dims2:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(digitsX)
    SSE[dim] = reconstructionError_ICA(ica, tmp, digitsX)
    print ("digits",dim, SSE[dim])
SSE = pd.Series(SSE) 
SSE.to_csv(out+'digits scree2.csv')

#raise

#%% task 4

grid ={'ica__n_components':dims1,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(diamondsX,diamondsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'diamonds dim red.csv')


grid ={'ica__n_components':dims2,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5, tol=2E-2)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digits dim red.csv')
raise

#%% data for task 3 but find the good dim values first, use clustering script to finish up
dim = 5
ica = FastICA(n_components=dim,random_state=10)

diamondsX2 = ica.fit_transform(diamondsX)
diamonds2 = pd.DataFrame(np.hstack((diamondsX2,np.atleast_2d(diamondsY).T)))
cols = list(range(diamonds2.shape[1]))
cols[-1] = 'Class'
diamonds2.columns = cols
diamonds2.to_hdf(out+'datasets.hdf','diamonds',complib='blosc',complevel=9)

dim = 35
ica = FastICA(n_components=dim,random_state=10)
digitsX2 = ica.fit_transform(digitsX)
digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
cols = list(range(digits2.shape[1]))
cols[-1] = 'Class'
digits2.columns = cols
digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)
'''
#%% extra dimension test
dim = 6
ica = FastICA(n_components=dim,random_state=10)

diamondsX2 = ica.fit_transform(diamondsX)
diamonds2 = pd.DataFrame(np.hstack((diamondsX2,np.atleast_2d(diamondsY).T)))
cols = list(range(diamonds2.shape[1]))
cols[-1] = 'Class'
diamonds2.columns = cols
diamonds2.to_hdf(out+'datasets1.hdf','diamonds',complib='blosc',complevel=9)

dim = 45
ica = FastICA(n_components=dim,random_state=10)
digitsX2 = ica.fit_transform(digitsX)
digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
cols = list(range(digits2.shape[1]))
cols[-1] = 'Class'
digits2.columns = cols
digits2.to_hdf(out+'datasets1.hdf','digits',complib='blosc',complevel=9)
'''