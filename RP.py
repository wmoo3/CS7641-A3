

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError, pairwiseDistCorr_chunked
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product

out = './RP/'
cmap = cm.get_cmap('Spectral') 

np.random.seed(0)
digits = pd.read_hdf('./BASE/datasets.hdf','digits')
digitsX = digits.drop('Class',1).copy().values
digitsY = digits['Class'].copy().values

diamonds = pd.read_hdf('./BASE/datasets.hdf','diamonds')        
diamondsX = diamonds.drop('Class',1).copy().values
diamondsY = diamonds['Class'].copy().values


diamondsX = StandardScaler().fit_transform(diamondsX)
digitsX= StandardScaler().fit_transform(digitsX)

clusters =  [2,5,10,15,20,25,30,35,40]
dims1 = [1,2,3,4,5,6,7,8,9]
dims2 = [2,5,10,15,20,25,30,35,40,45,50,55,60]
#raise
#%% task 2

#tmp = defaultdict(dict)
#for i,dim in product(range(10),dims1):
#    rp = GaussianRandomProjection(random_state=i, n_components=dim)
#    tmp[dim][i] = pairwiseDistCorr_chunked(rp.fit_transform(diamondsX), diamondsX)
#    print (dim, "scree3")
#tmp =pd.DataFrame(tmp).T
#tmp.to_csv(out+'diamonds scree3.csv')

#tmp = defaultdict(dict)
#for i,dim in product(range(10),dims1):
#    rp = GaussianRandomProjection(random_state=i, n_components=dim)
#    rp.fit(diamondsX)    
#    tmp[dim][i] = reconstructionError(rp, diamondsX)
#    print (dim, "scree4")
#tmp =pd.DataFrame(tmp).T
#tmp.to_csv(out+'diamonds scree4.csv')

#%% task 2

tmp = defaultdict(dict)
for i,dim in product(range(10),dims1):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr_chunked(rp.fit_transform(diamondsX), diamondsX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'diamonds scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims2):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(digitsX), digitsX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'digits scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims1):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(diamondsX)    
    tmp[dim][i] = reconstructionError(rp, diamondsX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'diamonds scree2.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims2):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(digitsX)  
    tmp[dim][i] = reconstructionError(rp, digitsX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'digits scree2.csv')

#%% task 4

grid ={'rp__n_components':dims1,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
rp = SparseRandomProjection(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(diamondsX,diamondsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'diamonds dim red.csv')


grid ={'rp__n_components':dims2,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
rp = SparseRandomProjection(random_state=5)           
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digits dim red.csv')
raise

#%% data for task 3 but find the good dim values first, use clustering script to finish up
dim = 6
rp = SparseRandomProjection(n_components=dim,random_state=5)

diamondsX2 = rp.fit_transform(diamondsX)
diamonds2 = pd.DataFrame(np.hstack((diamondsX2,np.atleast_2d(diamondsY).T)))
cols = list(range(diamonds2.shape[1]))
cols[-1] = 'Class'
diamonds2.columns = cols
diamonds2.to_hdf(out+'datasets.hdf','diamonds',complib='blosc',complevel=9)

dim = 25
rp = SparseRandomProjection(n_components=dim,random_state=5)
digitsX2 = rp.fit_transform(digitsX)
digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
cols = list(range(digits2.shape[1]))
cols[-1] = 'Class'
digits2.columns = cols
digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)
'''
#%% extra dimension test
dim = 6
rp = SparseRandomProjection(n_components=dim,random_state=5)

diamondsX2 = rp.fit_transform(diamondsX)
diamonds2 = pd.DataFrame(np.hstack((diamondsX2,np.atleast_2d(diamondsY).T)))
cols = list(range(diamonds2.shape[1]))
cols[-1] = 'Class'
diamonds2.columns = cols
diamonds2.to_hdf(out+'datasets1.hdf','diamonds',complib='blosc',complevel=9)

dim = 45
rp = SparseRandomProjection(n_components=dim,random_state=5)
digitsX2 = rp.fit_transform(digitsX)
digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
cols = list(range(digits2.shape[1]))
cols[-1] = 'Class'
digits2.columns = cols
digits2.to_hdf(out+'datasets1.hdf','digits',complib='blosc',complevel=9)
'''