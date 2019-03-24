# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:51:37 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import  nn_arch,nn_reg
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

out = './PCA/'
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
dims1 = [2,3,4,5,6,7,8]
dims2 = [2,5,10,15,20,25,30,35,40,45,50,55,60]
#raise

#%% task 2

pca = PCA(random_state=5)
pca.fit(diamondsX)
data = pca.explained_variance_
tmp = pd.Series(data,index = range(1,len(data)+1))
tmp.to_csv(out+'diamonds scree.csv')


pca = PCA(random_state=5)
pca.fit(digitsX)
data = pca.explained_variance_
tmp = pd.Series(data,index = range(1,len(data)+1))
tmp.to_csv(out+'digits scree.csv')

#%% task 4

grid ={'pca__n_components':dims1,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(diamondsX,diamondsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'diamonds dim red.csv')


grid ={'pca__n_components':dims2,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digits dim red.csv')
raise

#%% data for task 3 but find the good dim values first, use clustering script to finish up

dim = 6
pca = PCA(n_components=dim,random_state=10)

diamondsX2 = pca.fit_transform(diamondsX)
diamonds2 = pd.DataFrame(np.hstack((diamondsX2,np.atleast_2d(diamondsY).T)))
cols = list(range(diamonds2.shape[1]))
cols[-1] = 'Class'
diamonds2.columns = cols
diamonds2.to_hdf(out+'datasets.hdf','diamonds',complib='blosc',complevel=9)

dim = 30
pca = PCA(n_components=dim,random_state=10)
digitsX2 = pca.fit_transform(digitsX)
digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
cols = list(range(digits2.shape[1]))
cols[-1] = 'Class'
digits2.columns = cols
digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)
'''
#%% extra dimension test

dim = 6
pca = PCA(n_components=dim,random_state=10)

diamondsX2 = pca.fit_transform(diamondsX)
diamonds2 = pd.DataFrame(np.hstack((diamondsX2,np.atleast_2d(diamondsY).T)))
cols = list(range(diamonds2.shape[1]))
cols[-1] = 'Class'
diamonds2.columns = cols
diamonds2.to_hdf(out+'datasets.hdf','diamonds',complib='blosc',complevel=9)

dim = 30
pca = PCA(n_components=dim,random_state=10)
digitsX2 = pca.fit_transform(digitsX)
digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
cols = list(range(digits2.shape[1]))
cols[-1] = 'Class'
digits2.columns = cols
digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)
'''