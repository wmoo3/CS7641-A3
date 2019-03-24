# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:17:14 2017

@author: JTay
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = './BASE/'
np.random.seed(0)
digits = pd.read_hdf('./BASE/datasets.hdf','digits')
digitsX = digits.drop('Class',1).copy().values
digitsY = digits['Class'].copy().values

diamonds = pd.read_hdf('./BASE/datasets.hdf','diamonds')        
diamondsX = diamonds.drop('Class',1).copy().values
diamondsY = diamonds['Class'].copy().values


diamondsX = StandardScaler().fit_transform(diamondsX)
digitsX= StandardScaler().fit_transform(digitsX)

#%% benchmarking for chart type 2

grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(diamondsX,diamondsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Diamonds NN bmk.csv')


mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Digits NN bmk.csv')
raise