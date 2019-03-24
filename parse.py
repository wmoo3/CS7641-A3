# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:39:27 2017

@author: jtay
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
import os 
import sklearn.model_selection as ms
from sklearn.preprocessing import OneHotEncoder

for d in ['BASE','RP','PCA','ICA','RF']:
    n = './{}/'.format(d)
    if not os.path.exists(n):
        os.makedirs(n)

OUT = './BASE/'

####diamonds####
#diamondsX1 = pd.read_csv('./diamonds_train.data',header=None,sep=' ')
#diamondsX2 = pd.read_csv('./diamonds_valid.data',header=None,sep=' ')
#diamondsX = pd.concat([diamondsX1,diamondsX2],0).astype(float)
#diamondsY1 = pd.read_csv('./diamonds_train.labels',header=None,sep=' ')
#diamondsY2 = pd.read_csv('./diamonds_valid.labels',header=None,sep=' ')
#diamondsY = pd.concat([diamondsY1,diamondsY2],0)
#diamondsY.columns = ['Class']

diamondsX = pd.read_csv('C:\Personal\GA Tech\CS7641\Assignment3\diamonds.csv',sep=',')
diamondsY = diamondsX.copy()
diamondsX = diamondsX.drop(['Class'],axis=1)
diamondsY = diamondsY.drop(diamondsY.columns[0:9],axis=1)

diamonds_trgX, diamonds_tstX, diamonds_trgY, diamonds_tstY = ms.train_test_split(diamondsX, diamondsY, test_size=0.3, random_state=0,stratify=diamondsY)     

diamondsX1 = pd.DataFrame(diamonds_trgX)
diamondsY1 = pd.DataFrame(diamonds_trgY)
#diamondsY.columns = ['Class']

diamondsX2 = pd.DataFrame(diamonds_tstX)
diamondsY2 = pd.DataFrame(diamonds_tstY)
#diamondsY2.columns = ['Class']

diamonds1 = pd.concat([diamondsX1,diamondsY1],1)
#diamonds1 = diamonds1.dropna(axis=1,how='all')
diamonds1.to_hdf(OUT+'datasets.hdf','diamonds',complib='blosc',complevel=9)

diamonds2 = pd.concat([diamondsX2,diamondsY2],1)
#diamonds2 = diamonds2.dropna(axis=1,how='all')
diamonds2.to_hdf(OUT+'datasets.hdf','diamonds_test',complib='blosc',complevel=9)

####poker###
pokerX = pd.read_csv('C:\Personal\GA Tech\CS7641\Assignment3\poker-hand.csv',sep=',')
pokerY = pokerX.copy()
pokerX = pokerX.drop(['Class'],axis=1)
pokerY = pokerY.drop(pokerY.columns[0:10],axis=1)

poker_trgX, poker_tstX, poker_trgY, poker_tstY = ms.train_test_split(pokerX, pokerY, test_size=0.3, random_state=0,stratify=pokerY)     

pokerX1 = pd.DataFrame(poker_trgX)
pokerY1 = pd.DataFrame(poker_trgY)

pokerX2 = pd.DataFrame(poker_tstX)
pokerY2 = pd.DataFrame(poker_tstY)

poker1 = pd.concat([pokerX1,pokerY1],1)
poker1.to_hdf(OUT+'datasets.hdf','poker',complib='blosc',complevel=9)

poker2 = pd.concat([pokerX2,pokerY2],1)
poker2.to_hdf(OUT+'datasets.hdf','poker_test',complib='blosc',complevel=9)

####poker_oh####
onehot=OneHotEncoder(categories='auto')
onehot.fit(pokerX)
pokerX=pd.DataFrame(data=onehot.transform(pokerX).toarray(),columns=onehot.get_feature_names())

poker_trgX, poker_tstX, poker_trgY, poker_tstY = ms.train_test_split(pokerX, pokerY, test_size=0.3, random_state=0,stratify=pokerY)     

pokerX1 = pd.DataFrame(poker_trgX)
pokerY1 = pd.DataFrame(poker_trgY)

pokerX2 = pd.DataFrame(poker_tstX)
pokerY2 = pd.DataFrame(poker_tstY)

poker1 = pd.concat([pokerX1,pokerY1],1)
poker1.to_hdf(OUT+'datasets.hdf','poker_oh',complib='blosc',complevel=9)

poker2 = pd.concat([pokerX2,pokerY2],1)
poker2.to_hdf(OUT+'datasets.hdf','poker_oh_test',complib='blosc',complevel=9)


####digits####
digits = load_digits(return_X_y=True)
digitsX,digitsY = digits

digits = np.hstack((digitsX, np.atleast_2d(digitsY).T))
digits = pd.DataFrame(digits)
cols = list(range(digits.shape[1]))
cols[-1] = 'Class'
digits.columns = cols
digits.to_hdf(OUT+'datasets.hdf','digits',complib='blosc',complevel=9)

