

#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg, gmm_js
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys
import warnings

out = './{}/'.format(sys.argv[1])
#out = './{}/'.format('ICA')
np.random.seed(0)

digits = pd.read_hdf(out+'datasets.hdf','digits')
digitsX = digits.drop('Class',1).copy().values
digitsY = digits['Class'].copy().values

diamonds = pd.read_hdf(out+'datasets.hdf','diamonds')        
diamondsX = diamonds.drop('Class',1).copy().values
diamondsY = diamonds['Class'].copy().values


diamondsX = StandardScaler().fit_transform(diamondsX)
digitsX= StandardScaler().fit_transform(digitsX)

clusters = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,80,90,100]

#n_samples_diamondsX = int(len(diamondsX)/2)
#diamondsX_tmp = diamondsX.copy()
#diamondsX_1 = diamondsX_tmp[:n_samples_diamondsX]
#diamondsX_2 = diamondsX_tmp[n_samples_diamondsX:]

#n_samples_digitsX = int(len(digitsX)/2)
#digitsX_tmp = digitsX.copy()
#digitsX_1 = digitsX_tmp[:n_samples_digitsX]
#digitsX_2 = digitsX_tmp[n_samples_digitsX:]

#%% task 1,3
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    SSE = defaultdict(dict)
    ll = defaultdict(dict)
    acc = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)
    
    st = clock()
    for k in clusters:
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(diamondsX)
        gmm.fit(diamondsX)
        SSE[k]['Diamonds'] = km.score(diamondsX)
        ll[k]['Diamonds'] = gmm.score(diamondsX)    
        acc[k]['Diamonds']['Kmeans'] = cluster_acc(diamondsY,km.predict(diamondsX))
        acc[k]['Diamonds']['GMM'] = cluster_acc(diamondsY,gmm.predict(diamondsX))
        adjMI[k]['Diamonds']['Kmeans'] = ami(diamondsY,km.predict(diamondsX))
        adjMI[k]['Diamonds']['GMM'] = ami(diamondsY,gmm.predict(diamondsX))
        
        km.fit(digitsX)
        gmm.fit(digitsX)
        SSE[k]['Digits'] = km.score(digitsX)
        ll[k]['Digits'] = gmm.score(digitsX)
        acc[k]['Digits']['Kmeans'] = cluster_acc(digitsY,km.predict(digitsX))
        acc[k]['Digits']['GMM'] = cluster_acc(digitsY,gmm.predict(digitsX))
        adjMI[k]['Digits']['Kmeans'] = ami(digitsY,km.predict(digitsX))
        adjMI[k]['Digits']['GMM'] = ami(digitsY,gmm.predict(digitsX))
        print(k, clock()-st)
        
        
    SSE = (-pd.DataFrame(SSE)).T
    SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
    ll = pd.DataFrame(ll).T
    ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
    acc = pd.Panel(acc)
    adjMI = pd.Panel(adjMI)
    
    
    SSE.to_csv(out+'SSE.csv')
    ll.to_csv(out+'logliklihood.csv')
    acc.ix[:,:,'Digits'].to_csv(out+'Digits acc.csv')
    acc.ix[:,:,'Diamonds'].to_csv(out+'Diamonds acc.csv')
    adjMI.ix[:,:,'Digits'].to_csv(out+'Digits adjMI.csv')
    adjMI.ix[:,:,'Diamonds'].to_csv(out+'Diamonds adjMI.csv')


#%% task 5

grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,n_jobs=-1)

gs.fit(diamondsX,diamondsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Diamonds cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,n_jobs=-1)

gs.fit(diamondsX,diamondsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Diamonds cluster GMM.csv')


grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,n_jobs=-1)

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Digits cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,n_jobs=-1)

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Digits cluster GMM.csv')


# %% For chart 4/5
diamondsX2D = TSNE(verbose=10,random_state=5).fit_transform(diamondsX)
digitsX2D = TSNE(verbose=10,random_state=5).fit_transform(digitsX)

diamonds2D = pd.DataFrame(np.hstack((diamondsX2D,np.atleast_2d(diamondsY).T)),columns=['x','y','target'])
digits2D = pd.DataFrame(np.hstack((digitsX2D,np.atleast_2d(digitsY).T)),columns=['x','y','target'])

diamonds2D.to_csv(out+'diamonds2D.csv')
digits2D.to_csv(out+'digits2D.csv')

