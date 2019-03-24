# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:47:56 2017

@author: jtay
"""

import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score as acc
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_chunked
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.base import TransformerMixin,BaseEstimator
import scipy.sparse as sps
from scipy.linalg import pinv

nn_arch= [(25,),(50,),(25,25),(50,50),(100,100),(25,25,25),(100,25,100)]
nn_reg = [10**-x for x in range(1,5)]

def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X, _ = gmm_p.sample(n_samples)

    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y, _ = gmm_q.sample(n_samples)
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2
    '''
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    return (log_p_X.mean() - log_q_X.mean())
    '''        
    
def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
#    assert max(pred) == max(Y)
#    assert min(pred) == min(Y)    
    return acc(Y,pred)


class myGMM(GMM):
    def transform(self,X):
        return self.predict_proba(X)
        
        
def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)

    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

def pairwiseDistCorr_chunked(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    x1 = X1[:]
    x2 = X2[:]
    chunks = 0
    total_chunks = 0
    corref=[]
   
    while True:
        d1 = next(pairwise_distances_chunked(x1, working_memory=512))
        d2 = next(pairwise_distances_chunked(x2, working_memory=512))
        co=np.corrcoef(d1.ravel(),d2.ravel())[0,1]
        corref.append(co)
        chunks = d1.shape[0]
        total_chunks += chunks
        if total_chunks >= X1.shape[0]:
            break
        x1 = X1[total_chunks:,:]
        x2 = X2[total_chunks:,:]
    return sum(corref)/len(corref)

    
def aveMI(X,Y):    
    MI = MIC(X,Y) 
    return np.nanmean(MI)
    
  
def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)
    
def reconstructionError_ICA(projections,X1,X2):
    assert X1.shape[0] == X2.shape[0]
    ori = projections.inverse_transform(X1)
    sse = sum((ori-X2)**2)
    return sse.mean()

# http://datascience.stackexchange.com/questions/6683/feature-selection-using-feature-importances-in-random-forests-with-scikit-learn          
class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
         self.model = model
         self.n = n
    def fit(self, *args, **kwargs):
         self.model.fit(*args, **kwargs)
         return self
    def transform(self, X):
         return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]
                  
#http://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
from scipy.spatial import distance     
def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)                  