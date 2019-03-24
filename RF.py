

#%% Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg,ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    out = './RF/'
    
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


    #%% task 2
    
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=1)
    fs_diamonds = rfc.fit(diamondsX,diamondsY).feature_importances_ 
    fs_digits = rfc.fit(digitsX,digitsY).feature_importances_ 
  
    tmp = pd.Series(np.sort(fs_diamonds)[::-1])
    tmp.to_csv(out+'diamonds scree.csv')
    
    tmp = pd.Series(np.sort(fs_digits)[::-1])
    tmp.to_csv(out+'digits scree.csv')
    

    #%% task 4
    filtr = ImportanceSelect(rfc)
    grid ={'filter__n':dims1,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(diamondsX,diamondsY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'diamonds dim red.csv')
    
    
    grid ={'filter__n':dims2,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}  
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(digitsX,digitsY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'digits dim red.csv')
#raise

    #%% data for task 3 but find the good dim values first, use clustering script to finish up
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=1)
    
    dim = 6
    filtr = ImportanceSelect(rfc,dim)
    
    diamondsX2 = filtr.fit_transform(diamondsX,diamondsY)
    diamonds2 = pd.DataFrame(np.hstack((diamondsX2,np.atleast_2d(diamondsY).T)))
    cols = list(range(diamonds2.shape[1]))
    cols[-1] = 'Class'
    diamonds2.columns = cols
    diamonds2.to_hdf(out+'datasets.hdf','diamonds',complib='blosc',complevel=9)
    
    dim = 45
    filtr = ImportanceSelect(rfc,dim)
    digitsX2 = filtr.fit_transform(digitsX,digitsY)
    digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
    cols = list(range(digits2.shape[1]))
    cols[-1] = 'Class'
    digits2.columns = cols
    digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)

'''
    #%% extra dimension test
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=1)
    
    dim = 6
    filtr = ImportanceSelect(rfc,dim)
    
    diamondsX2 = filtr.fit_transform(diamondsX,diamondsY)
    diamonds2 = pd.DataFrame(np.hstack((diamondsX2,np.atleast_2d(diamondsY).T)))
    cols = list(range(diamonds2.shape[1]))
    cols[-1] = 'Class'
    diamonds2.columns = cols
    diamonds2.to_hdf(out+'datasets1.hdf','diamonds',complib='blosc',complevel=9)
    
    dim = 45
    filtr = ImportanceSelect(rfc,dim)
    digitsX2 = filtr.fit_transform(digitsX,digitsY)
    digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
    cols = list(range(digits2.shape[1]))
    cols[-1] = 'Class'
    digits2.columns = cols
    digits2.to_hdf(out+'datasets1.hdf','digits',complib='blosc',complevel=9)
'''