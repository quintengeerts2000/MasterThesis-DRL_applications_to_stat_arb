import pandas as pd
import numpy as np 
from numpy.linalg import eig, norm
from sklearn.linear_model import LinearRegression

def PCA(price_data:pd.DataFrame, 
                amount_of_factors:int=5,
                loadings_window_size:int=60)-> np.ndarray:
    '''
    Calculates the pca portfolio given a dataset with prices
    '''

    T, N         = price_data.shape 
    assert loadings_window_size < T, 'loading window larger than length of dataset supplied' 

    rets         = price_data.pct_change(1,fill_method=None).iloc[1:].to_numpy()
    idxsSelected = ~np.any(np.isnan(rets), axis = 0).ravel()
    if idxsSelected.sum() == 0:
            return np.zeros((N,N))
    
    rets_is     = rets[:,idxsSelected] # in sample returns: used for generating the portfolio

    # Calculate PCA
    rets_mean       = np.mean(rets_is, axis=0,keepdims=True)
    rets_vol        = np.sqrt(np.mean((rets_is-rets_mean)**2,axis=0,keepdims=True))
    rets_normalized = (rets_is - rets_mean) / rets_vol #TODO: wat als rets_vol = 0?
    Corr            = np.dot(rets_normalized.T, rets_normalized)
    _, eigenVectors = np.linalg.eigh(Corr)

    # Calculate loadings
    w           = eigenVectors[:,-amount_of_factors:].real  
    R           = rets_is[-loadings_window_size:,:]
    wtR         = R @ w  
    regr        = LinearRegression(fit_intercept=False, n_jobs=-1).fit(wtR,R)
    beta        = regr.coef_                                                    #beta
    psi         = (np.eye(beta.shape[0]) - beta @ w.T)

    # Calculate residual returns
    residual_portf = np.zeros((N,N))
    i = 0
    for idx, val in enumerate(idxsSelected):
         if val:
               residual_portf[idx,idxsSelected] = psi[i,:].reshape([1,-1])
               i += 1
    return residual_portf, idxsSelected