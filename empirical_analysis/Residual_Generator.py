import pandas as pd
import numpy as np 
from numpy.linalg import eig, norm
from sklearn.linear_model import LinearRegression
import datetime as dt

from ipca import InstrumentedPCA

class PCA:
    def __init__(self, price_data:pd.DataFrame, 
                 amount_of_factors:int=5,
                loadings_window_size:int=30,
                lookback_window_size:int=252)-> np.ndarray:
        
        self.price_data = price_data
        self.n_factors  = amount_of_factors
        self.loadings_window_size = loadings_window_size
        self.lookback_window_size = lookback_window_size
    
    def step(self, date:dt.datetime):
        self._calculate_weights(date)
        return self.residual_portf, self.idxsSelected

    def _calculate_weights(self, date:dt.datetime):
        '''
        Calculates the pca portfolio given a dataset with prices
        '''
        #price_data = self.price_data[(self.price_data.index <= date) & (self.price_data.index >= date - dt.timedelta(self.lookback_window_size))]
        price_data = self.price_data[(self.price_data.index <= date)].tail(self.lookback_window_size)


        T, N         = price_data.shape 
        assert self.loadings_window_size < T, 'loading window larger than length of dataset supplied' 

        #rets         = price_data.pct_change(1,fill_method=None).iloc[1:].to_numpy()
        rets         = price_data.to_numpy()
        idxsSelected = ~np.any(np.isnan(rets), axis = 0).ravel()
        if idxsSelected.sum() == 0:
                return np.zeros((N,N))
        
        rets_is     = rets[:,idxsSelected] # in sample returns: used for generating the portfolio
        if self.n_factors == 0:
            psi         = np.eye(idxsSelected.sum())
        else:
            # Calculate PCA
            rets_mean       = np.mean(rets_is, axis=0,keepdims=True)
            rets_vol        = np.sqrt(np.mean((rets_is-rets_mean)**2,axis=0,keepdims=True))
            rets_normalized = (rets_is - rets_mean) / rets_vol #TODO: wat als rets_vol = 0?
            Corr            = np.dot(rets_normalized.T, rets_normalized) / self.lookback_window_size
            _, eigenVectors = np.linalg.eigh(Corr)

            # Calculate loadings
            w           = eigenVectors[:,-self.n_factors:].real  
            R           = rets_is[-self.loadings_window_size:,:]
            wtR         = R @ w  
            regr        = LinearRegression(fit_intercept=False, n_jobs=-1).fit(wtR,R)
            beta        = regr.coef_                                                    #beta
            psi         = (np.eye(beta.shape[0]) - beta @ w.T)

        # reshape the loadings matrix
        residual_portf = np.zeros((N,N))
        i = 0
        for idx, val in enumerate(idxsSelected):
            if val:
                residual_portf[idx,idxsSelected] = psi[i,:].reshape([1,-1])
                i += 1
        self.residual_portf = residual_portf
        self.idxsSelected   = idxsSelected

class IPCA:
    def __init__(self, monthly_panel:pd.DataFrame, permnos, train_window:int=48, n_factors:int=5, retrain_period:int=252) -> None:
        self.permnos      = permnos
        self.train_window = train_window
        self.retrain_period= retrain_period
        self.n_factors    = n_factors
        self.instruments  = monthly_panel[monthly_panel.columns.difference(['TICKER', 'CUSIP', 'RET'])]
        self.returns      = monthly_panel[['RET','PERMNO','date']]
        self.dates        = pd.to_datetime(monthly_panel.date.sort_values().unique())
        self.train_date   = None
        self.rebalance_date  = None
        self.residual_portf, self.idxsSelected = None, None

    def step(self, date:dt.datetime):
        # check if the model needs to be refit
        if self.train_date == None:
            self._fit_model(date=date)
            self.train_date  = date
        elif date - self.train_date > dt.timedelta(days=self.retrain_period):
            self._fit_model(date=date)
            self.train_date = date
        
        #check if new weights can be used
        if self.rebalance_date == None:
             self._calculate_weights(date=date)
        elif self._find_nearest_date(date) > self.rebalance_date:
            self._calculate_weights(date=date)
        assert self.residual_portf is not None and self.idxsSelected is not None
        return self.residual_portf, self.idxsSelected
    
    def _fit_model(self, date:dt.datetime) -> None:
        start, end = self._find_start_end(date, diff=self.train_window)
        X = self.instruments[(self.instruments.date <= end) & (self.instruments.date >= start)]
        y = self.returns[(self.returns.date <= end) & (self.returns.date >= start)]
        X = X.set_index(['PERMNO', 'date']).replace(['C','B'],np.nan).astype(float)
        y = y.set_index(['PERMNO', 'date']).replace(['C','B'],np.nan).astype(float).squeeze()
        self.regr = InstrumentedPCA(n_factors=self.n_factors, intercept=False)
        self.regr = self.regr.fit(X=X, y=y, quiet=True)
        self.Gamma, self.Factors = self.regr.get_factors(label_ind=True)
    
    def _calculate_weights(self, date:dt.datetime) -> np.ndarray:
        G   = self.Gamma.values
        C = self.instruments[self.instruments.date == self._find_nearest_date(date)]
        C = C.drop(['date'],axis=1)
        C = C.dropna(axis=0)
        selected_permnos = C.PERMNO # not sure if this is enough to ensure that the right stocks are associated with the right weights
        C = C.drop(['PERMNO'],axis=1).values.astype(float)
        psi  = np.linalg.inv(G.T @ C.T @ C @ G) @ G.T @ C.T 
        beta = C @ G
        weights = np.eye(C.shape[0]) - (beta @ psi)
        out = pd.DataFrame(index=self.permnos, columns=self.permnos).fillna(0)
        out.loc[selected_permnos, selected_permnos] = weights
        self.residual_portf =  out.values
        self.idxsSelected = np.array([company in selected_permnos.values for company in self.permnos])
        assert np.all((out.sum(axis=0).values !=0) == self.idxsSelected)
    
    def _find_start_end(self, date:dt.datetime, diff:int) -> dt.datetime:
        return self.dates[self.dates < date][-diff], self.dates[self.dates < date][-1]
    
    def _find_nearest_date(self, date:dt.datetime) -> dt.datetime:
        return self.dates[self.dates < date][-1]

def PCA_old(price_data:pd.DataFrame, 
                amount_of_factors:int=5,
                loadings_window_size:int=60)-> np.ndarray:
    '''
    Calculates the pca portfolio given a dataset with prices
    '''

    T, N         = price_data.shape 
    assert loadings_window_size < T, 'loading window larger than length of dataset supplied' 

    #rets         = price_data.pct_change(1,fill_method=None).iloc[1:].to_numpy()
    rets         = price_data.to_numpy()
    idxsSelected = ~np.any(np.isnan(rets), axis = 0).ravel()
    if idxsSelected.sum() == 0:
            return np.zeros((N,N))
    
    rets_is     = rets[:,idxsSelected] # in sample returns: used for generating the portfolio
    if amount_of_factors == 0:
        psi         = np.eye(idxsSelected.sum())
    else:
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

    # reshape the loadings matrix
    residual_portf = np.zeros((N,N))
    i = 0
    for idx, val in enumerate(idxsSelected):
         if val:
               residual_portf[idx,idxsSelected] = psi[i,:].reshape([1,-1])
               i += 1
    return residual_portf, idxsSelected