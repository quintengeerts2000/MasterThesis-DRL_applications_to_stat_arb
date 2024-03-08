import torch
import torch.nn as nn
from collections import deque
import random
import numpy as np
import pandas as pd

class FourierExtractor():
    def __init__(self, signal_window:int=30) -> None:
        self.signal_window = signal_window
    
    def reset(self):
        None
    
    def train(self,train_data=pd.DataFrame):
        None

    def extract(self,residuals_data:pd.DataFrame):
        '''
        All the data input in this function should be considered in sample
        '''
        N, L = residuals_data.shape
        assert L == self.signal_window, "can't calculate fourier transform for more than the input amount of data"
        res_window = (residuals_data + 1).cumprod(axis=1) - 1
        Fourier    = np.fft.rfft(res_window,axis=1)
        n_f        = Fourier.shape[1]
        out        = np.zeros((N,n_f*2-2))
        out[:,:n_f]= np.real(Fourier)
        out[:,n_f:]= np.imag(Fourier[:,1:-1])
        return out.astype(float)
    
class CumsumExtractor():
    def __init__(self, signal_window:int=30) -> None:
        self.signal_window = signal_window
    
    def reset(self):
        None
    
    def train(self,train_data=pd.DataFrame):
        None

    def re_train(self, **kwargs):
        None

    def extract(self,residuals_data:pd.DataFrame):
        '''
        All the data input in this function should be considered in sample
        '''
        N, L = residuals_data.shape
        assert L == self.signal_window, "can't calculate fourier transform for more than the input amount of data"
        out = (residuals_data + 1).cumprod(axis=1) - 1
        return out
    
class CNNTransformerExtractor():
    def __init__(self, signal_window:int=60) -> None:
        self.signal_window   = signal_window
        # initializing the model
        self.optimizer_opts = {'lr':0.01}
        self.mse       = nn.MSELoss()
        self.losses    = deque(maxlen=200)
        self.loss_graph = list()
        self.reset()

    def reset(self):
        self.model = CNNTransformer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_opts)
    
    def train(self, train_data=pd.DataFrame):
        self.model.train()
        shuffled_idxs = random.sample(list(range(self.signal_window + 1,len(train_data)-1)), len(list(range(self.signal_window + 1,len(train_data)-1))))
        # training loop for training the vision transformer model
        for idx in shuffled_idxs:
            window    = train_data.iloc[idx - self.signal_window: idx+1].values.astype(float)
            idxsSelected = ~np.any(np.isnan(window), axis = 0).ravel()
            if idxsSelected.sum() == 0:
                continue
            inputVars = torch.FloatTensor((window[:-1,idxsSelected]+1).T.cumprod(axis=1))
            target    = torch.FloatTensor(window[-1,idxsSelected].T)

            pred      = self.model.forward(inputVars)
            loss      = self.mse(pred,target)
            self.losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print('Finished training the prediction module')
        self.model.eval()
    
    def re_train(self, train_data=pd.DataFrame, sample_size:int=64):
        self.model.train()
        idxs = list(range(self.signal_window + 1,len(train_data)-1))
        shuffled_idxs = random.sample(idxs, min(len(idxs),sample_size))
        # training loop for training the vision transformer model
        for idx in shuffled_idxs:
            window    = train_data.iloc[idx - self.signal_window: idx+1].values.astype(float)
            idxsSelected = ~np.any(np.isnan(window), axis = 0).ravel()
            if idxsSelected.sum() == 0:
                continue
            inputVars = torch.FloatTensor((window[:-1,idxsSelected]+1).T.cumprod(axis=1))
            target    = torch.FloatTensor(window[-1,idxsSelected].T)

            pred      = self.model.forward(inputVars)
            loss      = self.mse(pred,target)
            self.losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.model.eval()
    
    def predict(self, residuals_data):
        res_window = torch.FloatTensor((residuals_data + 1).cumprod(axis=1).astype(float))
        return self.model(res_window).detach().numpy()
    
    def extract(self, residuals_data):
        if type(residuals_data) == torch.Tensor:
            residuals_data = residuals_data.detach().numpy()
        res_window = torch.FloatTensor((residuals_data + 1).cumprod(axis=1).astype(float))
        return self.model.extr_sig(res_window).detach().numpy()

### deep learning model code below ###

class CNN_Block(nn.Module):
    def __init__(self, in_filters=1, out_filters=8, normalization=True, filter_size=2):
        super(CNN_Block, self).__init__()  
        self.in_filters = in_filters
        self.out_filters = out_filters
        
        self.conv1 = nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=filter_size,
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=out_filters, out_channels=out_filters, kernel_size=filter_size,
                                    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)
        self.left_zero_padding = nn.ConstantPad1d((filter_size-1,0),0)
        
        self.normalization1 = nn.InstanceNorm1d(in_filters)
        self.normalization2 = nn.InstanceNorm1d(out_filters)
        self.normalization = normalization
       
    def forward(self, x): #x and out have dims (N,C,T) where C is the number of channels/filters
        if self.normalization:
            x = self.normalization1(x)
        out = self.left_zero_padding(x)
        out = self.conv1(out)
        out = self.relu(out)
        if self.normalization: 
            out = self.normalization2(out)
        out = self.left_zero_padding(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out + x.repeat(1,int(self.out_filters/self.in_filters),1)   
        return out
    
class CNNTransformer(nn.Module):
    def __init__(self, 
                 random_seed = 0, 
                 device = "cpu", # other options for device are e.g. "cuda:0"
                 normalization_conv = True, 
                 filter_numbers = [1,8], #8
                 attention_heads = 4, 
                 use_convolution = True,
                 hidden_units = 2*8, #8
                 hidden_units_factor = 2,
                 dropout = 0.25, 
                 filter_size = 2, 
                 use_transformer = True):
        
        super(CNNTransformer, self).__init__()
        if hidden_units and hidden_units_factor and hidden_units != hidden_units_factor * filter_numbers[-1]:
            raise Exception(f"`hidden_units` conflicts with `hidden_units_factor`; provide one or the other, but not both.")
        if hidden_units_factor:
            hidden_units = hidden_units_factor * filter_numbers[-1]
        self.random_seed = random_seed 
        torch.manual_seed(self.random_seed)
        self.device = torch.device(device)
        self.filter_numbers = filter_numbers
        self.use_transformer = use_transformer
        self.use_convolution = use_convolution and len(filter_numbers) > 0
        self.is_trainable = True
        
        self.convBlocks = nn.ModuleList()
        for i in range(len(filter_numbers)-1):
            self.convBlocks.append(
                CNN_Block(filter_numbers[i],filter_numbers[i+1],normalization=normalization_conv,filter_size=filter_size))
        self.encoder = nn.TransformerEncoderLayer(d_model=filter_numbers[-1], nhead=attention_heads, dim_feedforward=hidden_units, dropout=dropout)
        self.linear = nn.Linear(filter_numbers[-1],1)
        #self.softmax = nn.Sequential(nn.Linear(filter_numbers[-1],num_classes))#,nn.Softmax(dim=1))
                 
    def forward(self,x): #x has dimension (N,T)
        N,T = x.shape
        x = x.reshape((N,1,T))  #(N,1,T)
        if self.use_convolution:
            for i in range(len(self.filter_numbers)-1):
                x = self.convBlocks[i](x) #(N,C,T), C is the number of channels/features
        x = x.permute(2,0,1)
        if self.use_transformer:
            x = self.encoder(x) #the input of the transformer is (T,N,C)
        return self.linear(x[-1,:,:]).squeeze() #this outputs the weights #self.softmax(x[-1,:,:]) #(N,num_classes)
    
    def extr_sig(self,x): #x has dimension (N,T)
        if len(x.shape) == 3:
            _,N,T = x.shape
        else:
            N,T = x.shape
        x = x.reshape((N,1,T))  #(N,1,T)
        if self.use_convolution:
            for i in range(len(self.filter_numbers)-1):
                x = self.convBlocks[i](x) #(N,C,T), C is the number of channels/features
        x = x.permute(2,0,1)
        if self.use_transformer:
            x = self.encoder(x) #the input of the transformer is (T,N,C)
        return x[-1,:,:] 

