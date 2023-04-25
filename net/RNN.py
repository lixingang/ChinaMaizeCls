import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributions as tdist

import math


#MLP模块
#参数介绍:
#*embed_chs:多层感知机每层的输入
#在模型的两个部分的两种使用方式的输入层数不同
class MLP(nn.Module):
    def __init__(self,fea_len,*embed_chs):
        super().__init__()
        #MLP模块    
        a_ch=fea_len 
        layers = []
        for b_ch in embed_chs:
            fc = nn.Linear(a_ch, b_ch)
            self.init_fc(fc)
            layers.append(nn.Sequential(fc, nn.Tanh(),))
            a_ch = b_ch
        self.layers = nn.Sequential(*layers)
    
        #结果输出层
        self.zdist = tdist.Normal(0, 1)
        self.mean = nn.Linear(a_ch, 1)
        self.std = nn.Linear(a_ch, 1)
    
    @staticmethod
    def init_fc(fc):
        """
        Xavier initialization for the fully connected layer
        """        
        r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)        
        fc.weight.data.uniform_(-r, r)        
        fc.bias.data.fill_(0)        
        
    
    def forward(self,x):
        
        y=self.layers(x)
        
        #m = self.mean(y)
        #s = self.std(y)
        #z = self.zdist.sample((y.shape[0], 1)).to(y.device)
        
        #return z * s + m, m, s
        return y
#(batch_size,1581,1)->(batch_size,128)
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super().__init__()
        self.rnn=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=False)
    def forward(self,x):
        x,_=self.rnn(x)
        x=x[:,-1,:]

        return x

class LSTM_MLP(nn.Module):
    def __init__(self,Option):
        super().__init__()
        self.Option=Option
        self.RNN=LSTM(self.Option.input_size,self.Option.hidden_size,self.Option.num_layers)
        #self.MLP=MLP(128,*[128,128,64])
        self.MLP=MLP(self.Option.s,*self.Option.chs)
        self.out=nn.Linear(self.Option.chs[-1],4)

    def forward(self,x):
        #(batch_size,12,10)

        x=self.RNN(x)
        #(batch_size,128)
        mid=self.MLP(x)
        y=self.out(mid)
        y = torch.sigmoid(y)
        y=torch.squeeze(y,1)

        return y,mid


class TOption():
    def __init__(self):
        self.input_size=8
        self.hidden_size=128
        self.num_layers=1
        self.output_size=1
        self.s=128
        self.chs=[128,128,64]

if __name__=="__main__":
    #x=torch.rand([16,3,1576])
    x=torch.rand([16,12,10])
    model=LSTM_MLP(TOption())


    y,mid=model(x)
    print(y.shape,mid.shape)


