import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributions as tdist

import math



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers):
        super(EncoderRNN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,num_layers,batch_first=True,bidirectional=False)

    def forward(self, inputdata):
        output, hidden = self.gru(inputdata)
        return output,hidden


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,num_layers,batch_first=True,bidirectional=False)


    def forward(self, inputdata, hidden):
        output = F.relu(inputdata)
        output, hidden = self.gru(output, hidden)
        return output, hidden

class Att(nn.Module):
    def __init__(self,Option):
        super(Att,self).__init__()
        self.Option=Option
        
        self.encoderRNN1=EncoderRNN(self.Option.input_size,self.Option.hidden_size,self.Option.num_layers)
        self.decoderRNN1=DecoderRNN(self.Option.hidden_size,self.Option.hidden_size,self.Option.num_layers)
        self.decoderRNN2=DecoderRNN(self.Option.hidden_size,self.Option.hidden_size,self.Option.num_layers)
        self.decoderRNN3=DecoderRNN(self.Option.hidden_size,self.Option.hidden_size,self.Option.num_layers)
        
        
        self.out = nn.Linear(self.Option.hidden_size, self.Option.output_size)

    def forward(self,inputdata):
        midden,hidden=self.encoderRNN1(inputdata)
        midden,hidden=self.decoderRNN1(midden,hidden)
        midden,hidden=self.decoderRNN2(midden,hidden)
        midden,hidden=self.decoderRNN3(midden,hidden)


        y=torch.mean(midden,dim=1)

        y=self.out(y)
        y=torch.squeeze(y,1)
        y=torch.sigmoid(y)
        return y



class uOption():
    def __init__(self):
        self.input_size=10
        self.hidden_size=128
        self.num_layers=1
        self.output_size=1

if __name__=="__main__":
    # encoderRNN=EncoderRNN(10,128,1)
    # x=torch.rand(64,12,10)
    # output,hidden=encoderRNN(x)
    # print(output.shape,hidden.shape)

    # decoderRNN=DecoderRNN(10,128,1,1)
    # output,hidden=decoderRNN(torch.rand(64,12,10),torch.rand(1,64,128))
    # print(output.shape,hidden.shape)


    att=Att(uOption())
    y=att(torch.rand(64,12,10))
    print(y)