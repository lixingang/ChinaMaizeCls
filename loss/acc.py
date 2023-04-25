import math
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score

class Accuracy():
    def __init__(self):
        self.loss=0

        #将正类预测为正类数
        self.TP=0
        #将正类预测为负类数
        self.FN=0
        #将负类预测为正类
        self.FP=0
        #将负类预测为负类数
        self.TN=0



    def getCC(self):
        return self.TP,self.FN,self.FP,self.TN

    def reinit(self):
        self.TP=0
        self.FN=0
        self.FP=0
        self.TN=0

    #准确率
    def accuracy(self):
        return (self.TP+self.TN)/(self.TP+self.FN+self.FP+self.TN+1e-6)

    #精确率
    def precision(self):
        return (self.TP)/(self.TP+self.FP+1e-6)

    #召回率
    def recall(self):
        return self.TP/(self.TP+self.FN+1e-6)
    
    #F1
    def F1(self):
        return 2*self.TP/(2*self.TP+self.FP+self.FN+1e-6)

    def update(self,y,outputs):

        outputs=outputs.argmax(dim=1)

        num=y.shape[0]

        for i in range(num):
            if y[i]==1 and outputs[i]==1:
                self.TP+=1
            elif y[i]==1 and outputs[i]==0:
                self.FN+=1
            elif y[i]==0 and outputs[i]==1:
                self.FP+=1
            elif y[i]==0 and outputs[i]==0:
                self.TN+=1
    
    def getacc(self):
        return self.accuracy(),self.precision(),self.recall(),self.F1()

if __name__=="__main__":
    Acc=Accuracy()
    Acc.update(torch.from_numpy(np.array([1.0,0.0,1.0,0.0])),torch.from_numpy(np.array([0.3,0.5,0.6,0.7])))
    print(Acc.TP,Acc.TN,Acc.FN,Acc.FP)