import math
import torch
import numpy as np
import sklearn.metrics as sm
# from sklearn.metrics import DDD
# from sklearn.metrics import confusion_matrix

class Accuracy():
    def __init__(self):
        self.loss = 0
        self.epoch_y = None
        self.epoch_outputs = None


    def getCC(self):
        # print(self.epoch_y, self.epoch_outputs)
        return sm.confusion_matrix(self.epoch_y,self.epoch_outputs, labels=[0,1,2,3])

    def reinit(self):
        self.epoch_y = None
        self.epoch_outputs = None

    # 准确率
    def accuracy(self):
        return sm.accuracy_score(self.epoch_y, self.epoch_outputs, )

    # 精确率
    def precision(self):
        return sm.precision_score(self.epoch_y, self.epoch_outputs, labels=[0,1,2 ], average="macro")

    # 召回率
    def recall(self):
        return sm.recall_score(self.epoch_y, self.epoch_outputs, labels=[0,1,2 ],  average="macro")

    # F1
    def F1(self):
        return sm.f1_score(self.epoch_y, self.epoch_outputs, labels=[0,1,2 ], average="macro")

    def update(self, y, outputs):
        y = y.detach().cpu().numpy()
        outputs = outputs.argmax(1)
        outputs = outputs.detach().cpu().numpy()
        self.epoch_y = y if self.epoch_y is None else np.concatenate([self.epoch_y, y] )

        # outputs = np.argmax(outputs, axis=1)
        self.epoch_outputs = outputs if self.epoch_outputs is None else np.concatenate([self.epoch_outputs, outputs] )

    def getacc(self):
        return self.precision(), self.precision(), self.recall(), self.F1()


if __name__ == "__main__":
    Acc = Accuracy()
    Acc.update(torch.from_numpy(np.array([0,1,2,3,1,1,2,3,3,3,3])), torch.from_numpy(np.array([0,1,1,1,1,2,2,3,3,3,3])))
    print(Acc.epoch_y,Acc.epoch_outputs)
    print(Acc.getacc())
    print(Acc.getCC())