from osgeo import gdal
import os
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DataItem():
    def __init__(self,data,label,Option):
        self.data=data
        self.label=label
        self.Option=Option

    #数据增强
    def DataEnhence(self,label,data):

        if len(data)==70:
            data = data.reshape(self.Option.size70,order='C')
            data = data[:,[0,1,2,3,4,6,8,9]]
        elif len(data) == 56:
            data = data.reshape(self.Option.size56, order='C')
        else:
            print("[ERROR] DATAENHANCE INPUT DATA SHAPE WRONG")
        # data=data.reshape(self.Option.size,order='C')
        

        # ['blue','green', 'red','red1','red2','nir','swir1','swir2']
        # masks=self.Option.masks
        # data=data[masks[0]:masks[1],:]
        # data = np.where(data>5000,0,data)
        # red=data[:,2]
        # nir=data[:,6]
        # NDVI=(nir-red)/(nir+red)
        # data=np.c_[data,NDVI]

        return label.astype(float),data.astype(float)/10000
    def getdata(self):
        return self.DataEnhence(self.label,self.data)


class CornDataset(Dataset):
    def __init__(self,Option,datas):
        self.Option=Option
        self.datas=datas
        self.mode='train'
        self.rundata={}



    def changemode(self,mode):
        self.mode=mode
        self.rundata={}

    def __len__(self):
        return self.datas[f'{self.mode}CDL'].shape[0]
    

    def __getitem__(self,id):
        if id not in self.rundata.keys():
            self.rundata[id]=DataItem(self.datas[f'{self.mode}s2'][id,:],self.datas[f'{self.mode}CDL'][id],self.Option)
        return self.rundata[id].getdata()


    

if __name__=="__main__":
    cornDataset=CornDataset(DatasetOpt())
    print(len(cornDataset))
    print(cornDataset[0][1].shape)
    cornDataset.mode='test'
    print(len(cornDataset))

    dataloader=DataLoader(
                dataset=cornDataset,
                batch_size=32,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=False
    )
    for y,x in dataloader:
        print(y.shape)
        print(x.shape)
        break
