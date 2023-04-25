from osgeo import gdal,osr,ogr,gdal_array,gdalconst
import os
import numpy as np
import random
import pandas as pd
def process_bar(percent, start_str='', end_str='100%', total_length=0,udata=''):
    bar = '#'.join(["\033[31m%s\033[0m"%''] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str+' '+udata
    print(bar, end='', flush=True)
class MakeData():
    def __init__(self,Option):
        self.Option=Option


    def read_data(self,path):
        # Traindata=pd.read_csv(os.path.join(path,'db2019.csv'))
        # Testdata=pd.read_csv(os.path.join(path,'db2018.csv'))
        # Vailddata=pd.read_csv(os.path.join(path,'db2018.csv'))
        Traindata = pd.read_csv(self.Option.Traindata)
        Testdata = pd.read_csv(self.Option.Testdata)
        Vailddata = pd.read_csv(self.Option.Vailddata)
        # Traindata=pd.read_csv(os.path.join(path,'various_years','db20182019.csv'))
        # Testdata=pd.read_csv(os.path.join(path,'various_years','db2017.csv'))
        # Vailddata=pd.read_csv(os.path.join(path,'various_years','db2017.csv'))

        result={}
        # print(Traindata.shape)
        result['trains2']=np.array(Traindata)[:,2:]
        result['trainCDL']=np.array(Traindata['class_id'])
        # result['trainCDL'] = np.where(result['trainCDL'] != 3, 1, 0)

        result['tests2']=np.array(Testdata)[:,2:]
        result['testCDL']=np.array(Testdata['class_id'])
        # result['testCDL'] = np.where(result['testCDL'] != 3, 1, 0)

        result['vailds2']=np.array(Vailddata)[:,2:]
        result['vaildCDL']=np.array(Vailddata['class_id'])
        # result['vaildCDL'] = np.where(result['vaildCDL'] != 3, 1, 0)
        print(result['vaildCDL'])
        self.useData=result
        return self.useData


