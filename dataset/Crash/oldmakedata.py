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
    def randomlabel(self,IDX,IDY,num=1):
        s2name=self.Option.s2title+str(IDX)+'_'+str(IDY)
        CDLname=self.Option.CDLtitle+str(IDX)+'_'+str(IDY)
        
        results={}
        results['data']=[]
        results['label']=[]
        
        paths=os.listdir(os.path.join(self.Option.workpath,CDLname))
        
        allnum=len(paths)
        for i,path in enumerate(paths):
            process_bar(i/allnum,start_str='',end_str='100%',total_length=0,udata='')
            if '.tif' in path:
                i=0
                CDLdata=gdal.Open(os.path.join(self.Option.workpath,CDLname,path))
                rows=CDLdata.RasterYSize
                cols=CDLdata.RasterXSize
                
                if rows!=self.Option.datasize or cols!=self.Option.datasize:
                    continue
                uCDLdata=CDLdata.ReadAsArray(0,0,cols,rows)          
                s2data=gdal.Open(os.path.join(self.Option.workpath,s2name,self.Option.s2title+path[4:]))
                us2data=s2data.ReadAsArray(0,0,cols,rows)
                while i<num: 

                    x=random.randint(0,self.Option.datasize-1)
                    y=random.randint(0,self.Option.datasize-1)
                    Tlabel=uCDLdata[x,y]  
                    Tdata=us2data[:,x,y]
                    results['data'].append(Tdata)
                    results['label'].append(Tlabel)
                    i+=1
        results['data']=np.array(results['data'])
        results['label']=np.array(results['label'])
        results['label']=np.where(results['label']==1,1,0)

        #调整正样本和负样本数量
        n=np.sum(results['label'])
        indexs0=[]
        indexs1=[]
        for i,label in enumerate(results['label']):
            if label==0:
                indexs0.append(i)
            else:
                indexs1.append(i)
        choices=np.random.choice(indexs0,n,replace=False)

        results['data']=np.append(results['data'][indexs1],results['data'][choices],axis=0)
        results['label']=np.append(results['label'][indexs1],results['label'][choices],axis=0)


        self.datas=results
        return self.datas


    def cutdata(self):
        result={}
        num=self.datas['label'].shape[0]

        cutindex=np.random.choice(num,int(num*self.Option.train_r),replace=False)

        result['trains2']=self.datas['data'][cutindex]
        result['trainCDL']=self.datas['label'][cutindex]
        
        rests2=np.delete(self.datas['data'],cutindex,0)
        restCDL=np.delete(self.datas['label'],cutindex,0)

        cutindex=np.random.choice(restCDL.shape[0],int(num*self.Option.test_r),replace=False)
        result['tests2']=rests2[cutindex]
        result['testCDL']=restCDL[cutindex]
        rests2=np.delete(rests2,cutindex,0)
        restCDL=np.delete(restCDL,cutindex,0)

        cutindex=np.random.choice(restCDL.shape[0],int(num*self.Option.vaild_r),replace=False)
        result['vailds2']=rests2[cutindex]
        result['vaildCDL']=restCDL[cutindex]

        self.useData=result
        return self.useData
    

    #读取外来数据
    def read_data(self,path1):
        if '.npy' in path1:
            datas=np.load(path1)
        elif 'csv' in path1:
            datas=np.array(pd.read_csv(path1))
        num=datas.shape[0]
        results={}
        results['data']=datas
        results['label']=np.array([0]*num)
        results['label'][0:int(num/2)]=1

        # adddata=np.load(path2)
        # deleteindex=np.random.choice(np.arange(int(num/2)),adddata.shape[0],replace=False)
        # deleteindex+=int(num/2)
        # results['data']=np.delete(results['data'],deleteindex,0)
        # results['label']=np.delete(results['label'],deleteindex,0)

        # results['data']=np.concatenate((results['data'],adddata),0)
        # results['label']=np.concatenate((results['label'],np.array([0]*adddata.shape[0])),0)



        self.datas=results

        return self.datas


    def mergedata(self,path1,path2):
        results={}
        datas1=np.array(pd.read_csv(path1))
        datas2=np.array(pd.read_csv(path2))
        datas=np.concatenate((datas1,datas2),0)
        results['data']=datas
 
        label1=np.array([1]*datas1.shape[0])
        label2=np.array([0]*datas2.shape[0])
        label=np.append(label1,label2)
        results['label']=label
        self.datas=results
        return self.datas


    #用##训练，用//验证
    def tt(self,path,classname='classL_0'):
        data=pd.read_csv(path)
        # bandname=['blue', 'green', 'red','red1','red2','red3','nir','red4','swir1', 'swir2']

        # names=[]
        # for i in range(12):
        #     for band in bandname:
        #         names.append(f'{band}_{i}')
        
        findata=data[data.keys()[2:]]
        findata=np.array(findata)
        label=np.array(data[classname])
        results={}
        results['data']=findata
        results['label']=label
        self.datas=results


        return self.datas
    
    def uu(self,path,remove=True,classname='classL_0'):
        data=pd.read_csv(path)
        bandname=['blue0', 'green0', 'red0','red1','red2','red3','nir','red4','swir1', 'swir2']
        names=[]
        for i in range(12):
            for band in bandname:
                names.append(f'{band}_{i}')
        findata=data[names]
        findata=np.array(findata)
        label=np.array(data[classname])
        if remove:
            judge=data['f']
            judge2=data['uuuu']
            removeindex=[]
            for i,abss in enumerate(judge):
                if judge2[i]==1 and abss>0.5:
                    removeindex.append(i)

            findata=np.delete(findata,removeindex,0)
            label=np.delete(label,removeindex,0)
                
        results={}
        results['data']=findata
        results['label']=label
        self.datas=results
        return self.datas


    def mcut(self,path,classname='classL_0'):
        data=pd.read_csv(path)
        bandname=['blue0', 'green0', 'red0','red1','red2','red3','nir','red4','swir1', 'swir2']
        names=[]
        for i in range(12):
            for band in bandname:
                names.append(f'{band}_{i}')
        findata=data[names]
        findata=np.array(findata)
        label=np.array(data[classname])

        judge=data['f']
        judge2=data['uuuu']
        removeindex=[]
        for i,abss in enumerate(judge):
            if judge2[i]==1 and abss>0.5:
                removeindex.append(i)

        findata=np.delete(findata,removeindex,0)
        label=np.delete(label,removeindex,0)

        udata=findata[removeindex,:]
        ulabel=label[removeindex]
        print(len(removeindex))
        self.datas={}
        self.datas['label']=label
        self.datas['data']=findata

        result={}
        num=self.datas['label'].shape[0]

        cutindex=np.random.choice(num,int(num*self.Option.train_r),replace=False)

        result['trains2']=self.datas['data'][cutindex]
        result['trainCDL']=self.datas['label'][cutindex]

        result['trains2']=np.concatenate((result['trains2'],udata),0)
        result['trainCDL']=np.concatenate((result['trainCDL'],ulabel),0)
        
        rests2=np.delete(self.datas['data'],cutindex,0)
        restCDL=np.delete(self.datas['label'],cutindex,0)

        cutindex=np.random.choice(restCDL.shape[0],int(num*self.Option.test_r),replace=False)
        result['tests2']=rests2[cutindex]
        result['testCDL']=restCDL[cutindex]
        rests2=np.delete(rests2,cutindex,0)
        restCDL=np.delete(restCDL,cutindex,0)

        cutindex=np.random.choice(restCDL.shape[0],int(num*self.Option.vaild_r),replace=False)
        result['vailds2']=rests2[cutindex]
        result['vaildCDL']=restCDL[cutindex]

        self.useData=result
        return self.useData

        
    def M1(self,path,IDX,IDY,num=1):
        datas=self.randomlabel(IDX,IDY,num)
        maizedata=np.array(pd.read_csv(path))
        indexs0=[]
        for i,label in enumerate(datas['label']):
            if label==0:
                indexs0.append(i)
        #choices=np.random.choice(indexs0,maizedata.shape[0],replace=False)
        nomaizedata=datas['data'][indexs0]
        label1=np.array([1]*maizedata.shape[0])
        label2=np.array([0]*nomaizedata.shape[0])
        results={}
        results['data']=np.concatenate((maizedata,nomaizedata),0)
        results['label']=np.append(label1,label2)
        self.datas=results
        return self.datas

    def read_ex(self,path):
        result={}
        
        exdata=pd.read_csv(path)
        inits=np.array(exdata['init_0'])
        whereinit=np.where(inits==1)[0]

        findata=np.array(exdata)
        findata=findata[:,4:]
        label=np.array(exdata['class_id'])

        result['tests2']=findata[whereinit,:]
        result['testCDL']=label[whereinit]

        findata=np.delete(findata,whereinit,0)
        label=np.delete(label,whereinit,0)

        num=findata.shape[0]
        cutindex=np.random.choice(num,int(num*self.Option.train_r),replace=False)
        result['trains2']=findata[cutindex]
        result['trainCDL']=label[cutindex]

        result['vailds2']=np.delete(findata,cutindex,0)
        result['vaildCDL']=np.delete(label,cutindex,0)

        self.useData=result
        return self.useData



    #用扩展的部分数据做训练
    def read_ex2(self,path):
        result={}
        
        exdata=pd.read_csv(path)
        inits=np.array(exdata['init_0'])
        whereinit=np.where(inits==1)[0]

        findata=np.array(exdata)
        findata=findata[:,4:]
        label=np.array(exdata['class_id'])

        result['tests2']=findata[whereinit,:]
        result['testCDL']=label[whereinit]

        findata=np.delete(findata,whereinit,0)
        label=np.delete(label,whereinit,0)

        num=findata.shape[0]
        useIndex=np.random.choice(num,int(num*0.5),replace=False)
        findata=np.delete(findata,useIndex,0)
        label=np.delete(label,useIndex,0)

        num=findata.shape[0]
        cutindex=np.random.choice(num,int(num*self.Option.train_r),replace=False)
        result['trains2']=findata[cutindex]
        result['trainCDL']=label[cutindex]

        result['vailds2']=np.delete(findata,cutindex,0)
        result['vaildCDL']=np.delete(label,cutindex,0)

        self.useData=result
        return self.useData

        #用扩展的部分数据做训练
    def read_ex3(self,initpath,EXpath):
        result={}
        
        initdata=pd.read_csv(initpath)
        findata=np.array(initdata)
        findata=findata[:,4:]
        label=np.array(initdata['class_id'])

        result['tests2']=findata
        result['testCDL']=label

        exdata=pd.read_csv(EXpath)
        findata=np.array(exdata)
        findata=findata[:,2:]
        label=np.array(exdata['class_id'])

        num=findata.shape[0]
        cutindex=np.random.choice(num,int(num*self.Option.train_r),replace=False)
        result['trains2']=findata[cutindex]
        result['trainCDL']=label[cutindex]

        result['vailds2']=np.delete(findata,cutindex,0)
        result['vaildCDL']=np.delete(label,cutindex,0)

        self.useData=result
        return self.useData


    def data_combine(self,paths,classname='class_id'):
        bandname=['blue0', 'green0', 'red0','red1','red2','red3','nir','red4','swir1', 'swir2']
        names=[]
        for i in range(12):
            for band in bandname:
                names.append(f'{band}_{i}')

        datas={}
        labels={}
        for i,path in enumerate(paths):
            data=pd.read_csv(path)
            findata=data[names]
            findata=np.array(findata)
            finlabel=np.array(data[classname])
            datas[i]=findata
            labels[i]=finlabel
        
        adata=None
        alabel=None
        keys=datas.keys()
        for key in keys:
            adata=datas[key] if adata is None else np.concatenate((adata,datas[key]),0)
            alabel=labels[key] if alabel is None else np.concatenate((alabel,labels[key]),0)

        # DBdata=pd.read_csv('/root/data/GH/area/code/work/DB_pts.csv')
        # HBdata=pd.read_csv('/root/data/GH/area/code/work/hebei_pts.csv')

        # finDBdata=DBdata[names]
        # finDBdata=np.array(finDBdata)
        # finHBdata=HBdata[names]
        # finHBdata=np.array(finHBdata)
        # DBlabel=np.array(DBdata['classL_0'])
        # HBlabel=np.array(HBdata['class_id'])

        # findata=np.concatenate((finDBdata,finHBdata),0)
        # label=np.concatenate((DBlabel,HBlabel),0)
        results={}
        results['data']=adata
        results['label']=alabel
        self.datas=results
        return self.datas
