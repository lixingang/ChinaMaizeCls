from torch import embedding
from Option import DataOption,DatasetOption
from dataset.makedata import MakeData
from dataset.dataset import CornDataset

if __name__=='__main__':
    makeData=MakeData(DataOption())
    #makeData.randomlabel(21,21,10)
    #makeData.mergedata('/root/data/GH/area/code/work/HBmaize.csv','/root/data/GH/area/code/work/HBnomaize.csv')
    #makeData.M1('/root/data/GH/area/code/work/t.csv',21,21,200)
    # result=makeData.mcut('/root/data/GH/area/code/work/DB_pts_judge.csv')

    useData=makeData.read_data('/data/GH/GH/area/code/data/ALL/D_7_17')


    for key in useData.keys():
        print(key,useData[key].shape)

    # for key in result.keys():
    #     print(result[key].shape)

    # makeData.read_data('/root/data/GH/area/code/work/data.npy')
    #result=makeData.cutdata()
    #u=True 
    #m=True
    #for i,label in enumerate(result['testCDL']):
    #    if u==True and label==1:
    #        print(result['tests2'][i])
    #       u=False
    #    if m==True and label==0:
    #        print(result['tests2'][i])
    #        m=False
        
    # result=makeData.cutdata()
    # for key in result.keys():
    #     print(key,result[key].shape)
    #     data=result[key]
    #     if 'CDL' in key:
    #         print(result[key],sum(result[key]))
    
    # for i in range(result['vailds2'].shape[0]):
    #     print(result['vailds2'][i][0])

    # dataset=CornDataset(DatasetOption(),result)
    # #print(len(dataset),dataset[0])
