import os
import sys

# sys.path.append(os.getcwd())

import logging
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR,LambdaLR


from dataset.dataset import CornDataset
from dataset.makedata import MakeData
from net.RNN import LSTM_MLP
from net.Attention import Att
from Option import DataOption,TrainOption,DatasetOption,GlobalOption
from loss.acc_multi import Accuracy
import numpy as np
from loss.loss import CenterLoss
from pic.fig import Draw


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)



logging.basicConfig(
    # filename='new.log', filemode='w',
    format='%(asctime)s  %(levelname)-10s %(processName)s  %(name)s \033[0;33m%(message)s\033[0m',
    datefmt="%Y-%m-%d-%H-%M-%S",
    level=logging.DEBUG if os.environ.get("DEBUG") is not None else logging.INFO
)

def process_bar(percent, start_str='', end_str='100%', total_length=0,udata=''):
    bar = '#'.join(["\033[31m%s\033[0m"%''] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str+' '+udata
    # print(bar, end='', flush=True)

def get_dataloader(Option,makeData):
    cornDataset=CornDataset(DatasetOption(),makeData.useData)
    cornDataset.mode='train'
    logging.info(f'trainsize={len(cornDataset)}')
    cornDataset.mode='vaild'
    logging.info(f'vaildsize={len(cornDataset)}')
    cornDataset.mode='test'
    logging.info(f'testsize={len(cornDataset)}')
    dataloader=DataLoader(
                dataset=cornDataset,
                batch_size=Option.batch_size,
                shuffle=True,
                num_workers=Option.num_workers,
                pin_memory=True,
                drop_last=False
    )
    return dataloader


def train(Option,makeData):

    draw=Draw(Option.workdir)

    Acc=Accuracy()

    device = torch.device(Option.device)
    #model
    model=LSTM_MLP(Option)
    model=model.to(device)


    num_epochs=Option.num_epochs
    #xiugai
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion=CenterLoss(num_classes=2, feat_dim=64,device=Option.device)
    
    #optimizer = optim.Adam(model.parameters(),lr=Option.lr,weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(),lr=Option.lr)
    #scheduler = StepLR(optimizer, **Option.scheduler)
    
    # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',factor=0.8,patience=10,verbose=True,
    # threshold=0.01,threshold_mode='abs',cooldown=0,min_lr=0,eps=1e-8)

    #固定为0.001 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)


    dataloader= get_dataloader(Option,makeData)

    dataset_size=len(dataloader.dataset)


    maxF1={'vaild':-99,'train':-99}
    logging.info('----- start train -----')
    for epoch in range(num_epochs):
        


        epoch_loss=0
        epoch_miou=0
        step=0

        #验证集
        model.eval()
        dataloader.dataset.changemode('vaild')
        dataset_size=len(dataloader.dataset)

        Acc.reinit()
        step=0
        for y,x in dataloader:

            x=x.to(device).float()
            y=y.to(device).float()

            outputs,mid=model(x)
            Acc.update(y,outputs)
            step+=1
            process_bar(step/(dataset_size//Option.batch_size),start_str='vaild:')
        # print("\n",Acc.getCC())
        accuracy,precision,recall,F1=Acc.getacc()
        if F1>maxF1['vaild']:
            maxF1['vaild']=F1
            torch.save(model,os.path.join(Option.workdir,'model.pkl'))   
        print('')
        logging.info(f'[vaild] {epoch+1}/{num_epochs} F1={F1:.4f} acc={accuracy:.4f} pre={precision:.4f} recall={recall:.4f}')
        
        #记录test的F1
        draw.update(F1,'testF1')

        vaildF1=F1

        #训练
        model.train()
        dataloader.dataset.changemode('train')
        dataset_size=len(dataloader.dataset)
        epoch_loss=0
        
        Acc.reinit()
        Acc.thres=Option.thres
        step=0
        for y,x in dataloader:
            optimizer.zero_grad()
            x=x.to(device).float()
            y=y.to(device).float()
            
            outputs,mid=model(x)
            loss=criterion1(outputs,y.long())+Option.c*criterion(mid,y)
            loss.backward()

            optimizer.step()
            
            epoch_loss+=loss.item()
            Acc.update(y,outputs)
            step+=1
            process_bar(step/(dataset_size//Option.batch_size),start_str='train:',udata=f'loss:{loss:.4f}')
        # print("\n",Acc.getCC())
        accuracy,precision,recall,F1=Acc.getacc()
        if F1>maxF1['train']:
            maxF1['train']=F1
        print('') 
        logging.info(f'[train] {epoch+1}/{num_epochs} epoch_loss={epoch_loss/step:.4f} F1={F1:.4f} acc={accuracy:.4f} pre={precision:.4f} recall={recall:.4f}')

        #scheduler.step(vaildF1)
        scheduler.step()
        ulr=optimizer.state_dict()['param_groups'][0]['lr']
        # print(ulr)
        #记录test的F1
        draw.update_mui([F1,epoch_loss/step,ulr],['trainF1','loss','lr'])
        
        
        #测试
        if epoch % 5 ==0:
            testmodel=torch.load(os.path.join(Option.workdir,'model.pkl'))
            testmodel.eval()
            dataloader.dataset.changemode('test')
            dataset_size=len(dataloader.dataset)
            Acc.reinit()
            step=0
            for y,x in dataloader:
                x=x.to(device).float()
                y=y.to(device).float()
                outputs,mid=testmodel(x)

                Acc.update(y,outputs)
                step+=1
                process_bar(step/(dataset_size//Option.batch_size),start_str='test:')
            accuracy,precision,recall,F1=Acc.getacc()
            print(Acc.getCC())
            print('')
            logging.info(f'[test] {epoch+1}/{num_epochs} F1={F1:.4f} acc={accuracy:.4f} pre={precision:.4f} recall={recall:.4f}')

#             draw.mpic('lr_loss.png','loss','lr')
#             draw.mpic('TT_F1.png','trainF1','testF1')



    #测试
    model=torch.load(os.path.join(Option.workdir,'model.pkl'))
    model.eval()
    dataloader.dataset.changemode('test')
    dataset_size=len(dataloader.dataset)
    Acc.reinit()
    step=0
    for y,x in dataloader:
        x=x.to(device).float()
        y=y.to(device).float()
        outputs,mid=model(x)

        Acc.update(y,outputs)
        step+=1
        process_bar(step/(dataset_size//Option.batch_size),start_str='test:')
    epoch_miou=epoch_miou/step
    accuracy,precision,recall,F1=Acc.getacc()
    print(Acc.getCC())
    print('')
    logging.info(f'[test] {epoch+1}/{num_epochs} F1={F1:.4f} acc={accuracy:.4f} pre={precision:.4f} recall={recall:.4f}')

    draw.mpic('lr_loss.png','loss','lr')
    draw.mpic('TT_F1.png','trainF1','testF1')

    

def createpath(name,rootdir='logs'):
    uname=os.path.join(rootdir,name)
    subprocess.call(['rm','-rf',uname])
    os.mkdir(uname)
    return uname



if __name__=='__main__':

    setup_seed(2)
    #sys.argv[1]:保存路径

    #全局设置
    globalOpt=GlobalOption()

    #数据切分为训练集和测试集
    makeData=MakeData(DataOption())
    #河北数据训练
    #makeData.read_data('/root/data/GH/area/code/work/data.npy')  
    #东北数据训练
    #0.93
    #makeData.tt('/root/data/GH/area/code/data/dongbei_ex_pts.csv','class_id')
    #判别式裁剪
    #makeData.uu('/root/data/GH/area/code/work/DB_pts_judge.csv')
    #makeData.uu('/root/data/GH/area/code/work/hebei_pts.csv',False,'class_id')
    
    #数据聚合
    #makeData.data_combine(['/root/data/GH/area/code/data/hebei_pts.csv','/root/data/GH/area/code/data/henan.csv'])


    # makeData.read_data('/data/GH/GH/area/code/data/ALL/D_7_25')
    makeData.read_data("/root/d12t/maize_cls/code_li/data/")


    #makeData.cutdata()
    
    #makeData.read_ex('/root/data/GH/area/code/data/expandDB_DE.csv')
    #makeData.read_ex('/root/data/GH/area/code/data/expandDB.csv')
    #makeData.read_ex3('/data/GH/GH/area/code/data/initDB.csv','/data/GH/GH/area/code/data/DongbeiMaize.csv')
    #更新工作路径
    trainOpt=TrainOption()
    trainOpt.workdir=createpath(sys.argv[1],trainOpt.rootdir)
    #进行训练
    train(trainOpt,makeData)

    os.system(f"cp Option.py {trainOpt.workdir}")
