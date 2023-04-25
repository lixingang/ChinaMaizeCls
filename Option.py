inputsize=8

class DataOption():
    def __init__(self):
        #训练数据基于某个区块
        self.workpath='/root/maize/samples/dongbei/2019'
        self.s2title='s2_'
        self.CDLtitle='CDL_'
        self.datasize=512

        self.train_r=0.7
        self.test_r=0.15
        self.vaild_r=0.15

        self.Traindata = "/root/d12t/maize_cls/code_li/data/various_years/db2018_4.csv"
        self.Testdata = "/root/d12t/maize_cls/code_li/data/various_years/db2019_4.csv"
        self.Vailddata = "/root/d12t/maize_cls/code_li/data/various_years/db2019_4.csv"


class DatasetOption():
    def __init__(self):
        self.name='corn'
        #12*10
        self.size70=(7,10)
        self.size56=(7,8)
        self.mask=True
        # self.masks=[0,3]
        # self.mask120 = [3,7]



#训练参数设置
class TrainOption():
    def __init__(self):
        #HB:500/DB:20
        self.batch_size=50
        self.num_workers=10
        self.num_epochs=150

        self.device='cuda:1'
        #HB:0.001
        self.lr=0.001
        self.scheduler={'step_size':20,'gamma': 0.1}

        self.miou_thr=0.5
        self.rootdir='logs'
        self.workdir='T1'

        self.thres=0.5
        self.cthres=0.8

        self.c=0.01

        #模型参数
        #input_size和size的第二维相同
        self.input_size=inputsize
        self.hidden_size=128
        self.num_layers=1
        self.output_size=1
        self.s=128
        self.chs=[128,128,64]

        

class GlobalOption():
    def __init__(self):
        self.name='corn'
        self.IDX=21
        self.IDY=21
        self.num=100

if __name__ =='__main__':
    d = DatasetOption()
    print(d.maskband)
