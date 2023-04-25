import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HEMLoss(nn.Module):
    
    def __init__(self, margin):
        super().__init__()
        self.mse = nn.MSELoss()
        self.margin = margin

    def forward(self, pred,real):
        cond = torch.abs(real - pred) > self.margin
        if cond.long().sum() > 0:
            real = real[cond]
            pred = pred[cond]
            return self.mse(real, pred)
        else:
            return 0.0 * self.mse(real, pred)


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=4, feat_dim=128, device='cuda:0'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device


        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))


    def forward(self, x, labels):

        if len(x.shape)==1:
            x=torch.unsqueeze(x,1)
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        

        distmat.addmm_( mat1=x, mat2=self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.device: classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        labels = (labels).int()
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss



'''
def compute_squared_EDM_method4(X):
  # 获得矩阵都行和列，因为是行向量，因此一共有n个向量
  n,m = X.shape
  # 计算Gram 矩阵
  G = np.dot(X,X.T)
  # 因为是行向量，n是向量个数,沿y轴复制n倍，x轴复制一倍
  H = np.tile(np.diag(G), (n,1))
  return np.sqrt(H + H.T - 2*G)
'''


if __name__=="__main__":
    centerLoss=CenterLoss(feat_dim=1,device='cpu')
    loss=centerLoss(torch.from_numpy(np.array([0.6,0.3,0.2])).float(),torch.from_numpy(np.array([1,0,1])))
    print(loss)

