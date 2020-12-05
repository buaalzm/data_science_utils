import torch
import numpy as np
from ipdb import set_trace


class LabelMapping(object):
    """
    label与颜色的映射
    """
    def __init__(self,n_classes = 6):
        self.n_classes = n_classes
        self.colormap=[
            [255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0], [0, 0, 0]
            #   white         dark blue    light blue        green         yellow          red
            #Impervious surface  buildings  Low vegetation   Tree            car         clutter
        ]
    
    def label2color(self, label_mask):
        """
        将label图映射成颜色图
        params:
        {
            label_mask[ndarray(B,H,W)]:
        }
        """
        if isinstance(label_mask,torch.Tensor):
            label_mask = label_mask.cpu().detach().numpy()
        assert isinstance(label_mask,np.ndarray)
        # set_trace()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = self.colormap[ll][0]
            g[label_mask == ll] = self.colormap[ll][1]
            b[label_mask == ll] = self.colormap[ll][2]
        rgb = np.zeros((label_mask.shape[0],3,label_mask.shape[1],label_mask.shape[2]))
        rgb[:, 0,:, :] = r / 255.0
        rgb[:, 1,:, :] = g / 255.0
        rgb[:, 2,:, :] = b / 255.0
        return rgb

    def ACC_map(self,predict,target):
        """
        比较predict的和ground truth
        将预测对的标记为绿色，错的标记为红色
        params:
        {
            predict[torch.Tensor/ndarray[B,H,W]]
            target[torch.Tensor/ndarray[B,H,W]]
        }
        """
        if isinstance(predict,torch.Tensor):
            predict = predict.cpu().detach().numpy()
        if isinstance(target,torch.Tensor):
            target = target.cpu().detach().numpy()
        acc = np.abs(predict-target)
        acc = acc.astype(np.uint8)
        
        r = predict.copy()
        g = predict.copy()
        b = np.zeros_like(predict)
        r[acc!=0]=[1.0]
        g[acc==0]=[1.0]
        r[acc==0]=[0.0]
        g[acc!=0]=[0.0]
        # set_trace()
        rgb = np.zeros((predict.shape[0],3,predict.shape[1],predict.shape[2]))
        rgb[:, 0,:, :] = r
        rgb[:, 1,:, :] = g
        rgb[:, 2,:, :] = b
        return rgb
