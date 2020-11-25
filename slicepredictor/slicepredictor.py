import torch
from torchvision import transforms
import sys
from dataloaders import custom_transforms as tr
import tifffile
import numpy as np
import os
import cv2
from PIL import Image
sys.path.append(os.path.dirname(__file__))
from slicepredictorbase import SlicePredictorBase


class SlicePredictorRS(SlicePredictorBase):
    def predict_patch(self,data):
        '''
        预测一个小块的图像
        params:
        {
            data[ndarray]:ndarray格式的输入图像
        }
        '''
        composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
        data_shape = (data.shape[0:2])
        sample = {'image': data, 'label': data}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = self.model(tensor_in)
        
        result_img = torch.max(output, 1)[1].detach().cpu().numpy() #*100
        result_img = result_img.reshape(*data_shape).astype(np.uint8)
        result_img[(result_img <=1)|(result_img >=17)] =17  # 替换异常预测值为17
        return result_img
    
    def image_read(self, input_file_path):
        data = np.array(Image.open(input_file_path)) # 非tif文件
        return data

    def image_save(self,input_file_path,output_dir,result):
        image_num = input_file_path.split('/')[-1].split('.')[0] # 图片的文件名上面的编号，用于存结果文件
        save_file_name = image_num + '.png'
        save_path =  os.path.join(output_dir,save_file_name)
        cv2.imwrite(save_path, result)