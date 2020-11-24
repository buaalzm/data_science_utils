import torch
from torchvision import transforms
from dataloaders import custom_transforms as tr
import tifffile
import numpy as np
import os
import cv2
from PIL import Image


class SlicePredictor():
    def __init__(self, model, crop_size, overlap_w,is_slice=False):
        self.model = model
        self.crop_size = crop_size # 切片的图像大小
        self.overlap_w = overlap_w # 交叠切片，交叠区域宽度
        self.is_slice = is_slice # 是否切片
    
    def pad_edge(self,im, new_w, new_h, bf=0):
        [h, w, c] = im.shape
        new_im = np.zeros([new_h, new_w, c], dtype=np.float32)
        new_im[bf:h + bf, bf:w + bf, :] = im # padding, 中间复制，边上置零
        return new_im

    def un_pad_edge(self,im, old_w, old_h, bf=0):
        new_im = im[bf:old_h + bf, bf:old_w + bf]
        return new_im


    def test_fdcnn(self,t1):
        '''
        params:
        {
            t1[nd.array]
        }
        return:cmm[ndarray]:处理完的拼成原大小的图像
        '''
        dim = self.crop_size
        bf = self.overlap_w
        [h, w, c] = t1.shape

        # Considering the edge

        """
        # to eliminate the splicing lines caused by image blocking, the
        # test image is cropped into 224 × 224 pixel-sized blocks with
        # 12 pixels of overlap as the inputs of the net
        """
        # write_dim = 200
        write_dim = dim - 2 * bf # 每次预测完，写进去的大小。定义为切图大小-2*交界带
        h_batch = int(int(h + write_dim - 1) / write_dim) # 竖着切出来的数量
        w_batch = int(int(w + write_dim - 1) / write_dim) # 横着切出来的数量
        new_size = (w_batch * write_dim + 2 * bf, h_batch * write_dim + 2 * bf) # 从这个尺寸，切出来write_dim大小，能切出来batch个，并且是整数数量
        # new_size = (int(new_size[0]), int(new_size[1]))

        im1 = self.pad_edge(t1, int(new_size[0]), int(new_size[1]), bf)

        # =================================================================== #
        cmm = np.zeros((new_size[1], new_size[0])) 
        all_count = h_batch * w_batch
        for i in range(h_batch):
            for j in range(w_batch):
                # print("Progress->Block", all_count)
                all_count = all_count - 1
                offset_x = j * write_dim
                offset_y = i * write_dim
                # cmm_b = net(t1)
                t1_b = im1[offset_y:offset_y + dim, offset_x:offset_x + dim, :] # 这个是切出来的小图像
                # print(t1_b.shape)
                output = self.predict_patch(t1_b)
                # plt.imshow(t1_b.astype(np.uint8))
                # plt.show()
                cmm_b = output

                cmm[offset_y + bf:offset_y + bf + write_dim, 
                    offset_x + bf:offset_x + bf + write_dim] = cmm_b[bf:bf + write_dim, bf:bf + write_dim]

        cmm = self.un_pad_edge(cmm, w, h, bf) # 拼成原图像大小的，处理完的图像

        # =================================================================== #

        return cmm

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
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        # erode_img = cv2.erode(result_img, kernel)
        # dilate_img = cv2.dilate(erode_img, kernel)
        return result_img
    
    def predict_one(self,input_file_path,output_dir):
        image_num = input_file_path.split('/')[-1].split('.')[0] # 图片的文件名上面的编号，用于存结果文件
        # data = tifffile.imread(input_file_path)
        data = np.array(Image.open(input_file_path)) # 非tif文件
        if self.is_slice:
            res = self.test_fdcnn(data) # 切片预测
        else:
            res = self.predict_patch(data) # 不切片
        
        save_file_name = image_num + '.png'
        save_path =  os.path.join(output_dir,save_file_name)
        cv2.imwrite(save_path, res)