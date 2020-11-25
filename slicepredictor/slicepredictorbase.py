import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from image_warp.imagewarp import ImgWarp
from abc import abstractmethod


class SlicePredictorBase():
    """
    待实现的方法：
    predict_patch,image_read,image_save
    """
    def __init__(self, model, crop_size, overlap_w=20,is_slice=False,padding='zeros',padding_w=50):
        self.model = model
        self.crop_size = crop_size # 切片的图像大小
        self.overlap_w = overlap_w # 交叠切片，交叠区域宽度
        self.is_slice = is_slice # 是否切片
        self.padding = padding
        self.padding_w = padding_w
    
    def pad_edge(self,im, new_w, new_h, bf=0):
        """
        把原图变大，四周填充bf的宽度
        self.padding=='mirror':镜像填充
        self.padding=='zeros':0填充
        params:
        {
            im[ndarray[h,w,c]]:3通道图像
            new_w[int]:填充后图像的宽度
            new_h[int]:填充后图像的高度
            bf[int]:交叠带的宽度
        }
        return:填充后的图像
        """
        [h, w, c] = im.shape
        new_im = np.zeros([new_h, new_w, c], dtype=np.float32)
        new_im[bf:h + bf, bf:w + bf, :] = im # padding, 中间复制，边上置零
        if self.padding=='mirror':
            dw = new_w-w-bf # 右边置零的像素数
            dh = new_h-w-bf # 下边置零的像素数
            block_lu = new_im[bf+1:2*bf+1,bf+1:2*bf+1,:]
            block_u = new_im[bf+1:2*bf+1,bf:w+bf,:]
            block_ru = new_im[bf+1:2*bf+1,w+bf-1-dw:w+bf-1,:]
            block_l = new_im[bf:h+bf,bf+1:2*bf+1,:]
            block_ld = new_im[h+bf-1-dh:h+bf-1,bf+1:2*bf+1,:]
            block_d = new_im[h+bf-1-dh:h+bf-1,bf:w+bf,:]
            block_r = new_im[bf:h+bf,w+bf-1-dw:w+bf-1,:]
            block_rd = new_im[h+bf-1-dh:h+bf-1,w+bf-1-dw:w+bf-1,:]
            new_im[0:bf,0:bf,:]=ImgWarp.rotate180(block_lu) # 左上
            new_im[0:bf,bf:w+bf,:]=ImgWarp.vertical_flip(block_u)
            new_im[0:bf,w+bf:,:]=ImgWarp.rotate180(block_ru)
            new_im[bf:bf+h,0:bf,:]=ImgWarp.horizontal_flip(block_l)
            new_im[bf+h:,0:bf,:]=ImgWarp.rotate180(block_ld)
            new_im[bf+h:,bf:w+bf,:]=ImgWarp.vertical_flip(block_d)
            new_im[bf:bf+h,w+bf:,:]=ImgWarp.horizontal_flip(block_r)
            new_im[bf+h:,w+bf:,:]=ImgWarp.rotate180(block_rd)
            # cv2.imwrite('./padimg.png', new_im)
        return new_im

    def un_pad_edge(self,im, old_w, old_h, bf=0):
        """
        从填充的图像中挖出原图像大小，输出预测结果
        params:
        {
            im[ndarray[h,w,c]]:3通道图像
            old_w[int]:原图像的宽度
            old_h[int]:原图像的高度
            bf[int]:交叠带的宽度
        }
        return:恢复出的预测结果
        """
        new_im = im[bf:old_h + bf, bf:old_w + bf]
        return new_im

    
    def slice_predict(self,t1):
        """
        切片并预测
        每一小块调用predict_patch预测
        params:
        {
            t1[nd.array]:输入的一张大图
        }
        return:cmm[ndarray]:处理完的拼成原大小的图像
        """
        dim = self.crop_size
        bf = self.overlap_w
        [h, w, c] = t1.shape

        write_dim = dim - 2 * bf # 每次预测完，写进去的大小。定义为切图大小-2*交界带
        h_batch = int(int(h + write_dim - 1) / write_dim) # 竖着切出来的数量
        w_batch = int(int(w + write_dim - 1) / write_dim) # 横着切出来的数量
        new_size = (w_batch * write_dim + 2 * bf, h_batch * write_dim + 2 * bf) # 从这个尺寸，切出来write_dim大小，能切出来batch个，并且是整数数量

        im1 = self.pad_edge(t1, int(new_size[0]), int(new_size[1]), bf)

        # =================================================================== #
        cmm = np.zeros((new_size[1], new_size[0])) 
        all_count = h_batch * w_batch
        for i in range(h_batch):
            for j in range(w_batch):
                all_count = all_count - 1
                offset_x = j * write_dim
                offset_y = i * write_dim
                t1_b = im1[offset_y:offset_y + dim, offset_x:offset_x + dim, :] # 这个是切出来的小图像
                output = self.predict_patch(t1_b)
                cmm_b = output

                cmm[offset_y + bf:offset_y + bf + write_dim, 
                    offset_x + bf:offset_x + bf + write_dim] = cmm_b[bf:bf + write_dim, bf:bf + write_dim]

        cmm = self.un_pad_edge(cmm, w, h, bf) # 拼成原图像大小的，处理完的图像

        # =================================================================== #

        return cmm

    @abstractmethod
    def predict_patch(self,data):
        '''
        预测一个小块的图像
        params:
        {
            data[ndarray]:ndarray格式的输入图像
        }
        return:预测的图像
        '''
        raise NotImplementedError
    
    def predict_one(self,input_file_path,output_dir):
        """
        预测一张图像，并存储
        params:
        {
            input_file_path[str]:输入图像的路径
        }
        """
        data = self.image_read(input_file_path)
        if self.is_slice:
            res = self.slice_predict(data) # 切片预测
        else:
            pad_width = self.padding_w
            [h, w, c] = data.shape
            im1 = self.pad_edge(data, w+2*pad_width, h+2*pad_width, pad_width)
            res = self.predict_patch(im1) # 不切片
            res = self.un_pad_edge(res, w, h, pad_width)
        
        self.image_save(input_file_path=input_file_path,output_dir=output_dir,result=res)
        

    @abstractmethod
    def image_read(self, input_file_path):
        """
        读取图像
        return:ndarray
        实现示例：
        def image_read(self, input_file_path):
            data = np.array(Image.open(input_file_path)) # 非tif文件
            return data
        """
        raise NotImplementedError

    @abstractmethod
    def image_save(self,input_file_path,output_dir,result):
        """
        存储预测结果
        示例：
        image_num = input_file_path.split('/')[-1].split('.')[0] # 图片的文件名上面的编号，用于存结果文件
        save_file_name = image_num + '.png'
        save_path =  os.path.join(output_dir,save_file_name)
        cv2.imwrite(save_path, res)
        """
        raise NotImplementedError