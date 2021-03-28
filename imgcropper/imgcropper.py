import tifffile
import cv2
import os
from itertools import product
from tqdm import tqdm
import shutil


def get_slicer(slice_rule):
    """
    return filenames list
    """
    if slice_rule == 'Postdam':
        all_num = ['2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11',
         '4_12', '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7',
          '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8', '7_9']
        test_num = ['2_12', '3_12', '4_12', '5_12', '6_12', '7_12']  
        train_num = list(set(all_num)-set(test_num))
    elif slice_rule == 'Vaihingen':
        train_num = [1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37]
        test_num = [11, 15, 28, 30, 34]
        train_num = [str(i) for i in train_num]
        test_num = [str(i) for i in test_num]
    else:
        raise NotImplementedError
    return train_num, test_num

def mkdirnx(path):
    if not os.path.exists(path):
        os.makedirs(path)

def tiff2png(imgpath):
    np_img = tifffile.imread(imgpath)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(imgpath[:-3]+'png', np_img)
    os.remove(imgpath)

class ImgCropper():
    def __init__(self, srcimgroot, labelroot, outroot, slice_rule,crop_size):
        """
        params{
            srcimgroot[str]:dir of image to be cropped
            outroot[str]:where to save results
            slice_rule[str]:slice which dataset and how
            crop_size[int]:box size 
        }
        """
        self.srcimgroot = srcimgroot
        self.labelroot = labelroot
        self.outroot = outroot
        self.slice_rule = slice_rule
        self.crop_size = crop_size
        self.train_img_out_root = os.path.join(outroot,'train','img')
        self.train_label_out_root = os.path.join(outroot,'train','label')
        self.test_img_out_root = os.path.join(outroot,'test','img')
        self.test_label_out_root = os.path.join(outroot,'test','label')
        mkdirnx(self.train_img_out_root)
        mkdirnx(self.train_label_out_root)
        mkdirnx(self.test_img_out_root)
        mkdirnx(self.test_label_out_root)
        self.train_nums, self.test_nums = get_slicer(slice_rule)
        self.counter = 0
    
    def get_name_by_num(self,num):
        if self.slice_rule == 'Vaihingen':
            train_src_name = 'top_mosaic_09cm_area{}.tif'.format(str(num))
            test_src_name = 'top_mosaic_09cm_area{}.tif'.format(str(num))
            label_src_name = 'top_mosaic_09cm_area{}.tif'.format(str(num))
        elif self.slice_rule == 'Postdam':
            train_src_name = 'top_potsdam_{}_RGB.tif'.format(num)
            test_src_name = 'top_potsdam_{}_RGB.tif'.format(num)
            label_src_name = 'top_potsdam_{}_label.tif'.format(num)
        else:
            raise NotImplementedError
        return train_src_name, test_src_name, label_src_name


    def crop_one(self,img_num):
        """
        裁剪一张图\n
        同时裁剪原图和label\n
        边角料被扔掉\n
        """
        img_name,_,label_name = self.get_name_by_num(img_num)
        img_path = os.path.join(self.srcimgroot,img_name)
        label_path = os.path.join(self.labelroot,label_name)
        src_img = self.load_img(img_path)
        label_img = self.load_img(label_path)
        height,width,_ = src_img.shape
        h_num = height//self.crop_size
        w_num = width//self.crop_size
        for h_index, w_index in product(range(h_num), range(w_num)):
            src_crop = src_img[h_index*self.crop_size:(h_index+1)*self.crop_size, w_index*self.crop_size:(w_index+1)*self.crop_size]
            label_crop = label_img[h_index*self.crop_size:(h_index+1)*self.crop_size, w_index*self.crop_size:(w_index+1)*self.crop_size]
            img_save_name = str(self.counter)+'.png'
            self.save_one(root=self.train_img_out_root,img=src_crop)
            self.save_one(root=self.train_label_out_root,img=label_crop)
            self.counter = self.counter + 1

    def crop_all(self):
        for img_num in tqdm(self.train_nums):
            self.crop_one(img_num)

    def copy_test(self):
        for num in self.test_nums:
            _, test_img_name, test_label_name = self.get_name_by_num(num)
            img_path = os.path.join(self.srcimgroot,test_img_name)
            label_path = os.path.join(self.labelroot,test_label_name)
            shutil.copy(img_path, self.test_img_out_root)
            shutil.copy(label_path, self.test_label_out_root)
            img_copied_path = os.path.join(self.test_img_out_root,test_img_name)
            label_copied_path = os.path.join(self.test_label_out_root,test_label_name)
            img_renamed_path = os.path.join(self.test_img_out_root,num + test_img_name[-4:])
            label_renamed_path = os.path.join(self.test_label_out_root,num + test_label_name[-4:])

            os.rename(img_copied_path,img_renamed_path)
            os.rename(label_copied_path,label_renamed_path)
            tiff2png(img_renamed_path)
            tiff2png(label_renamed_path)
    
    def load_img(self,img_path):
        """
        imread and return ndarray
        """
        if img_path[-3:]=='tif':
            np_img = tifffile.imread(img_path)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            np_img = cv2.imread(img_path)

        return np_img
        
    def save_one(self,root,img):
        save_name = str(self.counter)+'.png'
        cv2.imwrite(os.path.join(root,save_name), img)    


def test1():
    params = {
        'srcimgroot':r'D:\data\Vaihingen\ISPRS_semantic_labeling_Vaihingen\top',
        'labelroot':r'D:\data\Vaihingen\ISPRS_semantic_labeling_Vaihingen\gts_for_participants',
        'outroot':r'D:\data\mycrop\Vaihingen',
        'slice_rule':'Vaihingen',
        'crop_size':256
    }
    
    ic = ImgCropper(**params)
    ic.crop_one(r'top_mosaic_09cm_area1.tif')

def crop_Vaihingen():
    params = {
        'srcimgroot':r'D:\data\Vaihingen\ISPRS_semantic_labeling_Vaihingen\top',
        'labelroot':r'D:\data\Vaihingen\ISPRS_semantic_labeling_Vaihingen\gts_for_participants',
        'outroot':r'D:\data\mycrop\Vaihingen',
        'slice_rule':'Vaihingen',
        'crop_size':256
    }
    
    ic = ImgCropper(**params)
    ic.crop_all()
    ic.copy_test()

def crop_Postdam():
    params = {
        'srcimgroot':r'D:\data\Postdam\2_Ortho_RGB',
        'labelroot':r'D:\data\Postdam\5_Labels_for_participants',
        'outroot':r'D:\data\mycrop\Postdam',
        'slice_rule':'Postdam',
        'crop_size':512
    }
    
    ic = ImgCropper(**params)
    ic.crop_all()
    ic.copy_test()

if __name__ == '__main__':
    # crop_Vaihingen()
    crop_Postdam()