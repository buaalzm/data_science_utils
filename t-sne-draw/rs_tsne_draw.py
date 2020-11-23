from t_sne_plotter import TsnePlotter
import os
import tifffile
from tqdm import tqdm
from PIL import Image
import numpy as np


def load_data(data_path):
    LEVIR_CD_img_path = os.path.join(data_path,'LEVIR_CD','test','post_img') # D:\data\songkq_data\LEVIR_CD\train
    SemiCD_Google_img_path = os.path.join(data_path,'SemiCD_Google','test','sel_post_img')
    WHU_DSIFN_img_path = os.path.join(data_path,'WHU_DSIFN','test','post_img')
    data_list, label_list = [], []
    for path in os.listdir(LEVIR_CD_img_path):
        data_list.append(os.path.join(LEVIR_CD_img_path,path))
        label_list.append('LEVIR_CD')
    for path in os.listdir(SemiCD_Google_img_path):
        data_list.append(os.path.join(SemiCD_Google_img_path,path))
        label_list.append('SemiCD_Google')
    for path in os.listdir(WHU_DSIFN_img_path):
        data_list.append(os.path.join(WHU_DSIFN_img_path,path))
        label_list.append('WHU_DSIFN')
    return data_list, label_list

if __name__ == "__main__":
    tsne = TsnePlotter()
    data_path_list, label_list = load_data(r'D:\data\songkq_data')
    for index, path in tqdm(enumerate(data_path_list)):
        # 图片尺寸大小必须一样
        data = tifffile.imread(path)
        PILimage = Image.fromarray(np.uint8(data))
        resized_image = PILimage.resize((64, 64))
        data = np.asarray(resized_image)
        tsne.add_data(data,label_list[index])
    tsne.show_profile()
    tsne.draw()