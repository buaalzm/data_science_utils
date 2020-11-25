import numpy as np


class ImgWarp():
    """
    适用3通道ndarray的变换，水平镜像、竖直镜像、旋转180°
    """
    @staticmethod
    def horizontal_flip(img):
        """
        水平翻转
        """
        arr2 =img.copy()
        arr2= img.reshape(int(img.size/3),3)
        arr2 =np.array(arr2[::-1])
        arr2 = arr2.reshape(img.shape[0],img.shape[1],img.shape[2])
        return arr2[::-1]

    @staticmethod
    def vertical_flip(img):
        """
        竖直翻转
        """
        arr1 = img[::-1]
        return arr1

    @staticmethod
    def rotate180(img):
        """
        旋转180
        """
        arr2 =img.copy()
        arr2= img.reshape(int(img.size/3),3)#图像像素维度变形为 a*3格式，要保证每个RGB数组不发生改变；
        arr2 =np.array(arr2[::-1])#进行行逆置
        arr2 = arr2.reshape(img.shape[0],img.shape[1],img.shape[2])#再对图像进行一次变换，变成 源图像的维度
        return arr2


if __name__ == "__main__":
    import tifffile
    import matplotlib.pyplot as plt
    img = tifffile.imread(r'941.tif')
    imgh=ImgWarp.horizontal_flip(img)
    imgv=ImgWarp.vertical_flip(img)
    imgr=ImgWarp.rotate180(img)
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.title('original')
    plt.axis('off')
    plt.subplot(2,2,2)
    plt.imshow(imgh)
    plt.title('horizontal')
    plt.axis('off')
    plt.subplot(2,2,3)
    plt.imshow(imgv)
    plt.title('vertical')
    plt.axis('off')
    plt.subplot(2,2,4)
    plt.imshow(imgr)
    plt.title('rotate180')
    plt.axis('off')
    plt.show()