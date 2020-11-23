import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold


class TsnePlotter():
    '''
    t-sne绘图
    使用方法：一个一个用add_data添加数据，然后调用draw画图
    '''
    def __init__(self):
        self.class_num = 0 # 类别数量
        self.class_name_list = [] # 类别的名字
        self.data_list = [] # 数据转为一维的ndarray，存入列表
        self.label_list = [] # 存类别的序号

    def add_data(self,data,class_name):
        """
        添加数据
        params:
        {
            data[ndarray]:数据，添加的每一个数据长度应该一样
            class_name[str]:类别的名字
        }
        """
        self.data_list.append(data.reshape(-1))
        if class_name not in self.class_name_list:
            self.class_name_list.append(class_name)
            self.class_num+=1
        self.label_list.append(self.class_name_list.index(class_name))

    def draw(self):
        """
        画t-sne
        """
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        np_data = np.concatenate(tuple(self.data_list),axis=0)
        np_data = np_data.reshape(len(self.data_list),-1)
        X_tsne = tsne.fit_transform(np_data)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.scatter(X_norm[i, 0], X_norm[i, 1],color=plt.cm.Set1(self.label_list[i]),label=self.class_name_list[self.label_list[i]])
        plt.xticks([])
        plt.yticks([])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig('t-sne.png')
        plt.show()

    def show_profile(self):
        """
        打印数据基本信息
        """
        print('data num:{}'.format(len(self.data_list)))
        print('data dimention:{}'.format(self.data_list[0].shape[0]))
        print('label num:{}'.format(self.class_num))
        print('label categories:{}'.format(self.class_name_list))


if __name__ == "__main__":
    # 使用手写数字数据集测试
    from sklearn import datasets
    digits = datasets.load_digits(n_class=6)
    X, y = digits.data, digits.target
    tsne = TsnePlotter()
    for index in range(X.shape[0]):
        tsne.add_data(X[index,:],y[index])
    tsne.show_profile()
    tsne.draw()