from sklearn import svm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#画图设置坐标轴函数
def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')


def plt_support_(clf, X_, y_, kernel, c):
    pos = y_ == 1
    neg = y_ == -1
    ax = plt.subplot()
    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(0, 0.8, 600)
#构建一个二维网格平面
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
#np.c_按行连接两个矩阵 预测这个平面上的值
    Z_rbf = clf.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)
    # ax.contourf(X_, Y_, Z_rbf, alpha=0.75)
#绘制预测的边界图 即决策域
    cs = ax.contour(X_tmp, Y_tmp, Z_rbf, [0], colors='orange', linewidths=1)
    ax.clabel(cs, fmt={cs.levels[0]: 'decision boundary'})

    set_ax_gray(ax)

    ax.scatter(X_[pos, 0], X_[pos, 1], label='1', color='red')
    ax.scatter(X_[neg, 0], X_[neg, 1], label='0', color='purple')

    ax.scatter(X_[clf.support_, 0], X_[clf.support_, 1], marker='o', c='yellow', edgecolors='green', s=150,
               label='support_vectors')

    ax.legend()
    ax.set_title('{} kernel, C={}'.format(kernel, c))
    plt.show()


data = pd.read_csv('watermelon3a.csv',header=None)
#读取训练样本的特征值与标签
#X的前两列分别代表西瓜的
#y的1代表为好瓜 0代表为坏瓜
X = data.iloc[:, [0, 1]].values
y = data.iloc[:, 2].values
print(X)
print(y)
#将坏瓜标签记为-1
y[y == 0] = -1
#C为支持向量机的正则化参数.C越大正则化的强度越弱 与C的大小成反比
# The strength of the regularization is inversely proportional to C.
# Must be strictly positive. The penalty is a squared l2 penalty.
#规定C的值为100
C = 100

#创建径向基函数支持向量机
clf_rbf = svm.SVC(C=C)
#用训练数据拟合分类器模型
clf_rbf.fit(X, y.astype(int))
print('高斯核：')
print('预测值：', clf_rbf.predict(X))
print('真实值：', y.astype(int))
print('支持向量：', clf_rbf.support_)

print('-' * 40)

#创建线性核的支持向量机
clf_linear = svm.SVC(C=C, kernel='linear')
#用训练数据拟合分类器模型
clf_linear.fit(X, y.astype(int))
print('线性核：')
print('预测值：', clf_linear.predict(X))
print('真实值：', y.astype(int))
print('支持向量：', clf_linear.support_)

#绘制散点图实现支持向量机分类的可视化 观察支持向量的选取
plt_support_(clf_rbf, X, y, 'rbf', C)

plt_support_(clf_linear, X, y, 'linear', C)