#步骤
#（5）BP神经网络构建一个分类器的思路：
#第一步：初始化参数（包括权值和偏置）
#第二步：前向传播
#第三步：计算代价函数
#第四步：反向传播
#第五步：更新参数
#第六步：模型评估

#加载所需库文件
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import radviz
#参数初始化
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros(shape = (n_h,1))
    w2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros(shape = (n_y,1))
    parameters = {'w1':w1,"w2":w2,"b1":b1,"b2":b2}
    return  parameters
#前向传播
def forward(X,parameters):
    w1 = parameters['w1']
    w2 = parameters['w2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2,a1)+b2
    a2 = 1 / (1+np.exp(-z2))

    cache = {"z1":z1,"z2":z2,"a1":a1,"a2":a2}
    return a2,cache

def comp_Cost(a2,Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(a2),Y) + np.multiply((1-Y),np.log(1-a2))
    cost = - np.sum(logprobs) / m
    variance = np.linalg.norm((Y-a2),ord=None, axis=None, keepdims=False) **2 / 2
    return cost,variance

def backward_propagation(parameters,cache,X,Y):
    m = Y.shape[1]
    w2 = parameters['w2']
    a1 = cache['a1']
    a2 = cache['a2']

    #反向传播
    dz2 = a2 - Y
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1- np.power(a1, 2))
    dw1 = (1/m) * np.dot(dz1,X.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
    grads  = {'dw1':dw1, 'dw2':dw2 , 'db1':db1, 'db2':db2}
    return grads

def update_parameters(parameters,grads,learning_rate=0.3):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    w1 = w1 - dw1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b1 = b1 - db1 * learning_rate
    b2 = b2 - db2 * learning_rate
    parameters = {'w1':w1, 'w2':w2 , 'b1':b1, 'b2':b2}
    return parameters


def nn_model(X,Y,num_Hidden,num_Input,num_Output,num_Iteration,print_cost):
    np.random.seed(3)

    n_x = num_Input
    n_y = num_Output
    parameters = initialize_parameters(n_x,num_Hidden,n_y)
    for i in range(0,num_Iteration):
        a2, cache = forward(X,parameters)
        cost = comp_Cost(a2,Y)[0]
        variance = comp_Cost(a2,Y)[1]
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads)

        if print_cost and i % 100 ==0:
            print('迭代第%i次，代价函数为：%f ,均方差为：%f' %(i,cost,variance))

    return parameters

def predict(parameters, x_test, y_test):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    z1 = np.dot(w1, x_test) +b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2,a1)+b2
    a2 = 1/(1+np.exp(-z2))


    n_rows = y_test.shape[0]
    n_cols = y_test.shape[1]
    output = np.empty(shape=(n_rows,n_cols),dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            if a2[i][j] > 0.5:
                output[i][j] = 1
            else:
                output[i][j] = 0
    print('y_hat:',output)
    print('y:',y_test)
    count = 0
    for k in range(0,n_cols):
        if output[0][k] == y_test[0][k] and output[1][k] == y_test[1][k] and output[2][k] == y_test[2][k]:
            count += 1
        else:
            print('error_num:',k+1)
    acc = count /int(y_test.shape[1]) * 100
    print('accuracy : %.2f %%' % acc)
    return output

# 特征有4个维度，类别有1个维度，一共5个维度，故采用了RadViz图
def result_visualization(x_test, y_test, result):
    cols = y_test.shape[1]
    y = []
    pre = []

    # 反转换类别的独热编码
    for i in range(cols):
        if y_test[0][i] == 0 and y_test[1][i] == 0 and y_test[2][i] == 1:
            y.append('setosa')
        elif y_test[0][i] == 0 and y_test[1][i] == 1 and y_test[2][i] == 0:
            y.append('versicolor')
        elif y_test[0][i] == 1 and y_test[1][i] == 0 and y_test[2][i] == 0:
            y.append('virginica')

    for j in range(cols):
        if result[0][j] == 0 and result[1][j] == 0 and result[2][j] == 1:
            pre.append('setosa')
        elif result[0][j] == 0 and result[1][j] == 1 and result[2][j] == 0:
            pre.append('versicolor')
        elif result[0][j] == 1 and result[1][j] == 0 and result[2][j] == 0:
            pre.append('virginica')
        else:
            pre.append('unknown')

    # 将特征和类别矩阵拼接起来
    real = np.column_stack((x_test.T, y))
    prediction = np.column_stack((x_test.T, pre))

    # 转换成DataFrame类型，并添加columns
    df_real = pd.DataFrame(real, index=None, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])
    df_prediction = pd.DataFrame(prediction, index=None, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])

    # 将特征列转换为float类型，否则radviz会报错
    df_real[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_real[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)
    df_prediction[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_prediction[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)

    # 绘图
    plt.figure('真实分类')
    radviz(df_real, 'Species', color=['blue', 'green', 'red', 'yellow'])
    plt.figure('预测分类')
    radviz(df_prediction, 'Species', color=['blue', 'green', 'red', 'yellow'])
    plt.show()
#数据集

if __name__ == '__main__':
    data_set = pd.read_csv('iris_training.csv',header = None)
    #第一种取数据方法
    X = data_set.iloc[:,0:4].values.T
    Y = data_set.iloc[:,4:].values.T
    Y = Y.astype('uint8')
    #输入四个节点 隐层十个节点 输出三个节点 迭代10000次
    parameters = nn_model(X, Y, num_Hidden=10, num_Input=4, num_Output=3, num_Iteration =50000, print_cost = True)
    #对模型进行测试
    data_test = pd.read_csv('iris_test.csv',header=None)
    x_test = data_test.iloc[:,0:4].values.T
    y_test = data_test.iloc[:,4:].values.T
    y_test = y_test.astype('uint8')
    result = predict(parameters,x_test,y_test)

    #分类结果可视化
    result_visualization(x_test,y_test,result)