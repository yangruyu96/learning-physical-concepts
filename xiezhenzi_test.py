import torch
#from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import sympy as sp
import math
import torch.utils.data as Data
#import seaborn as sns
import cmath
from scipy import integrate
import os
from mogutda import SimplicialComplex

from mpl_toolkits.mplot3d import Axes3D

circle = []
for i in range(0,10000):
    x = (np.random.random() - 0.5) * 10
    p = (np.random.random() - 0.5) * 10

    radius = np.sqrt(2 * (x**2)  + p ** 2)

    x = x / radius
    p = p / radius


    y_pro = [x,-x,p]
    #print(y_pro)
    circle.append(y_pro) #生成随机观测量

def Save_list(list1,filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))              # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')                          # 相当于Tab一下，换一个单元格
        file2.write('\n')                              # 写完一行立马换行
    file2.close()

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.params1 = nn.Parameter(torch.tensor([[1,0],[1,0],[1,0]]),requires_grad=False)

        self.encoder1_1 = nn.Linear(3,20)
        self.encoder1_2 = nn.Linear(20,20)
        self.encoder1_3 = nn.Linear(20,3)

        self.Sigmoid = nn.ReLU()
        self.Sigmoid_T = nn.Tanh()

        self.linear_x_1 = nn.Linear(3,20)
        self.linear_x_2 = nn.Linear(20,20)
        self.linear_x_3 = nn.Linear(20,3)
        #latent的个数是3,有一个是冗余的。


    def forward(self,x,u0):
        loss_re = 0
        loss2_sum = 0
        #print(x)
        encoder = self.encoder1_1(x)
        encoder = self.Sigmoid_T(encoder)
        encoder = self.encoder1_2(encoder)
        encoder = self.Sigmoid_T(encoder)
        encoder = self.encoder1_3(encoder)
        #encoder就是latent
        l2 = self.linear_x_1(encoder)
        l2 = self.Sigmoid(l2)
        l2 = self.linear_x_2(l2)
        l2 = self.Sigmoid(l2)
        l2 = self.linear_x_3(l2)
        iden = torch.tensor([1,1,1])
        print(encoder)
        loss2_sum = loss2_sum + (torch.norm((l2 - x),p=2)**2) #看输出与输入的差别
        loss_re = loss_re + torch.abs(torch.norm((encoder[0,0]),p=2)**2 + torch.norm((encoder[0,1]),p=2)**2  - 1) + torch.abs(encoder[0,2]**2) * 100
        #我们选取的一种规范
        ge_z1 = encoder[0,0].cpu().detach().numpy()
        ge_z2 = (encoder[0,1].cpu().detach().numpy())
        #ge_x = (encoder[0,2].cpu().detach().numpy())
        #ge_x = (encoder[0,2].cpu().detach().numpy())
        ge_x = x[0,2].cpu().detach().numpy() #Z
        #把latent记录下来
        loss21 = (loss2_sum)

        loss =  loss21  + loss_re


        return loss,ge_z1,ge_x,ge_z2,loss2_sum/50

net = torch.load('/home/ruyu_yang/Desktop/learning/net_xiezhenzi_more')

baocunshuju = []
def train():
    output1 = 0
    y_total = []

    ran = np.random.randint(10000,size=100) #生成随即数
    for hangshu in ran:
        y_total.append(circle[hangshu]) #按随机数读取数据
    y_total = torch.tensor(y_total).view(100,3) #+ (torch.randn(100,5)/10000)
    u0 = torch.autograd.Variable(torch.ones(100, 1), requires_grad=True)
    mydataset = Data.TensorDataset(y_total.float(), u0) #打包数据
    data_loader = Data.DataLoader(dataset=mydataset, batch_size=1, shuffle=True, num_workers=0)
    ge_1_1 = []
    ge_2_1 = []
    ge_3_1 = []
    ge_1_2 = []
    ge_2_2 = []
    ge_3_2 = []
    ge_x_1 = []
    ge_x_2 = []
    ge_x_3 = []
    for step,(batch_x,batch_y) in enumerate(data_loader):
        output = net(batch_x, batch_y)
        output1 = output[0] + output1
        ge_1_1.append(output[1]) #X
        #ge_x_1.append(0)
        ge_x_1.append(output[2]) #Z
        ge_1_2.append(output[3]) #Y
        baocunshuju.append([output[1],output[2],output[3]]) #输出latent或者输入和输出

    #torch.save(net, 'net_xiezhenzi_more')
    return (output1).item(),ge_1_1,ge_2_1,ge_x_1,ge_x_2,ge_3_1,ge_x_3,output[4],ge_1_2,ge_2_2,ge_3_2



loss_model = []
for epoch in range(10):
    #optimizer = optim.Adam(net.parameters(),lr = 0.0001)
    t = train()
    '''
    ge_1_1 = t[1]
    ge_2_1 = t[2]
    ge_3_1 = t[5]
    ge_1_2 = t[8]
    ge_2_2 = t[9]
    ge_3_2 = t[10]
    ge_x_1 = t[3]
    ge_x_2 = t[4]
    ge_x_3 = t[6]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(ge_1_1, ge_1_2, ge_x_1)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
'''
Save_list(baocunshuju,r'C:\Users\yangruyu\Desktop\code\xiezhenzi_compare_p')

