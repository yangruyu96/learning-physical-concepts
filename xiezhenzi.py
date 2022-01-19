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
for i in range(0,1000):
    x = (np.random.random() - 0.5) * 10
    p = (np.random.random() - 0.5) * 10

    radius = np.sqrt(2 * (x**2)  + p ** 2)

    x = x / radius
    p = p / radius


    y_pro = [x,-x,p]
    #print(y_pro)
    circle.append(y_pro) #随机生成观测量


def Save_list(list1,filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))              # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')                          # 相当于Tab一下，换一个单元格
        file2.write('\n')                              # 写完一行立马换行
    file2.close()

#Save_list(circle,r'C:\Users\yangruyu\Desktop\code\myfile_xiezhenzi')

class LinearNet(nn.Module): #定义神经网络

    def __init__(self):
        super(LinearNet, self).__init__()
        self.params1 = nn.Parameter(torch.tensor([[1, 0], [1, 0], [1, 0]]), requires_grad=False)

        self.encoder1_1 = nn.Linear(3, 20)
        self.encoder1_2 = nn.Linear(20, 20)
        self.encoder1_3 = nn.Linear(20, 3)

        self.Sigmoid = nn.ReLU()
        # encoder
        self.Sigmoid_T = nn.Tanh()

        self.linear_x_1 = nn.Linear(3,20)
        self.linear_x_2 = nn.Linear(20,20)
        self.linear_x_3 = nn.Linear(20,3)
#decoder

    def forward(self,x,u0):
        loss_re = 0
        loss2_sum = 0
        #print(x)
        encoder = self.encoder1_1(x)
        encoder = self.Sigmoid_T(encoder)
        encoder = self.encoder1_2(encoder)
        encoder = self.Sigmoid_T(encoder)
        encoder = self.encoder1_3(encoder)
        #latent
        l2 = self.linear_x_1(encoder)
        l2 = self.Sigmoid(l2)
        l2 = self.linear_x_2(l2)
        l2 = self.Sigmoid(l2)
        l2 = self.linear_x_3(l2)
        #decoder
        iden = torch.tensor([1,1,1])
        print(encoder)
        loss2_sum = loss2_sum + (torch.norm((l2 - x),p=2)**2) #输出与输入的差别
        loss_re = loss_re + torch.abs(torch.norm((encoder[0,0]),p=2)**2 + torch.norm((encoder[0,1]),p=2)**2  - 1) + torch.abs(encoder[0,2]**2) * 100
        #这里选取规范，限制latent
        ge_z1 = encoder[0,0].cpu().detach().numpy()
        ge_z2 = (encoder[0,1].cpu().detach().numpy())
        #ge_x = (encoder[0,2].cpu().detach().numpy())
        ge_x = (encoder[0,2].cpu().detach().numpy())
        #记录latent
        loss21 = (loss2_sum)

        loss =  loss21  + loss_re


        return loss,ge_z1,ge_x,ge_z2,loss2_sum/50

net = LinearNet()
#net = torch.load('net_xiezhenzi')
def train():
    output1 = 0
    y_total = []

    ran = np.random.randint(1000,size=100)#生成随机数
    for hangshu in ran:
        y_total.append(circle[hangshu])#按照随机数读取观测量
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
        #print(output[0])
        output1 = output[0] + output1#累加loss
        ge_1_1.append(output[1])
        ge_x_1.append(output[2])
        ge_1_2.append(output[3])
        #收集latent
    output1 = output1/100
    print('loss:',output1)
    output1.backward(retain_graph=True)

    '''
    for name, parms in net.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
              ' -->grad_value:', parms.grad)
    '''
    #print('grad：',net.encoder.grad)
    optimizer.step()
    optimizer.zero_grad()

    #torch.save(net, 'net_xiezhenzi_more')
    return (output1).item(),ge_1_1,ge_2_1,ge_x_1,ge_x_2,ge_3_1,ge_x_3,output[4],ge_1_2,ge_2_2,ge_3_2

loss_model = []
for epoch in range(100000):
    optimizer = optim.Adam(net.parameters(),lr = 0.0001) #指定优化器和相关参数
    t = train()
    loss_model.append(t[0])
    if epoch%100 == 1:
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
        ax.scatter(ge_1_1,ge_1_2,ge_x_1)
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        plt.show()
    if epoch%100 == 1:
        print(loss_model)
        xarr = np.arange(0,len(loss_model))
        plt.plot(xarr,loss_model)
        plt.show()