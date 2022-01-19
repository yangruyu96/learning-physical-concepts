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
for t in range(0,10000):
    #t = z
    ro = math.pi/2
    theta_1 = t
    theta_2 = 2*t + math.pi/2
    coor_1 = [math.cos(theta_1),math.sin(theta_1),0]
    coor_2_i = [math.cos(theta_2),math.sin(theta_2),0]
    rotation = np.array([[1,0,0],[0,math.cos(ro),-math.sin(ro)],[0,math.sin(ro),math.cos(ro)]])
    coor_2 = np.matmul(rotation,coor_2_i)
    coor = coor_2 - coor_1
    #s = [0,1,0]
    #print(np.matmul(rotation,s))
    circle.append([coor[0],coor[1],coor[2]])



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


    def forward(self,x,u0):
        loss_re = 0
        loss2_sum = 0
        #print(x)
        encoder = self.encoder1_1(x)
        encoder = self.Sigmoid_T(encoder)
        encoder = self.encoder1_2(encoder)
        encoder = self.Sigmoid_T(encoder)
        encoder = self.encoder1_3(encoder)

        l2 = self.linear_x_1(encoder)
        l2 = self.Sigmoid_T(l2)
        l2 = self.linear_x_2(l2)
        l2 = self.Sigmoid_T(l2)
        l2 = self.linear_x_3(l2)
        iden = torch.tensor([1,1,1])
        #print(encoder)
        loss2_sum = loss2_sum + (torch.norm((l2 - x),p=2)**2)
        loss_re = loss_re + torch.abs(encoder[0,2]**2) * 100 + torch.abs((encoder[0,0]**2 + encoder[0,1]**2)**2 - 0.01 * ((encoder[0,0]**2 - encoder[0,1]**2)))*100
        #print(encoder[0,0]**2)
        #print(loss_re)
        ge_z1 = encoder[0,0].cpu().detach().numpy() #X
        #ge_z1 = x[0,0].cpu().detach().numpy() #X
        ge_z2 = (encoder[0,1].cpu().detach().numpy()) #Y
        #ge_z2 = 0
        #ge_x = (encoder[0,2].cpu().detach().numpy())
        #ge_x = (encoder[0,2].cpu().detach().numpy()) #Z
        ge_x = x[0,2].cpu().detach().numpy()
        #ge_x = 0
        loss21 = (loss2_sum)

        loss = loss_re + loss2_sum
        print('loss2_sum:',loss2_sum)
        print('loss_re:',loss_re)

        return loss,ge_z1,ge_x,ge_z2,loss2_sum/50


#net = LinearNet()
net = torch.load('net_orbit_bernoulli')
baocunshuju = []
def train():
    output1 = 0
    y_total = []

    ran = np.random.randint(10000,size=100)
    for hangshu in ran:
        y_total.append(circle[hangshu])
    y_total = torch.tensor(y_total).view(100,3) #+ (torch.randn(100,5)/10000)
    u0 = torch.autograd.Variable(torch.ones(100, 1), requires_grad=True)
    mydataset = Data.TensorDataset(y_total.float(), u0)
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
        output1 = output[0] + output1
        ge_1_1.append(output[1])
        ge_x_1.append(output[2])
        ge_1_2.append(output[3])
        baocunshuju.append([output[1],output[2],output[3]])

    #torch.save(net, 'net_orbit_bernoulli')
    return (output1).item(),ge_1_1,ge_2_1,ge_x_1,ge_x_2,ge_3_1,ge_x_3,output[4],ge_1_2,ge_2_2,ge_3_2

loss_model = []
for epoch in range(10):
    t = train()



def Save_list(list1,filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))              # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')                          # 相当于Tab一下，换一个单元格
        file2.write('\n')                              # 写完一行立马换行
    file2.close()

Save_list(baocunshuju, r'C:\Users\yangruyu\Desktop\code\orbit_compare_z')
'''
    loss_model.append(t[0])
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


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()
'''