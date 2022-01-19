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
import hanshuku
from mpl_toolkits.mplot3d import Axes3D
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.params1 = nn.Parameter(torch.tensor([[1,0],[1,0],[1,0]]),requires_grad=False)
        self.encoder1_1 = nn.Linear(5,20)
        self.encoder1_2 = nn.Linear(20,20)
        self.encoder1_3 = nn.Linear(20,3)
        self.Sigmoid = nn.ReLU()
        self.Sigmoid_T = nn.Tanh()
        self.linear_x_1 = nn.Linear(3,20)
        self.linear_x_2 = nn.Linear(20,20)
        self.linear_x_3 = nn.Linear(20,5)
# forward 定义前向传播
    def decoder_1(self,z,n):
        n = n.cpu().detach().numpy() - 1
        u = self.params1[n]
        u0=u.cpu().detach().numpy()
        #print(u0)
        z0 = torch.linspace(-100,100,steps=10**4)
        l = torch.exp((z0**2)*u0[0,0])
        x = np.float(z0[1] - z0[0])
        normal = 0
        for i in range(10**4):
            normal = normal + l[i]*x
        latent1 = torch.exp((z**2)*(u[0,0]))/normal
        return latent1
    def decoder_2(self,z):
        latent2 = z
        latent2 = self.linear_x_1(latent2)
        latent2 = self.linear_x_2(latent2)
        return latent2
    def generate(self,encoder):
        sample0 = torch.randn(1, 1) * torch.sqrt(torch.abs(encoder[0, 0])) + encoder[0, 3]
        sample1 = torch.randn(1, 1) * torch.sqrt(torch.abs(encoder[0, 1])) + encoder[0, 4]
        sample2 = torch.randn(1, 1) * torch.sqrt(torch.abs(encoder[0, 2])) + encoder[0, 5]
        sample = torch.Tensor([sample0, sample1, sample2])
        return sample
    def forward(self,x,u0):
        loss_re = 0
        loss2_sum = 0
        #print(x)
        encoder = self.encoder1_1(x)
        encoder = self.Sigmoid_T(encoder)
        encoder = self.encoder1_2(encoder)
        encoder = self.Sigmoid_T(encoder)
        encoder = self.encoder1_3(encoder)
        #print(encoder)
        ge_z1 = []
        ge_z2 = []
        ge_x = []
        for times in range(100):
            l2 = self.linear_x_1(encoder)
            l2 = self.Sigmoid(l2)
            l2 = self.linear_x_2(l2)
            l2 = self.Sigmoid(l2)
            l2 = self.linear_x_3(l2)
            iden = torch.tensor([1,1,1])
            loss2_sum = loss2_sum + (torch.norm((l2 - x),p=2)**2)
            loss_re = loss_re + torch.abs(torch.norm((encoder[0,0]),p=2)**2  + torch.norm((encoder[0,1]),p=2)**2 + torch.norm((encoder[0,2]),p=2)**2 - 1)
        ge_z1.append(encoder[0,0].cpu().detach().numpy())
        ge_z2.append(encoder[0,1].cpu().detach().numpy())
        ge_x.append(encoder[0,2].cpu().detach().numpy())
        loss21 = (loss2_sum/100)
        loss =  loss21
        return loss,encoder[0,0],encoder[0,2],encoder[0,1],loss2_sum/50#这里修改了输出，输出数而不是ge_z1等数列，便于用于重新输入net.




class LinearNet_hamiltonian(nn.Module):
    def __init__(self):
        super(LinearNet_hamiltonian, self).__init__()
        self.params1 = nn.Parameter(torch.tensor([[1,0],[1,0],[1,0]]),requires_grad=False)
        self.encoder1_1 = nn.Linear(15,20)
        self.encoder1_2 = nn.Linear(20,20)
        self.encoder1_3 = nn.Linear(20,6)
        self.Sigmoid = nn.ReLU()
        self.Sigmoid_T = nn.Tanh()
        self.linear_x_1 = nn.Linear(6,20)
        self.linear_x_2 = nn.Linear(20,20)
        self.linear_x_3 = nn.Linear(20,15)
# forward 定义前向传播
    def decoder_1(self,z,n):
        n = n.cpu().detach().numpy() - 1
        u = self.params1[n]
        u0=u.cpu().detach().numpy()
        #print(u0)
        z0 = torch.linspace(-100,100,steps=10**4)
        l = torch.exp((z0**2)*u0[0,0])
        x = np.float(z0[1] - z0[0])
        normal = 0
        for i in range(10**4):
            normal = normal + l[i]*x
        latent1 = torch.exp((z**2)*(u[0,0]))/normal
        return latent1
    def decoder_2(self,z):
        latent2 = z
        latent2 = self.linear_x_1(latent2)
        latent2 = self.linear_x_2(latent2)
        return latent2
    def generate(self,encoder):
        sample0 = torch.randn(1, 1) * torch.sqrt(torch.abs(encoder[0, 0])) + encoder[0, 3]
        sample1 = torch.randn(1, 1) * torch.sqrt(torch.abs(encoder[0, 1])) + encoder[0, 4]
        sample2 = torch.randn(1, 1) * torch.sqrt(torch.abs(encoder[0, 2])) + encoder[0, 5]
        sample = torch.Tensor([sample0, sample1, sample2])
        return sample
    def forward(self,x,u0):
        loss_re = 0
        loss2_sum = 0
        #print(x)
        encoder = self.encoder1_1(x)
        encoder = self.Sigmoid_T(encoder)
        encoder = self.encoder1_2(encoder)
        encoder = self.Sigmoid_T(encoder)
        encoder = self.encoder1_3(encoder)
        #print(encoder)
        ge_z1 = []
        ge_z2 = []
        ge_x = []
        for times in range(100):
            l2 = self.linear_x_1(encoder)
            l2 = self.Sigmoid(l2)
            l2 = self.linear_x_2(l2)
            l2 = self.Sigmoid(l2)
            l2 = self.linear_x_3(l2)
            iden = torch.tensor([1,1,1])
            loss2_sum = loss2_sum + (torch.norm((l2 - x),p=2)**2)
        loss_re = loss_re + torch.abs(torch.norm((encoder[0,0]),p=2)**2  + torch.norm((encoder[0,1]),p=2)**2 + torch.norm((encoder[0,2]),p=2)**2 - 1)
        ge_z1.append(encoder[0,0].cpu().detach().numpy())
        ge_z2.append(encoder[0,1].cpu().detach().numpy())
        ge_x.append(encoder[0,2].cpu().detach().numpy())
        loss21 = (loss2_sum/100)
        loss =  loss21
        junzhi = [encoder[0,3],encoder[0,4],encoder[0,5],loss_re,loss_re * encoder[0,3] * encoder[0,4] * encoder[0,5]]
        return loss,ge_z1,ge_x,ge_z2,loss2_sum/50,junzhi



net_state = LinearNet()
net_state = torch.load('net_complex_data_second.pth')
net_hamiltonian = LinearNet_hamiltonian()
def generate_hamiltonian_Data():
    time_hamiltonian = []
    for k in range(0,1000):
        H = hanshuku.generate_Hermitian(k)

        state = hanshuku.initial_state()

        time_series = []

        rho = state
        for times in range(0,5):
            rho = hanshuku.evolution(H,rho,0.1/(times+1))
            #print(times)
            output = net_state(torch.tensor(hanshuku.tomography(rho),dtype=torch.float32),1)
            time_series.extend([output[1],output[2],output[3]])
        #print(time_series)
        time_hamiltonian.append(time_series)
    return time_hamiltonian

list = generate_hamiltonian_Data()
def train_hamiltonian():
    output1 = 0
    y_total = []

    ran = np.random.randint(1000, size=100)
    #print(ran)
    for hangshu in ran:
        y_total.append(list[hangshu])
    print(y_total)
    y_total = torch.tensor(y_total).view(100, 15) #+ (torch.randn(100, 30) / 10000)
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
    x1,x2,x3,x4,x0 = 0,0,0,0,0
    for step, (batch_x, batch_y) in enumerate(data_loader):
        output = net_hamiltonian(batch_x, batch_y)
        # print(output[0])
        junzhi = output[-1]
        x0 = junzhi[0] + x0
        x1 = junzhi[1] + x1
        x2 = junzhi[2] + x2
        x3 = junzhi[3] + x3
        x4 = junzhi[4] + x4
        output1 = output[0] + output1
        ge_1_1.extend(output[1])
        ge_x_1.extend(output[2])
        ge_1_2.extend(output[3])
    output1 = (output1 / 100) + torch.abs(x4/100 - (x0/100) * (x1/100) * (x2/100) * (x3/100))
    print('loss:', output1)
    output1.backward(retain_graph=True)
    for name, parms in net_hamiltonian.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
              ' -->grad_value:', parms.grad)

    '''
    for name, parms in net.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
              ' -->grad_value:', parms.grad)
    '''
    # print('grad：',net.encoder.grad)
    optimizer.step()
    optimizer.zero_grad()
    torch.save(net_hamiltonian,'net_hamiltonian')
    for name, parms in net_hamiltonian.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
              ' -->grad_value:', parms.grad)

    return (output1).item(), ge_1_1, ge_2_1, ge_x_1, ge_x_2, ge_3_1, ge_x_3, output[4], ge_1_2, ge_2_2, ge_3_2

loss_model = []
for epoch in range(10000):
    optimizer = optim.Adam(net_hamiltonian.parameters(),lr = 0.0001)
    t = train_hamiltonian()
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
    #print(t[0])
