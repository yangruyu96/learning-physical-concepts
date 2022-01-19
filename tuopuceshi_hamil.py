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

from scipy.linalg import expm

state = np.array([[1],[0]])

Hamiltonian = np.array([[1,1j],[-1j,-1]])

'''
for i in range(0,100):
    state = np.matmul(expm(1j * Hamiltonian * i / 10000),state)
    print(state[0]*state[0].conjugate())
'''


def Q(z):
    return 1

def T(z):
    return -z**2


def L(u):
    return u

def TL(z,u):
    return - z**2 * ((u)**2) + z * u *2

def TL(z,u):
    return - z**2 * ((u)**2) + z * u *2


z = sp.symbols('z')
y0 = np.float(sp.integrate(Q(z)*(sp.E**(TL(z,2))),(z,-sp.oo,sp.oo)))
z = sp.symbols('z')
y1 = np.float(sp.integrate(Q(z)*(sp.E**(TL(z,3))),(z,-sp.oo,sp.oo)))
z = sp.symbols('z')
y2 = np.float(sp.integrate(Q(z)*(sp.E**(TL(z,4))),(z,-sp.oo,sp.oo)))
z = sp.symbols('z')
y3 = np.float(sp.integrate(Q(z)*(sp.E**(TL(z,5))),(z,-sp.oo,sp.oo)))

def pdf(z0,u,Q,TL,l):
    if l==0:
        return Q(z0)*(math.e**(TL(z0,u+1)))/np.float(y0)
    if l==1:
        return Q(z0)*(math.e**(TL(z0,u+1)))/np.float(y1)
    if l==2:
        return Q(z0) * (math.e ** (TL(z0,u+1))) / np.float(y2)
    if l==3:
        return Q(z0) * (math.e ** (TL(z0,u+1))) / np.float(y3)


def generate(u,Q,TL,l):
    x = (random.random()-0.5)*10
    y = random.random()*20
    if y < pdf(x,u,Q,TL,l):
        return x,y
    else:
        return -1,-1


def fun1(x,y,z):
    return torch.abs(x**2 + y**2 + z**2 - 1)

def fun2(average,sigma,x):
    return (1/math.sqrt(2*math.pi*abs(sigma))) * math.exp( -(x - average)**2 / (2*abs(sigma)) )

unseen = []


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.params1 = nn.Parameter(torch.tensor([[1,0],[1,0],[1,0]]),requires_grad=False)

        self.encoder1_1 = nn.Linear(5,20)
        self.encoder1_2 = nn.Linear(20,20)
        self.encoder1_3 = nn.Linear(20,4)

        self.Sigmoid = nn.ReLU()
        self.Sigmoid_T = nn.Tanh()

        self.linear_x_1 = nn.Linear(4,20)
        self.linear_x_2 = nn.Linear(20,20)
        self.linear_x_3 = nn.Linear(20,5)


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
        l2 = self.Sigmoid(l2)
        l2 = self.linear_x_2(l2)
        l2 = self.Sigmoid(l2)
        l2 = self.linear_x_3(l2)
        iden = torch.tensor([1,1,1])
        loss2_sum = loss2_sum + (torch.norm((l2 - x),p=2)**2)
        loss_re = loss_re + torch.abs(torch.norm((encoder[0,0]),p=2)**2  + torch.norm((encoder[0,1]),p=2)**2 + torch.norm((encoder[0,2]),p=2)**2 - 1) + (encoder[0,3]**2) * 100

        ge_z1 = encoder[0,0].cpu().detach().numpy()
        ge_z2 = (encoder[0,1].cpu().detach().numpy())
        ge_x = (encoder[0,2].cpu().detach().numpy())
        loss21 = (loss2_sum)

        loss =  loss21 + loss_re
        print(encoder)

        return loss,ge_z1,ge_x,ge_z2,loss2_sum/50



#net = LinearNet()
net = torch.load('net_more_latent')



def base(x,n):
    return math.sqrt(2) * cmath.sin(n * math.pi * x )

def wave(a,b,x):
    return (a * base(x,1) + b * base(x,2))/math.sqrt(abs(a)**2 + abs(b)**2)


def pro(a,b,x):
    return (abs(wave(a,b,x))**2)






def generate_hamiltonian_Data():
    time_hamiltonian = []
    for k in range(0,400):
        H = hanshuku.generate_Hermitian(k)
        state = hanshuku.initial_state()
        time_series = []
        rho = state
        for times in range(0,5):
            #print(times)
            #output = net(torch.tensor(hanshuku.tomography(rho),dtype=torch.float32),1)
            rho = hanshuku.evolution(H, rho, math.pi / 2)
            output = hanshuku.tomography(rho)
            time_series.extend([output[0,0].real])

        #print(time_series)
        time_hamiltonian.append(time_series)
    return time_hamiltonian

def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma
    return x


#list = Z_ScoreNormalization(circle,np.mean(circle),np.std(circle))
list = generate_hamiltonian_Data()



def Save_list(list1,filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))              # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')                          # 相当于Tab一下，换一个单元格
        file2.write('\n')                              # 写完一行立马换行
    file2.close()

Save_list(list, r'C:\Users\yangruyu\Desktop\code\myfile_hamil')
        #print(t[0])





