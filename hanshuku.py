import argparse
import errno
import os
import gudhi
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
import math
import numpy as np
from numpy import random
import cmath
import sympy as sp
from mogutda import SimplicialComplex
import torch
import torch.nn as nn





def Q(z):
    return 1

def T(z):
    return -z**2


def L(u):
    return u

def TL(z,u):
    return - z**2 * ((u)**2) + z * u *2

def pdf(z0,u,Q,TL,l):
    if l==0:
        return Q(z0)*(math.e**(TL(z0,u+1)))/np.float(y0)
    if l==1:
        return Q(z0)*(math.e**(TL(z0,u+1)))/np.float(y1)
    if l==2:
        return Q(z0) * (math.e ** (TL(z0,u+1))) / np.float(y2)

def generate(u,Q,TL,l):
    x = (random.random()-0.5)*10
    y = random.random()*20
    if y < pdf(x,u,Q,TL,l):
        return x,y
    else:
        return -1,-1



def Save_list(list1,filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))              # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')                          # 相当于Tab一下，换一个单元格
        file2.write('\n')                              # 写完一行立马换行
    file2.close()



def A(i):
    if i==0:
        return np.array([[1,0],[0,1]])
    elif i==1:
        return np.array([[0,1],[1,0]])
    elif i==2:
        return np.array([[0,-1j],[1j,0]])
    elif i==3:
        return np.array([[1,0],[0,-1]])

def H_PTM(i,k,H,p):
    x=np.matmul(H,A(i))-np.matmul(A(i),H)
    #x=np.matmul(np.matmul(expm(1j*H*p),A(i)),expm(-1j*H*p))
    return np.trace(np.matmul(x,A(k)))

def gen_SPAM():
    H1 = random.randn(2, 2)
    H2 = random.randn(2, 2) * 1j
    H = H1 + H2
    H = (H + (H.T).conjugate())


    H_0 = np.zeros((4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            H_0[i, j] = -1j * H_PTM(i, j, H, 0) / 2
    # 得到-iH的ptm，自然单位制。
    q = expm(H_0 * 0.00001 )
    #print(q)
    return q

def base(x,n):
    return math.sqrt(2) * cmath.sin(n * math.pi * x )

def wave(a,b,x):
    return (a * base(x,1) + b * base(x,2))/math.sqrt(abs(a)**2 + abs(b)**2)

def pro(a,b,x):
    return (abs(wave(a,b,x))**2)

def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma
    return x

def generate_Hermitian(i):
    random.seed(i)
    x = random.random() * 2 - 1
    y = random.random() * 2 - 1
    z = random.random() * 2 - 1
    r = np.sqrt(x**2 + y**2 + z**2)
    print(x,y,z)
    H = (x*A(1) + y*A(2) + z*A(3))/r
    #print(H)
    return H

def evolution(H,rho,jiange):
    U = expm(-1j * H * jiange)
    #print(U)
    rho = np.matmul(U,rho)
    motai = np.matmul(rho,(U.T).conjugate())
    return motai

def initial_state():
    state = np.array([[1,0],[0,0]])
    return state

def tomography(rho):
    trace = np.trace(rho)
    tomo1 = np.trace(np.matmul(A(1),rho))/trace
    tomo2 = -np.trace(np.matmul(A(2),rho))/trace
    tomo3 = np.trace(np.matmul(A(3),rho))/trace
    tomo4 = (tomo1 + tomo2)
    tomo5 = (tomo3 + tomo2)
    return np.array([[tomo1.real,tomo2.real,tomo3.real,tomo4.real,tomo5.real]])

def evolution_10(H,rho,jiange):
    for times in (0,10):
        rho = evolution(H,rho,jiange)
    return rho

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
        print(encoder)
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
        return loss,ge_z1,ge_x,ge_z2,loss2_sum/50

net = LinearNet()
#net = torch.load('net_complex_data_second.pth')

def net_state(rho):
    return net(torch.tensor(rho,dtype=torch.float32),1)

def sphere(x,y,z):
    theta = math.acos(z)
    if y>0:
        phi = math.acos(x/math.sin(theta))
    if y<0:
        phi = 2*math.pi - math.acos(x/math.sin(theta))
    return theta,phi



