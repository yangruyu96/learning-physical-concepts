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
        #self.params2 = nn.Parameter(torch.tensor([[-1.0,1.0],[-4.0,1.0],[-9.0,1.0]]),requires_grad=True)
        #self.params2 = torch.autograd.Variable(torch.rand(1,1,out=None).cuda(),requires_grad=True)
        #self.linear_1 = nn.Linear(10, 10)
        #self.linear_2 = nn.Linear(10,10)
        #self.linear_u_1 = nn.Linear(1,10)
        #self.linear_u_2 = nn.Linear(10,1)
        #self.linear_z_1 = nn.Linear(1,10)
        #self.linear_z_2 = nn.Linear(10,1)
        self.encoder1_1 = nn.Linear(5,20)
        self.encoder1_2 = nn.Linear(20,20)
        self.encoder1_3 = nn.Linear(20,4)
        #self.encoder2_1 = nn.Linear(12,20)
        #self.encoder2_2 = nn.Linear(20,1)
        #self.encoder3_1 = nn.Linear(12, 20)
        #self.encoder3_2 = nn.Linear(20, 1)
        self.Sigmoid = nn.ReLU()
        self.Sigmoid_T = nn.Tanh()
        #self.linear_Q_1 = nn.Linear(1,20)
        #self.linear_Q_2 = nn.Linear(20,1)
        self.linear_x_1 = nn.Linear(4,20)
        self.linear_x_2 = nn.Linear(20,20)
        self.linear_x_3 = nn.Linear(20,5)
        #self.nonlear_1 = nn.Linear(1,20)
        #self.nonlear_2 = nn.Linear(20,1)
        #self.second_1 = nn.Linear(1,20)
        #self.second_2 = nn.Linear(20,1)
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
        #normal = np.float(sp.integrate(l,(z0,-sp.oo,sp.oo)))
        #print(normal)
        latent1 = torch.exp((z**2)*(u[0,0]))/normal
        return latent1
    def decoder_2(self,z):
        latent2 = z
        latent2 = self.linear_x_1(latent2)
        #latent2 = self.Sigmoid(latent2)
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
        #ge_z1.append(encoder[0,0].cpu().detach().numpy())
        #ge_z2.append(encoder[0,1].cpu().detach().numpy())
        #ge_x.append(encoder[0,2].cpu().detach().numpy())
        x_final = encoder[0,0].cpu().detach().numpy()
        y_final = encoder[0,1].cpu().detach().numpy()
        z_final = encoder[0,2].cpu().detach().numpy()
        r_final = math.sqrt(x_final**2 + y_final**2 + z_final**2)
        sphere_final = hanshuku.sphere(x_final/r_final,y_final/r_final,z_final/r_final)
        theta_final = sphere_final[0]
        phi_final = sphere_final[1]
        x_0 = x[0,0]
        y_0 = x[0,1]
        z_0 = x[0,2]
        r_0 = math.sqrt(x_0**2 + y_0**2 + z_0**2)
        sphere_0 = hanshuku.sphere(x_0/r_0,y_0/r_0,z_0/r_0)
        ge_z1 = x_final
        ge_z2 = y_final
        #print('z:',z_0/r_0)
        #print('theta:',ge_z2)
        ge_x = z_final
        loss21 = (loss2_sum/100)
        #print('norm:',torch.norm(encoder))

        loss = loss21  + loss_re/100
        return loss,ge_z1,ge_x,ge_z2,loss2_sum/50



#net = LinearNet()
net = torch.load('net_more_latent')



def base(x,n):
    return math.sqrt(2) * cmath.sin(n * math.pi * x )

def wave(a,b,x):
    return (a * base(x,1) + b * base(x,2))/math.sqrt(abs(a)**2 + abs(b)**2)


def pro(a,b,x):
    return (abs(wave(a,b,x))**2)

'''
inter = 0
for dx in range(0,10000):
    inter = inter + pro(0.5,0,dx/10000)/10000

print('inter:',inter)
os.system("pause")
'''

circle = []
for i in range(0,100):
    a0_list = []
    y0_list = []
    for time in range(200000):
        # print(time)
        gene = generate(1, Q, TL, 0)
        if gene[1] > 0:
            a0_list.append(gene[0])
            y0_list.append(gene[1])
    b0_list = []
    y0_list = []
    for time in range(200000):
        # print(time)
        gene = generate(1, Q, TL, 0)
        if gene[1] > 0:
            b0_list.append(gene[0])
            y0_list.append(gene[1])
    k=0
    for k in range(0, 100):
        # print(k)
        phase = cmath.exp(1j * np.random.random() * 2 * math.pi)
        non_diag = a0_list[k] * b0_list[k] * phase
        state_matrix = np.array([[a0_list[k]*(a0_list[k]),a0_list[k]*b0_list[k]*(phase.conjugate())],[a0_list[k]*b0_list[k]*(phase),b0_list[k]*b0_list[k]]])
        tomo = hanshuku.tomography(state_matrix)
        tomo1 = tomo[0,0]
        tomo2 = tomo[0,1]
        tomo3 = tomo[0,2]
        tomo4 = tomo[0,3]
        tomo5 = tomo[0,4]
        y_pro = [tomo1.real, tomo2.real, tomo3.real,tomo4.real,tomo5.real]
        '''
        y_pro = []
        phase = cmath.exp(1j * np.random.random() * 10)
        for xrange in range(0, 5):
            probability = 0
            for repedx in range(0, 10000):
                probability = pro(a0_list[k], b0_list[k] * phase, xrange / 5 + repedx / 50000) / 50000 + probability
            y_pro.append(probability)
        '''
        circle.append(y_pro)




def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma
    return x


#list = Z_ScoreNormalization(circle,np.mean(circle),np.std(circle))
list = circle


state_theta = []

def train():
    output1 = 0
    y_total = []

    ran = np.random.randint(10000,size=100)
    for hangshu in ran:
        y_total.append(list[hangshu])
    y_total = torch.tensor(y_total).view(100,5) #+ (torch.randn(200,5)/10000)
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
        state_theta.append([output[1],output[2],output[3]])
        ge_1_1.append(output[1])
        ge_x_1.append(output[2])
        ge_1_2.append(output[3])
    #print('len:',len(state_theta))
    output1 = output1/100

    return (output1).item(),ge_1_1,ge_2_1,ge_x_1,ge_x_2,ge_3_1,ge_x_3,output[4],ge_1_2,ge_2_2,ge_3_2

loss_model = []

def Save_list(list1,filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))              # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')                          # 相当于Tab一下，换一个单元格
        file2.write('\n')                              # 写完一行立马换行
    file2.close()

circle_hamil = []
y_pro_hamil = []
for epoch in range(10):
    t = train()
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


#print(state_e)
#Save_list(state_theta,r'state_theta')
                  
'''
    loss_model.append(t[0])
    
    if epoch%2 == 1:
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
        
        plt.scatter(ge_1_1,ge_1_2)
        plt.show()
        plt.scatter(ge_x_1, ge_1_2)
        plt.show()
        plt.scatter(ge_x_1,ge_1_1)
        plt.show()
        
    if epoch%2 == 1:
        print(loss_model)
        xarr = np.arange(0,len(loss_model))
        plt.plot(xarr,loss_model)
        plt.show()
    #print(t[0])
'''




