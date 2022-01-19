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
import seaborn as sns



inistate = np.array([[1],[0],[0],[1]])







def Q(z):
    return 1

def T(z):
    return -z**2


def L(u):
    return u

def TL(z,u):
    return - z**2 * ((u)**2) + z * u *2


z = sp.symbols('z')
y0 = np.float(sp.integrate(Q(z)*(sp.E**(TL(z,2))),(z,-sp.oo,sp.oo)))
z = sp.symbols('z')
y1 = np.float(sp.integrate(Q(z)*(sp.E**(TL(z,3))),(z,-sp.oo,sp.oo)))
z = sp.symbols('z')
y2 = np.float(sp.integrate(Q(z)*(sp.E**(TL(z,4))),(z,-sp.oo,sp.oo)))


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

x0_list=[]
y0_list=[]
for time in range(20000000):
    #print(time)
    gene = generate(1,Q,TL,0)
    if gene[1] > 0:
        x0_list.append(gene[0])
        y0_list.append(gene[1])
x1_list=[]
y1_list=[]
for time in range(20000000):
    #print(time)
    gene = generate(2,Q,TL,1)
    if gene[1] > 0:
        x1_list.append(gene[0])
        y1_list.append(gene[1])

bathsize = 1



plt.scatter(x0_list,y0_list)
plt.scatter(x1_list,y1_list)
plt.show()






class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.params1 = nn.Parameter(torch.tensor([[-1.0,1.0],[-4.0,1.0],[-9.0,1.0]]),requires_grad=True)
        #self.params2 = nn.Parameter(torch.tensor([[-1.0,1.0],[-4.0,1.0],[-9.0,1.0]]),requires_grad=True)
        #self.params2 = torch.autograd.Variable(torch.rand(1,1,out=None).cuda(),requires_grad=True)
        self.linear_1 = nn.Linear(10, 10)
        self.linear_2 = nn.Linear(10,10)
        self.linear_u_1 = nn.Linear(1,10)
        self.linear_u_2 = nn.Linear(10,1)
        self.linear_z_1 = nn.Linear(1,10)
        self.linear_z_2 = nn.Linear(10,1)
        self.encoder1_1 = nn.Linear(5,20)
        self.encoder1_2 = nn.Linear(20,1)
        self.encoder2_1 = nn.Linear(5,20)
        self.encoder2_2 = nn.Linear(20,1)
        self.Sigmoid = nn.ReLU()
        self.Sigmoid_T = nn.Tanh()
        self.linear_Q_1 = nn.Linear(1,20)
        self.linear_Q_2 = nn.Linear(20,1)
        self.linear_x_1 = nn.Linear(1,20)
        self.linear_x_2 = nn.Linear(20,1)
        self.nonlear_1 = nn.Linear(1,20)
        self.nonlear_2 = nn.Linear(20,1)
        self.second_1 = nn.Linear(1,20)
        self.second_2 = nn.Linear(20,1)
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
    def forward(self,x,u0):
        z_generate = []
        z_cl_grad = []
        #y = self.linear_1(x)
        #y = self.Sigmoid(y)
        #y = self.linear_2(y)
        #u = self.linear_u_1(u0)
        #u = self.Sigmoid(u)
        #u = self.linear_u_2(u)
        loss = 0
        ge_z1=[]
        ge_z2=[]
        ge_x=[]
        loss2_sum = 0
        for times in range(100):
            zr_01 = torch.autograd.Variable((torch.rand(1,1,out=None)-0.5),requires_grad=True)
            #zr_02 = torch.autograd.Variable((torch.rand(1, 1, out=None) - 0.5), requires_grad=True)
            #print('zr:',zr)
            #print('time:',times)
            #ge_z = []
                #print(times)
                #print(z0)
                #print(u,z0)
                #print(y.size())
            encoder_01 = torch.cat((torch.cat((u0,zr_01),1),x),1)
            #encoder_02 = torch.cat((torch.cat((u0, zr_02), 1), x), 1)
                #print(encoder)
            encoder1 = self.encoder1_1(encoder_01)
            encoder1 = self.Sigmoid(encoder1)
            encoder1 = self.encoder1_2(encoder1)
            #encoder2 = self.encoder2_1(encoder_02)
            #encoder2 = self.Sigmoid(encoder2)
            #encoder2 = u0
                #encoder = self.Sigmoid_T(encoder)
            z_generate.append(encoder1)
                #encoder.to(torch.device('cpu'))
                #print(encoder)
            encoder1.backward(retain_graph=True)
            #print('grad:',zr.grad,'encoder:',encoder)
            z_cl_grad.append(zr_01.grad)
            z_grad_1 = zr_01.grad
                #print(z0.grad)
            #encoder2.backward(retain_graph=True)
            #z_grad_2 = zr_02.grad
            n = u0.cpu().detach().numpy() - 1
            ud1 = self.params1[n]
            #ud2 = self.params2[n]
                #u0 = ud1.cpu().detach().numpy()
                # print(u0)
            #z0 = torch.linspace(-100, 100, steps=10 ** 4).cuda()
            #l = torch.exp( (z0 ** 2) * ud1[0, 0] - z0*ud1[0,1])
            #dx = np.float(z0[1] - z0[0])
            #sigma = torch.sqrt(-1/(2*ud1[0,0]))
            #normal = 1/(sigma * torch.sqrt(2*torch.tensor(math.pi)))
            #normal = 0
            #for i in range(10**4):
                #normal = normal + l[i] * dx
                # normal = np.float(sp.integrate(l,(z0,-sp.oo,sp.oo)))
                # print(normal)
            #l1 = torch.exp(((encoder - ud1[0,1])**2) * (ud1[0, 0])) * normal.cuda()
                #l1 = self.decoder_1(encoder,u0)
                #l2 = self.decoder_2(encoder)
            l2 = self.linear_x_1(encoder1)
            l2 = self.Sigmoid(l2)
            l2 = self.linear_x_2(l2)
            l_second = self.second_1(encoder1)
            l_second = self.Sigmoid(l_second)
            l_second = self.second_2(l_second)
            nonli = self.nonlear_1(encoder1)
            nonli = self.Sigmoid(nonli)
            nonli = self.nonlear_2(nonli)
            loss1 = (1/abs(z_grad_1))
            #if zr.grad == 0:
            #    print('zr:',zr.grad)
            #print(zr.grad)
            l2= torch.cat((l2,l_second),1)
            l2 = torch.cat((l2,nonli),1)
            loss21 = - ((encoder1 - ud1[0,1])**2) * (ud1[0, 0]) - torch.log(-ud1[0,0])/2 + torch.norm((l2 - x),p=2)*20000
            loss= loss + loss21 + torch.log(loss1)
            #loss2_sum = loss2_sum - ((encoder - ud1[0, 1]) ** 2) * (ud1[0, 0]) - torch.log(-ud1[0, 0]) / 2
            #print(loss1)
                #loss.backward(retain_graph=True)
                #print(self.params1.grad)
            ge_z1.append(encoder1.cpu())
            #ge_z2.append(encoder2.cpu())
            ge_x.append(x[0,0].cpu())
            zr_01.grad.data.zero_()
            #zr_02.grad.data.zero_()
            #print('times:',times,'loss1:',loss1,'loss2:',loss2)
        #print('loss1:',loss1)
        #print('l2:',l2)
        #print('Tl:',self.params1,'loss:',loss)
        #print('z0grad',z_cl_grad)
        #print('l1:',l1,'encoder:',encoder)
        ge_z2=0
        return loss/100,ge_z1,ge_x,ge_z2,loss2_sum/100

net = LinearNet()

'''
x = torch.cat((x0,x1),0).cuda()
u = torch.cat((u0,u1),0).cuda()
mydataset1=Data.TensorDataset(x0,u0)
mydataset2=Data.TensorDataset(x1,u1)
mydataset = Data.TensorDataset(x,u)
data_loader = Data.DataLoader(dataset=mydataset,batch_size=1,shuffle=True,num_workers = 0)
'''
#print(net.parameters())
#for name,param in net.named_parameters():
#    print(name,param)

#print(net(x0,u0,0))
def train():
    output1 = 0
    x0_list = []
    y0_list = []
    for time in range(2000000):
        # print(time)
        gene = generate(1, Q, TL, 0)
        if gene[1] > 0:
            x0_list.append(gene[0])
            y0_list.append(gene[1])
    x1_list = []
    y1_list = []
    for time in range(2000000):
        # print(time)
        gene = generate(2, Q, TL, 1)
        if gene[1] > 0:
            x1_list.append(gene[0])
            y1_list.append(gene[1])
    x2_list = []
    y2_list = []
    for time in range(2000000):
        # print(time)
        gene = generate(3, Q, TL, 2)
        if gene[1] > 0:
            x2_list.append(gene[0])
            y2_list.append(gene[1])
    x0 = x0_list[0:20]
    x0 = torch.tensor(x0).view(20,1)/2
    x1 = x1_list[0:20]
    x1 = torch.tensor(x1).view(20,1)
    x2 = x2_list[0:20]
    x2 = torch.tensor(x2).view(20,1)
    u0 = torch.autograd.Variable(torch.ones(20, 1), requires_grad=True)
    u1 = u0 * 2
    u2 = u0 * 3
    x = torch.cat((x0, x1), 0)
    x = torch.cat((x,x2),0)
    x = x
    x_new1 = (x+100)**(-1)
    x_new2 = x**4
    x = torch.cat((x,x_new1),1)
    x = torch.cat((x,x_new2),1)+ (torch.randn(60,3)/100)
    u = torch.cat((u0, u1), 0)
    u = torch.cat((u,u2),0)
    mydataset = Data.TensorDataset(x, u)
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
        #print(batch_x)
        #print(batch_y)
        output = net(batch_x, batch_y)
        output1 = output[0] + output1
        if batch_y == 1:
            ge_1_1.extend(output[1])
            ge_x_1.extend(output[2])
            #ge_1_2.extend(output[3])
        if batch_y == 2:
            ge_2_1.extend(output[1])
            ge_x_2.extend(output[2])
            #ge_2_2.extend(output[3])
        if batch_y == 3:
            ge_3_1.extend(output[1])
            ge_x_3.extend(output[2])
            #ge_3_2.extend(output[3])
    optimizer.zero_grad()
    output1 = output1/60
    output1.backward(retain_graph=True)
    print('grad:',net.params1.grad)
    print('tl:',net.params1)
    net.params1.data = net.params1.data - 0.001 * net.params1.grad
    print('tl:',net.params1)
    net.params1.grad.data.zero_()
    optimizer.step()
    print('tl:', net.params1)
    optimizer.zero_grad()
    #for name, parameters in net.named_parameters():
    #    print(name, ':', parameters)
    #print(ge_1,ge_2)
    return (output1).item(),ge_1_1,ge_2_1,ge_x_1,ge_x_2,ge_3_1,ge_x_3,output[3]#,ge_1_2,ge_2_2,ge_3_2

loss_model = []
for epoch in range(10000):
    optimizer = optim.Adam(net.parameters(),lr = 0.001)
    t = train()
    print('loss:',t[0])
    #print('z_Grad:',t[-1])
    loss_model.append(t[0])
    if epoch%100 == 1:
        ge_1_1 = t[1]
        ge_2_1 = t[2]
        ge_3_1 = t[5]
        #ge_1_2 = t[8]
        #ge_2_2 = t[9]
        #ge_3_2 = t[10]
        ge_x_1 = t[3]
        ge_x_2 = t[4]
        ge_x_3 = t[6]
        #sns.distplot(ge_1)
        #sns.distplot(ge_2)
        #print(ge_x_1)
        plt.plot(ge_x_1,ge_1_1,'.')
        plt.plot(ge_x_2,ge_2_1,'.')
        plt.plot(ge_x_3,ge_3_1,'.')
        #plt.plot(ge_x_1,ge_1_2,'.')
        #plt.plot(ge_x_2,ge_2_2,'.')
        #plt.plot(ge_x_3,ge_3_2,'.')
        plt.show()
    #print(t[0])


#for name,param in net.named_parameters():
#    print(name,param)
print(loss_model)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

#print(get_parameter_number(net))