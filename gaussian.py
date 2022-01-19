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


def Q(z):
    return 1

def T(z):
    return -z**2


def L(u):
    return u

def TL(z,u):
    return - z**2 * ((u)**2)

z = sp.symbols('z')
y0 = np.float(sp.integrate(Q(z)*(sp.E**(TL(z,1))),(z,-sp.oo,sp.oo)))
z = sp.symbols('z')
y1 = np.float(sp.integrate(Q(z)*(sp.E**(TL(z,2))),(z,-sp.oo,sp.oo)))
z = sp.symbols('z')
y2 = np.float(sp.integrate(Q(z)*(sp.E**(TL(z,3))),(z,-sp.oo,sp.oo)))


def pdf(z0,u,Q,TL,l):
    if l==0:
        return Q(z0)*(math.e**(TL(z0,u)))/np.float(y0)
    if l==1:
        return Q(z0)*(math.e**(TL(z0,u)))/np.float(y1)
    if l==2:
        return Q(z0) * (math.e ** (TL(z0, u))) / np.float(y2)

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
        self.params1 = nn.Parameter(torch.tensor([[-1.0],[-4.0]]),requires_grad=False)
        #self.params2 = torch.autograd.Variable(torch.rand(1,1,out=None).cuda(),requires_grad=True)
        self.linear_1 = nn.Linear(10, 10)
        self.linear_2 = nn.Linear(10,10)
        self.linear_u_1 = nn.Linear(1,10)
        self.linear_u_2 = nn.Linear(10,1)
        self.linear_z_1 = nn.Linear(1,10)
        self.linear_z_2 = nn.Linear(10,1)
        self.encoder_1 = nn.Linear(4,50)
        self.encoder_2 = nn.Linear(50,2)
        self.Sigmoid = nn.ReLU()
        self.Sigmoid_T = nn.Tanh()
        self.linear_Q_1 = nn.Linear(1,10)
        self.linear_Q_2 = nn.Linear(10,1)
        self.linear_x_1 = nn.Linear(1,10)
        self.linear_x_2 = nn.Linear(10,2)
        self.nonlear_1 = nn.Linear(1,10)
        self.nonlear_2 = nn.Linear(10,1)
# forward 定义前向传播
    def decoder_1(self,z,n):
        n = n.cpu().detach().numpy() - 1
        u = self.params1[n]
        u0=u.cpu().detach().numpy()
        #print(u0)
        z0 = torch.linspace(-100,100,steps=10**4).cuda()
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
        ge_z=[]
        for times in range(100):
            #ge_z = []
            #zr = torch.autograd.Variable(torch.rand(1,1,out=None).cuda(),requires_grad=True)
            #print(times)
            #print(z0)
            #print(u,z0)
            #print(y.size())
            encoder = torch.cat((u0,x),1)
            #print(encoder)
            encoder = self.encoder_1(encoder)
            encoder = self.Sigmoid(encoder)
            encoder = self.encoder_2(encoder)
            #encoder = self.Sigmoid(encoder)
            #print(encoder)
            mean = encoder[0,0]
            variance = encoder[0,1]
            repa = torch.randn(1,1,out=None).cuda()
            encoder = mean + repa * variance
            #z_generate.append(encoder)
            #encoder.to(torch.device('cpu'))
            #print(encoder)
            #encoder.backward(retain_graph=True)
            #z_cl_grad.append(zr.grad)
            #print(z0.grad)
            n = u0.cpu().detach().numpy() - 1
            ud1 = self.params1[n].cuda()
            #u0 = ud1.cpu().detach().numpy()
            # print(u0)
            #z0 = torch.linspace(-100, 100, steps=10 ** 4).cuda()
            #l = torch.exp( (z0 ** 2) * ud1[0, 0])
            #dx = np.float(z0[1] - z0[0])
            sigma = torch.sqrt(-1/(2*ud1[0,0]))
            if ud1[0,0] > 0:
                print('ud1:',0)
            normal =  1/(sigma * torch.sqrt(2*torch.tensor(math.pi)))
            #for i in range(10**4):
                #normal = normal + l[i] * dx
            # normal = np.float(sp.integrate(l,(z0,-sp.oo,sp.oo)))
            # print(normal)
            l1 = torch.exp((encoder ** 2) * (ud1[0, 0]))/(normal.cuda())
            #l1 = self.decoder_1(encoder,u0)
            #l2 = self.decoder_2(encoder)
            l2 = self.linear_x_1(encoder)
            # latent2 = self.Sigmoid(latent2)
            l2 = self.linear_x_2(l2)
            nonli = self.nonlear_1(encoder)
            nonli = self.Sigmoid(nonli)
            nonli = self.nonlear_2(nonli)
            l2 = torch.cat((l2, nonli), 1)
            if variance == 0:
                print('variance:',0)
            loss1 = torch.exp(- repa**2/2)/(torch.sqrt(2*torch.tensor(math.pi))*abs(variance))
            #print(times,loss1)
            #loss1 = 1/np.abs((zr.grad).cpu())
            loss2 = - torch.log(l1) + torch.norm((l2 - x),p=2)*200
            if l1 ==0:
                print('l1:',l1,'normal:',normal,'sigma:',sigma,'u:',torch.exp((encoder ** 2) * (ud1[0, 0])))
            if loss1==0:
                print('loss1:',0)
            loss= loss + loss2 + torch.log(loss1.cuda())
            #print(loss)
            #loss.backward(retain_graph=True)
            #print(self.params1.grad)
            #zr.grad.data.zero_()
            ge_z.append(encoder)
            #if torch.isnan(loss) == True:
            #    print(loss1,loss2)
        #print('loss1:',loss1,'loss2:',loss2)
        #print('loss1:',loss1,'l1:',l1,'l2:',torch.norm((l2 - x),p=2))
        #print('Tl:',self.params1,'loss:',loss)
        #print('z0grad',z_cl_grad)
        #print('l1:',l1,'encoder:',encoder)
        #print(loss/100)
        #print(loss)
        return loss/100,ge_z

net = LinearNet().cuda()

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
    x0 = x0_list[0:10]
    x0 = torch.tensor(x0).view(10,1)  #+torch.randn(300,1)/10
    x1 = x1_list[0:10]
    x1 = torch.tensor(x1).view(10,1) #+torch.randn(300,1)/10
    u0 = torch.autograd.Variable(torch.ones(10, 1), requires_grad=True)
    u1 = u0 * 2
    x = torch.cat((x0, x1), 0).cuda()
    x_new1 = x * 2 + 4
    x_new2 = x ** 2
    x = torch.cat((x, x_new1), 1).cuda()
    x = torch.cat((x, x_new2), 1).cuda() + (torch.randn(20, 3) / 10).cuda()
    u = torch.cat((u0, u1), 0).cuda()
    mydataset = Data.TensorDataset(x, u)
    data_loader = Data.DataLoader(dataset=mydataset, batch_size=1, shuffle=True, num_workers=0)
    for step,(batch_x,batch_y) in enumerate(data_loader):
        ge_1 = []
        ge_2 = []
        #print(batch_x)
        #print(batch_y)
        output = net(batch_x, batch_y)
        #print(output1)
        output1 = output[0] + output1
        if batch_y == 1:
            ge_1.extend(output[1])
        if batch_y == 2:
            ge_2.extend(output[1])
        #sns.distplot(ge_1)
        #print(ge_1)
        #plt.show()
    optimizer.zero_grad()
    #print(output1)
    output1 = output1/20
    (output1).backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    return (output1),ge_1,ge_2

loss_model = []
for epoch in range(100000):
    optimizer = optim.Adam(net.parameters(),lr = 0.1)
    t = train()
    loss_model.append(t[0])
    print(t[0])
    if epoch%10 == 1:
        ge_1 = t[1]
        ge_2 = t[2]
        #print(ge_1)
        sns.distplot(ge_1)
        sns.distplot(ge_2)
        plt.show()

#for name,param in net.named_parameters():
#    print(name,param)
print(loss_model)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

#print(get_parameter_number(net))