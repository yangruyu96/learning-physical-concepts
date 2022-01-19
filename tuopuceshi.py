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

I_3 = np.array([[1,0,0],[0,1,0],[0,0,1]])
I_2 = np.array([[1,0],[0,1]])

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

Map = gen_SPAM()
print(Map)

x1 = []
x2 = []
x3 = []

fig=plt.figure()
ax1 = Axes3D(fig)

initial  = np.array([[1],[0],[0],[1]])
state = initial
circle = []



def base(x,n):
    return math.sqrt(2) * cmath.sin(n * math.pi * x )

def wave(a,b,x):
    return (a * base(x,1) + b * base(x,2))/math.sqrt(abs(a)**2 + abs(b)**2)

def pro(a,b,x):
    return (abs(wave(a,b,x))**2)





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
    for k in range(0, 10):
        # print(k)

        #y_pro = []
        phase = cmath.exp(1j * np.random.random() * 2 * math.pi)
        '''
        
        non_diag = a0_list[k]*b0_list[k]*phase
        tomo1 = (non_diag + non_diag.conjugate())/(abs(a0_list[k])**2 + abs(b0_list[k])**2)
        tomo2 = (1j * non_diag - 1j*(non_diag.conjugate()))/(abs(a0_list[k])**2 + abs(b0_list[k])**2)
        tomo3 = (a0_list[k]**2 - b0_list[k]**2)/(abs(a0_list[k])**2 + abs(b0_list[k])**2)
        y_pro = [tomo1.real,tomo2.real,tomo3.real]
        print(np.random.random() * 2 * math.pi)
        
        for xrange in range(0, 5):
            probability = 0
            for repedx in range(0, 10000):
                probability = pro(a0_list[k], b0_list[k] * phase, xrange / 5 + repedx / 50000) / 50000 + probability
            y_pro.append(probability)
        '''
        non_diag = a0_list[k]*b0_list[k]*phase
        tomo1 = (non_diag + non_diag.conjugate())/(abs(a0_list[k])**2 + abs(b0_list[k])**2)
        tomo2 = (1j * non_diag - 1j*(non_diag.conjugate()))/(abs(a0_list[k])**2 + abs(b0_list[k])**2)
        tomo3 = (a0_list[k]**2 - b0_list[k]**2)/(abs(a0_list[k])**2 + abs(b0_list[k])**2)
        tomo4 = tomo1 + tomo2
        tomo5 = tomo3 + tomo2
        y_pro = [tomo1.real,tomo2.real,tomo3.real,tomo4.real,tomo5.real]

        circle.append(y_pro)

#   state = np.matmul(Map,state)
 #   circle.append([math.sin((i+10)/100), math.cos((i+10)/100),math.sin(i/100),math.cos(i/100)])
 #   x1.append(math.sin(i/100) - 2 * math.sin(3*i/100))
 #   x2.append(math.cos(i/100) - 2 * math.cos(3*i/100))
 #   x3.append(math.sin(i/100))

#ax1.scatter3D(x1,x2,x3, cmap='Blues')  #绘制散点图
#plt.plot(x1,x2)

def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma
    return x


#list = Z_ScoreNormalization(circle,np.mean(circle),np.std(circle))
list = circle


torus_sc = list
torus_c = SimplicialComplex(simplices=torus_sc)
print(torus_c.betti_number(2))
print(torus_c.betti_number(1))
print(torus_c.betti_number(0))





Save_list(list,r'C:\Users\yangruyu\Desktop\code\myfile')



#print(circle)

#print(np.mean(circle),np.std(list))
