import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import math
from scipy.linalg import expm,logm
from scipy.linalg import pinv

def P(i,k,h):
    y=0
    for j in range(1,4):
        for l in range(1,4):
            x1=np.trace(np.matmul(np.matmul(np.matmul(A(j),A(i)),A(l)),A(k)))
            x2=np.trace(np.matmul(np.matmul(np.matmul(A(j),A(l)),A(i)),A(k)))
            x3=np.trace(np.matmul(np.matmul(np.matmul(A(i),A(j)),A(l)),A(k)))
            y=y+h[j-1,l-1]*(x1-(1/2)*(x2+x3))
    return y
#这个函数把开系统那一项携程ptm形式
def H_PTM(i,k,H,p):
    x=np.matmul(H,A(i))-np.matmul(A(i),H)
    #x=np.matmul(np.matmul(expm(1j*H*p),A(i)),expm(-1j*H*p))
    return np.trace(np.matmul(x,A(k)))
def M_PTM(i,k,M):
    x=np.matmul(np.matmul(M,A(i)),(M.T).conjugate())
    return np.trace(np.matmul(x,A(k)))
def A(i):
    if i==0:
        return np.array([[1,0],[0,1]])
    elif i==1:
        return np.array([[0,1],[1,0]])
    elif i==2:
        return np.array([[0,-1j],[1j,0]])
    elif i==3:
        return np.array([[1,0],[0,-1]])

def chai(M):
    kai1 = np.zeros((4,4))
    kai2 = np.ones((4,4))*1j
    kai = kai1 + kai2
    for m in range(0,4):
        for n in range(0,4):
            for i in range(0,4):
                for j in range(0,4):
                    kai[m,n] = kai[m,n] + M[i,j] * np.trace(np.matmul(np.matmul(np.matmul(A(m),A(i)),A(n)),A(j)))/2
    return (kai-kai2)/4


I_3 = np.array([[1,0,0],[0,1,0],[0,0,1]])
I_2 = np.array([[1,0],[0,1]])

h = np.array([[1/2,-math.pi/4,0],[-math.pi/4,1/2,0],[0,0,0]])
#h = np.matmul(h,h)
#h = np.array([[1/2,0,0],[0,1/2,0],[0,0,1/2]])
L1 = np.zeros((4, 4))
L2 = np.ones((4,4))*1j
L = L1 +L2
for i in range(0, 4):
    for j in range(0, 4):
        L[i, j] = P(i,j,h)
p = 1
H_0 = np.zeros((4, 4))
H_1 = 1j * np.zeros((4, 4))
H_0 = H_0 + H_1
H_new = (-math.pi/4)*A(3)
for i in range(0, 4):
    for j in range(0, 4):
        #print(H_PTM(i, j, H_new, p))
        H_0[i, j] = -1j * H_PTM(i, j, H_new, p) / 2
q = expm( H_0 * p + (L) * p * 0.01)
#print(q)
N = np.diag([1,1,0.5,0.5])
U = expm(H_0)
M = np.matmul(N,U)
q_log = logm(M)
#print(M)
#print(q)

#print(q_log)
#print(H_0 + L*0.01)
#print(q_log - H_0)
eig,trans = np.linalg.eig(q_log)
#print(np.matmul(np.matmul(trans,np.diag(eig)),np.linalg.inv(trans)) - q_log )
for i in range(-200,200):
    for j in range(-200,200):
        for k in range(-100,100):
            eig, trans = np.linalg.eig(q_log)
            x1 = 2*math.pi*1j*i
            x2 = 2*math.pi*1j*j
            x3 = 2*math.pi*1j*k
            eig[0] = eig[0] +x1#+ 2*math.pi*1j*random.randint(-10,10)
            eig[1] = eig[1] +x2#+ 2 * math.pi * 1j * random.randint(-10, 10)
            eig[2] = eig[2] +x3#+ 2 * math.pi * 1j * random.randint(-10, 10)
            q_log_new = np.matmul(np.matmul(trans,np.diag(eig)),np.linalg.inv(trans))
            #print(expm(q_log_new) - M)
            process = chai(q_log_new)
            process = np.delete(process,0,axis=0)
            process = np.delete(process,0,axis=1)
            a,b = np.linalg.eig(process)
            print(i,j,k)
            print('a:',a)
            if a[0].real > 0 and a[1].real>0 and a[2].real>0:
                print('eig:',eig)
                break

#q_0 = expm( H_0 * p)
#print(q_0)
#q_log = logm(q_0)
#print(q_log)
#print(H_0)
#print('L:',L)

#print('N:',np.matmul(q,np.linalg.inv(q_0)))