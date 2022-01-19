import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import math
from scipy.linalg import expm,logm
from scipy.linalg import pinv
import torch
def A(i):
    if i==0:
        return np.array([[1,0],[0,1]])
    elif i==1:
        return np.array([[0,1],[1,0]])
    elif i==2:
        return np.array([[0,-1j],[1j,0]])
    elif i==3:
        return np.array([[1,0],[0,-1]])

Matrix1 = np.zeros((16,16))
Matrix2 = np.ones((16,16))*1j
Matrix = Matrix1+ Matrix2
for i in range(0,4):
    for j in range(0,4):
        for k in range(0,4):
            for l in range(0,4):
                x1 = np.trace(np.matmul(np.matmul(np.matmul(A(i), A(k)), A(j)), A(l)))
                Matrix[i*4+j,k*4+l] = x1

a = torch.ones((1,2))

print(torch.norm((a),p=2))






