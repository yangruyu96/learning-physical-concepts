import numpy as np
from mogutda import SimplicialComplex
import argparse
import errno
import os
import gudhi
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x=[]
y=[]
z=[]
with open(r'C:\Users\yangruyu\Desktop\code\myfile.txt') as f:
    data = f.readlines()
    list_source = []
    for line in data:
        numbers = line.split()  # 将数据分隔
        # print(numbers)
        numbers_float = list(map(float, numbers))  # 转化为浮点数
        x.append(numbers_float[2])
        y.append(numbers_float[1])
        z.append(numbers_float[0])
        list_source.append(numbers_float)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()




torus_sc = list_source
torus_c = SimplicialComplex(simplices=torus_sc)

print(torus_c.betti_number(2))
