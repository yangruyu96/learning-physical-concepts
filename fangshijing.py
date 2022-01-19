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


h_bar = 1.05457266 * (10**(-34))

h = 6.62607015 * (10**(-34))

m = 6.6969 * (10**(-27))

k = 1.380649 * (10**(-23))

L = 10**(-8)

T = 70

energy0 = (math.pi**2) * (h_bar**2) / (2 * m * (L**2))

rebochang = h * ((2*math.pi * m * k * T )**(-1/2))
print(rebochang)
State0 = []
sum0 = 0

entropy = 0
for i in range(1,1000000):
    sum0 = sum0 + math.exp(-4*(i**2)*energy0/(k * T))

for i in range(1,1000000):
    State0.append(math.exp(-4 * (i ** 2) * energy0 / (k * T))/sum0)
    #print(math.log(math.exp(-4 * (i ** 2) * energy0 / (k * T))))
    if State0[-1]> 0:
        entropy = entropy - (k * State0[-1]) * (math.log(State0[-1]))



print(State0)
print(entropy)

print(k*math.log(2) / (entropy+ k * math.log(2)))