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
from scipy import stats

x1=np.linspace(0,10,1000)
y1=stats.norm(5,1).pdf(x1)
plt.plot(x1,y1)

x3=np.linspace(10,20,1000)
y3=stats.norm(15,1.5).pdf(x3)
plt.plot(x3,y3)
plt.ylim([0,0.5])
plt.show()

