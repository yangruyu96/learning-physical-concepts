from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import fsolve

from sympy import *


N = 10**23
m = 6.6969 * (10**(-27))
h_bar = 1.05457266 * (10 **(-34))
V = 1
Solution = N * (2**(1/2)) * (math.pi**2) * (h_bar**3) / (3 * V * (m**(3/2) ))
k = 1
T1 = 70
T2 = 470
beta1 = 1/(k * T1)
beta2 = 1/(k * T2)
print(Solution)
def number1(u):
    y = quad(lambda x : x**(1/2)/(math.exp(beta1*(x-u))-1), 0, 500)[0] - Solution
    #print(y)
    return y

result1 = fsolve(number1, 0, xtol=1.49012e-08)
#print(np.float(result))
#print(number(np.float(result)))

E1 = (N / Solution) * quad(lambda x : x**(3/2)/(math.exp(beta1*(x-np.float(result1)))-1), 0, 500)[0]


def number2(u):
    y = quad(lambda x : x**(1/2)/(math.exp(beta2*(x-u))-1), 0, 500)[0] - Solution
    #print(y)
    return y


result2 = fsolve(number2, 0, xtol=1.49012e-08)

E2 = (N / Solution) * quad(lambda x : x**(3/2)/(math.exp(beta2*(x-np.float(result2)))-1), 0, 500)[0]


beta3 = Symbol('beta3')
u = Symbol('u')
x = Symbol('x')
y1 = integrate(x**(1/2)/(math.exp(beta3*(x-u))-1), 0, 500)[0] - Solution
y2 = integrate(x**(3/2)/(math.exp(beta3*(x-u))-1), 0, 500)[0] - E1 - E2
result3 = solve([y1,y2],[beta3,u])


print(result3)

