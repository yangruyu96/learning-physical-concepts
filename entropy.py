from matplotlib import pyplot as plt
import numpy as np
import math
#N = 10**23
x1 = []
y1 = []
for i in range(0,400):
    R = 8.314492
    T1 = 20
    T2 = 270
    T4 = 20+i
    T3 = (T1+T4)/2
    V = 1
    k = 1.380649 * (10**(-23))
    m = 6.6969 * (10**(-27))
    h = 6.62607015 * (10**(-34))
    S1 = 3/2 * R * math.log(T1) + R * math.log(k * T1 / 100000) + 3/2 * R * (5/3 + math.log(2*math.pi * m * k / (h**2)))
    S2 = 3/2 * R * math.log(T2) #+ R * math.log(V / N)
    S4 = 3/2 * R * math.log(T4) + R * math.log(k * T1 / 100000) + 3/2 * R * (5/3 + math.log(2*math.pi * m * k / h**2))
    S3 = 2 * 3/2 * R * math.log(T3) + 2 * R * math.log(k * T1 / 100000) + 2 * 3/2 * R * (5/3 + math.log(2*math.pi * m * k / h**2))
    x1.append(i)
    y1.append((S3 - S1 - S4)/(S1 + S4))

x2 = []
y2 = []
for i in range(0,400):
    R = 8.314492
    T1 = 20
    T2 = 270
    T4 = 20+i
    T3 = (T1+T4)/2
    V = 1
    k = 1.380649 * (10**(-23))
    m = 6.6969 * (10**(-27))
    h = 6.62607015 * (10**(-34))
    S1 = 3/2 * R * math.log(T1) + R * math.log(k * T1 / 1000000) + 3/2 * R * (5/3 + math.log(2*math.pi * m * k / (h**2)))
    S2 = 3/2 * R * math.log(T2) #+ R * math.log(V / N)
    S4 = 3/2 * R * math.log(T4) + R * math.log(k * T1 / 1000000) + 3/2 * R * (5/3 + math.log(2*math.pi * m * k / h**2))
    S3 = 2 * 3/2 * R * math.log(T3) + 2 * R * math.log(k * T1 / 1000000) + 2 * 3/2 * R * (5/3 + math.log(2*math.pi * m * k / h**2))
    x2.append(i)
    y2.append((S3 - S1 - S4)/(S1 + S4))




plt.plot(x1,y1)
plt.plot(x2,y2)
plt.show()
print((S3 - S1 - S4)/(S1 + S4))
la = ((h**2) / (2*math.pi * m * k * T1))**(1/2)
print(la)
print(S1)
print(3/2 * R * (5/3 + math.log(2*math.pi * m * k / h**2)))


