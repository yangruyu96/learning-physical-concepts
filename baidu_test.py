import os
import platform
import matplotlib.pyplot as plt
import numpy as npy
import paddle
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import dagger
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
H_info = [[1.,'z0'],[-1.,'z1']]
class Linear_solver(paddle.nn.Layer):
    def __init__(self, shape, dtype='float64'):
        super(Linear_solver, self).__init__()
        self.theta = self.create_parameter(shape=shape,
                                           default_initializer=paddle.nn.initializer.Uniform(low=-1.0, high=0.0),
                                           dtype=dtype, is_bias=False)

    def forward(self):
        #print(self.theta)
        #final_state = cir_init(self.theta,2,6)
        #print(cir.run_state_vector()
        cir = UAnsatz(2)
        cir.ry(self.theta[0],0)
        # print(cir)
        cir.run_state_vector()
        loss = cir.expecval(H_info)
        return loss
net = Linear_solver([1])
loss_list = []
repetion = []
opt = paddle.optimizer.Adam(learning_rate = 0.001, parameters = net.parameters())
paddle.seed(0)
#print(net.parameters())
for times in range(0,10000):
    loss = net()[0]
    #print(loss)
    loss.backward()
    opt.minimize(loss)
    #print(opt)
    opt.clear_grad()
    loss_list.append(loss.numpy()[0])
    repetion.append(times)
print(net())
#print('损失函数: ', loss_list)
plt.plot(repetion,loss_list)
plt.show()

