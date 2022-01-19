import os
import platform
import matplotlib.pyplot as plt
import numpy as npy
import paddle
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import dagger
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
A = paddle.to_tensor(npy.array([[1.0,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=npy.float32))
state = paddle.to_tensor(npy.array([[1],[0],[0],[0]],dtype=npy.float32))
identity = paddle.to_tensor(npy.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=npy.float32))
state_matrix = paddle.matmul(state,dagger(state))
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
        cir.ry(self.theta[1],1)
        cir.cnot([0,1])
        #print(cir)
        # print(cir)
        final_state = cir.run_density_matrix()
        H_G = paddle.matmul(paddle.matmul(dagger(A),identity - state_matrix),A)
        #final_state = paddle.reshape(final_state,[1,4])
        #final_state_dagger = paddle.reshape(final_state,[4,1])
        loss = paddle.real(paddle.trace(paddle.matmul(final_state,H_G)))
        return loss,final_state
net = Linear_solver([2])
loss_list = []
repetion = []
opt = paddle.optimizer.Adam(learning_rate = 0.001, parameters = net.parameters())
paddle.seed(0)
#print(net.parameters())
for times in range(0,5000):
    loss = net()[0]
    #print(loss)
    loss.backward()
    opt.minimize(loss)
    #print(opt)
    opt.clear_grad()
    loss_list.append(loss.numpy()[0])
    repetion.append(times)
#print('损失函数: ', loss_list)
print(net()[1])
plt.plot(repetion,loss_list)
plt.show()

