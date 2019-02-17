#3층 신경망

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identify_function(x):
    return x

def init_network():
    network={}
    network['w1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1, 0.2,0.3])
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5],[0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network,x):
    w1, w2, w3 = network["w1"], network["w2"], network["w3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    y=identify_function(a3)

    return y

def softmax(a):
    c=np.max(a) #cut-off
    exp_a=np.exp(a-c) # 최댓값을 빼줌으로써 inf가 나오는 것을 막음. CS에서는 너무 큰 값도 다 inf로 인식.
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y


network=init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y)