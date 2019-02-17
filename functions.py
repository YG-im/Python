
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def identify_function(x):
    return x

def softmax(a):
    c=np.max(a) #cut-off
    exp_a=np.exp(a-c) # 최댓값을 빼줌으로써 inf가 나오는 것을 막음. CS에서는 너무 큰 값도 다 inf로 인식.
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y