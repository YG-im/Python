# Lab 3 Minimizing Cost
import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b to compute y_data = W * x_data
# We know that W should be 1
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name="weight")  #W가 변수야.

X = tf.placeholder(tf.float32)   #X랑 Y를 placeholder에다가 입력할거야
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1    #cost함수의 특정 위치의 기울기에 0.1만큼 W를 변화시켜서 또 대입해 보겠다.
gradient = tf.reduce_mean((W * X - Y) * X)  #(1/m)\sum^{m}_{i=1} (W X -Y)X
descent = W - learning_rate * gradient
update = W.assign(descent) #W에다가 decent값을 assign한다.
"""
cost함수를 Minimize하는 방법에는 위에 방법도 있지만 
매번 미분을 해서 reduce_mean을 취할 수 없으니 Gradient Descent Optimizer를 사용한다.
 optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
 train = optimizer.minimize(cost) 
 """



# Launch the graph in a session.
with tf.Session() as sess:   #그래프그릴라면 세션을 열어야해
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(21):
        _, cost_val, W_val = sess.run(  #_ 는 어떠한 변수에도 update를 할당하지 않겠다는거.
            [update, cost, W], feed_dict={X: x_data, Y: y_data}
        )
        print(step, cost_val, W_val)

"""
0 1.93919 [ 1.64462376]
1 0.551591 [ 1.34379935]
2 0.156897 [ 1.18335962]
3 0.0446285 [ 1.09779179]
4 0.0126943 [ 1.05215561]
5 0.00361082 [ 1.0278163]
6 0.00102708 [ 1.01483536]
7 0.000292144 [ 1.00791216]
8 8.30968e-05 [ 1.00421977]
9 2.36361e-05 [ 1.00225055]
10 6.72385e-06 [ 1.00120032]
11 1.91239e-06 [ 1.00064015]
12 5.43968e-07 [ 1.00034142]
13 1.54591e-07 [ 1.00018203]
14 4.39416e-08 [ 1.00009704]
15 1.24913e-08 [ 1.00005174]
16 3.5322e-09 [ 1.00002754]
17 9.99824e-10 [ 1.00001466]
18 2.88878e-10 [ 1.00000787]
19 8.02487e-11 [ 1.00000417]
20 2.34053e-11 [ 1.00000226]
"""
