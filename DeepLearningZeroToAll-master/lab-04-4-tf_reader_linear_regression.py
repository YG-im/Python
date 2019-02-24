# Lab 4 Multi-variable linear regression
# https://www.tensorflow.org/programmers_guide/reading_data

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'#,'data-01-test-score.csv', ...이렇게 여러개도 가
     ], shuffle=False, name='filename_queue')

#아래는 텍트스파일을 읽을 때 일반적으로 사용할 수 있는 것.
reader = tf.TextLineReader()  # 파일을 읽어올 리더 지 지정.
key, value = reader.read(filename_queue) #key와 value에 지정.

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)
#value를 어떻게 이해할 것인가?


# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
#batch가 일종의 펌프 같이 데이터를 축출해오는것. [xy[0:-1], xy[-1:]]를 batch_size=10개씩 불러온다.


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session. #통상적으로 이렇게 그냥 쓰임
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue. #통상적으로 이렇게 그냥 쓰임
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch]) #펌프질을 해서 데이터를 가져온다.
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

#통상적으로 이렇게 그냥 쓰임
coord.request_stop()
coord.join(threads)

# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

'''
Your score will be  [[185.33531]]
Other scores will be  [[178.36246]
 [177.03687]]
'''
