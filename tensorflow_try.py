import numpy as np
import pickle

data_filename = 'data.pkl'
print('Loading data from file \'' + data_filename + '\' ...')
with open(data_filename, 'rb') as f:
    data_x = pickle.load(f)
    data_y = pickle.load(f)
    alldata_x = pickle.load(f)
print('Data loading complete.')

train_x = data_x[:250000, :]
train_y = data_y[:250000, :]

test_x = data_x[250000:, :]
test_y = data_y[250000:, :]

import tensorflow as tf

L = [4, 20, 1]
LAYERS = len(L) - 1

learning_rate = 2e-3
iterations = 50000
batch_size = 100
regular_lambda = 1e-3
drop_keep_prob = 0.90

W = list(range(LAYERS + 1))
b = list(range(LAYERS + 1))
for i in range(LAYERS):
    W[i + 1] = weight_variable([L[i], L[i + 1]])
    b[i + 1] = bias_variable([L[i + 1]])

keep_prob = tf.placeholder("float")
x = tf.placeholder(tf.float32, [None, 4])
yt = list(range(LAYERS + 1))
yt[0] = x
for i in range(LAYERS):
    if i == LAYERS - 1:
        yt[i] = tf.nn.dropout(yt[i], keep_prob)
    yt[i + 1] = tf.nn.relu(tf.matmul(yt[i], W[i + 1]) + b[i + 1])
y = tf.nn.softmax(yt[LAYERS])
y_ = tf.placeholder("float", [None, 1])

l2_loss = 0
for i in range(LAYERS):
    l2_loss += tf.nn.l2_loss(W[i + 1])

rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_)))
cost_function = rmse + regular_lambda * l2_loss

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
init = tf.global_variables_initializer()

print('Start training ...')
print('Neural Network Layers:', L)
print('Learning Rate:', learning_rate)
print('Iterations:', iterations)
print('Batch Size:', batch_size)
print('Regularization lambda:', regular_lambda)
print('Dropout Keep Probability:', drop_keep_prob)

sess = tf.Session()
sess.run(init)

def new_batch(batch_size):
    batch_idx = np.random.choice(range(train_x.shape[0]), size = batch_size, replace = False)
    batch_x = np.zeros((batch_size, 4))
    batch_y_ = np.zeros((batch_size, 1))
    for i in range(batch_size):
        batch_x[i] = train_x[batch_idx[i]]
        batch_y_[i] = train_y[batch_idx[i]]
    return batch_x, batch_y_

mse_result = tf.reduce_mean(y - y_)

def test_rmse():
    mse_arr = []
    for i in range(0, test_x.shape[0], 100):
        mse_arr.append(sess.run(mse_result, feed_dict = {x: test_x[i : i + 100], y_: test_y[i : i + 100], keep_prob: 1.0}))
    return np.sqrt(np.mean(mse_arr))

if False:
    sess.run(train_step, feed_dict = {x: train_x, y_: train_y})
else:
    for i in range(iterations):
        batch_x, batch_y_ = new_batch(batch_size)
        sess.run(train_step, feed_dict = {x: batch_x, y_: batch_y_, keep_prob: drop_keep_prob})
        if i % (iterations // 100) == 0:
            print('RMSE:', test_rmse())
            print('Process: {}%'.format((i // (iterations // 100) + 1) * 1))

print('RMSE:', test_rmse())
