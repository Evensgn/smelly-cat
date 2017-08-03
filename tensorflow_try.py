import numpy as np
import pickle

data_filename = 'data.pkl'
print('Loading data from file \'' + data_filename + '\' ...')
with open(data_filename, 'rb') as f:
    data_x = pickle.load(f)
    data_y = pickle.load(f)
    alldata_x = pickle.load(f)
print('Data loading complete.')

ans_dict = {}
for i in range(data_x.shape[0]):
    ans_dict[('#').join(list(data_x[i].astype(str)))] = data_y[i]

data_x = data_x.reshape((-1, 4))
alldata_x = alldata_x.reshape((-1, 4))
data_y = data_y.reshape((-1, 1))

# train_x = data_x[:, :]
# train_y = data_y[:, :]

train_x = data_x[:250000, :]
train_y = data_y[:250000, :]

test_x = data_x[250000:, :]
test_y = data_y[250000:, :]

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

L = [7017, 1]
LAYERS = len(L) - 1

learning_rate = 1e-1
iterations = 100000
batch_size = 1000
regular_lambda = 0 #1e-5
drop_keep_prob = 0.90

W = list(range(LAYERS + 1))
b = list(range(LAYERS + 1))
for i in range(LAYERS):
    W[i + 1] = weight_variable([L[i], L[i + 1]])
    b[i + 1] = bias_variable([L[i + 1]])

keep_prob = tf.placeholder("float")
x = tf.placeholder(tf.float32, [None, 7017])
yt = list(range(LAYERS + 1))
yt[0] = x
for i in range(LAYERS):
    if i == LAYERS - 1:
        yt[i] = tf.nn.dropout(yt[i], keep_prob)
    yt[i + 1] = tf.matmul(yt[i], W[i + 1]) + b[i + 1]
    # if i + 1 < LAYERS:
    #     yt[i + 1] = tf.nn.relu6(yt[i + 1])
y = yt[LAYERS]
# y = tf.tanh(yt[LAYERS]) * 11. + 9.
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

def xraw_transform(raw_x):
    x = np.zeros((raw_x.shape[0], 7017))
    for i in range(x.shape[0]):
        x[i, int(raw_x[i, 0])] = 1
        x[i, int(raw_x[i, 1]) + 6000] = 1
        x[i, int(raw_x[i, 2]) + 7000] = 1
        x[i, int(raw_x[i, 3]) + 7007] = 1
    return x

mse_result = tf.reduce_mean(tf.square(y - y_))

def test_rmse():
    mse_arr = []
    for i in range(0, test_x.shape[0], 100):
        if i + 100 <= test_x.shape[0]:
            batch_x = xraw_transform(test_x[i : i + 100])
            batch_y_ = test_y[i : i + 100]
        else:
            batch_x = xraw_transform(test_x[i:])
            batch_y_ = test_y[i:]
        mse_arr.append(sess.run(mse_result, feed_dict = {x: batch_x, y_: batch_y_, keep_prob: 1.0}))
    return np.sqrt(np.mean(mse_arr))

if False:
    sess.run(train_step, feed_dict = {x: train_x, y_: train_y})
else:
    for i in range(iterations):
        raw_batch_x, batch_y_ = new_batch(batch_size)
        batch_x = xraw_transform(raw_batch_x)
        sess.run(train_step, feed_dict = {x: batch_x, y_: batch_y_, keep_prob: drop_keep_prob})
        if i % (iterations // 100) == 0:
            print('RMSE:', test_rmse())
            print('Process: {}%'.format((i // (iterations // 100) + 1) * 1))

print('RMSE:', test_rmse())

pred = []   
for i in range(0, alldata_x.shape[0], 100):
    if i + 100 <= alldata_x.shape[0]:
        batch_x = xraw_transform(alldata_x[i : i + 100])
    else:
        batch_x = xraw_transform(test_x[i:])
    batch_pred = sess.run(y, feed_dict = {x: batch_x, keep_prob: 1.0})
    for pred_i in batch_pred:
        pred.append(pred_i)

record_info = []
for info in alldata_x:
    record_info.append(('#').join(list(info.astype(str))))

for i in range(len(pred)):
    if record_info[i] in ans_dict:
        pred[i] = ans_dict[record_info[i]]

import csv

headers = ['cat#slave#weekday#food', 'pred']
rows = list(zip(record_info, pred))
with open('ans.csv', 'w', encoding = 'utf-8') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
