import numpy as np
from scipy.sparse import dok_matrix
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

def sparse_transform(raw_x):
    x = dok_matrix((raw_x.shape[0], 7017))
    for i in range(x.shape[0]):
        x[i, int(raw_x[i, 0])] = 1
        x[i, int(raw_x[i, 1]) + 6000] = 1
        x[i, int(raw_x[i, 2]) + 7000] = 1
        x[i, int(raw_x[i, 3]) + 7007] = 1
    return x

data_x = data_x.reshape((-1, 4))
alldata_x = alldata_x.reshape((-1, 4))
# data_y = data_y.reshape((-1, 1))

tr_data_x = sparse_transform(data_x)
tr_alldata_x = sparse_transform(alldata_x)

train_x = tr_data_x[:, :]
train_y = data_y[:]

# train_x = tr_data_x[:250000, :]
# train_y = data_y[:250000, :]

test_x = tr_data_x[250000:, :]
test_y = data_y[250000:]

from sklearn.neural_network import MLPRegressor
from sklearn import metrics

regressor = MLPRegressor(hidden_layer_sizes = (500, 100), activation = 'tanh', solver = 'lbfgs', tol = 5e-6, verbose = True)
# regressor = MLPRegressor(hidden_layer_sizes = (1, ), activation = 'tanh', solver = 'lbfgs', verbose = True)

print('Start training ...')
regressor.fit(train_x, train_y)
print('Training done.')

test_pred = regressor.predict(test_x)
print("Rooted Mean Squared Error: %s\n" % (np.sqrt(metrics.mean_squared_error(test_y, test_pred))))

pred = []   
for i in range(0, tr_alldata_x.shape[0], 100000):
    print('Prediction:', i)
    if i + 100000 <= tr_alldata_x.shape[0]:
        batch_x = tr_alldata_x[i : i + 100000]
    else:
        batch_x = tr_alldata_x[i:]
    batch_pred = regressor.predict(batch_x)
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