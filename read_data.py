import csv
import numpy as np

with open('data/training.csv') as f:
	f_csv = csv.DictReader(f)
	data_x = []
	data_y = []
	for row in f_csv:
		data_x.append(row['cat#slave#weekday#food'].split('#'))
		data_y.append(row['pred'])

data_x = np.array(data_x).astype(int)
data_y = np.array(data_y).astype(float)

with open('data/sample.csv') as f:
	f_csv = csv.DictReader(f)
	alldata_x = []
	for row in f_csv:
		alldata_x.append(row['cat#slave#weekday#food'].split('#'))
		
alldata_x = np.array(alldata_x).astype(int)

print('Data reading complete.')

import pickle

data_filename = 'data.pkl'
print('Dumping data to file \'' + data_filename + '\' ...')
with open(data_filename, 'wb') as f:
    pickle.dump(data_x, f)
    pickle.dump(data_y, f)
    pickle.dump(alldata_x, f)
print('Data dumping complete.')
