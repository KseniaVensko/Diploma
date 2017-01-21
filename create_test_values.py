from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras import initializations
from polygon_actions import *
#from shapely.geometry import mapping
from sklearn.preprocessing import MinMaxScaler
inputs=9
outputs=9
# a set of training sets
epoch_size=64*4
epoch = np.zeros(shape=(epoch_size, inputs), dtype=np.int)

#~ scaler = MinMaxScaler(feature_range=(0,1))
#~ x = scaler.transform(x)
#~ predict = scaler.inverse_transform(predict)

# generate correct inputs
for i in range(epoch_size):
	epoch[i][0] = np.random.random_integers(200, 1024)	#h1
	epoch[i][1] = np.random.random_integers(200, 1024)	#w1
	epoch[i][2] = np.random.random_integers(0, 100)		#r1
	
	temp = np.random.random_integers(1024)
	epoch[i][3] = 0 if temp < 200 else temp				#h2
	epoch[i][4] = 0 if epoch[i][3] == 0 else np.random.random_integers(200, 1024)	#w2
	epoch[i][5] = 0 if epoch[i][3] == 0 else np.random.random_integers(0, 100)	#r2
	
	temp = np.random.random_integers(1024)
	epoch[i][6] = 0 if temp < 200 else temp				#h3
	epoch[i][7] = 0 if epoch[i][6] == 0 else np.random.random_integers(200, 1024)	#w3
	epoch[i][8] = 0 if epoch[i][6] == 0 else np.random.random_integers(0, 100)	#r3

print epoch

output = np.zeros(shape=(epoch_size, outputs), dtype=np.int)

#generate correct outputs
for i in range(epoch_size):
	output[i][0] = np.random.random_integers(1024 - epoch[i][1] + 1) - 1	# x1 (1024 - w1)
	output[i][1] = np.random.random_integers(1024 - epoch[i][0] + 1) - 1	# y1 (1024 - h1)
	output[i][2] = np.random.random_integers(360) - 1				# angle1
	
	output[i][3] = 0 if epoch[i][3] == 0 else np.random.random_integers(1024 - epoch[i][4] + 1) - 1
	output[i][4] = 0 if epoch[i][3] == 0 else np.random.random_integers(1024 - epoch[i][3] + 1) - 1
	output[i][5] = 0 if epoch[i][3] == 0 else np.random.random_integers(360) - 1

	output[i][6] = 0 if epoch[i][6] == 0 else np.random.random_integers(1024 - epoch[i][7] + 1) - 1
	output[i][7] = 0 if epoch[i][6] == 0 else np.random.random_integers(1024 - epoch[i][6] + 1) - 1
	output[i][8] = 0 if epoch[i][6] == 0 else np.random.random_integers(360) - 1

print output

def my_init(shape, name=None):
    return initializations.uniform(shape, scale=0.5, name=name)

locate_model = Sequential()
locate_model.add(Dense(12, init=my_init, input_dim=9, activation='relu'))
locate_model.add(Dense(12, init=my_init, activation='relu'))
locate_model.add(Dense(9, init=my_init, activation='relu'))
locate_model.compile(optimizer='adam', loss='mse')


locate_model.fit(epoch, output, nb_epoch=4, batch_size=64)

for i in range(epoch_size):
	epoch[i][0] = np.random.random_integers(200, 1024)	#h1
	epoch[i][1] = np.random.random_integers(200, 1024)	#w1
	epoch[i][2] = np.random.random_integers(0, 100)		#r1
	
	temp = np.random.random_integers(1024)
	epoch[i][3] = 0 if temp < 200 else temp				#h2
	epoch[i][4] = 0 if epoch[i][3] == 0 else np.random.random_integers(200, 1024)	#w2
	epoch[i][5] = 0 if epoch[i][3] == 0 else np.random.random_integers(0, 100)	#r2
	
	temp = np.random.random_integers(1024)
	epoch[i][6] = 0 if temp < 200 else temp				#h3
	epoch[i][7] = 0 if epoch[i][6] == 0 else np.random.random_integers(200, 1024)	#w3
	epoch[i][8] = 0 if epoch[i][6] == 0 else np.random.random_integers(0, 100)	#r3

print epoch
pr = locate_model.predict(epoch, batch_size=64)
print pr
