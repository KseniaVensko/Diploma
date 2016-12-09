from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from polygon_actions import load_txt, save_txt, save_float_txt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

epoch_size = 64*4
model = Sequential()
model.add(Dense(12, init='normal', input_dim=9, activation='relu'))
model.add(Dense(18, init='normal'))
model.add(Dense(9, init='normal', activation='relu'))

# choose the optimizer and other variables
model.compile(optimizer='adam', loss='mse')

x,y = load_txt('try.txt')

history = model.fit(x, y, batch_size=1, nb_epoch=epoch_size)

x_test,y_test = load_txt('try2.txt')

train_predict = model.predict(x)
test_predict = model.predict(x_test)

save_float_txt(y_test, 'y_test')
save_float_txt(test_predict, 'test_predict')
save_float_txt(x_test, 'x_test')
save_float_txt(y, 'y')
save_float_txt(train_predict, 'train_predict')
save_float_txt(x, 'x')

score = model.evaluate(x_test, y_test, batch_size=1)
print score

#a = [400, 600, 50, 380, 500, 23, 700, 800, 90]
#pred = model.predict(a)
#print pred
