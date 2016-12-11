from keras.models import Sequential
from keras.layers import Dense
import numpy as np

epoch_size = 64*4
model = Sequential()
# input: n h1 w1 x1 y1 a1 h2 w2 x2 y2 a2 h3 w3 x3 y3 w3 (r123?)
model.add(Dense(12, init='normal', input_dim=7, activation='relu'))
model.add(Dense(12, init='normal'))
model.add(Dense(9, init='normal', activation='relu'))
# output: x1+d1 y1+d2 a1+d3 x2+d4 y2+d5 a2+d6 x3+d7 y3+d8 w3+d9

model.compile(optimizer='adam', loss='mse')
