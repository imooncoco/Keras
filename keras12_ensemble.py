#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101, 201), range(301, 401)])
x2 = np.array([range(1001, 1101), range(1101, 1201), range(1301, 1401)])

y1 = np.array([range(101, 201)])


# y2 = np.array(range(101, 201))

print(x1.shape)
print(x2.shape)
print(y1.shape)
# print(y2.shape)  # (100, )


x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

# x = np.append(x1, x2, axis=1)

# print(x.shape)   # (100, 3)
# print(y.shape)   # (100, 1)


from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test, = train_test_split(x, y1, train_size = 0.6, random_state=66, shuffle=False)
# x_val, x_test, y_val, y_test, = train_test_split(x_test, y_test, train_size = 0.6, random_state=66, shuffle=False)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size = 0.6, random_state=66, shuffle=False)
x1_val, x1_test, x2_val, x2_test, y1_val, y1_test = train_test_split(x1_test, x2_test, y1_test, train_size = 0.6, random_state=66, shuffle=False)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

# input2 = Input(shape=(3,))
# dense21 = Dense(5)(input2)
# dense22 = Dense(2)(dense21)
# dense23 = Dense(3)(dense22)
# output2 = Dense(1)(dense23)

# 이런 형식도 가능하다.
# 아직 히든레이어이기 때문에 가능
input2 = Input(shape=(3,))
dense21 = Dense(7)(input2)
dense23 = Dense(4)(dense21)
output2 = Dense(5)(dense23)

from keras.layers.merge import concatenate
merge1 = concatenate([output1, output2])

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)


model = Model(inputs = [input1, input2], outputs = output)

# model.add(Dense(5, input_dim = 1))
# input_dim = 1   1차원
# input_dim = 2   2차원  --> 열이 2개라는 것 의미/ 벡터가 2개이다 의미
# model.add(Dense(20, input_shape = (3, )))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))


model.summary()    # summary에 대한 설명  bias(절편)의 갯수까지 포함하여 계산하기 때문에


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])    # mse, mae
model.fit([x1_train, x2_train,], y1_train, epochs=100, batch_size=1, validation_data=([x1_val,x2_val], y1_val))

#4. 평가예측
loss, mse = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
print('mse : ', mse)


#              x    |    y
# train   | x_train | y_train
# test    | x_test  | y_test
# val     | x_val   | y_val
# predict | x_pred  |   ?
# y_pred는 없다. 왜? 우리가 알아내야하는 값이기 때문


x1_prd=np.array([[501, 502, 503], [504, 505, 506], [507, 508, 509]])
x2_prd=np.array([[601, 602, 603], [604, 605, 606], [607, 608, 609]])
x1_prd = np.transpose(x1_prd)
x2_prd = np.transpose(x2_prd)

aaa=model.predict([x1_prd,x2_prd], batch_size=1)
print(aaa)

y_predict = model.predict([x1_test, x2_test], batch_size=1)
#print(y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y_predict):
    return np.sqrt(mean_squared_error(y1_test, y_predict))
print('RMSE : ', RMSE(y1_test, y_predict))


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y_predict)
print('R2 : ', r2_y_predict)



