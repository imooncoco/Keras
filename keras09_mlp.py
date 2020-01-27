#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(101, 201)])
y = np.array([range(1, 101), range(101, 201)])

print(x.shape)   # (2, 10)
print(y.shape)   # (2, 10)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)   # (10, 2)
print(y.shape)   # (10, 2)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size = 0.6, random_state=66, shuffle=False)

x_val, x_test, y_val, y_test, = train_test_split(x_test, y_test, train_size = 0.6, random_state=66, shuffle=False)



#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim = 1))
# input_dim = 1   1차원
# input_dim = 2   2차원  --> 열이 2개라는 것 의미/ 벡터가 2개이다 의미
model.add(Dense(10, input_shape = (2, )))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))

# model.summary()    # summary에 대한 설명  bias(절편)의 갯수까지 포함하여 계산하기 때문에


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])    # mse, mae
model.fit(x_train, y_train, epochs=200, batch_size=5, validation_data=(x_val, y_val))

#4. 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=5)
print('mse : ', mse)

y_predict = model.predict(x_test, batch_size=5)
print(y_predict)


#              x    |    y
# train   | x_train | y_train
# test    | x_test  | y_test
# val     | x_val   | y_val
# predict | x_pred  |   ?
# y_pred는 없다. 왜? 우리가 알아내야하는 값이기 때문


x_prd=np.array([[301, 302, 303],[304, 305, 306]])
x_prd = np.transpose(x_prd)
aaa=model.predict(x_prd, batch_size=1)
print(aaa)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)



