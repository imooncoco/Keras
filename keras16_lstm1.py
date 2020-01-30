from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape)   #(13, 3)
print(y.shape)   #(13, )

x = x.reshape((x.shape[0], x.shape[1], 1))
print(x.shape)  # (13,3,1)


#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3, 1), return_sequences=True))
model.add(LSTM(10, activation='relu',  return_sequences=True))
model.add(LSTM(10, activation='relu',  return_sequences=True))
model.add(LSTM(10, activation='relu',  return_sequences=True))
model.add(LSTM(10, activation='relu',  return_sequences=True))
model.add(LSTM(10, activation='relu',  return_sequences=True))
model.add(LSTM(10, activation='relu',  return_sequences=True))
model.add(LSTM(10, activation='relu',  return_sequences=True))
model.add(LSTM(20, activation='relu', return_sequences=False))    # 마지막은 reutrn_sequences 안 넣어도 된다. -> return_sequences=False가 defalut  default
model.add(Dense(5, activation='linear'))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])    # mse, mae

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
# loss값을 모니터해서 과적합이 생기면 100번 더 돌고 끊음
# mode=auto loss면 최저값이100번정도 반복되면 정지, acc면 최고값이 100번정도 반복되면 정지
# mode=min, mode=max
model.fit(x, y, epochs=500, batch_size=1, verbose=2, callbacks=[early_stopping])
# verbose = 0 말수가 많은 = 0
# verbose = 1 default, verbose = 2, 3도 있다.


x_input = array([[25, 35, 45],[50, 60, 70],[70, 80, 90],[100, 110, 120]])  # predict용
x_input = x_input.reshape((4, 3, 1))

y_predict = model.predict(x_input)
print(y_predict)
