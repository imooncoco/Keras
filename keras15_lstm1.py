from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
y = array([4, 5, 6, 7, 8])

print(x.shape)   #(5, 3)
print(y.shape)   #(5,)

x = x.reshape(x.shape[0], x.shape[1], 1)
# x = x.reshape(5, 3, 1)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3, 1)))
# LSTM에서는 몇개씩 자르는지가 중요하다.
# 자르는 개수는 input_shape=()에 들어간다.
# input_shape(3, 1)에서 3은 열을 의미, 1은 자르는 개수 의미
# 1, 2, 3, ... 이렇게 자른다는 것 의미.
# 1 2, 3 4, ... 이렇다면 input_shape(3, 2)이다.
# 10은 출력노드를 의미 (히든레이어)
# x = (5, 3)이나 input_shape에 들어가야하는 것은 (5, 3, 1)이다.
# (5, 3, 1)에서 5는 행 무시
model.add(Dense(5))
model.add(Dense(1))

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mae'])    # mse, mae

model.fit(x, y, epochs=140, batch_size=1)

#4. 평가예측
loss, mae = model.evaluate(x, y, batch_size=1)
#aaa = model.evaluate(x, y, batch_size=1)
print(loss, mae)

x_input = array([6, 7, 8])    # (3, ) -> (1, 3) -> (1, 3, 1)
x_input = x_input.reshape(1, 3, 1)

y_predict = model.predict(x_input)
print(y_predict)


'''
compile
fit
evaluate
predict
'''