#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)
print(y.shape)


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])    # mse, mae
model.fit(x, y, epochs=100, batch_size=1)                       # batch_size 바꿔보기(1, 10, 없애보기)   # batch_size의 default값은 32이다.

#4. 평가예측
loss, mae = model.evaluate(x, y, batch_size=1)
print('mae : ', mae)


x_prd=np.array([11, 12, 13])
aaa=model.predict(x_prd, batch_size=1)
print(aaa)

bbb = model.predict(x, batch_size=1)
print(bbb)