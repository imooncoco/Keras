#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101, 201), range(301, 401)])
# x2 = np.array([range(1001, 1101), range(1101, 1201), range(1301, 1401)])
# y1 = np.array([range(101, 201)])

y1 = np.array([range(1, 101), range(101, 201), range(301, 401)])
y2 = np.array([range(1001, 1101), range(1101, 1201), range(1301, 1401)])
y3 = np.array([range(1, 101), range(101, 201), range(301, 401)])


# print(x1.shape)
# print(x2.shape)
# print(y1.shape)
# print(y2.shape)  # (100, )

x1 = np.transpose(x1)
# x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

# x = np.append(x1, x2, axis=1)

# print(x.shape)   # (100, 3)
# print(y.shape)   # (100, 1)


from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test, = train_test_split(x, y1, train_size = 0.6, random_state=66, shuffle=False)
# x_val, x_test, y_val, y_test, = train_test_split(x_test, y_test, train_size = 0.6, random_state=66, shuffle=False)

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x1, y1, y2, y3, train_size = 0.6, random_state=66, shuffle=False)
x1_val, x1_test, y1_val, y1_test, y2_val, y2_test, y3_val, y3_test = train_test_split(x1_test, y1_test, y2_test, y3_test, train_size = 0.6, random_state=66, shuffle=False)

print(y3_test.shape)
print(y3_train.shape)
print(y3_val.shape)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(3)(dense1)
output1 = Dense(4)(dense2)

# input2 = Input(shape=(3,))
# dense21 = Dense(5)(input2)
# dense22 = Dense(2)(dense21)
# dense23 = Dense(3)(dense22)
# output2 = Dense(1)(dense23)

# 이런 형식도 가능하다.
# 아직 히든레이어이기 때문에 가능
# input2 = Input(shape=(3,))
# dense21 = Dense(7)(input2)
# dense23 = Dense(4)(dense21)
# output2 = Dense(5)(dense23)

# from keras.layers.merge import concatenate
# merge1 = concatenate([output1, output2])

# middle1 = Dense(4)(merge1)
# middle2 = Dense(7)(middle1)
# middle3 = Dense(1)(middle2)   # 현재 merge된 마지막 레이어

output_1 = Dense(2)(output1)  # 1번째 아웃풋 모델
output_1 = Dense(3)(output_1)

output_2 = Dense(4)(output1) # 2번째 아웃풋 모델
output_2 = Dense(4)(output_2)
output_2 = Dense(3)(output_2)

output_3 = Dense(5)(output1)  # 3번째 아웃풋 모델
output_3 = Dense(3)(output_3)


model = Model(inputs = input1, outputs = [output_1, output_2, output_3])

model.summary()    # summary에 대한 설명  bias(절편)의 갯수까지 포함하여 계산하기 때문에



#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])    # mse, mae
# metrics에 mae를 넣으면 2가지 지표확인가능. acc도 넣을 수 있다 -> 그러나 acc는 분류모델에서 사용해야한다.
model.fit(x1_train, [y1_train, y2_train, y3_train], epochs=100, batch_size=1, validation_data=(x1_val, [y1_val, y2_val, y3_val]))

#4. 평가예측
#loss, mse = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=1)
aaa = model.evaluate(x1_test, [y1_test, y2_test, y3_test], batch_size=1)
print('aaa : ', aaa)
# 1. 변수를 1개
# 2. 변수를 mse 갯수별로

#print('mse : ', mse)


#              x    |    y
# train   | x_train | y_train
# test    | x_test  | y_test
# val     | x_val   | y_val
# predict | x_pred  |   ?
# y_pred는 없다. 왜? 우리가 알아내야하는 값이기 때문


x1_prd=np.array([[501, 502, 503], [504, 505, 506], [507, 508, 509]])
#x2_prd=np.array([[601, 602, 603], [604, 605, 606], [607, 608, 609]])
x1_prd = np.transpose(x1_prd)
#x2_prd = np.transpose(x2_prd)


predict_bbb=model.predict(x1_prd, batch_size=1)
print(predict_bbb)

print('-------------------')
y1_predict = model.predict(x1_test, batch_size=1)

#print(y1_predict)         #(20, 3) * 3 리스트
#print(y1_predict[0])


#y1_predict, y2_predict, y3_predict = model.predict(x1_test)

#print(y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y_predict):
    return np.sqrt(mean_squared_error(y1_test, y_predict))
RMSE1=RMSE(y1_test, y1_predict[0])
RMSE2=RMSE(y2_test, y1_predict[1])
RMSE3=RMSE(y3_test, y1_predict[2])
print('RMSE(y1_test) : ', RMSE1)
print('RMSE(y2_test) : ', RMSE2)
print('RMSE(y3_test) : ', RMSE3)
print('AVG(RMSE) : ', (RMSE1+RMSE2+RMSE3)/3)


#print('RMSE : ', RMSE(y1_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
#r2_y_predict = r2_score(y1_test, y1_predict)
#print('R2 : ', r2_y_predict)

R2_1 = r2_score(y1_test, y1_predict[0])
R2_2 = r2_score(y2_test, y1_predict[1])
R2_3 = r2_score(y3_test, y1_predict[2])
print('r2_score(y1_test) : ', R2_1)
print('r2_score(y2_test) : ', R2_2)
print('r2_score(y3_test) : ', R2_3)
print('AVG(r2_score) : ', (R2_1 + R2_2 + R2_3) / 3)



