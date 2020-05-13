import numpy as np
from sklearn.linear_model import LinearRegression

weight = np.array([0.3, 0.1, 0.2, 0.2, 0.2])
# 单次计算得分

# 合作态度 |及时响应|货物损失度|及时送达|好中差评
train_data = np.array(
    [[20, 60, 60, 40, 60], [60, 80, 60, 40, 60], [80, 60, 80, 60, 40], [80, 60, 60, 40, 60], [60, 60, 60, 40, 60],
     [70, 60, 60, 40, 60], [80, 80, 80, 80, 60], [20, 60, 60, 40, 80], [80, 60, 60, 40, 60], [80, 40, 40, 20, 60],
     [60, 60, 60, 40, 60], [80, 60, 20, 40, 60], [60, 20, 60, 40, 60], [80, 60, 60, 40, 60], [80, 80, 80, 80, 80]])

train_y = [np.dot(x, weight) for x in train_data]
# train_y=[50,62,60]*5
print(train_y)
model = LinearRegression()
model.fit(train_data, train_y)

text_data = np.array([[80, 60, 60, 80, 60], [80, 60, 80, 20, 80]])
test_y = np.dot(text_data, weight)
# test_y=[75,45]

predictions = model.predict(text_data)

for i, prediction in enumerate(predictions):
    print('Predicted: {}, Target: {}'.format(prediction, test_y[i]))

print(np.average(predictions))
