import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./data/1.salary.csv',)



array = data.values
array.shape
X = array[:, 0] #독립변수
Y = array[:,1]  #종립변수

#근속연수 * 연봉
XR = X.reshape(-1, 1)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(XR, Y, test_size=0.2)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)
model.coef_
model.intercept_

y_pred = model.predict(X_test)
error = mean_absolute_error(y_pred, y_test)
print(error)



# 실제 값 산점도
plt.scatter(range(len(X_test)), y_test, color='blue', label='Actual Values')
# 예측 값 선 그래프
plt.plot(range(len(X_test)), y_pred, color='red', linewidth=2, label='Predicted Values')

plt.xlabel('Experience Years')
plt.ylabel('Salary')
plt.title('Actual vs Predicted Salary')
plt.legend()
plt.show()