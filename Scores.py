import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('http://bit.ly/w-data')
data.head()

data.describe()

data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours reading')
plt.ylabel('Scores obtained')
plt.show()

X=data.iloc[:,:-1].values
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

l=regressor.coef_*X+regressor.intercept_
plt.scatter(X,y)
plt.plot(X,l)
plt.show()

print('cofficents :',regressor.coef_)
print('Interception :',regressor.intercept_)

y_pred=regressor.predict(X_test)
y_pred

df=pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
df


plt.scatter(X_test,y_test)
plt.xlabel('X_values')
plt.ylabel('y_values')
plt.title('test the actual values')
plt.show()

plt.scatter(X_test,y_pred,marker='v')
plt.xlabel('X_values')
plt.ylabel('y_values')
plt.title('test the actual values vs predicted values')
plt.show()

from sklearn import metrics
metrics.mean_absolute_error(y_test,y_pred)
metrics.mean_squared_error(y_test,y_pred)
np.sqrt(metrics.mean_squared_error(y_test,y_pred))