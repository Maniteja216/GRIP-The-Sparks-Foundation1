!git clone https://github.com/Maniteja216/Iris-prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree

data=pd.read_csv('/content/Iris-prediction/Iris.csv')
data.head()

X=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
y=label.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='gini')
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
y_pred

acc_score=accuracy_score(y_test,y_pred)
acc_score

confusion=confusion_matrix(y_test,y_pred)
confusion

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 2],
                color=ListedColormap(('red', 'blue', 'green'))(i), label=j)

plt.title('Decision Tree Classification Train Data')
plt.xlabel('SepalLength')
plt.ylabel('PetalLength')
plt.legend()
plt.show()

fig=plt.figure(figsize=(25,22))
fig=tree.plot_tree(classifier,filled=True, feature_names=data.iloc[:,1:-1].columns)










