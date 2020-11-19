import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

data=pd.read_csv('/content/Iris-prediction/Iris.csv')
data.head()

X=data.drop(columns=['Id','Species'], axis=1)
y=data.Species
X.head()

encode=LabelEncoder()
y=encode.fit_transform(y)
y

wcss_scores=[]
for i in range(1,10):
  model=KMeans(n_clusters=i, random_state=1)
  model.fit(X)
  wcss_scores.append(model.inertia_)

sns.set_style('darkgrid')
plt.plot(range(1,10),wcss_scores)
plt.xlabel('K values')
plt.ylabel('wcss_scores')
plt.show()

model=KMeans(n_clusters=3,random_state=1)
y_pred=model.fit_predict(X)
X=X.values

#Visuvalizing the clusters on the last two cloumns
plt.scatter(X[y_pred== 0,0],X[y_pred==0,1],
            s=100,c='magenta',label='Iris-setosa')
plt.scatter(X[y_pred== 1,0],X[y_pred==1,1],
            s=100,c='blue',label='Iris-versicolor')
plt.scatter(X[y_pred== 2,0],X[y_pred==2,1],
            s=100,c='green',label='Iris-vergnitia')

#ploting the centroids of the clusters
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],
            s=100,c='black',label='Centroids')

plt.legend()
plt.show()
