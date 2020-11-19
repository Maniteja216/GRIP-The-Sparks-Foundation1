!git clone https://github.com/Maniteja216/Data-Analysis-Retail

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('/content/Data-Analysis-Retail/SampleSuperstore.csv')
data.head()

data.isnull().sum()

data.describe()

data.duplicated().sum()

data.nunique()

data['Postal Code']=data['Postal Code'].astype('object')

data.drop_duplicates(subset=None, keep='first',inplace=True)

corr=data.corr()
sns.heatmap(corr,annot=True,cmap='Greens')

data=data.drop(['Postal Code'], axis=1)

sns.pairplot(data,hue='Ship Mode')

data['Ship Mode'].value_counts()

sns.pairplot(data,hue='Segment')

sns.countplot(x='Segment',data=data, palette='rainbow')

data['Category'].value_counts()

sns.countplot(x='Category',data=data, palette='tab10')

sns.pairplot(data,hue='Category')

data['Sub-Category'].value_counts()

plt.figure(figsize=(12,10))
data['Sub-Category'].value_counts().plot.pie(autopct='dark')
plt.show()

data['State'].value_counts()

plt.figure(figsize=(12,10))
sns.countplot(x='State',data=data, palette='rocket_r',order=data['State'].value_counts().index)
plt.xticks(rotation=90)
plt.show()

data.hist(figsize=(12,10),bins=50)
plt.show()

plt.figure(figsize=(12,10))
data['Region'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()

plt.subplots(figsize=(16,8))
plt.scatter(data['Sales'],data['Profit'])
plt.xlabel('sales')
plt.ylabel('profit')
plt.show()


sns.lineplot(x='Discount',y='Profit',label='profit',data=data)
plt.show()

sns.lineplot(x='Quantity',y='Profit',label='profit',data=data)
plt.show()

data.groupby('Segment')[['Profit','Sales']].sum().plot.bar(color=['red','blue'],figsize=(8,5))
plt.ylabel('profit/loss')
plt.show()

plt.figure(figsize=(12,10))
plt.title('Segment wise sales in each region')
sns.barplot(x='Region',y='Sales',data=data,hue='Segment',order=data['Region'].value_counts().index,palette='rocket')
plt.xlabel('Region',fontsize=15)
plt.show()

data.groupby('Region')[['Profit','Sales']].sum().plot.bar(color=['red','blue'],figsize=(8,5))
plt.ylabel('profit/loss and sales')
plt.show()

ps=data.groupby('State')[['Profit','Sales']].sum().sort_values(by='Sales',ascending=False)
ps[:].plot.bar(color=['blue','green'],figsize=(12,8))
plt.title('State wide sale in each region')
plt.xlabel('state')
plt.ylabel('profit/loss and sales')
plt.show()

top_states=data['State'].value_counts().nlargest(10)
top_states

data.groupby('Category')[['Profit','Sales']].sum().plot.bar(color=['red','blue'],figsize=(8,5))
plt.ylabel('profit/loss and sales')
plt.show()

ps=data.groupby('Sub-Category')[['Profit','Sales']].sum().sort_values(by='Sales',ascending=False)
ps[:].plot.bar(color=['blue','green'],figsize=(12,8))
plt.title('State wide sale in each region')
plt.xlabel('subcategory')
plt.ylabel('profit/loss and sales')
plt.show()





