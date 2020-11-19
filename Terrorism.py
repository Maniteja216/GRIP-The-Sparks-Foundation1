from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data=pd.read_csv('/content/drive/MyDrive/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
data.head()

data.shape

data.isnull().sum()

data.describe()

data.hist(figsize=(20,9))

corr=data.corr()
sns.heatmap(np.round(corr,2),annot=True,cmap='Greens')

pd.crosstab(data.iyear,data.region_txt).plot(kind='area',figsize=(16,8))
plt.title('Terrorist Activites by Region')
plt.ylabel('Number of Attacks')
plt.show()

plt.subplots(figsize=(16,8))
sns.countplot('iyear',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('YlOrBr',10))
plt.xticks(rotation=90)
plt.title('Number of Terrorist Activites')
plt.show()

Year=data.iyear.value_counts().to_dict()
rate=((Year[2017]-Year[1970])/Year[2017])*100
print(Year[1970],'attacks happend in 1970 &',Year[2017],'attack happend in 2017')
print('So the number of attacks form 1970 has incread by',np.round(rate,0),'%till 2017')

plt.figure(figsize=(13,6))
sns.countplot(data['attacktype1_txt'],order=data['attacktype1_txt'].value_counts().index,palette='hot')
plt.xticks(rotation=90)
plt.xlabel('Method')
plt.title('Method of attacks')
plt.show()

plt.figure(figsize=(13,6))
sns.countplot(data['targtype1_txt'],order=data['targtype1_txt'].value_counts().index,palette='magma')
plt.xticks(rotation=90)
plt.xlabel('Type')
plt.title('Type of target')
plt.show()

fig,axes=plt.subplots(figsize=(16,11),nrows=1,ncols=2)
sns.barplot(x=data['country_txt'].value_counts()[:20].values,y=data['country_txt'].value_counts()[:20].index,
            ax=axes[0],palette='magma');
axes[0].set_title('Terror Attacks per country')
sns.barplot(x=data['region_txt'].value_counts().values,y=data['region_txt'].value_counts().index,
            ax=axes[1])
axes[1].set_title('Terrorist Attacks per region')
fig.tight_layout()
plt.show()

terror=data.groupby(['country_txt'],as_index=False).count()

max_count=terror['iyear'].max()
max_id=terror['iyear'].idxmax()
max_name=terror['country_txt'][max_id]
min_count=terror['iyear'].min()
min_id=terror['iyear'].idxmin()
min_name=terror['country_txt'][min_id]

data_after=data[data['iyear']>=2001]
fig,ax=plt.subplots(figsize=(15,10),nrows=2,ncols=1)
ax[0]=pd.crosstab(data.iyear,data.region_txt).plot(ax=ax[0])
ax[0].set_title('Change in Region per Year')
ax[0].legend(loc='center left',bbox_to_anchor=(1,0.5))
ax[0].vlines(x=2001,ymin=0,ymax=7000,colors='red',linestyles='--')
pd.crosstab(data_after.iyear,data_after.region_txt).plot.bar(stacked=True,ax=ax[1])
ax[1].set_title('After Declaration of war on terror(2001-2017)')
ax[1].legend(loc='center left',bbox_to_anchor=(1,0.5))
plt.show()















