from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('/content/drive/My Drive/matches.csv')
data.head()

data.shape

data.isnull().sum()

#Drop missing values
data=data.drop(['umpire3'],axis=1)

corr=data.corr()
sns.heatmap(corr,annot=True,cmap='Greens')

corr=data.corr()
sns.heatmap(corr,annot=True,cmap='Greens')

data.describe()

data['player_of_match'].value_counts()

#top 10 players
data['player_of_match'].value_counts()[0:10]

data['player_of_match'].value_counts().keys()

plt.figure(figsize=(12,7))
plt.bar(data['player_of_match'].value_counts()[0:10].keys(),data['player_of_match'].value_counts()[0:10],color='r')
plt.show()

data['result'].value_counts()

data['toss_winner'].value_counts()

data['win_by_runs'].value_counts()

batting_first=data[data['win_by_runs']!=0]
batting_first.head()

plt.figure(figsize=(12,10))
plt.hist(batting_first['win_by_runs'])
plt.title('distribution of runs')
plt.xlabel('runs')
plt.show()

batting_first['winner'].value_counts()

batting_first['winner'].value_counts().keys()

batting_first['winner'].value_counts()[0:3].keys()

plt.figure(figsize=(12,10))
plt.bar(batting_first['winner'].value_counts()[0:3].keys(),batting_first['winner'].value_counts()[0:3],color=['red','green'])
plt.show()

plt.figure(figsize=(12,10))
plt.pie(batting_first['winner'].value_counts(),labels=batting_first['winner'].value_counts().keys(),autopct='%1.1f%%')
plt.show()

#win by wickets
data['win_by_wickets'].value_counts()

batting_secound=data[data['win_by_wickets']!=0]
batting_secound.head()

plt.figure(figsize=(12,10))
plt.hist(batting_secound['win_by_wickets'],bins=20)
plt.show()

batting_secound['winner'].value_counts()

#winner in secound batting
plt.figure(figsize=(12,10))
plt.bar(batting_secound['winner'].value_counts()[0:3].keys(),batting_secound['winner'].value_counts()[0:3],color=['red','green'])
plt.show()

plt.figure(figsize=(12,10))
plt.pie(batting_secound['winner'].value_counts(),labels=batting_secound['winner'].value_counts().keys(),autopct='%1.1f%%')
plt.show()

data['season'].value_counts()

data['city'].value_counts()

np.sum(data['toss_winner'] == data['winner'])

data1=pd.read_csv('/content/drive/My Drive/deliveries.csv')
data1.head()

data1.shape

data.isnull().sum()

data1.describe()

corr=data1.corr()
sns.heatmap(corr,annot=True,cmap='Greens')

data1['match_id'].unique()

match=data1[data1['match_id']==1]
match.head()

kkr=match[match['inning']==1]
kkr['batsman_runs'].value_counts()

kkr['dismissal_kind'].value_counts()

rcb=match[match['inning']==2]
rcb['batsman_runs'].value_counts()

rcb['dismissal_kind'].value_counts()

over=match[match['match_id']==1]['over']
run=match[(match['match_id']==1) & (match['over']==1)]['total_runs'].sum()
run

(match['over'].value_counts()/6).plot(kind='bar')




















