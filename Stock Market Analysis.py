import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import scipy.stats as stats
from statsmodels.tsa.vector_ar.var_model import VAR

url='https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/DPQMQH/P2Z4PM'
data=pd.read_csv(url)
data.head()

data.isnull().sum()

data.describe()

data['published_date'] = np.array([str(str(str(x)[:4]) + '/' + str(str(x)[4:6]) + '/' + str(str(x)[6:])) for x in data['publish_date']])

data.head()

data = data.drop('publish_date', axis=1)
data.head()

data['published_date'] = pd.to_datetime(data['published_date'])
data.info()

data = data[['published_date', 'headline_text']]

data.columns = ['published_date', 'headline']
data.head()

dict_news = {}

temp = data.loc[0, 'published_date']
temp2 = str(data.loc[0, 'headline'])
for x in range(1, len(data)):
    if data.loc[x, 'published_date']==temp:
        temp2 += '. ' + str(data.loc[x, 'headline'])
    else:
        dict_news[data.loc[x-1, 'published_date']] = temp2
        temp2 = ""
        temp = data.loc[x, 'published_date']


len(dict_news)

indexes = np.arange(0, len(dict_news))

df_news = pd.DataFrame(indexes)

df_news.head()

df_news['Published_Date'] = dict_news.keys()
df_news.head()

l = []
for i in dict_news.keys():
    l.append(dict_news[i])

l[0]

df_news['Headline'] = np.array(l)
df_news.head()

df_news = df_news.drop(0, axis=1)


polarity = []
subjectivity = []
tuples = []
for i in df_news['Headline'].values:
    my_valence = TextBlob(i)
    tuples.append(my_valence.sentiment)

for i in tuples:
    polarity.append(i[0])
    subjectivity.append(i[1])

df_news['Polarity'] = np.array(polarity)
df_news['Subjectivity'] = np.array(subjectivity)
df_news.head()

temp = ['Positive', 'Negative', 'Neutral']
temp1 = ['Factual', 'Public']
polarity = []
subjectivity = []
for i in range(len(df_news)):
    pol = df_news.iloc[i]['Polarity']
    sub = df_news.iloc[i]['Subjectivity']
    if pol >= 0:
        if pol >= 0.2:
            polarity.append(temp[0])
        else:
            polarity.append(temp[2])
    else:
        if pol <= -0.2:
            polarity.append(temp[1])
        else:
            polarity.append(temp[2])

    if sub >= 0.4:
        subjectivity.append(temp1[1])
    else:
        subjectivity.append(temp1[0])

df_news['Sentiment'] = polarity
df_news['Opinion'] = subjectivity
df_news.head()

plt.figure(figsize=(6,4))
df_news['Subjectivity'].hist()
plt.show()

plt.figure(figsize=(6,4))
df_news['Polarity'].hist()
plt.show()

sns.countplot(df_news['Opinion'])

!git clone https://github.com/Maniteja216/Global-Terrorism-Data-analysis

stocks=pd.read_csv('/content/Global-Terrorism-Data-analysis/ES=F.csv')
stocks.head()

len(stocks)

stocks.info()

stocks.describe()

stocks['HL_pct']=((stocks['High']-stocks['Low'])/stocks['Low'])*100
stocks['PCT_change']=((stocks['Close']-stocks['Open'])/stocks['Open'])*100

stocks = stocks.drop(['Open','High','Close','Low'], axis=1)
stocks.head()

stocks['Date'] = pd.to_datetime(stocks['Date'])
stocks.head()

df_news.columns = ['Date', 'Headline', 'Polarity', 'Subjectivity', 'Sentiment', 'Opinion']

merge_data = df_news.merge(stocks, how='inner', on='Date', left_index = True)
merge_data.head()

merge_data = merge_data.reset_index()
merge_data.head()

merge_data.info()

merge_data = merge_data.drop(0)

merge_data = merge_data[merge_data['Adj Close'].notna()]
merge_data.info()

merge_data = merge_data.reset_index()
merge_data.head()

merge_data = merge_data.drop(['level_0', 'index'], axis=1)
merge_data.head()

df_needed = merge_data[['Date', 'Polarity', 'Subjectivity','Adj Close','HL_pct','PCT_change']]
df_needed.head()

df_needed = df_needed.set_index('Date')
df_needed.head()

scaler = MinMaxScaler()
new = pd.DataFrame(scaler.fit_transform(df_needed))
new.columns = df_needed.columns
new.index=df_needed.index
new.head()

trainSet = new[: int(0.7*(len(new)))]
testSet = new[int(0.7*(len(new))):]

model = VAR(endog = trainSet)
fit = model.fit()

pred = fit.forecast(fit.y, steps=len(testSet))
predicted = pd.DataFrame(pred, columns=new.columns)

predicted.head()

testSet.head()

np.sqrt(mean_squared_error(predicted['Adj Close'], testSet['Adj Close']))

sns.jointplot(testSet['Adj Close'], predicted['Adj Close']).annotate(stats.pearsonr)
plt.show()














