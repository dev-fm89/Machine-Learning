
# coding: utf-8

# # Stock price prediction

# In[21]:


import quandl, math
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, linear_model


df = quandl.get('NSE/TCS')
#df.head()

#print(len(df))

df['PCT_Change'] = (df['Close'] - df['Open'])/df['Close']
df['HL_PCT'] = (df['High']-df['Low'])/df['High']

df['prev_day_close'] = df['Close'].shift(1)

df = df[['Close', 'Total Trade Quantity', 'prev_day_close', 'PCT_Change', 'HL_PCT']]

forecast_col = 'Close'

forecast_out = int(math.ceil(0.01*len(df)))

#print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)


df.dropna(inplace=True)


y = np.array(df['label'])

X = np.array(df.drop('label', 1))

X = preprocessing.scale(X)


X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size=0.2)

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test);


print(accuracy)



