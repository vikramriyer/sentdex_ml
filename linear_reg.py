import pandas as pd
import quandl
import math

#lec 1

df = quandl.get('WIKI/GOOGL')

#print df.head()

df = df[['Adj. Open','Adj. High','Adj. Low', 'Adj. Close', 'Adj. Volume',]]

#print df.head()

#lec 2

df['High_Low_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'High_Low_PCT', 'PCT_change', 'Adj. Volume']]

#print df.head()

#lec 3
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#every row is a days data or a candle, hence we predict 0.1*len means 10% of data
# for ex: if we have 100 days data, we predict 10 more days of data
forecast_out  = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

#print df.head()

#lec 4
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

# scaling new values along with the old values helps in training and testing, adds up processing time
X = preprocessing.scale(X)

#X = X[:-forecast_out+1]

y = np.array(df['label'])

print len(X), len(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=10)
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

print forecast_out

#lec 5