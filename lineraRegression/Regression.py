import pandas as pd 
import quandl, math,datetime
import numpy as np              
from sklearn import preprocessing,  model_selection , svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
# to save the classifier
import pickle

#style to make our graphs look decent.
style.use("ggplot")

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
#print(df['Adj. Open'].head())
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = "Adj. Close"
#replacing value of nan in dataset 
df.fillna(value = -99999 , inplace =True)
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)


df['label'] = df[forecast_col].shift(-forecast_out)

# X => feature (below preprocessing) y=> label here X=>y
# dataframe(......,1) means droping column  dataframe(......,0) means droping specific index or row
X = np.array(df.drop(['label'],1))
# proprocessing for feature, to make data in range -1 to 1 to speed up the process
X = preprocessing.scale(X)
#means it take the data from the ending bcz this is not training data in linear regression
X_lately  = X[-forecast_out:]
#Now seprate the data means remove the test data 
X =X[:-forecast_out]
#droping the one or more row in dataset if one of the column has nan
df.dropna(inplace=True)

y = np.array(df['label'])

# training and testing
#75% of your data, and use this to train the machine learning classifier.
#Then you take the remaining 25% of your data,
#and test the classifier.

# model_selection.train_test_split =>Split arrays or matrices into random train and test subsets
# here train size become rest 0.8
X_train , X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

# using algorithm
# 1 linear Regression
# we used n_jobs to define no of process at time if it is -1 then it depend on your computer speed 
#clf = LinearRegression(n_jobs=100)
#clf.fit(X_train, y_train)
"""    To save the trained classifier """
#with open("StockLinearRegression.pickle","wb") as f:
 #   pickle.dump(clf,f)
"""       To  Load classifier           """
pickle_load =open("StockLinearRegression.pickle","rb")
if(pickle_load!=None):
    clf = pickle.load(pickle_load)
"""                 ending                        """
confidance = clf.score(X_test, y_test)
#print(confidance)

# prediction predict can take one value or list of values
forecast_set= clf.predict(X_lately)
#plottig using matplotlib
# add a new column to our dataframe, the forecast column:
df['Forecast'] =np.nan
# We need to first grab the last day in the dataframe, 
#and begin assigning each new forecast to a new day.
# one thing is that here first column is date
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400# second in day
next_unix = last_unix + one_day
#add the forecast to the existing dataframe
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
# here we pass list[] to show the name of each graph and loc to represent the location
# where we want to show the graph details here 4 means bottom right
plt.legend(loc=4)
# means what is the name of value name in x axis(Date)
plt.xlabel('Date')
# means what is the name of value name in y axis(Price)
plt.ylabel('Price')
plt.show()
# 2 svm
''''clf = svm.SVR()
clf.fit(X_train, y_train)
confidance = clf.score(X_test, y_test)

print(confidance)
'''










