import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
from sklearn import  model_selection, preprocessing 
from sklearn.linear_model import LinearRegression
style.use("ggplot")

df = pd.read_csv("breast-cancer-wisconsin-data.txt")
df.replace('?',-99999999,inplace=True)
df.drop(["Id"],1,inplace=True)
X = np.array((df.drop(["Class"],1)).astype(float).values.tolist())
y = np.array((df['Class']).astype(float).values.tolist())

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf =LinearRegression()
clf.fit(X_train,y_train)
confidance = clf.score(X_test,y_test)
prediction = clf.predict(X_test)
print(confidance)