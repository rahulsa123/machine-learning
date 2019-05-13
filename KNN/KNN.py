import numpy as np
from sklearn import preprocessing, model_selection , neighbors
import pandas as pd
import csv

df = pd.read_csv("breast-cancer-wisconsin-data.txt", delimiter =",")
df.replace('?',-99999, inplace=True)
df.drop(["Id"], 1 , inplace =True)
print(type(df))
X = np.array(df.drop(["Class"],1))
y = np.array(df["Class"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
example_measures = np.array([4,2,1,1,1,2,3,2,1]).reshape(1,-1)
prediction = clf.predict(example_measures)
print(prediction)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
