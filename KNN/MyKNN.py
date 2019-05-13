"""
 Euclidean Distance
(sum i=1 to n of (ai,pi)**2)**1/2
ai ,pi = two point
i is number of cordinate
"""
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style 
from collections import Counter
import warnings
import pandas as pd 
import random
style.use("fivethirtyeight")
 # It is category of data type
# dataset = {"k":[[1,2],[2,3],[3,2]], "r":[[5,6],[6,5],[7,6]]}
# new_data = [3,4]
# for i in dataset:
# 	for j in dataset[i]:
# 		plt.scatter(j[0],j[1],color=i)
# plt.scatter(new_data[0],new_data[1],color="y")
# plt.show()


#default value of k(categories of data) is 3 or 5 in scikitlearn
def k_nearest_algorithm(data, predict, k=3):
	if(len(data)>=k):
		warnings.warn('K is set to a value less than total voting groups!')	
	distance = []
	for group in data:
		for feature in data[group]:
			#norm is same as formula
			euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
			distance.append([euclidean_distance,group])
	votes = [i[1] for i in sorted(distance)[:k]]		
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result

# vote_result = k_nearest_algorithm(dataset,new_data)
# for i in dataset:
# 	for j in dataset[i]:
# 		plt.scatter(j[0],j[1],color=i)
# plt.scatter(new_data[0],new_data[1],color=vote_result)
# plt.show()

df = pd.read_csv("breast-cancer-wisconsin-data.txt")
df.replace("?",-9999999,inplace = True)
df.drop(["Id"], 1, inplace = True)
#some of data is in string form to replace them
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
	# last column is class
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	# last column is class
	test_set[i[-1]].append(i[:-1])

correct = 0
total = 0
for group in test_set:
	for data in test_set[group]:
		vote = k_nearest_algorithm(train_set,data, k=5)
		if group == vote:
			correct+=1
		total+=1
print("Accuracy:" , correct/total)

