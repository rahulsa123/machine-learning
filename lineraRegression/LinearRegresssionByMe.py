'''  
Now to fit a line between many points we used line eq.
y = mx + c here m = slope of line and c = y-intercept

m = ( mean(x).mean(y) -mean(x.y) ) / ( mean(x)**2 - mean(x**2))

c= mean(y)-m.mean(x)
'''
import matplotlib.pyplot as plt 
import numpy as np 
from statistics import mean 
from matplotlib import style
from sklearn.linear_model import LinearRegression
#for test data
import random

style.use('fivethirtyeight')

def create_dataset(hm, variance, step=2, correlation = False):
	val = 1
	y =[]
	for i in range(hm):
		ref_y = val + random.randrange(-variance,variance)
		y.append(ref_y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	x = [i for i in range(len(y))]
	return np.array(x,dtype=np.float64), np.array(y,dtype=np.float64) 



#x = np.array([1,2,3,4,5], dtype=np.float64)
#y = np.array([5,4,6,5,6], dtype=np.float64)


def best_fit_slope_and_intercept(x,y):
	m = (((mean(x)*mean(y)) - mean(x*y)) / ((mean(x)*mean(x)) - mean(x*x)))
	b = mean(y) - m*mean(x)
	return m,b 

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


x, y = create_dataset(40, 40, 2, correlation = 'neg')

m, b = best_fit_slope_and_intercept(x,y)


#predict value
predict_x = 6
predict_y = m*predict_x + b
regression_line = [m*X + b for X in x]

r_squared = coefficient_of_determination(y,regression_line)
print(r_squared)

plt.scatter(predict_x, predict_y , label="predict", color='b')

plt.scatter(x,y,color='g',label="points")

plt.plot(x,regression_line,color='r',label="line")
plt.xlabel("X values")
plt.ylabel("y values")
plt.legend(loc=4)


plt.show()