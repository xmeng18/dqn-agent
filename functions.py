import numpy as np
import pandas as pd
import math

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

def getOtherDataVec():
	df = pd.read_csv("data/fulldata.csv")
	Vol = df.Vol.values
	Volume_ratio = df.Volume_ratio.values

	## sp500 returns
	
	return Vol, Volume_ratio

# returns the sigmoid
def sigmoid(x):
	try:
		if x < 0:
			return 1 - 1 / (1 + math.exp(x))
		return 1 / (1 + math.exp(-x))
	except OverflowError as err:
		print("Overflow err: {0} - Val of x: {1}".format(err, x))
	except ZeroDivisionError:
		print("division by zero!")
	except Exception as err:
		print("Error in sigmoid: " + err)

'''
Why does the Sigmoid function overflow?
As the training progress, the network improves its precision. As this precision approaches perfection, 
the sigmoid signal will either approach 1 from below or 0 for above. Eg: either 0.99999999999... or 0.00000000000000001...
This if condition here does not change the output of sigmoid function, it avoids the possible overflow error when computing the
sigmoid function
'''
	

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	Vol, Volume_ratio =  getOtherDataVec()
	#This part gets the data from csv file with OHLC data. If there is no n
	#periods before time t, the first row of data will be replicated to keep the
	#number of states to be n
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    #We do the similar thing the other data, which includes volume_ratio and Volumn
	otherdata = np.array([Volume_ratio,Vol])

	new_block = otherdata[:,d:t] if d >= 0 else np.concatenate((np.repeat(np.array([otherdata[:,0]]),-d-1, axis=0).T, otherdata[:,0:t + 1]), 1)
	#We do the sigmoid transformation to the difference of the data of the first part.
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.concatenate([np.array([res]),new_block])

