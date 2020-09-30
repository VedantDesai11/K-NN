import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from math import sqrt

style.use("dark_background")


def createData(mu, sigma, N, Data_Type, plot=True):

	# create N x 3 matrix for x, y and label values
	dataset = np.zeros((N,3))
	data = np.random.multivariate_normal(mu, sigma, N)
	epsilon = np.random.normal(0,0.5,N)

	for i, point in enumerate(data):
		y = 2 * point[0] + point[1] + epsilon[i]
		dataset[i][0], dataset[i][1], dataset[i][2] = point[0], point[1], y

	if plot == True:
		plt.scatter(data[:,0], data[:,1])
		plt.title(f'{Data_Type} data')
		plt.show()

	return dataset


def euclideanDistance(one, two):
    squared_distance = 0

    # Assuming correct input to the function where the lengths of two features are the same
    for i in range(len(one)):
        squared_distance += (one[i] - two[i]) ** 2

    ed = sqrt(squared_distance)

    return ed


def myKnnRegress(train, test, k=5):

	predictions = []

	# iterate through list of testing set
	for i, test_point in enumerate(test):
		testLabel = test[i][2]
		distanceList = []

		# iterate through list of training set
		for ii, train_point in enumerate(train):
			trainLabel = train[ii][2]

			# calculate distance for each point and append with label
			distanceList.append([euclideanDistance([train_point[0],train_point[1]],[test_point[0],test_point[1]]), trainLabel])

		distanceList = sorted(distanceList, key=lambda x: x[0])[:k]

		Weights = [1/i[0] for i in distanceList]

		s = 0
		for i, value in enumerate(distanceList):
			s += value[1] * Weights[i]

		predictions.append(s/sum(Weights))

	return predictions



if __name__ == '__main__':

	mu = [1, 0]
	sigma = np.array([[1, 0.75], [0.75, 1]])
	N_train = 300
	N_test = 100
	k_list = [1,2,3,5,10,20,50,100]
	#k_list = [3]
	errors = []
	train = createData(mu, sigma, N_train, 'train', False)
	test = createData(mu, sigma, N_test, 'test', False)

	for k in k_list:
		temp_error = []
		for i in range(5):
			predictions = myKnnRegress(train, test, k)

			error = 0

			for i, prediction in enumerate(predictions):
				error += sqrt((prediction - test[i][2]) ** 2)

			temp_error.append(error / N_test)

		errors.append(sum(temp_error) / len(temp_error))

	print(errors)
	plt.plot(k_list, errors)
	plt.show()

