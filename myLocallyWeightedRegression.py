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

		s = 0
		for value in distanceList:
			s += value[1]

		predictions.append(s//k)

	return predictions



if __name__ == '__main__':

	mu = [1, 0]
	sigma = np.array([[1, 0.75], [0.75, 1]])
	N_train = 300
	N_test = 100
	k_list = [1,2,3,5,10,20,50,100]
	#k_list = [1,2,3]
	accuracies = []


	train = createData(mu, sigma, N_train, 'train', False)
	test = createData(mu, sigma, N_test, 'test', False)

	for k in k_list:
		temp_accuracy = []
		for i in range(5):
			predictions = myKnnRegress(train, test, k)
			correct = 0

			for i, prediction in enumerate(predictions):
				if prediction == int(test[i][2]):
					correct += 1

			temp_accuracy.append(correct/N_test * 100)

		accuracies.append(f'Accuracy (k = {k}): {sum(temp_accuracy)/len(temp_accuracy):.2f}%')

	print(accuracies)

