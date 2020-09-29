from random import randint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from math import sqrt
from math import sqrt
style.use('dark_background')


def euclideanDistance(one, two):
    squared_distance = 0

    # Assuming correct input to the function where the lengths of two features are the same
    for i in range(len(one)):
        squared_distance += (one[i] - two[i]) ** 2

    ed = sqrt(squared_distance)

    return ed


def myKNNClassify(train, test, k=3):
	predictions = []

	# iterate through list of testing set
	for i, test_point in enumerate(test):
		testLabel = test[i][2]
		distanceList = []

		# iterate through list of training set
		for ii, train_point in enumerate(train):
			trainLabel = train[ii][2]

			# calculate distance for each point and append with label
			distanceList.append(
				[euclideanDistance([train_point[0], train_point[1]], [test_point[0], test_point[1]]), trainLabel])

		distanceList = sorted(distanceList, key=lambda x: x[0])[:k]

		ones = [item[1] for item in distanceList].count(1)
		zeros = [item[1] for item in distanceList].count(0)

		if ones > zeros:
			predictions.append(1)
		elif ones < zeros:
			predictions.append(0)
		else:
			predictions.append(randint(0,1))


	return predictions


def createData(mu, sigma, N, Data_Type, label, plot=False):

	# create N x 3 matrix for x, y and label values
	dataset = np.zeros((N,3))
	data = np.random.multivariate_normal(mu, sigma, N)

	for i, point in enumerate(data):
		dataset[i][0], dataset[i][1], dataset[i][2] = point[0], point[1], label

	if plot == True:
		plt.scatter(data[:,0], data[:,1])
		plt.title(f'{Data_Type} data')
		plt.show()

	return dataset


if __name__ == '__main__':
	mu_list = [[1, 0], [0, 1]]
	sigma_list = [np.array([[1, 0.75], [0.75, 1]]), np.array([[1, -0.5], [0.5, 1]])]
	N_train = 200
	N_test = 50
	k_list = [1,2,3,4,5,10,20]
	accuracies = []
	algoAccuracies = []

	# Creating 2 samples
	Xtrain = np.concatenate((createData(mu_list[0], sigma_list[0], N_train, 'Training', 0), createData(mu_list[1], sigma_list[1], N_train, 'Training', 1)))
	Xtest = np.concatenate((createData(mu_list[0], sigma_list[0], N_test, 'Testing', 0), createData(mu_list[1], sigma_list[1], N_test, 'Testing', 1)))

	for k in k_list:
		temp_accuracy = []

		# for loop to get avg of accuracy at k
		for i in range(5):
			predictions = myKNNClassify(Xtrain, Xtest, k)
			correct = 0

			for i, prediction in enumerate(predictions):
				if prediction == Xtest[i][2]:
					correct += 1

			temp_accuracy.append((correct/(N_test*2)) * 100)

		accuracies.append(sum(temp_accuracy) / len(temp_accuracy))

	plt.plot(k_list, accuracies)
	plt.xlabel('k')
	plt.ylabel('Accuracy %')
	plt.show()
	print(f'Average Accuracies = {sum(accuracies)/len(accuracies)}')

