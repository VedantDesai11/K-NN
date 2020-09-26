from random import randint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from math import sqrt
style.use('dark_background')


def myKNNClassify(train, test, k=3):

	predictions = []

	# iterate through list of testing set
	for i, test_point in enumerate(test[0]):
		testLabel = test[1][i]
		distanceList = []
		votes = []


		# iterate through list of training set
		for ii, train_point in enumerate(train[0]):

			trainLabel = train[1][ii]

			# calculate distance for each point and append with label
			distanceList.append([np.linalg.norm(train_point - test_point), trainLabel])

		distanceList = sorted(distanceList, key = lambda x: x[0])

		ones = 0
		zeros = 0
		for i in range(k):
			if distanceList[i][1] == 1:
				ones+=1
			else:
				zeros+=1

		if ones > zeros:
			predictions.append(1)
		if zeros > ones:
			predictions.append(0)
		else:
			predictions.append(randint(0,1))


	return predictions


def createData(mu_list, sigma_list, N, Data_Type):
	sample = []
	l = []
	for i, mu in enumerate(mu_list):
		data, label = np.random.multivariate_normal(mu, sigma_list[i], N), np.zeros(N, dtype='int') + i
		plt.scatter(data[:,0], data[:,1])

		sample.append(data)
		l.append(label)

	plt.title(f'{Data_Type} Data')
	plt.show()
	X = np.concatenate((sample[0], sample[1]))
	label = np.concatenate((l[0], l[1]))

	return X, label


if __name__ == '__main__':
	mu_list = [[1, 0], [0, 1]]
	sigma_list = [np.array([[1, 0.75], [0.75, 1]]), np.array([[1, -0.5], [0.5, 1]])]
	N_train = 200
	N_test = 50
	k_list = [1,2,3,4,5,10,20]
	accuracies = []
	algoAccuracies = []

	Xtrain = createData(mu_list, sigma_list, N_train, 'Training')
	Xtest = createData(mu_list, sigma_list, N_test, 'Testing')

	for k in k_list:

		predictions = myKNNClassify(Xtrain, Xtest, k)
		correct = 0
		algoCorrect = 0

		for i, label in enumerate(Xtest[1]):
			if label == predictions[i]:
				correct += 1

		accuracy = correct/(N_test*2) * 100
		accuracies.append(accuracy)


	plt.plot(k_list, accuracies)
	plt.xlabel('k')
	plt.ylabel('Accuracy %')
	plt.text(13, max(accuracies), f'Average Accuracies = {sum(accuracies)/len(accuracies):.2f}')
	plt.show()
	#print(f'Average Accuracies = {sum(accuracies)/len(accuracies)}')

