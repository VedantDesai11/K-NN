import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('dark_background')


def myKNNClassify(train, test, k):

	for test_point in test:
		for train_point in train:
			dist = np.linalg.norm(train_point - test_point)







def createData(mu_list, sigma_list, N):
    sample = []
    l = []
    for i, mu in enumerate(mu_list):
        sample.append(np.random.multivariate_normal(mu, sigma_list[i], N))
        l.append(np.zeros(N) + i)

    X = np.concatenate((sample[0], sample[1]))
    label = np.concatenate((l[0], l[1]))

    return X, label


if __name__ == '__main__':

	mu_list = [[1, 0], [0, 1]]
	sigma_list = [np.array([[1, 0.75], [0.75, 1]]), np.array([[1, -0.5], [0.5, 1]])]
	N_train = 200
	N_test = 50


	X, label = createData(mu_list, sigma_list, N_train)

	plt.scatter(X[:,0], X[:,1])
	plt.show()

