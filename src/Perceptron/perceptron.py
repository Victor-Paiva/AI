import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use(['ggplot'])

class Perceptron:
    def __init__(self, features, labels, lr=0.1, img=False):
        self.features = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1)
        self.labels = labels
        self.lr = lr
        self.img = img
        self.weights = np.random.normal(size=self.features.shape[-1])

    def show(self, margin=1.5, points=400):
        """
        Plots a meshgrid showing the decision boundary
        """
        x_min, x_max = np.min(self.features[:, 1]) - margin, np.max(self.features[:, 1]) + margin
        y_min, y_max = np.min(self.features[:, 2]) - margin, np.max(self.features[:, 2]) + margin

        x = np.linspace(x_min, x_max, points)
        y = np.linspace(y_min, y_max, points)

        xx, yy = np.meshgrid(x, y)
        xxr, yyr = xx.ravel().reshape((points**2, 1)), yy.ravel().reshape((points**2, 1))
        bias = np.ones(xxr.shape)
        feat = np.concatenate((bias, xxr, yyr), axis=1)

        Z = []
        for i in range(feat.shape[0]):
            Z.append(self.predict(feat[i]))

        Z = np.array(Z)
        Z = Z.reshape(xx.shape)

        plt.pcolormesh(xx, yy, Z, cmap='Pastel1', shading='auto')

        pos = self.labels == 1
        neg = self.labels == -1

        plt.scatter(self.features[pos, 1], self.features[pos, 2], color='purple')
        plt.scatter(self.features[neg, 1], self.features[neg, 2], color='green')
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        plt.savefig('meshgrid.png')

    def activation(self, value):
        """
        The activation function
        """
        return 1 if value >= 0 else -1

    def predict(self, features):
        """
        Make a prediction given a set of points
        """
        return self.activation(features @ self.weights.T)

    def accuracy(self):
        """
        Returns the accuracy of the perceptron
        """
        counter = 0
        for i in range(self.features.shape[0]):
            if self.labels[i] * self.predict(self.features[i]) > 0:
                counter += 1

        return counter / self.features.shape[0]

    def fit(self, epochs):
        """
        Trains the perceptron and, optionally, generates a file where one can visualize the perceptron's performance
        """
        for epoch in range(epochs):
            print('epoch:', epoch+1)
            
            for i in range(self.features.shape[0]):
                if self.labels[i] * self.predict(self.features[i]) < 0:
                    for j in range(self.weights.shape[0]):
                        self.weights[j] += self.labels[i] * self.features[i, j] * self.lr

        if self.img:
            self.show()

        return self.accuracy()