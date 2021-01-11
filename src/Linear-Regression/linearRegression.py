import numpy as np

class LinearRegression:
    def __init__(self, features, labels):
        self.features = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1)
        self.labels = labels
        self.weights = np.random.normal(size=2)

    def predict(self, point):
        """Returns a prediction using the given point"""
        return self.weights[0] + self.weights[1] * point

    def fit(self, epochs=30, lr=0.01):
        """Trains the model (30 epochs by default) using Gradient Descent"""
        for epoch in range(epochs):
            print('epoch', epoch+1)

            # Make the summations
            s_0, s_1 = 0, 0
            for i in range(self.features.shape[0]):
                s_0 += self.features[i] @ self.weights.T - self.labels[i]
                s_1 += (self.features[i] @ self.weights.T - self.labels[i])

            # Update the weights
            self.weights[0] -= lr * (1/self.features.shape[0]) * s_0
            self.weights[1] -= lr * (1/self.features.shape[0]) * s_1