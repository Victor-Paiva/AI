import numpy as np

class LinearRegression:
    def __init__(self, features, labels):
        self.features = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1)
        self.labels = labels
        self.weights = np.random.normal(size=2)

    def fit(epochs=10, lr=0.1):
        pass