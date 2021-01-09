import numpy as np
from perceptron import Perceptron

# Generate a set of points
features = np.random.uniform(low=0.0, high=20.0, size=(200, 2))
weights = np.array([2, -0.4])
labels = np.ones(200)
for i in range(200):
    if features[i] @ weights.T > features[i, 1]:
        labels[i] = -1

perc = Perceptron(features, labels, img=True)
acc = perc.fit(10)
print('accuracy:', acc)