import numpy as np
from linearRegression import LinearRegression

def generate_samples(coef, lims=(0, 50),  size=100):
    feat = np.linspace(lims[0], lims[1], size).reshape((size, 1))
    labs = feat * coef[0] + coef[1] + np.random.normal(size=feat.shape) / 10
    return feat, labs

def main():
    # Generate the samples
    weights = np.random.uniform(low=-2, high=2, size=2)
    features, labels = generate_samples(weights)

    reg = LinearRegression(features, labels)

if __name__ == '__main__':
    main()