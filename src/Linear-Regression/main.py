import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from linearRegression import LinearRegression

mpl.style.use(['ggplot'])

def generate_samples(coef, lims=(0, 50),  size=100, noise=True):
    """Generates sample data using the given coefficients and adds some noise to it (optional)"""
    feat = np.linspace(lims[0], lims[1], size).reshape((size, 1))
    labs = feat * coef[1] + coef[0]
    if noise:
        labs += np.random.normal(size=feat.shape) * 10
    return feat, labs

def show(x1, y1, x2, y2):
    """Plots the data points and the regressor"""
    plt.scatter(x1, y1, label='Data Points', color='steelblue')
    plt.plot(x2, y2, label='Regressor', color='red')
    plt.title('Linear Regression')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(fancybox=True)
    plt.savefig('regressor.png')

def main():
    # Generate the samples
    weights = np.random.uniform(low=-5, high=5, size=2)
    features, labels = generate_samples(weights)

    # Create and train the regressor
    reg = LinearRegression(features, labels)
    reg.fit()

    # Show the results
    x, y = generate_samples(reg.weights, noise=False)
    show(features, labels, x, y)


if __name__ == '__main__':
    main()