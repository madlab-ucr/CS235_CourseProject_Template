import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import numpy as np

def plot_decision_boundary(X, y, model):
    cMap = ListedColormap(["#6b76e8", "#c775d1"])
    cMapa = ListedColormap(["#c775d1", "#6b76e8"])

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
    Z = Z.reshape(xx.shape)

    plt.figure(1, figsize=(8, 6), frameon=True)
    plt.axis('off')
    plt.pcolormesh(xx, yy, Z, cmap=cMap)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker = "o", edgecolors='k', cmap=cMapa)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig(os.getcwd()+"/results/decision_boundary_plot.png")