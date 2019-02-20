import numpy as np
import random
import matplotlib.pyplot as plt

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))


def find_centers(X, K):
    converted = X.tolist()
    oldmu = random.sample(converted, K)
    oldmu
    mu = random.sample(converted, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)


def ConvertData(ClustersDict):
    clusters = [];
    x_axis = [];
    y_axis = [];
    for key in ClustersDict.keys():
        clusters.extend([key for c in range(len(ClustersDict[key]))])
        x_axis.extend([c[0] for c in ClustersDict[key]])
        y_axis.extend(c[1] for c in ClustersDict[key])

    return (clusters, x_axis, y_axis)

def ConvertCenters(centers):
    x_axis = [c[0] for c in centers]
    y_axis = [c[1] for c in centers]

    return (x_axis, y_axis)

count = int(input("Введите количество объектов"))
classCount = int(input("Введите количество классов"))

data = init_board_gauss(count, classCount)
(centers, clusters) = find_centers(data, classCount)

cluster = clusters[0]


(c, x_axis, y_axis) = ConvertData(clusters)
(x_centers, y_centers) = ConvertCenters(centers)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x_axis, y_axis, 2, c = c)
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(x_centers, y_centers, c=[20 for x in x_centers], marker=r'$\star$')

plt.show()




