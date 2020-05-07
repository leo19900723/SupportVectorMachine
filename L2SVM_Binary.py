import cvxpy
import random
import numpy
import matplotlib.pyplot as plt


def supportVectorMachine(x_dataPoints, y_labels):
    w_weights = cvxpy.Variable((x_dataPoints.shape[1], 1))
    b_baises = cvxpy.Variable()
    lambda_regularizationController = 1

    print(x_dataPoints.shape[0])

    primalCost = 0.5 * (cvxpy.norm(w_weights) ** 2)
    hingeLoss = cvxpy.sum(cvxpy.pos(1 - cvxpy.multiply(y_labels, x_dataPoints @ w_weights + b_baises)))

    objectiveFunction = cvxpy.Minimize(primalCost + lambda_regularizationController * hingeLoss)

    problem = cvxpy.Problem(objectiveFunction)
    problem.solve()
    print("status:", problem.status)
    print("optimal value", problem.value)
    print("optimal var w = {}, b = {}".format(w_weights.value, b_baises.value))
    return w_weights, b_baises


def _unitTest():
    # generate toy training data
    N1 = 200  # number of positive instances
    N2 = 100  # number of negative instances
    D = 2  # feature dimension
    eps = 1e-8  # select support vectors
    random.seed(1)  # For reproducibility
    r1 = numpy.sqrt(1.5 * numpy.random.rand(N1, 1))  # Radius
    t1 = 2 * numpy.pi * numpy.random.rand(N1, 1)  # Angle
    data1 = numpy.concatenate((r1 * numpy.cos(t1), r1 * numpy.sin(t1)), axis=1)  # Points
    r2 = numpy.sqrt(3 * numpy.random.rand(N2, 1))  # Radius
    t2 = 2 * numpy.pi * numpy.random.rand(N2, 1)  # Angle
    data2 = numpy.concatenate((2.5 + r2 * numpy.cos(t2), 1.5 + r2 * numpy.sin(t2)), axis=1)  # points
    # generate toy testing data
    Nt1 = 50  # number of positive instances
    Nt2 = 25  # number of negative instances
    D = 2  # feature dimension
    random.seed(1)  # For reproducibility
    r1 = numpy.sqrt(3.4 * numpy.random.rand(Nt1, 1))  # Radius
    t1 = 2 * numpy.pi * numpy.random.rand(Nt1, 1)  # Angle
    testdata1 = numpy.concatenate((r1 * numpy.cos(t1), r1 * numpy.sin(t1)), axis=1)  # Points
    r2 = numpy.sqrt(2.4 * numpy.random.rand(Nt2, 1))  # Radius
    t2 = 2 * numpy.pi * numpy.random.rand(Nt2, 1)  # Angle
    testdata2 = numpy.concatenate((3 + r2 * numpy.cos(t2), r2 * numpy.sin(t2)), axis=1)  # points
    ## training linear SVM based on CVX optimizer
    X = numpy.concatenate((data1, data2), axis=0)
    y = numpy.concatenate((numpy.ones((N1, 1)), - numpy.ones((N2, 1))), axis=0)

    w, b = supportVectorMachine(X, y)

    # visualize decision boundary for training data
    d = 0.02
    x1 = numpy.arange(numpy.min(X[:, 0]), numpy.max(X[:, 0]), d)
    x2 = numpy.arange(numpy.min(X[:, 1]), numpy.max(X[:, 1]), d)
    x1Grid, x2Grid = numpy.meshgrid(x1, x2)
    xGrid = numpy.stack((x1Grid.flatten('F'), x2Grid.flatten('F')), axis=1)
    scores1 = xGrid.dot(w.value) + b.value
    scores2 = -xGrid.dot(w.value) - b.value
    plt.figure(0)
    sup = y * (X.dot(w.value) + b.value) - 1
    sup_v1 = ((-eps < sup) & (sup < eps)).flatten()
    h3 = plt.scatter(X[sup_v1, 0], X[sup_v1, 1], s=21, marker='o', c='k')
    h1 = plt.scatter(data1[:, 0], data1[:, 1], s=15, marker='.', c='r')
    h2 = plt.scatter(data2[:, 0], data2[:, 1], s=15, marker='.', c='b')
    plt.contour(x1Grid, x2Grid, numpy.reshape(scores1, x1Grid.shape, order='F'), levels=0, colors='k')
    plt.axis('equal')
    plt.title('Decision boundary and support vectors for training data')
    plt.legend((h1, h2, h3), ('+1', '-1', 'support vecs'))

    # visualize decision boundary for test data
    Xt = numpy.concatenate((testdata1, testdata2), axis=0)
    yt = numpy.concatenate((numpy.ones((Nt1, 1)), - numpy.ones((Nt2, 1))), axis=0)
    xt1 = numpy.arange(numpy.min(Xt[:, 0]), numpy.max(Xt[:, 0]), d)
    xt2 = numpy.arange(numpy.min(Xt[:, 1]), numpy.max(Xt[:, 1]), d)
    xt1Grid, xt2Grid = numpy.meshgrid(xt1, xt2)
    xtGrid = numpy.stack((xt1Grid.flatten('F'), xt2Grid.flatten('F')), axis=1)
    test_scores1 = xtGrid.dot(w.value) + b.value
    test_scores2 = -xtGrid.dot(w.value) - b.value
    plt.figure(1)
    ht1 = plt.scatter(testdata1[:, 0], testdata1[:, 1], s=15, marker='.', c='r')
    ht2 = plt.scatter(testdata2[:, 0], testdata2[:, 1], s=15, marker='.', c='b')
    plt.contour(xt1Grid, xt2Grid, numpy.reshape(test_scores1, xt1Grid.shape, order='F'), levels=0, colors='k')
    plt.axis('equal')
    plt.title('Decision boundary and support vectors for test data')
    plt.legend((ht1, ht2), ('+1', '-1'))
    plt.show()


if __name__ == '__main__':
    _unitTest()
