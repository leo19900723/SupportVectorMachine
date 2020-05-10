import math
import cvxpy
import random
import numpy
import pandas
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from math import radians, cos, sin, asin, sqrt


class L2SVM(object):

    def __init__(self, x_trainDataPoints, y_trainLabels, K_allClassSet, lambda_regularizationController):
        self._x_trainDataPoints = x_trainDataPoints
        self._y_trainLabels = y_trainLabels
        self._k_allClassSet = K_allClassSet
        self._lambda_regularizationController = lambda_regularizationController

        self._w_weights = cvxpy.Variable((self._x_trainDataPoints.shape[1], len(self._k_allClassSet)))
        self._b_biases = cvxpy.Variable(len(self._k_allClassSet))
        self._b_biases.value = numpy.zeros(len(self._k_allClassSet))

    def train(self):
        print("Training...")
        delta = lambda a, b: int(bool(a - b)) * 1

        primalCost = 0.5 * cvxpy.sum_squares(self._w_weights)
        hingeLoss_multiClass_withoutPsi = cvxpy.sum([cvxpy.pos(self._x_trainDataPoints[i] @ self._w_weights[:, j] - self._x_trainDataPoints[i] @ self._w_weights[:, self._y_trainLabels[i]] + delta(j, self._y_trainLabels[i])) for j in self._k_allClassSet for i in range(self._x_trainDataPoints.shape[0])]) / len(self._k_allClassSet)

        trainingObjectiveFunction = cvxpy.Minimize(primalCost + self._lambda_regularizationController * hingeLoss_multiClass_withoutPsi)

        trainingProblem = cvxpy.Problem(trainingObjectiveFunction)
        trainingProblem.solve(solver="SCS")

        print("status:", trainingProblem.status)
        print("optimal value", trainingProblem.value)
        print("optimal var w =", self._w_weights.value, ", b =", self._b_biases.value)

        return self._w_weights.value, self._b_biases.value

    def predict(self, xt_testDataPoints, yt_testLabels=None):
        print("Predicting...")
        ycap_predictedLabels = numpy.empty(xt_testDataPoints.shape[0], dtype=int)
        w = self._w_weights.value
        b = self._b_biases.value

        for i in range(xt_testDataPoints.shape[0]):
            score = -math.inf
            for j in list(self._k_allClassSet)[::-1]:
                newScore = xt_testDataPoints[i] @ w[:, j] + b[j]

                if newScore > score:
                    ycap_predictedLabels[i] = j
                    score = newScore

        if yt_testLabels is not None:
            # print("y=\n", yt_testLabels)
            # print("ycap=\n", ycap_predictedLabels)
            print("RMSE Loss: ", sqrt(sklearn.metrics.mean_squared_error(yt_testLabels, ycap_predictedLabels)))
            print("Diff Ratio: ", numpy.sum(yt_testLabels != ycap_predictedLabels) / xt_testDataPoints.shape[0])
        return ycap_predictedLabels


    @staticmethod
    def printProgressBar(iteration, total, delimiter=None, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"):
        if iteration == total or delimiter is None or iteration % (delimiter * (total / 100)) == 0:
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + "-" * (length - filledLength)
            print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
            # Print New Line on Complete
            if iteration == total:
                print()


def main():
    print("Loading Files...")
    trainFile = pandas.read_csv("InputFiles/train.csv")
    testFile = pandas.read_csv("InputFiles/test.csv")

    offset = 0
    x_trainDataPoints, y_trainLabels = trainFile.loc[:, ["x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6"]].to_numpy() + offset, trainFile["Digit"].to_numpy()
    xt_testDataPoints = testFile.loc[:, ["x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6"]].to_numpy() + offset

    trainPortion = int(x_trainDataPoints.shape[0]*1)
    valPortion = int(x_trainDataPoints.shape[0]*0)

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x_trainDataPoints[:trainPortion], y_trainLabels[:trainPortion])
    ycap_lib = clf.predict(x_trainDataPoints[valPortion:])
    ytcap_lib = clf.predict(xt_testDataPoints)
    print("RMSE Loss: ", sqrt(sklearn.metrics.mean_squared_error(y_trainLabels[valPortion:], ycap_lib)))
    print("Diff Ratio: ", numpy.sum(y_trainLabels[valPortion:] != ycap_lib) / x_trainDataPoints.shape[0])

    model = L2SVM(x_trainDataPoints[:trainPortion], y_trainLabels[:trainPortion], {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, lambda_regularizationController=0.05)
    w = model.train()
    ycap = model.predict(x_trainDataPoints[valPortion:], y_trainLabels[valPortion:])
    ytcap = model.predict(xt_testDataPoints)
    print(ytcap)

    return


def _unitTest():
    # generate toy training data
    N1 = 100  # number of C1 instances
    N2 = 100  # number of C2 instances
    N3 = 100  # number of C3 instances
    N4 = 100  # number of C4 instances
    random.seed(1)  # For reproducibility

    r1 = numpy.sqrt(1.5 * numpy.random.rand(N1, 1))  # Radius
    t1 = 2 * numpy.pi * numpy.random.rand(N1, 1)  # Angle
    data1 = numpy.concatenate((2 + r1 * numpy.cos(t1), 4 + r1 * numpy.sin(t1)), axis=1)  # Points

    r2 = numpy.sqrt(3 * numpy.random.rand(N2, 1))  # Radius
    t2 = 2 * numpy.pi * numpy.random.rand(N2, 1)  # Angle
    data2 = numpy.concatenate((2.5 + r2 * numpy.cos(t2), 1.5 + r2 * numpy.sin(t2)), axis=1)  # points

    r3 = numpy.sqrt(1 * numpy.random.rand(N3, 1))  # Radius
    t3 = -2 * numpy.pi * numpy.random.rand(N3, 1)  # Angle
    data3 = numpy.concatenate((4 + r3 * numpy.cos(t3), -1 + r3 * numpy.sin(t3)), axis=1)  # points

    r4 = numpy.sqrt(4 * numpy.random.rand(N4, 1))  # Radius
    t4 = -5 * numpy.pi * numpy.random.rand(N4, 1)  # Angle
    data4 = numpy.concatenate((1 + r4 * numpy.cos(t4), 8 + r4 * numpy.sin(t4)), axis=1)  # points

    # generate toy testing data
    Nt1 = 50  # number of C1 instances
    Nt2 = 50  # number of C2 instances
    Nt3 = 50  # number of C3 instances
    Nt4 = 50  # number of C4 instances

    r1 = numpy.sqrt(3.4 * numpy.random.rand(Nt1, 1))  # Radius
    t1 = 2 * numpy.pi * numpy.random.rand(Nt1, 1)  # Angle
    testdata1 = numpy.concatenate((3 + r1 * numpy.cos(t1), 3.5 + r1 * numpy.sin(t1)), axis=1)  # Points

    r2 = numpy.sqrt(2.4 * numpy.random.rand(Nt2, 1))  # Radius
    t2 = 2 * numpy.pi * numpy.random.rand(Nt2, 1)  # Angle
    testdata2 = numpy.concatenate((3 + r2 * numpy.cos(t2), 0.8 + r2 * numpy.sin(t2)), axis=1)  # points

    r3 = numpy.sqrt(1.3 * numpy.random.rand(Nt3, 1))  # Radius
    t3 = -2 * numpy.pi * numpy.random.rand(Nt3, 1)  # Angle
    testdata3 = numpy.concatenate((3.5 + r3 * numpy.cos(t3), -1.2 + r3 * numpy.sin(t3)), axis=1)  # points

    r3 = numpy.sqrt(3.5 * numpy.random.rand(Nt4, 1))  # Radius
    t3 = -6.8 * numpy.pi * numpy.random.rand(Nt4, 1)  # Angle
    testdata4 = numpy.concatenate((0.3 + r3 * numpy.cos(t3), 7.5 + r3 * numpy.sin(t3)), axis=1)  # points

    # training linear SVM based on CVX optimizer
    x = numpy.concatenate((data1, data2, data3, data4), axis=0)
    y = numpy.array([0 for _ in range(N1)] + [1 for _ in range(N2)] + [2 for _ in range(N3)] + [3 for _ in range(N4)])

    xt = numpy.concatenate((testdata1, testdata2, testdata3, testdata4), axis=0)
    yt = numpy.array([0 for _ in range(Nt1)] + [1 for _ in range(Nt2)] + [2 for _ in range(Nt3)] + [3 for _ in range(Nt4)])

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x, y)
    ycap_lib = clf.predict(x)
    ytcap_lib = clf.predict(xt)
    print("=====================Train=====================")
    print("RMSE Loss: ", sqrt(sklearn.metrics.mean_squared_error(y, ycap_lib)))
    print("Diff Ratio: ", sqrt(numpy.sum(ycap_lib != y) / x.shape[0]))
    print("=====================Test======================")
    print("RMSE Loss: ", sklearn.metrics.mean_squared_error(yt, ytcap_lib))
    print("Diff Ratio: ", numpy.sum(ytcap_lib != yt) / xt.shape[0])

    model = L2SVM(x, y, {0, 1, 2, 3}, lambda_regularizationController=0.5)
    w, b = model.train()
    ycap = model.predict(x, y)
    ytcap = model.predict(xt, yt)

    # visualize decision boundary for training data
    vectorDrawingScalar = 1e1
    origin = [0], [0]
    y = lambda nv, c, x: (nv[0] * x - c) / (-nv[1])

    plt.figure(0)
    xGrid = numpy.linspace(-0.3, 0.3)
    d1 = plt.scatter(data1[:, 0], data1[:, 1], s=15, marker=".", c="r")
    d2 = plt.scatter(data2[:, 0], data2[:, 1], s=15, marker=".", c="b")
    d3 = plt.scatter(data3[:, 0], data3[:, 1], s=15, marker=".", c="g")
    d4 = plt.scatter(data4[:, 0], data4[:, 1], s=15, marker=".", c="y")

    # Draw Weights as Vectors
    plt.quiver(*origin, vectorDrawingScalar * w[0], vectorDrawingScalar * w[1], color=["r", "b", "g", "y"], scale=21)

    plt.legend((d1, d2, d3, d4), ("0", "1", "2", "3"))

    # visualize decision boundary for testing data
    plt.figure(1)
    xGrid = numpy.linspace(-0.3, 0.3)
    dt1 = plt.scatter(testdata1[:, 0], testdata1[:, 1], s=15, marker=".", c="r")
    dt2 = plt.scatter(testdata2[:, 0], testdata2[:, 1], s=15, marker=".", c="b")
    dt3 = plt.scatter(testdata3[:, 0], testdata3[:, 1], s=15, marker=".", c="g")
    dt4 = plt.scatter(testdata4[:, 0], testdata4[:, 1], s=15, marker=".", c="y")

    # Draw Weights as Vectors
    plt.quiver(*origin, vectorDrawingScalar * w[0], vectorDrawingScalar * w[1], color=["r", "b", "g", "y"], scale=21)

    plt.legend((dt1, dt2, dt3, dt4), ("0", "1", "2", "3"))

    plt.show()


if __name__ == '__main__':
    main()
