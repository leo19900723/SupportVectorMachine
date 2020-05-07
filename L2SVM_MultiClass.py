import math
import cvxpy
import random
import numpy
import matplotlib.pyplot as plt


class L2SVM(object):

    def __init__(self, x_trainDataPoints, y_trainLabels, K_allClassSet, lambda_regularizationController):
        self._x_trainDataPoints = x_trainDataPoints
        self._y_trainLabels = y_trainLabels
        self._K_allClassSet = K_allClassSet
        self._lambda_regularizationController = lambda_regularizationController

        self._w_weights = cvxpy.Variable((self._x_trainDataPoints.shape[1] * len(self._K_allClassSet), 1))
        self._b_biases = cvxpy.Variable(len(self._K_allClassSet))
        self._b_biases.value = numpy.zeros(len(self._K_allClassSet))

    def _psi_classSensitiveFeatureMapping(self, x, y):
        psi = numpy.zeros((self._x_trainDataPoints.shape[1] * len(self._K_allClassSet)))
        psi[int(y) * self._x_trainDataPoints.shape[1]: (int(y) + 1) * self._x_trainDataPoints.shape[1]] = x
        return psi

    def train(self):
        primalCost = 0.5 * (cvxpy.norm(self._w_weights) ** 2)
        # hingeLoss_multiClass_withPsi = cvxpy.sum([cvxpy.sum([cvxpy.pos(self._w_weights.T @ self._psi_classSensitiveFeatureMapping(self._x_trainDataPoints[instanceIndex], ycap) - self._w_weights.T @ self._psi_classSensitiveFeatureMapping(self._x_trainDataPoints[instanceIndex], self._y_trainLabels[instanceIndex])) for ycap in self._K_allClassSet]) for instanceIndex in range(self._x_trainDataPoints.shape[0])]) / (self._x_trainDataPoints.shape[0] * len(self._K_allClassSet))
        # hingeLoss_multiClass_withoutPsi = cvxpy.sum([cvxpy.sum([cvxpy.pos(self._w_weights[ycap * self._x_trainDataPoints.shape[1]: (ycap + 1) * self._x_trainDataPoints.shape[1]].T @ self._x_trainDataPoints[instanceIndex] - self._w_weights[self._y_trainLabels[instanceIndex] * self._x_trainDataPoints.shape[1]: (self._y_trainLabels[instanceIndex] + 1) * self._x_trainDataPoints.shape[1]].T @ self._x_trainDataPoints[instanceIndex]) for ycap in self._K_allClassSet]) for instanceIndex in range(self._x_trainDataPoints.shape[0])]) / (self._x_trainDataPoints.shape[0] * len(self._K_allClassSet))
        hingeLoss_multiClass_biasWithoutPsi = cvxpy.sum([cvxpy.sum([cvxpy.pos(self._w_weights[ycap * self._x_trainDataPoints.shape[1]: (ycap + 1) * self._x_trainDataPoints.shape[1]].T @ self._x_trainDataPoints[instanceIndex] - self._w_weights[self._y_trainLabels[instanceIndex] * self._x_trainDataPoints.shape[1]: (self._y_trainLabels[instanceIndex] + 1) * self._x_trainDataPoints.shape[1]].T @ self._x_trainDataPoints[instanceIndex] + self._b_biases[ycap] - self._b_biases[self._y_trainLabels[instanceIndex]]) for ycap in self._K_allClassSet]) for instanceIndex in range(self._x_trainDataPoints.shape[0])]) / (self._x_trainDataPoints.shape[0] * len(self._K_allClassSet))

        trainingObjectiveFunction = cvxpy.Minimize(primalCost + self._lambda_regularizationController * hingeLoss_multiClass_biasWithoutPsi)

        trainingProblem = cvxpy.Problem(trainingObjectiveFunction)
        trainingProblem.solve()

        print("status:", trainingProblem.status)
        print("optimal value", trainingProblem.value)
        print("optimal var w =", self._w_weights.value, ", b =", self._b_biases.value)

        return self._w_weights.value, self._b_biases.value

    def predict(self, xt_testDataPoints, yt_testLabels):
        ycap_predictedLabels = numpy.empty(xt_testDataPoints.shape[0], dtype=int)
        w = self._w_weights.value
        b = self._b_biases.value

        for instanceIndex in range(xt_testDataPoints.shape[0]):
            score = -math.inf
            for k in self._K_allClassSet:
                newScore = w.T @ self._psi_classSensitiveFeatureMapping(xt_testDataPoints[instanceIndex], k) + b[k]

                if newScore > score:
                    ycap_predictedLabels[instanceIndex] = k
                    score = newScore

        print("RMSE Loss: ", numpy.sqrt(numpy.nanmean((yt_testLabels - ycap_predictedLabels) ** 2)))
        print("Diff Ratio: ", numpy.sum(numpy.power(yt_testLabels - ycap_predictedLabels, 2)) / xt_testDataPoints.shape[0])
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
    #print("Loading Files...")
    #x_dataPoints, y_labels = datasets.load_svmlight_file("InputFiles/a1a")

    return


def _unitTest():
    # generate toy training data
    N1 = 100  # number of C0 instances
    N2 = 100  # number of C1 instances
    N3 = 100  # number of C2 instances
    random.seed(1)  # For reproducibility

    r1 = numpy.sqrt(1.5 * numpy.random.rand(N1, 1))  # Radius
    t1 = 2 * numpy.pi * numpy.random.rand(N1, 1)  # Angle
    data1 = numpy.concatenate((r1 * numpy.cos(t1), r1 * numpy.sin(t1)), axis=1)  # Points

    r2 = numpy.sqrt(3 * numpy.random.rand(N2, 1))  # Radius
    t2 = 2 * numpy.pi * numpy.random.rand(N2, 1)  # Angle
    data2 = numpy.concatenate((2.5 + r2 * numpy.cos(t2), 1.5 + r2 * numpy.sin(t2)), axis=1)  # points

    r3 = numpy.sqrt(1 * numpy.random.rand(N3, 1))  # Radius
    t3 = -2 * numpy.pi * numpy.random.rand(N3, 1)  # Angle
    data3 = numpy.concatenate((4 + r3 * numpy.cos(t3), -1 + r3 * numpy.sin(t3)), axis=1)  # points

    # generate toy testing data
    Nt1 = 50  # number of C0 instances
    Nt2 = 50  # number of C1 instances
    Nt3 = 50  # number of C2 instances

    r1 = numpy.sqrt(3.4 * numpy.random.rand(Nt1, 1))  # Radius
    t1 = 2 * numpy.pi * numpy.random.rand(Nt1, 1)  # Angle
    testdata1 = numpy.concatenate((r1 * numpy.cos(t1), r1 * numpy.sin(t1)), axis=1)  # Points

    r2 = numpy.sqrt(2.4 * numpy.random.rand(Nt2, 1))  # Radius
    t2 = 2 * numpy.pi * numpy.random.rand(Nt2, 1)  # Angle
    testdata2 = numpy.concatenate((3 + r2 * numpy.cos(t2), 0.8 + r2 * numpy.sin(t2)), axis=1)  # points

    r3 = numpy.sqrt(1.3 * numpy.random.rand(Nt3, 1))  # Radius
    t3 = -2 * numpy.pi * numpy.random.rand(Nt3, 1)  # Angle
    testdata3 = numpy.concatenate((3.5 + r3 * numpy.cos(t3), -1.2 + r3 * numpy.sin(t3)), axis=1)  # points

    # training linear SVM based on CVX optimizer
    x = numpy.concatenate((data1, data2, data3), axis=0)
    y = numpy.array([0 for _ in range(N1)] + [1 for _ in range(N2)] + [2 for _ in range(N2)])

    xt = numpy.concatenate((testdata1, testdata2, testdata3), axis=0)
    yt = numpy.array([0 for _ in range(Nt1)] + [1 for _ in range(Nt2)] + [2 for _ in range(Nt3)])

    model = L2SVM(x, y, {0, 1, 2}, lambda_regularizationController=0.5)
    w, b = model.train()
    ycap = model.predict(x, y)
    ytcap = model.predict(xt, yt)

    # visualize decision boundary for training data
    vectorDrawingScalar = 100000000
    origin = [0], [0]
    y = lambda nv, c, x: (nv[0] * x - c) / (-nv[1])
    v = vectorDrawingScalar * w.reshape((w.shape[0] // 2, 2))

    plt.figure(0)
    xGrid = numpy.linspace(-0.3, 0.3)
    d1 = plt.scatter(data1[:, 0], data1[:, 1], s=15, marker=".", c="r")
    d2 = plt.scatter(data2[:, 0], data2[:, 1], s=15, marker=".", c="b")
    d3 = plt.scatter(data3[:, 0], data3[:, 1], s=15, marker=".", c="g")

    # Draw Weights as Vectors
    plt.quiver(*origin, v[:, 0], v[:, 1], color=["r", "b", "g"], scale=21)

    plt.plot(xGrid, y(v[0], b[0], xGrid), c="r")
    plt.plot(xGrid, y(v[1], b[1], xGrid), c="b")
    plt.plot(xGrid, y(v[2], b[2], xGrid), c="g")

    plt.legend((d1, d2, d3), ("0", "1", "2"))

    # visualize decision boundary for testing data
    plt.figure(1)
    xGrid = numpy.linspace(-0.3, 0.3)
    dt1 = plt.scatter(testdata1[:, 0], testdata1[:, 1], s=15, marker=".", c="r")
    dt2 = plt.scatter(testdata2[:, 0], testdata2[:, 1], s=15, marker=".", c="b")
    dt3 = plt.scatter(testdata3[:, 0], testdata3[:, 1], s=15, marker=".", c="g")

    # Draw Weights as Vectors
    plt.quiver(*origin, v[:, 0], v[:, 1], color=["r", "b", "g"], scale=21)

    plt.plot(xGrid, y(v[0], b[0], xGrid), c="r")
    plt.plot(xGrid, y(v[1], b[1], xGrid), c="b")
    plt.plot(xGrid, y(v[2], b[2], xGrid), c="g")

    plt.legend((dt1, dt2, dt3), ("0", "1", "2"))

    plt.show()


if __name__ == '__main__':
    _unitTest()
