import math
import cvxpy
import random
import numpy
import pandas
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


class L2SVM_Binary(object):

    def __init__(self, x_trainDataPoints, y_trainLabels, supportVectorCriteria, lambda_regularizationController):
        self._w_weights = {
            "source": cvxpy.Variable((x_trainDataPoints.shape[1], 1)),
            "ht": cvxpy.Variable((x_trainDataPoints.shape[1], 1)),
            "it": cvxpy.Variable((x_trainDataPoints.shape[1], 1))
        }

        self._b_biases = {
            "source": cvxpy.Variable(),
            "ht": cvxpy.Variable(),
            "it": cvxpy.Variable()
        }

        self._lambda_regularizationController = lambda_regularizationController

        self.fit(x_trainDataPoints, y_trainLabels, transferMode="source")

        dist = (y_trainLabels * (x_trainDataPoints.dot(self._w_weights["source"].value) + self._b_biases["source"].value)) - 1
        self._xsv_supportVectorsDataPoints = x_trainDataPoints[((-supportVectorCriteria < dist) & (dist < supportVectorCriteria)).flatten(), :]
        self._ysv_supportVectorsLabels = y_trainLabels[((-supportVectorCriteria < dist) & (dist < supportVectorCriteria)).flatten(), :]
        print("# of SV:", self._xsv_supportVectorsDataPoints.shape[0])

    def fit(self, x_trainDataPoints, y_trainLabels, transferMode):
        print("[", transferMode, "] Training...")
        e_softError = cvxpy.Variable((x_trainDataPoints.shape[0], 1))
        primalCost = 0.5 * cvxpy.sum_squares(self._w_weights[transferMode])

        if transferMode == "source":
            constraints = [
                cvxpy.multiply(y_trainLabels, (x_trainDataPoints @ self._w_weights[transferMode] + self._b_biases[transferMode])) >= 1 - e_softError,
                e_softError >= 0
            ]

            trainingObjectiveFunction = cvxpy.Minimize(primalCost + self._lambda_regularizationController * cvxpy.sum(e_softError))
        elif transferMode == "ht":
            constraints = [
                cvxpy.multiply(y_trainLabels, (x_trainDataPoints @ self._w_weights[transferMode] + self._b_biases[transferMode])) >= 1 - e_softError,
                e_softError >= 0,
                cvxpy.sum_squares((self._w_weights["source"].value - self._w_weights[transferMode])) <= 100
            ]

            trainingObjectiveFunction = cvxpy.Minimize(primalCost + self._lambda_regularizationController * cvxpy.sum(e_softError))
        elif transferMode == "it":
            es_sourceSoftError = cvxpy.Variable((self._xsv_supportVectorsDataPoints.shape[0], 1))

            constraints = [
                cvxpy.multiply(y_trainLabels, (x_trainDataPoints @ self._w_weights[transferMode] + self._b_biases[transferMode])) >= 1 - e_softError,
                e_softError >= 0,
                cvxpy.multiply(self._ysv_supportVectorsLabels, (self._xsv_supportVectorsDataPoints @ self._w_weights[transferMode] + self._b_biases[transferMode])) >= 1 - es_sourceSoftError,
                es_sourceSoftError >= 0
            ]

            trainingObjectiveFunction = cvxpy.Minimize(primalCost + self._lambda_regularizationController * cvxpy.sum(e_softError) + self._lambda_regularizationController * cvxpy.sum(es_sourceSoftError))
        else:
            return

        trainingProblem = cvxpy.Problem(trainingObjectiveFunction, constraints)
        trainingProblem.solve(solver="SCS")

        print("status:", trainingProblem.status)
        print("optimal value", trainingProblem.value)

    def predict(self, xt_testDataPoints, transferMode):
        print("[", transferMode, "] Predicting...")
        return 2 * (xt_testDataPoints @ self._w_weights[transferMode].value + self._b_biases[transferMode].value > 0).astype(int) - 1

    def score(self, xt_testDataPoints, yt_testLabels, transferMode=None):
        rmseVal, scoreVal = {}, {}
        ycap_predictedLabels = {"source": None, "ht": None, "it": None}

        for methods in (ycap_predictedLabels.keys() if transferMode is None else {transferMode}):
            ycap_predictedLabels[methods] = self.predict(xt_testDataPoints, transferMode=methods)

            rmseVal[methods] = math.sqrt(sklearn.metrics.mean_squared_error(yt_testLabels, ycap_predictedLabels[methods]))
            scoreVal[methods] = numpy.sum(yt_testLabels == ycap_predictedLabels[methods]) / xt_testDataPoints.shape[0]
            print("=======================")
            print(methods + "- RMSE Loss: ", rmseVal[methods])
            print(methods + "- ACCU Ratio: ", scoreVal[methods])

        return scoreVal

    @property
    def viewWeights(self):
        return self._w_weights

    @property
    def viewBiases(self):
        return self._b_biases


def main():
    print("Loading Files...")
    trainFile = pandas.read_csv("InputFiles/train.csv")

    xs_sourceDataPoints, ys_sourceLabels = trainFile.loc[trainFile["Digit"].isin({1, 9}), "x3": "y6"].to_numpy(), trainFile.loc[trainFile["Digit"].isin({1, 9}), "Digit"].replace(9, -1).to_numpy()
    xt_targetDataPoints, yt_targetLabels = trainFile.loc[trainFile["Digit"].isin({1, 7}), "x3": "y6"].to_numpy(), trainFile.loc[trainFile["Digit"].isin({1, 7}), "Digit"].replace(7, -1).to_numpy()

    ys_sourceLabels = ys_sourceLabels.reshape((ys_sourceLabels.shape[0], 1))
    yt_targetLabels = yt_targetLabels.reshape((yt_targetLabels.shape[0], 1))

    xs_train, xs_val, ys_train, ys_val = train_test_split(xs_sourceDataPoints, ys_sourceLabels, test_size=0.2, random_state=1)
    xt_train, xt_val, yt_train, yt_val = train_test_split(xt_targetDataPoints, yt_targetLabels, test_size=0.3, random_state=1)

    model = L2SVM_Binary(xs_train, ys_train, supportVectorCriteria=1, lambda_regularizationController=0.05)
    model.fit(xt_train, yt_train, transferMode="ht")
    model.fit(xt_train, yt_train, transferMode="it")
    model.fit(xt_train, yt_train, transferMode="source")

    model.score(xt_val, yt_val)
    model.score(xt_targetDataPoints, yt_targetLabels)

    return


def _unitTest():
    # generate toy training data
    N1 = 200  # number of positive instances
    N2 = 100  # number of negative instances
    D = 2  # feature dimension
    eps = 1e-8  # select support vectors
    random.seed(1)  # For reproducibility
    r1 = numpy.sqrt(1.5 * numpy.random.rand(N1, 1))  # Radius
    t1 = 2 * numpy.pi * numpy.random.rand(N1, 1)  # Angle
    data1 = numpy.concatenate((r1 * numpy.cos(t1), 3 + r1 * numpy.sin(t1)), axis=1)  # Points
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
    testdata1 = numpy.concatenate((r1 * numpy.cos(t1), 3.25 + r1 * numpy.sin(t1)), axis=1)  # Points
    r2 = numpy.sqrt(2.4 * numpy.random.rand(Nt2, 1))  # Radius
    t2 = 2 * numpy.pi * numpy.random.rand(Nt2, 1)  # Angle
    testdata2 = numpy.concatenate((3 + r2 * numpy.cos(t2), r2 * numpy.sin(t2)), axis=1)  # points
    ## training linear SVM based on CVX optimizer
    X = numpy.concatenate((data1, data2), axis=0)
    y = numpy.concatenate((numpy.ones((N1, 1)), - numpy.ones((N2, 1))), axis=0)

    Xt = numpy.concatenate((testdata1, testdata2), axis=0)
    yt = numpy.concatenate((numpy.ones((Nt1, 1)), - numpy.ones((Nt2, 1))), axis=0)

    model = L2SVM_Binary(X, y, eps, 0.02)
    w, b = model.viewWeights["source"], model.viewBiases["source"]
    model.fit(X, y, transferMode="ht")
    model.fit(X, y, transferMode="it")
    model.score(Xt, yt)

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
    main()
