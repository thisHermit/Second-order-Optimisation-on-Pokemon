"""
Implementation of a Second Order Optimisation on a dataset.

Dataset taken from kaggle.
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

epsilon = 0.01
# Reading data
raw_data = pd.read_csv(
    "/media/guest/DATA2/Practice_for_ML/linearRegression/" +
    "pokemon_alopez247.csv")
d = {"Total": raw_data['Total'],
     "HP": raw_data['HP']}
data = pd.DataFrame(d).values


def jacobian(point, h=1e-5):
    n = len(point)
    jacobian = np.zeros(n)
    for i in range(n):
        smallIncrease = np.zeros(n)
        smallIncrease[i] += h
        jacobian[i] = (totalError(point + smallIncrease) -
                       totalError(point)) / h
    return jacobian


def hessian(point, h=1e-5):
    n = len(point)
    hessian = np.zeros((n, n))
    for i in range(n):
        smallIncrease = np.zeros(n)
        smallIncrease[i] += h
        # partOne = (np.array([jacobian(point + smallIncrease)]).T -
        #           np.array([jacobian(point)]).T)/h
        hessian[i] = (jacobian(point + smallIncrease) - jacobian(point)) / h
    return hessian


def newton(iterations, start_point):
    point = start_point
    for i in range(iterations):
        print("inv", np.dot(np.linalg.inv(hessian(point)), np.array([jacobian(point)]).T))
        tempPoint = np.array([point])
        tempPoint = tempPoint.T
        tempPoint = tempPoint - np.dot(np.linalg.inv(hessian(point)), np.array([jacobian(point)]).T)
        point = np.ndarray.flatten(tempPoint)
        print("point :", point)
        print("error :", totalError(point))


def compute_error_for_line(m, b):
    """Return the Error for Line given the points."""
    totalError = 0
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(data))


def totalError(one_point):
    return compute_error_for_line(one_point[0], one_point[1])


def main():
    testpoint = np.array(
        [2.72527749446, 231.602391353])
    startpoint = np.array(
        [2, 230])
    # print(jacobian(testpoint))
    # print(hessian(testpoint))
    newton(1000, startpoint)


if __name__ == '__main__':
    main()
