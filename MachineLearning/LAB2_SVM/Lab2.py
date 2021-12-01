# KTH-ECTS
# Machine Learning
# Lab2
# Donggyun and Daniel

import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':

    def kernel(point1, point2, type_kernel):
        # Function that given two vector points, returns the scalar product
        # according to type_ker, it will return one scalar product
        if type_kernel == 'lin':
            return np.matmul(np.transpose(point1), point2)
        elif type_kernel == 'poly':
            p = 2  # By default, can be higher, but more complex shapes
            product = np.matmul(np.transpose(point1), point2)
            return (product + 1) ** p
        else:  # Case for RBF (Radial Basis Function)
            sigma = 1  # Parameter that adjusts the boundary smoothness
            dist = np.linalg.norm(point1 - point2)
            return math.exp(-1 * dist ** 2 / (2 * sigma ** 2))


    def objective(input_vector_alpha):
        # Function that given a vector, returns a scalar value acording to the dual formation problem
        value = 0
        for i in range(len(input_vector_alpha)):
            for j in range(len(input_vector_alpha)):
                value += .5 * input_vector_alpha[i] * input_vector_alpha[j] * P[i][j] - input_vector_alpha[i]
        return value

    def zerofun(input_vector):
        # Function that given a vector, returns the value(scalar) that will be constrained to zero
        # XC is used to impose the equality constrains of SUM(alpha_i * t_i) = 0
        s = 0
        for value in range(len(input_vector)):
            s += input_vector[value] * t[value]
        return s


    def indicator(input_non_zero_alphas_extracted, data_to_check):
        # Function that classifies new points given the non zero alphas and their corresponding
        # data points and target values
        r = 0
        for xx in range(len(input_non_zero_alphas_extracted)):
            r += input_non_zero_alphas_extracted[xx][0] * input_non_zero_alphas_extracted[xx][2] \
                 * kernel(data_to_check, input_non_zero_alphas_extracted[xx][1], type_ker)
        return r - b


    # 3. Theory. Build a classifier that:
    #       1. (Optional) Transforms input data
    #       2. Separates the data into subsets.
    #           2.1. The separation is given by the decision boundary
    #           2.2. The decision boundary is descrived by w(weights) and b(bias)
    #           2.3. We must maximize the distance from the boundary to any given point
    #
    #

    # 4. Implementation
    # Variables
    np.random.seed(100)
    classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2) * 0.2 + [0, -0.5]
    x = np.concatenate((classA, classB))
    t = np.concatenate((np.ones(classA.shape[0]), -1 * np.ones(classB.shape[0])))
    N = x.shape[0]

    permute = list(range(N))
    random.shuffle(permute)
    x = x[permute, :]
    t = t[permute]

    C = 10  # IF no upper bound is wanted, C= None
    XC = {'type': 'eq', 'fun': zerofun}  # Imposes the equality constraint
    start = np.zeros(N)
    B = [(0, C) for b in range(N)]
    type_ker = 'lin'
    P = np.zeros((N, N))

    for row in range(N):
        for col in range(N):
            P[row][col] = t[row] * t[col] * kernel(x[row], x[col], type_ker)

    ret = minimize(objective, start, bounds=B, constraints=XC)

    alpha_minimize = ret['x']
    alpha_minimize_threshold = [alpha_minimize[i] if alpha_minimize[i] > 10 ** (-5)
                                else 0 for i in range(len(alpha_minimize))]
    data_extraction = []
    for p in range(len(alpha_minimize_threshold)):  # We save the values where alphas are not zero, their x and t
        if alpha_minimize_threshold[p] != 0:
            data_extraction.append([alpha_minimize_threshold[p], x[p], t[p]])  # Stored: alpha, x, t
    # b is computed
    # but first we obtain s, as one SV that is lower than C
    s = []
    for xx in range(len(data_extraction)):
        if 0 < data_extraction[xx][0] < C:
            s = data_extraction[xx]
            break
    # s = next(xx[0] for xx in data_extraction if xx[0] < C)
    print("shape of data_extraction: ", np.shape(data_extraction))
    b = 0
    '''
    for i in range(len(data_extraction[0])):  #
        b += data_extraction[i][0] * data_extraction[i][2] * kernel(s[1], data_extraction[i][1],type_ker)
    b -= s[2]
    '''
    # Plotting
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    plt.axis('equal')
    current_time = time.strftime("%m.%d.%y %H:%M", time.localtime())
    output_name = 'plot_Lab2_%s.jpg' % current_time
    plt.savefig(output_name)
    plt.show()

    # Plotting decision boundries
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[indicator(data_extraction, [xx, yy]) for xx in xgrid] for yy in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1, 0, 1), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    plt.title('Did the optimizer found a solution?' + str(ret['success'])+". C = "+str(C))
