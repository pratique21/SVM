from __future__ import division
import numpy
from benchmark import Benchmark
from shapely.geometry import LineString, Point, Polygon
from visualize import Visualize
from sklearn import svm
from sklearn.metrics import confusion_matrix
from auxillary import check_label_diversity, confusion_test, write_data_to_csv
import matplotlib.pyplot as plt
def datacollector(n_start,n_end,iterations,gamma,training_data_restriction,kernel_type,training_data_type="poly",degree_of_polynomial=1):
    """

    :param n_start: Starting point for number of data points
    :param n_end:  Ending point for number of data points
    :param iterations: Iterations over whih errors will be averaged for each n, the number of data points
    :param gamma:
    :param training_data_restriction: Values from 0-1 specify what percentage of available training data to be used for training
    :param kernel_type: "poly","rbf","linear" or "sigmoid"
    :param training_data_type: calls the corresponding function fro Benchmark. Currently wdefaulted to "poly"
    :param degree_of_polynomial: Degree of polynomial for the corresponding data type
    :return: Stores (x,y) on excel sheet with x: no. of data points and y: average number of errors over specified iterations
    """
    average_error = []
    sum = 0
    for j in range(n_start, n_end):
        for i in range(iterations):
            x = Benchmark.generate_polynomial(j, gamma, degree_of_polynomial)
            while (check_label_diversity(x[1])):
                x = Benchmark.generate_polynomial(j, gamma, degree_of_polynomial)
            sum += confusion_test(x, training_data_restriction, kernel_type)
        average_error.append([j, float(sum / iterations)])
        sum = 0
    write_data_to_csv(average_error, "_Tdatatype "+ training_data_type+"_KType "+ kernel_type+ "_n_range " + str(n_start)+"-" +str(n_end))
datacollector(100,105,10,0.2,0.5,"poly","poly",2)