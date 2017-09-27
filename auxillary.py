import numpy
from benchmark import Benchmark
from shapely.geometry import LineString, Point, Polygon
from sklearn.metrics import confusion_matrix
from visualize import Visualize
from sklearn import svm
import matplotlib.pyplot as plt
import xlsxwriter
def calculate_margin(distances):
    above_margin=[]
    below_margin=[]
    d_p=0
    d_m=0
    for d in distances:
        if d>0:
            above_margin.append(d)
        else:
            below_margin.append(d)
    if(len(above_margin) != 0):
        d_p=numpy.amin(above_margin)
    if(len(below_margin) != 0):
        d_m=numpy.amax(below_margin)
    return abs(0.5*(d_p-d_m))
def polynomial_kernel_margin_tester(gamma,degree_of_polynomial, n_start,n_end):
    margin_data=[]
    ok="True"
    for i in range(n_start,n_end):
        number_of_data_points = i
        while (ok):
            x = Benchmark.generate_polynomial(number_of_data_points, gamma, degree_of_polynomial)
            X = x[0]
            Y = x[1]
            ok = all(p == Y[0] for p in Y)
        clf = svm.SVC(kernel="sigmoid", gamma=3)
        clf.fit(x[0], x[1])
        margin_data.append([i,calculate_margin(clf.decision_function(x[0]))])
        ok="True"
    return margin_data
def check_label_diversity(labels):
    ok = all(p == labels[0] for p in labels)
    return ok
def confusion_test(dataset,restriction,kernel_type):
    restricted_training_data=[]
    restricted_training_label=[]
    chk=0
    for i in range(len(dataset[1])):
        if(dataset[1][0]*dataset[1][i] < 0):
            restricted_training_data.append(dataset[0][0])
            restricted_training_data.append(dataset[0][i])
            restricted_training_label.append(dataset[1][0])
            restricted_training_label.append(dataset[1][i])
            chk=i
            break
    data_limit=int(restriction*len(dataset[0]))
    for i in range(1,data_limit):
        if(i > data_limit):
            restricted_training_data.append(dataset[0][i])
            restricted_training_label.append(dataset[1][i])
        else:
            if(i != chk):
                restricted_training_data.append(dataset[0][i])
                restricted_training_label.append(dataset[1][i])
    clf = svm.SVC(kernel=kernel_type,gamma=3)
    clf.fit(restricted_training_data,restricted_training_label)
    predicted_labels=clf.predict(dataset[0])
    associated_conf_matrix = confusion_matrix(dataset[1],predicted_labels)
    total_errors=associated_conf_matrix[0][1]+associated_conf_matrix[1][0]
    return total_errors
def write_data_to_csv(average_errors,caselabel):
    workbook = xlsxwriter.Workbook('AverageErrors'+str(caselabel)+'.xlsx')
    worksheet = workbook.add_worksheet()
    row=0
    col=0
    for n, error in (average_errors):
        worksheet.write(row,col,n)
        worksheet.write(row,col+1,error)
        row+=1
    workbook.close()

