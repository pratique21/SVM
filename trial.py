""""
Filename:    trial.py

Description: This file contains sets of trials to characterize SK-Learn's Support Vector 
             Classification (SVC). We vary the following:
             
             Margin (gamma)
             Number of Points (N)
             C (Complexity Penalty)
             Kernel Function
                 
             And log the accuracy of the classifier (limiting training data) and distance of each 
             point from the decision boundary and the classifier into the CSV.
"""""

from sklearn import svm


def trial_pd(f, gamma, n, c, phi):
    """ Run the specific trial for the given margin (gamma), number of points (N), 
    and the complexity penalty (c). Record the distance of each point to the decision boundary in 
    the feature space, and the distance of each point to the classifying boundary in the vector 
    space (in a CSV file).
    
    :param f: File object to log data to.
    :param gamma: Minimum distance between the points and decision boundary.
    :param n: Number of data.
    :param c: Complexity penalty.
    :param phi: Kernel function to use. There exists: linear, polynomial, rbf, and sigmoid.
    :return: None.
    """
    d_from_line = lambda x, b_line: Point(x[0], x[1]).distance(b_line)
    d_from_polyg = lambda x, b_shape: Point(x[0], x[1]).distance(b_shape.exterior)

    # Generate our data for polynomials (1-4), ellipses, and rectangles.
    d_p1, d_p2, d_p3, d_p4 = [Benchmark.generate_polynomial(n, gamma, a) for a in range(1, 5)]
    d_e, d_r = Benchmark.generate_ellipse(n, gamma), Benchmark.generate_rectangle(n, gamma)

    # For each p* data-set, fit and record the distances to decision boundary and classifier.
    for d_p in [d_p1, d_p2, d_p3, d_p4]:
        clf = svm.SVC(c, kernel=phi)
        clf.fit(d_p[0], d_p[1])

        # Find the distance to the decision boundary, and to the classifier.
        delta_db = np.array(list(map(d_from_line, d_p)))
        delta_clf = np.array(clf.decision_function(d_p))

        # Record each set of distances in a different line.
        for d in [delta in [delta_db, delta_clf]]:
            list(map(lambda d_i: f.write(str(d_i)), d))
            f.write('\n')

    # For each polygon, fit and record the distances to decision boundary and classifier.
    for d_poly in [d_e, d_r]:
        clf = svm.SVC(c, kernel=phi)
        clf.fit(d_poly[0], d_poly[1])

        # Find the distance to the decision boundary, and to the classifier.
        delta_db = np.array(list(map(d_from_polyg, d_p)))
        delta_clf = np.array(clf.decision_function(d_p))

        # Record each set of distances in a different line.
        for d in [delta in [delta_db, delta_clf]]:
            list(map(lambda d_i: f.write(str(d_i)), d))
            f.write('\n')


def trial_cm(f, gamma, n, c, phi):
    """ Run the specific trial for the given margin (gamma), number of points (N), 
    and the complexity penalty (c). We restrict half the data to train the SVC, and obtain the 
    confusion matrix based on the latter half of the data. Record the number of false positives, 
    false negatives, true positives, and true negatives in a CSV file.

    :param f: File object to log data to.
    :param gamma: Minimum distance between the points and decision boundary.
    :param n: Number of data.
    :param c: Complexity penalty.
    :param phi: Kernel function to use. There exists: linear, polynomial, rbf, and sigmoid.
    :return: None.
    """
    training_d = lambda dl: [dl[i] for i in range(0, int(len(dl) / 2))]
    testing_d = lambda dl: [dl[i] for i in range(int(len(dl) / 2), len(dl))]

    # Generate our data for polynomials (1-4), ellipses, and rectangles.
    d_p1, d_p2, d_p3, d_p4 = [Benchmark.generate_polynomial(n, gamma, a) for a in range(1, 5)]
    d_e, d_r = Benchmark.generate_ellipse(n, gamma), Benchmark.generate_rectangle(n, gamma)

    # For each data-set, fit with half the data, and record the confusion matrix with other half.
    for d_p in [d_p1, d_p2, d_p3, d_p4, d_e, d_r]:
        tp, fp, tn, fn = [0 for a in range(0, 4)]

        clf = svm.SVC(c, kernel=phi)
        clf.fit(training_d(d_p[0]), training_d(d_p[1]))

        # Attach the labels to each point, loop through each point and record confusion matrix.
        for t_d in zip(testing_d[d_p[0]], testing_d[d_p[1]]):
            p_vs_a = lambda p, a: 1 if clf.predict([t_d[0], t_d[1]]) == p and t_d[2] == a else 0
            tp, fp, tn, fn = list(map(p_vs_a, [[1, 1], [1, -1], [-1, -1], [-1, 1]]))

        # Record each confusion matrix in a different line.
        list(map(lambda r: f.write(str(r) + ','), [tp, fp, tn, fn]))
        f.write('\n')
