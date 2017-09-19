""""
Filename:    benchmark.py

Description: This file contains an n-D training data generation class. This generates sets of 
linearly classifiable and non-linearly classifiable data passed in various classification tags.
"""""

import numpy as np
from shapely.geometry import LineString, Point, Polygon
import shapely.affinity


class Benchmark(object):
    """ Methods:    generate_linear = Generate a list of linearly separable classified points."""

    # Number of points to generate to check if gamma condition is satisfied for non-linear data.
    GCP = 1000

    @staticmethod
    def random_vector(i):
        """ Generate a normalized random vector of length 'i'. Range: [-1, 1].
        
        :param i: Length of the vector to generate.
        :return: A vector of length 'i'.
        """
        return 2 * np.random.rand(i) - 1

    @staticmethod
    def __d_passing_gamma(n, gamma, distance_to_b, i=2, scale=1):
        """ Generate a random list of points whose distance to the decision boundary is greater 
        than the given gamma.
        
        :param n: Number of points to generate. These are randomly plotted.
        :param gamma: Minimum separation between the two "classes" of data in D.
        :param distance_to_b: Distance finding function. 
        :param i: Dimensionality of the points themselves (2D, 3D, etc...).
        :param scale: The scale of the points and the decision boundary. Defaults to 1.
        :return: Vector of points that meet the defined gamma condition.
        """
        d = []
        while len(d) < n:
            p = random_vector(i) * scale
            distance_to_b(p) > gamma and d.append(p)

        return d

    @staticmethod
    def generate_linear(n, gamma, i, scale=1):
        """ Generate a random list of points that meet the given criteria. Generate a random 
        hyperplane and classify the data using this decision boundary.

        :param n: Number of points to generate. These are randomly plotted.
        :param gamma: Minimum separation between the two "classes" of data in D.
        :param i: Dimensionality of the points themselves (2D, 3D, etc...).
        :param scale: The scale of the points and the classifying weight vector. Defaults to 1.
        :return: n * i matrix of random points, a n-long vetor of labels corresponding to the 
                 first set of points, and the decision boundary weight vector.
        """
        d_from_gamma = lambda x, w: np.abs(np.dot(x, w[1:]) + w[0]) / (len(w) - 1)
        theta = lambda x, w: 1 if np.dot(w[1:], p.x) + w[0] > 0 else -1

        # Generate our decision boundary (w_star). Scale appropriately.
        w_star = random_vector(i + 1) * scale

        # Generate our random points. Gamma condition must be met for each point added.
        d = __d_passing_gamma(n, gamma, lambda a: d_from_gamma(a, w_star) > gamma, i, scale)

        # We classify each point given w_star. Perform for each point in D.
        ell = [theta(x, w_star) for x in d]

        return d, ell, w_star

    @staticmethod
    def generate_polynomial(n, gamma, degree, scale=1):
        """ Generate a random list of **2D** points that meet the given criteria and are able to 
        be classified by some polynomial of the given degree.
        
        :param n: Number of points to generate. These are randomly plotted.
        :param gamma: Minimum separation between the two "classes" of data in D.
        :param degree: Degree of the polynomial that will classify the given points.
        :param scale: The scale of the points and the decision boundary. Defaults to 1.
        :return: n * 2 matrix of random points, a n-long vector of labels corresponding to the 
                 first set of points, and the decision boundary polynomial coefficients.
        """
        d_from_gamma = lambda x, b_curve: Point(x[0], x[1]).distance(b_curve)
        evaluate_p = lambda i, b: sum([i * b[a] ** (degree - a) for a in range(0, degree + 1)])
        theta = lambda x, b: 1 if evaluate_p(x[0], b) < x[1] else -1

        # Generate our decision boundary curve. b[0] = a in ax^3, b[1] = a in ax^2, ...
        b = random_vector(3) * scale
        b_curve = LineString([evaluate_p(i, b) for i in range(-GCP, GCP)])

        # Generate our random points. Gamma condition must be met for each point added.
        d = __d_passing_gamma(n, gamma, lambda a: d_from_gamma(a, b_curve) > gamma, scale=scale)

        # We classify each point given decision boundary. Perform for each point in D.
        ell = [theta(x, b) for x in d]

        return d, ell, b

    @staticmethod
    def generate_ellipse(n, gamma, circle=False, scale=1):
        """ Generate a random list of **2D** points that meet the given criteria and are 
        classified by some ellipse. If desired, restrict the ellipse to just a circle.
        
        :param n: Number of points to generate. These are randomly plotted.
        :param gamma: Minimum separation between the two "classes" of data in D.
        :param circle: Flag to define decision ellipse as a circle.
        :param scale: The scale of the points and the decision boundary. Defaults to 1.
        :return: n * 2 matrix of random points, a n-long vector of labels corresponding to the 
                 first set of points, and the decision ellipse description.
        """
        d_from_gamma = lambda x, b_ellipse: Point(x[0], x[1]).distance(b_ellipse)
        theta = lambda x, b_ellipse: Point(x[0], x[1]).within(b_ellipse)

        # Generate our decision ellipse. b[0], b[1] = ellipse center. b[2], b[3] = a, b terms.
        b = [random_vector(2) * scale].append(abs(random_vector(2) * scale))
        b_circle = Point(b[0], b[1]).buffer(1)
        b_ellipse = b_circle if circle else shapely.affinity.scale(b_circle, b[2], b[3])

        # Generate our random points. Gamma condition must be met for each point added.
        d = __d_passing_gamma(n, gamma, lambda a: d_from_gamma(a, b_ellipse) > gamma, scale=scale)

        # We classify each point given decision boundary. Perform for each point in D.
        ell = [theta(x, b_ellipse) for x in d]

        return d, ell, b

    @staticmethod
    def generate_rectangle(n, gamma, scale=1):
        """ Generate a random list of **2D** points that meet the given crieteria and are 
        classified by some rectange. 
        
        :param n: Number of points to generate. These are randomly plotted.
        :param gamma: Minimum separation between the two "classes" of data in D.
        :param scale: The scale of the points and the decision boundary. Defaults to 1.
        :return: n * 2 matrix of random points, a n-long vector of labels corresponding to the 
                 first set of points, and the decision rectange upper right and bottom left 
                 corners.
        """
        d_from_gamma = lambda x, b_rectangle: Point(x[0], x[1]).distance(b_rectangle)
        theta = lambda x, b_rectangle: Point(x[0], x[1]).within(b_rectangle)

        # Generate our decision rectangle. b[0:1] = upper right, b[2:3] = bottom left
        b = sorted(random_vector(4))
        b_rectangle = Polygon([[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]])

        # Generate our random points. Gamma condition must be met for each point added.
        d = __d_passing_gamma(n, gamma, lambda a: d_from_gamma(a, b_rectangle) > gamma, scale=scale)

        # We classify each point given decision boundary. Perform for each point in D.
        ell = [theta(x, b_rectangle) for x in d]

        return d, ell, b
    