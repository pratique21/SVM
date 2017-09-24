""""
Filename:    visualize.py

Description: This file contains a set of methods (wrapped in a Visualize class) to display 
             training data and decision boundary curves. 
"""""

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString, Polygon


class Visualize(object):
    """ Methods: plot_2d = Produce a plot of a decision boundary and points. 
                 plot_clf = Produce a plot of the classification curve.
    """

    # Step size to generate meshgrid for.
    SSM = 0.002

    @staticmethod
    def __parse_line(b, scale):
        """
        
        :param b: Curve as a Shapely object or perceptron weight vector.
        :param scale: The axis to display.
        :return: X and Y coordinates to pass to Matplotlib.
        """
        if type(b) == LineString:
            return b.coords.xy[0], b.coords.xy[1]
        elif type(b) == Polygon:
            return b.exterior.coords.xy
        else:
            b_x = np.arange(-2 * scale, 2 * scale)
            return b_x, b_x * -b[1] / b[2] - b[0] / b[2]

    @staticmethod
    def plot_clf(d, clf, show=False, dynamic=False, pause=0.01):
        """ Plot the classifying curve as defined by our classifier 'clf'. If desired, the call 
        to "plt.show()" and "plt.ion" can be omitted.
        
        :param d: Data-set containing matrix of size n x 2.
        :param clf: Classifying object (SVC).
        :param show: If false, do not call "plt.show()".
        :param dynamic: If false, do not call "plt.ion()".
        :param pause: Number of seconds to pause plot for.
        """
        d_hat = np.array(d)

        # Define a mesh of points to plot in.
        xx, yy = np.meshgrid(np.arange(d_hat[:,0].min() - 1, d_hat[:,0].max() + 1, Visualize.SSM),
                             np.arange(d_hat[:,1].min() - 1, d_hat[:,1].max() + 1, Visualize.SSM))

        # Plot the classifying line.
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)

    @staticmethod
    def plot_2d(d, ell, b_star=None, focus=None, scale=1, show=False, dynamic=False, pause=0.01):
        """ Produce a plot given the data-set "d". If defined, draw the decision boundary curve 
        from "b_star" and adjust the axis according to "scale". If desired, the call to 
        "plt.show()" and "plt.ion" can be omitted.
        
        :param d: Data-set containing matrix of size n x 2.
        :param ell: Labels of same length as 'd', who correspond to the points in d.
        :param b_star: Decision boundary as a Shapely ojbect.
        :param focus: Point to focus. Represents the current point selected.
        :param scale: The axis to display.
        :param show: If false, do not call "plt.show()".
        :param dynamic: If false, do not call "plt.ion()".
        :param pause: Number of seconds to pause plot for.
        :param flash_b: Used to indicate that we have found a solution. Flashes "b" curve.
        :return: None.
        """
        assert len(d[0]) == 2 and len(d) == len(ell)

        # Enable interactive mode if desired. Clear the previous plot.
        dynamic and plt.cla() and plt.ion()

        # Plot our data-set. Colored red and blue.
        [plt.scatter(d[i][0], d[i][1], c=("b" if ell[i] == -1 else "r")) for i in range(0, len(d))]
        plt.axis([-scale, scale, -scale, scale])

        # Circle the focus point if defined.
        if focus is not None:
            plt.scatter(focus[0], focus[1], s=80, facecolors='none', edgecolors='m')

        # If defined, plot the decision boundary.
        if b_star is not None:
            db_x, db_y = Visualize.__parse_line(b_star, scale)
            plt.plot(db_x, db_y, "--", c="k")

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)

        return 0
