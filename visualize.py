""""
Filename:    visualize.py

Description: This file contains a set of methods (wrapped in a Visualize class) to display 
             training data and decision boundary curves. 
"""""

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from matplotlib import pyplot as plt

class Visualize(object):
    """ Methods: plot_2d = Produce a plot of a decision boundary and points.
    
    """

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
    def plot_2d(d, ell, b, b_star=None, focus=None, scale=1, show=False, dynamic=False, pause=0.01,
                flash_b=0):
        """ Produce a plot given the data-set "d" and the current classifier "b". If defined, 
        draw the decision boundary curve from "b_star" and adjust the axis according to "scale".
        If desired, the call to "plt.show()" and "plt.ion" can be omitted.
        
        :param d: Data-set containing list of Point instances.
        :param ell: Labels of same length as 'd', who correspond to the points in d.
        :param b: Classifier curve as a Shapely object.
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
        [plt.scatter(d[i][0], d[i][1], c=("r" if ell[i] == -1 else "b")) for i in range(0, len(d))]
        plt.axis([-scale, scale, -scale, scale])

        # Pull our classifying line.
        b_x, b_y = Visualize.__parse_line(b, scale)
        plt.plot(b_x, b_y, c="m")

        # Dynamic mode must be toggled off to call show.
        show and not dynamic and plt.show()

        # Dynamic mode must be on, show must be off to pause.
        not show and dynamic and plt.pause(pause)

        return 0
