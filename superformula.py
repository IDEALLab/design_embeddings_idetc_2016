"""
Creates superformula samples using two variables and fits them with B-splines.

Author(s): Jonah Chazan (jchazan@umd.edu)
"""

import numpy as np
from uniform_bspline import UniformBSpline
from fit_uniform_bspline import UniformBSplineLeastSquaresOptimiser
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import math

def fit_bspline(x, y, dim = 2, degree=2, num_control_points = 20,
                is_closed = False, num_init_points=1000):
    ''' Fits and returns a bspline curve to the given x and y points
    
        Parameters
        ----------
        x : list
            data x-coordinates
        y : list
            data y-coordinates
        dim : int
            the dimensionality of the dataset (default: 2)
        degree : int
            the degree of the b-spline polynomial (default: 2)
        num_control_points : int
            the number of b-spline control points (default: 20)
        is_closed : boolean
            should the b-spline be closed? (default: false)
        num_init_points : int
            number of initial points to use in the b-spline parameterization
            when starting the regression. (default: 1000)
        
        Returns
        -------
        c: a UniformBSpline object containing the optimized b-spline
    '''
    # TODO: extract dimensionality from the x,y dataset itself
    num_data_points = len(x)
    c = UniformBSpline(degree, num_control_points, dim, is_closed=is_closed)
    Y = np.c_[x, y] # Data num_points by dimension
    # Now we need weights for all of the data points
    w = np.empty((num_data_points, dim), dtype=float)
    # Currently, every point is equally important
    w.fill(1) # Uniform weight to the different points
    # Initialize `X` so that the uniform B-spline linearly interpolates between
    # the first and last noise-free data points.
    #t = np.linspace(0.0, 1.0, num_control_points)[:, np.newaxis]
    #X = Y[0] * (1 - t) + Y[-1] * t
    average_width = 0
    max_width = 0
    for i in range(len(Y)):
        wid = math.sqrt(Y[i][0] ** 2 + Y[i][1] ** 2)
        average_width += wid
        if wid > max_width:
            max_width = wid
    average_width /= len(Y)

    t = np.linspace(0.0, 2 * math.pi, num_control_points)
    X = max_width * np.c_[np.cos(t),np.sin(t)]
    # NOTE: Not entirely sure if the next three lines are necessary or not
    #m0, m1 = c.M(c.uniform_parameterisation(2), X)
    #x01 = 0.5 * (X[0] + X[-1])
    #X = (np.linalg.norm(Y[0] - Y[-1]) / np.linalg.norm(m1 - m0)) * (X - x01) + x01
    #print X
    # Regularization weight on the control point distance
    # This specifies a penalty on having the b-spline control points close
    # together, and in some sense prevents over-fitting. Change this is the
    # curve doesn't capture the curve variation well or smoothly enough
    lambda_ = 0.5 
    # These parameters affect the regression solver.
    # Presently, they are disabled below, but you can think about enabling them
    # if that would be useful for your use case.
    max_num_iterations = 1000
#    min_radius = 0
#    max_radius = 400
#    initial_radius = 100
    # Initialize U
    u0 = c.uniform_parameterisation(num_init_points)
    D = cdist(Y, c.M(u0, X))
    u = u0[D.argmin(axis=1)]
    # Run the solver
    (u, X, has_converged, states, num_iterations, 
        time_taken) = UniformBSplineLeastSquaresOptimiser(c,'lm').minimise(
        Y, w, lambda_, u, X,
        max_num_iterations = max_num_iterations,
        #min_radius = min_radius,
        #max_radius = max_radius,
        #initial_radius = initial_radius,
        return_all=True)

    #if not has_converged:
    #    print "Failed to converge!"
    return c,u0,X, has_converged

def superformula(a, b, m, n1, n2, n3, num_points=1000):
    phis = np.linspace(0, 2 * math.pi, num_points)

    def r(phi):
        # aux = abs(math.cos(m * phi / 4) / a) ** n2 + abs(math.sin(m * phi / 4) / b) ** n3
        aux = abs(math.cos(m * phi / 4)) ** n2 + abs(math.sin(m * phi / 4)) ** n3
        # force a, b to 1, use a, b to scale so we have a more linear example
        return aux ** (-1.0/n1)

    r = np.vectorize(r, otypes=[np.float])

    rs = r(phis)

    x = rs * np.cos(phis) * a
    y = rs * np.sin(phis) * b

    return (x, y)


def superformula_spline(a, b, m, n1, n2, n3, im_save_path, num_control_points, num_func_points=1000):

    x, y = superformula(a, b, m, n1, n2, n3, num_func_points)

    bspline,u0,x_plot, converged = fit_bspline(x, y, degree=2, num_control_points = num_control_points, is_closed = True)

    if im_save_path is not None and converged:
#        f = plt.figure(figsize=(10,10))
        ax1 = plt.gca()
        ax1.set_title('Image + Contour')
        ax1.plot( *zip(*bspline.M(u0, x_plot).tolist()),linewidth =2)#, c=u0, cmap="jet", alpha=0.5 )
        ax1.plot(*zip(*x_plot), marker="o", alpha=0.3)
        ax1.plot(x, y, linewidth=2, color="red")
        #ax1.set_title(imfile + ' n=' + str(num_control_points))
        #f.show()

        plt.savefig(im_save_path, dpi=300)
        plt.close()
    
    return x_plot, u0, bspline, converged
