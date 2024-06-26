'''Functions for drawing contours of Dirichlet distributions.'''

# Author: Thomas Boggs

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_AREA = 0.5 * 1 * 0.75**0.5
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

# For each corner of the triangle, the pair of other corners
_pairs = [_corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.
    Arguments:
        `xy`: A length-2 sequence containing the x and y value.
    '''
    coords = np.array([tri_area(xy, p) for p in _pairs]) / _AREA
    return np.clip(coords, tol, 1.0 - tol)

class Dirichlet(object):
    def __init__(self, alpha):
        '''Creates Dirichlet distribution with parameter `alpha`.'''
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     np.multiply.reduce([gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * np.multiply.reduce([xx ** (aa - 1)
                                                for (xx, aa)in zip(x, self._alpha)])
    def sample(self, N):
        '''Generates a random sample of size `N`.'''
        return np.random.dirichlet(self._alpha, N)

def draw_pdf_contours(dist, border=False, nlevels=200, subdiv=8, **kwargs):
    '''Draws pdf contours over an equilateral triangle (2-simplex).
    Arguments:
        `dist`: A distribution instance with a `pdf` method.
        `border` (bool): If True, the simplex border is drawn.
        `nlevels` (int): Number of contours to draw.
        `subdiv` (int): Number of recursive mesh subdivisions to create.
        kwargs: Keyword args passed on to `plt.triplot`.
    '''
    from matplotlib import ticker, cm
    import math

    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)
    # plt.axis('equal')
    # plt.xlim(0, 1)
    # plt.ylim(0, 0.75**0.5)
    # plt.axis('off')


    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 0.98)

    # plt.axis([-0.1, 1.1, -0.1, 1])

    if border is True:
        plt.triplot(_triangle, linewidth=1)

    # plt.xlim(-1.2, 1.2)
    # plt.ylim(-1.2, 1.2)
    # # plt.scatter(points.T[0], points.T[1], edgecolors = '#ffffff', alpha = 0.5)
    # # plt.text(p1[0]-0.1, p1[1]+0.1, 'P1', fontsize = 16)
    # # plt.text(p2[0]-0.1, p2[1]-0.15, 'P2', fontsize = 16)
    # # plt.text(p3[0], p3[1]-0.15, 'P3', fontsize = 16)
    plt.xticks([])
    plt.yticks([])
    # plt.show()

def plot_points(X, barycentric=True, border=True, **kwargs):
    '''Plots a set of points in the simplex.
    Arguments:
        `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                       (if in barycentric coords) of points to plot.
        `barycentric` (bool): Indicates if `X` is in barycentric coords.
        `border` (bool): If True, the simplex border is drawn.
        kwargs: Keyword args passed on to `plt.plot`.
    '''
    if barycentric is True:
        X = X.dot(_corners)
    # plt.plot(X[:, 0], X[:, 1], 'k.', color='steelblue', ms=1, **kwargs)
    plt.scatter(X[:, 0], X[:, 1], edgecolors = '#ffffff', alpha = 0.5)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.triplot(_triangle, linewidth=1)

if __name__ == '__main__':
    # plt.switch_backend('agg')
    plt.switch_backend('agg')

    # f = plt.figure(figsize=(8, 6))
    f = plt.figure(figsize=(12,3.6))
    alphas = [
        [1] * 3,
        [5] * 3,
        [10] * 3
        # [10] * 3
        # [2, 5, 15]
              ]
    for (i, alpha) in enumerate(alphas):
        # plt.subplot(2, len(alphas), i + 1)
        plt.subplot(1, len(alphas), i + 1)
        plt.subplots_adjust(wspace=0.1)

        dist = Dirichlet(alpha)
        draw_pdf_contours(dist)
        plt.title(f'Dirichlet({alpha[0]:.1f},{alpha[1]:.1f},{alpha[2]:.1f}) distribution')
        # plt.xlim(-1.2, 1.2)
        # plt.ylim(-1.2, 1)
        # # plt.scatter(points.T[0], points.T[1], edgecolors = '#ffffff', alpha = 0.5)
        # # plt.text(p1[0]-0.1, p1[1]+0.05, 'P1', fontsize = 16)
        # # plt.text(p2[0]-0.15, p2[1]-0.175, 'P2', fontsize = 16)
        # # plt.text(p3[0]-0.05, p3[1]-0.175, 'P3', fontsize = 16)
        # plt.xticks([])
        # plt.yticks([])

        # #标题
        # title = r'$\gamma$ = (%.2f, %.2f, %.2f)' % tuple(alpha)
        # # gamma_str = str(r'$\gamma$')
        # # title = f'Simplex of the Dirichlet({gamma_str}) with {title}'
        # plt.title(title, y=-0.15, fontdict={'fontsize': 10})           
                


        # plt.subplot(2, len(alphas), i + 1 + len(alphas))
        # # plot_points(dist.sample(5000))
        # plot_points(dist.sample(4000))
        # # title = r'$\gamma$ = (%.2f, %.2f, %.2f)' % tuple(alpha)
        # # title = f'Simplex of the Dirichlet({gamma_str}) with {title}'
        # plt.title(title, y=-0.15, fontdict={'fontsize': 10})           
             

    savepath = f'result/dirichlet'
    savename = f'dirichlet_simplex'
    plt.savefig(f'{savepath}/{savename}.png')        
    print('Wrote plots to "dirichlet_plots.png".')