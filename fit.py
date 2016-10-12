#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from scipy.optimize import minimize


class FitIt(object):
    """
    The main fitting class.

    :param data_x: ``numpy.ndarray``
        The points where the function is evaluated at.

    :param data_y: ``numpy.ndarray``
        The observed data. It must have the same shape as ``data_x``.

    :param guess: sequence
        Sequence containing the first guess for the parameters of the function.

    :params funct: callable
        The function that you wish to fit the data to. It must be designed so as
        to accept only two parameters: ``x`` and ``theta``, where ``x`` is a
        ``numpy.ndarray`` and ``theta`` is a sequence (with the same shape as
        ``guess``) containing the parameters to be fitted. It must return a
        ``numpy.ndarray`` with the values of the function sampled in the points
        of the ``x`` array.
    """
    def __init__(self, data_x, data_y, guess, funct):
        self.x = data_x
        self.y = data_y
        self.guess = guess
        self.funct = funct

    def chi_squared(self, theta):
        """
        Computes the chi-squared of the fit.

        :param theta: sequence
            The parameters to be passed to the model function.

        :return: ``float``
            The value of the chi-squared of the fit.
        """
        model = self.funct(self.x, theta)
        diff_sq = (self.y - model) ** 2
        return np.sum(diff_sq)

    def do_it(self, maxiter=200, display=False):
        """
        Do it (the fit). No fancy posteriors or co-variance matrix and shit.
        This method simply returns the best fit parameters and that is what you
        get.

        :param maxiter: ``int``
            Maximum number of iterations. Default is 200.

        :param display: ``bool``
            Switch to display the ``scipy.optimize.minimize`` print statements.
            Default is ``False``.

        :return: list
            The best fit parameters.
        """
        result = minimize(fun=self.chi_squared, method='nelder-mead',
                          options={'maxiter': maxiter, "disp": display})
        if display is True:
            print('Number of iterations performed = %i' % result['nit'])
            print('Minimization successful = %s' % repr(result['success']))
            print('Cause of termination = %s' % result['message'])

        params = result["x"]
        return params
