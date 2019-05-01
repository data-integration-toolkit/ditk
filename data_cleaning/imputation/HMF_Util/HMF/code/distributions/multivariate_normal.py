"""
Class representing a multivariate normal distribution, allowing us to sample from it.
"""
from numpy.random import multivariate_normal
import numpy

def MN_draw(mu,precision):
    sigma = numpy.linalg.inv(precision)
    return multivariate_normal(mean=mu,cov=sigma,size=None)