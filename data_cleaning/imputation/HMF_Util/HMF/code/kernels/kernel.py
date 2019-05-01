"""
General class for similarity kernels - so for Gaussian and Jaccard.

Child classes need to define a method construct_kernel() that uses self.features
and constructs a numpy array of similarity values (no_points by no_points).
"""

import numpy, pandas

class Kernel:
    def __init__(self):
        return
        
    """ Load the features from a csv file, sort by row name """
    def load_features(self,location_features,delimiter="\t",names=None):
        self.features = numpy.loadtxt(location_features,delimiter=delimiter)
        self.no_points,_ = self.features.shape
        
    """ Store the similarity kernel - first row is '# name1\tname2\t...', then kernel. """
    def store_kernel(self,location_output):
        numpy.savetxt(location_output, self.kernel, delimiter='\t')     
        
    """ Load the kernel from the above specified format """
    def load_kernel(self,location_input,delimiter="\t"):
        self.kernel = numpy.loadtxt(location_input,delimiter=delimiter)  
        self.no_points,_ = self.kernel.shape