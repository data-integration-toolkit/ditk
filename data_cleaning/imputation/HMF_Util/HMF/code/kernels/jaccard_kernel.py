"""
Classes for constructing Jaccard kernels for binary features. Methods for 
loading features, constructing the kernel, storing it, and loading it.

The features should be in a tab-delimited file. We can store the kernel as a 
tab-delimited file.

Usage:
    kernel = JaccardKernel()
    kernel.load_features("filename.csv")
    kernel.construct_kernel()
    kernel.store_kernel("kernel.txt")
Or:
    kernel = JaccardKernel()
    kernel.load_kernel("kernel.txt")

Can then use the kernel:
    kernel.kernel[i,j]
"""

from kernel import Kernel
import numpy

class JaccardKernel(Kernel):
    """ Load features, construct and store similarity kernel """
    def load_features_construct_store_kernel(self,location_features,location_output,delimiter="\t"):
        self.load_features(location_features=location_features,delimiter=delimiter)
        self.construct_kernel()
        self.store_kernel(location_output=location_output)
        
    """ Construct similarity kernel """
    def construct_kernel(self):
        print "Constructing kernel."
        
        # Jaccard coefficient
        def jaccard(a1,a2):
            if any(numpy.isnan(a1)) or any(numpy.isnan(a2)):
                return numpy.NaN
            a1, a2 = numpy.array(a1,dtype=int), numpy.array(a2,dtype=int)
            ands = sum(numpy.array(a1) & numpy.array(a2))
            ors = sum(numpy.array(a1) | numpy.array(a2))
            return 1 if ors == 0 else ands / float(ors)
            
        # Check whether values are in either 0 or 1, and convert values to ints
        assert all([True if val == 1. or val == 0. or numpy.isnan(val) else False for val in self.features.flatten()]), "Values are not binary - 1 or 0!"
        
        self.kernel = numpy.zeros((self.no_points,self.no_points))
        for i in range(0,self.no_points):
            for j in range(i,self.no_points):
                features1,features2 = self.features[i],self.features[j]
                similarity = jaccard(features1,features2)
                self.kernel[i,j] = similarity
                self.kernel[j,i] = similarity
        assert numpy.array_equal(self.kernel[~numpy.isnan(self.kernel)],
                                 self.kernel.T[~numpy.isnan(self.kernel.T)]), "Kernel not symmetrical!"
        assert numpy.nanmin(self.kernel) >= 0.0 and numpy.nanmax(self.kernel) <= 1.0, "Kernel values are outside [0,1]!"