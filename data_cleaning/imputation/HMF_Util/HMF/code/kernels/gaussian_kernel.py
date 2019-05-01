"""
Classes for constructing Gaussian kernels for real-values features. Methods for 
loading features, constructing the kernel, storing it, and loading it.

The features should be in a tab-delimited file. We can store the kernel as a 
tab-delimited file.

Usage: (we now need to specify the value for sigma^2 - tune to give a nice similarity distribution)
    kernel = JaccardKernel()
    kernel.load_features("filename.csv")
    kernel.construct_kernel(sigma_2=1)
    kernel.store_kernel("kernel.txt")
Or:
    kernel = JaccardKernel()
    kernel.load_kernel("kernel.txt")

Can then use the kernel:
    kernel.kernel[i,j]
"""

from kernel import Kernel
import numpy, math

class GaussianKernel(Kernel):
    """ Load features, construct and store similarity kernel """
    def load_features_construct_store_kernel(self,location_features,location_output,sigma_2,delimiter="\t"):
        self.load_features(location_features=location_features,delimiter=delimiter)
        self.construct_kernel(sigma_2=sigma_2)
        self.store_kernel(location_output=location_output)
        
    """ Construct similarity kernel """
    def construct_kernel(self,sigma_2):
        print "Constructing kernel."
        
        # Gaussian kernel
        def gaussian(a1,a2,sigma_2):
            #distance = sum([(x1-x2)**2 for x1,x2 in zip(a1,a2)])
            distance = numpy.power(a1-a2, 2).sum()
            return math.exp( -distance / (2.*sigma_2) )

        # First row and column are names, then similarities
        self.kernel = numpy.zeros((self.no_points,self.no_points))
        for i in range(0,self.no_points):
            print "Row %s/%s." % (i+1,self.no_points)
            for j in range(i,self.no_points):
                features1,features2 = self.features[i],self.features[j]
                similarity = gaussian(features1,features2,sigma_2)
                self.kernel[i,j] = similarity
                self.kernel[j,i] = similarity
        assert numpy.array_equal(self.kernel[~numpy.isnan(self.kernel)],
                                 self.kernel.T[~numpy.isnan(self.kernel.T)]), "Kernel not symmetrical!"
        assert numpy.nanmin(self.kernel) >= 0.0 and numpy.nanmax(self.kernel) <= 1.0, "Kernel values are outside [0,1]!"