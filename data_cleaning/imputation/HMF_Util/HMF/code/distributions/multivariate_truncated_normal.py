"""
Methods for drawing values from a multivariate Truncated Normal (MTN).

We use the methods from Wilhelm and Manjunath (2010), "tmvtnorm: A Package for 
the Truncated Multivariate Normal Distribution".
https://journal.r-project.org/archive/2010-1/RJournal_2010-1_Wilhelm+Manjunath.pdf

We use the R method rtmvnorm from rtmvnorm.R.
"""

import numpy, itertools

"""
# Load in the R method rtmvnorm
from rpy2.robjects import r
from rpy2.robjects.packages import importr

loc_mvtnorm = './mvtnorm/R/'
loc_tmvtnorm = './tmvtnorm/R/'
packages = [
    loc_mvtnorm+'mvt.R', # for pmvnorm
    loc_mvtnorm+'mvnorm.R',
    loc_tmvtnorm+'checkTmvArgs.R',
    loc_tmvtnorm+'rtmvnorm.R']
#r('Sys.setlocale('LC_ALL','C')')

for name in packages:
    r('source("%s")' % (name))
method_rtmvnorm = r('rtmvnorm')


def call_function_R(mean,sigma):
    ''' Convert the arguments to R strings, and return the query string to run rtmvnorm. '''
    str_n = str(1)
    
    str_mean = 'c(' + ','.join([str(m) for m in mean]) + ')'
    
    no_rows, no_cols = sigma.shape
    elements_sigma = [sigma[i,j] for i,j in itertools.product(range(0,no_rows),range(0,no_cols))]
    str_elements_sigma = 'c(' + ','.join([str(e) for e in elements_sigma]) + ')'
    str_sigma = 'matrix(%s,nrow=%s,ncol=%s,byrow=TRUE)' % \
        (str_elements_sigma,no_rows,no_cols)
    
    str_lower = 'rep(0,length=%s)' % len(mean)
    
    str_algorithm = "'gibbs'" # 'rejection', 'gibbs', 'gibbsR'
    
    string = 'rtmvnorm(n=%s,mean=%s,sigma=%s,lower=%s,algorithm=%s)' % \
        (str_n,str_mean,str_sigma,str_lower,str_algorithm)
    return string

def MTN_draw_R(mu,precision):
    sigma = numpy.linalg.inv(precision)
    string = call_function_R(mean=mu,sigma=sigma)
    result = r(string)
    print result
    return
"""

"""
from fortran_mtn import rtmvnormgibbs
print rtmvnormgibbs

def MTN_draw_Fortran(mu,precision):
    pass
"""

def MTN_draw(mu,precision):
    assert False, "Multivariate truncated normal draws not implemented yet!"

"""
mu = numpy.array([1,2,3,4,5])
precision = numpy.identity(5)
print MTN_draw_Fortran(mu,precision)
"""