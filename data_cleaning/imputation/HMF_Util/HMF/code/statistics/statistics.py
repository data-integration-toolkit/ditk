"""
Methods for computing performance statistics, like the MSE, RMSE, NRMSE, 
I-divergence.
"""

import numpy, math

def MSE_matrix(R,R_pred,M):
    ''' Compute the MSE of the real vs predicted values, for all 1 entries in M. '''
    (R,R_pred,M) = (numpy.array(R),numpy.array(R_pred),numpy.array(M))
    assert M.sum() > 0, "M has no 1 entries to calculate the MSE of!"
    return (M * (R-R_pred)**2).sum() / float(M.sum())
    
def i_div_matrix(R,R_pred,M):
    ''' Compute the I-divergence of the real vs predicted values, for all 1 entries in M. '''
    (R,R_pred,M) = (numpy.array(R),numpy.array(R_pred),numpy.array(M))
    assert M.sum() > 0, "M has no 1 entries to calculate the I-divergence of!"
    return (M*(R*numpy.log(R/R_pred)-R+R_pred)).sum()
    
def R2_matrix(R,R_pred,M):
    ''' Coefficient of determination, for all 1 entries in M. '''
    (R,R_pred,M) = (numpy.array(R),numpy.array(R_pred),numpy.array(M))
    assert M.sum() > 0, "M has no 1 entries to calculate the R^2 of!"
    mean = (M*R).sum() / float(M.sum())
    SS_total = float((M*(R-mean)**2).sum())
    SS_res = float((M*(R-R_pred)**2).sum())        
    return 1. - SS_res / SS_total
    
def Rp_matrix(R,R_pred,M):
    ''' Pearson correlation coefficient, for all 1 entries in M. '''
    (R,R_pred,M) = (numpy.array(R),numpy.array(R_pred),numpy.array(M))
    assert M.sum() > 0, "M has no 1 entries to calculate the Rp of!"
    mean_real = (M*R).sum() / float(M.sum())
    mean_pred = (M*R_pred).sum() / float(M.sum())
    covariance = (M*(R-mean_real)*(R_pred-mean_pred)).sum()
    variance_real = (M*(R-mean_real)**2).sum()
    variance_pred = (M*(R_pred-mean_pred)**2).sum()
    return covariance / float(math.sqrt(variance_real)*math.sqrt(variance_pred)) \
        if math.sqrt(variance_real)*math.sqrt(variance_pred) != 0.0 else 0.0
            
def all_statistics_matrix(R,R_pred,M):
    ''' Return tuple (MSE,R2,Rp), for all 1 entries in M. '''
    return (MSE_matrix(R,R_pred,M), R2_matrix(R,R_pred,M), Rp_matrix(R,R_pred,M))
        
        
''' Same methods but if we have a list of predictions. '''
def MSE_list(R,R_pred):
    (R,R_pred) = (numpy.array(R),numpy.array(R_pred))
    return ((R-R_pred)**2).sum()/float(len(R))
    
def R2_list(R,R_pred):
    (R,R_pred) = (numpy.array(R),numpy.array(R_pred))
    mean = R.sum()/float(len(R))
    return 1. - ((R_pred-R)**2).sum() / ((R-mean)**2).sum()

def Rp_list(R,R_pred):
    (R,R_pred) = (numpy.array(R),numpy.array(R_pred))
    mean_real = R.sum()/float(len(R))
    mean_pred = R_pred.sum()/float(len(R_pred))
    covariance = ((R-mean_real)*(R_pred-mean_pred)).sum()
    variance_real = ((R-mean_real)**2).sum()
    variance_pred = ((R_pred-mean_pred)**2).sum()
    return covariance / (math.sqrt(variance_real)*math.sqrt(variance_pred)) \
        if math.sqrt(variance_real)*math.sqrt(variance_pred) != 0.0 else 0.0
        
def all_statistics_list(R,R_pred):
    ''' Return tuple (MSE,R2,Rp), for all 1 entries in M. '''
    return (MSE_list(R,R_pred), R2_list(R,R_pred), Rp_list(R,R_pred))
        