'''
Methods for initialising the F, S, G, U, V, lambda parameters of the Gibbs 
sampling algorithms.

We initialise parameters using one of the following:
- expectation   - use the expectation values of the model definition, for the 
                  given hyperparameter values
- random        - draw values randomly from the model definition, using the 
                  given hyperparameter values
- kmeans        - run K-means on the main dataset and use the cluster 
                  assignments, adding +0.2 for smoothing
- least squares - solve least squares on the main dataset 

Matrix factorisation:
- U                -> expectation, random, kmeans
- V                -> expectation, random, least squares
- lambdak          -> expectation, random
- tau              -> expectation, random

Matrix tri-factorisation:
- F, G             -> expectation, random, kmeans
- S                -> expectation, random, least squares
- lambdak, lambdal -> expectation, random
- tau              -> expectation, random

The 'prior' parameter has to be either 'normal' or 'exponential'.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import updates_Gibbs
from HMF.code.kmeans.kmeans import KMeans
from HMF.code.distributions.exponential import exponential_draw
from HMF.code.distributions.gamma import gamma_draw
from HMF.code.distributions.normal import normal_draw
#from HMF.code.distributions.multivariate_normal import MN_draw
#from HMF.code.distributions.truncated_normal import TN_draw
#from HMF.code.distributions.truncated_normal_vector import TN_vector_draw
#from HMF.code.distributions.multivariate_truncated_normal import MTN_draw

import numpy, itertools

###############################################################################
############################# Initialisation tau ##############################
###############################################################################

def init_tau(init,alphatau,betatau,importance,R,M,F,G,S=None):
    ''' Initialise the tau parameter using the model definition. Init in ['random','exp']. '''
    options = ['random','exp']
    assert init in options, "Unknown initialisation option for tau: %s. Options are %s." % (init,options)
    
    alpha, beta = updates_Gibbs.alpha_beta_tau(
        alphatau=alphatau,betatau=betatau,importance=importance,dataset=R,mask=M,F=F,G=G,S=S)
        
    return gamma_draw(alpha,beta) if init == 'random' else alpha / float(beta)
        
###############################################################################
########################### Initialisation lambdak ############################
###############################################################################

def init_lambdak(init,K,alpha0,beta0):
    ''' Initialise the lambdak parameters using the model definition. Init in ['random','exp']. '''
    options = ['random','exp']
    assert init in options, "Unknown initialisation option for lambdak: %s. Options are %s." % (init,options)
    
    lambdak = numpy.zeros(K)
    for k in range(0,K):
        lambdak[k] = gamma_draw(alpha0,beta0) if init == 'random' else alpha0 / float(beta0)
    return lambdak
        
###############################################################################
########################### Initialisation lambdak ############################
###############################################################################

def init_lambdaS(init,K,L,alphaS,betaS):
    ''' Initialise the lambda^n_kl or lambda^m_kl parameters using the model definition. Init in ['random','exp']. '''
    options = ['random','exp']
    assert init in options, "Unknown initialisation option for element-wise sparsity lambda^S: %s. Options are %s." % (init,options)
    
    lambdaS = numpy.zeros((K,L))
    for k,l in itertools.product(range(0,K),range(0,L)):
        lambdaS[k,l] = gamma_draw(alphaS,betaS) if init == 'random' else alphaS / float(betaS)
    return lambdaS
        
###############################################################################
########################### Initialisation U and V ############################
###############################################################################
        
def init_matrix_random_exp(prior,init,I,K,lambdak):
    ''' Return a matrix initialised randomly from the prior or using expectation. '''
    options_init = ['random','exp']
    options_prior = ['normal','exponential']
    assert init in options_init, "Unknown initialisation option for matrix: %s. Options are %s." % (init,options_init)
    assert prior in options_prior, "Unknown prior option for matrix: %s. Options are %s." % (init,options_prior)
    assert len(lambdak) == K, 'lambdak is of length %s, rather than K=%s!' % (len(lambdak),K)
    
    matrix = numpy.zeros((I,K))
    for i,k in itertools.product(range(0,I),range(0,K)):
        if init == 'random':
            matrix[i,k] = normal_draw(mu=0,tau=lambdak[k]) if prior == 'normal' else exponential_draw(lambdax=lambdak[k])
        elif init == 'exp':
            # Set it to a small value, to avoid division by 0 in the model
            matrix[i,k] = 0.01 if prior == 'normal' else 1. / lambdak[k]
    return matrix

def init_matrix_kmeans(R,M,K):
    ''' Return a matrix initialised by running K-means on R, with K clusters. '''
    assert R is not None and M is not None, "Want to do K-means init but R or M is None."
    
    print "Initialising matrix using KMeans."
    kmeans_matrix = KMeans(X=R,M=M,K=K)
    kmeans_matrix.initialise()
    kmeans_matrix.cluster()
    return kmeans_matrix.clustering_results + 0.2   

def impute_missing_average(R,M):
    ''' Impute missing values in R indicated by M, using (row_avr+column_avr)/2 of observed values. '''
    I,J = R.shape
    indices_missing = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if M[i,j] == 0.]
    
    row_averages = [(M[i]*R[i]).sum() / float(M[i].sum()) for i in range(0,I)]
    column_averages = [(M[:,j]*R[:,j]).sum() / float(M[:,j].sum()) for j in range(0,J)]
    
    R_imp = numpy.copy(R)
    for i,j in indices_missing:
        R_imp[i,j] = ( float(row_averages[i])    if not numpy.isnan(float(row_averages[i]))    else 0.
                     + float(column_averages[j]) if not numpy.isnan(float(column_averages[j])) else 0. 
                     ) / 2.
    return R_imp
    
def init_V_leastsquares(R,M,U):
    ''' Return a matrix initialised by least squares. 
        We do this by solving R = UV.T -> V = (U_inv R).T (using pseudo-inverse).
        We impute the missing values in R using the row averages.'''
    assert R is not None and M is not None and U is not None, "Want to do least squares init but R, M, or U is None."
    
    print "Initialising V using least squares."
    
    R_imp = impute_missing_average(R,M)
    #(V,residuals,rank,singular_vals) = numpy.linalg.lstsq(a=U,b=R_imp) #returns x s.t. ||b-ax||^2 is minimised 
    U_pinv = numpy.linalg.pinv(U)
    V = numpy.dot(U_pinv,R_imp).T
    return V

def init_U(prior,init,I,K,lambdak,R=None,M=None):
    ''' Initialise U. Prior in ['normal','exponential']. Init in ['random','exp','kmeans']. '''
    U = init_matrix_kmeans(R=R,M=M,K=K) if init == 'kmeans' \
        else init_matrix_random_exp(prior=prior,init=init,I=I,K=K,lambdak=lambdak)
    return U

def init_V(prior,init,I,K,lambdak,R=None,M=None,U=None):
    ''' Initialise U. Prior in ['normal','exponential']. Init in ['random','exp','least']. '''
    
    V = init_V_leastsquares(R=R,M=M,U=U) if init == 'least' \
        else init_matrix_random_exp(prior=prior,init=init,I=I,K=K,lambdak=lambdak)
          
    ''' If V is nonnegative, set all negative elements to ~0 (bit more for smoothing). '''
    if prior == 'exponential':
        V = V.clip(min=0.01)    
      
    return V

###############################################################################
########################### Initialisation F, S, G ############################
###############################################################################

def init_FG(prior,init,I,K,lambdak,R=None,M=None):
    ''' Initialise F or G. Prior in ['normal','exponential']. Init in ['random','exp','kmeans']. '''
    return init_U(prior=prior,init=init,I=I,K=K,lambdak=lambdak,R=R,M=M)

def init_S_leastsquares(R,M,F,G):
    ''' Return a matrix initialised by least squares. 
        We do this by solving R = FSG.T -> S = F_inv R (G.T)_inv (using pseudo-inverse).
        We impute the missing values in R using the row averages.'''
    assert R is not None and M is not None and F is not None and G is not None, "Want to do least squares init but R, M, F, or G is None."
    
    print "Initialising S using least squares."
    
    R_imp = impute_missing_average(R,M)
    F_pinv = numpy.linalg.pinv(F)
    G_T_pinv = numpy.linalg.pinv(G.T)
        
    S = numpy.dot(F_pinv,numpy.dot(R_imp,G_T_pinv))
    return S
    
def init_S(prior,init,K,L,lambdaS,R=None,M=None,F=None,G=None,tensor_decomposition=False):
    ''' Initialise S. Prior in ['normal','exponential']. Init in ['random','exp','least']. '''
    assert lambdaS.shape == (K,L), "lambdaS should be shape %s, rather than %s." % ((K,L),lambdaS.shape)  
    
    lambdaS0 = lambdaS[0,:]
    S = init_S_leastsquares(R=R,M=M,F=F,G=G) if init == 'least' \
        else init_matrix_random_exp(prior=prior,init=init,I=K,K=L,lambdak=lambdaS0)
        
    ''' If S is nonnegative, set all negative elements to a very small value. '''
    if prior == 'exponential':
        S = S.clip(min=0.01)    

    ''' If doing tensor decomposition (CP), set off-diagonal elements to 0. '''
    if tensor_decomposition:
        S[~(numpy.eye(K,L,dtype=bool))] = 0.
        
    return S