"""
This file contains methods for drawing new values for the variables in the 
Gibbs sampling algorithm.

For F, G and S we return an entire new matrix.
For lambda (ARD) we return a new vector.
For the tau (noise) parameters we return a single new value.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import updates_Gibbs as updates
from HMF.code.distributions.gamma import gamma_draw
from HMF.code.distributions.normal import normal_draw
from HMF.code.distributions.multivariate_normal import MN_draw
from HMF.code.distributions.truncated_normal import TN_draw
from HMF.code.distributions.truncated_normal_vector import TN_vector_draw
from HMF.code.distributions.multivariate_truncated_normal import MTN_draw

import itertools
import numpy

###############################################################################
###################### Helpers for dimensionality checks ######################
###############################################################################

def check_dimensions_RCD(R,C,D):
    ''' Check dimensions of passed datasets (R,C,D) - number of rows should be the same. Return the F matrix. '''
    I, F = None, None
    
    for n in range(0,len(R)):
        (dataset,mask,F_n,S,G,tau,alpha) = R[n]
        (I_dataset,J_dataset), (I_mask,J_mask) = dataset.shape, mask.shape
        assert (I is None) or (I_dataset == I and I_mask == I and J_dataset == J_mask), \
            "R[%s] or M[%s] has wrong dimensions: %s and %s; should be %s rows." % (n,n,dataset.shape,mask.shape,I)
        assert (F is None) or (numpy.array_equal(F,F_n)), "F_%s is not equal to the other F's!" % n
        I, F = F_n.shape[0], F_n
    
    for m in range(0,len(C)):
        (dataset,mask,F_m,S,tau,alpha) = C[m]
        (I_dataset,J_dataset), (I_mask,J_mask) = dataset.shape, mask.shape
        assert (I is None) or (I_dataset == I and I_mask == I and J_dataset == J_mask), \
            "C[%s] or M[%s] has wrong dimensions: %s and %s; should be %s rows." % (m,m,dataset.shape,mask.shape,I)
        assert J_dataset == I_dataset and I_mask == J_mask, \
            "Matrix C[%s] and M[%s] should be square; are %s and %s instead." % (m,m,(I_dataset,J_dataset),(I_mask,J_mask))
        assert (F is None) or (numpy.array_equal(F,F_m)), "F_%s is not equal to the other F's!" % m
        I, F = F_m.shape[0], F_m
    
    for l in range(0,len(D)):
        (dataset,mask,F_l,G,tau,alpha) = D[l]
        (I_dataset,J_dataset), (I_mask,J_mask) = dataset.shape, mask.shape
        assert (I is None) or (I_dataset == I and I_mask == I and J_dataset == J_mask), \
            "D[%s] or M[%s] has wrong dimensions: %s and %s; should be %s rows." % (l,l,dataset.shape,mask.shape,I)
        assert (F is None) or (numpy.array_equal(F,F_l)), "F_%s is not equal to the other F's!" % l
        I, F = F_l.shape[0], F_l
        
    return F
        

###############################################################################
################## Draw new values for tau (noise) parameter ##################
###############################################################################

def draw_tau(alphatau,betatau,importance,dataset,mask,F,G,S=None):
    ''' Draw new values for tau (noise) parameter. '''
    alpha, beta = updates.alpha_beta_tau(alphatau=alphatau,betatau=betatau,importance=importance,dataset=dataset,mask=mask,F=F,G=G,S=S)
    tau = gamma_draw(alpha,beta)
    return tau
        

###############################################################################
########## Draw new values for alpha (dataset importance) parameter ###########
###############################################################################

def draw_importance(alphaA,betaA,tau,dataset,mask,F,G,S=None):
    ''' Draw new values for alpha (dataset importance) parameter. '''
    alpha, beta = updates.alpha_beta_importance(alphaA=alphaA,betaA=betaA,tau=tau,dataset=dataset,mask=mask,F=F,G=G,S=S)
    importance = gamma_draw(alpha,beta)
    return importance


###############################################################################
################# Draw new values for lambda (ARD) parameters #################
###############################################################################

def draw_lambdat(alpha0,beta0,Fs,K):
    ''' Draw new values for lambda (ARD) parameters. '''
    new_lambdas = numpy.empty(K)
    for k in range(0,K):
        alpha, beta = updates.alpha_beta_lambdat(alpha0=alpha0,beta0=beta0,Fs=Fs,k=k)
        new_lambdas[k] = gamma_draw(alpha,beta)
    return new_lambdas


###############################################################################
################# Draw new values for lambda (ARD) parameters #################
###############################################################################

def draw_lambdaS(alphaS,betaS,S,nonnegative):
    ''' Draw new values for lambda (ARD) parameters. '''
    (K,L) = S.shape
    new_lambdaS = numpy.empty((K,L))
    alpha, beta = updates.alpha_beta_lambdaS(alphaS=alphaS,betaS=betaS,S=S,nonnegative=nonnegative)
    for k,l in itertools.product(range(0,K),range(0,L)):
        new_lambdaS[k,l] = gamma_draw(alpha[k,l],beta[k,l])
    return new_lambdaS


###############################################################################
###################### Draw new values for F or G matrix ######################
###############################################################################
          
def draw_F(R,C,D,lambdaF,nonnegative,rows):
    ''' Draw new values for F or G matrix, and update in place.
        First do some dimensionality checks. 
        Then if rows=True, draw per row, otherwise per column.'''
        
    assert not nonnegative or not rows, "Nonnegative and row draws not implemented yet!"        
        
    F = check_dimensions_RCD(R,C,D)
    I, K = F.shape
    assert lambdaF.shape == (K,)
    
    if rows:
        # All independent so do not need to update F in between
        for i in range(0,I):
            mu_Fi, precision_Fi = updates.row_mu_precision_F(R=R,C=C,D=D,lambdaF=lambdaF,i=i,nonnegative=nonnegative)
            new_Fi = MTN_draw(mu_Fi,precision_Fi) if nonnegative else MN_draw(mu_Fi,precision_Fi)
            F[i,:] = new_Fi
    else:
        # Dependent so need to update R, C, D between updates
        for k in range(0,K):
            mu_Fk, tau_Fk = updates.column_mu_tau_F(R=R,C=C,D=D,lambdaF=lambdaF,k=k,nonnegative=nonnegative)
            new_Fk = TN_vector_draw(mu_Fk,tau_Fk) if nonnegative else MN_draw(mu_Fk,numpy.diag(tau_Fk))
            F[:,k] = new_Fk
      
    return F
      
      
###############################################################################
######################## Draw new values for S matrix #########################
###############################################################################

def draw_S(dataset,mask,tau,alpha,F,S,G,lambdaS,nonnegative,rows,tensor_decomposition):
    ''' Draw new values for S matrix, and update in place.
        First do some dimensionality checks.
        Then if rows=True, draw per row, otherwise per individual element.
        If tensor_decomposition = True, only draw new values for the diagonal. '''
        
    assert not nonnegative or not rows, "Nonnegative and row draws not implemented yet!"        
        
    (K, L), (I, J) = S.shape, dataset.shape
    assert lambdaS.shape == (K,L) and F.shape == (I,K) and G.shape == (J,L) and dataset.shape == mask.shape
    
    for k in range(0,K):
        ''' If doing CP, don't do row-wise draws. '''
        if rows and not tensor_decomposition:
            mu_Sk, precision_Fi = updates.row_mu_precision_S(
                dataset=dataset,mask=mask,tau=tau,alpha=alpha,F=F,S=S,G=G,lambdaSk=lambdaS[k],k=k,nonnegative=nonnegative)
            new_Sk = MTN_draw(mu_Sk,precision_Fi) if nonnegative else MN_draw(mu_Sk,precision_Fi)
            S[k,:] = new_Sk
        else:
            for l in range(0,L):
                ''' If doing CP, only draw value for l = k (diagonal). '''
                if tensor_decomposition and l != k:
                    S[k,l] = 0.
                else:
                    mu_Skl, tau_Skl = updates.individual_mu_tau_S(
                        dataset=dataset,mask=mask,tau=tau,alpha=alpha,F=F,S=S,G=G,lambdaSkl=lambdaS[k,l],k=k,l=l,nonnegative=nonnegative)
                    new_Skl = TN_draw(mu_Skl,tau_Skl) if nonnegative else normal_draw(mu_Skl,tau_Skl)    
                    S[k,l] = new_Skl
        
    return S