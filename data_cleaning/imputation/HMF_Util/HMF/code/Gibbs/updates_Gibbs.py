"""
This file contains the updates for the matrix factorisation models, for Gibbs
sampling.

We make them as general as possible, so they can be used for both single and 
multiple matrix factorisation and tri-factorisation models.

We can provide the parameters for row-wise draws (multivariate Gaussian or 
Truncated Normal), or column-wise draws (individual elements, but each column 
in parallel).

We use the following arguments for the updates for F, S:
    R - a list of main datasets and importances [Rn,Mn,Ftn,Sn,Fun,taun,alphan] for matrix tri-factorisation F(tn) S(n) F(un).T
    C - a list of constraint matrices and importances [Cm,Mm,Ftm,Sm,taum,alpham] for matrix tri-factorisation F(tm) S(m) F(tm).T
    D - a list of feature datasets and importances [Dl,Ml,Ftl,Gl,taul,alphal] for matrix factorisation F(tl) G(l).T
    lambdaF - a list of the hyperparameter or ARD lambda values (length K)
    i, k, l - the row or column number of F or S we are computing the parameters for
    nonnegative - True if we should use the nonnegative updates, False otherwise
    
We always use the rows of the Rn, Cm, Dl - if we want to compute the column 
factors, take the transpose of the matrix and its mask and pass that instead.
    
We use the following arguments for the updates for the lambdas (ARD):
    alpha0, beta0 - hyperparameters
    Fs - a list [Ft,nonneg] of all the factor matrices (F's, G's) that this ARD controls
Note that nonneg is True if we should use the nonnegative updates for Ft, False otherwise.
    
We use the following arguments for the updates for the tau (noise):
    alpha_tau, beta_tau - hyperparameters
    dataset, mask - the Rn, Cm, or Dl matrix
    F, G - factor matrices
    S - if MTF, None if MF
Note that nonneg is True if we should use the nonnegative updates for Ft, False otherwise.
    
Usage for single dataset matrix factorisation:
    U: column_tau_F([],[],[(R,M,U,V,tau,1)]),     column_mu_F(TODO)
    V: column_tau_F([],[],[(R.T,M.T,V,U,tau,1)]), column_mu_F(TODO)
Usage for single dataset matrix tri-factorisation:
    F: column_tau_F([(R,M,F,S,G,tau,1)],[],[]),         column_mu_F(TODO)
    G: column_tau_F([(R.T,M.T,G,S.T,F,tau,1)],[],[]),   column_mu_F(TODO)
"""

import numpy, math

###############################################################################
################################### Helpers ###################################
###############################################################################

def triple_dot(M1,M2,M3):
    ''' Do M1*M2*M3. If the matrices have dimensions I,K,L,J, then the complexity
        of M1*(M2*M3) is ~IJK, and (M1*M2)*M3 is ~IJL. So if K < L, we use the former. '''
    K,L = M2.shape
    if K < L:
        return numpy.dot(M1,numpy.dot(M2,M3))
    else:
        return numpy.dot(numpy.dot(M1,M2),M3)


###############################################################################
###### Updates for the alpha and beta parameters of the noise parameters ######
###############################################################################

def alpha_tau(alphatau,importance,mask):
    ''' Return the value for alpha for the Gibbs sampler, for the noise tau. '''
    return alphatau + importance * mask.sum() / 2.
          
def beta_tau(betatau,importance,dataset,mask,F,G,S=None):
    ''' Return the value for beta for the Gibbs sampler, for the noise tau. '''
    dataset_pred = numpy.dot(F,G.T) if S is None else triple_dot(F,S,G.T)
    squared_error = (mask*(dataset-dataset_pred)**2).sum()
    return betatau + importance * squared_error / 2.
        
        
###############################################################################
### Updates for the parameters of the posterior of importances alpha^n,m,l ####
############################################################################### 
        
def alpha_importance(alphaA):
    ''' Return the values for alpha for the Gibbs sampler, for the importance learning of alpha. '''
    return alphaA
          
def beta_importance(betaA,tau,dataset,mask,F,G,S=None):
    ''' Return the values for beta for the Gibbs sampler, for the importance learning of alpha. '''
    dataset_pred = numpy.dot(F,G.T) if S is None else triple_dot(F,S,G.T)
    squared_error = (mask*(dataset-dataset_pred)**2).sum()
    size_Omega = mask.sum()
    return betaA + tau * squared_error / 2. - size_Omega / 2. * math.log(tau / (2.*math.pi))
    
        
###############################################################################
########## Updates for the parameters of the posterior of lambda_t ############
############################################################################### 
        
def alpha_lambdat(alpha0,Fs):
    ''' Return the value for alpha for the Gibbs sampler, for the ARD. '''
    alpha = alpha0
    for F,nonneg in Fs:
        I,_ = F.shape
        alpha += I if nonneg else I / 2.
    return alpha
          
def beta_lambdat(beta0,Fs,k):
    ''' Return the value for beta for the Gibbs sampler, for the kth ARD. '''
    beta = beta0
    for F,nonneg in Fs:
        beta += F[:,k].sum() if nonneg else (F[:,k]**2).sum() / 2.
    return beta
        
        
###############################################################################
#### Updates for the parameters of the posterior of lambda^n and lambda^m #####
############################################################################### 
        
def alpha_lambdaS(alphaS,S,nonnegative):
    ''' Return the values for alpha for the Gibbs sampler, for the element-wise sparsity on S^n, S^m. '''
    return alphaS + numpy.ones(S.shape) * ( 1. if nonnegative else .5 )
          
def beta_lambdaS(betaS,S,nonnegative):
    ''' Return the value for beta for the Gibbs sampler, for the element-wise sparsity on S^n, S^m. '''
    return betaS + ( S if nonnegative else S**2 / 2. )
        
        
###############################################################################
####### Column-wise updates for the tau parameter of the posterior of F #######
###############################################################################

def column_tau_F(R,C,D,lambdaF,k,nonnegative):
    ''' Return the value for tau for the Gibbs posterior, for column-wise draws. '''
    tau_F = 0. if nonnegative else lambdaF[k]
    
    for Rn,Mn,Ftn,Sn,Fun,taun,alphan in R:
        tau_F += column_tau_individual_mtf(Rn,Mn,Ftn,Sn,Fun,taun,alphan,k)
    for Cm,Mm,Ftm,Sm,taum,alpham in C:
        tau_F += column_tau_individual_mtf(Cm,Mm,Ftm,Sm,Ftm,taum,alpham,k)
        tau_F += column_tau_individual_mtf(Cm.T,Mm.T,Ftm,Sm.T,Ftm,taum,alpham,k)
    for Dl,Ml,Ftl,Gl,taul,alphal in D:
        tau_F += column_tau_individual_mf(Dl,Ml,Ftl,Gl,taul,alphal,k)
        
    return tau_F

def column_tau_individual_mf(dataset,mask,F,G,tau,alpha,k):
    ''' Return the component of the tau update for an individual matrix, for matrix factorisation. '''
    return tau * alpha * ( mask * G[:,k]**2 ).sum(axis=1)

def column_tau_individual_mtf(dataset,mask,F,S,G,tau,alpha,k):
    ''' Return the component of the tau update for an individual matrix, for matrix tri-factorisation. '''
    return tau * alpha * ( mask * numpy.dot(S[k,:],G.T)**2 ).sum(axis=1)


###############################################################################
####### Column-wise updates for the mu parameter of the posterior of F ########
###############################################################################

def column_mu_F(R,C,D,lambdaF,tau_Fk,k,nonnegative):
    ''' Return the value for mu for the Gibbs posterior, for column-wise draws. '''
    mu_F = -lambdaF[k] if nonnegative else 0.
    
    for Rn,Mn,Ftn,Sn,Fun,taun,alphan in R:
        mu_F += column_mu_individual_mtf(Rn,Mn,Ftn,Sn,Fun,taun,alphan,k)
    for Cm,Mm,Ftm,Sm,taum,alpham in C:
        mu_F += column_mu_individual_mtf(Cm,Mm,Ftm,Sm,Ftm,taum,alpham,k)
        mu_F += column_mu_individual_mtf(Cm.T,Mm.T,Ftm,Sm.T,Ftm,taum,alpham,k)
    for Dl,Ml,Ftl,Gl,taul,alphal in D:
        mu_F += column_mu_individual_mf(Dl,Ml,Ftl,Gl,taul,alphal,k)
        
    mu_F /= tau_Fk
    return mu_F
    
def column_mu_individual_mf(dataset,mask,F,G,tau,alpha,k):
    ''' Return the component of the mu update for an individual matrix, for matrix factorisation. '''
    return tau * alpha * ( mask * ( ( dataset - numpy.dot(F,G.T) + numpy.outer(F[:,k],G[:,k])) * G[:,k] ) ).sum(axis=1)

def column_mu_individual_mtf(dataset,mask,F,S,G,tau,alpha,k):
    ''' Return the component of the mu update for an individual matrix, for matrix tri-factorisation. '''
    return tau * alpha * ( mask * ( ( dataset - triple_dot(F,S,G.T) + numpy.outer(F[:,k],numpy.dot(S[k,:],G.T)) ) * numpy.dot(S[k,:],G.T) ) ).sum(axis=1)


###############################################################################
##### Row-wise updates for the Precision parameter of the posterior of F ######
############################################################################### 
 
def row_precision_F(R,C,D,lambdaF,i,nonnegative):
    ''' Return the value for tau for the Gibbs posterior, for row-wise draws. '''
    precision_F = numpy.zeros((len(lambdaF),len(lambdaF))) if nonnegative else numpy.diag(lambdaF)
    
    for Rn,Mn,Ftn,Sn,Fun,taun,alphan in R:
        precision_F += row_precision_F_individual_mtf(Rn,Mn,Ftn,Sn,Fun,taun,alphan,i)
    for Cm,Mm,Ftm,Sm,taum,alpham in C:
        precision_F += row_precision_F_individual_mtf(Cm,Mm,Ftm,Sm,Ftm,taum,alpham,i)
        precision_F += row_precision_F_individual_mtf(Cm.T,Mm.T,Ftm,Sm.T,Ftm,taum,alpham,i)
    for Dl,Ml,Ftl,Gl,taul,alphal in D:
        precision_F += row_precision_F_individual_mf(Dl,Ml,Ftl,Gl,taul,alphal,i)
        
    return precision_F

def row_precision_F_individual_mf(dataset,mask,F,G,tau,alpha,i):
    ''' Return the component of the Precision update for an individual matrix, for matrix factorisation. '''
    G_masked = (mask[i] * G.T).T # zero rows when j not in mask[i]
    return tau * alpha * ( numpy.dot(G_masked.T,G_masked) )

def row_precision_F_individual_mtf(dataset,mask,F,S,G,tau,alpha,i):
    ''' Return the component of the Precision update for an individual matrix, for matrix tri-factorisation. '''
    GS_masked = (mask[i] * numpy.dot(G,S.T).T).T # zero rows when j not in mask[i]
    return tau * alpha * ( numpy.dot(GS_masked.T,GS_masked) )

 
###############################################################################
######### Row-wise updates for the mu parameter of the posterior of F #########
############################################################################### 

def row_mu_F(R,C,D,lambdaF,precision_Fi,i,nonnegative):
    ''' Return the value for mu for the Gibbs posterior, for row-wise draws. '''
    mu_F = -lambdaF if nonnegative else numpy.zeros(len(lambdaF))
    
    for Rn,Mn,Ftn,Sn,Fun,taun,alphan in R:
        mu_F += row_mu_F_individual_mtf(Rn,Mn,Ftn,Sn,Fun,taun,alphan,i)
    for Cm,Mm,Ftm,Sm,taum,alpham in C:
        mu_F += row_mu_F_individual_mtf(Cm,Mm,Ftm,Sm,Ftm,taum,alpham,i)
        mu_F += row_mu_F_individual_mtf(Cm.T,Mm.T,Ftm,Sm.T,Ftm,taum,alpham,i)
    for Dl,Ml,Ftl,Gl,taul,alphal in D:
        mu_F += row_mu_F_individual_mf(Dl,Ml,Ftl,Gl,taul,alphal,i)
      
    sigma_Fi = numpy.linalg.inv(precision_Fi)
    mu_F = numpy.dot(sigma_Fi,mu_F)
    return mu_F
    
def row_mu_F_individual_mf(dataset,mask,F,G,tau,alpha,i):
    ''' Return the component of the mu update for an individual matrix, for matrix factorisation. '''
    dataset_i_masked = mask[i] * dataset[i]
    return tau * alpha * numpy.dot(dataset_i_masked,G)

def row_mu_F_individual_mtf(dataset,mask,F,S,G,tau,alpha,i):
    ''' Return the component of the mu update for an individual matrix, for matrix tri-factorisation. '''
    dataset_i_masked = mask[i] * dataset[i]
    return tau * alpha * numpy.dot(dataset_i_masked,numpy.dot(G,S.T))
    

###############################################################################
######## Individual updates for the parameters of the posterior of S ##########
###############################################################################

def individual_tau_S(dataset,mask,tau,alpha,F,S,G,lambdaSkl,k,l,nonnegative):
    ''' Return the value for tau for the Gibbs posterior, for individual draws. '''
    tau_S = 0. if nonnegative else lambdaSkl
    tau_S += tau * alpha * ( mask * numpy.outer(F[:,k]**2,G[:,l]**2) ).sum()
    return tau_S

def individual_mu_S(dataset,mask,tau,alpha,F,S,G,lambdaSkl,k,l,tau_Skl,nonnegative):
    ''' Return the value for mu for the Gibbs posterior, for individual draws. '''
    mu_S = -lambdaSkl if nonnegative else 0. 
    mu_S += tau * alpha * ( mask * ( ( dataset - triple_dot(F,S,G.T) + S[k,l] * numpy.outer(F[:,k],G[:,l]) ) * numpy.outer(F[:,k],G[:,l]) ) ).sum() 
    mu_S /= tau_Skl      
    return mu_S
        

###############################################################################
######### Row-wise updates for the parameters of the posterior of S ###########
############################################################################### 
 
def row_precision_S(dataset,mask,tau,alpha,F,S,G,lambdaSk,k,nonnegative):
    ''' Return the value for Precision for the Gibbs posterior, for row draws. '''
    I,J = mask.shape
    precision_S = numpy.zeros((len(lambdaSk),len(lambdaSk))) if nonnegative else numpy.diag(lambdaSk)
    
    # Inefficient
    """
    indices_mask = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if mask[i,j]]
    precision_S += tau * alpha * numpy.array([F[i,k]**2 * numpy.outer(G[j],G[j]) for (i,j) in indices_mask]).sum(axis=0)
    """
    
    # Efficient - we dot F()k**2 with an I x (L x L) matrix that is sum_j in Omegai [ Gj * Gj.T ] 
    G_outer_masked = numpy.array([numpy.dot((mask[i] * G.T),(mask[i] * G.T).T) for i in range(0,I)])
    precision_S += tau * alpha * numpy.tensordot( F[:,k]**2, G_outer_masked, axes=1 )   
    
    return precision_S

def row_mu_S(dataset,mask,tau,alpha,F,S,G,lambdaSk,precision_Sk,k,nonnegative):
    ''' Return the value for tau for the Gibbs posterior, for row draws. '''
    mu_S = -lambdaSk if nonnegative else numpy.zeros(len(lambdaSk))
    mu_S += tau * alpha * numpy.dot( numpy.dot( 
        F[:,k], ( mask * ( dataset - triple_dot(F,S,G.T) + numpy.outer( F[:,k], numpy.dot(S[k,:],G.T) ) ) ) ),
        G )
    
    sigma_Sk = numpy.linalg.inv(precision_Sk)
    mu_S = numpy.dot(sigma_Sk,mu_S)
    return mu_S
        
        
###############################################################################
################# Return both parameters for the variables ####################
###############################################################################
        
def alpha_beta_tau(alphatau,betatau,importance,dataset,mask,F,G,S=None):
    ''' Return alpha and beta for the noise parameter tau. '''
    alpha = alpha_tau(
        alphatau=alphatau,importance=importance,mask=mask)
    beta = beta_tau(
        betatau=betatau,importance=importance,dataset=dataset,mask=mask,F=F,G=G,S=S)    
    return (alpha,beta)
    
def alpha_beta_importance(alphaA,betaA,tau,dataset,mask,F,G,S=None):
    ''' Return alpha and beta for the dataset importance alpha. '''
    alpha = alpha_importance(
        alphaA=alphaA)
    beta = beta_importance(
        betaA=betaA,tau=tau,dataset=dataset,mask=mask,F=F,G=G,S=S) 
    return (alpha,beta)
    
def alpha_beta_lambdat(alpha0,beta0,Fs,k):
    ''' Returna value for alpha and beta, for lambdak (ARD). '''
    alpha = alpha_lambdat(
        alpha0=alpha0,Fs=Fs)
    beta = beta_lambdat(
        beta0=beta0,Fs=Fs,k=k)
    return (alpha,beta)
    
def alpha_beta_lambdaS(alphaS,betaS,S,nonnegative):
    ''' Return an array of alpha and beta values for the element-wise sparsity of S^n or S^m. '''
    alpha = alpha_lambdaS(
        alphaS=alphaS,S=S,nonnegative=nonnegative)
    beta = beta_lambdaS(
        betaS=betaS,S=S,nonnegative=nonnegative)
    return (alpha,beta)

def column_mu_tau_F(R,C,D,lambdaF,k,nonnegative):
    ''' Return a vector of mu and tau values, for a column of Ft. '''
    tau_Fk = column_tau_F(
        R=R,C=C,D=D,lambdaF=lambdaF,k=k,nonnegative=nonnegative)
    mu_Fk = column_mu_F(
        R=R,C=C,D=D,lambdaF=lambdaF,k=k,tau_Fk=tau_Fk,nonnegative=nonnegative)
    return (mu_Fk,tau_Fk)
    
def row_mu_precision_F(R,C,D,lambdaF,i,nonnegative):
    ''' Return a vector for mu and matrix for the precision values, for a row of Ft. '''
    precision_Fi = row_precision_F(
        R=R,C=C,D=D,lambdaF=lambdaF,i=i,nonnegative=nonnegative)
    mu_Fi = row_mu_F(
        R=R,C=C,D=D,lambdaF=lambdaF,precision_Fi=precision_Fi,i=i,nonnegative=nonnegative)
    return (mu_Fi,precision_Fi)

def individual_mu_tau_S(dataset,mask,tau,alpha,F,S,G,lambdaSkl,k,l,nonnegative):
    ''' Return a value for mu and tau, for an element in S^n or S^m. '''
    tau_Skl = individual_tau_S(
        dataset=dataset,mask=mask,tau=tau,alpha=alpha,F=F,S=S,G=G,lambdaSkl=lambdaSkl,k=k,l=l,nonnegative=nonnegative)
    mu_Skl = individual_mu_S(
        dataset=dataset,mask=mask,tau=tau,alpha=alpha,F=F,S=S,G=G,lambdaSkl=lambdaSkl,k=k,l=l,tau_Skl=tau_Skl,nonnegative=nonnegative)
    return (mu_Skl,tau_Skl)
    
def row_mu_precision_S(dataset,mask,tau,alpha,F,S,G,lambdaSk,k,nonnegative):
    ''' Return a vector for mu and matrix for the precision values, for a row in S^n or S^m. '''
    precision_Sk = row_precision_S(
        dataset=dataset,mask=mask,tau=tau,alpha=alpha,F=F,S=S,G=G,lambdaSk=lambdaSk,k=k,nonnegative=nonnegative)
    mu_Sk = row_mu_S(
        dataset=dataset,mask=mask,tau=tau,alpha=alpha,F=F,S=S,G=G,lambdaSk=lambdaSk,precision_Sk=precision_Sk,k=k,nonnegative=nonnegative)
    return (mu_Sk,precision_Sk)