'''
Unit tests for the methods in updates_Gibbs.py.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import HMF.code.Gibbs.updates_Gibbs as updates_Gibbs

import numpy, itertools


###############################################################################
################################### Helpers ###################################
###############################################################################

def test_triple_dot():
    M1 = numpy.ones((3,4))
    M2 = numpy.ones((4,5))
    M3 = numpy.ones((5,6))
    M_expected = 20*numpy.ones((3,6))
    M = updates_Gibbs.triple_dot(M1,M2,M3)
    assert numpy.array_equal(M_expected,M)
    

###############################################################################
###### Updates for the alpha and beta parameters of the noise parameters ######
###############################################################################

I,J,K,L = 5,3,2,4
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0

F = 1./2. * numpy.ones((I,K))
S = 1./3. * numpy.ones((K,L))
G = 1./5. * numpy.ones((J,L))
U = 1./2. * numpy.ones((I,K))
V = 1./3. * numpy.ones((J,K))
R_pred_MTF = 11./15. * numpy.ones((I,J)) #1-4/15
R_pred_MF = 2./3. * numpy.ones((I,J)) #1-1/3

lambdaF = 2
lambdaS = 3
lambdaG = 5
alphatau, betatau = 3, 1
alpha0, beta0 = 6, 7

def test_alpha_tau():
    alphatau_s = alphatau + alpha * M.sum() / 2.
    assert alphatau_s == updates_Gibbs.alpha_tau(alphatau,alpha,M)

def test_beta_tau():
    # MTF case
    betatau_s = betatau + alpha * .5*((R_pred_MTF*M)**2).sum()
    assert abs(betatau_s - updates_Gibbs.beta_tau(betatau,alpha,R,M,F,G,S)) < 0.0000000000001
    
    # MF case
    betatau_s = betatau + alpha * .5*((R_pred_MF*M)**2).sum()
    assert abs(betatau_s - updates_Gibbs.beta_tau(betatau,alpha,R,M,U,V)) < 0.00000000001
        
def test_alpha_beta_tau():
    # MTF case
    expected_alphatau_s = alphatau + alpha * M.sum() / 2.
    expected_betatau_s = betatau + alpha * .5*((R_pred_MTF*M)**2).sum()
    
    alphatau_s, betatau_s = updates_Gibbs.alpha_beta_tau(alphatau,betatau,alpha,R,M,F,G,S)
    assert abs(expected_alphatau_s - alphatau_s) < 0.0000001
    assert abs(expected_betatau_s - betatau_s) < 0.0000001
    
    # MF case
    expected_betatau_s = betatau + alpha * .5*((R_pred_MF*M)**2).sum()
    
    alphatau_s, betatau_s = updates_Gibbs.alpha_beta_tau(alphatau,betatau,alpha,R,M,U,V)
    assert abs(expected_alphatau_s - alphatau_s) < 0.0000001
    assert abs(expected_betatau_s - betatau_s) < 0.0000001
    
       
###############################################################################
########## Updates for the parameters of the posterior of lambda_t ############
############################################################################### 
        
I,K = 5,4
F_list = [
    (1*numpy.ones((I,K)),True),
    (2*numpy.ones((I,K)),True),
    (3*numpy.ones((J,K)),False)
]        
alpha0, beta0 = 4., 5.
        
def test_alpha_lambdat():
    expected_alphalambda = alpha0 + (I + I + J/2.)
    alphalambda = updates_Gibbs.alpha_lambdat(alpha0,F_list)
    assert alphalambda == expected_alphalambda
          
def test_beta_lambdat():
    expected_betalambda = beta0 + (I + 2*I + 3**2*J/2.)
    for k in range(0,K):
        betalambda = updates_Gibbs.beta_lambdat(beta0,F_list,k)
        assert betalambda == expected_betalambda

def test_alpha_beta_lambdat():
    expected_alphalambda = alpha0 + (I + I + J/2.)
    expected_betalambda = beta0 + (I + 2*I + 3**2*J/2.)
    for k in range(0,K):
        alphalambda, betalambda = updates_Gibbs.alpha_beta_lambdat(alpha0,beta0,F_list,k)
        assert alphalambda == expected_alphalambda
        assert betalambda == expected_betalambda

       
###############################################################################
##### Updates for the parameters of the posterior of lambdan and lambdam ######
############################################################################### 
        
alphaS, betaS = 4., 5.
def test_alpha_lambdaS():
    # Real-valued
    expected_alpha1 = numpy.ones((K,L)) * (alphaS + .5)
    alpha1 = updates_Gibbs.alpha_lambdaS(alphaS,S,False)
    assert numpy.array_equal(alpha1, expected_alpha1)
    
    # Nonnegative
    expected_alpha2 = numpy.ones((K,L)) * (alphaS + 1.)
    alpha2 = updates_Gibbs.alpha_lambdaS(alphaS,S,True)
    assert numpy.array_equal(alpha2, expected_alpha2)
          
def test_beta_lambdaS():
    # Real-valued
    expected_beta1 = numpy.ones((K,L)) * (betaS + S[0,0]**2/2.)
    beta1 = updates_Gibbs.beta_lambdaS(betaS,S,False)
    assert numpy.array_equal(beta1, expected_beta1)
    
    # Nonnegative
    expected_beta2 = numpy.ones((K,L)) * (betaS + S[0,0])
    beta2 = updates_Gibbs.beta_lambdaS(betaS,S,True)
    assert numpy.array_equal(beta2, expected_beta2)

def test_alpha_beta_lambdaS():
    # Real-valued
    expected_alpha1 = numpy.ones((K,L)) * (alphaS + .5)
    expected_beta1 = numpy.ones((K,L)) * (betaS + S[0,0]**2/2.)
    alpha1, beta1 = updates_Gibbs.alpha_beta_lambdaS(alphaS,betaS,S,False)
    assert numpy.array_equal(alpha1, expected_alpha1)
    assert numpy.array_equal(beta1, expected_beta1)
        
    # Nonnegative
    expected_alpha2 = numpy.ones((K,L)) * (alphaS + 1.)
    expected_beta2 = numpy.ones((K,L)) * (betaS + S[0,0])
    alpha2, beta2 = updates_Gibbs.alpha_beta_lambdaS(alphaS,betaS,S,True)
    assert numpy.array_equal(alpha2, expected_alpha2)
    assert numpy.array_equal(beta2, expected_beta2)

        
###############################################################################
####### Column-wise updates for the tau parameter of the posterior of F #######
####### Column-wise updates for the mu parameter of the posterior of F ########
###############################################################################
        
I,J,K = 5,3,2
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0

U = 1./2. * numpy.ones((I,K))
V = 1./3. * numpy.ones((J,K))
lambdaU = 2 * numpy.ones((I,K))
lambdaV = 2 * numpy.ones((J,K))

tau = 3.
alpha = 2.

def test_column_tau_individual_mf():
    ''' Test for U '''
    expected_tauU = tau*alpha*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        tauUk = updates_Gibbs.column_tau_individual_mf(R,M,U,V,tau,alpha,k)
        assert tauUk[i] == expected_tauU[i,k]

    ''' Test for V '''
    expected_tauV = tau*alpha*numpy.array([[1.,1.],[1.,1.],[1.,1.]])
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        tauVk = updates_Gibbs.column_tau_individual_mf(R.T,M.T,V,U,tau,alpha,k)
        assert tauVk[j] == expected_tauV[j,k] 

def test_column_mu_individual_mf():
    ''' Test for U '''
    expected_muU = tau * alpha * numpy.array([[2.*(5./6.)*(1./3.),10./18.],[15./18.,15./18.],[10./18.,10./18.],[10./18.,10./18.],[15./18.,15./18.]])
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        muUk = updates_Gibbs.column_mu_individual_mf(R,M,U,V,tau,alpha,k)
        assert abs(muUk[i] - expected_muU[i,k]) < 0.000000000000001
        
    ''' Test for V '''
    expected_muV = tau * alpha * numpy.array([[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)]])
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        muVk = updates_Gibbs.column_mu_individual_mf(R.T,M.T,V,U,tau,alpha,k)
        assert muVk[j] == expected_muV[j,k]
      
###############################################################################
      
F = 1./2. * numpy.ones((I,K))
S = 1./3. * numpy.ones((K,L))
G = 1./5. * numpy.ones((J,L))

lambdaF = 2 * numpy.ones((I,K))
lambdaS = 3 * numpy.ones((K,L))
lambdaG = 5 * numpy.ones((J,L))
        
def test_column_tau_individual_mtf():
    ''' Test for F '''
    expected_tauF = tau*alpha*numpy.array([[32./225.,32./225.],[48./225.,48./225.],[32./225.,32./225.],[32./225.,32./225.],[48./225.,48./225.]])
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        tauFk = updates_Gibbs.column_tau_individual_mtf(R,M,F,S,G,tau,alpha,k)
        assert abs(tauFk[i] - expected_tauF[i,k]) < 0.00000001

    ''' Test for G '''
    expected_tauG = tau*alpha*numpy.array([[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.]])
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        tauGl = updates_Gibbs.column_tau_individual_mtf(R.T,M.T,G,S.T,F,tau,alpha,l)
        assert abs(tauGl[j] - expected_tauG[j,l]) < 0.00000001

def test_column_mu_individual_mtf():
    ''' Test for F '''
    expected_muF = tau*alpha*numpy.array([[2*(52./225.),2*(52./225.)],[3*(52./225.),3*(52./225.)],[2*(52./225.),2*(52./225.)],[2*(52./225.),2*(52./225.)],[3*(52./225.),3*(52./225.)]])
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        muFk = updates_Gibbs.column_mu_individual_mtf(R,M,F,S,G,tau,alpha,k)
        assert abs(muFk[i] - expected_muF[i,k]) < 0.00000001

    ''' Test for G '''
    expected_muG = tau*alpha*numpy.array([[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.],[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.],[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.]])
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        muGl = updates_Gibbs.column_mu_individual_mtf(R.T,M.T,G,S.T,F,tau,alpha,l)
        assert abs(muGl[j] - expected_muG[j,l]) < 0.00000001

###############################################################################

# For BSNMTF tests
Is,Ks = 5,2
Cs = numpy.ones((Is,Is))
Ms = numpy.ones((Is,Is))
Ms[0,0], Ms[2,2], Ms[3,1] = 0, 0, 0

Fs = 1./2. * numpy.ones((Is,Ks))
Ss = 1./3. * numpy.ones((Ks,Ks))

lambdaFs = 2 * numpy.ones((Is,Ks))
lambdaSs = 3 * numpy.ones((Ks,Ks))

def test_column_tau_F():
    ''' Test for MF - nonnegative '''
    expected_tauU = tau*alpha*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    expected_tauV = tau*alpha*numpy.array([[1.,1.],[1.,1.],[1.,1.]])
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        tauUk = updates_Gibbs.column_tau_F([],[],[(R,M,U,V,tau,alpha)],lambdaU[i,:],k,True)
        assert abs(tauUk[i] - expected_tauU[i,k]) < 0.0000001
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        tauVk = updates_Gibbs.column_tau_F([],[],[(R.T,M.T,V,U,tau,alpha)],lambdaV[j,:],k,True)
        assert abs(tauVk[j] - expected_tauV[j,k]) < 0.0000001
        
    ''' Test for MF - not nonnegative '''  
    expected_tauU = lambdaU + tau*alpha*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    expected_tauV = lambdaV + tau*alpha*numpy.array([[1.,1.],[1.,1.],[1.,1.]])
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        tauUk = updates_Gibbs.column_tau_F([],[],[(R,M,U,V,tau,alpha)],lambdaU[i,:],k,False)
        assert abs(tauUk[i] - expected_tauU[i,k]) < 0.0000001
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        tauVk = updates_Gibbs.column_tau_F([],[],[(R.T,M.T,V,U,tau,alpha)],lambdaV[j,:],k,False)
        assert abs(tauVk[j] - expected_tauV[j,k]) < 0.0000001
        
    ''' Test for MTF - nonnegative '''
    expected_tauF = tau*alpha*numpy.array([[32./225.,32./225.],[48./225.,48./225.],[32./225.,32./225.],[32./225.,32./225.],[48./225.,48./225.]])
    expected_tauG = tau*alpha*numpy.array([[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.]])
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        tauFk = updates_Gibbs.column_tau_F([(R,M,F,S,G,tau,alpha)],[],[],lambdaF[i,:],k,True)
        assert abs(tauFk[i] - expected_tauF[i,k]) < 0.00000001
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        tauGl = updates_Gibbs.column_tau_F([(R.T,M.T,G,S.T,F,tau,alpha)],[],[],lambdaG[j,:],l,True)
        assert abs(tauGl[j] - expected_tauG[j,l]) < 0.00000001

    ''' Test for SMTF - nonnegative ''' 
    expected_tauF = tau*alpha*numpy.array([[4./9.+4./9.,4./9.+4./9.],[5./9.+4./9.,5./9.+4./9.],[4./9.+4./9.,4./9.+4./9.],[4./9.+5./9.,4./9.+5./9.],[5./9.+5./9.,5./9.+5./9.]])
    
    for i,k in itertools.product(xrange(0,Is),xrange(0,Ks)):
        tauFk = updates_Gibbs.column_tau_F([],[(Cs,Ms,Fs,Ss,tau,alpha)],[],lambdaFs[i,:],k,True)
        assert abs(tauFk[i] - expected_tauF[i,k]) < 0.00000001
        
def test_column_mu_F():
    ''' Test for MF - nonnegative '''
    tauU = tau*alpha*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    tauV = tau*alpha*numpy.array([[1.,1.],[1.,1.],[1.,1.]])    
    
    expected_muU = 1./tauU * ( -lambdaU + tau * alpha * numpy.array([[2.*(5./6.)*(1./3.),10./18.],[15./18.,15./18.],[10./18.,10./18.],[10./18.,10./18.],[15./18.,15./18.]]))
    expected_muV = 1./tauV * ( -lambdaV + tau * alpha * numpy.array([[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)]]))
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        muUk = updates_Gibbs.column_mu_F([],[],[(R,M,U,V,tau,alpha)],lambdaU[i,:],tauU[:,k],k,True)
        assert abs(muUk[i] - expected_muU[i,k]) < 0.0000001
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        muVk = updates_Gibbs.column_mu_F([],[],[(R.T,M.T,V,U,tau,alpha)],lambdaV[j,:],tauV[:,k],k,True)
        assert abs(muVk[j] - expected_muV[j,k]) < 0.0000001
    
    ''' Test for MF - not nonnegative '''  
    expected_muU = 1./tauU * ( tau * alpha * numpy.array([[2.*(5./6.)*(1./3.),10./18.],[15./18.,15./18.],[10./18.,10./18.],[10./18.,10./18.],[15./18.,15./18.]]))
    expected_muV = 1./tauV * ( tau * alpha * numpy.array([[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)]]))
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        muUk = updates_Gibbs.column_mu_F([],[],[(R,M,U,V,tau,alpha)],lambdaU[i,:],tauU[:,k],k,False)
        assert abs(muUk[i] - expected_muU[i,k]) < 0.0000001
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        muVk = updates_Gibbs.column_mu_F([],[],[(R.T,M.T,V,U,tau,alpha)],lambdaV[j,:],tauV[:,k],k,False)
        assert abs(muVk[j] - expected_muV[j,k]) < 0.0000001
    
    ''' Test for MTF - nonnegative '''
    tauF = tau*alpha*numpy.array([[32./225.,32./225.],[48./225.,48./225.],[32./225.,32./225.],[32./225.,32./225.],[48./225.,48./225.]])
    tauG = tau*alpha*numpy.array([[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.]])
    
    expected_muF = 1./tauF * ( -lambdaF + tau*alpha*numpy.array([[2*(52./225.),2*(52./225.)],[3*(52./225.),3*(52./225.)],[2*(52./225.),2*(52./225.)],[2*(52./225.),2*(52./225.)],[3*(52./225.),3*(52./225.)]]) )
    expected_muG = 1./tauG * ( -lambdaG + tau*alpha*numpy.array([[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.],[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.],[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.]]) )
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        muFk = updates_Gibbs.column_mu_F([(R,M,F,S,G,tau,alpha)],[],[],lambdaF[i,:],tauF[:,k],k,True)
        assert abs(muFk[i] - expected_muF[i,k]) < 0.00000001
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        muGl = updates_Gibbs.column_mu_F([(R.T,M.T,G,S.T,F,tau,alpha)],[],[],lambdaG[j,:],tauG[:,k],k,True)
        assert abs(muGl[j] - expected_muG[j,l]) < 0.00000001

    ''' Test for SMTF - nonnegative ''' 
    tauF = tau*alpha*numpy.array([[4./9.+4./9.,4./9.+4./9.],[5./9.+4./9.,5./9.+4./9.],[4./9.+4./9.,4./9.+4./9.],[4./9.+5./9.,4./9.+5./9.],[5./9.+5./9.,5./9.+5./9.]])
    expected_muF = 1./tauF * ( alpha * tau * numpy.array([[8*10./36.,8*10./36.],[9*10./36.,9*10./36.],[8*10./36.,8*10./36.],[9*10./36.,9*10./36.],[10*10./36.,10*10./36.]]) - lambdaF )
    
    for i,k in itertools.product(xrange(0,Is),xrange(0,Ks)):
        muFk = updates_Gibbs.column_mu_F([],[(Cs,Ms,Fs,Ss,tau,alpha)],[],lambdaFs[i,:],tauF[:,k],k,True)
        assert abs(muFk[i] - expected_muF[i,k]) < 0.00000001
    
###############################################################################

def test_column_mu_tau_F():
    ''' Test with MF, MTF, SMTF '''
    expected_tauF = tau*alpha*numpy.array([[32./225.,32./225.],[48./225.,48./225.],[32./225.,32./225.],[32./225.,32./225.],[48./225.,48./225.]]) + \
                    tau*alpha*numpy.array([[4./9.+4./9.,4./9.+4./9.],[5./9.+4./9.,5./9.+4./9.],[4./9.+4./9.,4./9.+4./9.],[4./9.+5./9.,4./9.+5./9.],[5./9.+5./9.,5./9.+5./9.]]) + \
                    tau*alpha*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    expected_muF = 1./expected_tauF * ( -lambdaF + \
                    tau*alpha*numpy.array([[2*(52./225.),2*(52./225.)],[3*(52./225.),3*(52./225.)],[2*(52./225.),2*(52./225.)],[2*(52./225.),2*(52./225.)],[3*(52./225.),3*(52./225.)]]) + \
                    alpha * tau * numpy.array([[8*10./36.,8*10./36.],[9*10./36.,9*10./36.],[8*10./36.,8*10./36.],[9*10./36.,9*10./36.],[10*10./36.,10*10./36.]]) + \
                    tau * alpha * numpy.array([[2.*(5./6.)*(1./3.),10./18.],[15./18.,15./18.],[10./18.,10./18.],[10./18.,10./18.],[15./18.,15./18.]]) )
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        muFk, tauFk = updates_Gibbs.column_mu_tau_F(
            R=[(R,M,F,S,G,tau,alpha)],
            C=[(Cs,Ms,Fs,Ss,tau,alpha)],
            D=[(R,M,U,V,tau,alpha)],
            lambdaF=lambdaF[i,:],
            k=k,
            nonnegative=True
        )
        assert abs(muFk[i] - expected_muF[i,k]) < 0.00000001
        assert abs(tauFk[i] - expected_tauF[i,k]) < 0.00000001
    
        
###############################################################################
######## Row-wise updates for the tau parameter of the posterior of F #########
######## Row-wise updates for the mu parameter of the posterior of F ##########
###############################################################################
        
Up = numpy.ones((I,K))
Up[0,0] = 1.
Up[1,1] = 2. # to make the precision nonsingular
Vp = numpy.ones((J,K))
Vp[0,0] = 3.
Vp[1,1] = 4. # to make the precision nonsingular
         
Fp = numpy.ones((I,K))
Fp[0,0] = 5.
Fp[1,1] = 6.# to make the precision nonsingular
Gp = numpy.ones((J,L))
Gp[0,0] = 7.
Gp[1,1] = 8. 
Gp[2,2] = 9. # to make the precision nonsingular
Sp = numpy.ones((K,L))
Sp[0,0] = 10.
Sp[1,1] = 11. 
Sp[1,2] = 12. # to make the precision nonsingular
    
Fsp = 1./2. * numpy.ones((Is,Ks))
Ssp = 1./3. * numpy.ones((Ks,Ks))

Fsp[0,0] = 1.
Fsp[1,1] = 2. # to make precision non-singular
Ssp[0,1] = 3.
Ssp[0,1] = 4. # to make precision non-singular

def test_row_precision_F_individual_mf():
    indices_mask_i = [[j for j in range(0,J) if M[i,j]] for i in range(0,I)]
    indices_mask_j = [[i for i in range(0,I) if M[i,j]] for j in range(0,J)]
    
    for i in range(0,I):
        expected_precision_U = numpy.array([[
            tau*alpha*sum([V[j,k]*V[j,kp] for j in indices_mask_i[i]])
        for kp in range(0,K)] for k in range(0,K)])    
        precision_U = updates_Gibbs.row_precision_F_individual_mf(R,M,U,V,tau,alpha,i)
        assert numpy.array_equal(expected_precision_U,precision_U)
    
    for j in range(0,J):
        expected_precision_V = numpy.array([[
            tau*alpha*sum([U[i,k]*U[i,kp] for i in indices_mask_j[j]])
        for kp in range(0,K)] for k in range(0,K)])    
        precision_V = updates_Gibbs.row_precision_F_individual_mf(R.T,M.T,V,U,tau,alpha,j)
        assert numpy.array_equal(expected_precision_V,precision_V)
    
def test_row_mu_individual_mf():
    indices_mask_i = [[j for j in range(0,J) if M[i,j]] for i in range(0,I)]
    indices_mask_j = [[i for i in range(0,I) if M[i,j]] for j in range(0,J)]
    
    for i in range(0,I):
        expected_mu_U = numpy.array([
            tau*alpha*sum([R[i,j]*V[j,k] for j in indices_mask_i[i]])
        for k in range(0,K)])
        
        mu_U = updates_Gibbs.row_mu_F_individual_mf(R,M,U,V,tau,alpha,i)
        assert numpy.array_equal(expected_mu_U,mu_U)
        
    for j in range(0,J):
        expected_mu_V = numpy.array([
            tau*alpha*sum([R[i,j]*U[i,k] for i in indices_mask_j[j]])
        for k in range(0,K)])
        
        mu_V = updates_Gibbs.row_mu_F_individual_mf(R.T,M.T,V,U,tau,alpha,j)
        assert numpy.array_equal(expected_mu_V,mu_V)
        
###############################################################################

def test_row_precision_F_individual_mtf():
    indices_mask_i = [[j for j in range(0,J) if M[i,j]] for i in range(0,I)]
    indices_mask_j = [[i for i in range(0,I) if M[i,j]] for j in range(0,J)]
    
    for i in range(0,I):
        expected_precision_F = numpy.array([[
            tau*alpha*sum([numpy.dot(S[k],G[j]) * numpy.dot(S[kp],G[j]) for j in indices_mask_i[i]])
        for kp in range(0,K)] for k in range(0,K)])    
        
        precision_F = updates_Gibbs.row_precision_F_individual_mtf(R,M,F,S,G,tau,alpha,i)
        assert numpy.array_equal(expected_precision_F,precision_F)
    
    for j in range(0,J):
        expected_precision_G = numpy.array([[
            tau*alpha*sum([numpy.dot(F[i,:],S[:,l]) * numpy.dot(F[i,:],S[:,lp]) for i in indices_mask_j[j]])
        for lp in range(0,L)] for l in range(0,L)])    
        
        precision_G = updates_Gibbs.row_precision_F_individual_mtf(R.T,M.T,G,S.T,F,tau,alpha,j)
        assert numpy.array_equal(expected_precision_G,precision_G)
    
def test_row_mu_individual_mtf():
    indices_mask_i = [[j for j in range(0,J) if M[i,j]] for i in range(0,I)]
    indices_mask_j = [[i for i in range(0,I) if M[i,j]] for j in range(0,J)]
    
    for i in range(0,I):
        expected_mu_F = numpy.array([
            tau*alpha*sum([R[i,j]*numpy.dot(S[k],G[j]) for j in indices_mask_i[i]])
        for k in range(0,K)])
        
        mu_F = updates_Gibbs.row_mu_F_individual_mtf(R,M,F,S,G,tau,alpha,i)
        assert numpy.array_equal(expected_mu_F,mu_F)
        
    for j in range(0,J):
        expected_mu_G = numpy.array([
            tau*alpha*sum([R[i,j]*numpy.dot(F[i],S[:,l]) for i in indices_mask_j[j]])
        for l in range(0,L)])
        
        mu_G = updates_Gibbs.row_mu_F_individual_mtf(R.T,M.T,G,S.T,F,tau,alpha,j)
        assert numpy.array_equal(expected_mu_G,mu_G)

###############################################################################

def test_row_precision_F():
    indices_mask_i = [[j for j in range(0,J) if M[i,j]] for i in range(0,I)]
    indices_mask_j = [[i for i in range(0,I) if M[i,j]] for j in range(0,J)]
    
    indices_mask_i_s = [[j for j in range(0,I) if Ms[i,j]] for i in range(0,I)]
    indices_mask_j_s = [[i for i in range(0,I) if Ms[i,j]] for j in range(0,I)]
    
    ''' Test for MF - nonnegative '''
    for i in range(0,I):
        expected_precision_U = numpy.array([[
            tau*alpha*sum([V[j,k]*V[j,kp] for j in indices_mask_i[i]])
        for kp in range(0,K)] for k in range(0,K)])    
        precision_U = updates_Gibbs.row_precision_F([],[],[(R,M,U,V,tau,alpha)],lambdaU[i],i,True)
        assert numpy.array_equal(expected_precision_U,precision_U)
    
    for j in range(0,J):
        expected_precision_V = numpy.array([[
            tau*alpha*sum([U[i,k]*U[i,kp] for i in indices_mask_j[j]])
        for kp in range(0,K)] for k in range(0,K)])    
        precision_V = updates_Gibbs.row_precision_F([],[],[(R.T,M.T,V,U,tau,alpha)],lambdaV[j],j,True)
        assert numpy.array_equal(expected_precision_V,precision_V)
        
    ''' Test for MF - not nonnegative '''  
    for i in range(0,I):
        expected_precision_U = numpy.array([[
            tau*alpha*sum([V[j,k]*V[j,kp] for j in indices_mask_i[i]])
        for kp in range(0,K)] for k in range(0,K)]) 
        for k in range(0,K):
            expected_precision_U[k,k] += lambdaU[i,k]
        
        precision_U = updates_Gibbs.row_precision_F([],[],[(R,M,U,V,tau,alpha)],lambdaU[i],i,False)
        assert numpy.array_equal(expected_precision_U,precision_U)
    
    for j in range(0,J):
        expected_precision_V = numpy.array([[
            tau*alpha*sum([U[i,k]*U[i,kp] for i in indices_mask_j[j]])
        for kp in range(0,K)] for k in range(0,K)])    
        for k in range(0,K):
            expected_precision_V[k,k] += lambdaV[j,k]
            
        precision_V = updates_Gibbs.row_precision_F([],[],[(R.T,M.T,V,U,tau,alpha)],lambdaV[j],j,False)
        assert numpy.array_equal(expected_precision_V,precision_V)
    
    ''' Test for MTF - nonnegative '''
    for i in range(0,I):
        expected_precision_F = numpy.array([[
            tau*alpha*sum([numpy.dot(S[k],G[j]) * numpy.dot(S[kp],G[j]) for j in indices_mask_i[i]])
        for kp in range(0,K)] for k in range(0,K)])    
        precision_F = updates_Gibbs.row_precision_F([(R,M,F,S,G,tau,alpha)],[],[],lambdaF[i],i,True)
        assert numpy.array_equal(expected_precision_F,precision_F)
    
    for j in range(0,J):
        expected_precision_G = numpy.array([[
            tau*alpha*sum([numpy.dot(F[i,:],S[:,l]) * numpy.dot(F[i,:],S[:,lp]) for i in indices_mask_j[j]])
        for lp in range(0,L)] for l in range(0,L)])    
        precision_G = updates_Gibbs.row_precision_F([(R.T,M.T,G,S.T,F,tau,alpha)],[],[],lambdaG[j],j,True)
        assert numpy.array_equal(expected_precision_G,precision_G)
    
    ''' Test for SMTF - nonnegative ''' 
    for i in range(0,I):
        expected_precision_F = numpy.array([[
            tau*alpha*(
                sum([numpy.dot(Ss[k],Fs[j]) * numpy.dot(Ss[kp],Fs[j]) for j in indices_mask_i_s[i]]) + 
                sum([numpy.dot(Fs[ip,:],Ss[:,k]) * numpy.dot(Fs[ip,:],Ss[:,kp]) for ip in indices_mask_j_s[i]])
            )
        for kp in range(0,K)] for k in range(0,K)])    
        precision_F = updates_Gibbs.row_precision_F([],[(Cs,Ms,Fs,Ss,tau,alpha)],[],lambdaFs[i],i,True)
        assert all(numpy.isclose(expected_precision_F,precision_F).flatten())
    

def test_row_mu_F():
    indices_mask_i = [[j for j in range(0,J) if M[i,j]] for i in range(0,I)]
    indices_mask_j = [[i for i in range(0,I) if M[i,j]] for j in range(0,J)]
    
    indices_mask_i_s = [[j for j in range(0,I) if Ms[i,j]] for i in range(0,I)]
    indices_mask_j_s = [[i for i in range(0,I) if Ms[i,j]] for j in range(0,I)]
    
    ''' Test for MF - nonnegative '''
    for i in range(0,I):
        precision_U = numpy.array([[
            tau*alpha*sum([Vp[j,k]*Vp[j,kp] for j in indices_mask_i[i]])
        for kp in range(0,K)] for k in range(0,K)])   
        sigma_U = numpy.linalg.inv(precision_U)        
        
        expected_mu_U = numpy.array([
            - lambdaU[i,k] + tau*alpha*sum([R[i,j]*Vp[j,k] for j in indices_mask_i[i]])
        for k in range(0,K)])
        expected_mu_U = numpy.dot(sigma_U,expected_mu_U)
        
        mu_U = updates_Gibbs.row_mu_F([],[],[(R,M,Up,Vp,tau,alpha)],lambdaU[i],precision_U,i,True)
        assert numpy.array_equal(expected_mu_U,mu_U)
        
    for j in range(0,J):
        precision_V = numpy.array([[
            tau*alpha*sum([Up[i,k]*Up[i,kp] for i in indices_mask_j[j]])
        for kp in range(0,K)] for k in range(0,K)])   
        sigma_V = numpy.linalg.inv(precision_V)        
        
        expected_mu_V = numpy.array([
            - lambdaV[j,k] + tau*alpha*sum([R[i,j]*Up[i,k] for i in indices_mask_j[j]])
        for k in range(0,K)])
        expected_mu_V = numpy.dot(sigma_V,expected_mu_V)
        
        mu_V = updates_Gibbs.row_mu_F([],[],[(R.T,M.T,Vp,Up,tau,alpha)],lambdaV[j],precision_V,j,True)
        assert numpy.array_equal(expected_mu_V,mu_V)
        
    ''' Test for MF - not nonnegative '''  
    for i in range(0,I):
        precision_U = numpy.array([[
            tau*alpha*sum([Vp[j,k]*Vp[j,kp] for j in indices_mask_i[i]])
        for kp in range(0,K)] for k in range(0,K)])   
        for k in range(0,K):
            precision_U[k,k] += lambdaU[i,k]
        sigma_U = numpy.linalg.inv(precision_U)        
        
        expected_mu_U = numpy.array([
            tau*alpha*sum([R[i,j]*Vp[j,k] for j in indices_mask_i[i]])
        for k in range(0,K)])
        expected_mu_U = numpy.dot(sigma_U,expected_mu_U)
        
        mu_U = updates_Gibbs.row_mu_F([],[],[(R,M,Up,Vp,tau,alpha)],lambdaU[i],precision_U,i,False)
        assert numpy.array_equal(expected_mu_U,mu_U)
        
    for j in range(0,J):
        precision_V = numpy.array([[
            tau*alpha*sum([Up[i,k]*Up[i,kp] for i in indices_mask_j[j]])
        for kp in range(0,K)] for k in range(0,K)])   
        for k in range(0,K):
            precision_V[k,k] += lambdaV[j,k] 
        sigma_V = numpy.linalg.inv(precision_V)        
        
        expected_mu_V = numpy.array([
            tau*alpha*sum([R[i,j]*Up[i,k] for i in indices_mask_j[j]])
        for k in range(0,K)])
        expected_mu_V = numpy.dot(sigma_V,expected_mu_V)
        
        mu_V = updates_Gibbs.row_mu_F([],[],[(R.T,M.T,Vp,Up,tau,alpha)],lambdaV[j],precision_V,j,False)
        assert numpy.array_equal(expected_mu_V,mu_V)
    
    ''' Test for MTF - nonnegative '''
    for i in range(0,I):
        precision_F = numpy.array([[
            tau*alpha*sum([numpy.dot(Sp[k],Gp[j]) * numpy.dot(Sp[kp],Gp[j]) for j in indices_mask_i[i]])
        for kp in range(0,K)] for k in range(0,K)])   
        sigma_F = numpy.linalg.inv(precision_F)        
        
        expected_mu_F = numpy.array([
            - lambdaF[i,k] + tau*alpha*sum([R[i,j]*numpy.dot(Sp[k],Gp[j]) for j in indices_mask_i[i]])
        for k in range(0,K)])
        expected_mu_F = numpy.dot(sigma_F,expected_mu_F)
        
        mu_F = updates_Gibbs.row_mu_F([(R,M,Fp,Sp,Gp,tau,alpha)],[],[],lambdaF[i],precision_F,i,True)
        assert all(numpy.isclose(expected_mu_F,mu_F))
        
    for j in range(0,J):
        precision_G = numpy.array([[
            tau*alpha*sum([numpy.dot(Fp[i,:],Sp[:,l]) * numpy.dot(Fp[i,:],Sp[:,lp]) for i in indices_mask_j[j]])
        for lp in range(0,L)] for l in range(0,L)])  
        sigma_G = numpy.linalg.inv(precision_G)        
        
        expected_mu_G = numpy.array([
            - lambdaG[j,l] + tau*alpha*sum([R[i,j]*numpy.dot(Fp[i],Sp[:,l]) for i in indices_mask_j[j]])
        for l in range(0,L)])
        expected_mu_G = numpy.dot(sigma_G,expected_mu_G)
        
        mu_G = updates_Gibbs.row_mu_F([(R.T,M.T,Gp,Sp.T,Fp,tau,alpha)],[],[],lambdaG[j],precision_G,j,True)
        assert numpy.array_equal(expected_mu_G,mu_G)
    
    ''' Test for SMTF - nonnegative ''' 
    for i in range(0,I):
        precision_F = numpy.array([[
            tau*alpha*(
                sum([numpy.dot(Ssp[k],Fsp[j]) * numpy.dot(Ssp[kp],Fsp[j]) for j in indices_mask_i_s[i]]) + 
                sum([numpy.dot(Fsp[ip,:],Ssp[:,k]) * numpy.dot(Fsp[ip,:],Ssp[:,kp]) for ip in indices_mask_j_s[i]])
            )
        for kp in range(0,K)] for k in range(0,K)]) 
        sigma_F = numpy.linalg.inv(precision_F)  
        
        expected_mu_F = numpy.array([
            - lambdaFs[i,k] + tau*alpha*sum([Cs[i,j]*numpy.dot(Ssp[k],Fsp[j]) for j in indices_mask_i_s[i]]) \
                            + tau*alpha*sum([Cs[ip,i]*numpy.dot(Fsp[ip],Ssp[:,k]) for ip in indices_mask_j_s[i]])
        for k in range(0,K)])
        expected_mu_F = numpy.dot(sigma_F,expected_mu_F)
        
        mu_F = updates_Gibbs.row_mu_F([],[(Cs,Ms,Fsp,Ssp,tau,alpha)],[],lambdaFs[i],precision_F,i,True)
        assert numpy.array_equal(expected_mu_F,mu_F)

###############################################################################

def test_row_mu_precision_F():
    indices_mask_i = [[j for j in range(0,J) if M[i,j]] for i in range(0,I)]
    
    indices_mask_i_s = [[j for j in range(0,I) if Ms[i,j]] for i in range(0,I)]
    indices_mask_j_s = [[i for i in range(0,I) if Ms[i,j]] for j in range(0,I)]
    
    ''' Test with MF, MTF, SMTF '''
    for i in range(0,I):
        expected_precision_F = numpy.array([[
            tau*alpha*sum([Vp[j,k]*Vp[j,kp] for j in indices_mask_i[i]]) + \
            tau*alpha*sum([numpy.dot(Sp[k],Gp[j]) * numpy.dot(Sp[kp],Gp[j]) for j in indices_mask_i[i]]) + \
            tau*alpha*(
                sum([numpy.dot(Ssp[k],Fsp[j]) * numpy.dot(Ssp[kp],Fsp[j]) for j in indices_mask_i_s[i]]) + 
                sum([numpy.dot(Fsp[ip,:],Ssp[:,k]) * numpy.dot(Fsp[ip,:],Ssp[:,kp]) for ip in indices_mask_j_s[i]])
            )
        for kp in range(0,K)] for k in range(0,K)])  
        
        sigma_F = numpy.linalg.inv(expected_precision_F)  
        
        expected_mu_F = numpy.array([
            - lambdaF[i,k] + tau*alpha*sum([R[i,j]*Vp[j,k] for j in indices_mask_i[i]]) \
            + tau*alpha*sum([R[i,j]*numpy.dot(Sp[k],Gp[j]) for j in indices_mask_i[i]]) \
            + tau*alpha*sum([Cs[i,j]*numpy.dot(Ssp[k],Fsp[j]) for j in indices_mask_i_s[i]]) \
            + tau*alpha*sum([Cs[ip,i]*numpy.dot(Fsp[ip],Ssp[:,k]) for ip in indices_mask_j_s[i]])
        for k in range(0,K)])
        expected_mu_F = numpy.dot(sigma_F,expected_mu_F)
        
        mu_F, precision_F = updates_Gibbs.row_mu_precision_F([(R,M,Fp,Sp,Gp,tau,alpha)],[(Cs,Ms,Fsp,Ssp,tau,alpha)],[(R,M,Up,Vp,tau,alpha)],lambdaF[i],i,True)

        for k in range(0,K):        
            assert abs(expected_mu_F[k] - mu_F[k]) < 0.000001
        for k,kp in itertools.product(range(0,K),range(0,K)):   
            assert abs(expected_precision_F[k,k] - precision_F[k,k]) < 0.000001
        
        
###############################################################################
######## Individual updates for the parameters of the posterior of S ##########
###############################################################################
  
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0
   
F = 1./2. * numpy.ones((I,K))
S = 1./3. * numpy.ones((K,L))
G = 1./5. * numpy.ones((J,L))

lambdaF = 2 * numpy.ones((I,K))
lambdaS = 3 * numpy.ones((K,L))
lambdaG = 5 * numpy.ones((J,L))

tau = 3.
alpha = 2.
        
def test_individual_tau_S():
    ''' Nonnegative '''
    expected_tauS = tau*alpha*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        tauSkl = updates_Gibbs.individual_tau_S(R,M,tau,alpha,F,S,G,lambdaS[k,l],k,l,True)
        assert abs(tauSkl - expected_tauS[k,l]) < 0.000000000000001
        
    ''' Not nonnegative '''
    expected_tauS = lambdaS[k,l] + tau*alpha*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        tauSkl = updates_Gibbs.individual_tau_S(R,M,tau,alpha,F,S,G,lambdaS[k,l],k,l,False)
        assert abs(tauSkl - expected_tauS[k,l]) < 0.000000000000001

def test_individual_mu_S():
    ''' Nonnegative '''
    tauS = tau*alpha*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    expected_muS = 1./tauS * ( - lambdaS + alpha * tau * numpy.array([[12*23./300.,12*23./300.,12*23./300.,12*23./300.],[12*23./300.,12*23./300.,12*23./300.,12*23./300.]]) )
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        muSkl = updates_Gibbs.individual_mu_S(R,M,tau,alpha,F,S,G,lambdaS[k,l],k,l,tauS[k,l],True)
        assert abs(muSkl - expected_muS[k,l]) < 0.000000000000001
    
    ''' Not nonnegative '''
    tauS = lambdaS + tau*alpha*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    expected_muS = 1./tauS * ( alpha * tau * numpy.array([[12*23./300.,12*23./300.,12*23./300.,12*23./300.],[12*23./300.,12*23./300.,12*23./300.,12*23./300.]]) )
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        muSkl = updates_Gibbs.individual_mu_S(R,M,tau,alpha,F,S,G,lambdaS[k,l],k,l,tauS[k,l],False)
        assert abs(muSkl - expected_muS[k,l]) < 0.000000000000001
    
def test_individual_mu_tau_S():
    ''' Nonnegative '''
    expected_tauS = tau*alpha*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    expected_muS = 1./expected_tauS * ( - lambdaS + alpha * tau * numpy.array([[12*23./300.,12*23./300.,12*23./300.,12*23./300.],[12*23./300.,12*23./300.,12*23./300.,12*23./300.]]) )
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        muSkl, tauSkl = updates_Gibbs.individual_mu_tau_S(R,M,tau,alpha,F,S,G,lambdaS[k,l],k,l,True)
        assert abs(muSkl - expected_muS[k,l]) < 0.00000000000001
        assert abs(tauSkl - expected_tauS[k,l]) < 0.00000000000001
        
    ''' Not nonnegative '''
    expected_tauS = lambdaS + tau*alpha*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    expected_muS = 1./expected_tauS * ( alpha * tau * numpy.array([[12*23./300.,12*23./300.,12*23./300.,12*23./300.],[12*23./300.,12*23./300.,12*23./300.,12*23./300.]]) )
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        muSkl, tauSkl = updates_Gibbs.individual_mu_tau_S(R,M,tau,alpha,F,S,G,lambdaS[k,l],k,l,False)
        assert abs(muSkl - expected_muS[k,l]) < 0.00000000000001
        assert abs(tauSkl - expected_tauS[k,l]) < 0.00000000000001
        
        
###############################################################################
######### Row-wise updates for the parameters of the posterior of S ###########
############################################################################### 
        
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0
   
F = 1./2. * numpy.ones((I,K))
S = 1./3. * numpy.ones((K,L))
Gp = 1./5. * numpy.ones((J,L))
Gp[0,0] = 0.
Gp[1,1] = 0.1
Gp[2,2] = 0.2
Gp[2,3] = 0.3 # to make the precision non-singular

lambdaF = 2 * numpy.ones((I,K))
lambdaS = 3 * numpy.ones((K,L))
lambdaG = 5 * numpy.ones((J,L))

tau = 3.
alpha = 2.
        
def test_row_precision_S():
    ''' Nonnegative '''
    indices_mask = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if M[i,j]]
    
    for k in range(0,K):
        expected_precision_Sk = alpha * tau * numpy.array([[
            sum([F[i,k]**2 * Gp[j,l] * Gp[j,lp] for (i,j) in indices_mask])
        for lp in range(0,L)] for l in range(0,L)])    
        
        precision_Sk = updates_Gibbs.row_precision_S(R,M,tau,alpha,F,S,Gp,lambdaS[k,:],k,True)
        
        for l,lp in itertools.product(range(0,L),range(0,L)):
            assert abs(precision_Sk[l,lp] - expected_precision_Sk[l,lp]) < 0.0000001
    
    ''' Not nonnegative '''
    indices_mask = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if M[i,j]]
    
    for k in range(0,K):
        expected_precision_Sk = numpy.diag(lambdaS[k,:]) + alpha * tau * numpy.array([[
            sum([F[i,k]**2 * Gp[j,l] * Gp[j,lp] for (i,j) in indices_mask])
        for lp in range(0,L)] for l in range(0,L)])    
        
        precision_Sk = updates_Gibbs.row_precision_S(R,M,tau,alpha,F,S,Gp,lambdaS[k,:],k,False)
        
        for l,lp in itertools.product(range(0,L),range(0,L)):
            assert abs(precision_Sk[l,lp] - expected_precision_Sk[l,lp]) < 0.0000001
    

def test_row_mu_S():
    indices_mask = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if M[i,j]]
    
    ''' Nonnegative '''
    for k in range(0,K):   
        # Compute the expected mu
        precision_Sk = alpha * tau * numpy.array([[
            sum([F[i,k]**2 * Gp[j,l] * Gp[j,lp] for (i,j) in indices_mask])
        for lp in range(0,L)] for l in range(0,L)])
            
        sigma_Sk = numpy.linalg.inv(precision_Sk)
         
        expected_mu_Sk = numpy.array([
            -lambdaS[k,l] + alpha * tau * sum([
                (R[i,j] - sum([F[i,kp]*numpy.dot(S[kp,:],Gp[j,:]) for kp in range(0,K) if kp != k])) * F[i,k] * Gp[j,l]
            for (i,j) in indices_mask])
        for l in range(0,L)])
        expected_mu_Sk = numpy.dot(sigma_Sk,expected_mu_Sk)
        
        # Compare with the one from the updates
        mu_Sk = updates_Gibbs.row_mu_S(R,M,tau,alpha,F,S,Gp,lambdaS[k,:],precision_Sk,k,True)
    
        assert numpy.array_equal(mu_Sk,expected_mu_Sk)
          
    ''' Not nonnegative '''
    for k in range(0,K):   
        # Compute the expected mu
        precision_Sk = numpy.diag(lambdaS[k,:]) + alpha * tau * numpy.array([[
            sum([F[i,k]**2 * Gp[j,l] * Gp[j,lp] for (i,j) in indices_mask])
        for lp in range(0,L)] for l in range(0,L)])
        
        sigma_Sk = numpy.linalg.inv(precision_Sk)
         
        expected_mu_Sk = numpy.array([
            alpha * tau * sum([
                (R[i,j] - sum([F[i,kp]*numpy.dot(S[kp,:],Gp[j,:]) for kp in range(0,K) if kp != k])) * F[i,k] * Gp[j,l]
            for (i,j) in indices_mask])
        for l in range(0,L)])
        expected_mu_Sk = numpy.dot(sigma_Sk,expected_mu_Sk)
        
        # Compare with the one from the updates
        mu_Sk = updates_Gibbs.row_mu_S(R,M,tau,alpha,F,S,Gp,lambdaS[k,:],precision_Sk,k,False)
        assert numpy.array_equal(mu_Sk,expected_mu_Sk)

def test_row_mu_precision_S():
    indices_mask = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if M[i,j]]
    
    ''' Nonnegative '''
    for k in range(0,K):   
        # Compute the expected precision and mu
        expected_precision_Sk = numpy.zeros((L,L))
        G_outer_masked = numpy.array([numpy.dot((M[i] * Gp.T),(M[i] * Gp.T).T) for i in range(0,I)])
        expected_precision_Sk += tau * alpha * numpy.tensordot( F[:,k]**2, G_outer_masked, axes=1 )
            
        sigma_Sk = numpy.linalg.inv(expected_precision_Sk)
         
        expected_mu_Sk = numpy.array([
            -lambdaS[k,l] + alpha * tau * sum([
                (R[i,j] - sum([F[i,kp]*numpy.dot(S[kp,:],Gp[j,:]) for kp in range(0,K) if kp != k])) * F[i,k] * Gp[j,l]
            for (i,j) in indices_mask])
        for l in range(0,L)])
        expected_mu_Sk = numpy.dot(sigma_Sk,expected_mu_Sk)
        
        # Compare with the one from the updates
        mu_Sk, precision_Sk = updates_Gibbs.row_mu_precision_S(R,M,tau,alpha,F,S,Gp,lambdaS[k,:],k,True)
        for l,lp in itertools.product(range(0,L),range(0,L)):
            assert abs(precision_Sk[l,lp] - expected_precision_Sk[l,lp]) < 0.0000001
        for l in range(0,L):
            assert abs(mu_Sk[l] - expected_mu_Sk[l]) < 0.00000001
          
    ''' Not nonnegative '''
    for k in range(0,K):   
        # Compute the expected mu
        expected_precision_Sk = numpy.diag(lambdaS[k,:])
        G_outer_masked = numpy.array([numpy.dot((M[i] * Gp.T),(M[i] * Gp.T).T) for i in range(0,I)])
        expected_precision_Sk += tau * alpha * numpy.tensordot( F[:,k]**2, G_outer_masked, axes=1 )
        
        sigma_Sk = numpy.linalg.inv(expected_precision_Sk)
         
        expected_mu_Sk = numpy.array([
            alpha * tau * sum([
                (R[i,j] - sum([F[i,kp]*numpy.dot(S[kp,:],Gp[j,:]) for kp in range(0,K) if kp != k])) * F[i,k] * Gp[j,l]
            for (i,j) in indices_mask])
        for l in range(0,L)])
        expected_mu_Sk = numpy.dot(sigma_Sk,expected_mu_Sk)
        
        # Compare with the one from the updates
        mu_Sk, precision_Sk = updates_Gibbs.row_mu_precision_S(R,M,tau,alpha,F,S,Gp,lambdaS[k,:],k,False)
        for l,lp in itertools.product(range(0,L),range(0,L)):
            assert abs(precision_Sk[l,lp] - expected_precision_Sk[l,lp]) < 0.0000001
        for l in range(0,L):
            assert abs(mu_Sk[l] - expected_mu_Sk[l]) < 0.00000001
