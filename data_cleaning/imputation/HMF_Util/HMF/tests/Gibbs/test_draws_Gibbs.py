'''
Unit tests for the methods in draws_Gibbs.py.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import HMF.code.Gibbs.draws_Gibbs as draws_Gibbs

import numpy, itertools



###############################################################################
########################## Parameters for the tests ###########################
###############################################################################

I,J,K,L = 5,3,2,4
R = numpy.ones((I,J))/100.
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0

U = 1./2. * numpy.ones((I,K))
V = 1./3. * numpy.ones((J,K))
lambdaU = 2 * numpy.ones((I,K))
lambdaV = 2 * numpy.ones((J,K))

F = 1./2. * numpy.ones((I,K))
S = 100000. * numpy.ones((K,L)) # to force negative values
G = 1./5. * numpy.ones((J,L))
lambdaF = 2 * numpy.ones((I,K))
lambdaS = 3 * numpy.ones((K,L))
lambdaG = 5 * numpy.ones((J,L))

F[0,0] = 5.
F[1,1] = 6.# to make the precision nonsingular
G[0,0] = 7.
G[1,1] = 8. 
G[2,2] = 9. # to make the precision nonsingular
S[0,0] = 10.
S[1,1] = 11. 
S[1,2] = 12. # to make the precision nonsingular
        
Is,Ks = 5,2
Cs = numpy.ones((Is,Is))
Ms = numpy.ones((Is,Is))
Ms[0,0], Ms[2,2], Ms[3,1] = 0, 0, 0

Fs = 1./2. * numpy.ones((Is,Ks))
Ss = 1./3. * numpy.ones((Ks,Ks))

lambdaFs = 2 * numpy.ones((Is,Ks))
lambdaSs = 3 * numpy.ones((Ks,Ks))

tau = 3.
alpha = 2.
alphatau, betatau = 3, 1
alpha0, beta0 = 6, 7
alphaS, betaS = 4., 5.

R_F, C_F, D_F = [(R,M,F,S,G,tau,alpha)], [(Cs,Ms,F,Ss,tau,alpha)], [(R,M,F,V,tau,alpha)]


###############################################################################
##################### Draws for the noise parameters tau ######################
###############################################################################

def test_draw_tau():
    # MF case
    for i in range(0,100):
        new_tau = draws_Gibbs.draw_tau(alphatau,betatau,alpha,R,M,U,V)
        assert new_tau >= 0.
    
    # MTF case
    for i in range(0,100):
        new_tau = draws_Gibbs.draw_tau(alphatau,betatau,alpha,R,M,F,G,S)
        assert new_tau >= 0.
        
        
###############################################################################
#################### Draws for the ARD parameters lambdat #####################
###############################################################################

F_list = [
    (1*numpy.ones((I,K)),True),
    (2*numpy.ones((I,K)),True),
    (3*numpy.ones((J,K)),False)
]        

def test_draw_lambdat():
    for i in range(0,100):
        lambdat = draws_Gibbs.draw_lambdat(alpha0,beta0,F_list,K)
        assert lambdat.shape == (K,)
        for k in range(0,K):
            assert lambdat[k] > 0.
            
        
###############################################################################
######################## Draws for lambdan and lambdam ########################
############################################################################### 

def test_draw_lambdaS():
    for i in range(0,100):
        lambdan = draws_Gibbs.draw_lambdaS(alphaS,betaS,S,False)
        assert lambdan.shape == (K,L)
        for k,l in itertools.product(range(0,K),range(0,L)):
            assert lambdan[k,l] > 0.
            
            
###############################################################################
########################### Draws for the F matrix ############################
###############################################################################

def test_draw_F():
    copy_F = numpy.copy(F)    
    
    ''' Nonnegative, columns '''
    new_F = draws_Gibbs.draw_F(
        R=R_F,
        C=C_F,
        D=D_F,
        lambdaF=lambdaF[0],
        nonnegative=True,
        rows=False
    )
    for i,k in itertools.product(range(0,I),range(0,K)):
        assert new_F[i,k] != copy_F[i,k]
        assert new_F[i,k] > 0.
        assert numpy.array_equal(R_F[0][2],new_F)
        assert numpy.array_equal(C_F[0][2],new_F)
        assert numpy.array_equal(D_F[0][2],new_F)
    
    ''' Nonnegative, rows '''
    #TODO: implement
    
    ''' Not nonnegative, columns '''
    new_F = draws_Gibbs.draw_F(
        R=R_F,
        C=C_F,
        D=D_F,
        lambdaF=lambdaF[0],
        nonnegative=False,
        rows=False
    )
    for i,k in itertools.product(range(0,I),range(0,K)):
        assert new_F[i,k] != copy_F[i,k]
        assert numpy.array_equal(R_F[0][2],new_F)
        assert numpy.array_equal(C_F[0][2],new_F)
        assert numpy.array_equal(D_F[0][2],new_F)
    assert any([True if new_F[i,k] < 0. else False for i,k in itertools.product(range(0,I),range(0,K))])
    
    
    ''' Not nonnegative, rows '''
    new_F = draws_Gibbs.draw_F(
        R=R_F,
        C=C_F,
        D=D_F,
        lambdaF=lambdaF[0],
        nonnegative=False,
        rows=True
    )
    for i,k in itertools.product(range(0,I),range(0,K)):
        assert new_F[i,k] != copy_F[i,k]
        assert numpy.array_equal(R_F[0][2],new_F)
        assert numpy.array_equal(C_F[0][2],new_F)
        assert numpy.array_equal(D_F[0][2],new_F)
    assert any([True if new_F[i,k] < 0. else False for i,k in itertools.product(range(0,I),range(0,K))])
    
    
###############################################################################
########################### Draws for the S matrix ############################
###############################################################################

def test_draw_S():
    copy_S = numpy.copy(S)    
    
    ''' Nonnegative, columns '''
    new_S = draws_Gibbs.draw_S(
        dataset=R,
        mask=M,
        tau=tau,
        alpha=alpha,
        F=F,
        S=S,
        G=G,
        lambdaS=lambdaS,
        nonnegative=True,
        rows=False,
        tensor_decomposition=False,
    )
    for k,l in itertools.product(range(0,K),range(0,L)):
        assert new_S[k,l] != copy_S[k,l]
        assert new_S[k,l] > 0.
    
    ''' Nonnegative, rows '''
    #TODO: implement
    
    ''' Not nonnegative, columns '''
    new_S = draws_Gibbs.draw_S(
        dataset=R,
        mask=M,
        tau=tau,
        alpha=alpha,
        F=F,
        S=S,
        G=G,
        lambdaS=lambdaS,
        nonnegative=False,
        rows=False,
        tensor_decomposition=False,
    )
    for k,l in itertools.product(range(0,K),range(0,L)):
        assert new_S[k,l] != copy_S[k,l]
    assert any([True if new_S[k,l] < 0. else False for k,l in itertools.product(range(0,K),range(0,L))])
    
    
    ''' Not nonnegative, rows '''
    new_S = draws_Gibbs.draw_S(
        dataset=R,
        mask=M,
        tau=tau,
        alpha=alpha,
        F=F,
        S=S,
        G=G,
        lambdaS=lambdaS,
        nonnegative=False,
        rows=True,
        tensor_decomposition=False,
    )
    for k,l in itertools.product(range(0,K),range(0,L)):
        assert new_S[k,l] != copy_S[k,l]
    assert any([True if new_S[k,l] < 0. else False for k,l in itertools.product(range(0,K),range(0,L))])
    
    
    ''' Finally, test tensor decomposition (off-diagonal all zero). '''
    ''' Nonnegative, columns '''
    new_S = draws_Gibbs.draw_S(
        dataset=R,
        mask=M,
        tau=tau,
        alpha=alpha,
        F=F,
        S=S,
        G=G,
        lambdaS=lambdaS,
        nonnegative=True,
        rows=False,
        tensor_decomposition=True,
    )
    for k,l in itertools.product(range(0,K),range(0,L)):
        if k != l:
            assert new_S[k,l] == 0.
        else:
            assert new_S[k,l] != copy_S[k,l]
            assert new_S[k,l] > 0.