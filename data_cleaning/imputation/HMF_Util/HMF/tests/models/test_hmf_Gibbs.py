"""
Tests for the HMF Gibbs sampler.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

from HMF.code.models.hmf_Gibbs import HMF_Gibbs

import numpy, math, pytest, itertools


""" Test constructor """
def test_init():
    """
    We need to test the following cases:
    1. Dataset R relates same two entity types
    2. Rn and Cm are not 2-dimensional matrices
    3. Rn and Mn are of different sizes
    4. Cm and Mm are of different sizes
    5. Cm is not a square matrix
    6. R1 and R2 both relate E but have different no. of entities
    7. R and C both relate E but have different no. of entities
    8. An entity has no observed datapoints at all
    9. K does not have an entry for each entity
    10. Finally, we need to test whether all variables are correctly initialised
    """
    
    E0, E1, E2 = 'entity0','entity1',1337
    I0, I1, I2 = 10,9,8
    K0, K1, K2 = 3,2,1
    J0 = 4
    N, M, L, T = 3, 2, 1, 3
    
    R0 = numpy.ones((I0,I1)) # relates E0, E1
    R1 = numpy.ones((I0,I1)) # relates E0, E1
    R2 = numpy.ones((I1,I2)) # relates E1, E2
    C0 = numpy.ones((I0,I0)) # relates E0
    C1 = numpy.ones((I2,I2)) # relates E2
    D0 = numpy.ones((I2,J0)) # relates E2
    
    Mn0 = numpy.ones((I0,I1))
    Mn1 = numpy.ones((I0,I1))
    Mn2 = numpy.ones((I1,I2))
    Mm0 = numpy.ones((I0,I0))
    Mm1 = numpy.ones((I2,I2))
    Ml0 = numpy.ones((I2,J0))
    
    size_Omegan = [I0*I1,I0*I1,I1*I2]
    size_Omegam = [I0*(I0-1),I2*(I2-1)]
    size_Omegal = [I2*J0]
    
    alphan = [11.,12.,13.]
    alpham = [14.,15.]
    alphal = [16.]
    
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[1])]
    D = [(D0,Ml0,E2,alphal[0])]
    E = [E0,E1,E2]
    K = {E0:K0,E1:K1,E2:K2}
    I = {E0:I0,E1:I1,E2:I2}
    J = [J0]
    
    U1t = {'entity0':[0,1], 'entity1':[2], 1337:[] }
    U2t = {'entity0':[], 'entity1':[0,1], 1337:[2] }
    Vt = {'entity0':[0], 'entity1':[], 1337:[1] }
    Wt = {'entity0':[], 'entity1':[], 1337:[0]}
    E_per_Rn = [(E0,E1),(E0,E1),(E1,E2)]
    E_per_Cm = [E0,E2]
    E_per_Dl = [E2]
    
    alphatau, betatau = 1., 2.
    alpha0, beta0 = 6., 7.
    lambdaF, lambdaG = 3., 8.
    lambdaSn, lambdaSm = 4., 5.
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphatau':alphatau, 'betatau':betatau, 
               'lambdaF':lambdaF, 'lambdaG':lambdaG, 'lambdaSn':lambdaSn, 'lambdaSm':lambdaSm }
    settings = { 'priorF' : 'normal', 'priorG' : 'exponential', 'priorSn' : 'normal', 'priorSm' : 'exponential',
                 'orderF' : 'rows', 'orderG' : 'columns', 'orderSn' : 'individual', 'orderSm' : 'rows', 
                 'ARD' : True, 'element_sparsity' : True, }
    
    ''' 1. Dataset R relates same two entity types '''
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E1,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    with pytest.raises(AssertionError) as error:
        HMF_Gibbs(R,C,D,K,settings,priors)
    assert str(error.value) == "Gave same entity type for R1: entity1."
    
    ''' 2. Rn and Cm are not 2-dimensional matrices '''
    R1 = numpy.ones(I0)
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    with pytest.raises(AssertionError) as error:
        HMF_Gibbs(R,C,D,K,settings,priors)
    assert str(error.value) == "R1 is not 2-dimensional, but instead 1-dimensional."
    R1 = numpy.ones((I0,I1))
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
        
    C0 = numpy.ones((I0,I1,I2))
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[1])]
    with pytest.raises(AssertionError) as error:
        HMF_Gibbs(R,C,D,K,settings,priors)
    assert str(error.value) == "C0 is not 2-dimensional, but instead 3-dimensional."
    C0 = numpy.ones((I0,I0))
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[1])]
    
    ''' 3. Rn and Mn are of different sizes '''
    R2 = numpy.ones((I1,I2))
    Mn2 = numpy.ones((I0,I1))
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[1])]
    with pytest.raises(AssertionError) as error:
        HMF_Gibbs(R,C,D,K,settings,priors)
    assert str(error.value) == "Different shapes for R2 and M2: (9, 8) and (10, 9)."    
    R2 = numpy.ones((I1,I2))
    Mn2 = numpy.ones((I1,I2))
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    
    ''' 4. Cm and Mm are of different sizes '''
    C1 = numpy.ones((I2,I2))
    Mm1 = numpy.ones((I1,I1))
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[1])]
    with pytest.raises(AssertionError) as error:
        HMF_Gibbs(R,C,D,K,settings,priors)
    assert str(error.value) == "Different shapes for C1 and M1: (8, 8) and (9, 9)."
    C1 = numpy.ones((I2,I2))
    Mm1 = numpy.ones((I2,I2))
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[1])]
    
    ''' 5. Cm is not a square matrix '''
    C0 = numpy.ones((I1,I2))
    Mm0 = numpy.ones((I1,I2))
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[1])]
    with pytest.raises(AssertionError) as error:
        HMF_Gibbs(R,C,D,K,settings,priors)
    assert str(error.value) == "C0 is not a square matrix: (9, 8)."
    C0 = numpy.ones((I0,I0))
    Mm0 = numpy.ones((I0,I0))
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[1])]
    
    ''' 6. R1 and R2 both relate E but have different no. of entities '''
    R2 = numpy.ones((I1+1,I2))
    Mn2 = numpy.ones((I1+1,I2))
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    with pytest.raises(AssertionError) as error:
        HMF_Gibbs(R,C,D,K,settings,priors)
    assert str(error.value) == "Different number of rows (10) in R2 for entity type entity1 than before (9)!"
    R2 = numpy.ones((I1,I2))
    Mn2 = numpy.ones((I1,I2))
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    
    ''' 7. R and C both relate E but have different no. of entities '''
    R2 = numpy.ones((I1,I2+1))
    Mn2 = numpy.ones((I1,I2+1))
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    with pytest.raises(AssertionError) as error:
        HMF_Gibbs(R,C,D,K,settings,priors)
    assert str(error.value) == "Different number of rows (8) in C1 for entity type 1337 than before (9)!"
    R2 = numpy.ones((I1,I2))
    Mn2 = numpy.ones((I1,I2))
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    
    ''' 8. An entity has no observed datapoints at all '''
    Mn0[:,1] = numpy.zeros(I0)
    Mn1[:,1] = numpy.zeros(I0)
    Mn2[1,:] = numpy.zeros(I2)
    ''' Concurrently also test not getting an error for entity0 '''
    Mn0[2,:] = numpy.zeros(I1) 
    Mn1[2,:] = numpy.zeros(I1) 
    Mm0[2,:] = numpy.zeros(I0) 
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[0])]
    with pytest.raises(AssertionError) as error:
        HMF_Gibbs(R,C,D,K,settings,priors)
    assert str(error.value) == "No observed datapoints in any dataset for entity 1 of type entity1."
    Mn0 = numpy.ones((I0,I1))
    Mn1 = numpy.ones((I0,I1))
    Mn2 = numpy.ones((I1,I2))
    Mm0 = numpy.ones((I0,I0))
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[1])]
    
    ''' 9. K does not have an entry for each entity '''
    K = {E0:K0,E2:K2}
    with pytest.raises(AssertionError) as error:
        HMF_Gibbs(R,C,D,K,settings,priors)
    assert str(error.value) == "Did not get an entry for entity entity1 in K = {1337: 1, 'entity0': 3}."
    K = {E0:K0,E1:K1,E2:K2}
    
    ''' 10. Finally, we need to test whether all variables are correctly initialised '''
    HMF = HMF_Gibbs(R,C,D,K,settings,priors)
    
    assert numpy.array_equal(HMF.all_E,E)
    for R,Rtrue in zip(HMF.all_Rn,[R0,R1,R2]):
        assert numpy.array_equal(R,Rtrue)
    for Mn,Mntrue in zip(HMF.all_Mn,[Mn0,Mn1,Mn2]):
        assert numpy.array_equal(Mn,Mntrue)
    for C,Ctrue in zip(HMF.all_Cm,[C0,C1]):
        assert numpy.array_equal(C,Ctrue)
    for Mm,Mmtrue in zip(HMF.all_Mm,[Mm0,Mm1]):
        assert numpy.array_equal(Mm,Mmtrue)
    for Dl,Dltrue in zip(HMF.all_Dl,[D0]):
        assert numpy.array_equal(Dl,Dltrue)
        
    assert HMF.size_Omegan == size_Omegan
    assert HMF.size_Omegam == size_Omegam
    assert HMF.size_Omegal == size_Omegal
        
    assert HMF.E_per_Rn == E_per_Rn
    assert HMF.E_per_Cm == E_per_Cm
    assert HMF.E_per_Dl == E_per_Dl
    
    assert HMF.all_alphan == alphan
    assert HMF.all_alpham == alpham
    assert HMF.all_alphal == alphal
    
    assert numpy.array_equal(HMF.K,K)
    assert HMF.I == I
    assert HMF.J == J
    assert HMF.N == N
    assert HMF.M == M
    assert HMF.L == L
    assert HMF.T == T
    
    assert HMF.U1t == U1t
    assert HMF.U2t == U2t
    assert HMF.Vt == Vt
    assert HMF.Wt == Wt
    
    assert HMF.all_Ft == { 'entity0':[], 'entity1':[], 1337:[] }
    assert HMF.all_Sn == []
    assert HMF.all_Sm == []
    assert HMF.all_Gl == []
    
    assert HMF.all_taun == []
    assert HMF.all_taum == []
    assert HMF.all_taul == []
    
    assert HMF.alpha0 == alpha0
    assert HMF.beta0 == beta0
    assert HMF.alphatau == alphatau
    assert HMF.betatau == betatau
    assert HMF.lambdaF == lambdaF
    assert HMF.lambdaG == lambdaG
    assert HMF.lambdaSn == lambdaSn
    assert HMF.lambdaSm == lambdaSm
    
    # { 'priorF' : 'normal', 'priorG' : 'exponential', 'priorSn' : 'normal', 'priorSm' : 'exponential', 
    #   'orderF' : 'rows', 'orderG' : 'columns', 'orderSn' : 'individual', 'orderSm' : 'rows', 'ARD' : True }
    assert HMF.prior_F == 'normal'
    assert HMF.prior_G == ['exponential']
    assert HMF.prior_Sn == ['normal','normal','normal']
    assert HMF.prior_Sm == ['exponential','exponential']
    assert HMF.order_F == 'rows'
    assert HMF.order_G == ['columns']
    assert HMF.order_Sn == ['individual','individual','individual']
    assert HMF.order_Sm == ['rows','rows']
    assert HMF.ARD == True
    assert HMF.element_sparsity == True
    
    assert HMF.rows_F == True
    assert HMF.rows_G == [False]
    assert HMF.rows_Sn == [False,False,False]
    assert HMF.rows_Sm == [True,True]
    assert HMF.nonnegative_F == False
    assert HMF.nonnegative_G == [True]
    assert HMF.nonnegative_Sn == [False,False,False]
    assert HMF.nonnegative_Sm == [True,True]
    

""" Test initialing parameters """
def test_initialise():
    E0, E1, E2 = 'entity0','entity1',1337
    I0, I1, I2 = 10,9,8
    K0, K1, K2 = 3,2,1
    J0 = 4
    N, M, L, T = 3, 2, 1, 3
    
    R0 = numpy.ones((I0,I1)) # relates E0, E1
    R1 = numpy.ones((I0,I1)) # relates E0, E1
    R2 = numpy.ones((I1,I2)) # relates E1, E2
    C0 = numpy.ones((I0,I0)) # relates E0
    C1 = numpy.ones((I2,I2)) # relates E2
    D0 = numpy.ones((I2,J0)) # relates E2
    
    Mn0 = numpy.ones((I0,I1))
    Mn1 = numpy.ones((I0,I1))
    Mn2 = numpy.ones((I1,I2))
    Mm0 = numpy.ones((I0,I0))
    Mm1 = numpy.ones((I2,I2))
    Ml0 = numpy.ones((I2,J0))
    
    #size_Omegan = [I0*I1,I0*I1,I1*I2]
    #size_Omegam = [I0*(I0-1),I2*(I2-1)]
    #size_Omegal = [I2*J0]
    
    alphan = [11.,12.,13.]
    alpham = [14.,15.]
    alphal = [16.]
    
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[1])]
    D = [(D0,Ml0,E2,alphal[0])]
    E = [E0,E1,E2]
    K = {E0:K0,E1:K1,E2:K2}
    I = {E0:I0,E1:I1,E2:I2}
    J = [J0]
    
    #U1t = {'entity0':[0,1], 'entity1':[2], 1337:[] }
    #U2t = {'entity0':[], 'entity1':[0,1], 1337:[2] }
    #Vt = {'entity0':[0], 'entity1':[], 1337:[1] }
    #Wt = {'entity0':[], 'entity1':[], 1337:[0]}
    E_per_Rn = [(E0,E1),(E0,E1),(E1,E2)]
    E_per_Cm = [E0,E2]
    E_per_Dl = [E2]
    
    alphatau, betatau = 1., 2.
    alpha0, beta0 = 6., 7.
    alphaS, betaS = 9., 10.
    lambdaF, lambdaG = 3., 8.
    lambdaSn, lambdaSm = 4., 5.
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphaS':alphaS, 'betaS':betaS, 'alphatau':alphatau, 'betatau':betatau, 
               'lambdaF':lambdaF, 'lambdaG':lambdaG, 'lambdaSn':lambdaSn, 'lambdaSm':lambdaSm }
               
    """
    We need to test the following cases:
    - F ~ Exp or ~ N
    - G ~ Exp or ~ N
    - S ~ Exp or ~ N
    - ARD or no ARD
    - F init random, exp, kmeans
    - G init random, exp, least
    - S init random, exp, least
    - lambdat init random, exp
    - tau init random, exp
    """
    
    ''' F Exp, G Exp, S Exp, ARD, no element-wise sparsity. F exp, G exp, S exp, lambdat exp, tau exp. '''
    settings = { 'priorF' : 'exponential', 'priorG' : 'exponential', 'priorSn' : 'exponential', 'priorSm' : 'exponential',
                 'orderF' : 'rows', 'orderG' : 'columns', 'orderSn' : 'individual', 'orderSm' : 'individual',
                 'ARD' : True, 'element_sparsity': True }    
    init = { 'F' : 'exp', 'G' : 'exp', 'Sn' : 'exp', 'Sm' : 'exp', 'lambdat' : 'exp', 'lambdaS': 'exp', 'tau' : 'exp'}
    HMF = HMF_Gibbs(R,C,D,K,settings,priors)
    HMF.initialise(init)
    
    for E1 in E:
        for k in range(0,K[E1]):
            assert HMF.all_lambdat[E1][k] == alpha0 / float(beta0)
        for i,k in itertools.product(xrange(0,I[E1]),xrange(0,K[E1])):
            assert HMF.all_Ft[E1][i,k] == 1./HMF.all_lambdat[E1][k]
            
    expected_all_taun = [0.015369654419961557,0.015367151516936775,0.2442062783472021]
    for n in range(0,N):
        E1,E2 = E_per_Rn[n]
        for k,l in itertools.product(xrange(0,K[E1]),xrange(0,K[E2])):
            expected_lambdan_kl = alphaS / float(betaS)
            assert HMF.all_lambdan[n][k,l] == expected_lambdan_kl
            assert HMF.all_Sn[n][k,l] == 1./expected_lambdan_kl
        assert abs(HMF.all_taun[n] - expected_all_taun[n]) < 0.0000000001
            
    expected_all_taum = [0.0062975762814580696,3.7505835292008993]
    for m in range(0,M):
        E1 = E_per_Cm[m]
        for k,l in itertools.product(xrange(0,K[E1]),xrange(0,K[E1])):
            expected_lambdam_kl = alphaS / float(betaS)
            assert HMF.all_lambdam[m][k,l] == expected_lambdam_kl
            assert HMF.all_Sm[m][k,l] == 1./expected_lambdam_kl
        assert abs(HMF.all_taum[m] - expected_all_taum[m]) < 0.000000001
            
    expected_all_taul = [7.2634333565945441]
    for l in range(0,L):
        E1 = E_per_Dl[l]
        for j,k in itertools.product(xrange(0,J[l]),xrange(0,K[E1])):
            assert HMF.all_Gl[l][j,k] == 1./HMF.all_lambdat[E1][k]
        assert abs(HMF.all_taul[l] - expected_all_taul[l]) < 0.00000001
            
    ''' F Exp, G Exp, S N, no ARD, element-wise sparsity. F random, G exp, Sn exp, Sm random, tau random. '''
    settings = { 'priorF' : 'exponential', 'priorG' : 'exponential', 'priorSn' : 'normal', 'priorSm' : 'normal',
                 'orderF' : 'columns', 'orderG' : 'rows', 'orderSn' : 'rows', 'orderSm' : 'rows',
                 'ARD' : False, 'element_sparsity' : False }    
    init = { 'F' : 'random', 'G' : 'exp', 'Sn' : 'exp', 'Sm' : 'random', 'lambdaS': 'exp', 'tau' : 'random' }
    HMF = HMF_Gibbs(R,C,D,K,settings,priors)
    HMF.initialise(init)
    
    for E1 in E:
        for i,k in itertools.product(xrange(0,I[E1]),xrange(0,K[E1])):
            assert HMF.all_Ft[E1][i,k] != 1./lambdaF
            
    for n in range(0,N):
        E1,E2 = E_per_Rn[n]
        for k,l in itertools.product(xrange(0,K[E1]),xrange(0,K[E2])):
            assert HMF.all_Sn[n][k,l] == 0.01
        assert HMF.all_taun[n] >= 0.
            
    for m in range(0,M):
        E1 = E_per_Cm[m]
        for k,l in itertools.product(xrange(0,K[E1]),xrange(0,K[E1])):
            assert HMF.all_Sm[m][k,l] != 0.
        assert HMF.all_taum[m] >= 0.
            
    for l in range(0,L):
        E1 = E_per_Dl[l]
        for j,k in itertools.product(xrange(0,J[l]),xrange(0,K[E1])):
            assert HMF.all_Gl[l][j,k] == 1./lambdaG
        assert HMF.all_taul[l] >= 0.
            
    ''' F N, G N, Sn Exp, Sm N, ARD, no element-wise sparsity. F kmeans, G exp, S random, lambdat random, tau random. '''
    settings = { 'priorF' : 'normal', 'priorG' : 'normal', 'priorSn' : 'exponential', 'priorSm' : 'normal',
                 'ARD' : True, 'orderF' : 'columns', 'orderG' : 'rows', 'orderSn' : 'rows', 'orderSm' : 'rows' }    
    init = { 'F' : 'kmeans', 'G' : 'exp', 'Sn' : 'random', 'Sm' : 'random', 'lambdat' : 'random', 'tau' : 'random' }
    HMF = HMF_Gibbs(R,C,D,K,settings,priors)
    HMF.initialise(init)
    
    for E1 in E:
        for k in range(0,K[E1]):
            assert HMF.all_lambdat[E1][k] >= 0.
        for i,k in itertools.product(xrange(0,I[E1]),xrange(0,K[E1])):
            assert HMF.all_Ft[E1][i,k] == 0.2 or HMF.all_Ft[E1][i,k] == 1.2
            
    for n in range(0,N):
        E1,E2 = E_per_Rn[n]
        for k,l in itertools.product(xrange(0,K[E1]),xrange(0,K[E2])):
            assert HMF.all_Sn[n][k,l] >= 0.
        assert HMF.all_taun[n] >= 0.
            
    expected_all_taum = [0.47612886531245974,1.7230629295737439]
    for m in range(0,M):
        E1 = E_per_Cm[m]
        for k,l in itertools.product(xrange(0,K[E1]),xrange(0,K[E1])):
            assert HMF.all_Sm[m][k,l] != 0.
        assert HMF.all_taum[m] >= 0.
            
    expected_all_taul = [4.1601208459214458]
    for l in range(0,L):
        E1 = E_per_Dl[l]
        for j,k in itertools.product(xrange(0,J[l]),xrange(0,K[E1])):
            assert HMF.all_Gl[l][j,k] == 0.01
        assert HMF.all_taul[l] >= 0.
            
    ''' F Exp, G N, S N, no ARD, no element-wise sparsity. F kmeans, G least, S least, lambdat random, tau random. '''
    settings = { 'priorF' : 'exponential', 'priorG' : 'normal', 'priorSn' : 'normal', 'priorSm' : 'normal', 
                 'orderF' : 'columns', 'orderG' : 'rows', 'orderSn' : 'rows', 'orderSm' : 'rows',
                 'ARD' : False, 'element_sparsity' : False }    
    init = { 'F': 'kmeans', 'G': 'least', 'Sn': 'least', 'Sm': 'least', 'lambdat': 'random', 'lambdaS': 'exp', 'tau': 'random' }
    HMF = HMF_Gibbs(R,C,D,K,settings,priors)
    HMF.initialise(init)
    
    for E1 in E:
        for i,k in itertools.product(xrange(0,I[E1]),xrange(0,K[E1])):
            assert HMF.all_Ft[E1][i,k] == 0.2 or HMF.all_Ft[E1][i,k] == 1.2
            
    for n in range(0,N):
        E1,E2 = E_per_Rn[n]
        for k,l in itertools.product(xrange(0,K[E1]),xrange(0,K[E2])):
            assert HMF.all_Sn[n][k,l] != 0.
        assert HMF.all_taun[n] >= 0.
            
    expected_all_taum = [0.47612886531245974,1.7230629295737439]
    for m in range(0,M):
        E1 = E_per_Cm[m]
        for k,l in itertools.product(xrange(0,K[E1]),xrange(0,K[E1])):
            assert HMF.all_Sm[m][k,l] != 0.
        assert HMF.all_taum[m] >= 0.
            
    expected_all_taul = [4.1601208459214458]
    for l in range(0,L):
        E1 = E_per_Dl[l]
        for j,k in itertools.product(xrange(0,J[l]),xrange(0,K[E1])):
            assert HMF.all_Gl[l][j,k] != 1./lambdaG
        assert HMF.all_taul[l] >= 0.
    
    
""" Test some iterations, and that the values have changed in the F, S. """
def test_run():
    ''' Settings '''
    E0, E1, E2 = 'entity0','entity1',1337
    I0, I1, I2 = 10,9,8
    K0, K1, K2 = 3,2,1
    J0 = 4
    N, M, L, T = 3, 2, 1, 3
    
    R0 = numpy.ones((I0,I1)) # relates E0, E1
    R1 = numpy.ones((I0,I1)) # relates E0, E1
    R2 = numpy.ones((I1,I2)) # relates E1, E2
    C0 = numpy.ones((I0,I0)) # relates E0
    C1 = numpy.ones((I2,I2)) # relates E2
    D0 = numpy.ones((I2,J0)) # relates E2
    
    Mn0 = numpy.ones((I0,I1))
    Mn1 = numpy.ones((I0,I1))
    Mn2 = numpy.ones((I1,I2))
    Mm0 = numpy.ones((I0,I0))
    Mm1 = numpy.ones((I2,I2))
    Ml0 = numpy.ones((I2,J0))
    
    #size_Omegan = [I0*I1,I0*I1,I1*I2]
    #size_Omegam = [I0*(I0-1),I2*(I2-1)]
    #size_Omegal = [I2*J0]
    
    alphan = [11.,12.,13.]
    alpham = [14.,15.]
    alphal = [16.]
    
    R = [(R0,Mn0,E0,E1,alphan[0]),(R1,Mn1,E0,E1,alphan[1]),(R2,Mn2,E1,E2,alphan[2])]
    C = [(C0,Mm0,E0,alpham[0]),(C1,Mm1,E2,alpham[1])]
    D = [(D0,Ml0,E2,alphal[0])]
    E = [E0,E1,E2]
    K = {E0:K0,E1:K1,E2:K2}
    I = {E0:I0,E1:I1,E2:I2}
    J = [J0]
    
    #U1t = {'entity0':[0,1], 'entity1':[2], 1337:[] }
    #U2t = {'entity0':[], 'entity1':[0,1], 1337:[2] }
    #Vt = {'entity0':[0], 'entity1':[], 1337:[1] }
    #Wt = {'entity0':[], 'entity1':[], 1337:[0]}
    
    E_per_Rn = [(E0,E1),(E0,E1),(E1,E2)]
    E_per_Cm = [E0,E2]
    E_per_Dl = [E2]
    
    alphatau, betatau = 1., 2.
    alpha0, beta0 = 6., 7.
    lambdaF, lambdaG = 3., 8.
    lambdaSn, lambdaSm = 4., 5.
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphatau':alphatau, 'betatau':betatau, 
               'lambdaF':lambdaF, 'lambdaG':lambdaG, 'lambdaSn':lambdaSn, 'lambdaSm':lambdaSm }
    settings = { 'priorF' : 'exponential', 'priorG' : 'normal', 'priorSn' : 'normal', 'priorSm' : 'normal',
                 'orderF' : 'columns', 'orderG' : 'rows', 'orderSn' : 'rows', 'orderSm' : 'rows',
                 'ARD' : True, 'element_sparsity': True }    
    init = { 'F': 'kmeans', 'G': 'least', 'Sn': 'least', 'Sm': 'least', 'lambdat': 'random', 'lambdaS': 'random', 'tau': 'random' }
    iterations = 10
    
    HMF = HMF_Gibbs(R,C,D,K,settings,priors)
    HMF.initialise(init)
    HMF.run(iterations)
    
    ''' Do size checks '''
    for E0 in E:
        assert len(HMF.iterations_all_Ft[E0]) == iterations
        assert len(HMF.iterations_all_lambdat[E0]) == iterations
    for n in range(0,N):
        assert len(HMF.iterations_all_lambdan[n]) == iterations
        assert len(HMF.iterations_all_Sn[n]) == iterations
        assert len(HMF.iterations_all_taun[n]) == iterations
    for m in range(0,M):
        assert len(HMF.iterations_all_lambdam[m]) == iterations
        assert len(HMF.iterations_all_Sm[m]) == iterations
        assert len(HMF.iterations_all_taum[m]) == iterations
    for l in range(0,L):
        assert len(HMF.iterations_all_Gl[l]) == iterations
        assert len(HMF.iterations_all_taul[l]) == iterations
    
    ''' Check whether values change each iteration '''
    for iteration in range(1,iterations):
        for E0 in E:
            for k in range(0,K[E0]):
                assert HMF.iterations_all_lambdat[E0][iteration][k] != HMF.iterations_all_lambdat[E0][iteration-1][k]
            for i,k in itertools.product(xrange(0,I[E0]),xrange(0,K[E0])):
                assert HMF.iterations_all_Ft[E0][iteration][i,k] != HMF.iterations_all_Ft[E0][iteration-1][i,k]
        for n in range(0,N):
            E0,E1 = E_per_Rn[n]
            for k,l in itertools.product(xrange(0,K[E0]),xrange(0,K[E1])):
                assert HMF.iterations_all_lambdan[n][iteration][k,l] != HMF.iterations_all_lambdan[n][iteration-1][k,l]
                assert HMF.iterations_all_Sn[n][iteration][k,l] != HMF.iterations_all_Sn[n][iteration-1][k,l]
            assert HMF.iterations_all_taun[n][iteration] != HMF.iterations_all_taun[n][iteration-1]
        for m in range(0,M):
            E0 = E_per_Cm[m]
            for k,l in itertools.product(xrange(0,K[E0]),xrange(0,K[E0])):
                assert HMF.iterations_all_lambdam[m][iteration][k,l] != HMF.iterations_all_lambdam[m][iteration-1][k,l]
                assert HMF.iterations_all_Sm[m][iteration][k,l] != HMF.iterations_all_Sm[m][iteration-1][k,l]
            assert HMF.iterations_all_taum[m][iteration] != HMF.iterations_all_taum[m][iteration-1]
        for l in range(0,l):
            E0 = E_per_Dl[l]
            for j,k in itertools.product(xrange(0,J[l]),xrange(0,K[E0])):
                assert HMF.iterations_all_Gl[l][iteration][j,k] != HMF.iterations_all_Dl[l][iteration-1][j,k]
            assert HMF.iterations_all_taul[l][iteration] != HMF.iterations_all_taul[l][iteration-1]
            
            
""" Test approximating the expectations for the F, S, G, lambda, tau """
def test_approx_expectation():
    iterations = 10
    burn_in = 2
    thinning = 3 # so index 2,5,8 -> m=3,m=6,m=9
    
    E = ['entity0','entity1']
    I = {E[0]:5, E[1]:3}
    K = {E[0]:2, E[1]:4}
    J = [6]
    
    iterations_all_Ft = {
        E[0] : [numpy.ones((I[E[0]],K[E[0]])) * 3*m**2 for m in range(1,10+1)],
        E[1] : [numpy.ones((I[E[1]],K[E[1]])) * 1*m**2 for m in range(1,10+1)]
    }
    iterations_all_lambdat = {
        E[0] : [numpy.ones(K[E[0]]) * 3*m**2 for m in range(1,10+1)],
        E[1] : [numpy.ones(K[E[1]]) * 1*m**2 for m in range(1,10+1)]
    }
    iterations_all_Sn = [[numpy.ones((K[E[0]],K[E[1]])) * 2*m**2 for m in range(1,10+1)]]
    iterations_all_lambdan = [[numpy.ones((K[E[0]],K[E[1]])) * 2*m**2 for m in range(1,10+1)]]
    iterations_all_taun = [[m**2 for m in range(1,10+1)]]
    iterations_all_Sm = [[numpy.ones((K[E[1]],K[E[1]])) * 2*m**2 * 2 for m in range(1,10+1)]]
    iterations_all_lambdam = [[numpy.ones((K[E[1]],K[E[1]])) * 2*m**2 * 2 for m in range(1,10+1)]]
    iterations_all_taum = [[m**2*2 for m in range(1,10+1)]]
    iterations_all_Gl = [[numpy.ones((J[0],K[E[1]])) * 2*m**2 * 3 for m in range(1,10+1)]]
    iterations_all_taul = [[m**2*3 for m in range(1,10+1)]]
    
    expected_exp_F0 = numpy.array([[9.+36.+81. for k in range(0,2)] for i in range(0,5)])
    expected_exp_F1 = numpy.array([[(9.+36.+81.)*(1./3.) for k in range(0,4)] for i in range(0,3)])
    expected_exp_lambda0 = numpy.array([9.+36.+81. for k in range(0,2)])
    expected_exp_lambda1 = numpy.array([(9.+36.+81.)*(1./3.) for k in range(0,4)])
    expected_exp_Sn = numpy.array([[(9.+36.+81.)*(2./3.) for l in range(0,4)] for k in range(0,2)])
    expected_exp_lambdan = numpy.array([[(9.+36.+81.)*(2./3.) for l in range(0,4)] for k in range(0,2)])
    expected_exp_taun = (9.+36.+81.)/3.
    expected_exp_Sm = numpy.array([[(18.+72.+162.)*(2./3.) for l in range(0,4)] for k in range(0,4)])
    expected_exp_lambdam = numpy.array([[(18.+72.+162.)*(2./3.) for l in range(0,4)] for k in range(0,4)])
    expected_exp_taum = (18.+72.+162.)/3.
    expected_exp_Gl = numpy.array([[(27.+108.+243.)*(2./3.) for k in range(0,4)] for j in range(0,6)])
    expected_exp_taul = (27.+108.+243.)/3.
    
    R0, M0 = numpy.ones((I[E[0]],I[E[1]])), numpy.ones((I[E[0]],I[E[1]]))
    C0, M1 = numpy.ones((I[E[1]],I[E[1]])), numpy.ones((I[E[1]],I[E[1]]))
    D0, M2 = numpy.ones((I[E[1]],J[0])), numpy.ones((I[E[1]],J[0]))
    R, C, D = [(R0,M0,E[0],E[1],1.)], [(C0,M1,E[1],1.)], [(D0,M2,E[1],1.)]
    
    alphatau, betatau = 1., 2.
    alpha0, beta0 = 6., 7.
    lambdaF, lambdaG = 3., 8.
    lambdaSn, lambdaSm = 4., 5.
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphatau':alphatau, 'betatau':betatau, 
               'lambdaF':lambdaF, 'lambdaG':lambdaG, 'lambdaSn':lambdaSn, 'lambdaSm':lambdaSm }
    settings = { 'priorF' : 'exponential', 'priorG' : 'normal', 'priorSn' : 'normal', 'priorSm' : 'normal', 
                 'orderF' : 'columns', 'orderG' : 'rows', 'orderSn' : 'rows', 'orderSm' : 'rows',
                 'ARD' : True, 'element_sparsity': True }    
    
    HMF = HMF_Gibbs(R,C,D,K,settings,priors)
    HMF.iterations = iterations
    HMF.iterations_all_Ft = iterations_all_Ft
    HMF.iterations_all_lambdat = iterations_all_lambdat
    HMF.iterations_all_Sn = iterations_all_Sn
    HMF.iterations_all_lambdan = iterations_all_lambdan
    HMF.iterations_all_taun = iterations_all_taun
    HMF.iterations_all_Sm = iterations_all_Sm
    HMF.iterations_all_lambdam = iterations_all_lambdam
    HMF.iterations_all_taum = iterations_all_taum
    HMF.iterations_all_Gl = iterations_all_Gl
    HMF.iterations_all_taul = iterations_all_taul
    
    exp_F0 = HMF.approx_expectation_Ft(E[0],burn_in,thinning)
    exp_F1 = HMF.approx_expectation_Ft(E[1],burn_in,thinning)
    exp_lambda0 = HMF.approx_expectation_lambdat(E[0],burn_in,thinning)
    exp_lambda1 = HMF.approx_expectation_lambdat(E[1],burn_in,thinning)
    exp_Sn = HMF.approx_expectation_Sn(0,burn_in,thinning)
    exp_lambdan = HMF.approx_expectation_lambdan(0,burn_in,thinning)
    exp_taun = HMF.approx_expectation_taun(0,burn_in,thinning)
    exp_Sm = HMF.approx_expectation_Sm(0,burn_in,thinning)
    exp_lambdam = HMF.approx_expectation_lambdam(0,burn_in,thinning)
    exp_taum = HMF.approx_expectation_taum(0,burn_in,thinning)
    exp_Gl = HMF.approx_expectation_Gl(0,burn_in,thinning)
    exp_taul = HMF.approx_expectation_taul(0,burn_in,thinning)
    
    assert numpy.array_equal(expected_exp_F0,exp_F0)
    assert numpy.array_equal(expected_exp_F1,exp_F1)
    assert numpy.array_equal(expected_exp_lambda0,exp_lambda0)
    assert numpy.array_equal(expected_exp_lambda1,exp_lambda1)
    assert numpy.array_equal(expected_exp_Sn,exp_Sn)
    assert numpy.array_equal(expected_exp_lambdan,exp_lambdan)
    assert expected_exp_taun == exp_taun
    assert numpy.array_equal(expected_exp_Sm,exp_Sm)
    assert numpy.array_equal(expected_exp_lambdam,exp_lambdam)
    assert expected_exp_taum == exp_taum
    assert numpy.array_equal(expected_exp_Gl,exp_Gl)
    assert expected_exp_taul == exp_taul

    
""" Test computing the performance of the predictions using the expectations """
def test_predict():
    iterations = 10
    burn_in = 2
    thinning = 3 # so index 2,5,8 -> m=3,m=6,m=9
    
    E = ['entity0','entity1']
    I = {E[0]:5, E[1]:3}
    K = {E[0]:2, E[1]:4}
    J = [6]
    
    iterations_all_Ft = {
        E[0] : [numpy.ones((I[E[0]],K[E[0]])) * 3*m**2 for m in range(1,10+1)],
        E[1] : [numpy.ones((I[E[1]],K[E[1]])) * 1*m**2 for m in range(1,10+1)] 
    }
    iterations_all_lambdat = {
        E[0] : [numpy.ones(K[E[0]]) * 3*m**2 for m in range(1,10+1)],
        E[1] : [numpy.ones(K[E[1]]) * 1*m**2 for m in range(1,10+1)]
    }
    iterations_all_Ft['entity0'][2][0,0] = 24 #instead of 27 - to ensure we do not get 0 variance in our predictions
    iterations_all_Sn = [[numpy.ones((K[E[0]],K[E[1]])) * 2*m**2 for m in range(1,10+1)]]
    iterations_all_taun = [[m**2 for m in range(1,10+1)]]
    iterations_all_Sm = [[numpy.ones((K[E[1]],K[E[1]])) * 2*m**2 * 2 for m in range(1,10+1)]]
    iterations_all_taum = [[m**2*2 for m in range(1,10+1)]]
    iterations_all_Gl = [[numpy.ones((J[0],K[E[0]])) * 2*m**2 * 3 for m in range(1,10+1)]]
    iterations_all_taul = [[m**2*3 for m in range(1,10+1)]]
    
    R0 = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    C0 = numpy.array([[1,2,3],[4,5,6],[7,8,9]],dtype=float)
    D0 = numpy.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25,26,27,28,29,30]],dtype=float)
    M0, M1, M2 = numpy.ones((5,3)), numpy.ones((3,3)), numpy.ones((5,6))
    R, C, D = [(R0,M0,E[0],E[1],1.)], [(C0,M1,E[1],1.)], [(D0,M2,E[0],1.)]
    
    alphatau, betatau = 1., 2.
    alpha0, beta0 = 6., 7.
    lambdaF, lambdaG = 3., 8.
    lambdaSn, lambdaSm = 4., 5.
    priors = { 'alpha0':alpha0, 'beta0':beta0, 'alphatau':alphatau, 'betatau':betatau, 
               'lambdaF':lambdaF, 'lambdaG':lambdaG, 'lambdaSn':lambdaSn, 'lambdaSm':lambdaSm }
    settings = { 'priorF' : 'exponential', 'priorG' : 'normal', 'priorSn' : 'normal', 'priorSm' : 'normal', 
                 'ARD' : True, 'orderF' : 'columns', 'orderG' : 'rows', 'orderSn' : 'rows', 'orderSm' : 'rows' }    
    
    #expected_exp_F0 = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    #expected_exp_F1 = numpy.array([[(9.+36.+81.)*(1./3.) for k in range(0,4)] for i in range(0,3)])
    #expected_exp_Sn = numpy.array([[(9.+36.+81.)*(2./3.) for l in range(0,4)] for k in range(0,2)])
    #expected_exp_taun = (9.+36.+81.)/3.
    #R_pred = numpy.array([[ 3542112.,  3542112.,  3542112.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.]])
    
    #expected_exp_Sm = numpy.array([[(18.+72.+162.)*(2./3.) for l in range(0,4)] for k in range(0,4)])
    #expected_exp_taum = (18.+72.+162.)/3.
    #C_pred = array([[4741632.,4741632.,4741632.],[4741632.,4741632.,4741632.],[4741632.,4741632.,4741632.]])
    
    #expected_exp_Gl = numpy.array([[(27.+108.+243.)*(2./3.) for k in range(0,2)] for j in range(0,6)])
    #expected_exp_taul = (27.+108.+243.)/3. 
    #D_pred = array([[63252.,63252.,63252.,63252.,63252.,63252.],[63504.,63504.,63504.,63504.,63504.,63504.],[63504.,63504.,63504.,63504.,63504.,63504.],[63504.,63504.,63504.,63504.,63504.,63504.],[63504.,63504.,63504.,63504.,63504.,63504.]])
    
    M_test_R = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, R_pred->3542112,3556224,3556224,3556224
    MSE_R = ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / 4.
    R2_R = 1. - ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / (4.25**2+2.25**2+2.75**2+3.75**2) #mean=7.25
    Rp_R = 357. / ( math.sqrt(44.75) * math.sqrt(5292.) ) #mean=7.25,var=44.75, mean_pred=3552696,var_pred=5292, corr=(-4.25*-63 + -2.25*21 + 2.75*21 + 3.75*21)
    
    M_test_C = numpy.array([[0,0,1],[0,1,0],[1,1,0]]) #C->3,5,7,8, C_pred->4741632,4741632,4741632,4741632
    MSE_C = ((3.-4741632.)**2 + (5.-4741632.)**2 + (7.-4741632.)**2 + (8.-4741632.)**2) / 4.
    R2_C = 1. - ((3.-4741632.)**2 + (5.-4741632.)**2 + (7.-4741632.)**2 + (8.-4741632.)**2) / (2.75**2+0.75**2+1.25**2+2.25**2) #mean=5.75
    
    M_test_D = numpy.array([[0,0,1,0,0,1],[0,1,0,0,0,0],[1,1,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]) #D->3,6,8,13,14, D_pred->63252,63252,63504,63504,63504
    MSE_D = ((3.-63252.)**2 + (6.-63252.)**2 + (8.-63504.)**2 + (13.-63504.)**2 + (14.-63504.)**2) / 5.
    R2_D = 1. - ((3.-63252.)**2 + (6.-63252.)**2 + (8.-63504.)**2 + (13.-63504.)**2 + (14.-63504.)**2) / (5.8**2+2.8**2+0.8**2+4.2**2+5.2**2) #mean=8.8
    Rp_D = 0.84265143679484211    
    
    HMF = HMF_Gibbs(R,C,D,K,settings,priors)
    HMF.iterations = iterations
    HMF.iterations_all_Ft = iterations_all_Ft
    HMF.iterations_all_lambdat = iterations_all_lambdat
    HMF.iterations_all_Sn = iterations_all_Sn
    HMF.iterations_all_taun = iterations_all_taun
    HMF.iterations_all_Sm = iterations_all_Sm
    HMF.iterations_all_taum = iterations_all_taum
    HMF.iterations_all_Gl = iterations_all_Gl
    HMF.iterations_all_taul = iterations_all_taul
    
    performances_R = HMF.predict_Rn(0,M_test_R,burn_in,thinning)
    performances_C = HMF.predict_Cm(0,M_test_C,burn_in,thinning)
    performances_D = HMF.predict_Dl(0,M_test_D,burn_in,thinning)
    
    assert performances_R['MSE'] == MSE_R
    assert performances_R['R^2'] == R2_R
    assert performances_R['Rp'] == Rp_R
    
    assert performances_C['MSE'] == MSE_C
    assert performances_C['R^2'] == R2_C
    assert numpy.isnan(performances_C['Rp'])
    
    assert performances_D['MSE'] == MSE_D
    assert performances_D['R^2'] == R2_D
    assert abs(performances_D['Rp'] - Rp_D) < 0.00000000001


""" Test the evaluation measures MSE, R^2, Rp """
def test_compute_statistics():
    R0 = numpy.array([[1,2],[3,4]],dtype=float)
    M0 = numpy.array([[1,1],[0,1]])
    E = ['entity0','entity1']
    K = {E[0]:3,E[1]:4}
    
    R = [(R0,M0,E[0],E[1],1.)]
    C, D = [], []
    HMF = HMF_Gibbs(R,C,D,K,{},{})
    
    R_pred = numpy.array([[500,550],[1220,1342]],dtype=float)
    M_pred = numpy.array([[0,0],[1,1]])
    
    MSE_pred = (1217**2 + 1338**2) / 2.0
    R2_pred = 1. - (1217**2+1338**2)/(0.5**2+0.5**2) #mean=3.5
    Rp_pred = 61. / ( math.sqrt(.5) * math.sqrt(7442.) ) #mean=3.5,var=0.5,mean_pred=1281,var_pred=7442,cov=61
    
    assert MSE_pred == HMF.compute_MSE(M_pred,R0,R_pred)
    assert R2_pred == HMF.compute_R2(M_pred,R0,R_pred)
    assert Rp_pred == HMF.compute_Rp(M_pred,R0,R_pred)
    
    
""" Test the model quality measures. """
def test_log_likelihood():
    iterations = 10
    burn_in = 2
    thinning = 3 # so index 2,5,8 -> m=3,m=6,m=9
    
    E = ['entity0','entity1']
    I = {E[0]:5, E[1]:3}
    K = {E[0]:2, E[1]:4}
    J = [6]
    
    iterations_all_Ft = {
        E[0] : [numpy.ones((I[E[0]],K[E[0]])) * 3*m**2 for m in range(1,10+1)],
        E[1] : [numpy.ones((I[E[1]],K[E[1]])) * 1*m**2 for m in range(1,10+1)] 
    }
    iterations_all_lambdat = {
        E[0] : [numpy.ones(K[E[0]]) * 3*m**2 for m in range(1,10+1)],
        E[1] : [numpy.ones(K[E[1]]) * 1*m**2 for m in range(1,10+1)]
    }
    iterations_all_Ft['entity0'][2][0,0] = 24 #instead of 27 - to ensure we do not get 0 variance in our predictions
    iterations_all_Sn = [[numpy.ones((K[E[0]],K[E[1]])) * 2*m**2 for m in range(1,10+1)]]
    iterations_all_taun = [[m**2 for m in range(1,10+1)]]
    iterations_all_Sm = [[numpy.ones((K[E[1]],K[E[1]])) * 2*m**2 * 2 for m in range(1,10+1)]]
    iterations_all_taum = [[m**2*2 for m in range(1,10+1)]]
    iterations_all_Gl = [[numpy.ones((J[0],K[E[0]])) * 2*m**2 * 3 for m in range(1,10+1)]]
    iterations_all_taul = [[m**2*3 for m in range(1,10+1)]]
    
    R0 = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    C0 = numpy.array([[1,2,3],[4,5,6],[7,8,9]],dtype=float)
    D0 = numpy.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25,26,27,28,29,30]],dtype=float)
    
    M0 = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, R_pred->3542112,3556224,3556224,3556224    
    M1 = numpy.array([[0,0,1],[0,1,0],[1,1,0]]) #C->3,7,8, C_pred->4741632,4741632,4741632 - entry 5 gets set to 0 since it is the diagonal
    M2 = numpy.array([[0,0,1,0,0,1],[0,1,0,0,0,0],[1,1,0,0,0,0],[0,0,0,0,0,0],[1,0,0,0,0,0]]) #D->3,6,8,13,14,25, D_pred->63252,63252,63504,63504,63504,63504
     
    R, C, D = [(R0,M0,E[0],E[1],1.)], [(C0,M1,E[1],1.)], [(D0,M2,E[0],1.)]
    
    #expected_exp_F0 = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    #expected_exp_F1 = numpy.array([[(9.+36.+81.)*(1./3.) for k in range(0,4)] for i in range(0,3)])
    #expected_exp_Sn = numpy.array([[(9.+36.+81.)*(2./3.) for l in range(0,4)] for k in range(0,2)])
    #expected_exp_taun = (9.+36.+81.)/3.
    #R_pred = numpy.array([[ 3542112.,  3542112.,  3542112.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.]])
    
    #expected_exp_Sm = numpy.array([[(18.+72.+162.)*(2./3.) for l in range(0,4)] for k in range(0,4)])
    #expected_exp_taum = (18.+72.+162.)/3.
    #C_pred = array([[4741632.,4741632.,4741632.],[4741632.,4741632.,4741632.],[4741632.,4741632.,4741632.]])
    
    #expected_exp_Gl = numpy.array([[(27.+108.+243.)*(2./3.) for k in range(0,2)] for j in range(0,6)])
    #expected_exp_taul = (27.+108.+243.)/3. 
    #D_pred = array([[63252.,63252.,63252.,63252.,63252.,63252.],[63504.,63504.,63504.,63504.,63504.,63504.],[63504.,63504.,63504.,63504.,63504.,63504.],[63504.,63504.,63504.,63504.,63504.,63504.],[63504.,63504.,63504.,63504.,63504.,63504.]])
    
    MSE_R = ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / 4.
    MSE_C = ((3.-4741632.)**2 + (7.-4741632.)**2 + (8.-4741632.)**2) / 3.
    MSE_D = ((3.-63252.)**2 + (6.-63252.)**2 + (8.-63504.)**2 + (13.-63504.)**2 + (14.-63504.)**2 + (25.-63504.)**2) / 6.
      
    HMF = HMF_Gibbs(R,C,D,K,{},{})
    HMF.iterations = iterations
    HMF.iterations_all_Ft = iterations_all_Ft
    HMF.iterations_all_lambdat = iterations_all_lambdat
    HMF.iterations_all_Sn = iterations_all_Sn
    HMF.iterations_all_taun = iterations_all_taun
    HMF.iterations_all_Sm = iterations_all_Sm
    HMF.iterations_all_taum = iterations_all_taum
    HMF.iterations_all_Gl = iterations_all_Gl
    HMF.iterations_all_taul = iterations_all_taul
    
    log_likelihood = 4./2. * (math.log(42.) - math.log(2*math.pi)) - 42./2.*(MSE_R*4.) + \
                     3./2. * (math.log(84.) - math.log(2*math.pi)) - 84./2.*(MSE_C*3.) + \
                     6./2. * (math.log(126.) - math.log(2*math.pi)) - 126./2.*(MSE_D*6.)    
    no_parameters = (5*2+4*3+2*4+4*4+2*6+2+4+3)
    no_datapoints = 4+3+6
    AIC = -2*log_likelihood + 2*no_parameters #F0,F1,Sn0,Sm0,G,lambda0,lambda1,tau
    BIC = -2*log_likelihood + no_parameters*math.log(no_datapoints)
    
    assert HMF.no_datapoints() == no_datapoints
    assert HMF.no_parameters() == no_parameters
    assert abs(log_likelihood - HMF.quality('loglikelihood',burn_in,thinning)) <= 1.
    assert abs(AIC - HMF.quality('AIC',burn_in,thinning)) <= 1.
    assert abs(BIC - HMF.quality('BIC',burn_in,thinning)) <= 1.
    with pytest.raises(AssertionError) as error:
        HMF.quality('FAIL',burn_in,thinning)
    assert str(error.value) == "Unrecognised metric for model quality: FAIL."