"""
Generate a toy dataset for the data integration matrix tri-factorisation model, and store it.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

from HMF.code.distributions.exponential import exponential_draw
from HMF.code.distributions.normal import normal_draw
from ml_helpers.code.mask import generate_M

import numpy, itertools, matplotlib.pyplot as plt

''' 
Method for generating all the datasets - the Fs, Ss, Gs, Rn's, Cm's, Dl's - with noise.

Input:
- params_F is a list (E, I, K, lambdaF, prior)
- params_R is a list of tuples (E1, E2, lambdaSn, prior, taun)
- params_C is a list of tuples (E, lambdaSm, prior, taum)
- params_D is a list of tuples (E, J, lambdaGl, prior, taul)
Where prior is either 'normal' or 'exponential'.

Output is a tuple (all_R_true, all_R, all_C_true, all_C, all_D_true, all_D, all_Ft, all_Sn, all_Sm, all_Gl)
- all_R_true is a list [true_Rn] in order of params_R
- all_R is a list [Rn] in order of params_R
- all_C_true is a list [true_Cm] in order of params_C
- all_C is a list [true_Cm] in order of params_C
- all_D_true is a list [true_Dl] in order of params_D
- all_D is a list [true_Dl] in order of params_D
- all_Ft is a dictionary from entity names to F
- all_Sn is a list [Sn] in order of R
- all_Sm is a list [Sm] in order of C
- all_Gl is a list [Gl] in order of D
'''
def generate_dataset(params_F, params_R, params_C, params_D):  
    all_Ft, all_It, all_Kt = {}, {}, {}
    for (E, I, K, lambdaF, prior) in params_F:
        F = numpy.zeros((I,K))
        for i,k in itertools.product(xrange(0,I),xrange(0,K)):
            F[i,k] = exponential_draw(lambdaF) if prior == 'exponential' else normal_draw(0,lambdaF)
        all_Ft[E], all_It[E], all_Kt[E] = F, I, K
    
    all_Rn_true, all_Rn, all_Sn = [], [], []
    for (E1, E2, lambdaSn, prior, taun) in params_R:
        S = numpy.zeros((all_Kt[E1],all_Kt[E2]))
        for k,l in itertools.product(xrange(0,all_Kt[E1]),xrange(0,all_Kt[E2])):
            S[k,l] = exponential_draw(lambdaSn) if prior == 'exponential' else normal_draw(0,lambdaSn)
        all_Sn.append(S)      
        
        true_Rn = numpy.dot(all_Ft[E1],numpy.dot(S,all_Ft[E2].T))
        Rn = add_noise(true_Rn,taun)
        
        all_Rn_true.append(true_Rn)
        all_Rn.append(Rn)
            
    all_Cm_true, all_Cm, all_Sm = [], [], []
    for E1, lambdaSm, prior, taum in params_C:
        S = numpy.zeros((all_Kt[E1],all_Kt[E1]))
        for k,l in itertools.product(xrange(0,all_Kt[E1]),xrange(0,all_Kt[E1])):
            S[k,l] = exponential_draw(lambdaSm) if prior == 'exponential' else normal_draw(0,lambdaSm)
        all_Sm.append(S)
        
        true_Cm = numpy.dot(all_Ft[E1],numpy.dot(S,all_Ft[E1].T))
        Cm = add_noise(true_Cm,taum)

        all_Cm_true.append(true_Cm)        
        all_Cm.append(Cm)
            
    all_Dl_true, all_Dl, all_Gl = [], [], []
    for E1, J, lambdaGl, prior, taul in params_D:
        G = numpy.zeros((J,all_Kt[E1]))
        for j,k in itertools.product(xrange(0,J),xrange(0,all_Kt[E1])):
            G[j,k] = exponential_draw(lambdaGl) if prior == 'exponential' else normal_draw(0,lambdaGl)
        all_Gl.append(G)
        
        true_Dl = numpy.dot(all_Ft[E1],G.T)
        Dl = add_noise(true_Dl,taul)

        all_Dl_true.append(true_Dl)        
        all_Dl.append(Dl)
    
    return (all_Rn_true, all_Rn, all_Cm_true, all_Cm, all_Dl_true, all_Dl, all_Ft, all_Sn, all_Sm, all_Gl)
 
   
def add_noise(true_R,tau):
    ''' Add Gaussian noise of precision tau to the given matrix true_R. '''
    if numpy.isinf(tau):
        return numpy.copy(true_R)
    (I,J) = true_R.shape
    R = numpy.zeros((I,J))
    for i,j in itertools.product(xrange(0,I),xrange(0,J)):
        R[i,j] = normal_draw(true_R[i,j],tau)
    return R


'''
Method for generating all the mask matrices.
Input:
- all_It is a dictionary mapping entity names to I values
- params_R_M is a list of tuples (E1,E2,fraction_unknown)
- params_C_M is a list of tuples (E1,fraction_unknown)
- params_D_M is a list of tuples (E1,J,fraction_unknown)
Output is a tuple (all_Mn, all_Mm, all_Ml)
- all_Mn is a list of the mask matrices for Rn, in order of params_R
- all_Mm is a list of the mask matrices for Cm, in order of params_C
- all_Ml is a list of the mask matrices for Dl, in order of params_D
'''
def generate_all_masks(all_It, params_R_M, params_C_M, params_D_M):
    all_Mn, all_Mm, all_Ml = [], [], []
    for E1,E2,fraction_unknown in params_R_M:
        Mn = generate_M(all_It[E1],all_It[E2],fraction_unknown)
        all_Mn.append(Mn)
    for E1,fraction_unknown in params_C_M:
        Mm = generate_M(all_It[E1],all_It[E1],fraction_unknown)
        all_Mm.append(Mm)
    for E1,J,fraction_unknown in params_D_M:
        Ml = generate_M(all_It[E1],J,fraction_unknown)
        all_Ml.append(Ml)
    return all_Mn, all_Mm, all_Ml
    

###############################################################################


if __name__ == "__main__":
    output_folder = project_location+"HMF/toy_experiments/data/"
    
    ''' 
    Main dataset is R12_1.
    Second dataset is R12_2.
    Similarity kernel is C11.
    Feature datasets are D1 and D2.
    '''    
    # Parameters for the Ft
    E = [1,2]
    I = { 1:100, 2:80 }
    K = { 1:10,  2:8  }
    lambdaF = { 1: 1., 2: 1. }
    prior_F = 'exponential'
    params_F = [(E1,I[E1],K[E1],lambdaF[E1],prior_F) for E1 in E]
    
    # Parameters for R12_1 and R12_2
    N = 2
    lambdaSn = [1.,1.]
    taun = [1.,1.]
    prior_Sn = 'normal'
    params_R = [
        (1,2,lambdaSn[0],prior_Sn,taun[0]), # main dataset R12_1 
        (1,2,lambdaSn[1],prior_Sn,taun[1])  # second dataset R_12_2
    ]
    
    # Parameters for C11
    M = 1
    lambdaSm = [1000.]
    taum = [100.]
    prior_Sm = 'exponential'
    params_C = [(1,lambdaSm[0],prior_Sm,taum[0])] # similarity kernel entity 1 
    
    # Parameters for D1, D2
    L = 2
    lambdaGl = [1.,1.]
    taul = [1.,1.]
    prior_Gl = 'normal'
    J = [150, 100]
    params_D = [
        (1,J[0],lambdaGl[0],prior_Gl,taul[0]), # feature dataset D1   
        (2,J[1],lambdaGl[1],prior_Gl,taul[1])  # feature dataset D2   
    ]
    
    # Parameters M
    fractions_unknown_R = [0.1,0,0]
    fractions_unknown_C = [0]
    fractions_unknown_D = [0,0]
    params_R_M = [(E1,E2,fraction) for fraction,(E1,E2,_,_,_) in zip(fractions_unknown_R,params_R)]
    params_C_M = [(E1,fraction) for fraction,(E1,_,_,_) in zip(fractions_unknown_R,params_C)]
    params_D_M = [(E1,J1,fraction) for fraction,(E1,J1,_,_,_) in zip(fractions_unknown_R,params_D)]
    
    (all_Rn_true, all_Rn, all_Cm_true, all_Cm, all_Dl_true, all_Dl, all_Ft, all_Sn, all_Sm, all_Gl) = \
        generate_dataset(params_F,params_R,params_C,params_D)
    (all_Mn, all_Mm, all_Ml) = generate_all_masks(I, params_R_M, params_C_M, params_D_M)
    

    ''' Store all matrices in text files '''
    for E1 in E:
        numpy.savetxt(open(output_folder+"F_%s.txt" % E1,'w'),all_Ft[E1])
    for n in range(0,N):
        numpy.savetxt(open(output_folder+"R_%s_true.txt" % n,'w'),all_Rn_true[n])
        numpy.savetxt(open(output_folder+"R_%s.txt" % n,'w'),all_Rn[n])
        numpy.savetxt(open(output_folder+"M_%s.txt" % n,'w'),all_Mn[n])
        numpy.savetxt(open(output_folder+"Sn_%s.txt" % n,'w'),all_Sn[n])
        print "R_%s. Mean: %s. Variance: %s. Min: %s. Max: %s." % (n,numpy.mean(all_Rn[n]),numpy.var(all_Rn[n]),all_Rn[n].min(),all_Rn[n].max())
    for m in range(0,M):
        numpy.savetxt(open(output_folder+"C_%s_true.txt" % m,'w'),all_Cm_true[m])
        numpy.savetxt(open(output_folder+"C_%s.txt" % m,'w'),all_Cm[m])
        numpy.savetxt(open(output_folder+"M_%s.txt" % m,'w'),all_Mm[m])
        numpy.savetxt(open(output_folder+"Sm_%s.txt" % m,'w'),all_Sm[m])
        print "C_%s. Mean: %s. Variance: %s. Min: %s. Max: %s." % (m,numpy.mean(all_Cm[m]),numpy.var(all_Cm[m]),all_Cm[m].min(),all_Cm[m].max())
    for l in range(0,L):
        numpy.savetxt(open(output_folder+"D_%s_true.txt" % l,'w'),all_Dl_true[l])
        numpy.savetxt(open(output_folder+"D_%s.txt" % l,'w'),all_Dl[l])
        numpy.savetxt(open(output_folder+"M_%s.txt" % l,'w'),all_Ml[l])
        numpy.savetxt(open(output_folder+"G_%s.txt" % l,'w'),all_Gl[l])
        print "D_%s. Mean: %s. Variance: %s. Min: %s. Max: %s." % (l,numpy.mean(all_Dl[l]),numpy.var(all_Dl[l]),all_Dl[l].min(),all_Dl[l].max())
    