"""
Class for doing model selection for DI-MMTF, minimising the BIC, AIC, or MSE.
We try an entire grid of Kt values to find the best values.

We expect the following arguments:
- ranges_K      - a dictionary from the entity names to a list of values for K
- R             - the dataset
- M             - the mask matrix
- prior         - the prior values for BNMF. This should be a dictionary of the form:
                    { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS }
- initF         - the initialisation of F - 'kmeans', 'exp' or 'random'
- initS         - the initialisation of S - 'exp' or 'random'
- iterations    - number of iterations to run 
- restarts      - we run the classifier this many times and use the one with 
                  the highest log likelihood

The grid search can be started by running search(burn_in,thinning).
BIC and AIC we seek to minimise, loglikelihood maximise.

After that, the values for each metric ('BIC','AIC','loglikelihood') can be
obtained using all_values(metric) returning a list of performances corresponding
to all_values_K(), and the best value of K and L can be returned using
best_value(metric).
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

from DI_MMTF.code.models.di_mmtf_gibbs import di_mmtf_gibbs
import numpy, itertools

metrics = ['BIC','AIC','loglikelihood']

class GridSearch:
    def __init__(self,R,C,ranges_K,priors,initS,initF,iterations,restarts=1):
        self.ranges_K = ranges_K
        self.R = R
        self.C = C
        
        self.priors = priors
        self.initS = initS
        self.initF = initF
        
        self.iterations = iterations
        self.restarts = restarts
        assert self.restarts > 0, "Need at least 1 restart."
        
        self.entity_types = ranges_K.keys()
        self.grid_values = list(itertools.product(*ranges_K.values()))
        
        self.all_performances = { metric : [] for metric in metrics }
        
    def search(self,burn_in,thinning):
        for grid_point in self.grid_values:
            values_K = { entity:K for entity,K in zip(self.entity_types,grid_point) }
            print "Running grid search for DI-MMTF. Trying values_K = %s." % values_K
            
            # Run DI-MMTF <restart> times and use the best one
            best_DI_MMTF = None
            for r in range(0,self.restarts):
                print "Restart %s values_K = %s." % (r,values_K)
                DI_MMTF = di_mmtf_gibbs(R=self.R,C=self.C,K=values_K,priors=self.priors)
                DI_MMTF.initialise(init_S=self.initS,init_F=self.initF)
                DI_MMTF.run(iterations=self.iterations)
                
                args_quality = { 'metric':'loglikelihood', 'iterations':self.iterations, 'burn_in':burn_in, 'thinning':thinning }
                if best_DI_MMTF is None or (DI_MMTF.quality(**args_quality) > best_DI_MMTF.quality(**args_quality)):
                    best_DI_MMTF = DI_MMTF
            
            for metric in metrics:
                quality = best_DI_MMTF.quality(metric=metric,iterations=self.iterations,burn_in=burn_in,thinning=thinning)
                self.all_performances[metric].append(quality)
        
        print "Finished running line search for DI-MMTF."
    
    def all_values_K(self):
        return self.grid_values
    
    def all_values(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        return self.all_performances[metric]
    
    def best_value(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        min_or_max = numpy.argmin if metric in ['AIC','BIC'] else numpy.argmax
        index = min_or_max(self.all_values(metric))
        return self.all_values_K()[index]
        
    def print_all_performances(self):
        print "Entity types: %s." % self.entity_types
        print "All values K: %s." % self.all_values_K()
        for metric in metrics:
            print "All %s: %s." % (metric,self.all_values(metric))
        for metric in metrics:
            print "Best %s: %s." % (metric,self.best_value(metric))