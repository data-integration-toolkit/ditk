"""
Class for doing model selection for DI-MMTF, minimising the BIC, AIC, or MSE.
We now do a greedy search with the following strategy:
- We pick one entity type at a time
- For that entity type, we increase the current value of K by 1
- We try that model (restarts times, pick the one with the highest log likelihood)
  and if it improves the current model, set K = K + 1
- If it does not improve, no longer increase the K for this entity type
- Do this until we reach the end of our search range, or until no entity types 
  can be further increased.

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

The greedy search can be started by running search(metric,burn_in,thinning).
BIC and AIC we seek to minimise, loglikelihood maximise.

After that, the values for each metric ('BIC','AIC','loglikelihood') can be 
obtained using all_values(metric) returning a list of performances corresponding 
to values_K_tried(), and the best value of K and L can be returned using 
best_value(metric).
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

from DI_MMTF.code.models.di_mmtf_gibbs import di_mmtf_gibbs
import numpy

metrics = ['BIC','AIC','loglikelihood']

class GreedySearch:
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
        
        # Store all entity types, those we are still improving, the current values_K
        self.entity_types = ranges_K.keys()
        self.entity_types_left = list(numpy.copy(self.entity_types))
        self.current_values_K = { entity:ranges_K[entity][0] for entity in self.entity_types }
        
        # Store the values_K we tried so far and the performances
        self.values_K_tried = []
        self.all_performances = { metric : [] for metric in metrics }
        
        
    def search(self,search_metric,burn_in,thinning):
        # Try the first model
        assert search_metric in metrics, "Unrecognised metric: %s. Not in %s." % (search_metric,metrics)
        self.burn_in, self.thinning = burn_in, thinning
        self.args_quality = { 'metric':'loglikelihood', 'iterations':self.iterations, 'burn_in':burn_in, 'thinning':thinning }             
        initial_DI_MMTF = self.try_values_K(self.current_values_K)
        best_performance_so_far = initial_DI_MMTF.quality(metric=search_metric,iterations=self.iterations,burn_in=self.burn_in,thinning=self.thinning)
        self.store_performances(self.current_values_K,initial_DI_MMTF)
        
        # Then cycle through the entity types and each time increase the current one's K by 1
        while self.entity_types_left:
            current_entity_type = self.entity_types_left.pop(0)
            values_K = { entity:self.current_values_K[entity] for entity in self.entity_types }
            values_K[current_entity_type] += 1
            
            print "Running greedy search for DI-MMTF. Improving entity type %s, trying values_K = %s." % (current_entity_type,values_K)
               
            # Try the values_K restarts times, pick the best one, and store the performances
            best_DI_MMTF = self.try_values_K(values_K)
            self.store_performances(values_K,best_DI_MMTF)
            
            # Check whether we improved our metric's performance. If so, add the entity type back
            current_performance = best_DI_MMTF.quality(metric=search_metric,iterations=self.iterations,burn_in=self.burn_in,thinning=self.thinning)
            if self.improved_performance(best_performance_so_far,current_performance,search_metric):
                best_performance_so_far = current_performance
                self.current_values_K = values_K
                
                # If we have reached the end of ranges_K for this entity type, do not add it back
                if self.ranges_K[current_entity_type][-1] != self.current_values_K[current_entity_type]:
                    self.entity_types_left.append(current_entity_type)
        
        print "Finished running line search for DI-MMTF."
        self.print_all_performances()
    
    # Run DI-MMTF <restart> times and return the best one (highest log likelihood)
    def try_values_K(self,values_K):    
        best_DI_MMTF = None
        for r in range(0,self.restarts):
            print "Restart %s values_K = %s." % (r,values_K)
            DI_MMTF = di_mmtf_gibbs(R=self.R,C=self.C,K=values_K,priors=self.priors)
            DI_MMTF.initialise(init_S=self.initS,init_F=self.initF)
            DI_MMTF.run(iterations=self.iterations)
            
            if best_DI_MMTF is None or (DI_MMTF.quality(**self.args_quality) > best_DI_MMTF.quality(**self.args_quality)):
                best_DI_MMTF = DI_MMTF
        return best_DI_MMTF
        
    # Store the new performances
    def store_performances(self,values_K,best_DI_MMTF):
        self.values_K_tried.append(values_K)
        for metric in metrics:
            quality = best_DI_MMTF.quality(metric=metric,iterations=self.iterations,burn_in=self.burn_in,thinning=self.thinning)
            self.all_performances[metric].append(quality)
            
    # Return true if we improve the performance, false if not
    def improved_performance(self,old_performance,new_performance,metric):
        if (new_performance < old_performance and metric in ['AIC','BIC']):
            return True
        elif (new_performance > old_performance and metric in ['loglikelihood']):
            return True
        else:
            False
            
    # Return the list of values_K we tried
    def all_values_K(self):
        return self.values_K_tried
    
    # Return the performances of the specified metric
    def all_values(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        return self.all_performances[metric]
    
    # Return the best value for the K's
    def best_value(self,metric):
        assert metric in metrics, "Unrecognised metric name: %s." % metric
        min_or_max = numpy.argmin if metric in ['AIC','BIC'] else numpy.argmax
        index = min_or_max(self.all_values(metric))
        return self.all_values_K()[index]
        
    def print_all_performances(self):
        print "all_values_K = %s" % self.all_values_K()
        for metric in metrics:
            print "all_%s = %s" % (metric,self.all_values(metric))
        for metric in metrics:
            print "best_%s = %s" % (metric,self.best_value(metric))