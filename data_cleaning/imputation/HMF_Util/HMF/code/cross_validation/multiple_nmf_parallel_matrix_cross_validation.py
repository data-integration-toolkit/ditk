"""
For multiple NMF we concatenated the rows or columns of the matrix. 
When we then split the mask matrix, we only use the first n of the rows or columns.
The other rows or columns remain unchanged.

This class makes that modification.

Everything is the same, except the run() method. We now pass one of two arguments:
- rows_M:    number of rows we use for the cross-validation on M
- columns_M: number of columns we use for the cross-validation on M
One should be an integer, the other should be None.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import DI_MMTF.code.generate_mask.mask as mask
from parallel_matrix_cross_validation import ParallelMatrixCrossValidation, run_fold

from multiprocessing import Pool
import numpy

attempts_generate_M = 1000

# Class, redefining the run function
class MultipleNMFParallelMatrixCrossValidation(ParallelMatrixCrossValidation):
    # Run the cross-validation
    def run(self,rows_M=None,columns_M=None):
        assert rows_M is not None or columns_M is not None, "Either rows_M or columns_M should be a list of indices."    
        
        for parameters in self.parameter_search:
            print "Trying parameters %s." % (parameters)
            
            try:
                if rows_M is not None:
                    crossval_folds = mask.compute_crossval_folds_rows_attempts(self.M,no_rows=rows_M,no_folds=self.K,attempts=attempts_generate_M)                  
                elif columns_M is not None:
                    crossval_folds = mask.compute_crossval_folds_columns_attempts(self.M,no_columns=columns_M,no_folds=self.K,attempts=attempts_generate_M)  
                    
                # We need to put the parameter dict into json to hash it
                self.all_performances[self.JSON(parameters)] = {}
                
                # Create the threads for the folds, and run them
                pool = Pool(self.P)
                all_parameters = [
                    {
                        'parameters' : parameters,
                        'X' : numpy.copy(self.X),
                        'train' : train,
                        'test' : test,
                        'method' : self.method,
                        'train_config' : self.train_config                
                    }
                    for (train,test) in crossval_folds
                ]
                outputs = pool.map(run_fold,all_parameters)
                pool.close()
                
                for performance_dict in outputs:
                    self.store_performances(performance_dict,parameters)
                    
                self.log(parameters)
                
            except Exception as e:
                self.fout.write("Tried parameters %s but got exception: %s. \n" % (parameters,e))