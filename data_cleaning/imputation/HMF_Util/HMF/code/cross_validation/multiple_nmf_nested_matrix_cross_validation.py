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
from multiple_nmf_parallel_matrix_cross_validation import MultipleNMFParallelMatrixCrossValidation
from nested_matrix_cross_validation import MatrixNestedCrossValidation

import numpy

attempts_generate_M = 1000

class MultipleNMFNestedCrossValidation(MatrixNestedCrossValidation):
    # Run the cross-validation
    def run(self,rows_M=None,columns_M=None):
        assert rows_M is not None or columns_M is not None, "Either rows_M or columns_M should be a list of indices."    
        
        if rows_M is not None:
            crossval_folds = mask.compute_crossval_folds_rows_attempts(self.M,no_rows=rows_M,no_folds=self.K,attempts=attempts_generate_M)                  
        elif columns_M is not None:
            crossval_folds = mask.compute_crossval_folds_columns_attempts(self.M,no_columns=columns_M,no_folds=self.K,attempts=attempts_generate_M)  
                    
        for i,(train,test) in enumerate(crossval_folds):
            print "Fold %s of nested cross-validation." % (i+1)  
            assert numpy.array_equal(self.M,train+test), "Something went wrong with splitting M for nested cross-validation!"
            
            # Run the cross-validation
            crossval = MultipleNMFParallelMatrixCrossValidation(
            #crossval = MatrixCrossValidation(
                method=self.method,
                X=self.X,
                M=train,
                K=self.K,
                parameter_search=self.parameter_search,
                train_config=self.train_config,
                file_performance=self.files_nested_performances[i],
                P=self.P
            )
            crossval.run(rows_M=rows_M,columns_M=columns_M)
            
            try:
                (best_parameters,_) = crossval.find_best_parameters(evaluation_criterion='MSE',low_better=True)
                print "Best parameters for fold %s were %s." % (i+1,best_parameters)
            except KeyError:
                best_parameters = self.parameter_search[0]
                print "Found no performances, dataset too sparse? Use first values instead for fold %s, %s." % (i+1,best_parameters)
            
            # Train the model and test the performance on the test set
            performance_dict = self.run_model(train,test,best_parameters)
            self.store_performances(performance_dict)
            print "Finished fold %s, with performances %s." % (i+1,performance_dict)            
            
        self.log()