from tensor_factorization import TensorFactorization
from scipy.io.matlab import loadmat
import numpy as np
from scipy.sparse import lil_matrix
from rescal import als
import os
import pickle
from rescal_eval import innerfold
import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('RESCAL')

class RESCALModel(TensorFactorization):
    
    def __init__(self):
        TensorFactorization.__init__(self)
        self.T = None
        self.X = None
        self.A = None
        self.R = None

    def read_dataset(self, path_tensor, key_tensor):
        matrix = loadmat(path_tensor)
        self.T = np.array(matrix[key_tensor], np.float32)

        # construct array for rescal
        self.X = [lil_matrix(self.T[:, :, k]) for k in range(self.T.shape[2])]

        return self.X
    
    def factorize(self, rank):
        '''
            Run RESCAL algorithm to factorize tensor
        '''
        self.A, self.R, _, _, _ = als(self.X, rank, init='nvecs', conv=1e-3, lambda_A=10, lambda_R=10)

        return (self.A, self.R)

    def save_model(self, dir):
        '''
            Save matrices A and R to txt files in directory dir
        '''
        np.savetxt(dir+'\\rescal_output_A.txt', self.A, delimiter=',')
        with open(dir+"\\rescal_output_R.txt", "wb") as wb:
            pickle.dump(self.R, wb)
    
    def load_model(self, dir):
        '''
            Load matrices A and R from txt files in directory dir
        '''
        self.A = np.loadtxt(dir+'\\rescal_output_A.txt', delimiter=',')
        with open(dir+"\\rescal_output_R.txt", "rb") as rb:
            self.R = pickle.load(rb)
        
        return (self.A, self.R)

    def evaluate(self):
        K = self.T
        e, k = K.shape[0], K.shape[2]
        SZ = e * e

        # copy ground truth before preprocessing
        GROUND_TRUTH = K.copy()

        # construct array for rescal
        T = [lil_matrix(K[:, :, i]) for i in range(k)]

        # Do cross-validation
        FOLDS = 10
        IDX = list(range(SZ))
        np.random.shuffle(IDX)

        fsz = int(SZ / FOLDS)
        offset = 0
        AUC_train = np.zeros(FOLDS)
        AUC_test = np.zeros(FOLDS)
        for f in range(FOLDS):
            idx_test = IDX[offset:offset + fsz]
            idx_train = np.setdiff1d(IDX, idx_test)
            np.random.shuffle(idx_train)
            idx_train = idx_train[:fsz].tolist()
            _log.info('Train Fold %d' % f)
            AUC_train[f] = innerfold(T, idx_train + idx_test, idx_train, e, k, SZ, GROUND_TRUTH)
            _log.info('Test Fold %d' % f)
            AUC_test[f] = innerfold(T, idx_test, idx_test, e, k, SZ, GROUND_TRUTH)

            offset += fsz

        mean_train = AUC_train.mean()
        std_train = AUC_train.std()
        mean_test = AUC_test.mean()
        std_test = AUC_test.std()

        return (mean_train, std_train, mean_test, std_test)