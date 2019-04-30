#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aravindjyothi
"""
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import csv
from os import sys, path 

imputation_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
module_dir = path.dirname(path.dirname(path.abspath(__file__)))
data_dir = module_dir +'/data'

sys.path.append(imputation_dir)
sys.path.append(module_dir)


#Make the path for Imputation Visible Currently Set to Imputation and module folder

from Imputation import Imputation

class GAIN(Imputation):
    
    def __init__(self, mb_size, p_miss, p_hint, alpha, train_rate):
        self.mb_size= mb_size
        self.p_miss = p_miss
        self.p_hint = p_hint
        self.alpha = alpha
        self.train_rate = train_rate
        self.H_Dim1 = None
        self.H_Dim2 = None
        self.k = None
        
    def normalize(self, data, dimension):
        Min_Val = np.zeros(dimension)
        Max_Val = np.zeros(dimension)
        for i in range(dimension):
            Min_Val[i] = np.min(data[:,i])
            data[:,i] = data[:,i] - np.min(data[:,i])
            Max_Val[i] = np.max(data[:,i])
            data[:,i] = data[:,i] / (np.max(data[:,i]) + 1e-6)  
        return data 
    
    def introduce_missingness(self, Dim, No, Data):
        p_miss_vec = self.p_miss * np.ones((Dim,1))
        Missing = np.zeros((No, Dim))
        for i in range(Dim):
            A = np.random.uniform(0., 1., size = [len(Data),])
            B = A > p_miss_vec[i]
            Missing[:,i] = 1.*B
        return Missing
                    
    def train_test_split(self, No, Data, Missing):
        idx = np.random.permutation(No)
        Train_No = int(No * self.train_rate)
        Test_No = No - Train_No
        trainX = Data[idx[:Train_No],:]
        testX = Data[idx[Train_No:],:]
        trainM = Missing[idx[:Train_No],:]
        testM = Missing[idx[Train_No:],:]
        return trainX, testX, trainM, testM, Train_No, Test_No
    
    def gain_architecture(self, Dim):
        X = tf.placeholder(tf.float32, shape = [None, Dim])
        M = tf.placeholder(tf.float32, shape = [None, Dim])
        H = tf.placeholder(tf.float32, shape = [None, Dim])
        New_X = tf.placeholder(tf.float32, shape = [None, Dim])
        D_W1 = tf.Variable(self.xavier_init([Dim*2, self.H_Dim1]))     # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape = [ self.H_Dim1]))
        D_W2 = tf.Variable(self.xavier_init([self.H_Dim1, self.H_Dim2]))
        D_b2 = tf.Variable(tf.zeros(shape = [self.H_Dim2]))
        D_W3 = tf.Variable(self.xavier_init([self.H_Dim2, Dim]))
        D_b3 = tf.Variable(tf.zeros(shape = [Dim]))       # Output is multi-variate
        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
        G_W1 = tf.Variable(self.xavier_init([Dim*2, self.H_Dim1]))     # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = tf.Variable(tf.zeros(shape = [self.H_Dim1]))
        G_W2 = tf.Variable(self.xavier_init([self.H_Dim1, self.H_Dim2]))
        G_b2 = tf.Variable(tf.zeros(shape = [self.H_Dim2]))
        G_W3 = tf.Variable(self.xavier_init([self.H_Dim2, Dim]))
        G_b3 = tf.Variable(tf.zeros(shape = [Dim]))
        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
        return theta_D, theta_G, X, M, H, New_X

    @staticmethod
    def generator(new_x, m, G_W1, G_W2, G_W3, G_b1, G_b2, G_b3):
        inputs = tf.concat(axis = 1, values = [new_x,m])  # Mask + Data Concatenate
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output
        return G_prob
            
    @staticmethod
    def discriminator(new_x, h, D_W1, D_W2, D_W3, D_b1, D_b2, D_b3):
        inputs = tf.concat(axis = 1, values = [new_x,h])  # Hint + Data Concatenate
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output
        return D_prob
    
    def res_gen(self, inputData):
        return np.loadtxt(inputData, delimiter=",")
    
    def preprocess(self, inputData):
        Data = np.loadtxt(inputData, delimiter=",")
        self.k = self.res_gen(inputData)
        No = len(Data)
        Dim = len(Data[0,:])
        self.H_Dim1 = Dim
        self.H_Dim2 = Dim
        normalized_data = self.normalize(Data, Dim)
        return normalized_data, No, Dim
       
    @staticmethod
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape = size, stddev = xavier_stddev)
    
    @staticmethod
    def sample_M(m, n, p):
        A = np.random.uniform(0., 1., size = [m, n])
        B = A > p
        C = 1.*B
        return C
            
    @staticmethod
    def sample_Z(m, n):
        return np.random.uniform(0., 0.01, size = [m, n])        

    @staticmethod
    def sample_idx(m, n):
         A = np.random.permutation(m)
         idx = A[:n]
         return idx
        
    def return_imputed(self):
        f = open(module_dir + '/output.csv', 'w')
        cwriter = csv.writer(f)
        for each in self.k:
            cwriter.writerow(each)
   
    def train(self, normalized_data, No, Dim):
        missing_matrix  = self.introduce_missingness(Dim, No, normalized_data)
        trainX, testX, trainM, testM, Train_No, Test_No= self.train_test_split(No, normalized_data, missing_matrix)
        theta_D, theta_G, X, M, H, New_X = self.gain_architecture(Dim)
        G_sample = self.generator(New_X, M, theta_G[0],theta_G[1], theta_G[2], theta_G[3], theta_G[4], theta_G[5])
        Hat_New_X = New_X * M + G_sample * (1-M)
        D_prob = self.discriminator(Hat_New_X, H, theta_D[0], theta_D[1], theta_D[2], theta_D[3],theta_D[4],theta_D[5])
        D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8)) 
        G_loss1 = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
        MSE_train_loss = tf.reduce_mean((M * New_X - M * G_sample)**2) / tf.reduce_mean(M)
        D_loss = D_loss1
        G_loss = G_loss1 + gain_obj.alpha * MSE_train_loss
        MSE_test_loss = tf.reduce_mean(((1-M) * X - (1-M)*G_sample)**2) / tf.reduce_mean(1-M)
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for it in tqdm(range(5000)):    
            mb_idx = gain_obj.sample_idx(Train_No, gain_obj.mb_size)
            X_mb = trainX[mb_idx,:]      
            Z_mb = gain_obj.sample_Z(gain_obj.mb_size, Dim) 
            M_mb = trainM[mb_idx,:]  
            H_mb1 = gain_obj.sample_M(gain_obj.mb_size, Dim, 1-gain_obj.p_hint)
            H_mb = M_mb * H_mb1    
            New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce    
            _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict = {M: M_mb, New_X: New_X_mb, H: H_mb})
            _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
                                                                           feed_dict = {X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})
            if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr)))
                print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr)))
                print()
        return Test_No, testM, testX, MSE_test_loss, G_sample, sess, X, M, New_X
    
    def test(self, Test_No, testM, testX, MSE_test_loss, G_sample, sess, X, M, New_X):
        Z_mb = gain_obj.sample_Z(Test_No, Dim) 
        M_mb = testM
        X_mb = testX
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
        MSE_final, Sample = sess.run([MSE_test_loss, G_sample], feed_dict = {X: testX, M: testM, New_X: New_X_mb})
        print('Final Test RMSE: ' + str(np.sqrt(MSE_final)))
        
        
    def impute(self, trained_model=None, input=None):
        self.return_imputed()

    
    def evaluate(self, trained_model=None, input=None):
        self.return_imputed()
        
        
if __name__ == '__main__': 
    gain_obj = GAIN(128, 0.2, 0.9, 10, 0.8)
    data_path = data_dir + '/Letter.csv' # Dataset Chosen for execution: See other option in /data folder
    normalized_data, No, Dim= gain_obj.preprocess(data_path)
    Test_No, testM, testX, MSE_test_loss, G_sample, sess, X, M, New_X = gain_obj.train(normalized_data, No, Dim)
    gain_obj.test(Test_No, testM, testX, MSE_test_loss, G_sample, sess, X, M, New_X)
    gain_obj.impute()

    


    

    