# -*- coding: utf-8 -*-
import sys, os
# project_location = os.path.dirname(__file__)
# print project_location
# sys.path.insert(0,project_location+"/")

from HMF.code.models.nmf_np import nmf_np
from HMF.drug_sensitivity.load_dataset import load_data_without_empty

#to import the superclass, provide the path to the superclass file
import imputation

import time

import numpy
import codecs
import csv
numpy.set_printoptions(threshold=sys.maxsize)

R = []
M = []
K = ""
init_UV = ""
iterations = ""

class HMF_Class(imputation.Imputation):

	def preprocess(self, inputData, fileName):

		#this method receives the path to the folder and the dataset for which imputation is to be performed
		#It stores the data in two numpy arrays and returns them so that calculations can be performed on them
		#This method also returns the number of iterations to be performed and the initial U ans V values
		#the load_dataset_without_empty function introduces missingness into the data if required

		global R,M,K,init_UV,iterations, file_opener

		location = inputData
		location_data =                 location+"data_row_01/"
		location_features_drugs =       location+"features_drugs/"
		location_features_cell_lines =  location+"features_cell_lines/"
		location_kernels =              location+"kernels_features/"

		with open(inputData+fileName) as f:
			with open(inputData+"new_file.txt", 'w') as inpFile:
				reader = csv.DictReader(f, delimiter=',')
				writer = csv.DictWriter(inpFile, reader.fieldnames, delimiter='	')
				writer.writeheader()
				writer.writerows(reader)

		#R_gdsc,     M_gdsc,     _, _ = load_data_without_empty(location_data+"gdsc_ic50_row_01.txt")
		#R_ctrp,     M_ctrp,     _, _ = load_data_without_empty(location_data+"ctrp_ec50_row_01.txt")
		R_test,  M_test,  _, _ = load_data_without_empty(inputData+"new_file.txt")
		#R_ccle_ic,  M_ccle_ic,  _, _ = load_data_without_empty(location_data+"ccle_ic50_row_01.txt")

		R, M = R_test, M_test

		''' Settings NMF '''
		iterations = 1000
		init_UV = 'random'
		K = 10

		return R,M,K,init_UV,iterations

	def train(self, train_data):

		#Since data imputation is performed using matrix mfactorization,
		#this model does not require a train method

		pass

	def test(self, trained_model, test_data):

		#this model does not require a test method

		pass

	def impute(self, trained_model="", input=""):

		#the impute method is where the main imputation takes place
		#It calls the nmf_np method which performs non negative matrix factorization on the dataset
		#Non negative matrix factorization splits the dataset to compute latent factors and
		#Performs mathematical calculations to predict missing values in the dataset
		#The resulting dataset with predicted values is saved in a CSV file

		global R,M,K,init_UV,iterations

		time_start = time.time()

		NMF = nmf_np(R,M,K)
		NMF.initialise(init_UV)
		NMF.run(iterations)
		R_pred = NMF.return_R_predicted()

		#write the output to a file
		numpy.savetxt("output_file.csv", R_pred, delimiter=",")
		#print "spambase dataset predictions done"
		#print R_pred

		time_end = time.time()
		time_taken = time_end - time_start
		time_average = time_taken / iterations
		print "Time taken: %s seconds. Average per iteration: %s." % (time_taken, time_average)

		return R_pred

	def evaluate(self, trained_model = "", input = ""):
		#evaluation is done in the impute() method itself
		pass

# def main():

#     obj=HMF_Class()
#     inputData = "/Users/tushyagautam/Documents/USC/Information_Integration/Project/HMF_Submission/"
#     fileName = "wdbc.csv"
#     R,M,K,init_UV,iterations=obj.preprocess(inputData=inputData, fileName = fileName)

#     obj.impute()

# if __name__=='__main__':
#     main()

