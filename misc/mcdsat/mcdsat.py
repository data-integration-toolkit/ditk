#URL to the parent class - https://github.com/bjainvarsha/spring19_csci548_query_rewriting/blob/master/query_rewriting.py
import query_rewriting
import sys
import sys
from CQ import *
from CQ.Argumento import *
from CQ.Predicado import *
from CQ.SubObjetivo import *
from CQ.CQ import *
from CQ.SOComparacion import *
from Parser.CQparser import *
from Traductor.Traductor3 import *
from Traductor.GenerarReescrituras import *
from random import *
import subprocess
import os

class mcdsat(query_rewriting.Query_Rewriting):
	"""
	Child class which implements the MCDSAT algorithm for Query Rewriting and MCDs Genreration
	"""	

	def __init__(self):
		"""
		Task -- Initialize the algorithm specific data memebers using this constructor

		self.algorithm is set to mcdsat
		self.c2d_path -- string -- path to the c2d compiler executable
		self.models -- string -- path to the dnnf models folder
		self.cnf_file -- string -- path to the file which stores the intermediate result of this algorith, ie a CNF file
		self.compiled_cnf -- string -- path to the file containing the compiled dnnf models
		self.query_models -- string -- path to the file containing all the generated query models using MCDSAT
		"""
		self.algorithm = "mcdsat"
		self.c2d_path = ""
		self.models = ""
		self.cnf_file = ""
		self.compiled_dnnf = ""
		self.query_models = ""
		query_rewriting.Query_Rewriting.__init__(self)
		
	def read_input(self, viewsFile="examples/views_0.txt", queryFile="examples/query_0.txt", mcd_flag=True, rw_flag=True, c2d_path ="c2d/c2d_linux", models ="dnnf-models/models"):
		"""
		Abstract Function Implementation
		Task - Reads the query and view files and stores their path in class members. Saves the c2d compiler and models path if specified
			   Also sets the algorithm that the calling object will be using for Query Rewriting or MCDs generation	

		Input:
		self -- mcdsat object calling this function
		query -- string -- directory path to the queries file
		views -- string -- directory path to the views file
		mcd_flag -- boolean -- generate MCDs, if True
		rw_flag -- boolean -- generate rewrtings, if True
		c2d_path -- string -- path to c2d compiler, defaults to "c2d/c2d_linux"
		models -- string -- path to the dnnf models, defaults to "dnnf-models/models"

		Result:
		Stores the values of the above input values into the class members   
		"""
		self.algorithm = "mcdsat"
		if mcd_flag:
			self.mcd_flag = True
		if rw_flag:
			self.rw_flag = True
		self.c2d_path = c2d_path
		self.models = models
		self.query_file = queryFile
		self.views_file = viewsFile

	def generate_MCDs(self):
		"""
		Abstract Function Implementation
		Task -- Genrate MCDs using the input given input using the MCDSAT algorithm

		Input:
		self -- mcdsat object calling this function
		c2d_path -- string -- directory path to the off-the-shelf c2d compiler
		models -- string -- directory path to off-the-shelf models folder
		output_path -- string -- directory path to the resulting generated MCDs

		Result:
		Generates query rewritings and writes to the specified output_file path
		Sets self.mcd_flag to True
		Sets self.query_mcds to point to the output_file path
		"""
		MCDSATDIR = "mcdsat"
		C2D = self.c2d_path
		MODS = self.models
		EXP = ""
		if self.mcd_flag:
			EXP = "Sat"
		if self.rw_flag:
			EXP = "SatRW"
		VIS = "".join(self.views_file.split("/")[1:]).replace(".txt","")
		# print(VIS)
		CNF = EXP + VIS + ".cnf"
		# print(CNF)
		NNF = CNF + ".nnf"
		# print(NNF)
		LOG = EXP + VIS + ".log.txt"
		LOG1 = EXP + VIS + "_t1.txt"
		LOG2 = EXP + VIS + "_t2.txt"

		print("Translating to CNF {} ...".format(CNF))		
		traducir(EXP, self.views_file, self.query_file, VIS + ".pyo", LOG1, CNF)
		print("CNF file {} generated".format(CNF))
		print("Compiling the CNF to DNNF using c2d compiler ...")
		self.compileToDNNF(CNF, LOG)

	def generate_query_rewritings(self, c2d_path="c2d/c2d_linux", models="dnnf_models/models/", output_file="results/rewriting.txt"):
		"""
		Abstract Function Implementation
		Task -- Genrate Query Rewritings using the input given input using the MCDSAT algorithm

		Input:
		self -- mcdsat object calling this function
		c2d_path -- string -- directory path to the off-the-shelf c2d compiler
		models -- string -- directory path to off-the-shelf models folder
		output_path -- string -- directory path to the resulting query rewritings

		Result:
		Generates query rewritings and writes to the specified output_file path
		Sets self.rw_flag to True
		Sets the self.generated_rewritings to point to the output_file path
		"""
		pass

	def compileToDNNF(self, cnf_file, log_file):
		"""
		Task -- Compiles the generated CNF into DNNF using the c2d compiler

		Input:
		self -- mcdsat object calling this function
		cnf_file -- string -- (default self.cnf_file) Path containing the cnf model of the input queries
		output_file -- directory path to store the compiled dnnf models

		Result:
		Resulting dnnf models in output_file
		Sets the self.compiled_dnnf_file to output_file
	
		"""
		command = '{} -in {} -smooth -reduce -dt_method 4 > {}'.format(self.c2d_path, cnf_file, log_file)
		out = subprocess.check_output(command, shell=True)
		print("DNNF model generated and stored successfully ...")
		print("Generating models using d-DNNF ...")
		self.generateModels(cnf_file, cnf_file + ".nnf")

	def generateModels(self, cnf_file, nnf_file):
		"""
		Task -- Generated all models for the given input query using the cnf_file, dnnf_file and the dnnf models

		Input:
		self -- mcdsat object calling this function
		cnf_file -- string -- (default self.cnf_file) Path containing the cnf model of the input queries
		dnnf_file -- string -- (default self.compiled_dnnf_file) Path containing the compiled dnnf models
		output_file -- string -- directory path to store all the generated models

		Result:
		All the resulting models are stored in output_file
		Sets the self.query_models to point to this output_file
		"""
		EXP = ""
		if self.mcd_flag:
			EXP = "Sat"
		if self.rw_flag:
			EXP = "SatRW"
		VIS = "".join(self.views_file.split("/")[1:]).replace(".txt","")
		LOG1 = EXP + VIS + "_t1.txt"
		command = "{} -w {}".format(self.models, nnf_file)
		out = subprocess.check_output(command, shell=True)
		output = out.decode("utf-8")
		output = output.split("\n")
		print("Enumerated all the models for the given query and views ...")
		#output of above command must be given as last input in the next command
		generarReescrituras(EXP, self.views_file, self.query_file, VIS + ".pyo", LOG1, output)	
		print("Generated all rewritings ... ")
		print("Cleaning Up... \nDeleting all intermediate files...")
		self.cleanup()

	def cleanup(self):
		MCDSATDIR = "mcdsat"
		EXP = ""
		if self.mcd_flag:
			EXP = "Sat"
		if self.rw_flag:
			EXP = "SatRW"
		VIS = "".join(self.views_file.split("/")[1:]).replace(".txt","")
		# print(VIS)
		CNF = EXP + VIS + ".cnf"
		# print(CNF)
		NNF = CNF + ".nnf"
		# print(NNF)
		LOG = EXP + VIS + ".log.txt"
		LOG1 = EXP + VIS + "_t1.txt"
		LOG2 = EXP + VIS + "_t2.txt"
		command = "rm -f {} {} {} {}.pyo".format(LOG, LOG1, LOG2, VIS)	
		subprocess.check_output(command, shell=True)
		command = "rm {} {}".format(CNF, NNF)
		subprocess.check_output(command, shell=True)
		print("Done ...")

if __name__ == '__main__':
	mcdsat = mcdsat()
	mcdsat.read_input('examples/views_0.txt', 'examples/query_0.txt')
	# print(mcdsat.query_file, mcdsat.views_file)
	mcdsat.generate_MCDs()
