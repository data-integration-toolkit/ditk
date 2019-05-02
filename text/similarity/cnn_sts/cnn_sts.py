
# import DITKModel_SemanticSimilarity as SemSim
import numpy as np
import multiprocessing as mp
import random,copy,string
from nltk.tokenize import word_tokenize
from scipy.stats import pearsonr
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.layers import Input, Convolution1D, MaxPooling1D, Flatten
from tensorflow.python.keras.layers import Lambda, multiply, concatenate, Dense
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle
from text_similarity import TextSemanticSimilarity




class Embedder(object):
	# generating embeddings

	def __init__(self, dictname, wordvectdim):
		print('Loading GloVe...(This might take one or two minutes.)')
		self.wordtoindex   = dict()
		self.indextovector = []
		self.indextovector.append(np.zeros(wordvectdim))
		lines = open(dictname, 'r').readlines()
		blocksize = 1000
		r_list = mp.Pool(32).map(self._worker, ((lines[block:block+blocksize], block) for block in range(0,len(lines),blocksize)))
		for r in r_list:
		  self.wordtoindex.update(r[0])
		  self.indextovector.extend(r[1])
		self.indextovector = np.array(self.indextovector, dtype='float32')
	def _worker(self,args):
		wordtoindex   = dict()
		indextovector = []
		for line in args[0]:
			elements = line.split(' ')
			wordtoindex[elements[0]] = len(indextovector)+args[1]+1
			indextovector.append(np.array(elements[1:]).astype(float))
		return (wordtoindex,indextovector)
	def matrixize(self, sentencelist, sentencepad):
		indexlist = []
		for sentence in sentencelist:
			indexes = []
			for word in sentence:
				word = word.lower()
				if word not in self.wordtoindex: indexes.append(1)
				else: indexes.append(self.wordtoindex[word])
			indexlist.append(indexes)
		return self.indextovector[(pad_sequences(indexlist, maxlen=sentencepad, truncating='post', padding='post'))]

class STSTask(TextSemanticSimilarity):
	'''
	A class which Uses a  convolutional neural network to evaluate semantic textual similarity
	
	'''

	def __init__(self):
		""" c is the dictionary containing all the parameters in the following  example format"""
	
		self.c = dict()
		self.c['num_runs']   = 2
		self.c['num_epochs'] = 6
		self.c['num_batchs'] = 2
		self.c['batch_size'] = 3000
		self.c['wordvectdim']  = 300
		self.c['sentencepad']  = 60
		self.c['num_classes']  = 6
		self.c['cnnfilters']     = {1: 1800}
		self.c['cnninitial']     = 'he_uniform'
		self.c['cnnactivate']    = 'relu'
		self.c['densedimension'] = list([1800])
		self.c['denseinitial']   = 'he_uniform'
		self.c['denseactivate']  = 'tanh'
		self.c['optimizer']  = 'adam'
		
		
	def load_resc(self,dictname):
		# call embedding function

		self.embed = Embedder(dictname, self.c['wordvectdim'])
		
	def read_dataset(self,trainfile, validfile,testfile, *args, **kwargs): 
		"""
		Reads a dataset that is a CSV/Excel File.

		Args:
			fileName : With it's absolute path

		Returns:
			training_data_list : List of Lists that containes 2 sentences and it's similarity score 
			Note :
				Format of the output : [[S1,S2,Sim_score],[T1,T2,Sim_score]....]

		Raises:
			None
		"""
		#parse files to obtain the output
		self.traindata= self._load_data(trainfile)
		self.validdata= self._load_data(validfile)
		self.testdata = self._load_data(testfile)
		
	def _load_data(self, filename):
	
		"" " Takes a file and loads the data and returns a dict of labels """
		s0,s1,labels = [],[],[]
		lines=open(filename,'r').read().splitlines()
		for line in lines[1:]:
			# _,_,_,_, label, s0x, s1x = line.rstrip().split('\t')[:7]
			s0x, s1x, label = line.rstrip().split('\t')
			labels.append(float(label))
			s0.append([word.lower() for word in word_tokenize(s0x) if word not in string.punctuation])
			s1.append([word.lower() for word in word_tokenize(s1x) if word not in string.punctuation])
		m0 = self.embed.matrixize(s0, self.c['sentencepad'])
		m1 = self.embed.matrixize(s1, self.c['sentencepad'])
		classes = np.zeros((len(labels), self.c['num_classes']))
		for i, label in enumerate(labels):
			if np.floor(label) + 1 < self.c['num_classes']:
				classes[i, int(np.floor(label)) + 1] = label - np.floor(label)
			classes[i, int(np.floor(label))] = np.floor(label) - label + 1
		return {'labels': labels, 's0': s0, 's1': s1, 'classes': classes, 'm0': m0, 'm1': m1}
	
	
	
		
	def create_model(self):
		# create the model 

		K.clear_session()
		input0 = Input(shape=(self.c['sentencepad'], self.c['wordvectdim']))
		input1 = Input(shape=(self.c['sentencepad'], self.c['wordvectdim']))
		Convolt_Layer=[]
		MaxPool_Layer=[]
		Flatten_Layer=[]
		for kernel_size, filters in self.c['cnnfilters'].items():
			Convolt_Layer.append(Convolution1D(filters=filters,
											   kernel_size=kernel_size,
											   padding='valid',
											   activation=self.c['cnnactivate'],
											   kernel_initializer=self.c['cnninitial']))
			MaxPool_Layer.append(MaxPooling1D(pool_size=int(self.c['sentencepad']-kernel_size+1)))
			Flatten_Layer.append(Flatten())
		Convolted_tensor0=[]
		Convolted_tensor1=[]
		for channel in range(len(self.c['cnnfilters'])):
			Convolted_tensor0.append(Convolt_Layer[channel](input0))
			Convolted_tensor1.append(Convolt_Layer[channel](input1))
		MaxPooled_tensor0=[]
		MaxPooled_tensor1=[]
		for channel in range(len(self.c['cnnfilters'])):
			MaxPooled_tensor0.append(MaxPool_Layer[channel](Convolted_tensor0[channel]))
			MaxPooled_tensor1.append(MaxPool_Layer[channel](Convolted_tensor1[channel]))
		Flattened_tensor0=[]
		Flattened_tensor1=[]
		for channel in range(len(self.c['cnnfilters'])):
			Flattened_tensor0.append(Flatten_Layer[channel](MaxPooled_tensor0[channel]))
			Flattened_tensor1.append(Flatten_Layer[channel](MaxPooled_tensor1[channel]))
		if len(self.c['cnnfilters']) > 1:
			Flattened_tensor0=concatenate(Flattened_tensor0)
			Flattened_tensor1=concatenate(Flattened_tensor1)
		else:
			Flattened_tensor0=Flattened_tensor0[0]
			Flattened_tensor1=Flattened_tensor1[0]
		absDifference = Lambda(lambda X:K.abs(X[0] - X[1]))([Flattened_tensor0,Flattened_tensor1])
		mulDifference = multiply([Flattened_tensor0,Flattened_tensor1])
		allDifference = concatenate([absDifference,mulDifference])
		for ilayer, densedimension in enumerate(self.c['densedimension']):
			allDifference = Dense(units=int(densedimension), 
								  activation=self.c['denseactivate'], 
								  kernel_initializer=self.c['denseinitial'])(allDifference)
		output = Dense(name='output',
					   units=self.c['num_classes'],
					   activation='softmax', 
					   kernel_initializer=self.c['denseinitial'])(allDifference)
		self.model = Model(inputs=[input0,input1], outputs=output)
# 		self.model.compile(loss={'output': self._lossfunction}, optimizer=self.c['optimizer'])
		# losses.custom_loss = self._lossfunction
		self.model.compile(loss ='mse', optimizer=self.c['optimizer'])
	def _lossfunction(self,y_true,y_pred):
		ny_true = y_true[:,1] + 2*y_true[:,2] + 3*y_true[:,3] + 4*y_true[:,4] + 5*y_true[:,5]
		ny_pred = y_pred[:,1] + 2*y_pred[:,2] + 3*y_pred[:,3] + 4*y_pred[:,4] + 5*y_pred[:,5]
		my_true = K.mean(ny_true)
		my_pred = K.mean(ny_pred)
		var_true = (ny_true - my_true)**2
		var_pred = (ny_pred - my_pred)**2
		return -K.sum((ny_true-my_true)*(ny_pred-my_pred),axis=-1) / (K.sqrt(K.sum(var_true,axis=-1)*K.sum(var_pred,axis=-1)))
		
	def _sample_pairs(self, data, batch_size, shuffle=True, once=False):
		num = len(data['classes'])
		idN = int((num+batch_size-1) / batch_size)
		ids = list(range(num))
		while True:
			if shuffle: random.shuffle(ids)
			datacopy= copy.deepcopy(data)
			for name, value in datacopy.items():
				valuer=copy.copy(value)
				for i in range(num):
					valuer[i]=value[ids[i]]
				datacopy[name] = valuer
			for i in range(idN):
				sl  = slice(i*batch_size, (i+1)*batch_size)
				dataslice= dict()
				for name, value in datacopy.items():
					dataslice[name] = value[sl]
				x = [dataslice['m0'],dataslice['m1']]
				y = dataslice['classes']
				yield (x,y)
			if once: break

	def eval_model(self):
		results = []
		for data in [self.traindata, self.validdata, self.testdata]:
		# for data in [self.testdata]:
			predictionclasses = []
			for dataslice,_ in self._sample_pairs(data, len(data['classes']), shuffle=False, once=True):
				predictionclasses += list(self.model.predict(dataslice))
			prediction = np.dot(np.array(predictionclasses),np.arange(self.c['num_classes']))
			goldlabels = data['labels']
			result=pearsonr(prediction, goldlabels)[0]
			results.append(round(result,4))
		print('[Train, Valid, Test]=',end='')
		print(results)
		return tuple(results)
		
	def fit_model(self, wfname):
		kwargs = dict()
		kwargs['generator']       = self._sample_pairs(self.traindata, self.c['batch_size'])
		kwargs['steps_per_epoch'] = self.c['num_batchs']
		kwargs['epochs']          = self.c['num_epochs']
		class Evaluate(Callback):
			def __init__(self, task, wfname):
				self.task       = task
				self.bestresult = 0.0
				self.wfname     = wfname
			def on_epoch_end(self, epoch, logs={}):
				_,validresult,_ = self.task.eval_model()
				if validresult > self.bestresult:
					self.bestresult = validresult
					self.task.model.save(self.wfname)
		kwargs['callbacks'] = [Evaluate(self, wfname)]
		return self.model.fit_generator(verbose=2,**kwargs)
		
	def save_model(self, file):
		"""
		:param file: Where to save the model - Optional function
		:return:
		"""
		self.model.save(file)
		print("Model Saved!")
		return
		
	def load_model(self, file):
		"""
		:param file: From where to load the model - Optional function
		:return:
		"""
		
		model = load_model(file)
		self.model = model
		print("Model Loaded!")
		return


		
	def train(self,i_run, *args, **kwargs): 
	
		""" creates model and saves it as a class attribute"""
		self.create_model()
		print('Training')
		wfname = './weightfile'+str(i_run)
		self.fit_model(wfname)
		self.model.load_weights(wfname)
		return
		
		
	def predict(self, data):
		'''
		returns the model prediction value
		
		'''
		predictionclasses = []
		for dataslice,_ in self._sample_pairs(data, len(data['classes']), shuffle=False, once=True):
			predictionclasses += list(self.model.predict(dataslice))
		prediction = np.dot(np.array(predictionclasses),np.arange(self.c['num_classes']))
		return prediction

	def generate_embeddings(self, sent1, sent2, *args, **kwargs):  
		'''
			Task: Returns the hash that have been used to compute similarity ( Hash )

			Args:
				input_list : List of Words

			Returns:
				embeddings_list : List of embeddings/hash of those words

			Raises:
				None
	
		'''
		s0,s1 = [],[]
		s0.append([word.lower() for word in word_tokenize(sent1) if word not in string.punctuation])
		s1.append([word.lower() for word in word_tokenize(sent2) if word not in string.punctuation])
		m0 = self.embed.matrixize(s0, self.c['sentencepad'])
		m1 = self.embed.matrixize(s1, self.c['sentencepad'])

		return m0,m1
		

	def predict(self,sent1,sent2):

		m0,m1 = self.generate_embeddings(sent1,sent2)
		pred = self.model.predict([m0,m1])
		res = np.dot(np.array(pred),np.arange(self.c['num_classes']))
		return res[0]/5.0




	def evaluate(self): 
		"""
		Returns the correlation score(0-1) between the actual and predicted similarity scores

		Args:
			actual_values : List of actual similarity scores
			predicted_values : List of predicted similarity scores

		Returns:
			correlation_coefficient : Value between 0-1 to show the correlation between the values(actual and predicted)

		Raises:
			None
		"""
		
		results = []
		for data in [self.traindata, self.validdata, self.testdata]:
			prediction = self.predict(data)
			goldlabels = data['labels']
			result=pearsonr(prediction, goldlabels)[0]
			results.append(round(result,4))
		print('[Train, Valid, Test]=',end='')
		print(results)
		return tuple(results)
		
	

# if __name__ == "__main__":
# 	tsk = STSTask()
# 	tsk.load_resc('glove.840B.300d.txt')
# 	print("Loading Vectors - DONE!")
# 	print("Loading datasets....")
# 	tsk.read_dataset('sts2014-train.txt', 'sts2014-trial.txt', 'sts2014-test.txt')
# 	bestresult = 0.0
# 	bestwfname = None
# 	print("Done reading datasets!")
# 	print("Training")
# 	for i_run in range(tsk.c['num_runs']):
# 		print('RunID: %s' %i_run)
# 		tsk.train()
# 	tsk.save_model('cnn_model.h5')
# 	print("Model Saved!")
# 	print("Loading Model....")
# 	tsk.load_model('cnn_model.h5')
# 	sent1 = "checking the similarity of two sentences"
# 	sent2 = "checking if two sentences are similar"
# 	score = tsk.predict(sent1,sent2)
# 	print("Similarity:" + str(round(score,3)))
	# _,validresult,_ = tsk.evaluate()
	# print(validresult)
	# print("Donee")

		
	'''
	Sample Workflow:
	
	Takes the input file and loads data
	
	Generates  GloVe Embeddings 
	
	Creates CNN using specified hyperparameters and fits model
	
	Predicts similarity score based on model
	
	Evaluates predictions using Pearson Correlation Coeeffient 
	
	"""

	'''