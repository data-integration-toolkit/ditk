#Python 3.x

import abc
from relation_extraction_3 import RelationExtraction
import breds_wrapper as bd
import time

class BREDSModel(RelationExtraction):

	def __init__(self):
		"""
		Adds arguments needed for BREDS model.   
		"""
		self.config = {'max_tokens_away':40, 'min_tokens_away':1,'number_iterations':4,'context_window_size':2,'tag_type':'simple','wUpdt':0.5,'wNeg':2,'wUnk':0.0,'min_pattern_support':1,'alpha':0.1,'beta':0.8,'gamma':0.1,'word2vec_path':'afp_apw_xin_embeddings.bin'}


	def read_dataset(self,input_file, *args, **kwargs):  
		"""
		Reads dataset to be used for training and change it to useful format for BREDS model.
		Stores the data in appointed directory
              
		Args:
			input_file: Filepath with list of files to be read
           
		"""
		sentences = []
		seeds_candidates = {}
		output_format = {}
		with open(input_file) as fin:
			for line in fin:
				x = line.split('\t')
				e1 = x[1].lower()
				e2 = x[5].lower()

				e1_type = x[2]
				e2_type = x[6]
				#e1_type = 'e'
				#e2_type = 'e'
				if not x[3].isdigit():
					continue
				if not x[4].isdigit():
					continue
				if not x[7].isdigit():
					continue
				if not x[8].isdigit():
					continue
				e1_beg = int(x[3])
				e1_end = int(x[4])
				e2_beg = int(x[7])
				e2_end = int(x[8])
				rel = x[9].strip()
				#if rel=='None':
				#	continue
				s = str(x[0])
				s = s.lower()
				between = s[e1_end:e2_beg]
				between_token = between.split()
				if len(between_token) > 40 or len(between_token) < 1:
					continue
				new_sent = s[:e1_beg] + '<' + e1_type + '>' + s[e1_beg:e1_end] + '</' + e1_type + '>' + s[e1_end:e2_beg] + '<' + e2_type + '>' + s[e2_beg:e2_end] + '</' + e2_type + '>' + s[e2_end:]
				if new_sent in sentences:
					continue
				sentences.append(new_sent)
				output_format[new_sent] = [e1,e2,'NA',rel,0.0]

		self.output_format = output_format
		with open('sentences.txt','w') as fout:
			for sen in sentences:
				fout.write(sen + '\n')	

		return 'sentences.txt'
	
	def data_preprocess(self, input_data, *args, **kwargs):
		"""
		If the transformation will be too complex, use it to help change.
		"""
		pass 


	def tokenize(self,input_data,ngram_size=None, *args, **kwargs):  
		"""
		Common function from parent class, ignore it.
		"""
		pass

	
	def train(self, train_data, *args, **kwargs):  
		"""
		Prepare relations and positive seeds as well as negative seeds for each relations

		"""
		relation_count = {}
		seeds_candidates = {}
		with open(train_data) as fin:
			for line in fin:
				x = line.split('\t')
				e1 = x[1].lower()
				e2 = x[5].lower()

				e1_type = x[2]
				e2_type = x[6]

				#e1_type = 'e'
				#e2_type = 'e'
				rel = x[9].strip()
				if rel=='None':
					continue
				if rel not in relation_count:
					relation_count[rel] = []
				relation_count[rel].append((e1,e2))
				key = (e1_type,e2_type)
				if key not in seeds_candidates:
					seeds_candidates[key] = {}
				if rel not in seeds_candidates[key]:
					seeds_candidates[key][rel] = []
				seeds_candidates[key][rel].append((e1,e2))

		tmp_count = {}
		for k,v in relation_count.items():
			tmp_count[k] = len(set(v))

		print(seeds_candidates.keys())
		for k,v in seeds_candidates.items():
			for kk,vv in v.items():
				print(str(k) + ' ' + str(kk) + ' ' + str(len(vv)))

		sorted_x = sorted(tmp_count.items(), key=lambda kv: kv[1])
		print(sorted_x)
		selected_relation = [x[0] for x in sorted_x if x[1] > 50]
		#selected_relation = [sorted_x[-1][0]]

		max_rel_type = sorted_x[-1][0]
		for k,v in self.output_format.items():
			self.output_format[k][2] = max_rel_type

		seeds_for_each_relation = {}
		for k,v in seeds_candidates.items():
			for rel in selected_relation:
				if rel in v:
					seeds_for_each_relation[rel]={}
					seeds_for_each_relation[rel]['positive_seeds'] = v[rel]
					seeds_for_each_relation[rel]['entity_type'] = k
		neg_seeds = {}	
		for k,v in seeds_for_each_relation.items():
				types = v['entity_type']
				neg_seeds[k] = []
				for neg_rel,tup in seeds_candidates[types].items():
					if neg_rel != k:
						for t in tup:
							neg_seeds[k].append(t)

		for k,v in neg_seeds.items():
			seeds_for_each_relation[k]['negative_seeds'] = list(set(v))[:2]
			seeds_for_each_relation[k]['positive_seeds'] = list(set(seeds_for_each_relation[k]['positive_seeds']))[0:20]

		self.relation_seeds = seeds_for_each_relation

	def predict(self, test_data, entity_1 = None, entity_2= None,  trained_model = None, *args, **kwargs):   
		"""
		Predicts on test data using positive seeds and negative seeds. Call BREDS model from paper once for each relation
		Args:
            entity_1, entity_2: ignore it, just some common parameters.
			trained_model: ignore it.
		Returns:
              probablities: which relation is more probable given entity1, entity2 
                  or 
			relation: [tuple], list of tuples. (Eg - Entity 1, Relation, Entity 2) or in other format 
		"""
		for k,v in self.relation_seeds.items():

			pos_s = [v['entity_type']] + v['positive_seeds']
			neg_s = [v['entity_type']] + v['negative_seeds']
			#print(pos_s)
			self.relation = k
			self.init_bootstrap(pos_s,neg_s,0.1,0.1)
			breads = bd.BREDS(self.config, self.positive_seeds, self.negative_seeds, self.similarity, self.confidence)
			breads.generate_tuples(test_data)
			breads.init_bootstrap(tuples=None)

			tmp = sorted(list(breads.candidate_tuples.keys()), reverse=True)
		
			for t in tmp:
				if t.sentence not in self.output_format:
					continue
				if t.confidence > 0.1 and self.output_format[t.sentence][4] < t.confidence:
					self.output_format[t.sentence][4] = t.confidence
					self.output_format[t.sentence][2] = k
		
		with open('system_results.txt','w') as fout:
			for k,v in self.output_format.items():
				l = [k] + v
				#if v[2] !='NA':
				#	print(v[2])
				out = '\t'.join(l[:-1])
				fout.write(out + '\n')

		return 'system_results.txt'

	def evaluate(self, input_data, trained_model = None, *args, **kwargs):
		"""
		Evaluates the result based on the benchmark dataset and the evauation metrics  [Precision,Recall,F1, or others...]

		Returns:
			metrics: tuple with (p,r,f1) or similar...

		"""
		data = []

		with open(input_data) as f:
			for line in f:
				x = line.strip().split('\t')
				if len(x) < 5:
					continue
				#print(x)
				data.append(x)

		'''
		system_out = []
		standard_out = []
		for it in data:
			if it[3] != 'NA':
				print(it[3])
				system_out.append((it[1],it[2]))
			if it[4] in self.relation_seeds:
				standard_out.append((it[1],it[2]))

		standard_out = list(set(standard_out))
		system_out = list(set(system_out))
		
		inter = list(set(system_out) & set(standard_out)) 
		if len(system_out) != 0:
			precision = len(inter) / len(system_out)
		else:
			precision = 0
		if len(standard_out) != 0:
			recall = len(inter) / len(standard_out)
		else:
			recall = 0
		if precision and recall:
			f1 = 2.0 * precision * recall / (precision + recall)
		else:
			f1 = 0
		print('NA: precision %f recall %f f1 %f' % (precision,recall,f1))
		'''
		overall_pre = 0.0
		overall_recall = 0.0
		overall_f1 = 0.0
		for relation in self.relation_seeds:
			system_out = []
			standard_out = []
			for it in data:
				if it[3] == relation:
					system_out.append((it[1],it[2]))
				if it[4] == relation:
					standard_out.append((it[1],it[2]))

			standard_out = list(set(standard_out))
			system_out = list(set(system_out))

			#print(len(self.relation_seeds[relation]['positive_seeds']))
			#print(len(system_out))
			#print(len(standard_out))
			inter = list(set(system_out) & set(standard_out))
			if len(system_out) > 0: 
				precision = len(inter) / len(system_out)
			else:
				precision = 0
			if len(standard_out) > 0:
				recall = len(inter) / len(standard_out)
			else:
				recall = 0
			if precision != 0 and recall !=0:
				f1 = 2.0 * precision * recall / (precision + recall)
			else:
				f1 = 0
			if f1 > overall_f1:
				overall_f1 = f1
				overall_recall = recall
				overall_pre = precision
			#print('%s: precision %f recall %f f1 %f' % (relation,precision,recall,f1))
		print('results: precision %f recall %f f1 %f' % (overall_pre,overall_recall,overall_f1))
		return [overall_pre,overall_recall,overall_f1]

		#for k,v in self.relation_seeds.items():
		#	print(k)
		#	print(len(v['positive_seeds']))

		#print(self.relation_seeds.keys())

	def init_bootstrap(self, positive_seeds, negative_seeds, similarity, confidence):

		"""
		Automatically generate seeds files for each relation

		"""
		pos_file = 'positive_seeds.txt'
		neg_file = 'negative_seeds.txt'
		with open(pos_file,'w') as posf:
			posf.write('e1:'+positive_seeds[0][0]+'\n')
			posf.write('e2:'+positive_seeds[0][1]+'\n')
			posf.write('\n')
			for i in positive_seeds[1:]:
				posf.write(i[0]+';'+i[1]+'\n')

		with open(neg_file,'w') as negf:
			negf.write('e1:'+negative_seeds[0][0]+'\n')
			negf.write('e2:'+negative_seeds[0][1]+'\n')
			negf.write('\n')
			for i in negative_seeds[1:]:
				negf.write(i[0]+';'+i[1]+'\n')

		self.positive_seeds = pos_file
		self.negative_seeds = neg_file
		self.similarity = similarity
		self.confidence = confidence
		pass

def main():
	model = BREDSModel()
	#model.evaluate('ergfer')
	#positive_seeds = 'seeds_positive_wiki.txt'
	#negative_seeds = 'seeds_negative_wiki.txt'
	#model.init_bootstrap(positive_seeds,negative_seeds,0.2,0.3)
	#out_file = model.predict('wiki.txt')
	#print(out_file)
	#model.evaluate('relations_1555712866.txt')
	data = 'data/semeval/semeval.txt'
	test_data  = model.read_dataset(data)
	model.train(data)
	predicted_results = model.predict(test_data)
	model.evaluate(predicted_results)

if __name__=='__main__':
	main()

