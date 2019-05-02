# -*- coding: ms949 -*-
"""
Created on Fri Jan 27 09:31:47 2012
@author: dkna
"""

import sys
import SemanticSimilarity
import copy
import threading
import multiprocessing
import MyUtil
import time
import copy
import os


DEBUG = False
WX_PYTHON = False
PARENT_RELATIONSHIP = 2 # 1 - all parents, 2 - exclude part_of

# not only category-defined GO terms, but also their children are included into the definition
AUTOMATIC_INDLUSION_OF_CHILD_TERMS_IN_DEFINITION = True
CATEGORY_EXPANSION = 1 # 1- expands category terms

try:
	import wx
except:
	WX_PYTHON = False


MPI = False

if os.path.exists('Categorizer.exe'):
	MPI = False
else:
	MPI = True


OPT_DEF_FILE = 'DEF_FILE'
OPT_DEF_FILE_FLAG = '-d'

OPT_ANNOTATION_FILE = 'ANNOTATION_FILE'
OPT_ANNOTATION_FILE_FLAG = '-a'

OPT_GO_FILE = 'GO_FILE'
OPT_GO_FILE_FLAG = '-g'

OPT_INPUT_FILE = 'INPUT_FILE'
OPT_INPUT_FILE_FLAG = '-i'

OPT_METHOD = 'METHOD'
OPT_METHOD_FLAG_MULTIPLE = '-m'
OPT_METHOD_FLAG_SINGLE = '-s'



OPT_METHOD_MULTIPLE = 'MultipleCategories'
OPT_METHOD_SINGLE = 'SingleCategory'
OPT_METHOD_THRESHOLD = 'Cutoff'


OPT_PROGRESS_BAR = 'PROGRESS_BAR'
OPT_BASAL_COUNT = 'Basal_Counter'
OPT_MAX_COUNT = 'Max_Counter'
OPT_PROGRESS_BAR_TEXT = 'ProgressBarText'

OPT_CPU = 'CPU'
OPT_CPU_FLAG = '-cpu'

CATEGORY_NO_ANNOTATION = 'Uncategorized'

THREAD_TERMINATION = False

CACHE = None







def printf(msg):
	global DEBUG
	if DEBUG:
		print(msg)


class JOB(threading.Thread):


	p = None
	over = 0
	args = None
	method = None
	proc = None

	def __init__(self):

		threading.Thread.__init__(self)
		self.p = None
		self.over =0
		self.args = None
		self.method = None
		self.proc = None

	def kill(self):
		self.proc.terminate()

	def set_args(self, user_method, user_args):
		self.method = user_method
		self.args = [ user_args ]

	def is_over(self):
		return self.over

	def run(self):
		self.over = 1

		self.proc = multiprocessing.Process( target=self.method, args=self.args )

		if DEBUG:
			self.proc.run()
		else:
			try:
				self.proc.start()
				self.proc.join()
			except Exception as inst:
				print('Error in multiprocess: ', repr(inst))
				sys.exit(1)

		self.over = 2


def loadGOcategoryDefinitionFile(fname):


	parent_nodes = {}
	r = {}
	
	# key = category, values = [ GO ID , ... ]
	
	
	f=open(fname,'r')

	current_category = None
	cnt = 0

	for s in f.readlines():
		s = s.replace('\n','').strip().replace('\t','')
		cnt += 1


		if len(s)>0:
			comment_index = s.find('#')

			line = ''

			if comment_index < 0:
				line = s
			elif comment_index == 0:
				continue
			else:
				line = s[ :comment_index].strip()



			if len(line)>0:
				if line[0] == '-':
					current_category = line[1:].strip()
					r[current_category] = []
				else:
					if line.find('GO:') == 0:
						if current_category is None:
							print('Line number: ', cnt)
							print('Category not specified yet: ', s)
							sys.exit(1)
						else:
							if not line in r[current_category]:
								r[current_category].append(line)

					else:
						print('Line number: ', cnt)
						print('Unidentified GO ID: ', s)
						sys.exit(1)


	f.close()



	key = sorted(r.keys())


	#printf ('Category # = ' + str(len(key)))
	#for k in key:
	#	printf ( k + ' -> ' + str(len(r[k])) + ' terms')

	

	#printf ('Checking duplicates...')
	r = checkDuplicates(r)

	
	
	if AUTOMATIC_INDLUSION_OF_CHILD_TERMS_IN_DEFINITION:
		#printf ('Adding child nodes')
		r = addChildren(r)
		print('Category # = ' + str(len(key)))
		for k in key:
			print(k + ' -> ' + str(len(r[k])) + ' terms')
	

	
	


	parent_nodes = r
	
	return r





def checkDuplicates(r):
	

	dupe = {}
	
	for cat in r.keys():
		for gid in r[cat]:
			for cat2 in r.keys():
				if cat != cat2 and gid in r[cat2]:
					printf (gid + ' in '+ cat+ ' is also in '+ cat2)
					dupe[gid] = None
					
	printf ('>> There are ' + str(len(dupe)) + ' duplicate GO terms in your category file')
	printf ('>> These will be removed.')
	
	for cat in r.keys():
		new_r = []
		for gid in r[cat]:
			if not gid in dupe :
				new_r.append(gid)
		r[cat] = new_r
		
	return r
		
		
						
def addChildren(r):
	
	printf ('Expanding GO structure...')
	

	sem = SemanticSimilarity.SEMANTIC_SIMILARITY()
	new_r = {}
	
	
	# add child terms to categories, and put them in 'new_r]
	
	for cat in r.keys():
		
		new_r [cat] = []
		
		for gid in r[cat]:
			if gid in sem.index :
				for child in sem.index[gid][SemanticSimilarity.CHILD]:
					
					
					passed = True
					for c in r.keys():
						if child in r[c]:
							passed = False
							break
						
							
					if passed:
						if not child in new_r[cat]:
							new_r[cat].append(child)

	# find GO terms belonging to two or more categories
	dupe = []
	for cat in new_r.keys():
		
		for gid in new_r[cat]:
			for cat2 in new_r.keys():
				if cat != cat2:
					if gid in new_r[cat2] or gid in r[cat2]:
						if not gid in dupe:
							dupe.append(gid)


	
	if CATEGORY_EXPANSION==1:
		print('Expanding category definition...')
		xr = addVagueChildNodes(r, dupe)
		
		for cat in xr.keys():
			r[cat] = xr[cat] + r[cat]	
	



			
	for cat in new_r.keys():
		n = []
		for gid in new_r[cat]:
			if not gid in dupe:
				n.append(gid)
		new_r[cat] = n

		r[cat] = new_r[cat] + r[cat]
	
	
	
	
	
	for cat in r.keys():
		for gid in r[cat]:
			for cat2 in r.keys():
				if cat != cat2:
					if gid in r[cat2]:
						print(gid + ' in '+ cat+ ' is also in '+ cat2)
	
	return r


def __overlaps(a, b):
	r = []
	for x in a:
		if x in b:
			r.append(x)
	return r

def addVagueChildNodes(r, dupe):
	
	
	
	threshold = 0.0
	
	new_r = {}
	for cat in r.keys():
		new_r[cat] = []
	
	sim = SemanticSimilarity.SEMANTIC_SIMILARITY()
	
	for gid in dupe:
		
		# classify gid
		#parents = getParents(gid)
		
		parents = sim.index[gid][SemanticSimilarity.PARENT]
		
		max_score = -1
		found_cat = None
		
		for cat in r.keys():
			
			max_score_in_cat = -1
			for pid in __overlaps(parents, r[cat]):
				score = sim.getSimilarity(gid, pid)
				if score > max_score_in_cat:
					max_score_in_cat = score
			
			if max_score_in_cat > max_score:
				max_score = max_score_in_cat
				found_cat = cat
				
		if found_cat is not None and max_score > threshold:
			new_r[found_cat].append(gid)
				
	
	return new_r
	


def getParents(go_term):

	box = []
	
	if PARENT_RELATIONSHIP == 2:
		isa = go_term.getParentRelationshipGOIDsOfISA()
		preg = go_term.getParentRelationshipGOIDsOfPOSITIVELYREGULATE()
		nreg = go_term.getParentRelationshipGOIDsOfNEGATIVELYREGULATE()
		reg = go_term.getParentRelationshipGOIDsOfREGULATE()
		box = [  isa, reg, preg, nreg ]
	elif PARENT_RELATIONSHIP == 1:
		eve = go_term.getParentRelationshipGOIDs()
		box = [ eve ]
		
	r = []
	for n in box:
		for  i in n:
			if not i in r:
				r.append(i)
	return r
#-------------------------------------------------------






def processMPI(cat_def, gene_list, org_goid_dict, option, cache = {}):

	global WX_PYTHON

	printf ('MPI cpu = ' + str( option[OPT_CPU]))

	gene_category={}
	max_cpu = option[OPT_CPU]
	queue = multiprocessing.Queue()
	prev_value = -1

	lists = MyUtil.divideList( gene_list, max_cpu )
	jobs_list = []



	multi_option = {}
	for o in option.keys():
		multi_option [o] = option [o]
	multi_option [ OPT_PROGRESS_BAR ] = None
	multi_option [ OPT_PROGRESS_BAR_TEXT ] = None
	printf ('Copied option ===')
	printf (repr(multi_option))



	for i in range(max_cpu):

		t = JOB()
		t.set_args( processMPIpart,
		            ( cat_def,
		              lists[i],
		              org_goid_dict,
		              multi_option,
		              queue,
		              cache) )
		jobs_list.append(t)

		printf ('gene #=' + str( len(lists[i]) ) )


	while(True):

		cnt = 0
		done = 0

		for m in range( len(jobs_list)):
			if jobs_list[m].is_over() == 1:
				cnt += 1
			elif jobs_list[m].is_over() == 2:
				done += 1




		new_run = max_cpu - cnt
		running = cnt
		for m in range( len(jobs_list)):
			if jobs_list[m].is_over() == 0 and new_run>0:
				jobs_list[m].start()

				new_run -= 1
				running += 1

		printf ('DONE=' + str (done) +  '/' +  str(len(jobs_list)) +  ' Running=' + str(running) +  ' :: ' + repr(time.ctime()))



		while( not queue.empty()):
			x = queue.get()
			for key in x.keys():
				gene_category[key] = x[key]
				printf (key + '\t' + repr(x[key]))




		if THREAD_TERMINATION:
			printf ('[***] User wanted to terminate categorization...')

			for ij in range(len(jobs_list)):
				jobs_list[ij].kill()

			return None


		if WX_PYTHON:

			pointer = len(gene_category)



			##############################
			gauge = option[OPT_PROGRESS_BAR]
			if gauge is not None:
				max_val = option[OPT_MAX_COUNT]
				cur_val = pointer + option[OPT_BASAL_COUNT]
				per = int( float(cur_val) * 100.0 / float(max_val) )
				if per != prev_value:
					gauge.Value = per
					prg = option[OPT_PROGRESS_BAR_TEXT]
					prg.Label = str(per).strip() + ' %'
					prev_value = per

					printf ('\t>> ' + str(per).strip() + ' %')

					gauge.Refresh()
		else:
			cur = len(gene_category)
			tot = len(gene_list)
			per = float(cur) / float(tot) * 100.0
			txt = '%.2f %% done' % per
			print(txt, '\r',)

		time.sleep(1)

		if done == len(jobs_list):
			break


	return gene_category



def processMPIpart(args ):

	cat_def, gene_list, org_goid_dict, option, queue, cache = args




	sem_sim = SemanticSimilarity.SEMANTIC_SIMILARITY()


	gene_category={}
	category_gene={}

	cnt = 0.0
	total = float( len(gene_list) )

	prev_value = -1


	for gene in gene_list:

		cnt += 1.0

		if THREAD_TERMINATION:
			return None



		if gene in cache :
			gene_category[gene] = copy.deepcopy( cache[gene] )
			queue.put( copy.deepcopy( gene_category ))



			gene_category = {}

			continue



		if not org_goid_dict.has_key(gene):
			categories = {}
			categories [ CATEGORY_NO_ANNOTATION ] = 1
			gene_category[gene]=categories

		else:
			goids = org_goid_dict[gene]

			categories = {}

			if len(goids) == 0:
				categories [ CATEGORY_NO_ANNOTATION ] = 2
			else:

				#cats = categorizeGeneWith(sem_sim, goids, cat_def, option)
				cats = categorizeGeneWithBAK(sem_sim, goids, cat_def, option)

				threshold = option[OPT_METHOD_THRESHOLD]


				if option[OPT_METHOD] == OPT_METHOD_SINGLE:
					
					'''
					max_score = -1.0
					max_category = None

					for ccc in cats.keys():
						if cats[ccc] > max_score:
							max_score = cats[ccc]
							max_category = ccc

					#if max_score == 0 :
					if max_score < threshold :
						categories [ CATEGORY_NO_ANNOTATION ] = 4.0
					else:
						categories [ max_category ] = max_score
					'''
					
					max_score = -1.0
					for ccc in cats.keys():
						if cats[ccc] > max_score:
							max_score = cats[ccc]
							
					for ccc in cats.keys():
						if cats[ccc] >= max_score and cats[ccc] >= threshold:
							categories[ccc] = cats[ccc]
					
					if len(categories) == 0:
						categories[ CATEGORY_NO_ANNOTATION ] = 4.0
						
				else:


					for ccc in cats.keys():
						if cats[ccc] >= threshold:
							categories [ ccc ] = cats[ccc]

					if len(categories) == 0:
						categories[ CATEGORY_NO_ANNOTATION ] = 3.0


			gene_category[gene]=categories

		queue.put( copy.deepcopy(gene_category))
		gene_category = {}









def process(cat_def, gene_list, org_goid_dict, option, cache = {}):



	
	

	print('Loading GO structure and GOterm similarity scores...')
	sem_sim = SemanticSimilarity.SEMANTIC_SIMILARITY()
	

	gene_category={}
	category_gene={}

	cnt = 0.0
	total = float( len(gene_list) )
	
	prev_value = -1
	
	
	for gene in gene_list:

		cnt += 1.0

		if THREAD_TERMINATION:
			printf ('[***] User wanted to terminate categorization...')
			return None


		print '\r', cnt/total*100,'%               ','\r', 


		

		if WX_PYTHON:
			

			
			gauge = option[OPT_PROGRESS_BAR]
			if gauge is not None:
				max_val = option[OPT_MAX_COUNT]
				cur_val = cnt + option[OPT_BASAL_COUNT]
				per = int( float(cur_val) * 100.0 / float(max_val) )
				if per != prev_value:
					gauge.Value = per
					prg = option[OPT_PROGRESS_BAR_TEXT]
					prg.Label = str(per).strip() + ' %'
					prev_value = per
					gauge.Refresh()
			

		if cache.has_key(gene):
			gene_category[gene] = copy.deepcopy( cache[gene] )
			continue
		
		
		
		if not org_goid_dict.has_key(gene):
			categories = {}
			categories [ CATEGORY_NO_ANNOTATION ] = 1
			gene_category[gene]=categories
			
		

		else:
			goids = org_goid_dict[gene]
			
			categories = {}
			
			if len(goids) == 0:
				categories [ CATEGORY_NO_ANNOTATION ] = 2
			else:
				
				cats = categorizeGeneWithBAK(sem_sim, goids, cat_def, option)
		
				threshold = option[OPT_METHOD_THRESHOLD]

				if option[OPT_METHOD] == OPT_METHOD_SINGLE:
					
					'''
					max_score = -1.0
					max_category = None
					
					for ccc in cats.keys():
						if cats[ccc] > max_score:
							max_score = cats[ccc]
							max_category = ccc
					
					#if max_score == 0 :
					if max_score < threshold:
						categories [ CATEGORY_NO_ANNOTATION ] = 4.0
					else:
						categories [ max_category ] = max_score
					'''
					
					max_score = -1.0
					for ccc in cats.keys():
						if cats[ccc] > max_score:
							max_score = cats[ccc]
							
					for ccc in cats.keys():
						if cats[ccc] >= max_score and cats[ccc] >= threshold:
							categories[ccc] = cats[ccc]
					
					if len(categories) == 0:
						categories[ CATEGORY_NO_ANNOTATION ] = 4.0
			
				else:

					
					for ccc in cats.keys():
						if cats[ccc] >= threshold:
							categories [ ccc ] = cats[ccc]
							
					if len(categories) == 0:
						categories[ CATEGORY_NO_ANNOTATION ] = 3.0
				
						
			gene_category[gene]=categories




	return gene_category




def loadAnnotationFile(annot_file):

	'''

	Annot을 dict 형태로 저장한다.

	'''

	gene_goid_dict = {}
	init = True

	f=open(annot_file,'r')

	while(True):
		s = f.readline()
		if not s:
			break

		s = s.replace('\n','').replace('\r','')

		if len(s) == 0: continue
		if s[0] == '!': continue
		if init:
			init = False
			continue


		x = s.split('\t')

		db_name = x[0].strip()
		uid = x[1].strip()
		name = x[2].strip()
		no = x[3].strip()
		go_id = x[4].strip()
		category = x[8].strip()
		desc = x[9].strip()


		if no == 'NOT':
			continue

		# unique id
		if not uid in gene_goid_dict :
			gene_goid_dict[ uid ] = []

		# gene name
		if not name in gene_goid_dict :
			gene_goid_dict[ name ] = []

		if not go_id in gene_goid_dict[uid]:
			gene_goid_dict[uid].append(go_id)

		if not go_id in gene_goid_dict[name]:
			gene_goid_dict[name].append(go_id)


	f.close()

	return gene_goid_dict





def reportStat(category_gene, output):

	f=open(output,'w')


	ccc = category_gene.keys()
	ccc.sort()

	total = 0.0
	for cat in ccc:
		genes = category_gene[cat]
		total = total + float( len(genes) )


	for cat in ccc:
		genes = category_gene[cat]


		per = float(len(genes))/total * 100.0
		s = cat + '\t' + str(len(genes)) + '\t' + str(per)
		f.write(s+'\n')

	f.close()

def reportCategoryGene(category_gene):
	#f=open(output,'w')


	ccc = category_gene.keys()
	ccc.sort()

	# ---------------------------------------------------
	new_ccc = []
	for c in ccc:
		if c != CATEGORY_NO_ANNOTATION:
			new_ccc.append(c)
	new_ccc.append(CATEGORY_NO_ANNOTATION)
	# ----------------------------------------------------
	

	res = []
	for cat in new_ccc:
		genes = category_gene[cat]

		s = cat + '\t' + str(len(genes)) + '\t' + ','.join(genes)
		res.append(s)
	return res


def reportGeneCategory(gene_category, output):
	
	f=open(output,'w')





	for gene in gene_category.keys():
		categories = gene_category[gene]

		box = []

		
		for c in sorted(categories, key=categories.get, reverse=True):
		
			score = categories[c]
			
			if c != CATEGORY_NO_ANNOTATION:
				box.append( c + '(' + str(score) + ')' )
			else:
				box.append( '-' )

		s = gene + '\t' + ','.join(box)
		f.write(s+'\n')

	f.close()



def report(all_categories, gene_category):


	category_gene = {}
	for c in all_categories:
		category_gene [c] = []

	
		
	for g in gene_category.keys():
		for c in gene_category[g].keys():
			if not g in category_gene[c]:
				category_gene[c].append(g)
				
	

	return reportCategoryGene(category_gene)
	#reportGeneCategory(gene_category, output+'_gene.txt')



def loadGeneList(fname):
	r = []
	f=open(fname,'r')
	for s in f.readlines():
		s = s.replace('\n','').strip()
		if len(s)>0:
			r.append(s)
	f.close()
	return r

def loadSimCache(sim_index, gene_list):
	
	r = {}
	
	gene_dict={}
	for g in gene_list:
		gene_dict[g]=None
	
	f=open(sim_index,'r')
	
	while(True):
		s = f.readline()
		
		if not s:
			break
		
		s = s.replace('\n','').strip()
		if len(s) == 0: continue

		x = s.split('\t')
		key = x[0]
		
		index = key.find(':',4)
		g1 = key[:index]
		g2 = key[index+1:]
		
		
		sim_score = eval(x[1])

		if gene_dict.has_key(g1) or gene_dict.has_key(g2):
			r[key] = sim_score

	f.close()

	return r

def run(option):


	def_file = option[OPT_DEF_FILE]
	input_file = option[OPT_INPUT_FILE]
	annot_file = option[OPT_ANNOTATION_FILE]
	method = option[OPT_METHOD]
	threshold = option[OPT_METHOD_THRESHOLD]
	cpu = option[OPT_CPU]
	sim_index = 'go_sim.txt'

	print('Loading category file: ', def_file)
	cat_def = loadGOcategoryDefinitionFile(def_file)

	print('Loading annotation file: ', annot_file)
	org_goid_dict = loadAnnotationFile(annot_file)

	print('Loading genes: ', input_file)
	gene_list = loadGeneList(input_file)

	#print 'Loading caches: ', sim_index
	#global CACHE
	#CACHE = loadSimCache(sim_index, gene_list)

	print('-----------------')
	print('Categorizing... ')
	ct = cat_def.keys()
	ct.sort()
	all_categories = ct + [ CATEGORY_NO_ANNOTATION ]


	if not MPI:
		cpu = 1

	gene_category = None

	if WX_PYTHON:
		if cpu == 1:
			gene_category = process(cat_def, gene_list, org_goid_dict, option)
		else:
			gene_category = processMPI(cat_def, gene_list, org_goid_dict, option)
	else:
		if cpu == 1:
			gene_category = process(cat_def, gene_list, org_goid_dict, option)
		else:
			gene_category = processMPI(cat_def, gene_list, org_goid_dict, option)

	out = input_file + '.result.txt'

	report(all_categories, gene_category, out)
	
	print()
	print('Done')













def getFromCache(g1, g2):

	global CACHE

	key = __makeAkey(g1, g2)
	try:
		score = CACHE[key]
		return score
	except:
		return 0.0



def __makeAkey(g1, g2):
	if g1>g2:
		return g1+':'+g2
	else:
		return g2+':'+g1



def sortParentsByProb(sem_sim, go):
	
	if sem_sim.index.has_key(go):
		genes = [go] + sem_sim.index[go][SemanticSimilarity.PARENT]
		for i in range( len(genes) - 1):
			for j in range(i+1, len(genes)):
				gi = genes[i]
				gj = genes[j]
				if sem_sim.prob[gi] > sem_sim.prob[gj]:
					genes[i] = gj
					genes[j] = gi
		return genes	
	else:
		return []
	
			
def findSiblings(p, category_goids, sem_sim, done):
	
	r = []
	
	for g in category_goids:
		
		if not done.has_key(g) and sem_sim.index.has_key(g):
			parents = [g] + sem_sim.index[g][SemanticSimilarity.PARENT]
			if p in parents:
				done[g] = None
				r.append(g)
				
	return r
				
		
	
	
def bit_faster(sem_sim, user_goid, category_goids, option):
	
	done = {}
	
	sortedParents = sortParentsByProb(sem_sim, user_goid)
	
	max_sim = 0.0
	threshold = option[OPT_METHOD_THRESHOLD]
	
	for p in sortedParents:
		
		max_sim_in = 0.0
		
		siblings = findSiblings(p, category_goids, sem_sim, done)
		# calculate semantic similarity
		
		for s in siblings:
		
			if sem_sim.index.has_key(user_goid) and sem_sim.index.has_key(s):
				
				sim_score = sem_sim.getSimilarity(user_goid, s)
				if sim_score>max_sim:
					max_sim = sim_score
				if sim_score>max_sim_in:
					max_sim_in = sim_score
					
		if max_sim_in < threshold or max_sim_in < max_sim:
			break
		
	return max_sim



def categorizeGeneWith(sem_sim, user_goids, cat_def, option):


	global CACHE

	goids = []

	'''
	for gid in user_goids:
		if not gid in [ 'GO:0005488', 'GO:0005634', 'GO:0005737', 'GO:0008150',\
		                'GO:0005622', 'GO:0016020', 'GO:0009987','GO:0005515' ]:
				
			goids.append(gid)
	'''
	goids = user_goids



	categories = {}
	for c in cat_def.keys():
		categories[c] = 0.0
	
	
	for gid in goids:
		

		for cat in cat_def.keys():

			#  to avoid unnecessary calculations
			if categories[cat] == 1.0: # already max
				continue

			done = {}
			
			categories[cat] = bit_faster(sem_sim, gid, cat_def[cat], option)
	


	return categories


def categorizeGeneWithBAK(sem_sim, user_goids, cat_def, option):


	global CACHE
	PARENTS_ONLY = True

	goids = []
	goids = user_goids


	categories = {}
	for c in cat_def.keys():
		categories[c] = -1.0
	
	
	for gid in goids:


		# calculate a semantic similarity of gid and its parents that are included in the user-defined categories
		parents = None
		if PARENTS_ONLY:
			if sem_sim.index.has_key(gid):
				parents = sem_sim.index[gid][SemanticSimilarity.PARENT]
		

		for cat in cat_def.keys():

			#  to avoid unnecessary calculations
			if categories[cat] == 1.0: # already max
				continue

			#done = {}
			
			
			for cgid in cat_def[cat]:
				
				
				
				#---------------------------------
				if PARENTS_ONLY:
					if parents is None:
						continue
					if (not cgid in parents) and (gid != cgid):
						continue
				#---------------------------------
				
				
				
				
				#if not done.has_key(cgid):

				if sem_sim.index.has_key(gid) and sem_sim.index.has_key(cgid):
					
					sim_score = -1.0

					if CACHE is None:
						sim_score = sem_sim.getSimilarity(gid, cgid)
					else:
						sim_score = getFromCache(gid, cgid)
					

					if categories[cat] < sim_score:
						categories[cat] = sim_score
						
				#done[cgid] = None
				
				# ----------------------------------------------------------------
				#if sem_sim.index.has_key(cgid):
				#	for pid in sem_sim.index[cgid][SemanticSimilarity.PARENT]:
				#		done[pid] = None
				#
				# ----------------------------------------------------------------

	


	return categories















def processOptions(argv):


	#--------------------------------------------------------
	option = {

	        OPT_DEF_FILE: None,
	    OPT_INPUT_FILE: None,
	    OPT_METHOD: None,
	    OPT_METHOD_THRESHOLD: None,
	    OPT_ANNOTATION_FILE: None,
	    OPT_PROGRESS_BAR: None,
	    OPT_CPU: 1
	}



	for i in range( 0, len(argv), 2 ):

		flag = argv[i]
		value = argv[i+1]


		if flag == OPT_DEF_FILE_FLAG:
			option[OPT_DEF_FILE] = value
		elif flag == OPT_GO_FILE_FLAG:
			option[OPT_GO_FILE] = value
		elif flag == OPT_INPUT_FILE_FLAG:
			option [OPT_INPUT_FILE] = value
		elif flag == OPT_ANNOTATION_FILE_FLAG:
			option [OPT_ANNOTATION_FILE] = value
		elif flag == OPT_CPU_FLAG:
			option [OPT_CPU] = eval(value)
		elif flag == OPT_METHOD_FLAG_MULTIPLE:
			option[OPT_METHOD] = OPT_METHOD_MULTIPLE
			option[OPT_METHOD_THRESHOLD] = float( value)
		elif flag == OPT_METHOD_FLAG_SINGLE:
			option[OPT_METHOD] = OPT_METHOD_SINGLE
			option[OPT_METHOD_THRESHOLD] = float( value)

		else:
			print('Error: ', flag, value)
			sys.exit(1)


	keys = option.keys()
	keys.sort()

	print('---------------')
	print(' Parameters...')
	print('---------------')
	for k in keys:
		print(k, ':', option[k])
	print('---------------')



	return option



def displayHelp():
	print('''
Categorizer v1.0
	- categorizes genes/proteins based on GO annotations

[parameters]
	-d [category file]
		User-specified category definition file
	-a [annotation file]
		Gene-GO annotation file downloaded from GeneOntology
	-i [gene list file]
		A list of genes to categorize

	-m [threshold]
		With this paramter, genes can belong into multiple
			categories with a cutoff value of 'threshold'.
			(0 < threshold <= 1.0)
	-s [threshold]
		With this paramter, genes belong to only one category that 
	                has the highest similarity score.
			(0 < threshold <= 1.0)

	* -m and -s are mutually exclusive!

''')
	
	if MPI:
		print ('''
	*optional*
	-cpu [n]
		integer n>=1. Uses multiple cores (default = 1)
		
Example:
		
	python Categorizer.py -d ./data/example_categories.txt -a ./data/example_gene_association.fb -i ./data/example_genes.txt -m 0.3 -cpu 3
''')
		
	else:
		print('''
Example:
	Categorizer.exe -d .\\data\\example_categories.txt -a .\\data\\example_gene_association.fb -i .\\data\\example_genes.txt -m 0.3
''')

def checkOptions(opt):


	if opt[OPT_DEF_FILE] is not None and \
		opt[OPT_INPUT_FILE] is not None and \
		opt[OPT_ANNOTATION_FILE] is not None and \
	        opt[OPT_METHOD_THRESHOLD] is not None:

		return True
	else:
		return False


if __name__ == '__main__':




	print()
	print('[[ Categozier v1.0]] ')

	# 초기화 함수

	if len(sys.argv) >= 6:

		opt = processOptions( sys.argv[1:] )
		if checkOptions(opt):
			run(opt)
		else:
			print('*** Error ***')
			print('You must specify category file(-d), annotation file(-a), input file(-i), and category method (-m/-s).')
	else:
		displayHelp()
	#print 'DONE'

