# -*- coding: ms949 -*-

import GO
import sys
import MyUtil
import SemanticSimilarity
import Categorizer

DEBUG = True
CHILD = 'C'
PARENT = 'P'

GO_MF = 'GO:0003674'
GO_CC = 'GO:0005575'
GO_BP = 'GO:0008150'

def msg(s):
	if DEBUG:
		print s







def __getParents(gt, current_go_id):
	
	# NOPE: include IS_A, REGULATES, NEG_REGULATES, POS_REGLATES
	# include all parents
	
	box = {}
	
	#gt = GO.GOTerm()
	

	
	if Categorizer.PARENT_RELATIONSHIP == 2:
		# exclude part_of
		for g in gt.getParentRelationshipGOIDsOfISA():
			box[g] = None
		for g in gt.getParentRelationshipGOIDsOfREGULATE():
			box[g] = None
		for g in gt.getParentRelationshipGOIDsOfNEGATIVELYREGULATE():
			box[g] = None
		for g in gt.getParentRelationshipGOIDsOfPOSITIVELYREGULATE():
			box[g] = None

	elif Categorizer.PARENT_RELATIONSHIP == 1:
		# include all parent nodes
		for g in gt.getParentRelationshipGOIDs():
			box[g] = None	
		
	return box.keys()
	
	
	

def __traceUp(go_class, box, user_go_id, current_go_id):

	gt = go_class.getGOTermOf(current_go_id)


	parents = __getParents(gt, current_go_id)
	

	for pid in parents:

		if not box.has_key(user_go_id):
			box[user_go_id] = {
				PARENT: [],
				CHILD: []
				}

		if not pid in box[user_go_id][PARENT]:
			box[user_go_id][PARENT].append( pid )
		__traceUp(go_class, box, user_go_id, pid)


def __getChildren(gt, current_go_id):
	
	# include IS_A, REGULATES, NEG_REGULATES, POS_REGLATES
	
	box = {}
	
	
	if Categorizer.PARENT_RELATIONSHIP == 2:
		for g in gt.getChildRelationshipGOIDsOfISA():
			box[g] = None
		for g in gt.getChildRelationshipGOIDsOfREGULATE():
			box[g] = None
		for g in gt.getChildRelationshipGOIDsOfNEGATIVELYREGULATE():
			box[g] = None
		for g in gt.getChildRelationshipGOIDsOfPOSITIVELYREGULATE():
			box[g] = None
	elif Categorizer.PARENT_RELATIONSHIP == 1:
		for g in gt.getChildRelationshipGOIDs():
			box[g] = None
		
	
	
	return box.keys()	

def __traceDown(go_class, box, user_go_id, current_go_id):

	gt = go_class.getGOTermOf(current_go_id)
	
	children = __getChildren(gt, current_go_id)
	

	for pid in children:

		if not box.has_key(user_go_id):
			box[user_go_id] = {
				PARENT: [],
				CHILD: []
				}
		if not pid in box[user_go_id][CHILD]:
			box[user_go_id][CHILD].append( pid )
		__traceDown(go_class, box, user_go_id, pid)


def __indexingGO(go_class):




	box = {}
	'''
	box[GO_ID] = { 'P': [GOID, GOID, ...], 'C':[GOID, GOID, ...] }
	'''

	for gid in go_class.go_terms.keys():

		__traceUp(go_class, box, gid, gid)

		__traceDown(go_class, box, gid, gid)

	return box


def __saveIndex(index, output):
	f=open(output, 'w')

	for goid in index.keys():

		s = goid + '\t' + \
			','.join( index[goid][PARENT] ) + '\t' + \
			','.join( index[goid][CHILD] )
		f.write(s + '\n' )

	f.close()


def __loadIndex(input_file):
	index = {}

	f=open(input_file,'r')

	for s in f.readlines():
		s = s.replace('\n','')
		if len(s) == 0: continue

		x = s.split('\t')
		goid = x[0] # go id
		pids = x[1].split(',') # parents
		cids = x[2].split(',') # children

		index[goid] = {
			PARENT: pids,
			CHILD: cids
			}

	f.close()

	return index




def __loadUniProtAnnotationFile(input_file):

	box = {}

	import os
	fsize = os.path.getsize(input_file)
	pos = 0
	cnt = 0

	f = open(input_file, 'r')

	while(True):
		s = f.readline()
		pos += len(s)

		cnt += 1
		if cnt == 100:
			per = round( float(pos) / float(fsize) * 100.0, 2)
			print per,'%     ', '\r',
		
		
		if not s:
			break

		s = s.replace('\n','').strip()
		if len(s) == 0: continue
		if s[0] == '!': continue

		x = s.split('\t')
		uid = x[2]
		gid = x[4]

		if not box.has_key(uid):
			box[uid] = []

		box[uid].append(gid)


	f.close()

	print '100.00 %'


	return box


def __precalculate(index, uniprot_go_map):

	go_p = {}
	for gid in index.keys():
		go_p[gid] = 0.0


	__precalculate2(go_p, index, uniprot_go_map, GO_MF)
	__precalculate2(go_p, index, uniprot_go_map, GO_CC)
	__precalculate2(go_p, index, uniprot_go_map, GO_BP)



	return go_p


def __precalculate2(go_p, index, uniprot_go_map, category):

	total = 0.0

	for gid in index.keys():

		pids = index[gid][PARENT]

		if category in pids or category == gid:
			total += 1.0

			go_p[gid] += 1.0

			for p in pids:
				go_p[p] += 1.0

	for gid in index.keys():

		pids = index[gid][PARENT]

		if category in pids or gid == category:

			if 0<go_p[gid]<1:
				print 'What? ', gid, go_p[gid]
				sys.exit(1)

			go_p[gid] = (1.0 + go_p[gid] ) / (1.0 + total )


			if go_p[gid]>1:
				print 'Larger than 1: ', gid, go_p[gid]
				go_p[gid] = 1.0



def __saveProbability(go_prob, output):

	f=open(output, 'w')

	for gid in go_prob.keys():
		s = gid + '\t' + str(go_prob[gid])
		f.write(s+'\n')

	f.close()


def __makeAkey(g1, g2):
	if g1>g2:
		return g1+':'+g2
	else:
		return g2+':'+g1


def __saveSimilarityScores(go_prob, go_index, go_sim_index_file):

	all_go_terms = go_prob.keys()

	# please note that sim class uses the created go_index and go_prob files during this run.
	sim = SemanticSimilarity.SEMANTIC_SIMILARITY()

	#sim_index = {}

	per = 0
	f=open(go_sim_index_file,'w')
	total = float(len(all_go_terms))

	for g1 in range(len(all_go_terms)):

		per = float(g1+1)/total*100.0
		print per, '%              ', '\r',

		for g2 in range(g1, len(all_go_terms)):
			g1_term = all_go_terms[g1]
			g2_term = all_go_terms[g2]

			parents_of_g1 = go_index[g1_term][PARENT]
			parents_of_g2 = go_index[g2_term][PARENT]

			if (GO_MF in parents_of_g1 and parents_of_g2) or \
				 (GO_BP in parents_of_g1 and parents_of_g2) or \
				 (GO_CC in parents_of_g1 and parents_of_g2):
				# they are in the same GO tree
				score = sim.getSimilarity(g1_term, g2_term)
				key = __makeAkey(g1_term, g2_term)
				#sim_index[key] = score

				if score != 0.0:
					s = key + '\t' + str( score )
					f.write(s+'\n')

	print
	# saving...
	#f=open(go_sim_index_file, 'w')
	#for key in sim_index.keys():
	#	s = key + '\t' + str( sim_index[key])
	#	f.write(s+'\n')
	f.close()





def process(uniprot_go_annotation_file, go_file, index_file, go_prob_file, go_sim_index_file):


	msg('Loading GO: ' + go_file)
	go_class = GO.GO(go_file)

	msg('Indexing...')
	index = __indexingGO(go_class)

	__saveIndex(index, index_file)


	msg('Loading UniProt annotations: ' + uniprot_go_annotation_file)
	uniprot_go_map = __loadUniProtAnnotationFile(uniprot_go_annotation_file)

	msg('Calculating probabilities...')
	go_prob = __precalculate( index, uniprot_go_map)
	__saveProbability(go_prob, go_prob_file)



	# two index files (go_prob.txt and go_index.txt) were already generated.
	# Using these files, create GO-GO semantic similarity scores.
	# __saveSimilarityScores(go_prob, index, go_sim_index_file)

	msg('done')


def run(uniprot_annotation_file, go_file):
	print 'Generating information contents from uniprot data...'

	index_file = 'go_index.txt'
	go_prob_file = 'go_prob.txt'
	go_sim_index_file = 'go_sim.txt'

	print 'Inputs...'
	print '\tUniprot data = ', uniprot_annotation_file
	print '\tGO data = ', go_file
	print
	print 'Outputs...'
	print '\tIndex file = ', index_file
	print '\tInformation contents file = ', go_prob_file
	print '\tSemantic Similarity file = ', go_sim_index_file


	process(uniprot_annotation_file, go_file, index_file, go_prob_file, go_sim_index_file)

	print 'Done'

def display():

	print '''
-----------------------------
Indexing tool for Categorizer
-----------------------------

This tool generates index files (go_index.txt, go_prob.txt, go_sim.txt)
from UniProt GO annotation file and GeneOntology file.

Example)
	python rebuild.py ./data/gene_association.goa_uniprot_noiea.txt ./data/gene_ontology_ext.obo

	You can download above two files at
	http://www.geneontology.org/GO.downloads.annotations.shtml

	Copy generated go_index.txt, go_prob.txt and go_sim.txt 
	to Categorizer or CategorizerGUI folder.
	Those index files will be automatically loaded by the 
	CategorizerGUI and Categorizer.
'''


if __name__ == '__main__':

	if len(sys.argv) == 3:
		uni_file = sys.argv[1]
		go_file = sys.argv[2]
		run(uni_file, go_file)
	else:
		display()
