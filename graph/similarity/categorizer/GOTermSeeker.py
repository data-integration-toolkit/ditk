# -*- coding: ms949 -*-


import sys
import GO



go = None


def seekGOTerms(keywords, output_file = None):
	'''
	keywords: A list of words to search for.
	'''

	ret = {} # Key: GO ID, Value: Term
	
	
	for gid in go.go_terms.keys():
		gx = GO.GOTerm()
		gx = go.getGOTermOf(gid)
		
		desc = gx.getGOName()
		
		
		for k in keywords:
		
			k = k.upper()
			
			
			if desc.upper().find(k) >= 0:
				ret[gid] = desc
				print gid, desc
				break
			
			
		
	
	f=None
	if output_file is not None:
		f=open(output_file, 'w')
	
	for gid in ret.keys():
		s = '\t'+gid+'\t# '+ret[gid]
		if f is not None:
			f.write(s+'\n')
		else:
			print s
		
		
	if f is not None:
		f.close()
	






def displayHelp():

	print '''
	This tool searches for GO terms that have a particular keyword.

	python GOTermSeeker.py [GeneOntology file] [keyword] [output file]

	example: python GOTermSeeker.py ./data/gene_ontology_ext.obo "cell cycle" temp.txt

'''

if __name__ == '__main__':

	'''
	모든 GOTerm을 뒤져서 원하는 keyword가 있으면 다 모은다.
	'''

	#keywords = [ 'mitochondria', 'mitochondrion' ]
	#keywords = [ 'lysosome', 'lysosomal', 'phagocytosis']
	#keywords = ['autophagy' ]
	#keywords = ['endocytosis', 'endosomal', 'endosome' ]
	#keywords = ['transcription', 'translation', 'transcribed' ]
	#keywords = ['cell cycle', 'division' ]
	#keywords = [ 'cytoskeleton' ]
	#keywords = [ 'metabolism' ]
	#keywords = ['Protein folding', 'chaperone', 'heat shock' ]
	#keywords = ['proteolysis', 'proteasom', 'ubiquitin' ]
	#keywords = [ 'splicing', 'spliceosom' ]
	#keywords = [ 'transport', 'localiz' ]

	#keywords = [ 'oxidoreduct' ]
	
	#output_file = 'r:/tmp.txt'


	if len(sys.argv) == 4:

		global go

		go_file = sys.argv[1]
		keyword = sys.argv[2].replace('"', '')
		keywords = [keyword]
		output_file = sys.argv[3]

		go = GO.GO(obo_file = go_file)

		seekGOTerms(keywords, output_file)

		print 'DONE'
	else:
		displayHelp()
