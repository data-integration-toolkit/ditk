# -*- coding: ms949 -*-
'''

GO 파일을 읽어서 가지고 있는다.
이를 이용해 다른 계산에 쓴다.

'''

EXE_PATH = '.'

GO_FILE = EXE_PATH + '/gene_ontology.obo'
GO_EXT_FILE = EXE_PATH + '/gene_ontology_ext.obo'

class GOTerm():

	go_id = ''

	RELATION_IS_A = 'is_a'
	RELATION_PART_OF = 'part_of'
	RELATION_REGULATES = 'regulates'
	RELATION_NEGATIVELY_REGULATES = 'negatively_regulates'
	RELATION_POSITIVELY_REGULATES = 'positively_regulates'

	CATEGORY_PROCESS = 'biological_process'
	CATEGORY_FUNCTION = 'molecular_function'
	CATEGORY_COMPONENT = 'cellular_component'

	is_obsolete = False

	name = ''
	definition = ''
	name_space = ''


	relation = { }
	child_relation = {}

	alt_ids = []

	# =============
	count = {}
	flyids = {}
	flyids_cnt = {}


	def __init__(self):
		self.go_id = ''
		self.name = ''
		self.name_space = ''
		self.definition = ''

		self.relation = {}
		self.child_relation = {}

		self.count = {}
		self.flyids = {}
		self.flyids_cnt = {}

		self.is_obsolete = False
		self.alt_ids = []



	def addAlternativeID(self, aid):
		self.alt_ids.append(aid)
	def getAlternativeIDs(self):
		return self.alt_ids

	def setObsolete(self, obsolete):
		self.is_obsolete = obsolete
	def isObsolete(self):
		return self.is_obsolete

	def setGOID( self, go_id ):
		self.go_id = go_id
	def getGOID(self):
		return self.go_id

	def setGOName(self, name):
		self.name = name
	def getGOName(self): return self.name
	def setGODefinition(self, definition):
		self.definition = definition
	def getGODefinition(self):
		return self.definition

	def setGONameSpace(self, namespace):
		self.name_space = namespace
	def getGONameSpace(self):
		return self.name_space

	def addParentRelationship(self, go_id, relation):
		self.relation [ go_id ] = relation
	def getParentRelationshipSize(self):
		return len( self.relation )


	def getParentRelationshipGOIDs(self):
		''' relationship 에 저장된 모든 GO id를 넘겨준다. List '''
		return self.relation.keys()

	def getParentRelationshipGOIDsOfISA(self):
		r=[]
		for k in self.relation.keys():
			if self.relation[k] == self.RELATION_IS_A :
				r.append(k)
		return r


	def getParentRelationshipGOIDsOfREGULATE(self):
		r=[]
		for k in self.relation.keys():
			if self.relation[k] == self.RELATION_REGULATES :
				r.append(k)
		return r

	def getParentRelationshipGOIDsOfNEGATIVELYREGULATE(self):
		r=[]
		for k in self.relation.keys():
			if self.relation[k] == self.RELATION_NEGATIVELY_REGULATES :
				r.append(k)
		return r


	def getParentRelationshipGOIDsOfPOSITIVELYREGULATE(self):
		r=[]
		for k in self.relation.keys():
			if self.relation[k] == self.RELATION_POSITIVELY_REGULATES :
				r.append(k)
		return r





	def getChildRelationshipGOIDsOfISA(self):
		r=[]
		for k in self.child_relation.keys():
			if self.child_relation[k] == self.RELATION_IS_A :
				r.append(k)
		return r


	def getChildRelationshipGOIDsOfPARTOF(self):
		r=[]
		for k in self.child_relation.keys():
			if self.child_relation[k] == self.RELATION_PART_OF :
				r.append(k)
		return r


	def getChildRelationshipGOIDsOfREGULATE(self):
		r=[]
		for k in self.child_relation.keys():
			if self.child_relation[k] == self.RELATION_REGULATES :
				r.append(k)
		return r


	def getChildRelationshipGOIDsOfNEGATIVELYREGULATE(self):
		r=[]
		for k in self.child_relation.keys():
			if self.child_relation[k] == self.RELATION_NEGATIVELY_REGULATES :
				r.append(k)
		return r

	def getChildRelationshipGOIDsOfPOSITIVELYREGULATE(self):
		r=[]
		for k in self.child_relation.keys():
			if self.child_relation[k] == self.RELATION_POSITIVELY_REGULATES :
				r.append(k)
		return r

	def addChildRelationship(self, go_id, relation):
		self.child_relation [ go_id ] = relation
	def getChildRelationshipSize(self):
		''' 저장된 relationship 전체 개수를 넘겨준다 '''
		return len( self.child_relation )
	def getChildRelationshipGOIDs(self):
		''' relationship 에 저장된 모든 GO id를 넘겨준다. List '''
		return self.child_relation.keys()


	def getParentRelationshipOf(self, go_id):
		''' GO ID값을 넣어주면 해당하는 relationship 값을 돌려준다 '''
		return self.relation[go_id]

	def getChildRelationshipOf(self, go_id):
		''' GO ID값을 넣어주면 해당하는 relationship 값을 돌려준다 '''
		return self.child_relation[go_id]


	def toString(self):
		r = 'ID:\t' + self.getGOID() + '\t' + \
		  'Name:\t' + self.getGOName() + '\t' + \
		  'NameSpace\t'+ self.getGONameSpace() + '\t' + \
		  'Definition\t' + self.getGODefinition() + '\t' + \
		  'Relation\t'

		for ids in self.getParentRelationshipGOIDs():
			r = r + '\t' + ids + '\t' + self.getParentRelationshipOf(ids) + '\n'

		for k in self.count.keys():
			r = r + k + '\t' + str( self.count[k] ) + '\n'

		r = r + '====================================='
		return r


	def increasePoint(self, cnt_id, point, flyDB_id):
		if not self.count.has_key(cnt_id):
			self.count [cnt_id ] = 0

		if not self.flyids.has_key(cnt_id):
			self.flyids[cnt_id] = []
			self.flyids_cnt[cnt_id] = 0

		self.count [cnt_id] = self.count [cnt_id] + point

		if not flyDB_id in self.flyids[cnt_id]:
			self.flyids[cnt_id].append( flyDB_id )
			self.flyids_cnt[cnt_id] = self.flyids_cnt[cnt_id] + 1


	def getCount(self, cnt_id):

		if self.count.has_key (cnt_id) :
			return self.count [cnt_id]
		else:
			return 0.0

	def getFlyDBIDSizeOf(self, cnt_id):
		if self.flyids_cnt.has_key(cnt_id):
			return self.flyids_cnt[cnt_id]
		else:
			return 0

	def getFlyDBIDs(self, cnt_id):
		if self.flyids.has_key(cnt_id):

			return self.flyids[cnt_id]
		else:
			return []




	# =====================================

class GO():

	go_terms = {}

	def __init__(self, obo_file = GO_EXT_FILE):
		self.go_terms = {}
		self.loadFile(obo_file)

	def writeGO2FlyDBID(self, fname):

		f = open(fname, 'w')

		for g_id in self.go_terms.keys():
			go = GOTerm()
			go = self.go_terms[g_id]
			s = g_id + '\t' + go.getGOName() + '\t'

			mm = go.getFlyDBIDSizeOf('m')
			nn = go.getFlyDBIDSizeOf('x')

			if mm>30:
				s = s + '\t [ '+ str(mm)+' / ' + str(nn) + ' ] 많아서 생략'
				f.write(s+'굈')
			elif mm > 0:
				x = '\t [ ' + str(mm)+ ' / ' + str(nn)+ ' ] ' + '\n'
				x = x + '\tModifiers = '

				for m in go.getFlyDBIDs('m'):
					x = x + ',' + m

				x = x + '\n'
				x = x + '\nNonmodifiers = '

				for m in go.getFlyDBIDs('x'):
					x = x + ',' + m

				s = s + x

				f.write(s+'\n')

		f.close()


	def size(self):
		return len(self.go_terms)

	def getGOTermOf(self, go_id):
		if self.go_terms.has_key(go_id):
			return self.go_terms[go_id]
		else:

			for k in self.go_terms.keys():
				aids = self.go_terms[k].getAlternativeIDs()
				if go_id in aids:
					return self.go_terms[k]

			return None

	def loadFile(self, fname):


		f=open(fname,'r')

		flag = 0
		goterm = None

		for s in f.readlines():
			s=s.replace('\n','')


			if len(s)>0:
				if s == '[Term]' or s == '[Typedef]':
					flag = 1
					if goterm != None:
						self.go_terms [ goterm.getGOID() ] = goterm


					goterm = GOTerm()

				else:
					if flag == 1:
						x = s.split(":")
						if x[0] == 'id':
							y = s.split(' ')
							goterm.setGOID( y[1] )
						elif x[0] == 'name':
							goterm.setGOName( x[1].strip() )
						elif x[0] == 'namespace':
							goterm.setGONameSpace( x[1] )
						elif x[0] == 'def':
							goterm.setGODefinition( x[1].strip().replace('"','') )
						elif x[0] == 'alt_id':
							y = s.split(' ')
							aid = y[1].strip()
							goterm.addAlternativeID(aid)

						elif x[0] == 'is_obsolete':
							if x[1].strip() == 'true':
								goterm.setObsolete(True)
						elif x[0] == 'is_a':
							y = s.split(' ')
							goterm.addParentRelationship( y[1], goterm.RELATION_IS_A )
						elif x[0] == 'relationship':
							y = s.split(' ')
							rel = y[1]
							gid = y[2]

							if rel == 'regulates' :
								goterm.addParentRelationship( gid, goterm.RELATION_REGULATES )
							elif rel == 'part_of':
								goterm.addParentRelationship( gid, goterm.RELATION_PART_OF )
							elif rel == 'negatively_regulates':
								goterm.addParentRelationship( gid, goterm.RELATION_NEGATIVELY_REGULATES)
							elif rel == 'positively_regulates':
								goterm.addParentRelationship( gid, goterm.RELATION_POSITIVELY_REGULATES)

							else:
								pass





		f.close()

		self.assignRelation()

	def assignRelation(self):


		g = GOTerm()

		for g_id in self.go_terms.keys():

			g = self.go_terms[g_id]

			v = g.getParentRelationshipGOIDs()

			for p_gi in v:
				pg = GOTerm()
				pg = self.getGOTermOf( p_gi )


				pg.addChildRelationship( g_id, g.getParentRelationshipOf(p_gi) )



	def printAll(self):
		for k in self.go_terms.keys():
			g = GOTerm()
			g = self.go_terms[k]
			print(g.toString())

	def increaseCounter (self, go_id, cnt_id, flyDB_id):
		g = GOTerm()
		g = self.go_terms[go_id]

		g.increaseCount()


		parent_ids = g.getParentRelationshipGOIDs()

		for p in parent_ids:
			i = g.getParentRelationshipOf(p)
			if i == g.RELATION_IS_A:
				self.increaseCounter( p , cnt_id )

	def increasePoint (self, go_id, cnt_id, point, flyDBID):
		g = GOTerm()
		g = self.go_terms[go_id]

		g.increasePoint( cnt_id, point, flyDBID )


		parent_ids = g.getParentRelationshipGOIDs()

		cnt = 0
		for p in parent_ids:
			i = g.getParentRelationshipOf(p)
			if i == g.RELATION_IS_A:
				cnt = cnt + 1

		for p in parent_ids:
			i = g.getParentRelationshipOf(p)
			if i == g.RELATION_IS_A:


				self.increasePoint( p, cnt_id, point, flyDBID  )

	def findRelation(self, go1_child, go2_parent):

		storate = self.__findRelation(go1_child, go2_parent)


		return storate


	def getDistance(self, go_child, go_parent):



		box = self.findRelation(go_child, go_parent)
		if len(box) == 0:
			return -1
		else:
			distance = 0
			for d, goid in box:
				if d<distance:
					distance = d
			return distance



	def __findRelation(self, go1, go2):

		storage = []

		if go1 == go2:
			storage =  [ [ 0, go2 ] ]
			return storage

		self.__traceUp(0, go1, go2, storage)

		return storage

	def __traceUp(self, cnt, go1, go2, storage):

		go1_term = self.getGOTermOf(go1)

		if go1_term is None:
			return


		parent_ids = go1_term.getParentRelationshipGOIDsOfISA()

		cnt  += 1

		for p in parent_ids:


			if p == go2:
				storage.append( [ cnt, go2 ] )
			else:

				self.__traceUp( cnt, p, go2, storage)







