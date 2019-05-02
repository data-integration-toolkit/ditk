import os,sys

import gensim
import gensim.models

from scipy import spatial
from graph.embedding.graph_embedding import GraphEmbedding
#from graph.embedding.opa2vec import eval_main





class OPA2VEC(GraphEmbedding):

    @classmethod
    def read_dataset(self, fileNames, options = {}): 
        """
		Reads datasets and convert them to proper format for train or test. Returns data in proper format for train, validation and test.

		Args:
			fileNames: list-like. List of files representing the dataset to read. Each element is str, representing
            				filename. The required list of files are
            1.ontology file in OWL
            2.association file specifiying association of entity to the class in ontology
                           
            
		Returns:
			data: data in proper [arbitrary] format contiaining ontology and association details.
		Raises:
			None
		"""
        #set up variables:
        ontology_file = fileNames[0]
        association_file = fileNames[1]
        reasoner= "elk"
        
        
        print("***********OPA2Vec Running ...***********\n")
        print("***********Ontology Processing ...***********\n")
        commandF ="groovy ProcessOntology.groovy " + str(ontology_file) +" "+str(reasoner)
        os.system(commandF)
        print("***********Ontology Processing Complete ...***********\n")
        
        with open(ontology_file) as f:
            ontology = f.read().splitlines()
            
        with open(association_file) as f:
            association = f.read().splitlines()
        
        return ontology, association
    
        


    @classmethod
    def learn_embeddings(self,data, options = {}):  #<--- implemented PER class
        """
		Learns embeddings with data, build model and train the model

		Args:
			data: Intermediate list of files for creating embeddings
				
            options['output_file_path'] : Specifiy the output the path
            options['path_of_pretrained_word2vec'] = Specify path of word2vec model trained on pubmed extracts
		Returns:
			ret: List of Embeddings

		Raises:
			None
		"""
        
        ditk_path = ""
        for path in sys.path:
            if "ditk" in path:
                ditk_path = path
            print(ditk_path)

        ontology_file = data[0]
        association_file= data[1]
        classes_file=ditk_path+"/graph/embedding/opa2vec/finalclasses.lst"
        #window=5
        #embedding=200
        mincoun=0
        model= "sg"
        pretrained = ditk_path+"/graph/embedding/opa2vec/RepresentationModel_pubmed.txt"
        listofuri="all"
        outfile = data[2]
        
        commandExtra1="perl getclasses.pl "+str(association_file)
        os.system(commandExtra1)
        commandMerge ="cat axiomsorig.lst axiomsinf.lst > axioms.lst"
        os.system(commandMerge)
        print("***********Metadata Extraction ...***********\n")
        commandS ="groovy getMetadata.groovy "+ str(ontology_file)+" "+str(listofuri)
        os.system(commandS)
        print("***********Metadata Extraction Complete ...***********\n")
        print("***********Propagate Associations through hierarchy ...***********\n")
        commandT="perl AddAncestors.pl "+ str(association_file)
        os.system(commandT)
        commandF= "cat "+str(association_file)+" associationAxiomInferred.lst > AllAssociations1.lst"
        os.system(commandF)
        Addacommand="sort -u AllAssociations1.lst > AllAssociations.lst"
        os.system(Addacommand)
        print("***********Association propagation Complete ...***********")
        print("***********Corpus Creation ...***********\n")
        commandFif="cat axioms.lst metadata.lst AllAssociations.lst  > ontology_corpus.lst"
        os.system(commandFif)
        print("***********Corpus Creation Complete ...***********\n")
        print("***********Running Word2Vec ...*********** \n")
        myclasses = classes_file
        #mywindow= window
        #mysize= embedding
        mincount=mincoun
        model = model
        pretrain= pretrained
        outfile=outfile
        mymodel=gensim.models.Word2Vec.load (pretrain)
        mymodel.min_count = mincount
        sentences =gensim.models.word2vec.LineSentence('ontology_corpus.lst')
        mymodel.build_vocab(sentences, update=True)
        #mymodel =gensim.models.Word2Vec(sentences,sg=0,min_count=0, size=200 ,window=5, sample=1e-3)
        mymodel.train (sentences,total_examples=mymodel.corpus_count, epochs=100)
        #print (len(mymodel.wv.vocab));
        # Store vectors for each given class
        word_vectors=mymodel.wv
        final_vec = []
        file= open (outfile, 'w')
        with open(myclasses) as f:
            for line in f:
                myclass1=line.rstrip()
                if myclass1 in word_vectors.vocab:		
                #myvectors[myclass1]=mymodel[myclass1]
                     file.write(str(myclass1) +' '+ str(mymodel[myclass1]) +'\n')
                     temp = mymodel[myclass1]
                     final_vec.append(temp)
            file.close()
        print("***********Vector representations available at AllVectorResults.lst ***********\n")
        print("***********OPA2Vec Complete ...***********\n")
        
        #Read the embedding file and return
        #with open(outfile) as f:
        #    embedding = f.read().splitlines()
            
        return final_vec


    @classmethod
    def evaluate(self,data, options = {}):  #<--- common ACROSS ALL classes. Requirement that INPUT format uses output from predict()!
        """
		Calculate the cosine similarity among the elements

		Args:
			data: embeddings
			

		Returns:
			metrics: cosine similarity

		Raises:
			None
		"""
        
        element1 = data[0]
        element2 = data[1]
        
        cosine_similarity = 1 - spatial.distance.cosine(element1, element2)
        
        results = {}
        
        results['cosine_similarity'] = cosine_similarity
        
        return results
        
    @classmethod
    def save_model(cls, clf, file_name):
        pass

    @classmethod
    def load_model(cls, file_name):
        pass
    
