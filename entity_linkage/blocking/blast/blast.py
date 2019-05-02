from entity_linkage.blocking.blocking import Blocking
import entity_linkage.blocking.blast.py_sparker as sparker
import os
import tempfile
from pyspark import SparkContext, SparkConf
import csv
import itertools
import pandas as pd

class Blast(Blocking):
	realIdIds1 = None
	realIdIds1 = None

	def __init__(self, blocking_module = "SparkER"):
		sc_conf = SparkConf()
		sc_conf = SparkConf()
		sc_conf.set('spark.executor.memory', '2g')
		sc_conf.set('spark.executor.cores', '4')
		sc_conf.set('spark.cores.max', '4')
		sc_conf.set("spark.driver.memory",'3g')
		self.sc = SparkContext.getOrCreate(conf=sc_conf)

	def read_dataset(self, filepath_list):
		self.profiles1 = sparker.CSVWrapper.loadProfiles(filepath_list[0], header = True, realIDField = "id")
		self.separatorID = self.profiles1.map(lambda profile: profile.profileID).max()
		self.profiles2 = sparker.CSVWrapper.loadProfiles(filepath_list[1], header = True, realIDField = "id", startIDFrom = self.separatorID+1, sourceId=1)
		self.separatorIDs = [self.separatorID]
		self.maxProfileID = self.profiles2.map(lambda profile: profile.profileID).max()
		self.profiles = self.profiles1.union(self.profiles2)
		#print(self.profiles1.collect())
		profiles1_list = []
		count = 0
		for obj in self.profiles1.collect():
			temp_dict = {}
			temp_dict["sourceId"] = obj.sourceId
			temp_dict["profileId"] = obj.profileID
			temp_dict["originalId"] = obj.originalID
			for temp in obj.attributes:
				temp_dict[temp.key] = temp.value
			count += 1
			profiles1_list.append(temp_dict)

		profiles2_list = []
		#count = 0
		for obj in self.profiles2.collect():
			temp_dict = {}
			temp_dict["sourceId"] = obj.sourceId
			temp_dict["profileId"] = obj.profileID
			temp_dict["originalId"] = obj.originalID
			for temp in obj.attributes:
				temp_dict[temp.key] = temp.value

			profiles2_list.append(temp_dict)

		return [pd.DataFrame(profiles1_list), pd.DataFrame(profiles2_list)]

	def train(self, dataframe):
		return None


	def predict(self, model = None, dataframe_list = None):
		self.profileid_to_realId = self.sc.broadcast(self.profiles.map(lambda p:(p.profileID, p.originalID)).collectAsMap())
		self.clusters = sparker.AttributeClustering.clusterSimilarAttributes(self.profiles, 128, 0, computeEntropy=True)
		self.blocks = sparker.TokenBlocking.createBlocksClusters(self.profiles, self.clusters, self.separatorIDs)
		blocksPurged = sparker.BlockPurging.blockPurging(self.blocks, 1.005)
		(profileBlocks, self.profileBlocksFiltered, blocksAfterFiltering) = sparker.BlockFiltering.blockFilteringQuick(blocksPurged, 0.1, self.separatorIDs)
		blockIndexMap = blocksAfterFiltering.map(lambda b : (b.blockID, b.profiles)).collectAsMap()
		self.blockIndex = self.sc.broadcast(blockIndexMap)
		self.profileBlocksSizeIndex = self.sc.broadcast(self.profileBlocksFiltered.map(lambda pb : (pb.profileID, len(pb.blocks))).collectAsMap())


		entropiesMap = self.blocks.map(lambda b : (b.blockID, b.entropy)).collectAsMap()
		self.entropies = self.sc.broadcast(entropiesMap)

		self.use_entropy = True
		results = sparker.WNP.wnp(self.profileBlocksFiltered,
                          		  self.blockIndex,
                                  self.maxProfileID,
                          		  self.separatorIDs,
                          		  None,
                          		  sparker.ThresholdTypes.AVG,#Threshold type
                          		  sparker.WeightTypes.CBS,#Weighting schema
                          		  self.profileBlocksSizeIndex,
                          		  self.use_entropy,
                          		  self.entropies,
                          		  2.0,#Blast c parameter
                          		  sparker.ComparisonTypes.OR#Pruning strategy
                         		  )
		self.candidate_set = results.flatMap(lambda x: x[2])
		return self.rdd_to_csv2(self.candidate_set)

	def evaluate(self, groundtruth, dataframe_list):
		realIdIds1 = self.sc.broadcast(self.profiles1.map(lambda p:(p.originalID, p.profileID)).collectAsMap())
		realIdIds2 = self.sc.broadcast(self.profiles2.map(lambda p:(p.originalID, p.profileID)).collectAsMap())
		def convert(gtEntry):
			if gtEntry.firstEntityID in realIdIds1.value and gtEntry.secondEntityID in realIdIds2.value:
				first = realIdIds1.value[gtEntry.firstEntityID]
				second = realIdIds2.value[gtEntry.secondEntityID]
				if (first < second):
					return (first, second)
				else:
					return (second, first)
			else:
				return (-1, -1)

		orig = self.blocks.map(lambda x: x.getComparisonSize()).sum()
		self.groundtruth_rdd = sparker.CSVWrapper.loadGroundtruth(groundtruth, id1="id1", id2="id2")
		self.mid_gt = self.groundtruth_rdd.map(convert)
		newGT = self.sc.broadcast(set(self.mid_gt.filter(lambda x: x[0] >= 0).collect()))
		
		results = sparker.WNP.wnp(self.profileBlocksFiltered,
                          		  self.blockIndex,
                                  self.maxProfileID,
                          		  self.separatorIDs,
                          		  newGT,
                          		  sparker.ThresholdTypes.AVG,#Threshold type
                          		  sparker.WeightTypes.CBS,#Weighting schema
                          		  self.profileBlocksSizeIndex,
                          		  self.use_entropy,
                          		  self.entropies,
                          		  2.0,#Blast c parameter
                          		  sparker.ComparisonTypes.OR#Pruning strategy
                         		  )
		match_found = float(results.map(lambda x: x[1]).sum())
		num_edges = results.map(lambda x: x[0]).sum()
		#candidate_set = results.flatMap(lambda x: x[2])
		pc = float (match_found) / float(len(newGT.value))
		pq = float(match_found) / float(num_edges)
		block_reduction = float(orig - num_edges)/ float(orig)
		return pq, pc, block_reduction

	def save_model(self, file):
		return None

	def load_model(self, file):
		return None

	def rdd_iterate2(self, rdd, chunk_size=10000):
		indexed_rows = rdd.zipWithIndex().cache()
		count = indexed_rows.count()
		start = 0
		end = start + chunk_size
		while start < count:
			chunk = indexed_rows.filter(lambda r: r[1] >= start and r[1] < end).collect()
			for row in chunk:
				yield row[0]
			start = end
			end = start + chunk_size

	def rdd_to_csv2(self, rdd):
		count = 0
		list_return = []
		for row in self.rdd_iterate2(rdd): # with abstraction, iterates through entire RDD
			count+= 1
			temp_list = [self.profileid_to_realId.value[row[0]], self.profileid_to_realId.value[row[1]]]
			list_return.append(tuple(temp_list))
		return [list_return]

def main():
	blast = Blast()
	dataset1 = "dataset-sample/Dataset1.csv"
	dataset2 = "dataset-sample/Dataset2.csv"
	file_list = [dataset1, dataset2]
	dataframes = blast.read_dataset(file_list)
	print(dataframes[0].head(5))
	blast.train(dataframes)
	predicted_pair_list = blast.predict(dataframe_list = dataframes)
	print(predicted_pair_list[0][10])
	groundtruth = "dataset-sample/articlesGround.csv"
	print(blast.evaluate(groundtruth, dataframes))



if __name__ == '__main__':
    main()
