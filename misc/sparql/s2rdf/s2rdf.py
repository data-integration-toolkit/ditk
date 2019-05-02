import abc, time
from s2rdf_base import S2RdfBase
from datetime import datetime
from dateutil import parser
import googleapiclient.discovery
from s2rdf_util import util
from google.cloud import storage
import csv

class S2RDF(S2RdfBase):

    '''
        Since the Spark job will be run on Google cloud there are three things that are needed:
             1) Id of the bucket where all of the relavant files are and will be stored
             2) Region where the cluster is located
             3) Name of the Hadoop cluster   
             4) Project ID
        NOTE: You must have an envirionment variable named GOOGLE_APPLICATION_CREDENTIALS that points to your credentials file (JSON).
              Otherwise you will not be able to connect to the cluster
    '''
    def __init__(self, storage_bucket, region, cluster_name, project_id):
        self.storage_bucket = storage_bucket
        self.region = region
        self.cluster_name = cluster_name
        self.project_id = project_id

        self.dataproc = googleapiclient.discovery.build('dataproc', 'v1')

    # write some line to the log file
    def _writeToLog(self,line,filename):
        print(line)
        with open("./" + filename, "a") as logFile:
            logFile.write(line+"\n")

    # [START __submit_spark_job__]
    def __submit_spark_job__(self, jar_file, main_class, jar_args):
        """Submits the Spark job to the cluster, assuming `jar_file` has
            already been uploaded to `self.storage_bucket`.
            Returns job id.
        """

        # TODO use google-api-python-client
        job_details = {
            "projectId": "csci-548-project-sp2019-237020",
            "job": {
                "placement": {
                    "clusterName": "cluster-4a25"
                },
                    "sparkJob": {
                        "mainClass": main_class,
                        "jarFileUris": ['gs://{}/{}'.format(self.storage_bucket,jar_file)],
                        "args": jar_args
                    }
            }
        }
        result = self.dataproc.projects().regions().jobs().submit(
        projectId=self.project_id,
        region=self.region,
        body=job_details).execute()
        job_id = result['reference']['jobId']
        return job_id
    # [END __submit_spark_job__]

    # [START _string_timedelta]
    def _string_timedelta(self, t_delta):
        days = t_delta.days

        remaining_seconds = t_delta.seconds

        hours = int(t_delta.seconds / 3600)
        remaining_seconds = remaining_seconds - hours*3600

        minutes = int(remaining_seconds / 60)
        remaining_seconds = remaining_seconds - minutes*60

        return '{} days {} hours {} min {} sec'.format(days, hours, minutes, remaining_seconds)
    # [END _string_timedelta]


    # [START _wait_for_job]
    def _wait_for_job(self, job_id, logfile):
        self._writeToLog('Waiting for job {} to finish...'.format(job_id),logfile)
        while True:
            time.sleep(10)
            result = self.dataproc.projects().regions().jobs().get(
                projectId=self.project_id,
                region=self.region,
                jobId=job_id).execute()
        # Handle exceptions
            if result['status']['state'] == 'ERROR':
                raise Exception(result['status']['details'])
            elif result['status']['state'] == 'DONE':
                # Calculate time of job
                for status in result['statusHistory']:
                    if status['state'] == 'RUNNING':
                        start = parser.parse(status['stateStartTime'])
                        end = parser.parse(result['status']['stateStartTime'])
                        self._writeToLog('Job {} finished in {}.'.format(job_id, self._string_timedelta(end-start)),logfile)
                return result
    # [END _wait_for_job]

    # [START load_rdf]
    def load_rdf(self, input_rdf_path, path_to_jar, wait_for_jobs=False, scale=0.25):

        dt = datetime.now()
        logfilename = "dataset_creator_" + str(dt.month) + str(dt.day) + str(dt.year) + "_" + str(dt.hour) + str(dt.minute) + str(dt.second) + ".log"

         # Put tables in same folder as rdf file
        begin = input_rdf_path.rfind("/") + 1
        db_dir, input_rdf = input_rdf_path[:begin], input_rdf_path[begin:]
        db_dir = "gs://{}/{}".format(self.storage_bucket, db_dir)

        className = "runDriver"
         # Donwload stat files (SMALL DATASETS ONLY!)
        client = storage.Client(project=self.project_id)
        bucket = client.get_bucket(self.storage_bucket)
                        
        # Make the Vertical Partioning (VP) tables first
        self._writeToLog('Starting Spark Jobs for project {} on cluster {} in bucket {}'.format(self.project_id, self.cluster_name,self.storage_bucket), logfilename)
        job_id = self.__submit_spark_job__(path_to_jar, className, [db_dir, input_rdf, 'VP', str(scale)])
        self._writeToLog("Started Vertical Partitioning job, Job ID: {}".format(job_id), logfilename)
        # all VP tables ust exist before creating ExtVP tables
        self._wait_for_job(job_id, logfilename)

        for relation_type in ['SO', 'OS', 'SS']:
            j_id = self.__submit_spark_job__(path_to_jar, className, [db_dir, input_rdf, relation_type, str(scale)])
            # log job ID 
            self._writeToLog("Started Extended Vertical Partitioning subset {}, Job ID: {}".format(relation_type, j_id), logfilename)
            self._wait_for_job(j_id, logfilename)
        self._writeToLog('Tables are (will be) located at {}VP and {}ExtVP'.format(db_dir, db_dir), logfilename)
        
        # Download Stats files
        dir = input_rdf_path.split('/')[0] + '/statistics'
        file_index = 1
        for b in bucket.list_blobs(prefix=dir):
            if '.csv' in b.name and 'summary' not in b.name:
                s = str(bucket.blob(b.name).download_as_string(), 'utf-8')
                rows = s.split('\n')
                mode = 'a+'
                f_name = 's2rdf_vp_stats.csv'
                if '><' in rows[0]:
                    f_name = 's2rdf_extVp_stats_{}.csv'.format(file_index) 
                    mode = 'w'
                file_index = file_index + 1
                with open(f_name, mode) as out:
                    w = csv.writer(out,quotechar='"')
                    file_index = file_index + 1
                    for r in rows:
                        if len(r) > 2:
                            w.writerow(r.split(','))
       


    # [END load_rdf]
    def sparql_translator(self, input_sparql, output_dir, path_to_jar):
        # get all stat files from
        util.translate_sparql_queries(util, input_sparql, output_dir, path_to_jar, '')
