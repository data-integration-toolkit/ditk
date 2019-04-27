import abc
from prost_base import ProstBase
from datetime import datetime
from dateutil import parser
import googleapiclient.discovery
from google.cloud import storage
import time
import csv

class Prost(ProstBase):
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
                # raise Exception(result['status']['de
                self._writeToLog('Error during job {}'.format(job_id),logfile)
                return result
            elif result['status']['state'] == 'DONE':
                # Calculate time of job
                for status in result['statusHistory']:
                    if status['state'] == 'RUNNING':
                        start = parser.parse(status['stateStartTime'])
                        end = parser.parse(result['status']['stateStartTime'])
                        self._writeToLog('Job {} finished in {}.'.format(job_id, self._string_timedelta(end-start)),logfile)
                return result
    # [END _wait_for_job]


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

    def load_rdf(self, input_rdf, output_dir, path_to_jar):
        className = 'run.Main'
        input_location = 'gs://{}/{}'.format(self.storage_bucket, input_rdf)

        job_id = self.__submit_spark_job__(path_to_jar, className, ['-i',input_location, '-o',output_dir])

        self._writeToLog("Started Prost Loading job, Job ID: {}".format(job_id),'prost_load.log')

        # FOR SMALL DATASET ONLY
        self._wait_for_job(job_id, 'prost_load.log')

        # Get VP table stats
        client = storage.Client(project=self.project_id)
        bucket = client.get_bucket(self.storage_bucket)
        dir = input_rdf.split('/')[0] + '/{}_vp_stats.csv'.format(output_dir)
        blobs = bucket.list_blobs(prefix=dir)
        for b in blobs:
            if b.name != dir and b.name != dir + '_SUCCESS':
                s = str(bucket.blob(b.name).download_as_string(), 'utf-8')
                rows = s.split('\n')
                with open('prost_vp_stats.csv', 'w') as out:
                    w = csv.writer(out,quotechar='"')
                    for r in rows:
                        if len(r) > 2:
                            w.writerow(r.split(','))



    def query_executor(self, sparql_query_file_or_dir, db_name, output_dir, path_to_jar):
        from os import listdir
        from os.path import isfile, join, isdir, exists
        
        mainClass = 'run.Main'
        
        query = ''
        with open(sparql_query_file_or_dir, 'r') as f:
            query = f.read()
       
        # db_arg = 'gs://{}/{}'.format(self.storage_bucket, db_dir)
        out_arg = 'gs://{}/{}'.format(self.storage_bucket, output_dir)

        job_id = self.__submit_spark_job__(path_to_jar, mainClass, ['-q', query, '-o', out_arg, '-d', db_name, '-wpt'])
        self._writeToLog("Started PROST Query Executor job for query {}, JOB ID: {}".format(sparql_query_file_or_dir, job_id),'prost.log')
        self._wait_for_job(job_id, 'prost.log')
        self._download_results(output_dir)


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
    
    def _download_results(self, output_dir):
        client = storage.Client(project=self.project_id)
        bucket = client.get_bucket(self.storage_bucket)

        print(output_dir)
        dir = output_dir + '.csv/'
        out_file_name = output_dir.split('/')[1] + '.csv'
        print(out_file_name)
        blobs = bucket.list_blobs(prefix=dir)
        for b in blobs:
            if b.name != dir and b.name != dir + '_SUCCESS':
                s = str(bucket.blob(b.name).download_as_string(), 'utf-8')
                rows = s.split('\n')
                with open(out_file_name, 'w') as out:
                    w = csv.writer(out,quotechar='"')
                    for r in rows:
                        if len(r) > 2:
                            w.writerow(r.split(','))

