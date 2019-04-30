
# Copyright Simon Skilevic
# Master Thesis for Chair of Databases and Information Systems
# Uni Freiburg
#
import sys, getopt, subprocess

class util:
    # get the list of files in mypath
    @staticmethod
    def _loadListOfQueries(self, mypath):
            from os import listdir
            from os.path import isfile, join
            onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
            return onlyfiles

    def _readFileToString(self, fileName):    
        with open(fileName) as myFile: return myFile.read()
        
    def _addStringToFile(self, fileName, str, queryName):
        with open(fileName, "a") as myFile: myFile.write(">>>>>>"+queryName+"\n"+str)
        
               
    def translate_sparql_queries(self,sparql_dir, output_sql_dir, jar_location, stats_dir):
        
        print ('Input Dir is "' + sparql_dir)
        print ('Output Dir is "' + output_sql_dir) 
        from os import listdir, mkdir, getcwd
        from os.path import isfile, join, isdir, exists
        sparql_list = [ f for f in listdir(sparql_dir) if isfile(join(sparql_dir,f)) ]
        # subprocess.call("rm -f "+sqlDir+"/*.*", shell=True)
        
        output_sql_dir = sparql_dir + "sql/"
        if not exists(output_sql_dir):
            mkdir(output_sql_dir)

        for fileName in sparql_list:
            if ((not "~" in fileName) and (not "rdf3x" in fileName)):
                print ("Parse " + sparql_dir + fileName)
                outPutFileName = (output_sql_dir+"/" + fileName[:4]).replace("//", "/")

                stat_files = []
                for f in listdir(stats_dir):
                    if isfile(join(stats_dir,f)):
                        stat_files.append(f)
                rel_args = []
                if "stat_os.csv" in stat_files:
                    rel_args.append("-os")
                if "stat_so.csv" in stat_files:
                    rel_args.append("-so")
                if "stat_ss.csv" in stat_files:
                    rel_args.append("-ss")
                if len(rel_args) > 0:
                    rel_args.insert(0, " ")
                # execute translator
                command = ("java -jar " + jar_location
                + " -i " + sparql_dir + fileName + " -o " + outPutFileName
                + " ".join(rel_args)
                +" -sd " + stats_dir
                # +" -sUB " + "1"
                )
                print("\n" + command)
                status =  subprocess.call(command, shell=True)

                # write query to files
                with open(sparql_dir + fileName) as myFile:
                    query = myFile.read()
                    with open(output_sql_dir+"compositeQueryFile.txt", "a") as comp_file:
                        comp_file.write(">>>>>>" + fileName[:-4] + "\n" + query)

                # add to composite query file (all queries of the input directory)
                command = command.replace("//", "/")

