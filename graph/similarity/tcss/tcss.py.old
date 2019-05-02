'''
Created on 2010-07-27

@author: Shobhit Jain

@contact: shobhit@cs.toronto.edu
'''

import sys
import os
import getopt
from main import load_semantic_similarity, calculate_semantic_similarity


def _usage():
    '''
    Details on how to use TCSS.
    '''
    print "\n\n tcss.py [-options] geneA geneB\
    \n or \n tcss.py [-options] -i input_file"
    print "\n -options \
    \n    -i [file name] or --input [=file name]       Input file (two genes separted by comma per line)\
    \n    -o [file name] or --output [=file name]      Output file\
    \n    -c [domain:cutoff] or                        Domain [C/P/F], cutoff [int/float] in any combination\
    \n         --topology-cutoff [=domain:cutoff]      (default: C:2.4,P:3.5,F:3.3)\
    \n    --detail                                     Detailed output (default: False)\
    \n    --gene [=file name]                          Gene annotation file (default: SGD file provided)\
    \n    --go [=file name]                            Gene Ontology (GO) obo file (default: GO file provided)\
    \n    --drop [=evidence code]                      GO evidence code not to be used \
    \n    -h or --help                                 Usage\n\n\n"



def _command_line_arguments(argv):
    '''
    Process the command line arguments. Returns options variable.
    '''
    options = {'args':None, 'input':None, 'output':None, 'detail':None, 'ontology':"C:2.4,P:3.5,F:3.3", \
               'gene':'gene_association.sgd', 'go':'gene_ontology.obo.txt', 'drop':''}
    try:
        opts, args = getopt.getopt(argv, "hi:o:c:", ["help", "input=", "output=", "topology-cutoff=", \
                                                     "detail", "ontology=", "gene=", "go=", "drop="])
    except getopt.GetoptError:
        _usage()
        sys.exit(2)
    if len(opts) == len(args) == 0:
        _usage()
        sys.exit()
    if 1 < len(args) < 3:
        options['args'] = args
    if len(args) > 2:
        _usage()
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            _usage()
            sys.exit()
        elif opt in ("-i", "--input"):
            if os.path.isfile(arg):
                if len(args) > 0: 
                    print "Excluding arguments:", args
                    options['args'] = None
                options['input'] = arg
            else:
                print 'Input file not found'
                sys.exit()
        elif opt in ("-o", "--output"):
            options['output'] = arg
        elif opt in ("-c", "--ontology"):
            options['ontology'] = arg
        elif opt in ("--detail"):
            options['detail'] = True
	elif opt in ("--drop"):
	    options['drop'] = arg
        elif opt in ("--gene"):
            if os.path.isfile(arg):
                options['gene'] = arg
            else:
                print 'Gene annotation file not found'
                sys.exit()
        elif opt in ("--go"):
            if os.path.isfile(arg):
                options['go'] = arg
            else:
                print 'GO file not found'
                sys.exit()
        else:
            _usage()
            sys.exit()     
    if options['args'] == None and options['input'] == None:
        _usage()
        sys.exit()
    return options



def input_file_run(objs, infile, outfile, detail):
    '''
    Run TCSS when input file is provided.
    '''
    file = open(infile, 'r')
    result = ""
    for line in file:
        line = line.strip().split(",")
        if len(line) > 2:
            print 'File format not proper: refer README'
            sys.exit()
        result += calculate_semantic_similarity(objs, line[0], line[1], detail)
    file.close()
    output_results(result, outfile)     
        
        
        
def output_results(result, outfile):
    '''
    Output the results on screen or on file. 
    '''
    if outfile:
        try:
            file = open(outfile, 'w')
            file.write(result)
            file.close()
        except:
            print "Cannot open output file: check path or permissions \n Printing results on screen"
            print result
    else:
        print result
        

    
if __name__ == '__main__':
    
    options = _command_line_arguments(sys.argv[1:])
    objs = load_semantic_similarity(options['go'], options['gene'], options['ontology'], options['drop'])
    
    if options['input']:
        input_file_run(objs, options['input'], options['output'], options['detail'])
    else:
        result = calculate_semantic_similarity(objs, options['args'][0], options['args'][1], options['detail'])
        output_results(result, options['output'])
    
