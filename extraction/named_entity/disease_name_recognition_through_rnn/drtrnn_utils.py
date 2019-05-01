
import os
import sys
import codecs

DITK_FILE_HEADER = 'WORD TRUE_LABEL PRED_LABEL\n\n'
DATA_PATH_BASE = 'binary_data/'


def write_drtrnn_format_to_file(data,outputFilePath):
    """
    Write the data to output file. Format expected to be in format required by drtrnn.

    Args:
        data: list of list. inner list is [token,tag]. token and tag are str
        outputFilePath: str, filepath for location of file to write

    Returns:
        None

    Raises:
        None
    """
    with open(outputFilePath, 'w') as f:
        for line in data:
            if isinstance(line, list):
                line = ' '.join(line)
                line = line
            try:
                f.write(line+'\n')
            except:
                line = line+'\n'
                f.write(line.encode('utf8'))


def ditk_to_drtrnn_format(ditkFilePath,tdt='train'):
    """
    Convert data from ditk format to drtrnn format AND write the resulting
        file to the fixed input data location expected by drtrnn.

    Args:
        ditkFilePath: str, location of input file
        ttd: str, in set {'train','dev','test'} to indicate if file is train, dev, or test

    Returns:
        converted_lines: list of lists. inner list is [token,tag]

    Raises:
        None
    """

    ditk_n_header_lines =2
    with codecs.open(ditkFilePath, mode='r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    converted_lines = []
    header_count = ditk_n_header_lines
    for line in lines:
        if ditk_n_header_lines > 0:  # skip header
            ditk_n_header_lines -= 1
            continue
        if len(line.strip()) > 0:
            data = line.split()
            token = data[0]
            tag = data[3]
            if tag.startswith('B') or tag.startswith('I'):
                tag = str(1)
            converted_line = [token,tag]
            converted_lines.append(converted_line)
        else:
            converted_lines.append(line)

    outputFilePath = DATA_PATH_BASE + tdt + '.txt'
    write_drtrnn_format_to_file(converted_lines,outputFilePath)

    return converted_lines


def extract_file_locations(file_dict):
    """
    Helper to ectract filenames from file_dict.

    Args:
        file_dict: dictionary
            {
                "train": dict, {key="file description":value="file location"},
                "dev" : dict, {key="file description":value="file location"},
                "test" : dict, {key="file description":value="file location"},
            }
        PRECONDITION: 'file description' expected to be str in set {'data',...}

    Returns:
        train_file_location,dev_file_location,test_file_location. all are str

    Raises:
        None
    """
    train_file_location = file_dict['train']['data']  # PRECONDITION THAT THIS EXISTS
    if file_dict['dev']:  # dev set provided
        dev_file_location = file_dict['dev']['data']
    else:
        dev_file_location = train_file_location  # no dev data. use train as placeholder
    if file_dict['test']:  # dev set provided
        test_file_location = file_dict['test']['data']
    else:
        test_file_location = train_file_location  # no dev data. use train as placeholder

    return train_file_location,dev_file_location,test_file_location


def convert_ditk_to_train_format(file_dict):
    """
    Helper function to convert from ditk ner format [example found in test/sample_input.txt] to format
    required by drtrnn methods.

    Args:
        file_dict: dictionary
            {
                "train": dict, {key="file description":value="file location"},
                "dev" : dict, {key="file description":value="file location"},
                "test" : dict, {key="file description":value="file location"},
            }
        PRECONDITION: 'file description' expected to be str in set {'data',...}

    Returns:
        data_dict: dictionary
            {
                'train': list of lists.
                'dev': list of lists.
                'test': list of lists.
            }
        NOTE: list of list. inner list is [token,tag]

    Raises:
        None
    """
    train_file_location,dev_file_location,test_file_location = extract_file_locations(file_dict)

    data_train = ditk_to_drtrnn_format(train_file_location,'train')
    data_dev = ditk_to_drtrnn_format(dev_file_location,'dev')
    data_test = ditk_to_drtrnn_format(test_file_location,'test')

    data_dict = {'train':data_train,'dev':data_dev,'test':data_test}

    return data_dict

def convert_token_tag_to_ditk_format(filebase,token_tag):
    """
    Helper function to act as an intermediary. Converts ['token', 'tag'] to 
    ['token' '_' '_' 'tag' '_'*11]. Basically expands data to 15 columns [not pretty
    but useful in creating a go-to spot for ditk imtermediary]

    Writes the intermediate files to 'binary_data/train_ditk_format.txt',
        'binary_data/dev_ditk_format.txt', 'binary_data/test_ditk_format.txt'

    Args:
        filebase: str in {'train','dev','test'}. Base filename to use for ditk format files
        token_tag: list of list. inner list is [token_tag]

    Returns:
        ditk_filename: str, name of the file that was written in ditk format
    """

    ditk_filename_suffix = '_ditk_format.txt'

    ditk_filename = DATA_PATH_BASE + filebase + ditk_filename_suffix

    prevLineSpace = False
    with open(ditk_filename,'w') as outFile:
        outFile.write(DITK_FILE_HEADER)  # add header expected by test format
        for t in token_tag:
            if len(t) > 1:
                outLine = '%s _ _ %s '%(t[0],t[1])+' '.join(['_' for _ in range(11)])+'\n'
                try:
                    outFile.write(outLine)
                except:
                    outFile.write(outLine.encode('utf8'))
                prevLineSpace = False
            else:
                if prevLineSpace:
                    continue  # don't write double blank lines
                outFile.write('\n')
                prevLineSpace = True

    return ditk_filename



def convert_conll_to_token_tag(file):
    """
    Helper function to convert conll2003 to token_tag format.

    Args:
        file: str, file location of the file to convert to token_tag format 

    Returns:
        converted_lines: list of lists. inner list is [token,tag]
    """
    with codecs.open(file, mode='r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    converted_lines = []
    for line in lines:
        if len(line.strip()) > 0:
            if '-DOCSTART-' in line:
                continue
            data = line.split()
            token = data[0]
            tag = data[3]
            if tag.startswith('B') or tag.startswith('I'):
                tag = str(1)
            converted_line = [token,tag]
            converted_lines.append(converted_line)
        else:
            converted_lines.append(line)

    return converted_lines


def convert_ontonotes_to_token_tag(file):
    """
    Helper function to convert OntoNotes5.0 to token_tag format.

    Args:
        file: str, file location of the file to convert to token_tag format 

    Returns:
        converted_lines: list of lists. inner list is [token,tag]
    """
    with codecs.open(file, mode='r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    converted_lines = []
    for line in lines:
        if len(line.strip()) > 0:
            if '-DOCSTART-' in line:
                continue
            data = line.split()
            token = data[0]
            tag = data[3]
            if tag.startswith('B') or tag.startswith('I'):
                tag = str(1)
            converted_line = [token,tag]
            converted_lines.append(converted_line)
        else:
            converted_lines.append(line)

    return converted_lines


def master_token_tag_to_drtrnn(ditk_train,ditk_dev,ditk_test):
    """
    Helper function to complete the transformation from token_tag format to drtrnn format!
    Once specific dataset is put into token_tag format, remaining steps are the same. Hence,
    the purpose of this function is mainly to refactor repretetive code.

    Args:
       ditk_train,ditk_dev,ditk_test: each is list of lists. inner list is [token,tag]

    Returns:
        data_dict: dictionary
            {
                'train': list of lists.
                'dev': list of lists.
                'test': list of lists.
            }
        NOTE: list of list. inner list is [token,tag]
    """
    filebase_ditk_train = 'train'
    filebase_ditk_dev = 'dev'
    filebase_ditk_test = 'test'

    filename_ditk_train = convert_token_tag_to_ditk_format(filebase_ditk_train,ditk_train)
    filename_ditk_dev = convert_token_tag_to_ditk_format(filebase_ditk_dev,ditk_dev)
    filename_ditk_test = convert_token_tag_to_ditk_format(filebase_ditk_test,ditk_test)

    data_train = ditk_to_drtrnn_format(filename_ditk_train,'train')
    data_dev = ditk_to_drtrnn_format(filename_ditk_dev,'dev')
    data_test = ditk_to_drtrnn_format(filename_ditk_test,'test')

    data_dict = {'train':data_train,'dev':data_dev,'test':data_test}

    return data_dict



def convert_dataset_conll_to_train_format(file_dict):
    """
    Helper function to convert from conll2003 format to format required by drtrnn methods.

    Args:
        file_dict: dictionary
            {
                "train": dict, {key="file description":value="file location"},
                "dev" : dict, {key="file description":value="file location"},
                "test" : dict, {key="file description":value="file location"},
            }
        PRECONDITION: 'file description' expected to be str in set {'data',...}

    Returns:
        data_dict: dictionary
            {
                'train': list of lists.
                'dev': list of lists.
                'test': list of lists.
            }
        NOTE: list of list. inner list is [token,tag]

    Raises:
        None
    """

    train_file_location,dev_file_location,test_file_location = extract_file_locations(file_dict)

    ditk_train = convert_conll_to_token_tag(train_file_location)
    ditk_dev = convert_conll_to_token_tag(dev_file_location)
    ditk_test = convert_conll_to_token_tag(test_file_location)

    data_dict = master_token_tag_to_drtrnn(ditk_train,ditk_dev,ditk_test)

    return data_dict


def convert_dataset_ontoNotes_to_train_format(file_dict):
    """
    Helper function to convert from OntoNotes5.0 format to format required by drtrnn methods.

    Args:
        file_dict: dictionary
            {
                "train": dict, {key="file description":value="file location"},
                "dev" : dict, {key="file description":value="file location"},
                "test" : dict, {key="file description":value="file location"},
            }
        PRECONDITION: 'file description' expected to be str in set {'data',...}

    Returns:
        data_dict: dictionary
            {
                'train': list of lists.
                'dev': list of lists.
                'test': list of lists.
            }
        NOTE: list of list. inner list is [token,tag]

    Raises:
        None
    """

    train_file_location,dev_file_location,test_file_location = extract_file_locations(file_dict)

    ditk_train = convert_ontonotes_to_token_tag(train_file_location)
    ditk_dev = convert_ontonotes_to_token_tag(dev_file_location)
    ditk_test = convert_ontonotes_to_token_tag(test_file_location)

    data_dict = master_token_tag_to_drtrnn(ditk_train,ditk_dev,ditk_test)

    return data_dict


def copy_predictions_to_predictions_with_header(raw_predictions_filename='predictions.txt'):
    """
    Helper function to convert raw predictions to final predictions. The only difference
    between the two is that final predictions have a header.

    Args:
        raw_predictions_filename: str, the filename of the raw predictions

    Returns:
        finalPredictionsFileName: str, filename of the final predictions file
    """
    finalPredictionsFileName = 'predictions_withHeader.txt'

    with open(finalPredictionsFileName,'w') as outFile:
        outFile.write(DITK_FILE_HEADER)  # add header expected by test format
        with open(raw_predictions_filename,'r') as pf:
            for line in pf:
                outFile.write(line)

    return finalPredictionsFileName


def load_groundTruth_from_predictions(raw_predictions_filename='predictions.txt'):
    """
    Helper function to load groundTruths from predictions file. This function
    should not be called unless predict() has already completed.

    Args:
        raw_predictions_filename: str, the filename of the raw predictions

    Returns:
        groundTruths: list of tuples in same format as output of predict()
    """
    groundTruths = []
    with open(raw_predictions_filename,'r') as inFile:
        for line in inFile:
            info = line.split()
            if len(info) < 3:
                continue
            token = info[0].strip()
            tag = info[1].strip()
            truth = [None,None,token,tag]
            groundTruths.append(truth)

    return groundTruths


def bio(test_only=False):
    """
    Intermediate function between raw data to drtrnn format and training. This
    generates the *_tags and *_words and vocab files.

    PRECONDITION: Expect data_path_base+'train.txt' to be present and in proper drtrnn format!

    Args:
        None [uses fixed path locations and naming conventions, hence the requirement on precondition!]
        test_only: boolean. If true, only convert test data

    Returns:
        None

    Raises:
        None
    """

    VOCAB_FILE = 'word2vec/vocab.txt'
    unique_words = set()

    bases = ['train','dev','test']

    if test_only:
        bases = ['test']

    for base in bases:

        fl = DATA_PATH_BASE+base+'.txt'  #"test.txt"
        if not (base+'.txt' in os.listdir(DATA_PATH_BASE)):
            print(os.listdir(DATA_PATH_BASE))
            if base == 'train':  # at a minimum, we MUST have train
                print('Error on train(). Training file does not exist. Exiting now.')
                sys.exit()
            if (base == 'test') and test_only:  # at a minimum, we MUST have train
                print('Error on predict(). Testing file does not exist. Exiting now.')
                sys.exit()
            continue  # go to next one
        f = open(fl, "r")

        sent_list = f.read().strip().split("\n\n")
        sents = []
        tag_sents = []
        for sent in sent_list :
            word_list = sent.split('\n')
            words = []
            tags = []
            for line in word_list :
                splitted = line.strip().split()
                if not (len(splitted)==2):
                    continue
                w,tag = line.strip().split()
                t = 0  # case tag is 'O'
                if tag in ['1']:  # tag is B-Dis or I-Dis...
                    t = 1
                words.append(w)
                unique_words.add(w.lower())
                tags.append(int(t))
            sents.append(words)
            tag_sents.append(tags)

        # print len(sents)
        # print len(tag_sents)

        tag_list = []
        for i in xrange(len(sents)):
            tag = []
            cond = 1
            for j in xrange(len(sents[i])):
                if(tag_sents[i][j] == 1 and cond == 1):
                    tag.append("B-Dis")
                    cond = 0
                elif(tag_sents[i][j] == 1 and cond == 0):
                    tag.append("I-Dis")
                elif(tag_sents[i][j] == 0 and cond == 0):
                    tag.append("O")
                    cond = 1
                else:
                    tag.append("O")
            tag_list.append(tag)

        g = open(DATA_PATH_BASE+base+"_words.txt", "w")
        h = open(DATA_PATH_BASE+base+"_tags.txt", "w")
        for i in xrange(len(sents)):
            if not (len(sents[i])==len(tag_list[i])):
                print 'do a concern'  # TEMPORARY
            for j in xrange(len(sents[i])):
                g.write(sents[i][j])
                g.write(" ")
                h.write(tag_list[i][j])
                h.write(" ")
            g.write("\n")
            h.write("\n")

    # write vocab file that has words from all sets...train, test, dev...[technically this is questionable...]
    if not (test_only):  # if this is test_only, we do NOT write
        with open(VOCAB_FILE,'w') as vvv:
            for word in unique_words:
                outLine = word+'\n'
                vvv.write(outLine)
