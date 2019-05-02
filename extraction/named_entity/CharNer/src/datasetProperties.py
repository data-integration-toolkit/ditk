class datasetProperties:
    no_of_sentences_train = 0
    no_of_sentences_test = 0
    no_of_sentences_dev = 0
    max_length_train = 0
    max_length_test = 0
    max_length_dev = 0
    train_data = []
    test_data = []
    dev_data = []



    def __init__(self, data):


        self.getData(data['train'], "train")
        self.getData(data['test'], "test")
        self.getData(data['dev'], "dev")


    def getData(self, data, dataType):

        sentences = []
        max_length = 0
        no_of_sentences = 0


        length_sentence = 0

        for l in data:
            if len(l):
                sentences.append(l)
                length_sentence+=len(l[0])+1
            else:  # emtpy line

                if(max_length<length_sentence):
                    max_length = length_sentence
                sentences.append(l)
                no_of_sentences += 1
                length_sentence = 0



        if (dataType == "train"):
            datasetProperties.train_data = sentences
            datasetProperties.no_of_sentences_train = no_of_sentences
            datasetProperties.max_length_train = max_length

        if (dataType == "test"):
            datasetProperties.test_data = sentences
            datasetProperties.no_of_sentences_test = no_of_sentences
            datasetProperties.max_length_test = max_length

        if (dataType == "dev"):
            datasetProperties.dev_data = sentences
            datasetProperties.no_of_sentences_dev = no_of_sentences
            datasetProperties.max_length_dev = max_length



