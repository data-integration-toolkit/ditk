from extraction.named_entity.Multi_Task_NER import multi_task_ner

def main(fileNames):

    multi_task =  multi_task_ner.multi_task_ner()
    data = multi_task.read_dataset(fileNames,"dataset")
    embedding = multi_task.generating_encoding(data)
    multi_task.train([data, embedding])

    train_file = fileNames['train']
    predix = ''
    if train_file.rfind('/') >= 0:
        predix = train_file[:train_file.rfind('/')] + '/'

    multi_task.save_model(predix)

    decoded_predictions = multi_task.predict([data, embedding])
    ground_truth = multi_task.convert_ground_truth(data)[0]
    NN_pre = decoded_predictions[0]
    #ground_truth = ground_truth
    test_words = data[2][0]
    CRF_pre = decoded_predictions[1]

    file_path = 'output.txt'
    file = open(file_path, 'w')
    for i in range(len(test_words)):
        file.write(str(test_words[i])+" "+str(ground_truth[i])+" "+ str(CRF_pre[i])+'\n')

    file.close()

    target = multi_task.convert_ground_truth(data)
    evaluation_result = multi_task.evaluate(decoded_predictions, target)

    for i in range(len(evaluation_result)):
        if i == 0:
            print('NN precision: ', str(evaluation_result[i][0]))
            print('NN recall: ', str(evaluation_result[i][1]))
            print('NN f1: ', str(evaluation_result[i][2]))
        else:
            print('CRF precision: ', str(evaluation_result[i][0]))
            print('CRF recall: ', str(evaluation_result[i][1]))
            print('CRF f1: ', str(evaluation_result[i][2]))
    return file_path

if __name__ == '__main__':
    #multi_task  = Multi_Task_NER()
    #multi_task.train_file = "chemdner_new/train.txt"
    #multi_task.valid_file = "chemdner_new/valid.txt"
    #multi_task.test_file = "chemdner_new/test.txt"
    #multi_task.fileNames = {}
    #multi_task.fileNames['train'] = multi_task.train_file
    #multi_task.fileNames['valid'] = multi_task.valid_file
    #multi_task.fileNames['test'] = multi_task.test_file
    fileNames = {}
    fileNames['train'] = "tests/ner_test_input/train.txt"
    fileNames['valid'] = "tests/ner_test_input/valid.txt"
    fileNames['test'] = "tests/ner_test_input/test.txt"
    outputfile = main(fileNames)
