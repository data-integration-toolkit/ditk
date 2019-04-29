from ner_base import NER

from chemlistem import tradmodel
from chemlistem import get_ensemble_model
import xml.etree.ElementTree as et
import tqdm
import re
from collections import defaultdict

conll_2000_tagset = {'NP', 'ADVP', 'ADJP', 'VP', 'PP', 'SBAR', 'CONJP', 'PRT', 'INTJ', 'LST', 'UCP'}



class DITKChemlistem(NER):
    @staticmethod
    def convert_ground_truth(data, *args, **kwargs):
        annotation_file = data[0]
        text_file = data[1]
        id_dict = defaultdict(list)
        for line in open(text_file).readlines():
            id = line.split("\t")[0]
            id_dict[id + "T"] = []
            id_dict[id + "A"] = []

        for line in open(annotation_file).readlines():
            line = line.split("\t")
            value =(line[2],line[3],line[4],None,None)
            id_dict[line[0] + line[1]].append(value)
        out = []
        for id, value in sorted(id_dict.items()):
            out.append(value)
        return out


    def read_dataset(self, file_dict, dataset_name, *args, **kwargs):
        self.dataset_name = dataset_name
        out_dict = dict()
        for key,value in file_dict.items():
            if isinstance(value,tuple):
                #Correct format
                out_dict[key] = value
            elif isinstance(value,str):
                if "ditk" in dataset_name.lower():
                    out_dict[key] = DITKChemlistem.parse_conll_to_biocreative_format(DITKChemlistem.ditk_to_conll2003(value))
                else:
                    out_dict[key] = DITKChemlistem.parse_conll_to_biocreative_format(value)
            else:
                raise Exception("Unknown format: " + str(value))
        return out_dict

    def train(self, data, *args, **kwargs):
        self.model = tradmodel.TradModel()
        self.model.train(data[1], data[0],None,self.dataset_name,gpu=False)

    def predict(self, data, *args, **kwargs):
        try:
            self.model = tradmodel.TradModel()
            self.model.load("tradmodel_"+self.dataset_name+ ".json", 'tradmodel_' +self.dataset_name+'.h5')
        except Exception as e:
            print("Couldn't load trained model, loading pretrained model")
            self.model = get_ensemble_model(False)

        d = dict()
        with open(data[1],"r") as f:
            for line in f.readlines():
                line = line.split("\t")
                id = line[0]
                abstract = line[1]
                text = line[2]
                d[id + "T"] = abstract
                d[id + "A"] = text

        texts = []
        for id, value in sorted(d.items()):
            texts.append(value)
        predictions = self.model.batchprocess(texts)
        return predictions


    def evaluate(self, predictions, groundTruths, *args, **kwargs):
        true_positives = set()
        false_positives = set()
        false_negatives = set()
        for prediction, truth in zip(predictions, groundTruths):
            prediction_set = set(map(lambda x: (x[0],x[1],x[2]), prediction))
            truth_set = set(map(lambda x: (int(x[0]),int(x[1]),x[2]),truth))
            true_positives.update(prediction_set.intersection(truth_set))
            false_positives.update(prediction_set.difference(truth_set))
            false_negatives.update(truth_set.difference(prediction_set))
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        return (precision, recall, f1)

    @staticmethod
    def write_ditk_output(predictions,data):
        annotatations_fname = data[0]
        text_fname = data[1]
        ground_truths = DITKChemlistem.convert_ground_truth(data)
        text_dict = dict()
        with open(text_fname,"r") as text_f:
            for line in text_f.readlines():
                line_split = line.split("\t")
                text_dict[line_split[0] + "T"] = line_split[1]
                text_dict[line_split[0] + "A"] = line_split[2]

        folder = data[1].split("/")[:-1]
        outfile_name = "/".join(folder) + "/out.txt"
        sorted_items = sorted(text_dict.items())
        with open(outfile_name,"w") as f:
            for i in range(len(sorted_items)):
                prediction = predictions[i]
                ground_truth = ground_truths[i]
                text = sorted_items[i][1]
                offset = 0
                for word in text.split(" "):
                    predicted_tag = "O"
                    ground_truth_tag = "O"
                    for value in prediction:
                        if int(value[0]) == offset and value[2] == word:
                            predicted_tag = "I-MISC"
                    for value in ground_truth:
                        if int(value[0]) == offset:
                            ground_truth_tag = "I-MISC"
                    offset += len(word + " ")
                    f.write(word)
                    f.write(" ")
                    f.write(ground_truth_tag)
                    f.write(" ")
                    f.write(predicted_tag)
                    f.write("\n")
        return outfile_name


    @staticmethod
    def parse_conll_to_biocreative_format(data):
        #Format (id, start, end, entity, type)
        annotations = []
        #Format (id, text)
        texts = []
        id = 0
        if isinstance(data,str):
            with open(data,"r") as f:
                lines = list(f.readlines())
        else:
            lines = data
        cur_line = ""
        offset = 0
        in_title = True
        title = ""
        for i,line in enumerate(lines):
            if 'DOCSTART' not in line and not(i == len(lines) - 1):
                if isinstance(line,str):
                    split_line = line.split(" ")
                else:
                    split_line = line
                text = split_line[0] + " "
                if len(split_line) == 4:
                    if in_title:
                        title += text
                    else:
                        cur_line += text
                    if "B" in split_line[3] or "I" in split_line[3]:
                        annotations.append((id,in_title,offset,offset + len(text),text,split_line[3].split("-")[1].strip("\n")))
                    offset += len(text)
                if split_line[0] == ".":
                    in_title = False
            else:
                in_title = True
                texts.append((id, title, cur_line))
                cur_line = ""
                title = ""
                offset = 0
                id += 1
        folder = data.split("/")[:-1]
        annotations_fname = "/".join(folder) + "/annotations.tsv"
        texts_fname = "/".join(folder) + "/texts.txt"
        with open(annotations_fname,"w") as f:
            for annotation in annotations:
                f.write(str(annotation[0])) #ID
                f.write("\t")
                if annotation[1]:
                    f.write("T\t")
                else:
                    f.write("A\t")
                f.write(str(annotation[2])) #Start
                f.write("\t")
                f.write(str(annotation[3])) #end
                f.write("\t")
                f.write(annotation[4]) #entity
                f.write("\t")
                f.write(annotation[5]) #Type
                f.write("\n")
        with open(texts_fname,"w") as f:
            for text in texts:
                f.write(str(text[0]))
                f.write("\t")
                f.write(text[1])
                f.write("\t")
                f.write(text[2])
                f.write("\n")
        return (annotations_fname,texts_fname)

    @staticmethod
    def is_chunk_tag(chunk):
        if chunk == 'O' or str(chunk).startswith('B-') or str(chunk).startswith('I-'):
            return True
        return False
    @staticmethod
    def ditk_to_conll2003(input_file):
        with open(input_file,"r") as f:
                lines = list(f.readlines())
        converted_lines = []

        chunk_tag_format = None
        for i, line in enumerate(lines):
            if len(line.strip()) > 0:
                data = line.split()
                if chunk_tag_format is None:
                    chunk_tag_format = DITKChemlistem.is_chunk_tag(data[2])

                if not chunk_tag_format:
                    tokens = str(data[2])
                    tokens = re.findall(r"[\w']+", tokens)
                    for token in tokens:
                        if token in conll_2000_tagset:
                            data[2] = 'I-' + token
                        else:
                            data[2] = 'O'

                converted_line = list()
                converted_line.extend(data[0:4])
                converted_lines.append(converted_line)

            else:
                converted_lines.append(line)
        folder = input_file.split("/")[:-1]
        fname = "/".join(folder) + "/tmp.txt"
        with open(fname,"w") as f:
            for line in converted_lines:
                if len(line) != 4:
                    f.write(line)
                else:
                    f.write(line[0].strip("/"))
                    f.write(" ")
                    f.write(line[1])
                    f.write(" ")
                    f.write(line[2])
                    f.write(" ")
                    f.write(line[3])
                    f.write("\n")

        return fname

    def convert_prediction_to_standard_format(self, predictions):
        return list(map(lambda p: (p[0],p[1],p[2], None, None),predictions))

if __name__ == "__main__":
    dataset = {"train": ("cemp/train/chemdner_cemp_gold_standard_train.tsv","cemp/train/chemdner_patents_train_text.txt"),
               "test": ("cemp/test/chemdner_cemp_gold_standard_development_v03.tsv","cemp/test/chemdner_patents_development_text.txt")}
    #dataset = {"train": "conll/eng.train.txt", "test": "conll/eng.testa.txt"}
    #dataset = {"train": "ditk/ner_test_input.txt"}
    model = DITKChemlistem()
    dataset = model.read_dataset(dataset, "cemp")

    model.train(dataset["train"])
    predictions = model.predict(dataset["test"])

    ground_truth = model.convert_ground_truth(dataset["test"])
    assert(len(ground_truth) == len(predictions))
    print(model.evaluate(predictions,ground_truth))
    #model.write_ditk_output(predictions,dataset["test"])
