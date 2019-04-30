import os, sys, fnmatch
import load_conll_2012.coreference_reading as coreference_reading

def load_data(config):
    suffix = config["file_suffix"]
    dir_prefix = config["dir_prefix"]

    print("Load conll documents from:", dir_prefix, " with suffix = ", suffix)
    data = None
    count = 0
    source = ""
    for root, dirnames, filenames in os.walk(dir_prefix):
        #if lang not in root or sets not in root:
            #continue
        for filename in fnmatch.filter(filenames, '*' + suffix):
            file_path = os.path.join(root, filename)
            
            index = filename.find("_")
            if index == -1:
                source2 = filename
            else:
                source2 = filename[:index]
            if source != source2:
                source = source2
                print(" <%s>" % source)
            #print '    ' + filename
            
            data = coreference_reading.read_conll_doc(file_path, data)
            count += 1
    if data is None or len(data) == 0:
        print("Cannot load data in '%s' with suffix '%s'" %(dir_prefix, suffix))
        sys.exit(1)
    print("Total doc.: " + str(count))

    return data


def load_raw_data(raw_data):
    data = coreference_reading.read_conll_raw_data(raw_data)
    return data

if __name__ == '__main__':
    config = {"file_suffix": "gold_conll",
              "dir_prefix": "D:/Study/CSCI 548 IIW/Project/Resources/DataSets/OntoNotes-5.0-NER-BIO-master/conll-formatted-ontonotes-5.0/data/test/data/english/annotations/bn"}
    data = load_data(config)
    for doc in data:
        print('document:', doc)
        for part in data[doc]:
            yolo = False
            for text in data[doc][part]["text"]:
                if "Rumsfeld" in text:
                    yolo = True
                    break
            if not yolo: continue
            
            print('part:', part)
            print('attrs.:', data[doc][part].keys())
            
            print("\narrtr: <text>")
            text = data[doc][part]["text"]
            print(type(text))
            print(len(text))
            print(text[0])
            
            print("\narrtr: <parses>")
            parses = data[doc][part]["parses"]
            print(type(parses))
            print(len(parses))
            print(parses[0])
            
            print("\narrtr: <ner>")
            ner = data[doc][part]["ner"]
            print(type(ner))
            print(len(ner))
            print(ner)
            
            print("\narrtr: <heads>")
            heads = data[doc][part]["heads"]
            print(type(heads))
            print(len(heads))
            for i, j in heads[0].items(): print(i,j)
            print(len(heads[0]))
            print(heads[0][((5,13), u"VP")])
            print(heads[0][((0,14), u"S")])
            exit()
