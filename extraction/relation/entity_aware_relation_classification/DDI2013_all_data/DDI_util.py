import xml.etree.ElementTree as ET
import os
# from sets import Set


def read(dir_list):
    data = []
    for d in dir_list:
        file_list = os.listdir(d)
        for fname in file_list:
            #		print fname
            parser = ET.XMLParser(encoding="UTF-8")  # etree.XMLParser(recover=True)
            tree = ET.parse(d + '/' + fname, parser=parser)
            root = tree.getroot()
            for sent in root:
                sent_id = sent.attrib['id']
                sent_text = sent.attrib['text'].strip()
                ent_dict = {}
                pair_list = []
                for c in sent:
                    if c.tag == 'entity':
                        d_type = c.attrib['type']
                        d_id = c.attrib['id']
                        d_ch_of = c.attrib['charOffset']
                        d_text = c.attrib['text']
                        ent_dict[d_id] = [d_text, d_type, d_ch_of]
                    elif c.tag == 'pair':
                        p_id = c.attrib['id']
                        e1 = c.attrib['e1']
                        entity1 = ent_dict[e1]
                        e2 = c.attrib['e2']
                        entity2 = ent_dict[e2]
                        ddi = c.attrib['ddi']
                        if ddi == 'true':
                            if 'type' in c.attrib:
                                ddi = c.attrib['type']
                            else:
                                ddi = 'int'
                        pair_list.append([entity1, entity2, ddi])
                data.append([sent_id, sent_text, pair_list])
    return data





def readData(inputfiles):
    input_dir = []
    for filepath in inputfiles:
        input_dir.append(filepath)

    tr_data = read(input_dir)
    fw = open('common_input.txt', 'w')
    for sid, stext, pair in tr_data:
        if len(pair) == 0:
            continue;

        # fw.write(sid + "\t" + stext + "\n")
        for e1, e2, ddi in pair:
            e1_indices=e1[2].split('-')
            e2_indices=e2[2].split('-')
            fw.write(stext+ '\t' + e1[0] + '\t' + e1[1] + '\t'+e1_indices[0] + '\t' + e1_indices[1] + '\t' + e2[0] +'\t' + e2[1] +' \t' + e2_indices[0]+ '\t' + e2_indices[1] +'\t' + ddi)
            # fw.write(e1[0] + '\t' + e1[1] + '\t' + e1[2] + '\t' + e2[0] + '\t' + e2[1] + '\t' + e2[2] + '\t' + ddi)
            fw.write('\n')


def extract_relation(input_data):
    relation = set()
    for line in open(input_data):
        relation.add(line.split('\t')[-1])

    print(relation)

# inputfiles is the path to the folder containing all the DDI xml files
# You can include type of dataset(train or test) as a parameter and write to different files, as per your requirement
# inputfiles=["test/MedLine", "test/DrugBank"]
# readData(inputfiles)
extract_relation('common_input.txt')