import re
conll_2000_tagset = {'NP', 'ADVP', 'ADJP', 'VP', 'PP', 'SBAR', 'CONJP', 'PRT', 'INTJ', 'LST', 'UCP'}


def is_tag_bio(tag):
    return str(tag).startswith('B-') or str(tag).startswith('I-') or tag == 'O'

def ditk_ner_to_conll2012(lines):

    converted_lines = []
    for i, line in enumerate(lines):
        try:
            if len(line.strip()) > 0:
                data = line.split()
                converted_line = list()
                converted_line.extend(data[4:7])
                converted_line.extend(data[0:3])
                converted_line.extend(data[7:11])
                converted_line.append(data[3])
                converted_line.extend(data[11:])
                converted_lines.append(converted_line)
        except Exception as e:
            print('Skiping invalid line.')


    end_doc = "#end document"
    start_doc_template = "#begin document (%s); part %s"

    conll2012_lines =[]

    prev_part = None
    prev_word_num = None
    bio_tag = None

    for i,line in enumerate(converted_lines):

        set_name = line[0]
        curr_part = line[1]
        curr_word_num = int(line[2])

        ner_tag = line[10]
        if bio_tag is None:
            bio_tag = is_tag_bio(ner_tag)

        if bio_tag:
            if ner_tag == 'O':
                line[10] = '*'
            elif str(ner_tag).startswith('I-'):
                if i+1 < len(converted_lines) and str(converted_lines[i+1][10]).startswith("I-"):
                    line[10] = '*'
                else:
                    line[10] = '*)'
            elif str(ner_tag).startswith('B-'):
                if i+1 < len(converted_lines) and str(converted_lines[i+1][10]).startswith("I-"):
                    line[10] = '('+ ner_tag[2:] + '*'
                else:
                    line[10] = '(' + ner_tag[2:] + ')'

        if len(conll2012_lines) == 0:
            conll2012_lines.append(start_doc_template % (set_name, curr_part))

        if prev_part is None:
            prev_part = curr_part

        if prev_word_num is not None  and (prev_word_num != curr_word_num - 1 or prev_word_num == curr_word_num):
            conll2012_lines.append('\n')

        if curr_part != prev_part:
            conll2012_lines.append(end_doc)
            conll2012_lines.append(start_doc_template % (set_name, curr_part))
            prev_part = curr_part

        prev_word_num = curr_word_num
        conll2012_lines.append(line)

    conll2012_lines.append('\n')
    conll2012_lines.append(end_doc)

    return conll2012_lines

def is_chunk_tag(chunk):
    if chunk =='O' or str(chunk).startswith('B-') or str(chunk).startswith('I-'):
        return True
    return False

def ditk_ner_to_conll2003(lines):
    converted_lines = []

    chunk_tag_format = None
    for i,line in enumerate(lines):
        if len(line.strip()) > 0:
            data = line.split()
            if chunk_tag_format is None:
                chunk_tag_format = is_chunk_tag(data[2])

            if not chunk_tag_format:
                tokens = str(data[2])
                tokens = re.findall(r"[\w']+", tokens)
                for token in tokens:
                    if token in conll_2000_tagset:
                        data[2] = 'I-'+token
                    else:
                        data[2] = 'O'

            converted_line = list()
            converted_line.extend(data[0:4])
            converted_lines.append(converted_line)

        else:
            converted_lines.append(line)


    return converted_lines

def conll2012_to_ditk(lines):
    converted_lines =[]

    flag = None
    for line in lines:
        l = line.strip()
        if len(l) > 0 and str(l).startswith('#begin') or str(l).startswith('#begin'):
            continue

        l = ' '.join(l.split())
        ls = l.split(" ")
        converted_line = []

        if len(ls) >= 11:
            extra = ls[0:3]  # 4-7
            extra.extend(ls[6:10])  # 7 - 11
            extra.extend(ls[11:])  # 11:
            word = ls[3]  # 0
            pos = ls[4]  # 1
            cons = ls[5]  # 2
            ori_ner = ls[10]  # 3
            ner = ori_ner
            # print(word, pos, cons, ner)

            if ori_ner == "*":
                if flag == None:
                    ner = "O"
                else:
                    ner = "I-" + flag
            elif ori_ner == "*)":
                ner = "I-" + flag
                flag = None
            elif ori_ner.startswith("(") and ori_ner.endswith("*") and len(ori_ner) > 2:
                flag = ori_ner[1:-1]
                ner = "B-" + flag
            elif ori_ner.startswith("(") and ori_ner.endswith(")") and len(ori_ner) > 2 and flag == None:
                ner = "B-" + ori_ner[1:-1]

            converted_line.extend([word, pos, cons, ner])
            converted_line.extend(extra)

            # text += " ".join([word, pos, cons, ner]) + " "
            # text += " ".join(extra) + '\n'
            converted_lines.append(converted_line)
        else:
            # text += '\n'
            converted_lines.append(line)
    # text += '\n'

    return converted_lines

def conll2003_to_ditk(lines):
    converted_lines = []
    for line in lines:
        if len(line.strip()) > 0:
            if '-DOCSTART-' in line:
                continue
            data = line.split()
            converted_line = list()
            converted_line.extend(data[0:4])
            converted_line.extend(list(' -' * 8))
            converted_lines.append(converted_line)
        else:
            converted_lines.append(line)
            print('Here')
    return converted_lines

def is_data_conll12(data):
    conll_2012 = False
    for line in data[:min(5, len(data))]:
        if not conll_2012 and len(line) > 4 and line[4] and line[4] != '-' and line[5] and line[5] != '-' and line[6] and line[6] != '-':
            conll_2012 = True
    return conll_2012

def convert_to_line_format(data):
    converted_data = []
    for line in data:
        if isinstance(line, list):
            converted_data.append(' '.join(line))
        else:
            converted_data.append(line)
    return converted_data

def run_fn_over_keys(data_dict, function):
    for key in data_dict.keys():
        split_data = data_dict[key]
        data_dict[key] = function(split_data)
    return data_dict

def convert_data_to_conll(file):
    lines = None
    with open(file, mode='r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    if not lines:
        return None

    is_conll_2012 = False

    converted_data = None
    if is_conll_2012:
        converted_data = ditk_ner_to_conll2012(lines)
    else:
        converted_data = ditk_ner_to_conll2003(lines)

    print('Writing to file')
    with open('temp.txt', 'w') as f:
        for data in converted_data:
            if isinstance(data, list):
                data = "\t".join(data)
                data = data
            f.write(data +'\n')
    return converted_data

def convert_data_to_ditk(file):
    with open(file, mode='r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    converted_data = conll2003_to_ditk(lines)
    with open('temp2.txt', 'w') as f:
        for data in converted_data:
            if isinstance(data, list):
                data = "\t".join(data)
                data = data
            f.write(data+'\n')

def get_ne_tags(data):
    ne_list = set()
    for line in data:
        if len(line) > 0:
            if str(line[3]).startswith('B-') or str(line[3]).startswith('I-'):
                ne_tag = line[3][2:]
                ne_list.add(ne_tag)
    return sorted(ne_list)

def get_pos_tags(data):
    pos_list = set()
    for line in data:
        if len(line) > 0:
            pos_tag = line[1]
            pos_list.add(pos_tag)

            chunk_tag_format = is_chunk_tag(line[2])

            if not chunk_tag_format:
                tokens = str(line[2])
                tokens = re.findall(r"[\w']+", tokens)
                pos_list = pos_list.union(set(tokens))
    return sorted(pos_list)

def get_raw_sentences(data):
    sentence_data = []
    sentence = []
    for line in data:
        if len(line) > 1:
            sentence.append((line[0], line[3]))
        else:
            if len(sentence) > 0:
                sentence_data.append(sentence)
                sentence = []

    sentence_data.append(sentence)
    return sentence_data

if __name__ == "__main__":
    # convert_data_to_conll("C:/Users/kkuna/PycharmProjects/tf_rnn/load_conll_2012/nbc_0001.train.ner")
    convert_data_to_ditk('D:\\Study\\CSCI 548 IIW\\Project\\Tasks\\TestCases\\conll2003/sample.txt')