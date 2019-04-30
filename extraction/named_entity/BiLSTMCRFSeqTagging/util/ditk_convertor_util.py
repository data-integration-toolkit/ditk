import re
conll_2000_tagset = {'NP', 'ADVP', 'ADJP', 'VP', 'PP', 'SBAR', 'CONJP', 'PRT', 'INTJ', 'LST', 'UCP'}


def is_tag_bio(tag):
    return str(tag).startswith('B-') or str(tag).startswith('I-') or tag == 'O'


def ditk_ner_to_conll2012(lines):
    converted_lines = []
    for i, line in enumerate(lines):
        if len(line.strip()) > 0:
            data = line.split()
            converted_line = list()
            converted_line.extend(data[4:7])
            converted_line.extend(data[0:3])
            converted_line.extend(data[7:11])
            converted_line.append(data[3])
            converted_line.extend(data[11:])
            converted_lines.append(converted_line)

    end_doc = "#end document\n"
    start_doc_template = "#begin document (%s); part %s\n"

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

    conll2012_lines.append('\n' + end_doc)

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
    return converted_lines


# Expected usage: Internally when using the code. i.e while preprocessing during train, predict
def convert_data_to_conll(file):
    with open(file, mode='r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    converted_data = ditk_ner_to_conll2003(lines)

    print('Writing to file')
    with open('temp.txt', 'w') as f:
        for data in converted_data:
            if isinstance(data, list):
                data = "\t".join(data)
                data = data
            f.write(data +'\n')
    return converted_data


# Expected usage: Before running the module to generate files in DITK format
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