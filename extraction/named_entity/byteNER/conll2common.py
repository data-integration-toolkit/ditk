# Expected usage: Before running the module to generate files in DITK format
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

def convert_data_to_ditk(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    converted_data = conll2003_to_ditk(lines)
    with open(output_file, 'w') as f:
        for data in converted_data:
            if isinstance(data, list):
                data = "\t".join(data)
                data = data
            f.write(data+'\n')

input_path = "/Users/lixiangci/repos/NER/corpus/CoNLL-2003/"
output_path = "/Users/lixiangci/repos/ditk/extraction/named_entity/byteNER/examples/"
input_names = ["eng.train","eng.testa","eng.testb"]
output_names = ["training.tsv","development.tsv","evaluation.tsv"]

for input_name, output_name in zip(input_names, output_names):
    convert_data_to_ditk(input_path+input_name, output_path+output_name)