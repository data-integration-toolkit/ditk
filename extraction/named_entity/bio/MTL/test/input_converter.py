
def converter(input_path, output_file):
    f = open(input_path, "r")
    lines = f.readlines()
    f.close()
    result = []
    for line in lines:
        if len(line)>2:
            tmp = line.split(' ')
            result.append("\t".join([tmp[0],tmp[3]]) + "\n")
        else:
            result.append("\n")

    open(output_file, 'w').write(''.join(result))

converter("./ner_input.txt", "./ner_input_converted.txt")