def common(c):
    if (c >= 'a' and c <= 'z'):
        return True
    if (c >= 'A' and c <= 'Z'):
        return True
    if (c == '-'):
        return True
    return False

def convert(origin_file, coverted_file):
    fp_input = open(origin_file, 'r')
    fp_output = open(coverted_file, "w")
    inputs = fp_input.read().strip().split('\n')

    diction = {}

    for i in range(len(inputs)):
        input_attr = inputs[i].strip().split('\t')
        fp_output.write(str(i) + "\n")
        e1_s = int(input_attr[3])
        e1_e = int(input_attr[4].split(';')[len(input_attr[4].split(';')) - 1])
        e2_s = int(input_attr[7])
        e2_e = int(input_attr[8].split(';')[len(input_attr[8].split(';')) - 1])
        origin = input_attr[0].strip()
        length = len(origin)
        sentence = ""
        cnt = 0
        for i in range(0, e1_s):
            if (origin[i] == ' '):
                sentence += ' '
                cnt += 1
            else:
                if not common(origin[i]):
                    if ((i + 1) != length) and (origin[i + 1] == ' ') and (len(sentence) != 0) and (sentence[len(sentence) - 1] != ' '):
                        sentence += " " + origin[i]
                        cnt += 1
                    else:
                        sentence += origin[i]
                else:
                    sentence += origin[i]
        if (e1_s > 0) and (origin[e1_s - 1] != ' '):
            sentence += " "
            cnt += 1
        s1 = cnt
        sentence += input_attr[1]
        cnt += len(input_attr[1].strip().split(' ')) - 1
        e1 = cnt
        if (e1_e + 1 < length) and (origin[e1_e + 1] != ' '):
            sentence += ' '
            cnt += 1

        for i in range(e1_e + 1, e2_s):
            if (origin[i] == ' '):
                sentence += ' '
                cnt += 1
            else:
                if not common(origin[i]):
                    if ((i + 1) != length) and (origin[i + 1] == ' ') and (sentence[len(sentence) - 1] != ' '):
                        sentence += " " + origin[i]
                        cnt += 1
                    else:
                        sentence += origin[i]
                else:
                    sentence += origin[i]
        if (e2_s > 0) and (origin[e2_s - 1] != ' '):
            sentence += " "
            cnt += 1
        s2 = cnt
        sentence += input_attr[5]
        cnt += len(input_attr[5].strip().split(' ')) - 1
        e2 = cnt
        if (e2_e + 1 < length) and (origin[e2_e + 1] != ' '):
            sentence += ' '
            cnt += 1
        for i in range(e2_e + 1, length):
            if (origin[i] == ' '):
                sentence += ' '
                cnt += 1
            else:
                if not common(origin[i]):
                    if ((i + 1) != length) and (origin[i + 1] == ' ') and (sentence[len(sentence) - 1] != ' '):
                        sentence += " " + origin[i]
                        cnt += 1
                    else:
                        sentence += origin[i]
                else:
                    sentence += origin[i]
        if (sentence[len(sentence) - 1] == '.') and (sentence[len(sentence) - 2] != ' '):
            sentence = sentence[:-1] +  ' .'

        fp_output.write(sentence + '\n')
        fp_output.write("(['" + input_attr[1].lower() + "', " + str(s1) + ", "
        + str(e1) + ", '" + input_attr[2].strip() + "']")
        fp_output.write(", ['" + input_attr[5].lower() + "', " + str(s2) + ", "
        + str(e2) + ", '" + input_attr[6].strip() + "'])\n")
        if (input_attr[9] == 'false'):
            fp_output.write('[0]')
        else:
            fp_output.write("['" + input_attr[5].lower() + "', '" + input_attr[9] + "', '" + input_attr[1].lower() + "']")
        diction[input_attr[9]] = 1
        fp_output.write('\n\n\n')
