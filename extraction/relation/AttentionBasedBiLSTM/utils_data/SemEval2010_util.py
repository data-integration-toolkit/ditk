def Common_to_SemEval2010(common_data,dataType):
    data = []
    lines = common_data
    # with open("SemEval2010_train", "w") as f:
    for idx in range(len(lines)):
        tokens = lines[idx]
        sentence = tokens[0]
        e1 = tokens[1]
        if( ';' in tokens[3]):
            tokens[3] = tokens[3].split(';')[0]
        e1_pos_start = int(tokens[3])
        if (';' in tokens[4]):
            tokens[4] = tokens[4].split(';')[0]
        e1_pos_end = int(tokens[4])
        e2 = tokens[5]
        if (';' in tokens[7]):
            tokens[7] = tokens[7].split(';')[0]
        e2_pos_start = int(tokens[7])
        if (';' in tokens[8]):
            tokens[8] = tokens[8].split(';')[0]
        e2_pos_end = int(tokens[8])
        relation = tokens[9]

        sentence = sentence[:e1_pos_start] + '<e1>' + e1 + '</e1>' + sentence[e1_pos_end:e2_pos_start] + '<e2>' + e2 + '</e2>' + sentence[e2_pos_end:]
        # f.writelines(str(idx+1) + '\t\"' + sentence + '\"\n' + relation + '\n' + 'Comment:' + '\n\n')
        #

        data.append([idx + 1, sentence, e1, e1_pos_start, e1_pos_end, e2, e2_pos_start, e2_pos_end, relation])


    return data

def SemEval2010_to_common(inputfile):
    data = []
    lines = [line.strip() for line in open(inputfile)]

    with open("train.txt", "w") as f:
        for idx in range(0, len(lines), 4): # Change 4 to 3 when the inputfile does not have "comment" (3rd line).
            relation = lines[idx + 1]
            sentence = lines[idx].split("\t")[1][1:-1].strip('\"')
            e1_pos_start = sentence.find('<e1>')
            e1_pos_end = sentence.find('</e1>')
            e1 = sentence[e1_pos_start:e1_pos_end+len('</e1>')]
            sentence = sentence.replace(e1, e1[len('<e1>'):-len('</e1>')])
            e1 = e1[len('<e1>'):-len('</e1>')]
            e1_pos_end = e1_pos_end-len('<e1>')

            e2_pos_start = sentence.find('<e2>')
            e2_pos_end = sentence.find('</e2>')
            e2 = sentence[e2_pos_start:e2_pos_end + len('</e2>')]
            sentence = sentence.replace(e2, e2[len('<e2>'):-len('</e2>')])
            e2 = e2[len('<e2>'):-len('</e2>')]
            e2_pos_end = e2_pos_end - len('<e2>')

            data.append([sentence, e1, 'Null', e1_pos_start, e1_pos_end, e2, 'Null', e2_pos_start, e2_pos_end, relation])
           # print("\t".join([sentence, e1, 'Null', str(e1_pos_start), str(e1_pos_end), e2, 'Null', str(e2_pos_start), str(e2_pos_end), relation]))
            f.writelines("\t".join([sentence, e1, 'Null', str(e1_pos_start), str(e1_pos_end), e2, 'Null', str(e2_pos_start), str(e2_pos_end), relation, '\n']))

    return data


if __name__ == "__main__":
    common_data = SemEval2010_to_common('SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT')
    data =Common_to_SemEval2010(common_data)
