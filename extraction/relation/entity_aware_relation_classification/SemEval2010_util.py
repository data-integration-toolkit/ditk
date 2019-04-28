def Common_to_SamEval2010(inputfile):
    def Common_token_to_SamEval2010_token(tokens):
        sentence = tokens[0]
        e1 = tokens[1]
        e1_pos_start = int(tokens[3])
        e1_pos_end = int(tokens[4])
        e2 = tokens[5]
        e2_pos_start = int(tokens[7])
        e2_pos_end = int(tokens[8])
        relation = tokens[9]
        sentence = sentence[:e1_pos_start] + '<e1>' + e1 + '</e1>'\
                   + sentence[e1_pos_end:e2_pos_start] + '<e2>' + e2 + '</e2>' + sentence[e2_pos_end:]
        return [sentence, relation]

    data = []
    lines = [line.strip() for line in open(inputfile)]
    # with open("SamEval2010_"+inputfile, "w") as f:
    for idx in range(len(lines)):
        tokens = lines[idx].split("\t")
        sam_eval = Common_token_to_SamEval2010_token(tokens)
        # f.writelines(str(idx+1) + '\t\"' + sam_eval[0] + '\"\n' + sam_eval[7] + '\n' + 'Comment:' + '\n\n')
        data.append(sam_eval)

    return data


def SemEval2010_to_common(inputfile):
    data = []
    lines = [line.strip() for line in open(inputfile)]

    with open("./data/SamEval2010/testfile.txt", "w") as f:
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

            # data.append([sentence, e1, None, e1_pos_start, e1_pos_end, e2, None, e2_pos_start, e2_pos_end, relation])
            f.writelines("\t".join([sentence, e1, 'Null', str(e1_pos_start), str(e1_pos_end), e2, 'Null', str(e2_pos_start), str(e2_pos_end), relation, '\n']))

    # return data


if __name__ == "__main__":
    SemEval2010_to_common('./SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT')
    # Common_to_SamEval2010('common_test.txt')