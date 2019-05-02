def SemEval_to_common(inputfile):
    data = []
    lines = inputfile

    for line in lines: # Change 4 to 3 when the inputfile does not have "comment" (3rd line).
        # relation = lines[idx + 1]
        sentence = line[1]
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

        data.append(sentence)
        print("\t".join([sentence, e1, 'Null', str(e1_pos_start), str(e1_pos_end), e2, 'Null', str(e2_pos_start), str(e2_pos_end)]))
        # f.writelines("\t".join([sentence, e1, 'Null', str(e1_pos_start), str(e1_pos_end), e2, 'Null', str(e2_pos_start), str(e2_pos_end), relation, '\n']))

    return data
