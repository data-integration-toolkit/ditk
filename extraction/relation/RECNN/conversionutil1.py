import utils
def Common_to_SemEval(inputfile):
    data = []
    semevallist=[]
    relationlist={}
    revrelationlist={}
    flag=0
    i=0
    lines = [line.strip() for line in open(inputfile)]
    for line in lines:
        semevallist.append(line)
    with open("ouptutfile", "w") as f:
        for idx in range(len(lines)):
            tokens = lines[idx].split("\t")
            sentence = tokens[0]
            e1 = tokens[1]
            e1_pos_start = int(tokens[3])
            e1_pos_end = int(tokens[4])
            e2 = tokens[5]
            e2_pos_start = int(tokens[7])
            e2_pos_end = int(tokens[8])
            relation = tokens[9]
            if relation not in utils.class2label.keys():
                relationlist[relation]=i
                revrelationlist[i]=relation
            i=i+1
            sentence = sentence[:e1_pos_start] + '<e1>' + e1 + '</e1>' + sentence[e1_pos_end:e2_pos_start] + '<e2>' + e2 + '</e2>' + sentence[e2_pos_end:]
            f.writelines(str(idx+1) + '\t\"' + sentence + '\"\n' + relation + '\n' + 'Comment:' + '\n\n')

            data.append([idx+1, sentence, e1, e1_pos_start, e1_pos_end, e2, e2_pos_start, e2_pos_end, relation])
    utils.class2label=relationlist
    utils.label2class=revrelationlist
    return data

