def Common_to_NYT(inputfile):
    data = []
    m = 1
    semevallist=[]
    relationlist={"NA":0}
    lines = [line.strip() for line in open(inputfile)]
    for line in lines:
        semevallist.append(line)
    with open("ouptutfile", "w") as f:
        for idx in range(len(lines)):
            tokens = lines[idx].split("\t")
            sentence = tokens[0]
            e1 = tokens[1]
            e1_pos_start = tokens[3]
            e1_pos_end = tokens[4]
            e2 = tokens[5]
            e2_pos_start = tokens[7]
            e2_pos_end = tokens[8]
            relation = tokens[9]
            i=0
            j=0
            if relation not in relationlist:
                relationlist[relation]=m
                m=m+1
            sentence1=str(i)+'\t'+str(j)+'\t'+e1+'\t'+e2+'\t'+relation+'\t'+sentence+'\t'+'###END### '
            # sentence = sentence[:e1_pos_start] + '<e1>' + e1 + '</e1>' + sentence[e1_pos_end:e2_pos_start] + '<e2>' + e2 + '</e2>' + sentence[e2_pos_end:]
            f.writelines(sentence1)
            i+=1
            j+=1

            data.append(sentence1)

    return data,relationlist

