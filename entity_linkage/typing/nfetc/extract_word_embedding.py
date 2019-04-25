def remove_words():
    wordSet = set()
    with open("./data/corpus/Wiki/tokens.txt", "r") as f:
        for line in f:
            s = line.strip("\n")
            wordSet.add(s)

    cnt = 0
    embeds = []
    with open("./data/glove.840B.300d_original.txt", "r") as f:
        for line in f:
            line_list = line.strip("\n").split()
            word, embed = line_list[0], line_list[1:]
            if len(embed) > 1:
                if word in wordSet:
                    cnt += 1
                    embeds.append(line)


    with open("./data/glove.840B.300d_revised.txt", "w") as f:
        f.write(str(len(embeds))+" 300\n")
        for embed in embeds:
            f.write(embed)

remove_words()