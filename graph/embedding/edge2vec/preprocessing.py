def prep():
    f = open("input.txt","r")
    o = open("data2.txt","w")
    counte = 1
    countr = 1
    count = 1
    entity = {}
    relations = {}
    for line in f:
        arr = line.strip().split("\t")
        if len(arr) != 3:
            break
        if arr[0] not in entity.keys():
            entity[arr[0]] = counte
            counte+=1

        if arr[2] not in entity.keys():
            entity[arr[2]] = counte
            counte+=1

        if arr[1] not in relations.keys():
            relations[arr[1]] = countr
            countr+=1
    f.close()
    f = open("data1.txt","r")
    for line in f:
        arr = line.strip().split("\t")
        if len(arr) != 3:
            break
        # print str(entity[arr[0]])+str(entity[arr[2]])+str(relations[arr[1]])+str(count)
        o.writelines(str(entity[arr[0]])+" "+str(entity[arr[2]])+" "+str(relations[arr[1]])+" "+str(count)+"\n")
        count+=1

    return (relations[max(relations, key=relations.get)])

    # print entity
    # print relations
prep()
