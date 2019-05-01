def NYT_to_Common():
    inputfile="train.txt"
    data = []
    lines = [line.strip() for line in open(inputfile)]
    print("hello")
    with open("common_"+inputfile, "w") as f:
        for line in lines:
            linecontent=line.split()
            print(linecontent[0])
            print(linecontent[1])
            print(linecontent[2])
            print(linecontent[3])
            print(linecontent[4])
            print(' '.join(linecontent[5:-2]))

            data.append([' '.join(linecontent[5:-2]), linecontent[2], 'Null','Null', 'Null', linecontent[3], 'Null', 'Null', 'Null', linecontent[4]])
            f.writelines("\t".join([' '.join(linecontent[5:-2]), linecontent[2], 'Null','Null', 'Null', linecontent[3], 'Null', 'Null', 'Null', linecontent[4],'\n']))



if __name__ == "__main__":
    label = NYT_to_Common()
    print(label)
