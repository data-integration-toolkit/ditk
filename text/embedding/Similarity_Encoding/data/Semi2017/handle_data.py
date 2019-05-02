import pandas as pd

my_data = pd.read_csv('./sts-test-2017-org.csv', delimiter=',', header=None)
my_data = my_data.loc[:,5:6]
my_data.columns = ["S1","S2"]

file = open("input.txt","w", encoding="utf8")
for index,row in my_data.iterrows():
    file.write(str(row["S1"]))
    file.write("\n")
    file.write(str(row["S2"]))
file.close()
