#!/bin/bash
mkdir -p DDI
mkdir -p NYT
mkdir -p SemEval

datafiles=(DDI.txt SemEval.txt NYT.txt)
datafileIDs=(1FiRAKARX_ivYTRC8sqi3QHNs9JWdQhXx 16n_ggzqtI6JoVj0t3azPtGN__MNsGGlV 1vE5F3XPTINv9XvnsbmXwlHCo8osQA9pe)

for index in ${!datafiles[*]}; do
  bash gdl.sh ${datafileIDs[$index]} ${datafiles[$index]}
done

mv DDI.txt DDI
mv SemEval.txt SemEval
mv NYT.txt NYT

wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

gunzip GoogleNews-vectors-negative300.bin.gz
