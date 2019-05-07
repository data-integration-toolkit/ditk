#!/usr/bin/env python
# encoding: utf-8

import sys
infile = sys.argv[1]
outfile = sys.argv[2]
fout = open(outfile,'w')
values = ['DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL']

for line in open(infile).readlines():
    ls = line.strip().split()
    if len(ls) > 5:
        need_change = False
        for v in values:
            if v in ls[-1]:
                need_change = True
                break
        if need_change:
            ls[-1] = 'O'
        line = '\t'.join([ls[1],ls[-1]])+'\n'
    fout.write(line)

