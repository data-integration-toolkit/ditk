For NFETC model, you can put 2 types of input; **raw_data**, **clean_data**.
<br>
1) **clean_data.tsv**<br>
Corpus input format is ["start position", "end position", "text", "mention", "Type(s)"]. 
There are 3 lines of sample clean data input.
```
6	7	OK , we 'll call him General Ye Ting .	him General Ye	/person/title
7	8	OK , we 'll call him General Ye Ting .	General Ye Ting	/location
0	1	Very naughty ?	<PAD> Very naughty	/other/art/music
```

2) **raw_data.txt**<br>
If you download publicabily avaliable data that the original NFETC github suggests to download, you will get raw data. (http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip)
It will have one more column, "f", at the end; ["start position", "end position", "text", "mention", "Type(s)", "f"]

```
18	19	Usually directors , otherwise , they have beards and very long hair , or otherwise they shave their heads .	/other /other/body_part	HEAD|heads PARENT|shave CLUSTER|1011 CLUSTER|101110 CLUSTER|1011100010 CLUSTER|10111000100 CHARACTERS|:he CHARACTERS|hea CHARACTERS|ead CHARACTERS|ads CHARACTERS|ds: SHAPE|a ROLE|dobj BEFORE|their AFTER|. TOPIC|12 
1	2	Usually directors , otherwise , they have beards and very long hair , or otherwise they shave their heads .	/person/title /person	HEAD|directors PARENT|have CLUSTER|1011 CLUSTER|101110 CLUSTER|1011101110 CLUSTER|1011101110100 CHARACTERS|:di CHARACTERS|dir CHARACTERS|ire CHARACTERS|rec CHARACTERS|ect CHARACTERS|cto CHARACTERS|tor CHARACTERS|ors CHARACTERS|rs: SHAPE|a ROLE|nsubj BEFORE|Usually AFTER|, TOPIC|12 
0	1	Oh .	/person/athlete /person	HEAD|Oh PARENT|ROOT CLUSTER|1111 CLUSTER|111100 CLUSTER|1111000100 CLUSTER|11110001001 CHARACTERS|:oh CHARACTERS|oh: SHAPE|Aa ROLE|root BEFORE|<s> AFTER|. TOPIC|12 
```

## To run test.py with sample test data
Initial setting is that model will run with sample clean data and it will check number of input/output rows and cols. 
Run on the nfetc folder with following command line; ``` python test/test.py```


<br><br>
### Embedding (glove.840B.300d)<br>
If you want to use other inputs, you must download word embedding file form http://nlp.stanford.edu/data/glove.840B.300d.zip and replace current golve file to full size of golve word embedding file.
I uploaded downsized word embedding file due to the limitation of size (under data folder in develop branch)
