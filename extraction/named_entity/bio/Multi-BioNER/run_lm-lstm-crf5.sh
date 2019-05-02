python3 train_wc.py --train_file data_bioner_5/BC2GM-IOBES/merge.tsv data_bioner_5/BC4CHEMD-IOBES/merge.tsv \
                    --dev_file data_bioner_5/BC2GM-IOBES/devel.tsv data_bioner_5/BC4CHEMD-IOBES/devel.tsv \
                    --test_file data_bioner_5/BC2GM-IOBES/test.tsv data_bioner_5/BC4CHEMD-IOBES/test.tsv \
                    --caseless --fine_tune --emb_file data_bioner_5/source.txt --word_dim 50 --gpu 0 --shrink_embedding --patience 1 --epoch 3
