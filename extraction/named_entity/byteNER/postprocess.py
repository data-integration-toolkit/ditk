"""
Postprocessing functions
- Convert ordinal byte IOB file into input format (src/tgt)
- Convert ordinal byte IOB file into byte IOB file
"""


def byte_iob_to_src_tgt(iob_file, ordinal=True):
    """
    Convert ordinal byte IOB file (prediction file = expect col 1 to be token, col 2 to be pred tag) into input format (src/tgt)
    :param iob_file:
    :return:
    """
    output_src_file = iob_file + '.src'
    output_tgt_file = iob_file + '.tgt'
    sentences = []
    anns = []
    sentence = []
    ann = []
    curr_ann = []
    curr_ann_idx = 0

    with open(iob_file, 'rb') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if len(line) > 0:
                split = line.split()
                if ordinal:
                    chr_tok = chr(int(split[0].split(',')[0]))
                else:
                    if len(split) == 1:
                        chr_tok = ' '
                    else:
                        chr_tok = split[0]
                sentence.append(chr_tok)
                tag = split[-1]

                if tag.startswith('B-'):
                    if len(curr_ann) == 2:  # insert length and type of prev entity
                        curr_ann = ('S' + str(curr_ann[1]), 'L' + str(curr_ann_idx - curr_ann[1]), curr_ann[0])
                        ann.insert(0, ' '.join(curr_ann))
                        curr_ann = []
                    tag_type = tag[2:]
                    curr_ann.append(tag_type)
                    curr_ann.append(curr_ann_idx)
                elif tag == 'O':
                    if len(curr_ann) == 2:  # insert length and type of prev entity
                        curr_ann = ['S' + str(curr_ann[1]), 'L' + str(curr_ann_idx - curr_ann[1]), curr_ann[0]]
                        ann.insert(0, ' '.join(curr_ann))
                        curr_ann = []
                else:
                    if len(curr_ann) != 2:
                        print 'seeing stray I- tag at line ', line_idx
                        # treat as B- tag
                        tag_type = tag[2:]
                        curr_ann.append(tag_type)
                        curr_ann.append(curr_ann_idx)
                curr_ann_idx += 1
                # prev_prev_chr_tok = prev_chr_tok
                # prev_chr_tok = chr_tok
            else:
                if len(curr_ann) == 2:  # insert length of prev entity
                    curr_ann = ('S' + str(curr_ann[1]), 'L' + str(curr_ann_idx - curr_ann[1]), curr_ann[0])
                    ann.insert(0, ' '.join(curr_ann))
                    curr_ann = []
                if len(sentence) > 0:
                    sentences.append(''.join(sentence))
                    anns.append(' '.join(ann))
                sentence = []
                ann = []
                curr_ann_idx = 0
                # prev_chr_tok = None

        if len(sentence) > 0:  # last sentence and annotations
            sentences.append(''.join(sentence))
            anns.append(' '.join(ann))

    with open(output_src_file, 'wb') as s, open(output_tgt_file, 'wb') as t:
        s.write('\n'.join(sentences) + '\n')
        t.write('\n'.join(anns) + '\n')


def ord_byte_iob_to_byte_iob(iob_file):
    """
    Convert ordinal byte IOB file into byte iob format
    :param iob_file:
    :return:
    """
    output_byte_iob_file = iob_file + '.chr'
    with open(iob_file, 'rb') as i, open(output_byte_iob_file, 'wb') as o:
        for line in i:
            if len(line.strip()) > 0:
                split = line.split()
                tok = split[0].split(',')[0]
                tok_char = chr(int(tok))
                if chr(int(tok)) == ' ':
                    tok_char = '<SPACE>'
                o.write(tok_char + ' ' + ' '.join(split[1:]) + '\n')
            else:
                o.write(line)
