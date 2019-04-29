import json
import os

def nyt_to_common(inputfile, outputfile, num_of_relations_per_sentence=0):
    label = set()
    with open(outputfile, 'w') as f:
        outputs = []
        for line in open(inputfile, mode='r', encoding='utf-8'):
            data_dic = json.loads(line)

            sent_text = data_dic['sentText'].strip().strip('"').strip("'")

            entities = {}
            entity_mentions = data_dic['entityMentions']
            for i in range(len(entity_mentions)):
                entity = entity_mentions[i]['text']
                start_pos = sent_text.index(entity)
                end_pos = start_pos + len(entity)
                entity_type = entity_mentions[i]['label']
                entities[entity] = {'s': start_pos, 't': end_pos, 'type': entity_type}

            relation_mentions = data_dic['relationMentions']
            cnt = 0
            for i in range(len(relation_mentions)):
                if 0 != num_of_relations_per_sentence and num_of_relations_per_sentence == cnt:
                    break

                e1 = relation_mentions[i]['em1Text']
                e2 = relation_mentions[i]['em2Text']
                relation = relation_mentions[i]['label']
                try:
                    res = [sent_text, e1, entities[e1]['type'], str(entities[e1]['s']), str(entities[e1]['t']),
                           e2, entities[e2]['type'], str(entities[e2]['s']), str(entities[e2]['t']), relation]

                    outputs.append('\t'.join(res) + '\n')
                    cnt = cnt + 1
                    label.add(relation)
                except KeyError:
                    continue
                except UnicodeEncodeError:
                    continue
        
        for output in outputs[:10000]:
            f.write(output)
    return label


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label = nyt_to_common(dir_path+'\\train.json', dir_path+'\\relation_extraction_input_train.txt', 1)
    print(label)
    nyt_to_common(dir_path+'\\test.json', dir_path+'\\relation_extraction_input_test.txt', 1)
