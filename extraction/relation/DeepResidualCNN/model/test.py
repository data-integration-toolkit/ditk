import numpy as np
from utils.DataManager import DataManager

def test(testing_data, input_x, input_p1, input_p2, s, p, dropout_keep_prob, datamanager, sess, num_epoch,relationdict,dataobject):
    output_file='predictions.txt'
    results = []
    total = 0
    sequence_length=100
    i = 0
    t = 0
    c = 0
    j=0
    prediction_file = open(output_file, 'w')
    for test in testing_data:
        i += 1
        x_test = datamanager.generate_x(testing_data[test])
        p1, p2 = datamanager.generate_p(testing_data[test])
        y_test = datamanager.generate_y(testing_data[test])
        scores, pre = sess.run([s, p], {input_x: x_test, input_p1:p1, input_p2:p2, dropout_keep_prob: 1.0})
        max_pro = 0
        prediction = -1
        for score in scores:
            score = np.exp(score-np.max(score))
            score = score/score.sum(axis=0)
            score[0] = 0
            pro = score[np.argmax(score)]
            if pro > max_pro and np.argmax(score)!=0:
                max_pro = pro
                prediction = np.argmax(score)
        for i in range(len(testing_data[test])):
            results.append((test, testing_data[test][i].relation.id, max_pro, prediction))
            dm = DataManager(sequence_length,relationdict)
            relation_data = list(open("data/relation2id.txt", encoding="utf8").readlines())
            relation_data = [s.split() for s in relation_data]
            for relation in relation_data:
                if int(relation[1]) == testing_data[test][i].relation.id:
                    print("Predicted relation is",relation[0])
                    prelation=relation[0]

            if testing_data[test][i].relation.id == pre and pre!=0:
                c += 1
            t += 1
            if testing_data[test][i].relation.id != 0:
                total += 1
        list1=dataobject[j].split()
        j=j+1
        prediction_file.writelines("\t".join([' '.join(list1[5:-1]),list1[2],list1[3],list1[4],prelation+'\n']))
    print("Correct: "+str(c))
    print("Total: "+str(t))
    print("Accuracy: "+str(float(c)/float(t)))
    results = sorted(results, key=lambda t: t[2])
    results.reverse()
    correct = 0
    f = open("re-9-128_precision_recall_"+str(num_epoch)+".txt", "w")
    for i in range(total):
        if results[i][1] == results[i][3]:
            correct += 1
        if i%100 == 0:
            print("Precision: "+str(float(correct)/float(i+1))+"  Recall: "+str(float(correct)/float(total)))
        f.write(str(float(correct)/float(i+1))+"    "+str(float(correct)/float(total))+"    "+str(results[i][2])+"  "
                +results[i][0]+"  "+str(results[i][3])+"\n")