import subprocess
import os
import datetime

from configure import FLAGS
import utils


class Logger:
    def __init__(self, out_dir):
        self.log_dir = os.path.abspath(os.path.join(out_dir, "logs"))
        try:
            os.makedirs(self.log_dir)
        except FileExistsError:
            pass

        timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.log_path = os.path.abspath(os.path.join(self.log_dir, "logs_{0}.txt".format(timestamp)))
        self.log_file = open(self.log_path, "w")

        self.print_hyperparameters()

        self.best_f1 = 0.0
        self.best_p = 0.0
        self.best_r = 0.0

    def print_hyperparameters(self):
        self.log_file.write("\n================ Hyper-parameters ================\n\n")
        for arg in vars(FLAGS):
            self.log_file.write("{}={}\n".format(arg.upper(), getattr(FLAGS, arg)))
        self.log_file.write("\n==================================================\n\n")

    def logging_train(self, step, loss, accuracy):
        time_str = datetime.datetime.now().isoformat()
        log = "{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)
        self.log_file.write(log+"\n")
        print(log)

    def logging_eval(self, step, loss, accuracy, predictions):
        self.log_file.write("\nEvaluation:\n")
        # loss & acc
        time_str = datetime.datetime.now().isoformat()
        log = "{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)
        self.log_file.write(log + "\n")
        print(log)

        # f1-score
        prediction_path = os.path.abspath(os.path.join(self.log_dir, "predictions.txt"))
        prediction_file = open(prediction_path, 'w')
        for i in range(len(predictions)):
            prediction_file.write("{}\t{}\n".format(i, utils.label2class[predictions[i]]))
        prediction_file.close()
        perl_path = os.path.join(os.path.curdir,
                                 "SemEval2010_task8_all_data",
                                 "SemEval2010_task8_scorer-v1.2",
                                 "semeval2010_task8_scorer-v1.2.pl")
        target_path = os.path.join(os.path.curdir, "resource", "target.txt")
        process = subprocess.Popen(["perl", perl_path, prediction_path, target_path], stdout=subprocess.PIPE)
        process_str = str(process.communicate()[0])
        str_tokens = process_str.split("\\n")
        # str_parse = str_tokens[-2]
        # idx = str_parse.find('%')
        # f1_score = float(str_parse[idx-5:idx])
        scores_tokens =  str_tokens[-6].split("\\t")
        p_score = float(scores_tokens[0][scores_tokens[0].find("=")+2:scores_tokens[0].find("%")])
        r_score = float(scores_tokens[1][scores_tokens[1].find("=")+2:scores_tokens[1].find("%")])
        f1_score = float(scores_tokens[2][scores_tokens[2].find("=")+2:scores_tokens[2].find("%")])

        if self.best_f1 < f1_score:
            self.best_f1 = f1_score
            self.best_r = r_score
            self.best_p = p_score

        f1_log = "<<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>>:\n" \
                 "macro-averaged F1-score = {:g}%, Best = {:g}%\n".format(f1_score, self.best_f1)
        f1_log = "{}Precision: {:g}%\nRecall: {:g}%".format(f1_log, p_score, r_score)
        self.log_file.write(f1_log + "\n")
        print(f1_log)
