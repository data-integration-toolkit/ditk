import pandas as pd

def load_dict_from_txt(path):
    d = {}
    with open(path) as f:
        for line in f.readlines():
            a, b = line.strip().split()
            d[a] = int(b)
    return d

class DataSet:
    def __init__(self, d):
        self.train_raw = d["train_raw"]
        self.valid_raw = d["valid_raw"]
        self.test_raw = d["test_raw"]
        self.train = d["train"]
        self.valid = d["valid"]
        self.test = d["test"]
        self.e2id = d["e2id"]
        self.r2id = d["r2id"]

    def load_raw_data(self):
        df_train = pd.read_csv(self.train_raw, sep="\t", names=["e1", "r", "e2"])
        df_valid = pd.read_csv(self.valid_raw, sep="\t", names=["e1", "r", "e2"])
        df_test = pd.read_csv(self.test_raw, sep="\t", names=["e1", "r", "e2"])
        return df_train, df_valid, df_test

    def load_data(self):
        df_train = pd.read_csv(self.train, names=["e1", "r", "e2"]).as_matrix()
        df_valid = pd.read_csv(self.valid, names=["e1", "r", "e2"]).as_matrix()
        df_test = pd.read_csv(self.test, names=["e1", "r", "e2"]).as_matrix()
        return df_train, df_valid, df_test

    def load_idx(self):
        e2id = load_dict_from_txt(self.e2id)
        r2id = load_dict_from_txt(self.r2id)
        return e2id, r2id

    def save_e2id(self, eSet):
        outfile = open(self.e2id, "w")
        e2id = {}
        for idx, e in enumerate(sorted(eSet)):
            e2id[e] = idx
            outfile.write("%s %d\n" % (e, idx))
        outfile.close()
        return e2id

    def save_r2id(self, rSet):
        outfile = open(self.r2id, "w")
        r2id = {}
        for idx, r in enumerate(sorted(rSet)):
            r2id[r] = idx
            outfile.write("%s %d\n" % (r, idx))
        outfile.close()
        return r2id

    def save_data(self, train, valid, test):
        train.to_csv(self.train, header=False, index=False)
        valid.to_csv(self.valid, header=False, index=False)
        test.to_csv(self.test, header=False, index=False)
