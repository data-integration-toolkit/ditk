import numpy as np
from hyperopt import hp

# required params:
# - embedding_size
# - lr
# - batch_size
# - max_iter
# - neg_ratio
# - contiguous_sampling
# - valid_every: set it to 0 to enable early stopping

param_space_TransE = {
    # "embedding_size": hp.quniform("embedding_size", 50, 200, 10),
    "embedding_size": 200,
    "margin": hp.quniform("margin", 0.5, 5, 0.5),
    "lr": hp.qloguniform("lr", np.log(1e-4), np.log(1e-2), 1e-4),
    "batch_size": 5000,
    "max_iter": 100000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 5000,
}


param_space_TransE_fb3m = {
    # "embedding_size": hp.quniform("embedding_size", 50, 200, 10),
    "embedding_size": 50,
    "margin": hp.quniform("margin", 0.5, 5, 0.5),
    "lr": hp.qloguniform("lr", np.log(1e-3), np.log(1e-2), 2e-4),
    "batch_size": 5000,
    "max_iter": 500000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 20000,
}

param_space_best_TransE_L2_wn18 = {
    "embedding_size": 200,
    "margin": 0.5,
    "lr": 0.001,
    "batch_size": 2000,
    "max_iter": 2000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_best_TransE_L1_fb15k = {
    "embedding_size": 190,
    "margin": 3.5,
    "lr": 0.001,
    "batch_size": 5000,
    "max_iter": 15000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_best_TransE_L1_fb3m = {
    "embedding_size": 100,
    "margin": 4.5,
    "lr": 0.001,
    "batch_size": 5000,
    "max_iter": 120000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_DistMult = {
    # "embedding_size": hp.quniform("embedding_size", 50, 200, 10),
    "embedding_size": 200,
    "l2_reg_lambda": hp.qloguniform("l2_reg_lambda", np.log(1e-3), np.log(1e-1), 1e-3),
    "lr": hp.qloguniform("lr", np.log(1e-4), np.log(1e-2), 1e-4),
    "batch_size": 5000,
    "max_iter": 100000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 5000,
}

param_space_DistMult_fb3m = {
    # "embedding_size": hp.quniform("embedding_size", 50, 200, 10),
    "embedding_size": 50,
    "l2_reg_lambda": hp.qloguniform("l2_reg_lambda", np.log(1e-3), np.log(1e-1), 1e-3),
    "lr": hp.qloguniform("lr", np.log(1e-4), np.log(1e-2), 1e-4),
    "batch_size": 5000,
    "max_iter": 500000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 20000,
}

param_space_best_DistMult_tanh_wn18 = {
    "embedding_size": 150,
    "l2_reg_lambda": 0.0026,
    "lr": 0.011,
    "batch_size": 2000,
    "max_iter": 15000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_best_DistMult_tanh_fb15k = {
    "embedding_size": 200,
    "l2_reg_lambda": 0.0009,
    "lr": 0.001,
    "batch_size": 5000,
    "max_iter": 55000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_best_DistMult_tanh_fb3m = {
    "embedding_size": 100,
    "l2_reg_lambda": 0.054,
    "lr": 0.0035,
    "batch_size": 5000,
    "max_iter": 60000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_NTN = {
    "embedding_size": 50,
    "k": 2,
    "l2_reg_lambda": hp.qloguniform("l2_reg_lambda", np.log(1e-3), np.log(1e-1), 1e-3),
    "lr": hp.qloguniform("lr", np.log(1e-4), np.log(1e-2), 1e-4),
    "batch_size": 5000,
    "max_iter": 100000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 5000,
}

param_space_best_NTN_wn18 = {
    "embedding_size": 66,
    "k": 2,
    "l2_reg_lambda": 0.0002,
    "lr": 0.001,
    "batch_size": 2000,
    "max_iter": 100000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_best_NTN_fb15k = {
    "embedding_size": 120,
    "k": 2,
    "l2_reg_lambda": 0.0001,
    "lr": 0.001,
    "batch_size": 5000,
    "max_iter": 50000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_Complex = {
    # "embedding_size": hp.quniform("embedding_size", 50, 200, 10),
    "embedding_size": 200,
    "l2_reg_lambda": hp.qloguniform("l2_reg_lambda", np.log(1e-3), np.log(1e-1), 1e-3),
    "lr": hp.qloguniform("lr", np.log(1e-4), np.log(1e-2), 1e-4),
    "batch_size": 5000,
    "max_iter": 100000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 5000,
}

param_space_Complex_fb3m = {
    "embedding_size": 50,
    "l2_reg_lambda": hp.qloguniform("l2_reg_lambda", np.log(1e-3), np.log(1e-1), 1e-3),
    "lr": hp.qloguniform("lr", np.log(1e-4), np.log(1e-2), 1e-4),
    "batch_size": 5000,
    "max_iter": 500000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 20000,
}

param_space_best_Complex_wn18 = {
    "embedding_size": 180,
    "l2_reg_lambda": 0.0073,
    "lr": 0.002,
    "batch_size": 2000,
    "max_iter": 25000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_best_Complex_tanh_fb15k = {
    "embedding_size": 140,
    "l2_reg_lambda": 0.0172,
    "lr": 0.001,
    "batch_size": 5000,
    "max_iter": 500,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_best_Complex_tanh_fb3m = {
    "embedding_size": 50,
    "l2_reg_lambda": 0.031,
    "lr": 0.0013,
    "batch_size": 5000,
    "max_iter": 60000,
    "neg_ratio": 1,
    "contiguous_sampling": False,
    "valid_every": 0,
}

param_space_dict = {
    "TransE_L2": param_space_TransE,
    "TransE_L1": param_space_TransE,
    "TransE_L2_fb3m": param_space_TransE_fb3m,
    "TransE_L1_fb3m": param_space_TransE_fb3m,
    "best_TransE_L2_wn18": param_space_best_TransE_L2_wn18,
    "best_TransE_L1_fb15k": param_space_best_TransE_L1_fb15k,
    "best_TransE_L1_fb3m": param_space_best_TransE_L1_fb3m,
    "DistMult": param_space_DistMult,
    "DistMult_tanh": param_space_DistMult,
    "DistMult_tanh_fb3m": param_space_DistMult_fb3m,
    "best_DistMult_tanh_wn18": param_space_best_DistMult_tanh_wn18,
    "best_DistMult_tanh_fb15k": param_space_best_DistMult_tanh_fb15k,
    "best_DistMult_tanh_fb3m": param_space_best_DistMult_tanh_fb3m,
    "NTN": param_space_NTN,
    "best_NTN_wn18": param_space_best_NTN_wn18,
    "best_NTN_fb15k": param_space_best_NTN_fb15k,
    "Complex": param_space_Complex,
    "Complex_tanh": param_space_Complex,
    "Complex_fb3m": param_space_Complex_fb3m,
    "Complex_tanh_fb3m": param_space_Complex_fb3m,
    "best_Complex_wn18": param_space_best_Complex_wn18,
    "best_Complex_tanh_fb15k": param_space_best_Complex_tanh_fb15k,
    "best_Complex_tanh_fb3m": param_space_best_Complex_tanh_fb3m,
}

int_params = [
    "embedding_size", "batch_size", "max_iter", "neg_ratio", "valid_every", "k",
    "fe_size", "hidden_size", "hidden_layers",
]

class ModelParamSpace:
    def __init__(self, learner_name):
        s = "Invalid model name! (Check model_param_space.py)"
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_into_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k, v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_into_param(v[i])
                elif isinstance(v, dict):
                    self._convert_into_param(v)
        return param_dict
