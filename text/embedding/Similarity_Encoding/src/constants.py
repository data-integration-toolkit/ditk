from keras.metrics import binary_crossentropy, mean_squared_error
import re

sample_seed = 5
clf_seed = 42
shuffle_seed = 32
common_network_shape = (100, 100, 1)
batch_size = 32
val_split = 0.05
epochs = 20
default_encoding_depth = 10
rescaling_factor = 1 / 2
dropout = None
LOSS_MAP = {'binary_clf': binary_crossentropy,
            'multiclass_clf': binary_crossentropy,
            'regression': mean_squared_error}
tf_allowed_scope = re.compile('[A-Za-z0-9_.\\-/]*')
