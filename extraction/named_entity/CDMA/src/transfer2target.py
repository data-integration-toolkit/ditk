from .model.data_utils import NERDataset
from .model.sal_blstm_oal_crf_model import SAL_BLSTM_OAL_CRF_Model
from .model.config import Config
import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def get_tensors_in_checkpoint_file(file_name,all_tensors=True,tensor_name=None):
    varlist=[]
    var_value =[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            varlist.append(key)
            var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)


def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = list()
    # Loop all loaded tensors
    for i, tensor_name in enumerate(loaded_tensors[0]):
        # Extract tensor
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
        except:
            print('Not found: '+tensor_name)
        full_var_list.append(tensor_aux)
    return full_var_list
 

def main():
    # create instance of config
    config = Config()
    config.filename_train = "../datasets/ritter2011/train"
    config.filename_dev = "../datasets/ritter2011/dev"
    config.filename_test = "../datasets/ritter2011/test"

    config.filename_chars = config.filename_chars.replace("source", "target")
    config.filename_glove = config.filename_glove.replace("source", "target")
    config.filename_tags = config.filename_tags.replace("source", "target")
    config.filename_words = config.filename_words.replace("source", "target")

    config.dir_model = config.dir_model.replace("source", "target")
    config.dir_output = config.dir_output.replace("source", "target")
    config.path_log = config.path_log.replace("source", "target")


    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids[0])
    # build model
    model = SAL_BLSTM_OAL_CRF_Model(config)
    model.build()
    CHECKPOINT_NAME = "source_model/model_weights"
    restored_vars  = get_tensors_in_checkpoint_file(file_name=CHECKPOINT_NAME)
    tensors_to_load = build_tensors_in_checkpoint_file(restored_vars)
    model.saver = tf.train.Saver(tensors_to_load)
    model.restore_session("source_model/")
    model.reinitialize_weights("proj")

    # create datasets
    train = NERDataset(config.filename_train, config.processing_word,
                       config.processing_tag, config.max_iter)

    dev   = NERDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
