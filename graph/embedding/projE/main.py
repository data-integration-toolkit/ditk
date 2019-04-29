from ProjE import ProjE
import tensorflow as tf

def main(_):

    model = ProjE(embed_dim=5, combination_method='simple',
                   dropout=0.5, neg_weight=0.5)

    args = model.read_dataset('./yago_c/')

    model.train_hrt_input, model.train_hrt_weight, model.train_trh_input, model.train_trh_weight, \
    model.train_loss, model.train_op = model.learn_embeddings(data = args['data_dir'], argDict = args)
    print("training done")
    model.test_input, model.test_head, model.test_tail = model.test_ops()
    print("testing done")
    load_dir = ""
    load_dir = model.load_model("./trainFiles/ProjE_DEFAULT_0.ckpt")

    model.evaluate(data = args['data_dir'], args = args, load_dir = load_dir)

if __name__ == '__main__':
    tf.app.run()
