from ProjE import ProjE
import tensorflow as tf

def main(input_file):

    model = ProjE(embed_dim=200, combination_method='simple',
                   dropout=0.5, neg_weight=0.5)

    args, trainList, validList, testList = model.read_dataset(input_file)

    model.train_hrt_input, model.train_hrt_weight, model.train_trh_input, model.train_trh_weight, model.train_loss, model.train_op, model.ent_embeddings, model.rel_embeddings = model.learn_embeddings(data = args['data_dir'], argDict = args)
    model.test_input, model.test_head, model.test_tail = model.test_ops()
    load_dir = ""
    #load_dir = model.load_model("./trainFiles/ProjE_DEFAULT_0.ckpt")

    model.evaluate(data = args['data_dir'], args = args, load_dir = load_dir)
    return model.ent_embeddings, model.rel_embeddings

if __name__ == '__main__':
    main(input_file)
    tf.app.run()
