from ned_with_web_links import EntityNormalization

def main():
    ned = EntityNormalization()
    train_set, eval_set, test_set = ned.read_dataset("/data0/linking/wikipedia/dumps/20150901/", (0.8, 0.1, 0.1))

    model = ned.train(train_set)

    ned.evaluate(model, eval_set)

    ned.predict(model, test_set)


if __name__ == '__main__':
    main()