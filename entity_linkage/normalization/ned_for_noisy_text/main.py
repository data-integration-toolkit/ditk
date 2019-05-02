from ned_for_noisy_text import EntityNormalization


def main():
    ned = EntityNormalization()

    train_config = "experiments/CoNLL/experiment.conf"

    model = ned.train(train_config)

    ned.evaluate(model, "")

    train_config = "experiments/CoNLL/experiment.conf"

    model = ned.train(train_config)

    ned.evaluate(model, "")


if __name__ == '__main__':
    main()
