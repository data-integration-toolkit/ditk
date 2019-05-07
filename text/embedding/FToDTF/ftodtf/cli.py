""" This module handles parsing of cli-flags and then calls the needed function
from the library"""
import sys
import argparse
from multiprocessing import Process
from multiprocessing.queues import Empty

from tqdm import tqdm

import ftodtf.model
import ftodtf.training
import ftodtf.input
import ftodtf.inference
import ftodtf.export
from ftodtf.settings import FasttextSettings


PREPROCESS_REQUIRED_PARAMETERS = ["corpus_path"]
TRAIN_REQUIRED_PARAMETERS = []
SETTINGS = FasttextSettings()
PARSER = argparse.ArgumentParser(
    description="Unsupervised training of word-vector-embeddings.")

SUBPARSER = PARSER.add_subparsers(dest="command")
PREPROCESS_PARSER = SUBPARSER.add_parser(
    "preprocess", help="Convert raw text to training-data for use in the train step.")
TRAIN_PARSER = SUBPARSER.add_parser(
    "train", help="Train word-embeddings using previously created training-data.")

INFER_PARSER = SUBPARSER.add_parser("infer", help="Use trained embeddings.")
INFER_SUBPARSER = INFER_PARSER.add_subparsers(dest="subcommand")
INFER_SIMILARITIES = INFER_SUBPARSER.add_parser(
    "similarities", help="Compute the similarities between given words.")

EXPORT_PARSER = SUBPARSER.add_parser(
    "export", help="Export trained embeddings.")
EXPORT_SUBPARSER = EXPORT_PARSER.add_subparsers(dest="subcommand")
EXPORT_EMBEDDINGS = EXPORT_SUBPARSER.add_parser(
    "embeddings", help="Export the embeddings in tensorflows checkpoint-format. Can be used for fasttext infer and is smaller than regular training-chekpoints")


def add_arguments_to_parser(arglist, parser, required, group=None):
    """ Adds arguments (obtained from the settings-class) to an agrparse-parser

    :param list(str) arglist: A list of strings representing the names of the flags to add
    :param argparse.ArgumentParser parser: The parser to add the arguments to
    :param list(str) required: A list of argument-names that are required for the command
    :param str group: If set place the arguments in an argument-group of the specified name
    """
    if group:
        parser = parser.add_argument_group(group)
    for parameter, default in filter(lambda x: x[0] in arglist, vars(SETTINGS).items()):
        parser.add_argument("--"+parameter, type=type(default),
                            help=SETTINGS.attribute_docstring(parameter),
                            required=parameter in required, default=default)


add_arguments_to_parser(SETTINGS.preprocessing_settings(),
                        PREPROCESS_PARSER,
                        PREPROCESS_REQUIRED_PARAMETERS)

add_arguments_to_parser(SETTINGS.training_settings(),
                        TRAIN_PARSER,
                        TRAIN_REQUIRED_PARAMETERS)

add_arguments_to_parser(SETTINGS.distribution_settings(),
                        TRAIN_PARSER,
                        [],
                        "Distribution settings")

add_arguments_to_parser(SETTINGS.inference_settings(),
                        INFER_PARSER,
                        [])

INFER_SIMILARITIES.add_argument("words", type=str, nargs="+",
                                help=" A list of words of which the similarities to each other should be computed.")

# Export uses the same parameters as settigns, because both build an InferenceModel (and therefore need exactly the same settings).
add_arguments_to_parser(SETTINGS.inference_settings(),
                        EXPORT_PARSER,
                        [])

EXPORT_PARSER.add_argument("-outdir", type=str, default="./export",
                           help="The directory to store the exports in. Default ./export")


def spawn_progress_bar():
    """ This function will spawn a new process using multiprocessing module.

    :return: A child process.
    """
    p = Process(target=show_prog, args=(ftodtf.input.QUEUE, ))
    p.daemon = True
    return p


def show_prog(q):
    """ Show progressbar, converges against the next max progress_bar.n and
    finishes only when the function "write_batches_to_file" ends.

    :param q: Process which handles the progressbar.
    """
    proggress_bar = tqdm(total=100, desc="Segmen./Cleaning",
                         bar_format='{desc}:{percentage:3.0f}%|{bar}[{elapsed}]')
    n = 40
    j = 1
    while True:
        try:
            finished_function = q.get(timeout=1)
            if finished_function == "_process_text":
                proggress_bar.n = 66
                n, j = 10, 1
                proggress_bar.desc = "Writing Batches"
            elif finished_function == "write_batches_to_file":
                proggress_bar.n = 100
                proggress_bar.close()
                return 0
        except (TimeoutError, Empty):
            if n <= 0:
                j *= 10
                n = j
            proggress_bar.update(1/j)
            n -= 1
            continue


def cli_main():
    """ Program entry point. """
    flags, unknown = PARSER.parse_known_args()
    if unknown:
        print(
            "Unknown flag '{}'. Run --help for a list of all possible "
            "flags".format(unknown[0]))
        sys.exit(1)
    # copy specified arguments over to the SETTINGS object
    for k, v in vars(flags).items():
        SETTINGS.__dict__[k] = v

    if flags.command == "preprocess":
        try:
            SETTINGS.validate_preprocess()
        except Exception as e:
            print(": ".join(["ERROR", e.__str__()]))
            sys.exit(1)
        else:
            p = spawn_progress_bar()
            p.start()
            ipp = ftodtf.input.InputProcessor(SETTINGS)
            ipp.preprocess()
            try:
                ftodtf.input.write_batches_to_file(ipp.batches(),
                                                   SETTINGS.batches_file,
                                                   SETTINGS.num_batch_files)
            except Warning as w:
                # write_batches_to_file will raise a warning if there is not enough input-data
                print(w)
                sys.exit(1)
            p.join()
    elif flags.command == "train":
        try:
            SETTINGS.validate_train()
        except Exception as e:
            print(": ".join(["ERROR", e.__str__()]))
            sys.exit(1)
        else:
            ftodtf.training.train(SETTINGS)
    elif flags.command == "infer" and flags.subcommand == "similarities":
        ftodtf.inference.compute_similarities(flags.words, SETTINGS)
    elif flags.command == "export" and flags.subcommand == "embeddings":
        ftodtf.export.export_embeddings(SETTINGS, flags.outdir)
    else:
        PARSER.print_help()


if __name__ == "__main__":
    try:
        cli_main()
    except KeyboardInterrupt as e:

        # Kill all subprocess
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            child.kill()

        print("Program interrupted!")
