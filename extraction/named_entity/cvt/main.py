import os
import sys
PACKAGE_PARENT = '../../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
print(os.getcwd())
from extraction.named_entity.cvt.cross_view import CrossViewTraining
import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError


def main(input_file_path):
    timeout_val = 180
    files_dict = {'train': input_file_path,
                  'test': input_file_path,
                  'dev': input_file_path}

    my_model = CrossViewTraining()
    data = my_model.read_dataset(file_dict=files_dict)
    test_labels = my_model.convert_ground_truth(data=data['test'])

    with timeout(timeout_val+10):
        with timeout(timeout_val+5):
            with timeout(timeout_val):
                my_model.train(data=data)

    pred_vals = my_model.predict(data=data['test'])
    my_model.evaluate(predictions=pred_vals, ground_truths=test_labels)
    output_file_path = my_model.data_dir+"/output.txt"

    return output_file_path


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("supply input file name as cmd arg")
