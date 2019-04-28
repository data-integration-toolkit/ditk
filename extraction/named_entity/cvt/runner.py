from extraction.named_entity.cvt.cross_view import CrossViewTraining
import signal
from contextlib import contextmanager

timeout_val = 180
files_dict = {'train': './our_data_format/ner_test_input.txt',
              'test': './our_data_format/ner_test_input.txt',
              'dev': './our_data_format/ner_test_input.txt'}


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


my_model = CrossViewTraining()
data = my_model.read_dataset(file_dict=files_dict)
test_labels = my_model.convert_ground_truth(data=data['test'])

with timeout(timeout_val+10):
    with timeout(timeout_val+5):
        with timeout(timeout_val):
            my_model.train()
my_model.predict()
my_model.evaluate()
