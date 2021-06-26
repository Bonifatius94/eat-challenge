
import sys
from train import NaiveTrainingSession, NoisyTrainingSession
from dataset import write_tfrecord_datasets


def main():

    # make sure there is a task specified as first script arg
    if len(sys.argv) <= 1:
        raise ValueError('Invalid script arguments! Requires the task to be executed as first argument!')

    # parse the task from script args
    task = sys.argv[1]

    # run the task
    if task == 'preprocess_dataset': preprocess_dataset()
    if task == 'naive_training': run_naive_training()
    if task == 'noisy_training': run_noisy_training()
    # TODO: add more tasks here ...


def preprocess_dataset():

    # define hyper-params required for preprocessing the dataset
    params = {
        'seconds_per_sample': 4,
        'sample_rate': 48000,
        'num_map_threads': 16,
    }

    # write a tfrecord cache file for each dataset (train / test)
    write_tfrecord_datasets(params)


def run_naive_training():

    # define hyper-params required for creating the training session
    params = {
        # training hparams
        'skip_training': False,
        'num_epochs': 10,
        'batch_size': 16,  # batch size seems to be very volatile

        # dataset hparams
        'num_train_samples': 945,
        'num_test_samples': 469,
        'seconds_per_sample': 4,
        # 'sample_rate': 48000,
        'sample_rate': 16000,
        'num_map_threads': 3,
        'num_classes': 7,
        # 'melspec_shape': (128, 376, 1),
        'melspec_shape': (128, 126, 1),
    }

    # load a dataset for testing purposes
    session = NaiveTrainingSession(params)
    session.run_training()



def run_noisy_training():

    # define hyper-params required for creating the training session
    params = {
        # training hparams
        'skip_training': False,
        'num_epochs': 10,
        'batch_size': 16,  # batch size seems to be very volatile

        # dataset hparams
        'num_train_samples': 945,
        'num_test_samples': 469,
        'seconds_per_sample': 4,
        # 'sample_rate': 48000,
        'sample_rate': 16000,
        'num_map_threads': 3,
        'num_classes': 7,
        # 'melspec_shape': (128, 376, 1),
        'melspec_shape': (128, 126, 1),
    }

    # load a dataset for testing purposes
    session = NoisyTrainingSession(params)
    session.run_training()


if __name__ == '__main__':
    main()
