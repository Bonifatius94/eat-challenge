
import sys
from train import NaiveTrainingSession
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
    # TODO: add more tasks here ...


def preprocess_dataset():

    # define hyper-params required for preprocessing the dataset
    params =  {
        'seconds_per_sample': 4,
        'sample_rate': 16000,
        'num_map_threads': 3,
    }

    # write a tfrecord cache file for each dataset (train / test)
    write_tfrecord_datasets(params)


def run_naive_training():

    # define hyper-params required for creating the training session
    params = {
        # training hparams
        'num_epochs': 20,
        'batch_size': 32,

        # dataset hparams
        'seconds_per_sample': 4,
        'sample_rate': 16000,
        'num_map_threads': 3,
        'num_classes': 32,
    }

    # load a dataset for testing purposes
    session = NaiveTrainingSession(params)
    session.run_training()


if __name__ == '__main__':
    main()
