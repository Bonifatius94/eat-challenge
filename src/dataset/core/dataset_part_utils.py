
import tensorflow as tf
import numpy as np


def partition_dataset(dataset: tf.data.Dataset, splits: list):

    # make sure the splits are valid
    if np.sum(np.array(splits)) != 1.0:
        raise ValueError('Invalid dataset splits! The sum of all splits needs to be 1!')

    # determine the amount of batches on the dataset
    num_samples = int(np.sum([1 for _ in iter(dataset)]))

    # determine the amount of train / eval / test batches (test batches = rest)
    num_train_batches = int(num_samples * splits[0])
    num_eval_batches = int(num_samples * splits[1])

    # partition the train / eval / test datasets
    train_data = dataset.take(num_train_batches)
    eval_data = dataset.skip(num_train_batches).take(num_eval_batches)
    test_data = dataset.skip(num_train_batches + num_eval_batches)

    return train_data, eval_data, test_data