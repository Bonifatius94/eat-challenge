
import tensorflow as tf
import numpy as np


def partition_dataset(batched_dataset: tf.data.Dataset, splits: list, batch_size: int):

    # make sure the splits are valid
    if np.sum(np.array(splits)) != 1.0:
        raise ValueError('Invalid dataset splits! The sum of all splits needs to be 1!')

    # determine the amount of samples on the dataset
    num_samples = tf.data.experimental.cardinality(batched_dataset).numpy() # TODO: check if this works

    # determine the amount of train / eval / test batches (test batches = rest)
    num_batches = int(np.ceil(num_samples / batch_size))
    num_train_batches = int(num_batches * splits[0])
    num_eval_batches = int(num_batches * splits[1])

    # partition the train / eval / test datasets
    train_data = batched_dataset.take(num_train_batches)
    eval_data = batched_dataset.skip(num_train_batches).take(num_eval_batches)
    test_data = batched_dataset.skip(num_train_batches + num_eval_batches)

    return train_data, eval_data, test_data