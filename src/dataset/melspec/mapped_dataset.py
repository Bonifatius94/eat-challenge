
import tensorflow as tf
import glob, os

from dataset.utils import load_audio_file, load_labels
from dataset.core import load_labeled_audio_filepaths_dataset


mode2wildcard = lambda train: 'train' if train else 'test'


def load_dataset(params: dict, dataset_path: str='./dataset', train: bool=True):

    # TODO: refactor this dataset into following steps:
    #       1) load audio file paths (with labels / sample ids)
    #       2) partition the file paths into train / eval / test splits
    #       3) map the audio file paths to load audio features (-> can be one-to-many relation)
    #       4) flatten the one-to-many relations for each data split

    # create a dataset from audio file paths (filepath, label_or_sample_id) tuples
    dataset = load_labeled_audio_filepaths_dataset(dataset_path, train)

    # define the function for loading and preprocessing audio files
    file2audio = audio_preprocessing_func(params)

    # map the audio file paths to preprocessed audio features
    # info: there can be multiple audio shards per file (-> one-to-many)
    dataset = dataset.map(
        lambda file, l_or_sid: (file2audio(file), l_or_sid),
        num_parallel_calls=params['num_map_threads'], # scale with multi-core
        deterministic=train # ensure deterministic datasets for training
                            # -> train / eval / test splits are always the same
    )

    # TODO: add a flat_map operation to handle one-to-many audio shards

    # squeeze the batch dimension (batching is handled in calling function)
    dataset = dataset.map(lambda x, y: (tf.squeeze(x, axis=0), y))

    return dataset


def audio_preprocessing_func(params: dict):

    # read audio sampling configuration settings
    sample_rate = params['sample_rate']
    target_steps = params['seconds_per_sample'] * params['sample_rate']

    # define a mapping functions for audio proprocessing
    tf_load_audio_file = lambda file, sr, steps: load_audio_file(file.numpy(), sr, steps)
    pyfunc_load_audio = lambda file: \
        tf.py_function(tf_load_audio_file, [file, sample_rate, target_steps], [tf.float32])

    return pyfunc_load_audio
