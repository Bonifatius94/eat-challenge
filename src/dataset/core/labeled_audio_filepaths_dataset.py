
import tensorflow as tf
import glob, os

from dataset.utils import load_labels
from dataset.core import load_audio_filepaths_dataset


mode2wildcard = lambda train: 'train' if train else 'test'


def load_dataset(dataset_path: str='./dataset', train: bool=True):

    # create a dataset from audio file paths
    dataset = load_audio_filepaths_dataset(dataset_path, train)

    # map the sample's audio features and label / sample id
    filepath_to_label_or_sample_id = label_or_sampleid_func(dataset_path, train)
    dataset = dataset.map(lambda path: (path, filepath_to_label_or_sample_id(path)))

    return dataset


def label_or_sampleid_func(dataset_path, train):

    # prepare funcs for sample id extraction from audio file paths
    get_sample_id = lambda file: os.path.basename(file[:-4])
    tf_get_sample_id = lambda file: get_sample_id(file.numpy().decode('ascii'))

    # define a mapping function for label retrieval
    labels_by_sample_id = \
        load_labels_by_sample_id(dataset_path, train) if train else None
    tf_get_label = lambda file: labels_by_sample_id[tf_get_sample_id(file)]

    # define a mapping functions for label / sample id retrieval
    pyfunc_load_label_or_sample_id = lambda file: \
        tf.py_function(tf_get_label, [file], [tf.int32]) if train \
            else tf.py_function(tf_get_sample_id, [file], [tf.string])

    return pyfunc_load_label_or_sample_id


def load_labels_by_sample_id(dataset_path, train):

    # load labels from CSV file
    wildcard = mode2wildcard(train)
    labels_file_wildcard = os.path.join(dataset_path, 'labels', f'*{ wildcard }*.csv')
    labels_filepath = glob.glob(labels_file_wildcard)[0]
    labels_by_sample_id = load_labels(labels_filepath)

    return labels_by_sample_id