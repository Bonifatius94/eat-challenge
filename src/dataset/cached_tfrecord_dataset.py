
import numpy as np
import librosa
import tensorflow as tf
import glob, os
import pandas as pd

from .mapped_dataset import load_dataset as load_mapped_dataset
from tensorflow.data import TFRecordDataset
from tensorflow.data.experimental import TFRecordWriter
from tensorflow.train import Example, Feature, Features, FloatList, Int64List
from tensorflow.io import FixedLenFeature, FixedLenSequenceFeature


# create lambdas for tfrecord example feature creation
_float_feature = lambda values: Feature(float_list=FloatList(value=values))
_int64_feature = lambda values: Feature(int64_list=Int64List(value=values))


def write_tfrecord_datasets(params: dict, dataset_path: str='./dataset'):

    # create the tfrecord directory in dataset_path (if not already existing)
    tfrecord_dir = './dataset/tfrecord'
    if not (os.path.exists(tfrecord_dir) and os.path.isdir(tfrecord_dir)):
        os.makedirs(tfrecord_dir)

    # create tfrecord files for both training and test data
    write_tfrecord_dataset(params, dataset_path, 'train')
    write_tfrecord_dataset(params, dataset_path, 'test')


def write_tfrecord_dataset(params: dict, dataset_path: str, wildcard: str):

    # load the numpy dataset containing the features to be exported
    feature_dataset = load_mapped_dataset(params, dataset_path, wildcard)

    # map the feature dataset using the serialization function
    serialized_features_dataset = feature_dataset.map(tf_serialize_example)

    # determine the target filepath of the tfrecord file
    tfrecord_filepath = os.path.join(dataset_path, 'tfrecord', f'{ wildcard }.tfrecord')

    # write the contents to the tfrecord file
    writer = TFRecordWriter(tfrecord_filepath)
    writer.write(serialized_features_dataset)
    # TODO: split the dataset in multiple shards (~100MB)


def tf_serialize_example(melspectrogram, label):

    # wrap the serialization function to make it usable within dataset.map()
    tf_string = tf.py_function(serialize_example, (melspectrogram, label), tf.string)
    return tf.reshape(tf_string, ())


def serialize_example(melspectrogram, label):

    # serialize the example's features
    example_proto = Example(features=Features(feature={
        'melspectrogram': _float_feature(melspectrogram.numpy().flatten()),
        'label': _int64_feature(label.numpy()),
    }))
    return example_proto.SerializeToString()


# ===================================================================================


def load_datasets(params: dict, dataset_path: str='./dataset'):

    # load audio datasets
    train_data = load_dataset(dataset_path, 'train')
    test_data = load_dataset(dataset_path, 'test')

    # batch both datasets properly
    train_data = train_data.batch(params['batch_size'])
    test_data = test_data.batch(params['batch_size'])

    # shuffle the training dataset properly
    #train_data = train_data.shuffle(5)

    # tune the performance by prefetching several batches in advance
    train_data = train_data.prefetch(5)
    test_data = test_data.prefetch(5)

    return train_data, test_data


def load_dataset(dataset_path: str, wildcard: str):

    # load dataset from tfrecord file
    labels_file_wildcard = os.path.join(dataset_path, 'tfrecord', f'*{ wildcard }*.tfrecord')
    labels_filepath = glob.glob(labels_file_wildcard)[0]

    # define tfrecord dataset features to be parsed
    feature_description = {
        'melspectrogram': tf.io.FixedLenSequenceFeature(shape=(128, 126, 1), dtype=tf.float32,
                                                        default_value=0.0, allow_missing=True),
        'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)
    }

    # create a tfrecord dataset from file and extract features
    dataset = TFRecordDataset(labels_filepath)
    dataset = dataset.map(lambda example: tf.io.parse_single_example(example, feature_description))
    dataset = dataset.map(lambda x: (x['melspectrogram'], x['label']))
    dataset = dataset.map(lambda x, y: (tf.squeeze(x, axis=0), y))

    return dataset





