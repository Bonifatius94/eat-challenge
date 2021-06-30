
import numpy as np
import tensorflow as tf
import glob, os

from tensorflow.data import TFRecordDataset
from tensorflow.data.experimental import TFRecordWriter
from tensorflow.train import Example, Feature, Features, FloatList, Int64List, BytesList
from tensorflow.io import FixedLenFeature, FixedLenSequenceFeature

from .mapped_dataset import load_dataset as load_mapped_dataset


# ===================================================================================
#                         COMMON CONFIGURATION SETTINGS
# ===================================================================================

# define the name of the newly created tfrecord dataset sub-directory
tfrecord_subdir = 'tfrecord'

# define the feature keys of each data example in the tfrecord file
key_melspectrogram = 'amplitude'
key_label = 'label'
key_sample_id = 'sample_id'


# ===================================================================================
#                         WRITE TFRECORD DATASET TO FILE
# ===================================================================================

def write_tfrecord_dataset(params: dict, dataset_path: str='./dataset', train: bool=True):

    wildcard = 'train' if train else 'test'

    # create the tfrecord directory in dataset_path (if not already existing)
    tfrecord_dir = os.path.join(dataset_path, tfrecord_subdir)
    if not (os.path.exists(tfrecord_dir) and os.path.isdir(tfrecord_dir)):
        os.makedirs(tfrecord_dir)

    # determine the target filepath of the tfrecord file
    sample_rate = params['sample_rate']
    tfrecord_filepath = os.path.join(dataset_path, tfrecord_subdir,
        f'{ wildcard }-amplitude-{ sample_rate }.tfrecord')

    # use the mapped dataset utils to load all features to be exported
    # info: the mapped dataset is preferred as it facilitates CPU scaling
    dataset = load_mapped_dataset(params, dataset_path, train)

    # map the feature dataset using the serialization function
    ser_dataset = dataset.map(lambda x, y: tf_serialize_example(x, y, train))

    # write the contents to the tfrecord file
    writer = TFRecordWriter(tfrecord_filepath)
    writer.write(ser_dataset)


def tf_serialize_example(melspectrogram, label_or_sample_id, train: bool):

    # wrap the serialization function to make it usable within dataset.map()
    tf_ser_train = lambda mel, l: serialize_train_example(mel.numpy(), l.numpy())
    tf_ser_test = lambda mel, s: serialize_test_example(mel.numpy(), s.numpy())
    tf_ser_example_wrapper = lambda mel, l_or_s: \
        tf_ser_train(mel, l_or_s) if train else tf_ser_test(mel, l_or_s)
    tf_string = tf.py_function(tf_ser_example_wrapper, (melspectrogram, label_or_sample_id), tf.string)
    return tf.reshape(tf_string, ())


# create lambdas for tfrecord example feature creation
_float_feature = lambda values: Feature(float_list=FloatList(value=values))
_int64_feature = lambda values: Feature(int64_list=Int64List(value=values))
_bytes_feature = lambda values: Feature(bytes_list=BytesList(value=values))


def serialize_train_example(melspectrogram: np.ndarray, label: int):

    tf.print(f'melspec shape: { melspectrogram.shape }')

    # create an example with the given features
    example_proto = Example(features=Features(feature={
        key_melspectrogram: _float_feature(melspectrogram.flatten()),
        key_label: _int64_feature(label),
    }))

    # serialize the example as string
    return example_proto.SerializeToString()


def serialize_test_example(melspectrogram: np.ndarray, sample_id: str):

    #tf.print(f'melspec shape: { melspectrogram.shape }')

    # create an example with the given features
    example_proto = Example(features=Features(feature={
        key_melspectrogram: _float_feature(melspectrogram.flatten()),
        key_sample_id: _bytes_feature(sample_id),
    }))

    # serialize the example as string
    return example_proto.SerializeToString()


# ===================================================================================
#                         READ TFRECORD DATASET FROM FILE
# ===================================================================================

def load_dataset(params: dict, dataset_path: str='./dataset', train: bool=True):

    wildcard = 'train' if train else 'test'

    # load dataset from tfrecord file
    melspec_shape = params['inputs_shape']
    sample_rate = params['sample_rate']
    tfrecord_file_wildcard = os.path.join(dataset_path, tfrecord_subdir,
        f'*{ wildcard }*amplitude*{ sample_rate }*.tfrecord')
    tfrecord_filepath = glob.glob(tfrecord_file_wildcard)[0]

    # define the tfrecord dataset features to be parsed
    train_feature_description = {
        key_melspectrogram: FixedLenSequenceFeature(shape=melspec_shape, dtype=tf.float32,
                                                    default_value=0.0, allow_missing=True),
        key_label: FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)
    }
    test_feature_description = {
        key_melspectrogram: FixedLenSequenceFeature(shape=melspec_shape, dtype=tf.float32,
                                                    default_value=0.0, allow_missing=True),
        key_sample_id: FixedLenFeature(shape=[], dtype=tf.string, default_value='')
    }
    feature_description = train_feature_description if train else test_feature_description

    # create a tfrecord dataset from file and extract all features
    dataset = TFRecordDataset(tfrecord_filepath)
    dataset = dataset.map(lambda example: tf.io.parse_single_example(example, feature_description))
    dataset = dataset.map(lambda x: (x[key_melspectrogram], x[key_label if train else key_sample_id]))
    dataset = dataset.map(lambda x, y: (tf.squeeze(x, axis=0), y))

    return dataset
