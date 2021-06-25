
import tensorflow as tf
import glob, os

from tensorflow.data import TFRecordDataset
from tensorflow.data.experimental import TFRecordWriter
from tensorflow.train import Example, Feature, Features, FloatList, Int64List
from tensorflow.io import FixedLenFeature, FixedLenSequenceFeature

from .mapped_dataset import load_dataset as load_mapped_dataset


# create lambdas for tfrecord example feature creation
_float_feature = lambda values: Feature(float_list=FloatList(value=values))
_int64_feature = lambda values: Feature(int64_list=Int64List(value=values))

# define the name of the newly created tfrecord dataset sub-directory
tfrecord_subdir = 'tfrecord'

# define the feature keys of each data example in the tfrecord file
key_melspectrogram = 'melspectrogram'
key_label = 'label'


def write_tfrecord_datasets(params: dict, dataset_path: str='./dataset'):

    # create the tfrecord directory in dataset_path (if not already existing)
    tfrecord_dir = os.path.join(dataset_path, tfrecord_subdir)
    if not (os.path.exists(tfrecord_dir) and os.path.isdir(tfrecord_dir)):
        os.makedirs(tfrecord_dir)

    # create tfrecord files for both training and test data
    write_tfrecord_dataset(params, dataset_path, 'train')
    write_tfrecord_dataset(params, dataset_path, 'test')


def write_tfrecord_dataset(params: dict, dataset_path: str, wildcard: str):

    # determine the target filepath of the tfrecord file
    tfrecord_filepath = os.path.join(dataset_path, tfrecord_subdir, f'{ wildcard }.tfrecord')

    # use the mapped dataset utils to load all features to be exported
    # info: the mapped dataset is preferred as it facilitates CPU scaling
    dataset = load_mapped_dataset(params, dataset_path, wildcard)

    # map the feature dataset using the serialization function
    ser_dataset = dataset.map(tf_serialize_example)

    # write the contents to the tfrecord file
    writer = TFRecordWriter(tfrecord_filepath)
    writer.write(ser_dataset)
    # TODO: split the dataset in multiple shards (~100MB)


def tf_serialize_example(melspectrogram, label):

    # wrap the serialization function to make it usable within dataset.map()
    tf_string = tf.py_function(serialize_example, (melspectrogram, label), tf.string)
    return tf.reshape(tf_string, ())


def serialize_example(melspectrogram, label):

    # serialize the example's features
    example_proto = Example(features=Features(feature={
        key_melspectrogram: _float_feature(melspectrogram.numpy().flatten()),
        key_label: _int64_feature(label.numpy()),
    }))
    return example_proto.SerializeToString()


# ===================================================================================


def load_datasets(params: dict, dataset_path: str='./dataset', shuffle: bool=True):

    # load audio datasets
    train_data = load_dataset(dataset_path, 'train')
    test_data = load_dataset(dataset_path, 'test')

    # batch both datasets properly
    train_data = train_data.batch(params['batch_size'])
    test_data = test_data.batch(params['batch_size'])

    # shuffle the training dataset properly
    if shuffle: train_data = train_data.shuffle(20)

    # tune the performance by prefetching several batches in advance
    train_data = train_data.prefetch(5)
    test_data = test_data.prefetch(5)

    return train_data, test_data


def load_dataset(dataset_path: str, wildcard: str):

    # load dataset from tfrecord file
    tfrecord_file_wildcard = os.path.join(dataset_path, tfrecord_subdir, f'*{ wildcard }*.tfrecord')
    tfrecord_filepath = glob.glob(tfrecord_file_wildcard)[0]

    # define tfrecord dataset features to be parsed
    feature_description = {
        key_melspectrogram: FixedLenSequenceFeature(shape=(128, 126, 1), dtype=tf.float32,
                                                    default_value=0.0, allow_missing=True),
        key_label: FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)
    }

    # create a tfrecord dataset from file and extract features
    dataset = TFRecordDataset(tfrecord_filepath)
    dataset = dataset.map(lambda example: tf.io.parse_single_example(example, feature_description))
    dataset = dataset.map(lambda x: (x[key_melspectrogram], x[key_label]))
    dataset = dataset.map(lambda x, y: (tf.squeeze(x, axis=0), y))

    return dataset
