
import numpy as np
import tensorflow as tf
import glob, os

from tensorflow.data import TFRecordDataset
from tensorflow.data.experimental import TFRecordWriter
from tensorflow.train import Example, Feature, Features, FloatList, Int64List, BytesList
from tensorflow.io import FixedLenFeature, FixedLenSequenceFeature

from . import MappedMelspecDatasetLoader

# ===================================================================================
#                         COMMON CONFIGURATION SETTINGS
# ===================================================================================

# define the name of the newly created tfrecord dataset sub-directory
tfrecord_subdir = 'tfrecord'

# define the feature keys of each data example in the tfrecord file
key_melspectrogram = 'melspectrogram'
key_label = 'label'
key_sample_id = 'sample_id'

# create lambdas for tfrecord example feature creation
_float_feature = lambda values: Feature(float_list=FloatList(value=values))
_int64_feature = lambda values: Feature(int64_list=Int64List(value=[values]))
_bytes_feature = lambda values: Feature(bytes_list=BytesList(value=[values]))


class CachedTfrecordMelspecDatasetLoader:

    def __init__(self, params: dict):
        self.params = params

    # ===================================================================================
    #                         WRITE TFRECORD DATASET TO FILE
    # ===================================================================================

    def write_tfrecord_dataset(self, dataset_path: str='./dataset', train: bool=True):
        # create the tfrecord directory in dataset_path (if not already existing)
        tfrecord_dir = os.path.join(dataset_path, tfrecord_subdir)
        if not (os.path.exists(tfrecord_dir) and os.path.isdir(tfrecord_dir)):
            os.makedirs(tfrecord_dir)

        # use the mapped dataset utils to load all features to be exported
        ds_loader = MappedMelspecDatasetLoader(self.params)
        if train:
            train_ds = ds_loader.load_dataset(train, speaker_ids=self.params['train_speaker_ids'])
            test_ds = ds_loader.load_dataset(train, speaker_ids=self.params['test_speaker_ids'])
            self._write_dataset_to_file(self._get_tfrecord_filepath('train'), train_ds, train)
            self._write_dataset_to_file(self._get_tfrecord_filepath('test'), test_ds, train)
        else:
            dataset = ds_loader.load_dataset(train)
            self._write_dataset_to_file(self._get_tfrecord_filepath('eval'), dataset, train)

    def _get_tfrecord_filepath(self, wildcard: str, dataset_path: str='./dataset'):
        sample_rate = self.params['sample_rate']
        return os.path.join(dataset_path, tfrecord_subdir,
            f'{ wildcard }-melspec-{ sample_rate }.tfrecord')

    def _write_dataset_to_file(self, filepath: str, dataset: tf.data.Dataset, train: bool):
        ser_dataset = dataset.map(lambda x, y: self._tf_serialize_example(x, y, train))
        writer = TFRecordWriter(filepath)
        writer.write(ser_dataset)

    def _tf_serialize_example(self, melspectrogram, label_or_sample_id, train: bool):
        tf_ser_train = lambda mel, l: self._serialize_train_example(mel.numpy(), l.numpy())
        tf_ser_test = lambda mel, s: self._serialize_test_example(mel.numpy(), s.numpy())
        tf_ser_example_wrapper = lambda mel, l_or_s: \
            tf_ser_train(mel, l_or_s) if train else tf_ser_test(mel, l_or_s)
        tf_string = tf.py_function(tf_ser_example_wrapper, (melspectrogram, label_or_sample_id), tf.string)
        return tf.reshape(tf_string, ())

    def _serialize_train_example(self, melspectrogram: np.ndarray, label: int):
        return Example(features=Features(feature={
                key_melspectrogram: _float_feature(melspectrogram.flatten()),
                key_label: _int64_feature(label),
            })).SerializeToString()

    def _serialize_test_example(self, melspectrogram: np.ndarray, sample_id: str):
        return Example(features=Features(feature={
                key_melspectrogram: _float_feature(melspectrogram.flatten()),
                key_sample_id: _bytes_feature(sample_id),
            })).SerializeToString()


    # ===================================================================================
    #                         READ TFRECORD DATASET FROM FILE
    # ===================================================================================

    def load_dataset(self, dataset_path: str='./dataset', train: bool=True):
        if train:
            print('loading train set')
            train_ds = self._load_dataset(ds_wildcard='train', train=train)
            print('loading test set')
            test_ds = self._load_dataset(ds_wildcard='test', train=train)
            return (train_ds, test_ds)
        return self._load_dataset(ds_wildcard='eval', train=train)

    def _load_dataset(self, ds_wildcard: str, dataset_path: str='./dataset', train: bool=True):
        # load dataset from tfrecord file
        melspec_shape = self.params['inputs_shape']
        sample_rate = self.params['sample_rate']
        tfrecord_file_wildcard = os.path.join(dataset_path, tfrecord_subdir,
            f'*{ ds_wildcard }*melspec*{ sample_rate }*.tfrecord')
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
