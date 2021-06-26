
import tensorflow as tf
import glob, os

from dataset.utils import load_audio_file, load_labels


def load_dataset(params: dict, dataset_path: str='./dataset', train: bool=True):

    wildcard = 'train' if train else 'test'

    # read audio sampling configuration settings
    sample_rate = params['sample_rate']
    target_steps = params['seconds_per_sample'] * params['sample_rate']

    # load audio file paths
    audio_files_wildcard = os.path.join(dataset_path, 'audio', f'*{ wildcard }*.wav')
    audio_filepaths = sorted(glob.glob(audio_files_wildcard))

    # prepare funcs for sample id extraction from audio file paths
    get_sample_id = lambda file: os.path.basename(file[:-4])
    tf_get_sample_id = lambda file: get_sample_id(file.numpy().decode('ascii'))

    if train:
        # load labels from CSV file
        labels_file_wildcard = os.path.join(dataset_path, 'labels', f'*{ wildcard }*.csv')
        labels_filepath = glob.glob(labels_file_wildcard)[0]
        labels_by_sample_id = load_labels(labels_filepath)

        # define a mapping function for label retrieval
        tf_get_label = lambda file: labels_by_sample_id[tf_get_sample_id(file)]

    # create a dataset from audio file paths
    dataset = tf.data.Dataset.from_tensor_slices((audio_filepaths))

    # define a mapping functions for audio file preprocessing and label / sample id retrieval
    tf_load_audio_file = lambda file, sr, steps: load_audio_file(file.numpy(), sr, steps)
    pyfunc_load_audio = lambda file: \
        tf.py_function(tf_load_audio_file, [file, sample_rate, target_steps], [tf.float32])
    pyfunc_load_label_or_sample_id = lambda file: \
        tf.py_function(tf_get_label, [file], [tf.int32]) if train \
            else tf.py_function(tf_get_sample_id, [file], [tf.string])

    # map the sample's audio features and label / sample id
    dataset = dataset.map(lambda filepath: (
            pyfunc_load_audio(filepath),
            pyfunc_load_label_or_sample_id(filepath)
        ),
        num_parallel_calls=params['num_map_threads'],
        deterministic=train # ensure deterministic datasets for training
                            # -> train / eval / test splits are always to same
    )

    # squeeze the batch dimension (batching is handled in calling function)
    dataset = dataset.map(lambda x, y: (tf.squeeze(x, axis=0), y))

    return dataset
