
import tensorflow as tf
import glob, os

from dataset.utils import load_audio_file, load_labels


def load_datasets(params: dict, dataset_path: str='./dataset', shuffle: bool=True):

    # load audio datasets
    train_data = load_dataset(params, dataset_path, 'train')
    test_data = load_dataset(params, dataset_path, 'test')

    # batch both datasets properly
    train_data = train_data.batch(params['batch_size'])
    test_data = test_data.batch(params['batch_size'])

    # shuffle the training dataset properly
    if shuffle: train_data = train_data.shuffle(5)

    # tune the performance by prefetching several batches in advance
    train_data = train_data.prefetch(5)
    test_data = test_data.prefetch(5)

    return train_data, test_data


def load_dataset(params: dict, dataset_path: str, wildcard: str):

    # read audio sampling configuration settings
    sample_rate = params['sample_rate']
    target_steps = params['seconds_per_sample'] * params['sample_rate']

    # load audio file paths
    audio_files_wildcard = os.path.join(dataset_path, 'audio', f'*{ wildcard }*.wav')
    audio_filepaths = sorted(glob.glob(audio_files_wildcard))

    # load labels from CSV file
    labels_file_wildcard = os.path.join(dataset_path, 'labels', f'*{ wildcard }*.csv')
    labels_filepath = glob.glob(labels_file_wildcard)[0]
    labels_by_file = load_labels(labels_filepath)

    # create a dataset from audio file paths
    dataset = tf.data.Dataset.from_tensor_slices((audio_filepaths))

    # define mapping functions for audo file preprocessing / label retrieval
    tf_get_label = lambda file: labels_by_file[os.path.basename(file.numpy().decode('ascii')[:-4])]
    tf_load_audio_file = lambda file, sr, steps: load_audio_file(file.numpy(), sr, steps)

    # map the sample's audio features and label -> (audio content, label) tuples
    dataset = dataset.map(lambda filepath: (
            tf.py_function(tf_load_audio_file, [filepath, sample_rate, target_steps], [tf.float32]),
            tf.py_function(tf_get_label, [filepath], [tf.int32])
        ),
        num_parallel_calls=params['num_map_threads'],
        deterministic=False
    )
    dataset = dataset.map(lambda x, y: (tf.squeeze(x, axis=0), y))

    return dataset
