
import numpy as np
import tensorflow as tf
import glob, os

from dataset.utils import load_audio_file, load_labels


def load_datasets(params: dict, dataset_path: str='./dataset', shuffle: bool=True):

    # TODO: implement train / eval / test splits

    # load audio datasets
    train_data = load_dataset(params, dataset_path, 'train')
    #test_data = load_dataset(params, dataset_path, 'test')

    # batch both datasets properly
    train_data = train_data.batch(params['batch_size'])
    #test_data = test_data.batch(params['batch_size'])

    # shuffle the training dataset properly
    if shuffle: train_data = train_data.shuffle(5)

    # tune the performance by prefetching several batches in advance
    train_data = train_data.prefetch(5)
    #test_data = test_data.prefetch(5)

    return train_data#, test_data


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

    # cache the melspectrograms and labels as numpy arrays
    get_label_func = lambda file: labels_by_file[os.path.basename(file[:-4])]
    labels = np.array([get_label_func(file) for file in audio_filepaths])
    spectrograms = np.array([load_audio_file(file, sample_rate, target_steps) for file in audio_filepaths])

    # create a mapped dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, labels))

    return dataset
