
import tensorflow as tf
import glob, os


def load_dataset(dataset_path: str='./dataset', train: bool=True):

    wildcard = 'train' if train else 'test'

    # load audio file paths
    audio_files_wildcard = os.path.join(dataset_path, 'audio', f'*{ wildcard }*.wav')
    audio_filepaths = sorted(glob.glob(audio_files_wildcard))

    # create a dataset from audio file paths
    dataset = tf.data.Dataset.from_tensor_slices((audio_filepaths))

    return dataset
