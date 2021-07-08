
# import numpy as np
# import tensorflow as tf
# import glob, os

# from dataset.utils import load_audio_file, load_labels


# def load_dataset(params: dict, dataset_path: str='./dataset', train: bool=True):

#     wildcard = 'train' if train else 'test'

#     # read audio sampling configuration settings
#     sample_rate = params['sample_rate']
#     target_steps = params['seconds_per_sample'] * params['sample_rate']

#     # load audio file paths
#     audio_files_wildcard = os.path.join(dataset_path, 'audio', f'*{ wildcard }*.wav')
#     audio_filepaths = sorted(glob.glob(audio_files_wildcard))

#     # extract sample ids from audio file paths
#     get_sample_id_func = lambda file: os.path.basename(file[:-4])
#     sample_ids = np.array([get_sample_id_func(file) for file in audio_filepaths])

#     # only load labels for the training dataset (test dataset is unlabeled)
#     if train:
#         # load labels from CSV file
#         labels_file_wildcard = os.path.join(dataset_path, 'labels', f'*{ wildcard }*.csv')
#         labels_filepath = glob.glob(labels_file_wildcard)[0]
#         labels_by_sample_id = load_labels(labels_filepath)
#         labels = np.array([labels_by_sample_id[id] for id in sample_ids])

#     # cache the melspectrograms as numpy array
#     spectrograms = np.array([load_audio_file(file, sample_rate, target_steps) for file in audio_filepaths])

#     # create a mapped dataset from file paths
#     dataset = tf.data.Dataset.from_tensor_slices(
#         (spectrograms, labels) if train else (spectrograms, sample_ids))

#     return dataset
