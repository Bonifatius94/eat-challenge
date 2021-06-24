
import numpy as np
import librosa
import tensorflow as tf
import glob, sys, os
import pandas as pd


def load_datasets(params: dict, dataset_path: str='./dataset'):

    # load audio datasets
    train_data = load_dataset(params, dataset_path, 'train')
    test_data = load_dataset(params, dataset_path, 'test')

    # batch both datasets properly
    train_data = train_data.batch(params['batch_size'])
    test_data = test_data.batch(params['batch_size'])

    # tune the performance by prefetching data
    train_data = train_data.prefetch(5)
    test_data = test_data.prefetch(5)

    # shuffle the training dataset properly
    train_data = train_data.shuffle(5)

    return train_data, test_data


def load_dataset(params: dict, dataset_path: str, wildcard: str):

    # load audio file paths
    audio_files_wildcard = os.path.join(dataset_path, 'audio', f'*{ wildcard }*.wav')
    audio_filepaths = sorted(glob.glob(audio_files_wildcard))

    # load labels from CSV file
    labels_file_wildcard = os.path.join(dataset_path, 'labels', f'*{ wildcard }*.csv')
    labels_filepath = glob.glob(labels_file_wildcard)[0]
    labels_by_file = load_labels(labels_filepath)

    #print(labels_by_file.keys())
    #print(labels_by_file['train_001'])
    #pass

    # create a mapped dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((audio_filepaths))

    # read audio sampling configuration settings
    sample_rate = params['sample_rate']
    target_steps = params['seconds_per_sample'] * params['sample_rate']

    # map the sample's label and audio file contents -> (audio content, label) tuples
    get_label_func = lambda file: labels_by_file[os.path.basename(file.numpy().decode('ascii')[:-4])]
    dataset = dataset.map(lambda filepath: (
            tf.py_function(load_audio_file, [filepath, sample_rate, target_steps], [tf.float32]),
            tf.py_function(get_label_func, [filepath], [tf.int32])
        ),
        num_parallel_calls=params['num_threads'],
        deterministic=False
    )
    # info: when using the map() function, the audio content is only loaded when needed
    #       -> data can be prefetched for better performance (multi-threading, etc.)
    #       -> machine's RAM / GPU RAM won't explode from loading the dataset

    # TODO: think of loading pre-processed melspectrograms (from TFRecord), no more raw wav files

    return dataset


def load_labels(labels_filepath: str):

    # load labels into a pandas data frame
    df = pd.read_csv(labels_filepath, sep=';')

    # extract the file_name and subject_id columns
    file_names = df['file_name']
    subject_ids = df['subject_id']

    # return (file name, subject id) tuples as dictionary
    return dict(zip(file_names, subject_ids)) # TODO: check if this works


def load_audio_file(audio_filepath: str, sample_rate: int, target_steps: int):

    # load the wave file in waveform
    x, _ = librosa.load(audio_filepath.numpy(), sr=sample_rate)
    sampled_steps = x.shape[0]

    # add zero-padding if the sample is too short
    if sampled_steps < target_steps:
        diff = target_steps - sampled_steps
        padding = np.zeros(diff, dtype = np.int16)
        x = np.concatenate((x, padding))

    # cut samples that are too long
    if sampled_steps > target_steps:
        x = x[:target_steps]

    # convert the waveform into a spectrogram
    x = librosa.feature.melspectrogram(x, sr=sample_rate)
    x = librosa.power_to_db(x)
    x = np.expand_dims(x, axis=-1)

    return x
