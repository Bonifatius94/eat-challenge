import os
from glob import glob
import pandas as pd
import tensorflow as tf

from dataset.utils import load_audio_file_as_melspec

type2id = {
    'Apple':     0,
    'Banana':    1,
    'Biscuit':   2,
    'Crisp':     3,
    'Haribo':    4,
    'Nectarine': 5,
    'No_Food':   6
}


class MappedMelspecDatasetLoader:

    def __init__(self, params: dict):
        self.params = params

    def load_dataset(self, train: bool, speaker_ids: list=None) -> tf.data.Dataset:
        # load labels and audio files as pandas dataframes
        df_labels = self._get_labels_from_csv(self._get_labels_csv_filepath(train))
        df_audio_filepaths = self._get_audio_filepaths(train)

        # join labels by audio filename
        df_dataset = pd.merge(df_labels, df_audio_filepaths, on=['file_name'])

        # when in training mode, select audio files by speaker
        if train:
            df_speakers = pd.DataFrame()
            df_speakers['subject_id'] = speaker_ids
            df_dataset = pd.merge(df_dataset, df_speakers, on=['subject_id'])
            df_dataset['food_type'] = df_dataset['food_type'].map(lambda ft: type2id[ft])

        # prepare the label content to be attached to the audio data
        # info: file_name for inference, food_type for training
        label_col = 'food_type' if train else 'file_name'
        df_dataset = df_dataset[['audio_filepath', label_col]]
        print(df_dataset, train)

        # create a tensorflow dataset from the pandas dataframe
        dataset = tf.data.Dataset.from_tensor_slices(
            tensors=(list(df_dataset['audio_filepath']), list(df_dataset[label_col])))

        # load the audio files using a preprocessing function
        # info: do this lazily by a mapped tensorflow dataset
        audio_preproc_func = self._audio_preprocessing_func()
        dataset = dataset.map(
            lambda file, l_or_sid: (audio_preproc_func(file), l_or_sid),
            num_parallel_calls=self.params['num_map_threads'], # scale with multi-core
            deterministic=True # ensure deterministic datasets
        )

        return dataset

    def _audio_preprocessing_func(self):
        # read audio sampling configuration settings
        sample_rate = self.params['sample_rate']
        target_steps = self.params['seconds_shard'] * self.params['sample_rate']

        # define a mapping functions for audio proprocessing
        tf_load_audio_file = lambda file, sr, steps: \
            load_audio_file_as_melspec(file.numpy(), sr, steps)
        pyfunc_load_audio = lambda file: \
            tf.py_function(tf_load_audio_file, \
                [file, sample_rate, target_steps], [tf.float32])
        return pyfunc_load_audio

    def _get_labels_csv_filepath(self, train: bool):
        wildcard = 'train' if train else 'test'
        return glob(f'./dataset/labels/*{wildcard}*.csv')[0]

    def _get_audio_filepaths(self, train: bool):
        wildcard = 'train' if train else 'test'
        audio_filepaths = sorted(glob(f'./dataset/audio/*{ wildcard }*.wav'))
        df = pd.DataFrame()
        df['audio_filepath'] = audio_filepaths
        df['file_name'] = [os.path.basename(path)[:-4] for path in audio_filepaths]
        return df

    def _get_labels_from_csv(self, filepath: str):
        return pd.read_csv(filepath, sep=';', header=0)
