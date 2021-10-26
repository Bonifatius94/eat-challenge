
# import numpy as np
# import tensorflow as tf

# from dataset.utils import load_audio_file_as_melspec, load_audio_file_as_overlapping_melspec
# from dataset.core import load_labeled_audio_filepaths_dataset


# mode2wildcard = lambda train: 'train' if train else 'test'


# def load_dataset(params: dict, dataset_path: str='./dataset', train: bool=True):

#     tf.print('creating new mapped melspec dataset')

#     # create a dataset from audio file paths yielding (filepath, label_or_sample_id) tuples
#     dataset = load_labeled_audio_filepaths_dataset(dataset_path, train)

#     # define the function for loading and preprocessing audio files
#     file2audio = overlapping_audio_preprocessing_func(params) \
#         if params['multi_shard'] else audio_preprocessing_func(params)

#     # map the audio file paths to preprocessed audio features
#     # info: there can be multiple audio shards per file (-> one-to-many)
#     dataset = dataset.map(
#         lambda file, l_or_sid: (file2audio(file), l_or_sid),
#         num_parallel_calls=params['num_map_threads'], # scale with multi-core
#         deterministic=True # ensure deterministic datasets
#     )

#     # flat map multiple shards accordingly
#     if params['multi_shard']:
#         dataset = dataset.flat_map(lambda x, y: 
#             tf.data.Dataset.range(tf.shape(x, out_type=tf.int64)[1], out_type=tf.int64)
#                 .map(lambda i: (x[0][i], y)))

#     return dataset


# def audio_preprocessing_func(params: dict):

#     # read audio sampling configuration settings
#     sample_rate = params['sample_rate']
#     target_steps = params['seconds_shard'] * params['sample_rate']

#     # define a mapping functions for audio proprocessing
#     tf_load_audio_file = lambda file, sr, steps: load_audio_file_as_melspec(file.numpy(), sr, steps)
#     pyfunc_load_audio = lambda file: \
#         tf.py_function(tf_load_audio_file, [file, sample_rate, target_steps], [tf.float32])

#     return pyfunc_load_audio


# def overlapping_audio_preprocessing_func(params: dict):

#     # read audio sampling configuration settings
#     sample_rate = params['sample_rate']
#     shard_steps = params['seconds_shard'] * params['sample_rate']
#     overlap_steps = params['seconds_overlap'] * params['sample_rate']

#     # define a mapping functions for audio proprocessing
#     tf_load_audio_file = lambda file: load_audio_file_as_overlapping_melspec(
#         file.numpy(), sample_rate, shard_steps, overlap_steps)
#     pyfunc_load_audio = lambda file: tf.py_function(tf_load_audio_file, [file], [tf.float32])

#     return pyfunc_load_audio
