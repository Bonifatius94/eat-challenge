
import numpy as np
import librosa


def load_audio_file(audio_filepath: str, sample_rate: int, target_steps: int):

    # load the wave file in waveform
    x, _ = librosa.load(audio_filepath, sr=sample_rate)
    sampled_steps = x.shape[0]

    # add zero-padding if the sample is too short
    if sampled_steps < target_steps:
        diff = target_steps - sampled_steps
        padding = np.zeros(diff, dtype = np.int16)
        x = np.concatenate((x, padding))

    # cut samples that are too long
    if sampled_steps > target_steps:
        x = x[:target_steps]

    # convert the waveform into a spectrogram with log-scale
    x = librosa.feature.melspectrogram(x, sr=sample_rate)
    x = librosa.power_to_db(x)
    x = np.expand_dims(x, axis=-1)

    return x


def load_audio_file_overlapping(audio_filepath: str,
    sample_rate: int, shard_steps: int, overlap_steps: int):

    # load the wave file in waveform
    raw_audio, _ = librosa.load(audio_filepath, sr=sample_rate)
    sampled_steps = raw_audio.shape[0]

    step = 0
    audio_shards = []

    # loop until the last shard is reached
    while step <= sampled_steps - shard_steps:

        # cut the next audio shard
        shard = raw_audio[step:(step+shard_steps)]
        audio_shards.append(shard)
        step += overlap_steps

    # collect the last shard (padded with trailing zeros)
    if step < sampled_steps:
        diff = shard_steps - (sampled_steps - step)
        padding = np.zeros(diff, dtype = np.int16)
        raw_audio = np.concatenate((raw_audio[step:-1], padding))

    # convert all shards into logarithmized melspectrograms
    melspec_shards = []
    for shard in audio_shards:

        # convert the waveform into a spectrogram with log-scale
        shard = librosa.feature.melspectrogram(shard, sr=sample_rate)
        shard = librosa.power_to_db(shard)
        shard = np.expand_dims(shard, axis=-1)
        melspec_shards.append(shard)

    return melspec_shards