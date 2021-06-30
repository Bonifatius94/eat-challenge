
import numpy as np
import librosa


def load_audio_file_as_melspec(audio_filepath: str, sample_rate: int, target_steps: int):

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


def load_audio_file_as_overlapping_melspec(audio_filepath: str,
    sample_rate: int, shard_steps: int, overlap_steps: int):

    # load the wave file in waveform
    raw_audio, _ = librosa.load(audio_filepath, sr=sample_rate)
    sample_steps = raw_audio.shape[0]

    step = 0
    audio_shards = []

    # loop until the last shard is reached
    while step <= sample_steps - shard_steps:

        # cut the next audio shard
        shard = raw_audio[step:(step+shard_steps)]
        audio_shards.append(shard)
        step += (shard_steps - overlap_steps)

    # collect the last shard (padded with trailing zeros)
    # ensure to always sample at least 1 shard, filter too small shards
    if not (step > 0 and sample_steps - step < 0.5 * shard_steps):
        diff = shard_steps - (sample_steps - step)
        padding = np.zeros(diff, dtype = np.int16)
        last_shard = np.concatenate((raw_audio[step:], padding))
        audio_shards.append(last_shard)

    # convert all shards into logarithmized melspectrograms
    melspec_shards = []
    for shard in audio_shards:

        # convert the waveform into a spectrogram with log-scale
        shard = librosa.feature.melspectrogram(shard, sr=sample_rate)
        shard = librosa.power_to_db(shard)
        shard = np.expand_dims(shard, axis=-1)
        melspec_shards.append(shard)

    print('num_shards', len(melspec_shards))

    return np.array(melspec_shards)


def load_audio_file_as_amplitude(audio_filepath: str, sample_rate: int, target_steps: int):

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
    # x = librosa.feature.melspectrogram(x, sr=sample_rate)
    # x = librosa.power_to_db(x)
    x = np.expand_dims(x, axis=-1)

    return x


def load_audio_file_as_overlapping_amplitude(audio_filepath: str,
    sample_rate: int, shard_steps: int, overlap_steps: int):

    # load the wave file in waveform
    raw_audio, _ = librosa.load(audio_filepath, sr=sample_rate)
    sample_steps = raw_audio.shape[0]

    step = 0
    audio_shards = []

    # loop until the last shard is reached
    while step <= sample_steps - shard_steps:

        # cut the next audio shard
        shard = raw_audio[step:(step+shard_steps)]
        audio_shards.append(shard)
        step += (shard_steps - overlap_steps)

    # collect the last shard (padded with trailing zeros)
    # ensure to always sample at least 1 shard, filter too small shards
    if not (step > 0 and sample_steps - step < 0.5 * shard_steps):
        diff = shard_steps - (sample_steps - step)
        padding = np.zeros(diff, dtype = np.int16)
        last_shard = np.concatenate((raw_audio[step:], padding))
        audio_shards.append(last_shard)

    # convert all shards into logarithmized melspectrograms
    amplitude_shards = []
    for shard in audio_shards:

        # convert the waveform into a spectrogram with log-scale
        # shard = librosa.feature.melspectrogram(shard, sr=sample_rate)
        # shard = librosa.power_to_db(shard)
        shard = np.expand_dims(shard, axis=-1)
        amplitude_shards.append(shard)

    print('num_shards', len(amplitude_shards))

    return np.array(amplitude_shards)
