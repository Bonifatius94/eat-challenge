
from dataset.melspec import load_mapped_dataset, load_cached_np_dataset
from dataset.melspec import load_tfrecord_dataset, write_tfrecord_dataset


class DatasetFactory:

    def create_cached_dataset(self, dataset_specifier: str, params: dict):

        if dataset_specifier == 'tfrecord_melspec':
            write_tfrecord_dataset(params, train=True)
            write_tfrecord_dataset(params, train=False)
        # TODO: add more cached datasets here ...
        else: raise ValueError(f'No dataset caching available for specifier { dataset_specifier }!')


    def load_dataset(self, dataset_specifier: str, params: dict, train: bool):

        # load the dataset of the given type
        if dataset_specifier == 'mapped_melspec': return load_mapped_dataset(params, train=train)
        elif dataset_specifier == 'tfrecord_melspec': return load_tfrecord_dataset(params, train=train)
        elif dataset_specifier == 'np_ram_melspec': return load_cached_np_dataset(params, train=train)
        else: raise ValueError(f'Cannot load data for unknown dataset specifier { dataset_specifier }!')
