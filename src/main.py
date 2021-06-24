
from dataset import load_datasets

# TODO: add the program entrypoint here ...

def main():

    # TODO: evaluate script args

    # define hyper-params required for creating the datasets
    params = {
        'batch_size': 32,
        'seconds_per_sample': 4,
        'sample_rate': 16000,
        'num_threads': 3
    }

    # load a dataset for testing purposes
    train_data, _ = load_datasets(params)

    # write the first batch to console
    print(iter(train_data).next())


if __name__ == '__main__':
    main()