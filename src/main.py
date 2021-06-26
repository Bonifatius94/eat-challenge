
import sys, json, glob, os
from train import TrainSessionFactory
from dataset import DatasetFactory
from eval import EvalSession


def main():

    # make sure the correct amount script args is provided
    if len(sys.argv) <= 1:
        raise ValueError('Invalid script arguments! Requires the task to be executed as first argument!')
    if len(sys.argv) <= 2:
        raise ValueError('Invalid script arguments! Requires the training type as second argument!')

    # parse the task from script args
    task = sys.argv[1]
    train_type = sys.argv[2]

    # run the task
    if task == 'preprocess_dataset': preprocess_dataset(train_type)
    elif task == 'run_training': run_training(train_type)
    elif task == 'eval_results': eval_results(train_type)
    else: raise ValueError(f'Unknown main task { task }!')
    # TODO: add more tasks here ...


def preprocess_dataset(train_type: str):

    train_params = load_train_params(train_type)

    # create a preprocessed melspec dataset (TFRecord) from raw audio data
    factory = DatasetFactory()
    factory.create_cached_dataset(train_params)


def run_training(train_type: str):

    train_params = load_train_params(train_type)

    # create a training session from the loaded training params
    train_sess_factory = TrainSessionFactory()
    train_sess = train_sess_factory.create_train_session(train_type, train_params)

    # run the training
    train_sess.run_training()


def eval_results(train_type: str):

    train_params = load_train_params(train_type)

    # load the best pre-trained model of the given training type
    train_sess_factory = TrainSessionFactory()
    train_sess = train_sess_factory.create_train_session(train_type, train_params)
    train_sess.load_best_model()

    # load the test dataset to be sampled on
    factory = DatasetFactory()
    dataset = factory.load_dataset(train_params['dataset_specifier'], train_params, train=False)
    dataset = dataset.batch(train_params['batch_size'])

    # make sure the eval results outdir exists
    eval_outdir = './eval_results'
    if not (os.path.exists(eval_outdir) and os.path.isdir(eval_outdir)):
        os.makedirs(eval_outdir)

    # perform the evaluation and write the results to csv file
    session = EvalSession(dataset, train_sess.model)
    session.export_results(os.path.join(eval_outdir, f'{ train_type }.csv'))


def load_train_params(train_type: str):

    # load the training parameters of the given training type
    train_config_file = glob.glob(f'./config/*{ train_type }*.json')[0]
    return load_config_params(train_config_file)


def load_config_params(config_filepath: str):
    with open(config_filepath, 'r') as file:
        return json.load(file)


if __name__ == '__main__':
    main()
