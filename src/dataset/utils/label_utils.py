
import pandas as pd


def load_labels(labels_filepath: str):

    # load labels into a pandas data frame
    df = pd.read_csv(labels_filepath, sep=';')

    # extract the file_name and subject_id columns
    file_names = df['file_name']
    food_types = df['food_type']

    # apply the indices to the labels
    _, type2id = get_label_dicts()
    food_types = [type2id[type] for type in food_types]

    # return (file name, subject id) tuples as dictionary
    return dict(zip(file_names, food_types))


def get_label_dicts():

    # create (label_id, label_name) mappings
    id2type = {
        0: 'Apple',
        1: 'Banana',
        2: 'Biscuit',
        3: 'Crisp',
        4: 'Haribo',
        5: 'Nectarine',
        6: 'No_Food'
    }

    # create reverse mappings (label_name, label_id)
    type2id = dict(zip(id2type.values(), id2type.keys()))

    return id2type, type2id