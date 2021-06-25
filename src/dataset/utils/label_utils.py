
import pandas as pd


def load_labels(labels_filepath: str):

    # load labels into a pandas data frame
    df = pd.read_csv(labels_filepath, sep=';')

    # extract the file_name and subject_id columns
    file_names = df['file_name']
    food_types = df['food_type']

    # create a unique indexing dictionary
    food_types_set = sorted(set(food_types))
    type2id = dict([(food_types_set[i], i) for i in range(len(food_types_set))])

    # apply the indices to the labels
    food_types = [type2id[type] for type in food_types]

    # return (file name, subject id) tuples as dictionary
    return dict(zip(file_names, food_types))
