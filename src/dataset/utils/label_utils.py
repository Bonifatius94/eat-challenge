
import pandas as pd


def load_labels(labels_filepath: str):

    # load labels into a pandas data frame
    df = pd.read_csv(labels_filepath, sep=';')

    # extract the file_name and subject_id columns
    file_names = df['file_name']
    subject_ids = df['subject_id']

    # return (file name, subject id) tuples as dictionary
    return dict(zip(file_names, subject_ids))
