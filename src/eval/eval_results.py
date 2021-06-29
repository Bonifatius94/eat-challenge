
import pandas as pd
import numpy as np
import tensorflow as tf

from dataset.utils import get_label_dicts


class EvalSession:

    def __init__(self, dataset, model):
        super(EvalSession, self).__init__()

        # load the test dataset and the pre-trained model
        self.dataset = dataset
        self.model = model


    def export_results(self, out_filepath: str):

        # predict all test samples
        sampled_preds = np.array([])
        sample_ids = np.array([])

        # load the label conversion dictionaries
        id2type, _ = get_label_dicts()

        # extract the related sample IDs from the dataset
        for batch in iter(self.dataset):

            # unpack the batch and predict on it
            inputs, ids = batch
            preds = tf.argmax(self.model(inputs), axis=1)
            # TODO: handle the prediction of multiple audio shards

            # preprocess preds and sample ids for export
            preds = np.array([id2type[id] for id in preds.numpy()])
            ids = [id.decode('ascii') for id in ids.numpy()]

            # apply the results to the lists
            sample_ids = np.concatenate((sample_ids, ids))
            sampled_preds = np.concatenate((sampled_preds, preds))

        # write the results to the given output file (csv format)
        df = pd.DataFrame({ 'sample_id': sample_ids, 'sampled_preds': sampled_preds })
        df.to_csv(out_filepath, header=False, index=False, sep=';', quoting=1, quotechar='\'')
