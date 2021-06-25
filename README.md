
# Deep Learning - EAT Challenge

## About
This project is part of an AI challenge of the DeepLearning course 2021 at the University of Augsburg.
The objective to be learned is a classification task telling which food people are eating on audio recordings.

## Dataset / Metrics Preparation
For your convenience, the download of all external project assets (dataset and evaluation metrics) has been
automated by the shell script 'init_dataset_and_metrics.sh'. After executing the following command
you should be ready to run / develop the project code.

```sh
# download and unpack the dataset and metric files
./init_dataset_and_metrics.sh <dataset zip password>
```

## How to Run
First, cache the audio input data (melspectrogram) as TFRecord files.
This should massively improve your training performance.

```sh
# cache the melspectrograms dataset as TFRecord file
python src/main.py preprocess_dataset
```

Now, you can launch a training session (e.g. the naive training).

```sh
# process a naive training session
python src/main.py naive_training
```

After that you can sample all inputs of the unknown test dataset using a trained model
and export the prediction results for EAT challenge submission.

```sh
# evaluate trained model results for submission
# TODO: add the command when the export function is ready to use
```

## Training Results

| Training | Approach Description                        | Test Acc. | Trained Weights                                   | Logs                                              |
| -------- | ------------------------------------------- | --------- | ------------------------------------------------- | ------------------------------------------------- |
| Naive    | Train on audio melspectrograms using Conv2D |      0.34 | https://megastore.uni-augsburg.de/get/zWkOmMK5qK/ | https://megastore.uni-augsburg.de/get/8tkquPq0Ee/ |
