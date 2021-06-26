
# Deep Learning - EAT Challenge

## About
This project is part of an AI challenge of the DeepLearning course 2021 at the University of Augsburg.
The objective to be learned is a classification task telling which food people are eating on audio recordings.

## EAT Dataset Setup
For your convenience, the download of all external project assets (dataset and evaluation metrics) has been
automated by a shell script. After executing the script you should be ready to run / develop the project code.

```sh
# download and unpack the dataset and metric files
./init_dataset_and_metrics.sh <dataset zip password>
```

## How to Run
First, cache the input dataset as TFRecord files for a training session (e.g. naive training).
This should massively improve your training performance.

```sh
# cache the melspectrograms dataset as TFRecord file
python src/main.py preprocess_dataset naive
```

Now, you can launch a training session (e.g. naive training).

```sh
# process a naive training session
python src/main.py run_training naive
```

After that you can sample all inputs of the unknown test dataset using a trained model
and export the prediction results for EAT challenge submission.

```sh
# evaluate trained model results for submission
python src/main.py eval_results naive
```

## Training Results

| Training | Approach Description                        | Test Acc. | Trained Weights                                   | Logs                                              |
| -------- | ------------------------------------------- | --------- | ------------------------------------------------- | ------------------------------------------------- |
| Naive    | Train on audio melspectrograms using Conv2D |      0.40 | TODO: upload weights | TODO: upload logs |
