
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

```sh
# run the project's startup script
python src/main.py
```

TODO: add main script arguments to specify the training approach

## Training Results

| Training | Approach Description                        | Test Acc. | Trained Weights                                   | Logs                                              |
| -------- | ------------------------------------------- | --------- | ------------------------------------------------- | ------------------------------------------------- |
| Naive    | Train on audio melspectrograms using Conv2D |      0.92 | https://megastore.uni-augsburg.de/get/uwXwBXYScZ/ | https://megastore.uni-augsburg.de/get/o3nDS5DN1U/ |
