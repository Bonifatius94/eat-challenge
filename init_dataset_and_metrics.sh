#!/bin/bash

# ===================================
#   DATASET AND METRICS PREPARATION
# ===================================

# ===================================
#        1)  CONFIGURATION
# ===================================

# make sure that a zip password was specified, otherwise abort with error
if [ -z "$1" ]; then
    echo 'Invalid script args! Please specify a zip password as first parameter!'
    exit -1
fi

# interpret the first script arg as zip file password
ZIP_PASSWORD=$1

# define other parameters required for downloading and unzipping
ZIP_URL=https://megastore.uni-augsburg.de/get/rXvJzBqyAJ/
ZIP_FILENAME=CMI2018-EAT_package_audio.zip
#ZIP_ROOTDIR=ICMI2018-EAT_package
DATASET_OUTDIR=./dataset
METRICS_OUTDIR=./src/metrics
DOCS_OUTDIR=./doc

# ===================================
#         2) ZIP DOWNLOAD
# ===================================

# download the zip file
wget "$ZIP_URL" -O "$ZIP_FILENAME"

# make sure that the downloaded zip file exists, otherwise abort with error
if [ ! -f "$ZIP_FILENAME" ]; then
    echo 'Downloading the zip file failed! Please try again!'
    exit -2
fi

# ===================================
#         3) ZIP UNPACKING
# ===================================

# create extraction target directories
mkdir -p "$DATASET_OUTDIR/audio"
mkdir -p "$DATASET_OUTDIR/audio_features"
mkdir -p "$DATASET_OUTDIR/audio_features_instance"
mkdir -p "$DATASET_OUTDIR/labels"
mkdir -p "$METRICS_OUTDIR/baseline_end2you"
mkdir -p "$METRICS_OUTDIR/baseline_xbow"

# unzip all files and directories required for training
unzip -j -q -P "$ZIP_PASSWORD" "$ZIP_FILENAME" */audio/* -d "$DATASET_OUTDIR/audio"
unzip -j -q -P "$ZIP_PASSWORD" "$ZIP_FILENAME" */audio_features/* -d "$DATASET_OUTDIR/audio_features"
unzip -j -q -P "$ZIP_PASSWORD" "$ZIP_FILENAME" */audio_features_instance/* \
      -d "$DATASET_OUTDIR/audio_features_instance"
unzip -j -q -P "$ZIP_PASSWORD" "$ZIP_FILENAME" */labels/* -d "$DATASET_OUTDIR/labels"

# unzip all files and directories required for metrics evaluation
unzip -j -q -P "$ZIP_PASSWORD" "$ZIP_FILENAME" */baseline_end2you/* \
      -d "$METRICS_OUTDIR/baseline_end2you"
unzip -j -q -P "$ZIP_PASSWORD" "$ZIP_FILENAME" */baseline_xbow/* -d "$METRICS_OUTDIR/baseline_xbow"

# unzip all documentation files
unzip -j -q -P "$ZIP_PASSWORD" "$ZIP_FILENAME" */README */README_Test -d "$DOCS_OUTDIR"

# make sure that the unzipped files and folders exist, otherwise abort with error
if [ ! -d "$DATASET_OUTDIR/audio" ] || [ ! -d "$DATASET_OUTDIR/audio_features" ] || [ ! -d "$DATASET_OUTDIR/labels" ]; then
    echo 'Unzipping dataset failed! Please try again!'
    exit -3
fi
if [ ! -d "$METRICS_OUTDIR/baseline_end2you" ] || [ ! -d "$METRICS_OUTDIR/baseline_xbow" ]; then
    echo 'Unzipping metrics failed! Please try again!'
    exit -3
fi
if [ ! -f "$DOCS_OUTDIR/README" ] || [ ! -f "$DOCS_OUTDIR/README_Test" ]; then
    echo 'Unzipping docs failed! Please try again!'
    exit -3
fi

# ===================================
#         4) ZIP CLEANUP
# ===================================

# remove the zip file to save space on disk
rm -rf CMI2018-EAT_package_audio.zip

# ===================================
#         5) RESULT REPORT
# ===================================

echo "Successfully initialized datasets and metrics!"
echo "unpacked datasets to '$DATASET_OUTDIR'"
echo "unpacked metrics to '$METRICS_OUTDIR'"
echo "unpacked documentation to '$DOCS_OUTDIR'"

# ===================================
#     Marco Tröster, 2021-06-24
# ===================================

