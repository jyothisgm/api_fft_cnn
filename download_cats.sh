#! /bin/bash
mkdir cat_images && cd cat_images &&\
curl -L -o archive.zip https://www.kaggle.com/api/v1/datasets/download/borhanitrash/cat-dataset &&\
unzip archive.zip && rm archive.zip
