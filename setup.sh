#!/bin/bash

git clone --depth 1 --branch v0.1.0  https://github.com/s3prl/s3prl.git

pip instal -r requirements.txt


gcloud auth login

mkdir model_weights

cd model_weights

gsutil -m cp -r "gs://rahul_chaurasia_rfcx/model_weights/mockingjay_pretrained_v4/" .


