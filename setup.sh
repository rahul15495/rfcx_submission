#!/bin/bash

git clone --depth 1 --branch v0.1.0  https://github.com/s3prl/s3prl.git
apt-get install tmux
pip instal -r requirements.txt

