#!/bin/bash

# activate virtual environment
#source venv/bin/activate

cd env

echo 'testing on small testset...'
echo 'resnet model'
python3 evaluate_result.py -t acc -d data/wikipaintings_small/wikipaintings_test -cm -s --model_name small --report --show
echo 'vgg model'
python3 evaluate_result.py -t acc --m_type vgg -d data/wikipaintings_small/wikipaintings_test -cm -s --model_name small --report --show


echo 'plotting history...'
python3 evaluate_result.py --his b -f models/resnet50_model/resnet_eg_history.pck --show -s

echo 'predict on an image... vgg'
python3 evaluate_result.py -t pred --m_type vgg -d data/images/gustav-klimt_the-sunflower-1907.jpg

echo 'predict on an image... resnet'
python3 evaluate_result.py -t pred -d data/images/gustav-klimt_the-sunflower-1907.jpg

echo 'generate activation map...'
python3 evaluate_result.py --act conv1 --show
