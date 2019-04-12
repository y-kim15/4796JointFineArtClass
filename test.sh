#!/bin/bash

# activate virtual environment
source venv/bin/activate

cd env

echo '============================================================================='
echo 'testing on small testset...'
echo 'resnet model'
echo '-----------------------------------------------------------------------------'
python3 evaluate_result.py -t acc -d data/wikipaintings_small/wikipaintings_test -cm -s --cm_type --model_name small --report --show
echo '-----------------------------------------------------------------------------'
echo 'vgg model'
echo '-----------------------------------------------------------------------------'
python3 evaluate_result.py -t acc --m_type vgg -d data/wikipaintings_small/wikipaintings_test -cm -s --model_name small --report --show

echo '============================================================================='
echo 'plotting history...'
python3 evaluate_result.py --his b -f models/resnet50_model/resnet_eg_history.pck --show -s

echo '============================================================================='
echo 'predict on an image... vgg'
echo '-----------------------------------------------------------------------------'
python3 evaluate_result.py -t pred --m_type vgg -d data/images/gustav-klimt_the-sunflower-1907.jpg
echo '-----------------------------------------------------------------------------'
echo 'predict on an image... resnet'
echo '-----------------------------------------------------------------------------'
python3 evaluate_result.py -t pred -d data/images/gustav-klimt_the-sunflower-1907.jpg

echo '============================================================================='
echo 'generate activation map...'
echo 'resnet model'
echo '-----------------------------------------------------------------------------'
python3 evaluate_result.py --act conv1 --show
echo '-----------------------------------------------------------------------------'
echo 'vgg model'
echo '-----------------------------------------------------------------------------'
python3 evaluate_result.py --m_type vgg --act block1_conv1 --show
