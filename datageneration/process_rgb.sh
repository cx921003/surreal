#!/bin/bash

set -e
# SET PATHS HERE
FFMPEG_PATH=/home/gvarol/tools/ffmpeg/ffmpeg_build_sequoia_h264
PYTHON2_PATH=/usr # PYTHON 2
cd /home/xu/workspace/surreal/datageneration

# BUNLED PYTHON
BUNDLED_PYTHON=${BLENDER_PATH}/2.78/python
export PYTHONPATH=${BUNDLED_PYTHON}/lib/python3.4:${BUNDLED_PYTHON}/lib/python3.4/site-packages
export PYTHONPATH=${BUNDLED_PYTHON}:${PYTHONPATH}

# FFMPEG
export LD_LIBRARY_PATH=${FFMPEG_PATH}/lib:${X264_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${FFMPEG_PATH}/bin:${PATH}

data_root=/home/xu/workspace/surreal/car_full_bright
N=10545
k=3

echo 'Creating Folders'
cd $data_root
mkdir $data_root/A -p
mkdir $data_root/B -p

echo 'Splitting RGB'
cd $data_root/rgb
ls | cat -n | while read n f; do cp "$f" `printf "%05d.png" $((n-1))`; done
for i in $(seq 1 $k $((N-1)));do cp `printf "%05d.png" $i` $data_root/A/; done
for i in $(seq 2 $k $((N-1)));do cp `printf "%05d.png" $i` $data_root/B/; done


echo 'Renaming RGB'
cd $data_root/A
ls | cat -n | while read n f; do mv "$f" `printf "%05d.png" $((n-1))`; done
cd $data_root/B
ls | cat -n | while read n f; do mv "$f" `printf "%05d.png" $((n-1))`; done


echo 'Creating Dataset'
python /home/xu/workspace/view_synthesis/datasets/combine_A_and_B.py \
--fold_A $data_root/A \
--fold_B $data_root/B \
--fold_AB $data_root/train


cd $data_root/train
for i in $(seq $((N*4/15)) $N);do mv `printf "%05d.png" $i` $data_root/test/; done

