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


data_root=/home/xu/workspace/surreal/datageneration/dataset256
N=24000

echo 'Creating Folders'
cd $data_root
mkdir $data_root/A -p
mkdir $data_root/B -p
mkdir $data_root/A_depth -p
mkdir $data_root/B_depth -p
mkdir $data_root/flow_map
mkdir $data_root/depth_map

echo 'Converting exr to npy for Flow'
cd $data_root/gtflow
#ls | cat -n | while read n f; do mv "$f" `printf "%05d.exr" $((n-1))`; done
python /home/xu/workspace/surreal/datageneration/exr.py --input $data_root/gtflow --output $data_root/flow_map --type flow

echo 'Renaming flow'
cd $data_root/flow_map
ls | cat -n | while read n f; do mv "$f" `printf "%05d.npy" $((n-1))`; done

echo 'Removing redundent flow'
cd $data_root/flow_map
for i in $(seq 0 2 $((N-1)) );do rm `printf "%05d.npy" $i`; done


echo 'Converting exr to npy for Depth'
cd $data_root/depth
#ls | cat -n | while read n f; do mv "$f" `printf "%05d.exr" $((n-1))`; done
python /home/xu/workspace/surreal/datageneration/exr.py --input $data_root/depth --output $data_root/depth_map --type depth

echo 'Splitting Depth'
cd $data_root/depth_map
ls | cat -n | while read n f; do mv "$f" `printf "%05d.npy" $((n-1))`; done
for i in $(seq 0 2 $((N-1)));do if [ ! -f `printf "%05d.npy" $i` ];then break; fi; cp `printf "%05d.npy" $i` $data_root/A_depth/; done
for i in $(seq 1 2 $((N-1)));do if [ ! -f `printf "%05d.npy" $i` ];then break; fi; cp `printf "%05d.npy" $i` $data_root/B_depth/; done

echo 'Renaming Depth'
cd $data_root/A_depth
ls | cat -n | while read n f; do mv "$f" `printf "%05d.npy" $((n-1))`; done
cd $data_root/B_depth
ls | cat -n | while read n f; do mv "$f" `printf "%05d.npy" $((n-1))`; done


echo 'Splitting RGB'
cd $data_root/rgb
#ls | cat -n | while read n f; do cp "$f" `printf "%05d.png" $((n-1))`; done
for i in $(seq 0 2 $((N-1)));do cp `printf "%05d.png" $i` $data_root/A/; done
for i in $(seq 1 2 $((N-1)));do cp `printf "%05d.png" $i` $data_root/B/; done


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


