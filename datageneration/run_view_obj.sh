#!/bin/bash

JOB_PARAMS=${1:-'--idx 13 --ishape 0 --stride 50'} # defaults to [0, 0, 50]
set -e
# SET PATHS HERE
FFMPEG_PATH=/home/gvarol/tools/ffmpeg/ffmpeg_build_sequoia_h264
PYTHON2_PATH=/usr # PYTHON 2
BLENDER_PATH=/home/xu/Software/blender-2.78a-linux-glibc211-x86_64
cd /home/xu/workspace/surreal/datageneration

# BUNLED PYTHON
BUNDLED_PYTHON=${BLENDER_PATH}/2.78/python
export PYTHONPATH=${BUNDLED_PYTHON}/lib/python3.4:${BUNDLED_PYTHON}/lib/python3.4/site-packages
export PYTHONPATH=${BUNDLED_PYTHON}:${PYTHONPATH}

# FFMPEG
export LD_LIBRARY_PATH=${FFMPEG_PATH}/lib:${X264_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${FFMPEG_PATH}/bin:${PATH}

### RUN PART 1  --- Uses python3 because of Blender
#if [ -d ./output_dataset ];then
#    rm ./output_dataset/ -r
#fi
for idx in $(seq 0 20);do
JOB_PARAMS=${1:-'--idx 13 --ishape 0 --stride 50 --idx_cloth '$((idx*200))}
echo $JOB_PARAMS
$BLENDER_PATH/blender -b -P rendering_objects.py -- ${JOB_PARAMS}
done
