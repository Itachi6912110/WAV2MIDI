#!/bin/bash

WAV_FILE=$1
FEAT_DIR="feat"
NOTE_DIR="note"
OUT_DIR="result"
MODEL_DIR="model"

START=15
END=18

mkdir -p ${FEAT_DIR}
mkdir -p ${NOTE_DIR}
mkdir -p ${OUT_DIR}

# Feature Extraction
#python3 MelodyExt.py ${WAV_FILE} ${FEAT_DIR}/demo.feat ${FEAT_DIR}/demo.z ${FEAT_DIR}/demo.cf ${FEAT_DIR}/demo.pitch

# Note Segmentation
python3 NoteSeg.py -d ${FEAT_DIR}/demo.feat -p ${FEAT_DIR}/demo.pitch -of ${NOTE_DIR}/demo.est -sm ${NOTE_DIR}/demo.sdt \
                   -m ${MODEL_DIR}/resnet18_80.model --feat 9 --threshold 0.5

# Transcription to Results
python3 Est2Midi.py ${NOTE_DIR}/demo.est ${OUT_DIR}/demo.midi

# Visualization
python3 Visualize.py -pitch ${FEAT_DIR}/demo.pitch -est ${NOTE_DIR}/demo.est -z ${FEAT_DIR}/demo.z \
                     -cf ${FEAT_DIR}/demo.cf -out ${OUT_DIR}/demo.png -sm ${NOTE_DIR}/demo.sdt \
                     -start ${START} -end ${END}