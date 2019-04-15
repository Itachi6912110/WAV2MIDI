#!/bin/bash

DHEAD="data/ISMIR2014_note/"
AHEAD="ans/ISMIR2014_ans/"
PHEAD="pitch/ISMIR2014/"
TDHEAD="data/TONAS_note/"
TAHEAD="ans/TONAS_ans/"
TPHEAD="pitch/TONAS/"
SIZE_LR=1
BASE_LR=0.001
LR=$(echo "${BASE_LR} * ${SIZE_LR}" | bc)
HS1=150
HL1=3
WS=9
SE=5
BIDIR1=1
NORM=ln

END_EPOCH=80
BATCH=10
FEAT=SN_SF1_SIN_SF1_ZN_F9
FEAT_NUM=9

THRESHOLD=0.5

MDIR="model/sdt6_resnet50_${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${END_EPOCH}b${BATCH}_${FEAT}_sample"
EMFILE1="${MDIR}/onoffset_attn_${NORM}_onenc_k${WS}l${HL1}h${HS1}b${BIDIR1}e${END_EPOCH}b${BATCH}_${FEAT}"
DMFILE1="${MDIR}/onoffset_attn_${NORM}_ondec_k${WS}l${HL1}h${HS1}b${BIDIR1}e${END_EPOCH}b${BATCH}_${FEAT}"
EFILE="output/single/ISMIR2014_sdt6_resnet50_${NORM}k${WS}_l${HL1}h${HS1}b{BIDIR1}_e${END_EPOCH}b${BATCH}_${FEAT}.csv"
VFILE="output/total/ISMIR2014_sdt6_resnet50_${NORM}k${WS}_l${HL1}h${HS1}b{BIDIR1}_e${END_EPOCH}b${BATCH}_${FEAT}.csv"
TEFILE="output/train_single/ISMIR2014_sdt6_resnet50_${NORM}k${WS}_l${HL1}h${HS1}b{BIDIR1}_e${END_EPOCH}b${BATCH}_${FEAT}_sample.csv"
TVFILE="output/train_total/ISMIR2014_sdt6_resnet50_${NORM}k${WS}_l${HL1}h${HS1}b{BIDIR1}_e${END_EPOCH}b${BATCH}_${FEAT}_sample.csv"
TROUTDIR="output/sdt6_resnet50_est"
HMMFEATDIR="output/sdt6_resnet_hmmfeat"

echo -e "Evaluating OnOffset Model Info:"
echo -e "HS1=${HS1} HL1=${HL1} BIDIR1=${BIDIR1} NORM=${NORM}"
echo -e "WS=${WS} SE=${SE}"
echo -e "EPOCHS=${END_EPOCH} BATCH=${BATCH} FEAT=${FEAT}"
echo -e "Onset Encoder Model: ${EMFILE1}"
echo -e "Onset Decoder Model: ${DMFILE1}"


mkdir -p ${TROUTDIR}
mkdir -p ${HMMFEATDIR}

echo -e "Start Evaluation on ISMIR2014 Validation Set"
for num in $(seq 1 38)
do
    python3 eval_resnet50_fmeasure.py -d ${DHEAD}/${FEAT}/${num}_${FEAT} -a ${AHEAD}${num}.GroundTruth -pf ${PHEAD}${num}_P -em1 ${EMFILE1} -dm1 ${DMFILE1} -p ${num} -ef ${EFILE} -tf ${VFILE} -l ${LR} \
    --hs1 ${HS1} --hl1 ${HL1} --ws ${WS} --single-epoch ${SE} --bidir1 ${BIDIR1} --norm ${NORM} --feat ${FEAT_NUM} --threshold ${THRESHOLD} -of ${TROUTDIR}/${num}_test -sf ${TROUTDIR}/${num}_sdt_test -sm ${TROUTDIR}/${num}_sm_test \
    -hmm ${HMMFEATDIR}/${num}_feat_test
done

#echo -e "Start Evaluation on TONAS Training Set"
#for num in $(seq 1 82)
#do
#    python3 eval_resnet50_fmeasure.py -d ${TDHEAD}/${FEAT}/${num}_${FEAT} -a ${TAHEAD}${num}.GroundTruth -pf ${TPHEAD}${num}_P -em1 ${EMFILE1} -dm1 ${DMFILE1} -p ${num} -ef ${TEFILE} -tf ${TVFILE} -l ${LR} \
#    --hs1 ${HS1} --hl1 ${HL1} --ws ${WS} --single-epoch ${SE} --bidir1 ${BIDIR1} --norm ${NORM} --feat ${FEAT_NUM} --threshold ${THRESHOLD} -of ${TROUTDIR}/${num}_train -sf ${TROUTDIR}/${num}_sdt_train -sm ${TROUTDIR}/${num}_sm_train \
#    -hmm ${HMMFEATDIR}/${num}_feat_train
#done