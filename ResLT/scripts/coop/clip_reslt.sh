#!/bin/bash

# custom config
DATA=D:/CV/CLIP-New/HRB/
TRAINER=ResLT
# DATA_PATH=D:/CV/CLIP-New/HRB/Samples/
# TRAIN_META=D:/CV/CLIP-New/HRB/train_metadata.json
DATASET=hrb
CFG=rn50  # config file
CTP=$1  # class token position (end or middle)
NCTX=$2  # number of context tokens
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
CSC=$4  # class-specific context (False or True)

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python reslt_train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.ResLT.N_CTX ${NCTX} \
        TRAINER.ResLT.CSC ${CSC} \
        TRAINER.ResLT.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
    
    fi
done