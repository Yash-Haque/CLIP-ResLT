#!/bin/bash

# custom config
DATA=D:/CV/CLIP-New/HRB/ # Path to Dataset
TRAINER=ResLT
TRAIN_META=D:/CV/CLIP-New/HRB/train_metadata.json # Path to Train Metadata
TEST_META=D:/CV/CLIP-New/HRB/test_metadata.json # Path to Test Metadata
DATASET=hrb
CFG=rn50  # config file
CTP=$1  # class token position (end or middle)
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)
BETA=$3 # Beta value (0.85)
GAMMA=$4 # Gamma value (0.5)
DROPOUT=$5  # dropout during training (True or False)

# Command to run this script: ./clip_reslt.sh end 16 0.85 0.5 True

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/beta${BETA}_gamma${GAMMA}_ctp${CTP}/seed${SEED}
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
        TRAINER.ResLT.BETA ${BETA} \
        TRAINER.ResLT.GAMMA ${GAMMA} \
        TRAINER.ResLT.DROPOUT ${DROPOUT} \
        TRAINER.ResLT.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.HERB.TRAIN_META ${TRAIN_META}  
    fi
done