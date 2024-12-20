#!/bin/bash

# custom config
DATA=D:/CV/CLIP-New/HRB/
TRAINER=CLIP_Adapter

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
        python D:/CV/CLIP-New/CoOp/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/ResLT/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.CLIP_Adapter.N_CTX ${NCTX} \
        TRAINER.CLIP_Adapter.CSC ${CSC} \
        TRAINER.CLIP_Adapter.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done