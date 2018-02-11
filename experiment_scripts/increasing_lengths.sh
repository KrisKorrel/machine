#!/bin/bash

CUDA=$1
TRAIN_MAX_LENGTH=$2
RUN_COUNTER=$3

LOG_LEVEL='debug'

# set values
EPOCHS=70
OPTIMIZER='adam'
LR=0.001
RNN_CELL='lstm'
EMB_SIZE=200
H_SIZE=200
N_LAYERS=2
DROPOUT_ENCODER=0.5
DROPOUT_DECODER=0.5
TF=0.5
BATCH_SIZE=128
ATTENTION=true
BIDIRECTIONAL=false
PRINT_EVERY=20
SAVE_EVERY=131

if [ "$BIDIRECTIONAL" = true ]
then
    BIDIRECTIONAL="--bidirectional"
else
    BIDIRECTIONAL=""
fi

if [ "$ATTENTION" = true ]
then
    ATTENTION="--attention"
else
    ATTENTION=""
fi

# Define the train data and checkpoint path
TRAIN_PATH=data/CLEANED-SCAN/length_split/increasing_lengths/$TRAIN_MAX_LENGTH/tasks_train.txt
DEV_PATH=data/CLEANED-SCAN/length_split/increasing_lengths/$TRAIN_MAX_LENGTH/tasks_dev.txt
EXPT_DIR=checkpoints_exp_increasing_lengths/train_max_$TRAIN_MAX_LENGTH/run-$CUDA-$RUN_COUNTER

# Start training
echo "Train model with max train length" $TRAIN_MAX_LENGTH
python train_model.py \
    --train "$TRAIN_PATH" \
    --dev "$DEV_PATH" \
    --output_dir "$EXPT_DIR" \
    --epochs $EPOCHS \
    --optim $OPTIMIZER \
    --lr $LR \
    --rnn_cell $RNN_CELL \
    --embedding_size $EMB_SIZE \
    --hidden_size $H_SIZE \
    --n_layers $N_LAYERS \
    --dropout_p_encoder $DROPOUT_ENCODER \
    --dropout_p_decoder $DROPOUT_DECODER \
    --teacher_forcing_ratio $TF \
    --batch_size $BATCH_SIZE \
    $BIDIRECTIONAL \
    $ATTENTION \
    --print_every $PRINT_EVERY \
    --save_every $SAVE_EVERY \
    --cuda_device $CUDA \
    --log-level $LOG_LEVEL \