#!/bin/bash

ATTENTION=true
CUDA=$1

# set values
EPOCHS=1
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
BIDIRECTIONAL=false
PRINT_EVERY=20
SAVE_EVERY=131

while true
do
    # Generate random number in [2, 10] (inclusive) (number of output lengths that are deleted)
    NUMBER_OF_LENGTHS=$[ ( $RANDOM % 9 )  + 2 ]

    LENGHTS_TO_LEAVE_OUT=()
    for i in $(seq 1 $NUMBER_OF_LENGTHS)
    do
        # Generate random number in [2-21] (output length to leave out)
        LEAVE_OUT_LENGTH=$[ ( $RANDOM % 20 )  + 2 ]
        # Add it to array
        LENGHTS_TO_LEAVE_OUT+=($LEAVE_OUT_LENGTH)
    done

    # Call script to split data
    python create_length_splits.py --leave_outs ${LENGHTS_TO_LEAVE_OUT[*]}

    # Define the train data and checkpoint path
    TRAIN_PATH=data/CLEANED-SCAN/length_split/${LENGHTS_TO_LEAVE_OUT[*]}/tasks_train.txt
    EXPT_DIR=checkpoints_experiment_random_leave_out/${LENGHTS_TO_LEAVE_OUT[*]}
    DEV_PATH=data/CLEANED-SCAN/length_split/tasks_test_length.txt

    # Start training
    echo "Train model with leave out" ${LENGHTS_TO_LEAVE_OUT[*]}
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
        $( (( $BIDIRECTIONAL )) && echo '--bidirectional' ) \
        $( (( $ATTENTION )) && echo '--attention' ) \
        --print_every $PRINT_EVERY \
        --save_every $SAVE_EVERY \
        --cuda_device $CUDA

    echo 'Test on length > 22'
    DEV_PATH=data/CLEANED-SCAN/length_split/tasks_test_length.txt
    python evaluate.py --checkpoint_path "$EXPT_DIR"/$(ls -t "$EXPT_DIR"/ | head -1) --test_data "$DEV_PATH" --cuda_device $CUDA

    for test_length in 24 25 26 27 28 30 32 33 36 40 48
    do
        echo 'Test on length '$test_length
        DEV_PATH=data/CLEANED-SCAN/length_split/exp_2_test_$test_length/tasks_train.txt
        python evaluate.py --checkpoint_path "$EXPT_DIR"/$(ls -t "$EXPT_DIR"/ | head -1) --test_data "$DEV_PATH" --cuda_device $CUDA
    done

    echo ''
    echo ''
    echo ''

done
