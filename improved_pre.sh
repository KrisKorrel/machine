#! /bin/sh

# NOT USED
CUDA=false
BIDIR=false
PONDERING=false
USE_ATTENTION_LOSS=false

TRAIN=data/lookup-3bit/train.csv
DEV=data/lookup-3bit/validation.csv
OUTPUT_DIR=improved_pre
TEST_PATH1=data/lookup-3bit/test1_heldout.csv
TEST_PATH2=data/lookup-3bit/test2_subset.csv
TEST_PATH3=data/lookup-3bit/test3_hybrid.csv
TEST_PATH4=data/lookup-3bit/test4_unseen.csv
TEST_PATH5=data/lookup-3bit/test5_longer.csv

EPOCHS=200
MAX_LEN=50
RNN_CELL='gru'
EMBEDDING_SIZE=300
HIDDEN_SIZE=300
N_LAYERS=1
DROPOUT_P_ENCODER=0.1
DROPOUT_P_DECODER=0.1
TEACHER_FORCING_RATIO=0
BATCH_SIZE=1
EVAL_BATCH_SIZE=128
OPTIM='sgd'
LR=0.01
SAVE_EVERY=100
PRINT_EVERY=99999999999999999
ATTENTION='pre-rnn'
ATTTENTION_METHOD='hard'

echo "Start training"
python train_model.py \
    --train $TRAIN \
    --dev $TRAIN \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --max_len $MAX_LEN \
    --rnn_cell $RNN_CELL \
    --embedding_size $EMBEDDING_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --n_layers $N_LAYERS \
    --dropout_p_encoder $DROPOUT_P_ENCODER \
    --dropout_p_decoder $DROPOUT_P_DECODER \
    --teacher_forcing_ratio $TEACHER_FORCING_RATIO \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --optim $OPTIM \
    --lr $LR \
    --save_every $SAVE_EVERY \
    --print_every $PRINT_EVERY \
    --attention $ATTENTION \
    --attention_method $ATTTENTION_METHOD \
    --use_input_eos \
    --ignore_output_eos \

echo 'Stop training'
echo 'Start testing'

echo '\nTrain set'
python evaluate.py \
    --checkpoint_path $OUTPUT_DIR/$(ls -t $OUTPUT_DIR/ | head -1) \
    --test_data $TRAIN \
    --max_len $MAX_LEN \
    --batch_size $EVAL_BATCH_SIZE \
    --attention $ATTENTION \
    --attention_method $ATTTENTION_METHOD \
    --use_input_eos \
    --ignore_output_eos \
    --use_attention_loss

echo '\nDev set'
python evaluate.py \
    --checkpoint_path $OUTPUT_DIR/$(ls -t $OUTPUT_DIR/ | head -1) \
    --test_data $DEV \
    --max_len $MAX_LEN \
    --batch_size $EVAL_BATCH_SIZE \
    --attention $ATTENTION \
    --attention_method $ATTTENTION_METHOD \
    --use_input_eos \
    --ignore_output_eos \
    --use_attention_loss

echo '\nTest test1_heldout'
python evaluate.py \
    --checkpoint_path $OUTPUT_DIR/$(ls -t $OUTPUT_DIR/ | head -1) \
    --test_data $TEST_PATH1 \
    --max_len $MAX_LEN \
    --batch_size $EVAL_BATCH_SIZE \
    --attention $ATTENTION \
    --attention_method $ATTTENTION_METHOD \
    --use_input_eos \
    --ignore_output_eos \
    --use_attention_loss

echo '\nTest test2_subset'
python evaluate.py \
    --checkpoint_path $OUTPUT_DIR/$(ls -t $OUTPUT_DIR/ | head -1) \
    --test_data $TEST_PATH2 \
    --max_len $MAX_LEN \
    --batch_size $EVAL_BATCH_SIZE \
    --attention $ATTENTION \
    --attention_method $ATTTENTION_METHOD \
    --use_input_eos \
    --ignore_output_eos \
    --use_attention_loss


echo '\nTest test3_hybrid'
python evaluate.py \
    --checkpoint_path $OUTPUT_DIR/$(ls -t $OUTPUT_DIR/ | head -1) \
    --test_data $TEST_PATH3 \
    --max_len $MAX_LEN \
    --batch_size $EVAL_BATCH_SIZE \
    --attention $ATTENTION \
    --attention_method $ATTTENTION_METHOD \
    --use_input_eos \
    --ignore_output_eos \
    --use_attention_loss

echo '\nTest test4_unseen'
python evaluate.py \
    --checkpoint_path $OUTPUT_DIR/$(ls -t $OUTPUT_DIR/ | head -1) \
    --test_data $TEST_PATH4 \
    --max_len $MAX_LEN \
    --batch_size $EVAL_BATCH_SIZE \
    --attention $ATTENTION \
    --attention_method $ATTTENTION_METHOD \
    --use_input_eos \
    --ignore_output_eos \
    --use_attention_loss

echo '\nTest test5_longer'
python evaluate.py \
    --checkpoint_path $OUTPUT_DIR/$(ls -t $OUTPUT_DIR/ | head -1) \
    --test_data $TEST_PATH5 \
    --max_len $MAX_LEN \
    --batch_size $EVAL_BATCH_SIZE \
    --attention $ATTENTION \
    --attention_method $ATTTENTION_METHOD \
    --use_input_eos \
    --ignore_output_eos \
    --use_attention_loss
