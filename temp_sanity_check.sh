#! /bin/sh

TRAIN_PATH=test/test_data/train.txt
DEV_PATH=test/test_data/dev.txt
EXPT_DIR=example

# set values
EMB_SIZE=16
H_SIZE=32
N_LAYERS=1
CELL='gru'
EPOCH=1
PRINT_EVERY=5
SAVE_EVERY=5
TF=0.5

### TODO ###
MODEL_TYPE=baseline
SAMPLE_TRAIN=full
SAMPLE_INFER=full

echo "Train model on example data"
python train_model.py \
    --train $TRAIN_PATH \
    --batch_size 1024 \
    --output_dir $EXPT_DIR \
    --embedding_size $EMB_SIZE \
    --hidden_size $H_SIZE \
    --rnn_cell $CELL \
    --n_layers $N_LAYERS \
    --epoch $EPOCH \
    --print_every $PRINT_EVERY \
    --save_every $SAVE_EVERY \
    --teacher_forcing $TF \
    --attention 'pre-rnn' \
    --attention_method 'mlp' \
    --model_type $MODEL_TYPE \
    --sample_train $SAMPLE_TRAIN \
    --sample_infer $SAMPLE_INFER

MODEL_TYPE=seq2attn
SAMPLE_TRAIN=gumbel_hard
SAMPLE_INFER=argmax
ATTN_KEYS=outputs
ATTN_VALS=embeddings
INIT_TEMP=5
LEARN_TEMP=unconditioned

echo "Train model on example data"
python train_model.py \
    --train $TRAIN_PATH \
    --batch_size 1024 \
    --output_dir $EXPT_DIR \
    --embedding_size $EMB_SIZE \
    --hidden_size $H_SIZE \
    --rnn_cell $CELL \
    --n_layers $N_LAYERS \
    --epoch $EPOCH \
    --print_every $PRINT_EVERY \
    --save_every $SAVE_EVERY \
    --teacher_forcing $TF \
    --attention 'pre-rnn' \
    --attention_method 'mlp' \
    --model_type $MODEL_TYPE \
    --sample_train $SAMPLE_TRAIN \
    --sample_infer $SAMPLE_INFER \
    --attn_keys $ATTN_KEYS \
    --attn_vals $ATTN_VALS \
    --initial_temperature $INIT_TEMP \
    --learn_temperature $LEARN_TEMP