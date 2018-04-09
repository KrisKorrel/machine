#! /bin/sh

TRAIN_PATH=data/lookup/train.csv
DEV_PATH=data/lookup/unseen.csv
EXPT_DIR=example

# use small parameters for quicker testing
EMB_SIZE=100
H_SIZE=100
CELL2='lstm'
EPOCH=50
CP_EVERY=3000

EX=0
ERR=0

# Start training
echo "Test training"

TF=0.5
CELL='lstm'
ATTN='post-rnn'
ATTN_METHOD='mlp'
BS=10


# test attention loss and pondering
echo "\n\nTest training with attention loss and ponderer"
#python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'post-rnn' --attention_method 'dot' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio 0 --use_attention_loss --pondering --batch_size=2 --scale_attention_loss 1 --optim adam

#python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'post-rnn' --attention_method 'mlp' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio 0 --use_attention_loss --pondering  --batch_size=1 --scale_attention_loss 1 --optim adam

python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio $TF --attention $ATTN --attention_method $ATTN_METHOD --batch_size $BS --remove_output_eos
