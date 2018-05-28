#! /bin/sh

TRAIN_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/train.tsv
DEV_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/train.tsv
EXPT_DIR=example

# set values
EMB_SIZE=256
H_SIZE=256
N_LAYERS=1
CELL='lstm'
EPOCH=1000
PRINT_EVERY=99999999999
TF=0.5
BS=1024
EBS=1024
ATTN='new'
ATTN_METHOD='mlp'

# Start training
echo "Train model on example data"
python train_model.py --batch_size $BS --eval_batch_size $EBS --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF --bidirectional --attention $ATTN --attention_method $ATTN_METHOD

# echo "Evaluate model on test data"
# python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $DEV_PATH

# echo "Run in inference mode"
# python infer.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) 
