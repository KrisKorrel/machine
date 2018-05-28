#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=2:00:00

module load python

cd ~/machine

TRAIN_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/train.tsv
DEV_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/validation.tsv
EXPT_DIR=example

# set values
EMB_SIZE=32
H_SIZE=256
N_LAYERS=1
CELL='gru'
EPOCH=1000
PRINT_EVERY=99999999999999999
TF=0.5

# Start training
echo "Train model on example data"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF --bidirectional --attention 'new' --attention_method 'mlp' > out.txt 2> out2.txt