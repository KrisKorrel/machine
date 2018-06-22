#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=0:01:00:00

cd ~/machine

module load python

DATA_PATH='../machine-tasks/test_output_symbol_eos'

TRAIN_PATH="${DATA_PATH}/grammar_std.train.full.tsv"
DEV_PATH="${DATA_PATH}/grammar.val.tsv"
TEST_PATH_STD="${DATA_PATH}/grammar_std.tst.full.tsv"
TEST_PATH_REPEAT="${DATA_PATH}/grammar_repeat.tst.full.tsv"
TEST_PATH_SHORT="${DATA_PATH}/grammar_short.tst.full.tsv"
TEST_PATH_LONG="${DATA_PATH}/grammar_long.tst.full.tsv"

# set values
EMB_SIZE=128
H_SIZE=128
N_LAYERS=1
CELL='lstm'
EPOCH=20
PRINT_EVERY=200
SAVE_EVERY=99999999999999999
TF=1
ATTN='pre-rnn'
ATTN_METHOD='mlp'
DROPOUT=0
OPTIM='adam'
LR=0.001
BATCH_SIZE=125
EVAL_BATCH_SIZE=10000
ATTN_SCALE=1

NAME='baseline_with_eos'

CUDA=0

for RUN in {1,2,3,4}; do
    EXPT_DIR="${NAME}_E${EMB_SIZE}_H${H_SIZE}_run_${RUN}"
    python train_model.py \
        --train $TRAIN_PATH \
        --dev $DEV_PATH \
        --monitor $DEV_PATH $TEST_PATH_STD $TEST_PATH_REPEAT $TEST_PATH_SHORT $TEST_PATH_LONG \
        --write-logs "${EXPT_DIR}_LOG" \
        --output_dir $EXPT_DIR \
        --print_every $PRINT_EVERY \
        --embedding_size $EMB_SIZE \
        --hidden_size $H_SIZE \
        --rnn_cell $CELL \
        --n_layers $N_LAYERS \
        --epoch $EPOCH \
        --batch_size $BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --print_every $PRINT_EVERY \
        --save_every $SAVE_EVERY \
        --dropout_p_encoder $DROPOUT \
        --dropout_p_decoder $DROPOUT \
        --teacher_forcing $TF \
        --attention $ATTN \
        --attention_method $ATTN_METHOD \
        --optim $OPTIM \
        --scale_attention_loss $ATTN_SCALE \
        --lr $LR \
        --cuda_device $CUDA \
        --attention_method mlp \
        --understander_train_method supervised \
        --init_exec_dec_with encoder \
        --train_regime simultaneous \
        --add_k_grammar_metric \
        --remove_eos_k_grammar \
        > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &

    CUDA=$((CUDA+1))
    if [ "$CUDA" -eq "4" ]; then
       CUDA=0
    fi
done

wait
