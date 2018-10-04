#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error.txt
#PBS -o output.txt

cd ~/exp_ag_sr_hard

module load python

DATA_PATH="../machine-tasks/SymbolRewriting"
TRAIN_PATH="${DATA_PATH}/grammar_std.train.full.tsv"
VALIDATION_PATH="${DATA_PATH}/grammar.val.tsv"
TEST1_PATH="${DATA_PATH}/grammar_std.tst.full.tsv"
TEST2_PATH="${DATA_PATH}/grammar_repeat.tst.full.tsv"
TEST3_PATH="${DATA_PATH}/grammar_short.tst.full.tsv"
TEST4_PATH="${DATA_PATH}/grammar_long.tst.full.tsv"
MONITOR_DATA="${VALIDATION_PATH} ${TEST1_PATH} ${TEST2_PATH} ${TEST3_PATH} ${TEST4_PATH}"

TF=1
EPOCHS=30
BATCH_SIZE=128
METRICS="word_acc seq_acc sym_rwr_acc"
EMB_SIZE=32
H_SIZE=256
ATTN_METHOD='hard'

# set values
N_LAYERS=1
CELL='lstm'
PRINT_EVERY=250
SAVE_EVERY=250
ATTN='pre-rnn'
DROPOUT=0
OPTIM='adam'
LR=0.001
EVAL_BATCH_SIZE=2000

CUDA=0
REAL_CUDA=0

for RUN in 1 2 3 4 5 6 7 8 9 10; do
    EXPT_DIR="exp_ag_sr_hard_run_${RUN}"

    python train_model.py \
        --train $TRAIN_PATH \
        --dev $VALIDATION_PATH \
        --monitor $MONITOR_DATA \
        --write-logs "${EXPT_DIR}_LOG" \
        --output_dir $EXPT_DIR \
        --model_type baseline \
        --print_every $PRINT_EVERY \
        --embedding_size $EMB_SIZE \
        --hidden_size $H_SIZE \
        --rnn_cell $CELL \
        --n_layers $N_LAYERS \
        --epoch $EPOCHS \
        --batch_size $BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --print_every $PRINT_EVERY \
        --save_every $SAVE_EVERY \
        --dropout_p_encoder $DROPOUT \
        --dropout_p_decoder $DROPOUT \
        --teacher_forcing $TF \
        --attention $ATTN \
        --attention_method $ATTN_METHOD \
        --sample_train full \
        --sample_infer full \
        --optim $OPTIM \
        --lr $LR \
        --cuda_device $CUDA \
        --full_focus \
        --metrics $METRICS \
        > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &

        CUDA=$((CUDA+1))
        if [ "$CUDA" -eq "4" ]; then
           CUDA=0
        fi

        REAL_CUDA=$((REAL_CUDA+1))
        if [ "$REAL_CUDA" -eq "8" ]; then
           REAL_CUDA=0
           wait
        fi
done

wait