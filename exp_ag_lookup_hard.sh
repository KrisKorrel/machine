#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error_hard.txt
#PBS -o output_hard.txt

cd ~/exp_ag_lookup_hard

module load python

DATA_PATH="../machine-tasks/LookupTables/lookup-3bit/samples/sample1"
TRAIN_PATH="${DATA_PATH}/train.tsv"
VALIDATION_PATH="${DATA_PATH}/validation.tsv"
TEST1_PATH="${DATA_PATH}/heldout_compositions.tsv"
TEST2_PATH="${DATA_PATH}/heldout_inputs.tsv"
TEST3_PATH="${DATA_PATH}/heldout_tables.tsv"
TEST4_PATH="${DATA_PATH}/new_compositions.tsv"
TEST5_PATH="${DATA_PATH}/longer_compositions_incremental.tsv"
TEST6_PATH="${DATA_PATH}/longer_compositions_new.tsv"
TEST7_PATH="${DATA_PATH}/longer_compositions_seen.tsv"
TEST8_PATH="${DATA_PATH}/heldout_tables_sn.tsv"
TEST9_PATH="${DATA_PATH}/heldout_tables_ns.tsv"
MONITOR_DATA="${VALIDATION_PATH} ${TEST1_PATH} ${TEST2_PATH} ${TEST3_PATH} ${TEST4_PATH} ${TEST5_PATH} ${TEST6_PATH} ${TEST7_PATH} ${TEST8_PATH} ${TEST9_PATH}"

TF=0
EPOCHS=100
BATCH_SIZE=1
METRICS="word_acc seq_acc"
EMB_SIZE=128
H_SIZE=512
ATTN_METHOD='hard'

# set values
N_LAYERS=1
CELL='gru'
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
    EXPT_DIR="exp_ag_lookup_hard_run_${RUN}"

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