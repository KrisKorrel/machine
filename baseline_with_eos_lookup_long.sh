#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00

cd ~/machine

module load python/3.5.0

DATA_PATH='../machine-tasks/LongLookupTables'

TRAIN_PATH="${DATA_PATH}/train.tsv"
DEV_PATH="${DATA_PATH}/validation.tsv"
TEST_PATH_1="${DATA_PATH}/heldout_compositions.tsv"
TEST_PATH_2="${DATA_PATH}/heldout_inputs.tsv"
TEST_PATH_3="${DATA_PATH}/heldout_tables.tsv"
TEST_PATH_4="${DATA_PATH}/new_compositions.tsv"

TEST_PATH_5="${DATA_PATH}/longer_new_1.tsv"
TEST_PATH_6="${DATA_PATH}/longer_incremental_1.tsv"
TEST_PATH_7="${DATA_PATH}/longer_seen_1.tsv"

TEST_PATH_8="${DATA_PATH}/longer_new_2.tsv"
TEST_PATH_9="${DATA_PATH}/longer_incremental_2.tsv"
TEST_PATH_10="${DATA_PATH}/longer_seen_2.tsv"

TEST_PATH_11="${DATA_PATH}/longer_new_3.tsv"
TEST_PATH_12="${DATA_PATH}/longer_incremental_3.tsv"
TEST_PATH_13="${DATA_PATH}/longer_seen_3.tsv"

TEST_PATH_14="${DATA_PATH}/longer_new_4.tsv"
TEST_PATH_15="${DATA_PATH}/longer_incremental_4.tsv"
TEST_PATH_16="${DATA_PATH}/longer_seen_4.tsv"

TEST_PATH_17="${DATA_PATH}/longer_new_5.tsv"
TEST_PATH_18="${DATA_PATH}/longer_incremental_5.tsv"
TEST_PATH_19="${DATA_PATH}/longer_seen_5.tsv"

# set values
EMB_SIZE=512
H_SIZE=512
N_LAYERS=1
CELL='lstm'
EPOCH=50
PRINT_EVERY=100
SAVE_EVERY=99999999999999999
TF=0
ATTN='pre-rnn'
ATTN_METHOD='mlp'
DROPOUT=0
OPTIM='adam'
LR=0.001
BATCH_SIZE=16
EVAL_BATCH_SIZE=10000
ATTN_SCALE=1

NAME='lookup_long_baseline_with_eos'
SAMPLE_TRAIN='gumbel_hard'
SAMPLE_INFER='gumbel_hard'
INIT_TEMP=2
LEARN_TEMP=unconditioned
KEYS=understander_encoder_outputs
VALS=understander_encoder_embeddings

CUDA=0

for RUN in {1,2,3,4}; do
    EXPT_DIR="${NAME}_E${EMB_SIZE}_H${H_SIZE}_run_${RUN}"
    python3 train_model.py \
        --train $TRAIN_PATH \
        --dev $DEV_PATH \
        --monitor $DEV_PATH $TEST_PATH_1 $TEST_PATH_2 $TEST_PATH_3 $TEST_PATH_4 $TEST_PATH_5 $TEST_PATH_6 $TEST_PATH_7 \
                    $TEST_PATH_8 $TEST_PATH_9 $TEST_PATH_10 $TEST_PATH_11 $TEST_PATH_12 $TEST_PATH_13 $TEST_PATH_14 \
                    $TEST_PATH_15 $TEST_PATH_16 $TEST_PATH_17 $TEST_PATH_18 $TEST_PATH_19 \
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
        --attention_method $ATTN_METHOD \
        --understander_train_method supervised \
        --sample_train $SAMPLE_TRAIN \
        --sample_infer $SAMPLE_INFER \
        --initial_temperature $INIT_TEMP \
        --learn_temperature $LEARN_TEMP \
        --init_exec_dec_with new \
        --train_regime simultaneous \
        --attn_keys $KEYS \
        --attn_vals $VALS \
        > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &

    CUDA=$((CUDA+1))
    if [ "$CUDA" -eq "4" ]; then
       CUDA=0
    fi
done

wait
