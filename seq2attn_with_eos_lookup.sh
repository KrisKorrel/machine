#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00

cd ~/machine

module load python

DATA_PATH='../machine-tasks/test_output_lookup_eos/lookup-3bit/samples/sample1'

TRAIN_PATH="${DATA_PATH}/train.tsv"
DEV_PATH="${DATA_PATH}/validation.tsv"
TEST_PATH_1="${DATA_PATH}/heldout_compositions.tsv"
TEST_PATH_2="${DATA_PATH}/heldout_inputs.tsv"
TEST_PATH_3="${DATA_PATH}/heldout_tables.tsv"
TEST_PATH_4="${DATA_PATH}/new_compositions.tsv"
TEST_PATH_5="${DATA_PATH}/longer_compositions_new.tsv"
TEST_PATH_6="${DATA_PATH}/longer_compositions_incremental.tsv"
TEST_PATH_7="${DATA_PATH}/longer_compositions_seen.tsv"

# set values
EMB_SIZE=512
H_SIZE=512
N_LAYERS=1
CELL='lstm'
EPOCH=200
PRINT_EVERY=50
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

NAME='lookup_seq2attn_with_eos'
SAMPLE_TRAIN='gumbel_hard'
SAMPLE_INFER='gumbel_hard'
INIT_TEMP=2
LEARN_TEMP=unconditioned
KEYS=understander_encoder_outputs
VALS=understander_encoder_embeddings
# NOT USED:
# DROPOUT_ENC_DEC=0.
# ADD_K_GRAMMAR_METRIC
# INIT_EXEC_WITH
# INPUT_EOS_K_GRAMMAR
# IGNORE_OUTPUT_EOS

CUDA=0

for RUN in {1,2,3,4}; do
    EXPT_DIR="${NAME}_E${EMB_SIZE}_H${H_SIZE}_run_${RUN}"
    python train_model.py \
        --train $TRAIN_PATH \
        --dev $DEV_PATH \
        --monitor $DEV_PATH $TEST_PATH_1 $TEST_PATH_2 $TEST_PATH_3 $TEST_PATH_4 $TEST_PATH_5 $TEST_PATH_6 $TEST_PATH_7 \
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
        --ignore_output_eos \
        > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &

    CUDA=$((CUDA+1))
    if [ "$CUDA" -eq "4" ]; then
       CUDA=0
    fi
done

wait
