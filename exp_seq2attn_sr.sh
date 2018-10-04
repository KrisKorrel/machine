#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error.txt
#PBS -o output.txt

module load python

cd ~/exp_seq2attn_sr

FIX NUMBER OR RUNS 10
LSTM OR GRU

# COMMON PARAMETERS
DROPOUT=0.5
ATTENTION="pre-rnn"
ATTN_METHOD="mlp"
SAVE_EVERY=100
PRINT_EVERY=100
EMB_SIZE=512
HIDDEN_SIZE=512
EVAL_BATCH_SIZE=2000
INIT_TEMP=5
LEARN_TEMP=no
DROPOUT_ENC_DEC=0
CUDA=0
FULL_ATTENTION_FOCUS=yes
MODEL_TYPE="seq2attn"

DATA_PATH="../machine-tasks/test_output_symbol_eos"
TRAIN_PATH="${DATA_PATH}/train.tsv"
VALIDATION_PATH="${DATA_PATH}/val_std.tsv"
VALIDATION2_PATH="${DATA_PATH}/val_mix.tsv"
TEST1_PATH="${DATA_PATH}/test_std.tsv"
TEST2_PATH="${DATA_PATH}/test_repeat.tsv"
TEST3_PATH="${DATA_PATH}/test_short.tsv"
TEST4_PATH="${DATA_PATH}/test_long.tsv"
MONITOR_DATA="${VALIDATION_PATH} ${VALIDATION2_PATH} ${TEST1_PATH} ${TEST2_PATH} ${TEST3_PATH} ${TEST4_PATH}"

TF=1
EPOCHS=10
BATCH_SIZE=128

MODEL_TYPE="seq2attn"
SAMPLE_TRAIN=gumbel_hard
SAMPLE_INFER=gumbel_hard
ATTN_KEYS=understander_encoder_outputs
ATTN_VALS=understander_encoder_embeddings
INIT_EXEC_DEC_WITH=new
METRICS="word_acc seq_acc sym_rwr_acc"

for RNN_CELL in gru lstm; do
    for RUN in 1 2; do
        EXPT_DIR="seq2attn_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_fullfocus${FULL_ATTENTION_FOCUS}_run_${RUN}"
        python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --full_attention_focus $FULL_ATTENTION_FOCUS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type seq2attn > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
        CUDA=$((CUDA+1))
        if [ "$CUDA" -eq "4" ]; then
            wait
            CUDA=0
        fi
    done
done

wait
