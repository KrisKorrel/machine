#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error.txt
#PBS -o output.txt

module load python

cd ~/nmt

# COMMON PARAMETERS
DROPOUT=0.2
ATTN_METHOD="mlp"
EMB_SIZE=512
HIDDEN_SIZE=512
INIT_TEMP=5
LEARN_TEMP=no
DROPOUT_ENC_DEC=0
CUDA=0

DATA_PATH="../machine-tasks/NMT/OpenNMT"
TRAIN_PATH="${DATA_PATH}/train.tsv"
VALIDATION_PATH="${DATA_PATH}/validation.tsv"
TEST_PATH="${DATA_PATH}/validation.tsv"
MONITOR_DATA="${TEST_PATH}"

TF=0.5
EPOCHS=150
BATCH_SIZE=64
EVAL_BATCH_SIZE=450
SAVE_EVERY=100
PRINT_EVERY=100
METRICS="word_acc seq_acc"

RNN_CELL=gru

ATTENTION="pre-rnn"
MODEL_TYPE="baseline"
FULL_ATTENTION_FOCUS=yes
SAMPLE_TRAIN=full
SAMPLE_INFER=full
INIT_EXEC_DEC_WITH=encoder
ATTN_KEYS=executor_encoder_outputs
ATTN_VALS=executor_encoder_outputs

for RUN in 1 2; do
    EXPT_DIR="baseline_nmt_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
    python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type $MODEL_TYPE --full_attention_focus $FULL_ATTENTION_FOCUS > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
    CUDA=$((CUDA+1))
    if [ "$CUDA" -eq "4" ]; then
        wait
        CUDA=0
    fi
done

ATTENTION="pre-rnn"
MODEL_TYPE="seq2attn"
FULL_ATTENTION_FOCUS=yes
SAMPLE_TRAIN=gumbel_hard
SAMPLE_INFER=argmax
INIT_EXEC_DEC_WITH=new
ATTN_KEYS=understander_encoder_outputs
ATTN_VALS=understander_encoder_embeddings

for RUN in 1 2; do
    EXPT_DIR="seq2attn_nmt_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
    python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type $MODEL_TYPE --full_attention_focus $FULL_ATTENTION_FOCUS > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
    CUDA=$((CUDA+1))
    if [ "$CUDA" -eq "4" ]; then
        wait
        CUDA=0
    fi
done

wait