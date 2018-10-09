#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error.txt
#PBS -o output.txt

module load python

cd ~/exp_scan_embeddings

ATTENTION="pre-rnn"
ATTN_METHOD="mlp"
DROPOUT_ENC_DEC=0
METRICS="word_acc seq_acc"

SAVE_EVERY=100
PRINT_EVERY=100
BATCH_SIZE=128
EVAL_BATCH_SIZE=2000
TF=0.5
EPOCHS=30

CUDA=0
REAL_CUDA=0

# MUST BE INITIALIZED..
INIT_EXEC_DEC_WITH=new
INIT_TEMP=5
LEARN_TEMP=no

# BASELINE
MODEL_TYPE=baseline
EMB_SIZE=200
HIDDEN_SIZE=200
DROPOUT=0
N_LAYERS=2
RNN_CELL=gru

ATTN_KEYS=understander_encoder_outputs
ATTN_VALS=understander_encoder_outputs
SAMPLE_TRAIN=full
SAMPLE_INFER=full

FULL_ATTENTION_FOCUS=no

for RUN in 1 2 3 4; do
    for FILLER in 0; do
        DATA_PATH="../machine-tasks/SCAN/CLEANED-SCAN/filler_split"
        TRAIN_PATH="${DATA_PATH}/tasks_train_filler_num${FILLER}.txt"
        VALIDATION_PATH=$TRAIN_PATH
        TEST_PATH="${DATA_PATH}/tasks_test_filler_num${FILLER}.txt"
        MONITOR_DATA="${TEST_PATH}"

        EXPT_DIR="baseline_scan_exp2_${FILLER}_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_L${N_LAYERS}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
        python3 train_model.py --train $TRAIN_PATH --n_layers $N_LAYERS --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type $MODEL_TYPE --full_attention_focus $FULL_ATTENTION_FOCUS > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
    
        CUDA=$((CUDA+1))
        REAL_CUDA=$((REAL_CUDA+1))
        if [ "$CUDA" -eq "4" ]; then
            CUDA=0
        fi
        if [ "$REAL_CUDA" -eq "8" ]; then
            REAL_CUDA=0
            wait
        fi
    done
done

# SEQ2ATTN
MODEL_TYPE=seq2attn
RNN_CELL=gru
EMB_SIZE=512
HIDDEN_SIZE=512
DROPOUT=0.2
N_LAYERS=1

ATTN_KEYS=understander_encoder_outputs
ATTN_VALS=understander_encoder_embeddings

SAMPLE_TRAIN=gumbel_hard
SAMPLE_INFER=argmax
INIT_TEMP=5
LEARN_TEMP=no

FULL_ATTENTION_FOCUS=yes

for RUN in 1 2 3 4; do
    for FILLER in 0; do
        DATA_PATH="../machine-tasks/SCAN/CLEANED-SCAN/filler_split"
        TRAIN_PATH="${DATA_PATH}/tasks_train_filler_num${FILLER}.txt"
        VALIDATION_PATH=$TRAIN_PATH
        TEST_PATH="${DATA_PATH}/tasks_test_filler_num${FILLER}.txt"
        MONITOR_DATA="${TEST_PATH}"

        EXPT_DIR="seq2attn_scan_exp2_${FILLER}_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_L${N_LAYERS}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
        python3 train_model.py --train $TRAIN_PATH --n_layers $N_LAYERS --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type $MODEL_TYPE --full_attention_focus $FULL_ATTENTION_FOCUS > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
    
        CUDA=$((CUDA+1))
        REAL_CUDA=$((REAL_CUDA+1))
        if [ "$CUDA" -eq "4" ]; then
            CUDA=0
        fi
        if [ "$REAL_CUDA" -eq "8" ]; then
            REAL_CUDA=0
            wait
        fi
    done
done

wait