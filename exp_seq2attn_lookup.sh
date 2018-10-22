#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error.txt
#PBS -o output.txt

module load python

cd ~/exp_seq2attn_lookup

TODO ADD FULL FOCUS AND MODEL TYPE
NOT really checked for correctness

# COMMON PARAMETERS
DROPOUT=0
ATTENTION="pre-rnn"
ATTN_METHOD="mlp"
SAVE_EVERY=100
PRINT_EVERY=100
EMB_SIZE=512
HIDDEN_SIZE=512
EVAL_BATCH_SIZE=2000
INIT_TEMP=5
LEARN_TEMP=unconditioned
DROPOUT_ENC_DEC=0
CUDA=0
RNN_CELL=gru
FULL_ATTENTION_FOCUS=no

# TASK SPECIFIC PARAMETERS
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

TF=0.5
EPOCHS=50
BATCH_SIZE=1
METRICS="word_acc seq_acc"

MODEL_TYPE="seq2attn"
INIT_EXEC_DEC_WITH=new
RNN_CELL=gru
ATTN_KEYS=understander_encoder_outputs
ATTN_VALS=understander_encoder_embeddings
EMB_SIZE=256
HIDDEN_SIZE=256
DROPOUT=0.5
SAMPLE_TRAIN=gumbel_hard
SAMPLE_INFER=argmax
INIT_TEMP=5
LEARN_TEMP=no

FULL_ATTENTION_FOCUS=no
for RUN in 1 2; do
    EXPT_DIR="seq2attn_lookup_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_focus_${FULL_ATTENTION_FOCUS}_run_${RUN}"
    python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --full_attention_focus $FULL_ATTENTION_FOCUS --model_type $MODEL_TYPE > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
    
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

FULL_ATTENTION_FOCUS=yes
for RUN in 1 2; do
    EXPT_DIR="seq2attn_lookup_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_focus_${FULL_ATTENTION_FOCUS}_run_${RUN}"
    python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --full_attention_focus $FULL_ATTENTION_FOCUS --model_type $MODEL_TYPE > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
    
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

SAMPLE_TRAIN=sparsemax
SAMPLE_INFER=sparsemax
FULL_ATTENTION_FOCUS=no
for RUN in 1 2; do
    EXPT_DIR="seq2attn_lookup_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_focus_${FULL_ATTENTION_FOCUS}_run_${RUN}"
    python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --full_attention_focus $FULL_ATTENTION_FOCUS --model_type $MODEL_TYPE > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
    
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

FULL_ATTENTION_FOCUS=yes
for RUN in 1 2; do
    EXPT_DIR="seq2attn_lookup_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_focus_${FULL_ATTENTION_FOCUS}_run_${RUN}"
    python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --full_attention_focus $FULL_ATTENTION_FOCUS --model_type $MODEL_TYPE > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
    
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