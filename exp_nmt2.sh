#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error.txt
#PBS -o output.txt

module load python

cd ~/exp_nmt2

# COMMON PARAMETERS
DROPOUT=0.2
ATTN_METHOD="mlp"
EMB_SIZE=512
HIDDEN_SIZE=512
INIT_TEMP=5
LEARN_TEMP=no
DROPOUT_ENC_DEC=0
CUDA=0

DATA_PATH="../machine-tasks/LookupTables/lookup-3bit/samples/sample1"
TRAIN_PATH="${DATA_PATH}/validation.tsv"
VALIDATION_PATH="${DATA_PATH}/validation.tsv"
MONITOR_DATA="${VALIDATION_PATH}"

DATA_PATH="../machine-tasks/NMT/OpenNMT"
TRAIN_PATH="${DATA_PATH}/validation.tsv"
VALIDATION_PATH="${DATA_PATH}/validation.tsv"
MONITOR_DATA="${VALIDATION_PATH}"

TF=0.5
EPOCHS=1
BATCH_SIZE=10
EVAL_BATCH_SIZE=10
SAVE_EVERY=10
PRINT_EVERY=100
METRICS="word_acc seq_acc bleu"
LOWER="--lower"

RNN_CELL=lstm

ATTENTION="pre-rnn"
MODEL_TYPE="baseline"
FULL_ATTENTION_FOCUS=no
INIT_EXEC_DEC_WITH=new
ATTN_KEYS=understander_encoder_outputs
ATTN_VALS=understander_encoder_outputs

SAMPLE_TRAIN=full
SAMPLE_INFER=full
EXPT_DIR="baseline"
python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type $MODEL_TYPE --full_attention_focus $FULL_ATTENTION_FOCUS $LOWER
CUDA=$((CUDA+1))

# LOAD_DIR="$(ls -t ${EXPT_DIR}/ | head -2 | tail -1)"
# python3 train_model.py --load_checkpoint $LOAD_DIR --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type $MODEL_TYPE --full_attention_focus $FULL_ATTENTION_FOCUS $LOWER


# ### SEQ2ATTN ###
# ATTENTION="pre-rnn"
# MODEL_TYPE="seq2attn"
# FULL_ATTENTION_FOCUS=no
# INIT_EXEC_DEC_WITH=new
# ATTN_KEYS=understander_encoder_outputs
# ATTN_VALS=understander_encoder_embeddings

# SAMPLE_TRAIN=sparsemax
# SAMPLE_INFER=sparsemax
# EXPT_DIR="seq2attn_nmt_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}"
# python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type $MODEL_TYPE --full_attention_focus $FULL_ATTENTION_FOCUS $LOWER
# CUDA=$((CUDA+1))

# LOAD_DIR="$(ls -t ${EXPT_DIR}/ | head -2 | tail -1)"
# python3 train_model.py --load_checkpoint $LOAD_DIR --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type $MODEL_TYPE --full_attention_focus $FULL_ATTENTION_FOCUS $LOWER

wait