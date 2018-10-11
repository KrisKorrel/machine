#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error.txt
#PBS -o output.txt

module load python

cd ~/exp_ag_scan

# Use the overall best model of original SCAN paper and tests baseline, learned guidance and oracle guidance
# on SCAN experiments 1, 2 and 3a. 

# COMMON
ATTENTION="pre-rnn"
DROPOUT_ENC_DEC=0
METRICS="seq_acc"

SAVE_EVERY=100
PRINT_EVERY=100
BATCH_SIZE=128
EVAL_BATCH_SIZE=2000
TF=0.5
EPOCHS=30

# MUST BE INITIALIZED..
INIT_EXEC_DEC_WITH=new
INIT_TEMP=5
LEARN_TEMP=no

# BASELINE
MODEL_TYPE=baseline
EMB_SIZE=200
HIDDEN_SIZE=200
DROPOUT=0.5
N_LAYERS=2
RNN_CELL=lstm

ATTN_KEYS=understander_encoder_outputs
ATTN_VALS=understander_encoder_outputs
SAMPLE_TRAIN=full
SAMPLE_INFER=full

FULL_ATTENTION_FOCUS=no

CUDA=0
REAL_CUDA=0

for EXPERIMENT in 1 2 3; do
    for MODEL in baseline learned oracle; do
        for RUN in 1 2 3 4 5 6 7 8 9 10; do
            if [ "$EXPERIMENT" == "1" ]; then
                DATA_PATH="../machine-tasks/SCAN/hard_attention/simple_split"
            fi
            if [ "$EXPERIMENT" == "2" ]; then
                DATA_PATH="../machine-tasks/SCAN/hard_attention/length_split"
            fi
            if [ "$EXPERIMENT" == "3" ]; then
                DATA_PATH="../machine-tasks/SCAN/hard_attention/addprim_jump_split"
            fi

            TRAIN_PATH="${DATA_PATH}/train.tsv"
            TEST_PATH="${DATA_PATH}/test.tsv"
            MONITOR_DATA="${TEST_PATH}"

            if [ "$MODEL" == "baseline" ]; then
                ATTN_METHOD=mlp
                EXTRAARG=""
            fi
            if [ "$MODEL" == "learned" ]; then
                ATTN_METHOD=mlp
                EXTRAARG="--use_attention_loss"
            fi
            if [ "$MODEL" == "oracle" ]; then
                ATTN_METHOD=hard
                EXTRAARG=""
            fi


            EXPT_DIR="${MODEL}_SCAN_exp_${EXPERIMENT}_run_${RUN}"
            python3 train_model.py --train $TRAIN_PATH --n_layers $N_LAYERS --dev $TEST_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type $MODEL_TYPE --full_attention_focus $FULL_ATTENTION_FOCUS $EXTRAARG > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &

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
done

wait
