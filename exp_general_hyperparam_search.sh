#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error.txt
#PBS -o output.txt

module load python

cd ~/$1_$2

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
FULL_ATTENTION_FOCUS=yes

TODO ADD FULL FOCUS AND MODEL TYPE


# TASK SPECIFIC PARAMETERS
if [ "$2" == "lookup" ]; then
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
    EPOCHS=100
    BATCH_SIZE=1
    METRICS="word_acc seq_acc"
fi
if [ "$2" == "sr" ]; then
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
    METRICS="word_acc seq_acc sym_rwr_acc"
fi
if [ "$2" == "scan" ]; then
    DATA_PATH="../machine-tasks/test_scan/simple_split"
    TRAIN_PATH="${DATA_PATH}/tasks_train_simple.txt"
    VALIDATION_PATH="${DATA_PATH}/tasks_val_simple.txt"
    TEST_PATH="${DATA_PATH}/tasks_test_simple.txt"
    MONITOR_DATA="${VALIDATION_PATH} ${TEST_PATH}"

    TF=0.5
    EPOCHS=50
    BATCH_SIZE=128
    METRICS="word_acc seq_acc"
fi
if [ "$2" == "nmt" ]; then
    DATA_PATH="../machine-tasks/NMT"
    TRAIN_PATH="${DATA_PATH}/train.tsv"
    VALIDATION_PATH="${DATA_PATH}/test.tsv"
    TEST_PATH="${DATA_PATH}/test.tsv"
    MONITOR_DATA="${TEST_PATH}"

    TF=0.5
    EPOCHS=50
    BATCH_SIZE=128
    METRICS="word_acc seq_acc"
fi

# MODEL SPECIFIC PARAMETERS
if [ "$1" == "baseline" ]; then
    MODEL_TYPE="baseline"
    SAMPLE_TRAIN=full
    SAMPLE_INFER=full
    ATTN_KEYS=understander_encoder_outputs
    ATTN_VALS=understander_encoder_outputs
    INIT_EXEC_DEC_WITH=encoder
fi
if [ "$1" == "seq2attn" ]; then
    MODEL_TYPE="seq2attn"
    SAMPLE_TRAIN=gumbel_hard
    SAMPLE_INFER=gumbel_hard
    ATTN_KEYS=understander_encoder_outputs
    ATTN_VALS=understander_encoder_embeddings
    INIT_EXEC_DEC_WITH=new
fi



if [ "$3" == "rnn_cell" ]; then
    for RNN_CELL in "lstm" "gru"; do
        for RUN in 1 2; do
            EXPT_DIR="$1_$2_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
            python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
            CUDA=$((CUDA+1))
            if [ "$CUDA" -eq "4" ]; then
wait
                CUDA=0
            fi
        done

    done
else
    RNN_CELL=optimal
fi



if [ "$3" == "attn_keys_vals" ]; then
    for ATTN_KEYS in "understander_encoder_outputs" "understander_encoder_embeddings"; do
        for ATTN_VALS in "understander_encoder_embeddings"; do
            for RUN in 1 2; do
                EXPT_DIR="$1_$2_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
                python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
                CUDA=$((CUDA+1))
                if [ "$CUDA" -eq "4" ]; then
wait
                   CUDA=0
                fi
            done
    
        done
    done
else
    ATTN_KEYS=optimal
    ATTN_VALS=optimal
fi




if [ "$3" == "hidden_size" ]; then
    for EMB_SIZE in 32 64 128 256 512 1024; do
        for HIDDEN_SIZE in 32 64 128 256 512 1024; do
            echo $EMB_SIZE
            echo $HIDDEN_SIZE
            if [ "$EMB_SIZE" -gt "$HIDDEN_SIZE" ]; then
                continue
            fi

            for RUN in 1 2; do
                EXPT_DIR="$1_$2_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
                python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
                CUDA=$((CUDA+1))
                if [ "$CUDA" -eq "4" ]; then
wait
                   CUDA=0
                fi
            done
            done
    done
else
    EMB_SIZE=optimal
    HIDDEN_SIZE=optimal
fi





if [ "$3" == "dropout" ]; then
    for DROPOUT in 0 0.2 0.5; do
        for RUN in 1 2; do
            EXPT_DIR="$1_$2_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
            python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
            CUDA=$((CUDA+1))
            if [ "$CUDA" -eq "4" ]; then
wait
               CUDA=0
            fi
        done
    done
else
    DROPOUT=optimal
fi



if [ "$3" == "sample" ]; then
    for SAMPLE_TRAIN in "full" "full_hard" "gumbel_soft" "gumbel_hard"; do
        SAMPLE_INFER=$SAMPLE_TRAIN

        for RUN in 1 2; do
            EXPT_DIR="$1_$2_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
            python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
            CUDA=$((CUDA+1))
            if [ "$CUDA" -eq "4" ]; then
wait
               CUDA=0
            fi
        done
    done

    SAMPLE_TRAIN="full_hard"
    SAMPLE_INFER="argmax"

    for RUN in 1 2; do
        EXPT_DIR="$1_$2_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
        python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
        CUDA=$((CUDA+1))
        if [ "$CUDA" -eq "4" ]; then
wait
           CUDA=0
        fi
    done

    SAMPLE_TRAIN="gumbel_hard"
    SAMPLE_INFER="argmax"

    for RUN in 1 2; do
        EXPT_DIR="$1_$2_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
        python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
        CUDA=$((CUDA+1))
        if [ "$CUDA" -eq "4" ]; then
wait
           CUDA=0
        fi
    done
else
    SAMPLE_TRAIN=optimal
    SAMPLE_INFER=optimal
fi




if [ "$3" == "temp" ]; then
    for INIT_TEMP in 0.5 1 5; do
        for LEARN_TEMP in "no" "unconditioned" "conditioned"; do

            for RUN in 1 2; do
                EXPT_DIR="$1_$2_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
                echo $INIT_TEMP
                echo $EXPT_DIR
                python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
                CUDA=$((CUDA+1))
                if [ "$CUDA" -eq "4" ]; then
wait
                   CUDA=0
                fi
            done
    
        done
    done
else
    INIT_TEMP=optimal
    LEARN_TEMP=optimal
fi

wait