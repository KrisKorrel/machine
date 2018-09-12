#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error.txt
#PBS -o output.txt

module load python

cd ~/$1


# COMMON PARAMETERS
DROPOUT=0.5
ATTN_METHOD="mlp"
SAVE_EVERY=100
PRINT_EVERY=50
EMB_SIZE=256
HIDDEN_SIZE=256
EVAL_BATCH_SIZE=2000
INIT_TEMP=5
LEARN_TEMP=no
DROPOUT_ENC_DEC=0
CUDA=0
RNN_CELL=gru
ATTENTION="pre-rnn"

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
EPOCHS=100
BATCH_SIZE=1
METRICS="word_acc seq_acc"

MODEL=baseline
SAMPLE_TRAIN=full
SAMPLE_INFER=full
INIT_EXEC_DEC_WITH=new
ATTN_KEYS=understander_encoder_outputs
ATTN_VALS=understander_encoder_outputs


CUDA=0
CUDA_REAL=0

for TEST_GUMBEL in "yes" "no"; do
    for TEST_EMBED in "yes" "no"; do
        for FULL_ATTENTION_FOCUS in "yes" "no"; do
            for TEST_TRANSCODER in "yes" "no"; do

                if [ "$TEST_GUMBEL" == "yes" ]; then
                    SAMPLE_TRAIN=gumbel_hard
                    SAMPLE_INFER=argmax
                else
                    SAMPLE_TRAIN=full
                    SAMPLE_INFER=full
                fi

                if [ "$TEST_EMBED" == "yes" ]; then
                    ATTN_KEYS=understander_encoder_outputs
                    ATTN_VALS=understander_encoder_embeddings
                else
                    ATTN_KEYS=understander_encoder_outputs
                    ATTN_VALS=understander_encoder_outputs
                fi

                if [ "$TEST_TRANSCODER" == "yes" ]; then
                    MODEL=seq2attn
                else
                    MODEL=baseline
                fi


                for RUN in 1 2 3; do
                    EXPT_DIR="gumbel_${TEST_GUMBEL}_embed_${TEST_EMBED}_focus_${FULL_ATTENTION_FOCUS}_transcoder_${TEST_TRANSCODER}_${RNN_CELL}_${ATTN_KEYS}_${ATTN_VALS}_E${EMB_SIZE}_H${HIDDEN_SIZE}_D${DROPOUT}_${SAMPLE_TRAIN}_${SAMPLE_INFER}_${LEARN_TEMP}_${INIT_TEMP}_run_${RUN}"
                    python3 train_model.py --train $TRAIN_PATH --dev $VALIDATION_PATH --monitor $MONITOR_DATA --metrics $METRICS --output_dir $EXPT_DIR --epochs $EPOCHS --rnn_cell $RNN_CELL --embedding_size $EMB_SIZE --hidden_size $HIDDEN_SIZE --dropout_p_encoder $DROPOUT --dropout_p_decoder $DROPOUT --teacher_forcing_ratio $TF --attention $ATTENTION --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --save_every $SAVE_EVERY --print_every $PRINT_EVERY --write-logs "${EXPT_DIR}_LOG" --cuda_device $CUDA --understander_train_method supervised --sample_train $SAMPLE_TRAIN --sample_infer $SAMPLE_INFER --initial_temperature $INIT_TEMP --learn_temperature $LEARN_TEMP --init_exec_dec_with $INIT_EXEC_DEC_WITH --attn_keys $ATTN_KEYS --attn_vals $ATTN_VALS --dropout_enc_dec $DROPOUT_ENC_DEC --model_type $MODEL --full_attention_focus $FULL_ATTENTION_FOCUS > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &
                done

                CUDA=$((CUDA+1))
                CUDA_REAL=$((CUDA_REAL+1))
                if [ "$CUDA" -eq "4" ]; then
                    CUDA=0
                fi

                if [ "$CUDA_REAL" -eq "8" ]; then
                    wait
                fi
            done
        done
    done
done

GENERALPARAMSCRIPT: MODEL, FULLFOCUS
SPARSEMAX

wait
