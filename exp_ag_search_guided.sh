#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error_learned.txt
#PBS -o output_learned.txt

cd ~/search_learned

ADD K GRAMMAR ACC

module load python

TRAIN_PATH=../machine-tasks/SymbolRewriting/grammar_std.train.full.tsv
DEV_PATH=../machine-tasks/SymbolRewriting/grammar.val.tsv
TEST_PATH_STD=../machine-tasks/SymbolRewriting/grammar_std.tst.full.tsv
TEST_PATH_REPEAT=../machine-tasks/SymbolRewriting/grammar_repeat.tst.full.tsv
TEST_PATH_SHORT=../machine-tasks/SymbolRewriting/grammar_short.tst.full.tsv
TEST_PATH_LONG=../machine-tasks/SymbolRewriting/grammar_long.tst.full.tsv

# set values
N_LAYERS=1
CELL='lstm'
EPOCH=30
PRINT_EVERY=250
SAVE_EVERY=250
TF=1
ATTN='pre-rnn'
ATTN_METHOD='mlp'
DROPOUT=0
OPTIM='adam'
LR=0.001
BATCH_SIZE=128
EVAL_BATCH_SIZE=2000

CUDA=0
REAL_CUDA=0

for RUN in {1,2}; do
    for EMB_SIZE in 32 64; do
        for H_SIZE in {64,128,256}; do
            for ATTN_SCALE in {0.1,1,10}; do
                EXPT_DIR="learned_E${EMB_SIZE}_H${H_SIZE}_SCALE${ATTN_SCALE}_run_${RUN}"

                python train_model.py \
                    --train $TRAIN_PATH \
                    --dev $DEV_PATH \
                    --monitor $DEV_PATH $TEST_PATH_STD $TEST_PATH_REPEAT $TEST_PATH_SHORT $TEST_PATH_LONG \
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
                    --use_attention_loss \
                    --full_focus \
                    > "${EXPT_DIR}_out.txt" 2> "${EXPT_DIR}_err.txt" &

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
        done
    done
done

wait