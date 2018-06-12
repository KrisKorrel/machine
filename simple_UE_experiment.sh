#! /bin/sh

# NOT USED
CUDA=false
BIDIR=false
PONDERING=false
USE_ATTENTION_LOSS=false

DATASETS_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1
TRAIN="${DATASETS_PATH}/train.tsv"
DEV="${DATASETS_PATH}/validation.tsv"
TEST_PATH1="${DATASETS_PATH}/heldout_inputs.tsv"
TEST_PATH2="${DATASETS_PATH}/heldout_tables.tsv"
TEST_PATH3="${DATASETS_PATH}/longer_compositions_seen.tsv"
TEST_PATH4="${DATASETS_PATH}/longer_compositions_incremental.tsv"
TEST_PATH5="${DATASETS_PATH}/longer_compositions_new.tsv"
OUTPUT_DIR=example

MAX_LEN=50
RNN_CELL='gru'
EMBEDDING_SIZE=256
HIDDEN_SIZE=256
N_LAYERS=1
DROPOUT_P_ENCODER=0
DROPOUT_P_DECODER=0
TEACHER_FORCING_RATIO=0
BATCH_SIZE=1
EVAL_BATCH_SIZE=1024
OPTIM='adam'
LR=0.001
SAVE_EVERY=9999999999999999
PRINT_EVERY=99999999999999
ATTENTION='pre-rnn'
ATTTENTION_METHOD='hard'

EPOCHS=2000 # first 50% of epochs, only the executor is trained with hard guidance. Second half, the understander is trained
GAMMA=0.1 # Discount factor for rewards. Since we don't have sparse rewards, we can keep this low
EPSILON=0.99 # Sample stochastically from policy 99% of times, sample unifomly 1%
TRAIN_METHOD='supervised' # Train understander with either 'supervised' or 'rl'
SAMPLE_TRAIN='gumbel_hard' # In supervised setting we can either use the 'full' attention vector, sample using 'gumbel_soft', or sample using gumbel ST ('gumbel_hard')
SAMPLE_INFER='gumbel_hard' # In supervised setting we can either use the 'full' attention vector, sample using 'gumbel_soft', sample using gumbel ST ('gumbel_hard'),  or use 'argmax' ar inference
INIT_TEMP=5 # (Initial) temperature for gumbel-softmax
LEARN_TEMP='unconditioned' # Fix temperature with 'no', make it a latent, unconditioned, learnable parameter with 'unconditioned', learn it conditioned on encoder-decoder concatenation with 'conditioned'
INIT_EXEC_DEC_WITH='new' # Initialize the executor's decoder with it's last encoder, or with a new learable vector
TRAIN_REGIME='simultaneous' # In 'two-stage' training we first train the executor with hard guidance for n/2 epochs and then the understander for n/2 epochs
                            # In 'simultaneous' training, we train both models together without any supervision on the attention.

# The understander will compute the attention scores based on a concatenation of the decoder hidden states with the 'keys'
# The keys can be: 'understander_encoder_embeddings', 'understander_encoder_outputs', 'executor_encoder_embeddings', 'executor_encoder_outputs'
ATTN_KEYS='understander_encoder_outputs'
# With the attention scores/probs, the executor will create a context vector as a weightes averages over the 'values'
# The vals can be: 'understander_encoder_embeddings', 'understander_encoder_outputs', 'executor_encoder_embeddings', 'executor_encoder_outputs'
ATTN_VALS='understander_encoder_embeddings'

echo "Start training"
python train_model.py \
    --train $TRAIN \
    --pre_train $TRAIN \
    --dev $DEV \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --max_len $MAX_LEN \
    --rnn_cell $RNN_CELL \
    --embedding_size $EMBEDDING_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --n_layers $N_LAYERS \
    --dropout_p_encoder $DROPOUT_P_ENCODER \
    --dropout_p_decoder $DROPOUT_P_DECODER \
    --teacher_forcing_ratio $TEACHER_FORCING_RATIO \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --optim $OPTIM \
    --lr $LR \
    --save_every $SAVE_EVERY \
    --print_every $PRINT_EVERY \
    --attention $ATTENTION \
    --attention_method $ATTTENTION_METHOD \
    --gamma $GAMMA \
    --epsilon $EPSILON \
    --understander_train_metho $TRAIN_METHOD \
    --sample_train $SAMPLE_TRAIN \
    --sample_infer $SAMPLE_INFER \
    --initial_temperature $INIT_TEMP \
    --init_exec_dec_with $INIT_EXEC_DEC_WITH \
    --train_regime $TRAIN_REGIME \
    --learn_temperature $LEARN_TEMP \
    --attn_keys $ATTN_KEYS \
    --attn_vals $ATTN_VALS