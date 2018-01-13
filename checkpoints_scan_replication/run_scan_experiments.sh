#! /bin/sh

Note that is used NLLLoss

# Set parameters true for all experiments
EMB_SIZE=50
H_SIZE=50
N_LAYERS=1
CELL='lstm'
EPOCH=30
PRINT_EVERY=50
PLOT_EVERY=-1
SAVE_EVERY=200
PLOT_DIR=embedding_plots
TF=0.5

TRAIN_PATH=data/CLEANED-SCAN/simple_split/tasks_train_simple.txt
DEV_PATH=data/CLEANED-SCAN/simple_split/tasks_test_simple.txt
EXPT_DIR=checkpoints_simple_split
echo "Train model on simple split"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF

TRAIN_PATH=data/CLEANED-SCAN/length_split/tasks_train_length.txt
DEV_PATH=data/CLEANED-SCAN/length_split/tasks_test_length.txt
EXPT_DIR=checkpoints_length_split
echo "Train model on simple split"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF

TRAIN_PATH=data/CLEANED-SCAN/add_prim_split/tasks_train_addprim_jump.txt
DEV_PATH=data/CLEANED-SCAN/add_prim_split/tasks_test_addprim_jump.txt
EXPT_DIR=checkpoints_addprim_jump_split
echo "Train model on simple split"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF

TRAIN_PATH=data/CLEANED-SCAN/add_prim_split/tasks_train_addprim_turn_left.txt
DEV_PATH=data/CLEANED-SCAN/add_prim_split/tasks_test_addprim_turn_left.txt
EXPT_DIR=checkpoints_addprim_turn_left_split
echo "Train model on simple split"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF
