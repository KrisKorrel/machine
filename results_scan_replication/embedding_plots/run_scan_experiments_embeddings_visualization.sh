#! /bin/sh

# Set parameters true for all experiments
EMB_SIZE=2
H_SIZE=50
N_LAYERS=1
CELL='lstm'
EPOCH=10
PRINT_EVERY=50
PLOT_EVERY=25
SAVE_EVERY=2000
TF=0.5

TRAIN_PATH=data/CLEANED-SCAN/simple_split/tasks_train_simple.txt
DEV_PATH=data/CLEANED-SCAN/simple_split/tasks_test_simple.txt
EXPT_DIR=checkpoints_length_split
PLOT_DIR=embedding_plots_simple_split_1
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF
PLOT_DIR=embedding_plots_simple_split_2
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF
PLOT_DIR=embedding_plots_simple_split_3
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF

TRAIN_PATH=data/CLEANED-SCAN/length_split/tasks_train_length.txt
DEV_PATH=data/CLEANED-SCAN/length_split/tasks_test_length.txt
EXPT_DIR=checkpoints_length_split
PLOT_DIR=embedding_plots_length_split_1
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF
PLOT_DIR=embedding_plots_length_split_2
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF
PLOT_DIR=embedding_plots_length_split_3
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF

TRAIN_PATH=data/CLEANED-SCAN/add_prim_split/tasks_train_addprim_jump.txt
DEV_PATH=data/CLEANED-SCAN/add_prim_split/tasks_test_addprim_jump.txt
EXPT_DIR=checkpoints_length_split
PLOT_DIR=embedding_plots_addprim_jump_split_1
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF
PLOT_DIR=embedding_plots_addprim_jump_split_2
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF
PLOT_DIR=embedding_plots_addprim_jump_split_3
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF

TRAIN_PATH=data/CLEANED-SCAN/add_prim_split/tasks_train_addprim_turn_left.txt
DEV_PATH=data/CLEANED-SCAN/add_prim_split/tasks_test_addprim_turn_left.txt
EXPT_DIR=checkpoints_length_split
PLOT_DIR=embedding_plots_addprim_turn_left_split_1
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF
PLOT_DIR=embedding_plots_addprim_turn_left_split_2
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF
PLOT_DIR=embedding_plots_addprim_turn_left_split_3
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --save_every $SAVE_EVERY --plot_every $PLOT_EVERY --plot_dir $PLOT_DIR --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF
