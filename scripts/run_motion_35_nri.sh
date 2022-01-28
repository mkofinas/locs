#!/bin/bash

GPU=0 # Set to whatever GPU you want to use

# First: process data

# Make sure to replace this with the directory containing the data files
IN_DIR='data/motion/raw/35/'
DATA_PATH='data/motion/processed/35'
mkdir -p $DATA_PATH
# python -u locs/datasets/cmu_motion_data.py --data_path $IN_DIR --out_path $DATA_PATH

BASE_RESULTS_DIR="results/motion_35"

for SEED in {1..5}
do
    MODEL_TYPE="nri"
    EXPERIMENT_EXT='_release'
    WORKING_DIR="${TMP_BASE_RESULTS_DIR}/${MODEL_TYPE}${EXPERIMENT_EXT}/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 4 --encoder_hidden 256 --skip_first --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type $MODEL_TYPE --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--batch_size 8 --lr 5e-4 --use_adam --num_epochs 600 --lr_decay_factor 0.5 --lr_decay_steps 300 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing'
    mkdir -p $WORKING_DIR
    CUDA_VISIBLE_DEVICES=$GPU python -u locs/experiments/motion_experiment.py \
      --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR \
      $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u locs/experiments/motion_experiment.py \
      --gpu --report_error_norm --mode eval --load_best_model \
      --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS \
      $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
    MODEL_ARGS="--model_type $MODEL_TYPE --graph_type dynamic ${ENCODER_ARGS} ${DECODER_ARGS}"
    CUDA_VISIBLE_DEVICES=$GPU python -u locs/experiments/motion_experiment.py \
      --gpu --report_error_norm --mode eval --load_best_model \
      --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS \
      $TRAINING_ARGS --error_out_name prediction_errors_dynamic.npy |& tee "${WORKING_DIR}eval_results_dynamic.txt"
done
