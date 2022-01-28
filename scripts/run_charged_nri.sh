#!/bin/bash

GPU=0 # Set to whatever GPU you want to use

# Make sure to replace this with the directory containing the data files
DATA_PATH='data/charged_3d_5_unboxed/'

BASE_RESULTS_DIR="results/charged_3d_5_unboxed"

for SEED in {1..5}
do
    MODEL_TYPE="nri"
    EXPERIMENT_EXT="_release"
    WORKING_DIR="${BASE_RESULTS_DIR}/${MODEL_TYPE}${EXPERIMENT_EXT}/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 2 --encoder_hidden 256 --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type ${MODEL_TYPE} --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--no_edge_prior 0.5 --batch_size 128 --lr 5e-4 --use_adam --num_epochs 200 --lr_decay_factor 0.5 --lr_decay_steps 200 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    CUDA_VISIBLE_DEVICES=$GPU python -u locs/experiments/charged_3d_experiment.py \
      --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR \
      $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u locs/experiments/charged_3d_experiment.py \
      --report_error_norm --gpu --load_best_model --test_pred_steps 20 \
      --mode eval --data_path $DATA_PATH --working_dir $WORKING_DIR \
      $MODEL_ARGS $TRAINING_ARGS
    CUDA_VISIBLE_DEVICES=$GPU python -u locs/experiments/charged_3d_experiment.py \
      --report_error_norm --gpu --load_best_model --test_pred_steps 20 \
      --mode eval --data_path $DATA_PATH --working_dir $WORKING_DIR \
      --interactive $MODEL_ARGS $TRAINING_ARGS

    # Uncomment to run inference and save predictions
    # CUDA_VISIBLE_DEVICES=$GPU python -u locs/experiments/charged_3d_experiment.py \
      # --report_error_norm --gpu --load_best_model --test_pred_steps 50 \
      # --mode save_pred --data_path $DATA_PATH --working_dir $WORKING_DIR \
      # $MODEL_ARGS $TRAINING_ARGS
    # CUDA_VISIBLE_DEVICES=$GPU python -u locs/experiments/charged_3d_experiment.py \
      # --report_error_norm --gpu --load_best_model --test_pred_steps 50 \
      # --mode save_pred --interactive --data_path $DATA_PATH --working_dir $WORKING_DIR \
      # $MODEL_ARGS $TRAINING_ARGS
done
