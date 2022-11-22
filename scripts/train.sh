#!/bin/bash
~/.conda/envs/transformers/bin/python src/train.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --train_file data/squad/train.json \
    --validation_file data/squad/valid.json \
    --output_dir training_output/squad/ \
    --overwrite_output_dir \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --predict_with_generate \
    --num_train_epochs 100 \
    --save_strategy epoch \
    --save_total_limit 11 \
    --load_best_model_at_end=True \
    --evaluation_strategy epoch \
    --metric_for_best_model bleu