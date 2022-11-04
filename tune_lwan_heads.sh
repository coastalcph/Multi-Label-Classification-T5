MODEL_NAME='t5-base'
BATCH_SIZE=16
DATASET='uklex-l1'
USE_LWAN=true
GEN_MAX_LENGTH=32
TRAINING_MODE='lwan-v2'
OPTIMIZER='adamw_torch'
LEARNING_RATE=3e-5
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

for HEADS in 4 6 12
do
  python experiments/train_classifier.py \
  --model_name_or_path ${MODEL_NAME} \
  --use_lwan ${USE_LWAN} \
  --lwan_heads ${HEADS} \
  --dataset_name ${DATASET} \
  --output_dir data/logs/${OPTIMIZER}/${DATASET}/${MODEL_NAME}-${TRAINING_MODE}-heads-${HEADS}/seed_42 \
  --max_seq_length 512 \
  --do_train \
  --do_eval \
  --do_pred \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model micro-f1 \
  --greater_is_better True \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 5 \
  --num_train_epochs 20 \
  --learning_rate ${LEARNING_RATE} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --seed 84 \
  --optim ${OPTIMIZER} \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine
done