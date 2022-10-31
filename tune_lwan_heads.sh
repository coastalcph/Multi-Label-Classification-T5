MODEL_NAME='t5-base'
BATCH_SIZE=8
DATASET='uklex-l1'
USE_LWAN=true
SEQ2SEQ=false
TRAINING_MODE='lwan'
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false

for HEADS in 1 4 12
do
  python experiments/train_classifier.py \
  --model_name_or_path ${MODEL_NAME} \
  --seq2seq ${SEQ2SEQ} \
  --use_lwan ${USE_LWAN} \
  --lwan_heads ${HEADS} \
  --dataset_name ${DATASET} \
  --output_dir data/logs/${DATASET}/${MODEL_NAME}-${TRAINING_MODE}/heads_${HEADS} \
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
  --learning_rate 3e-5 \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --seed 32 \
  --warmup_ratio 0.05 \
  --fp16 \
  --fp16_full_eval \
  --lr_scheduler_type cosine
done