MODEL_NAME='t5-base'
BATCH_SIZE=16
DATASET='uklex-l1'
GEN_MAX_LENGTH=32
TRAINING_MODE='seq2seq'
OPTIMIZER='adafactor'
LEARNING_RATE=1e-4
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=5
export TOKENIZERS_PARALLELISM=false

for LABEL_DESC_TYPE in original numbers simplified
do
  for SEED in 21 32 42 84
  do
    python experiments/train_classifier.py \
    --model_name_or_path ${MODEL_NAME} \
    --seq2seq true \
    --label_descriptors_mode ${LABEL_DESC_TYPE} \
    --dataset_name ${DATASET} \
    --output_dir data/logs/${OPTIMIZER}/${DATASET}/${MODEL_NAME}-${TRAINING_MODE}-${LABEL_DESC_TYPE}/fp32/seed_${SEED} \
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
    --seed ${SEED} \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 2 \
    --optim ${OPTIMIZER} \
    --warmup_ratio 0.05 \
    --lr_scheduler_type constant_with_warmup
  done
done

python report_label_desc_results.py