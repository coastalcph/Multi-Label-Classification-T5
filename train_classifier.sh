MODEL_NAME='t5-base'
BATCH_SIZE=16
DATASET='uklex-l1'
USE_LWAN=false
USE_T5ENC2DEC=true
SEQ2SEQ=false
GEN_MAX_LENGTH=32
T5ENC2DEC_MODE='multi-step'
TRAINING_MODE='t5enc-multi'
OPTIMIZER='adafactor'
SCHEDULER='constant_with_warmup'
LEARNING_RATE=1e-4
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false

for SEED in 21 32 42 84
do
  python experiments/train_classifier.py \
  --model_name_or_path ${MODEL_NAME} \
  --seq2seq ${SEQ2SEQ} \
  --use_lwan ${USE_LWAN} \
  --lwan_version 1 \
  --t5_enc2dec ${USE_T5ENC2DEC} \
  --t5_enc2dec_mode ${T5ENC2DEC_MODE} \
  --dataset_name ${DATASET} \
  --output_dir data/logs/${OPTIMIZER}/${DATASET}/${MODEL_NAME}-${TRAINING_MODE}/fp32/seed_${SEED} \
  --max_seq_length 512 \
  --generation_max_length ${GEN_MAX_LENGTH} \
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
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --seed ${SEED} \
  --warmup_ratio 0.05 \
  --lr_scheduler_type ${SCHEDULER} \
  --optim ${OPTIMIZER} \
  --gradient_accumulation_steps 1 \
  --eval_accumulation_steps 1 \
  --learning_rate ${LEARNING_RATE}
done

python report_dataset_results.py --optimizer ${OPTIMIZER} --fp fp32 --dataset ${DATASET}