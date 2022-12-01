MODEL_NAME='t5-base'
BATCH_SIZE=64
GEN_MAX_LENGTH=128
TRAINING_MODE='seq2seq'
OPTIMIZER='adafactor'
LEARNING_RATE=1e-4
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false

for DATASET in uklex-l1
do
  for SEED in 21 32 42 84
  do
    python experiments/train_classifier.py \
    --model_name_or_path data/logs/${OPTIMIZER}/${DATASET}/${MODEL_NAME}-${TRAINING_MODE}-original/fp32/seed_${SEED} \
    --seq2seq true \
    --label_descriptors_mode original \
    --dataset_name ${DATASET} \
    --output_dir data/logs/${OPTIMIZER}/${DATASET}/${MODEL_NAME}-${TRAINING_MODE}-original-greedy/fp32/seed_${SEED} \
    --max_seq_length 512 \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --n_beams  1 \
    --do_eval \
    --do_pred \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --seed ${SEED}
  done
done

python table_dataset_results.py