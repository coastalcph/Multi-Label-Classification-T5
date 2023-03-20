#!/bin/bash
#SBATCH --job-name=bert-uklex-l1
#SBATCH --cpus-per-task=8 --mem=8000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --output=/home/rwg642/MultiLabelConditionalGeneration/bert-uklex-l1.txt
#SBATCH --time=6:00:00

module load miniconda/4.12.0
conda activate kiddothe2b

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES

MODEL_NAME='bert-base-cased'
BATCH_SIZE=16
DATASET='uklex-l1'
SCHEDULER='cosine'
LEARNING_RATE=3e-5

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

for SEED in 21 32 42 84
do
  python experiments/train_classifier.py \
  --model_name_or_path ${MODEL_NAME} \
  --seq2seq false \
  --use_lwan false \
  --lwan_version 1 \
  --t5_enc2dec false \
  --dataset_name ${DATASET} \
  --output_dir data/logs/adam_w/${DATASET}/${MODEL_NAME}/fp32/seed_${SEED} \
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
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --seed ${SEED} \
  --warmup_ratio 0.05 \
  --lr_scheduler_type linear \
  --gradient_accumulation_steps 1 \
  --eval_accumulation_steps 1 \
  --learning_rate ${LEARNING_RATE}
done