#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -p gpu
#SBATCH -n 64#one GPU, n<16
#SBATCH -A hpc_sunsmic3m
#SBATCH -o /project/tzhao3/Time-LLM-main/job/ettm1_out_3 # File name for stdout
#SBATCH -e /project/tzhao3/Time-LLM-main/job/ettm1_error_3 # File name for error
#SBATCH --mail-type END # Send email when job ends
#SBATCH --mail-user tzhao3@lsu.edu # Send mail to this address
#SBATCH --gres=gpu:4
#job on super mike3

model_name=TimeLLM
train_epochs=100
learning_rate=0.001
llama_layers=6
percent=100
num_process=4
batch_size=24
d_model=32
d_ff=128

comment='TimeLLM-ETTm1'
for pred_len in 96 
do

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process  run_main_1.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --llm_model GPT2 \
  --llm_dim 768 \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj 'type1' \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --percent $percent \
  --model_comment $comment

  done