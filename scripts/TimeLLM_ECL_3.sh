#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -p gpu4
#SBATCH -n 64#one GPU, n<16
#SBATCH -A hpc_sunsmic3m
#SBATCH -o /project/tzhao3/TimeLLM_git_clone/Reprogramming-multi-level-time-series-forecasting-by-LLMs/job/timellm_ecl_3_out # File name for stdout
#SBATCH -e /project/tzhao3/TimeLLM_git_clone/Reprogramming-multi-level-time-series-forecasting-by-LLMs/job/timellm_ecl_3_error # File name for error
#SBATCH --mail-type END # Send email when job ends
#SBATCH --mail-user tzhao3@lsu.edu # Send mail to this address
#SBATCH --gres=gpu:4
#job on super mike3

model_name=TimeLLM
d_model=32
d_ff=128
train_epochs=50
seq_len=512
learning_rate=0.0001
patience=3
llama_layers=6
num_process=4
batch_size=48
eval_batch_size=48
n_heads=8
percent=10
decomp_level=1
decomp_method='STL'
comment='1'

accelerate launch --multi_gpu --num_processes $num_process run_main_1.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ECL_512_336 \
  --model $model_name \
  --datasets electricity \
  --target_data electricity \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
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
  --n_heads $n_heads \
  --patience $patience \
  --batch_size $batch_size \
  --eval_batch_size $eval_batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --lradj 'COS' \
  --train_epochs $train_epochs \
  --percent $percent \
  --align_text \
  --decomp_level $decomp_level \
  --decomp_method $decomp_method \
  --combination 'late' \
  --model_comment $comment
