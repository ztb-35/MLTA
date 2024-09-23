#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -p gpu
#SBATCH -n 48#one GPU, n<16
#SBATCH -A hpc_sundeepby4
#SBATCH -o /work/tzhao3/TimeLLM/Reprogramming-multi-level-time-series-forecasting-by-LLMs/job/db_st_m4_out # File name for stdout
#SBATCH -e /work/tzhao3/TimeLLM/Reprogramming-multi-level-time-series-forecasting-by-LLMs/job/db_st_m4_error # File name for error
#SBATCH --mail-type END # Send email when job ends
#SBATCH --mail-user tzhao3@lsu.edu # Send mail to this address
#SBATCH --gres=gpu:2
#job on super mike3

model_name=ST_TimeLLM_3
d_model=32
d_ff=128
train_epochs=50
learning_rate=0.0001
patience=4
llama_layers=6
num_process=2
batch_size=48
eval_batch_size=48
n_heads=8
percent=100
decomp_level=3
decomp_method='moving_avg'
comment='3'

accelerate launch --multi_gpu --num_processes $num_process run_m4.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --data_path ETTh1.csv \
  --model_id M4_512_yearly \
  --model $model_name \
  --data m4 \
  --seasonal_pattern 'Yearly' \
  --features M \
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

accelerate launch --multi_gpu --num_processes $num_process run_m4.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --data_path ETTh1.csv \
  --model_id M4_512_quarterly \
  --model $model_name \
  --data m4 \
  --seasonal_pattern 'Quarterly' \
  --features M \
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

  accelerate launch --multi_gpu --num_processes $num_process run_m4.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --data_path ETTh1.csv \
  --model_id M4_512_monthly \
  --model $model_name \
  --data m4 \
  --seasonal_pattern 'Monthly' \
  --features M \
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

  accelerate launch --multi_gpu --num_processes $num_process run_m4.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --data_path ETTh1.csv \
  --model_id M4_512_weekly \
  --model $model_name \
  --data m4 \
  --seasonal_pattern 'Weekly' \
  --features M \
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

  accelerate launch --multi_gpu --num_processes $num_process run_m4.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --data_path ETTh1.csv \
  --model_id M4_512_daily \
  --model $model_name \
  --data m4 \
  --seasonal_pattern 'Daily' \
  --features M \
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

  accelerate launch --multi_gpu --num_processes $num_process run_m4.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --data_path ETTh1.csv \
  --model_id M4_512_hourly \
  --model $model_name \
  --data m4 \
  --seasonal_pattern 'Hourly' \
  --features M \
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