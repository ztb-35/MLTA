#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -p gpu
#SBATCH -n 48#one GPU, n<16
#SBATCH -A hpc_sundeepby4
#SBATCH -o /work/tzhao3/TimeLLM/Reprogramming-multi-level-time-series-forecasting-by-LLMs/job/db_st_traffic_out_3 # File name for stdout
#SBATCH -e /work/tzhao3/TimeLLM/Reprogramming-multi-level-time-series-forecasting-by-LLMs/job/db_st_traffic_error_3 # File name for error
#SBATCH --mail-type END # Send email when job ends
#SBATCH --mail-user tzhao3@lsu.edu # Send mail to this address
#SBATCH --gres=gpu:2
#job on super mike3

model_name=ST_TimeLLM_1
d_model=32
d_ff=128
train_epochs=50
seq_len=512
learning_rate=0.0001
patience=3
llama_layers=6
num_process=2
batch_size=24
eval_batch_size=24
n_heads=8
percent=100
decomp_level=3
decomp_method='STL'
comment='1'

accelerate launch --multi_gpu --num_processes $num_process run_main_1.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id traffic_512_336 \
  --model $model_name \
  --datasets traffic \
  --target_data traffic \
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
