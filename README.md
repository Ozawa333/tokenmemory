# Token Memory
This is the repository for the paper Token Memory Transformer with Infinite Context

# Enviroment 
Please run the following commands to install the environment
```bash
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers
```

# Training and Evaluation
## On pg-19
```bash
accelerate launch --num_processes=3 --mixed_precision='bf16' \
    train_token_llama.py \
    --model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B' \
    --segment_length=4096 \
    --dataset_name='deepmind/pg19' \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --checkpointing_steps='epoch' \
    --num_train_epochs=5 \
    --seed=42 \
    --low_cpu_mem_usage \
    --report_to='wandb' \
    --preprocessing_num_workers=1 \
    --with_tracking \
    --trust_remote_code True \
    --max_train_steps 3040 \
    --learning_rate=1e-3 \
    --output_dir='./checkpoint/pg19_32k_008gra_1e-3' \
    --gradient_accumulation_steps 8 \
    --block_size=32768 \
    --tracking_name 'pg19_32k' \
    --lr_scheduler_type 'cosine'
```bash

## On c4-en
accelerate launch --num_processes=3 --mixed_precision='bf16' \
    train_token_llama.py \
    --model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B' \
    --segment_length=4096 \
    --dataset_name='vllg/looong_c4' \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --checkpointing_steps='epoch' \
    --num_train_epochs=5 \
    --seed=42 \
    --low_cpu_mem_usage \
    --report_to='wandb' \
    --preprocessing_num_workers=16 \
    --with_tracking \
    --trust_remote_code True \
    --max_train_steps 3040 \
    --learning_rate=1e-3 \
    --output_dir='./checkpoint/c4_32k_008gra_1e-3' \
    --gradient_accumulation_steps 8 \
    --block_size=32768 \
    --tracking_name 'c4_32k' \
    --lr_scheduler_type 'cosine'

## On booksum
accelerate launch --num_processes=3 --mixed_precision='bf16' \
    train_token_llama_sum.py \
    --model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B' \
    --segment_length=4096 \
    --dataset_name='kmfoda/booksum' \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --checkpointing_steps='epoch' \
    --num_train_epochs=10 \
    --seed=42 \
    --low_cpu_mem_usage \
    --report_to='wandb' \
    --preprocessing_num_workers=32 \
    --with_tracking \
    --trust_remote_code True \
    --learning_rate=1e-3 \
    --output_dir='./checkpoint/booksum_32k_008gra_1e-3' \
    --gradient_accumulation_steps 8 \
    --block_size=32768 \
    --tracking_name 'booksum_32k' \
    --lr_scheduler_type 'cosine'

