#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=run_llama2_hf_13b_chat
#SBATCH --output=run_llama2_hf_13b_chat.out
#SBATCH --mem=80GB
#SBATCH --time=1-12:00
#SBATCH --gres=gpu:a100:1

source ~/.bashrc
conda activate /scratch/as17582/.conda/envs/penv

python run_llama2_hf.py \
    --input_filename=/scratch/as17582/GPT3_Ques_Decomp/jsonl \
    --output_filename=/scratch/as17582/GPT3_Ques_Decomp/model_outputs/llama2_13b_chat \
    --ckpt_dir=/vast/work/public/ml-datasets/llama-2/Llama-2-13b-chat-hf/ \
    --tokenizer_path=/vast/work/public/ml-datasets/llama-2/Llama-2-13b-chat-hf/ \
    --max-gen-len=512 --max_batch_size=1 --temperature=0.0 \
    --stop-word="Q:" --max_examples=1000 --resume=True
