#!/bin/bash

rdzv_id=$1
head_node_ip=$2
head_node_port=$3

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$head_node_port
export WORLD_SIZE=$4
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

#echo executing torchrun with node rank $SLURM_NODEID on node $SLURMD_NODENAME
#torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$SLURM_NODEID \
#	--rdzv_id $rdzv_id --rdzv_backend c10d --rdzv_endpoint $head_node_ip:$head_node_port \
#	example_text_completion.py \
#	--ckpt_dir=/vast/work/public/ml-datasets/llama-2/llama-2-70b/ \
#	--tokenizer_path=/vast/work/public/ml-datasets/llama-2/tokenizer.model \
#	--max_seq_len=128 --max_batch_size=1

echo starting worker with RANK $RANK
python run_llama2.py \
	--input_filename=/scratch/ad6489/GPT3_Ques_Decomp/jsonl \
	--output_filename=/scratch/ad6489/GPT3_Ques_Decomp/model_outputs/llama2_70b \
	--ckpt_dir=/vast/work/public/ml-datasets/llama-2/llama-2-70b/ \
	--tokenizer_path=/vast/work/public/ml-datasets/llama-2/tokenizer.model \
	--max_seq_len=4096 --max-gen-len=512 --max_batch_size=1 --temperature=0.0 --resume=True --max_examples=2000
