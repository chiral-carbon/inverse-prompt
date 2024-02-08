#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --requeue
#SBATCH --time=23:00:00
#SBATCH --mem=48GB
#SBATCH --output=logs/res_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --job-name='mh'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ad6489@nyu.edu

overlay=/scratch/ad6489/pytorch-example/overlay_img
img=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

singularity exec --nv \
	--overlay $overlay:ro \
	$img \
    /bin/bash -c \
	"source /ext3/env.sh; python mh.py" \
    > log.out 2> log.err
