## Description

[This is a work in progress] 

Using Metropolis-Hastings to recover the LLM prompts, given its responses. 
We use BERT based models for the MH proposal distribution. Currently using the RoBERTa model to sample proposal tokens. 

Sequence probabilities were originally computed using GPT3.5, now switched to Llama-2 (7B). 

---------------------------------------------------------------------------------------

### Prerequisites

Since the code is submitted as a job on Greene HPC, please refer to [this Greene singularity setup guide](https://github.com/alexholdenmiller/nyu_cluster) by [@alexholdenmiller](https://github.com/alexholdenmiller).

Then run your singularity container with the command:
```
$ singularity exec --overlay /scratch/netID/PATH/TO/OVERLAY/IMG:rw /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash
```

This code is in Python 3.11, so keep this python version when creating the container.

Inside singularity conda environment (after following the setup), install PyTorch:
```
$ conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

For the remaining libraries: 
```
$ pip install -r requirements.txt
```

The `Llama-2 7b` model used in the code is hosted on NYU Greene HPC at the location:
`/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf`.

### Running the algorithm

The file `mh.py` contains the test code that presently runs 1 example (batch size=1). 

Submit the following command after exiting the singularity container.
```$ sbatch run.sh``` 
to run this as a sbatch job.

*Before submitting*, change the email ID, overlay location and image location to your own. 

Change the GPU/CPU and memory resources as required. 