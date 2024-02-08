## Description

[This is a work in progress.] 

Using Metropolis-Hastings to recover the LLM prompts, given its responses. 
We use BERT based models for the MH proposal distribution. Currently using the RoBERTa model to sample proposal tokens. 

The file `mh.py` contains the test code that runs 1 example. Submit `sbatch run.sh` to run this as a sbatch job.

---------------------------------------------------------------------------------------