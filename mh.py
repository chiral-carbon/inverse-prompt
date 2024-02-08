import torch
import torch.nn.functional as F
import os
import openai
import numpy as np
import random
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM,pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_llama = AutoModelForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf")
tokenizer_llama = AutoTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf")
model_llama = model_llama.to(device)

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=False):
    """ Generate a word from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx

def get_init_text(text, tokenizer, model):
    """
    text: a string of text with a question and answer
    tokenizer: a HuggingFace tokenizer
    model: a HuggingFace model

    returns: a string of text generated from masking the question tokens and sampling from the model to fill in the masks
    """
    init_tokens = tokenizer.tokenize(text)
    question_mark_index = init_tokens.index('?')

    for i in range(question_mark_index):
        init_tokens[i] = tokenizer.mask_token
    masked_init_text = tokenizer.convert_tokens_to_string(init_tokens)

    input_encoded = tokenizer(masked_init_text, return_tensors='pt').to(device)
    input_ids = input_encoded["input_ids"]
    output_logits = model(input_ids).logits
    # print(F.softmax(output_logits, dim=-1))

    pred_ids = input_ids.clone()[0].tolist()

    for i in range(question_mark_index):
        pred_ids[i+1] = generate_step(output_logits, i+1, sample=True)
    
    filled_toks = tokenizer.convert_ids_to_tokens(pred_ids[1:-1])
    filled_text = tokenizer.convert_tokens_to_string(filled_toks)

    return filled_toks, filled_text, question_mark_index


def get_gpt_logprobs(sequence):
    """
    Arguments:
      - sequence:
      - question_mark_index: index of '?' in the token list that separates the question and answer tokens

    Returns: 
      - the sum of token logprobs from GPT
    """
    openai.organization = os.environ.get("OPENAI_API_ORG_ID")
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    # prefix = tokenizer.convert_tokens_to_string(tokens[:question_mark_index+1])
    # completion = tokenizer.convert_tokens_to_string(tokens[question_mark_index+1:])
    # sequence = tokenizer.convert_tokens_to_string(tokens)

    response = openai.Completion.create(model='text-davinci-003',
                                        prompt=sequence,
                                        max_tokens=0,
                                        logprobs=1,
                                        echo=True
                                        )
    # print(response.choices[0].logprobs.token_logprobs) 
    return sum(filter(None, response.choices[0].logprobs.token_logprobs))

# def get_logprobs(sequence):
#     inputs = tokenizer_llama(sequence, return_tensors="pt")
#     outputs = model_lp(**inputs, return_dict_in_generate=True, output_scores=True, do_sample=True)
#     transition_scores = model_llama.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
#     # output_length = inputs.input_ids.shape[1] + np.sum(transition_scores.numpy() < 0, axis=1)
#     probabilities = torch.exp(transition_scores.sum(axis=1) )
#     return probabilities.item()

def get_llama2_logprobs(sequence):
    encoded = tokenizer_llama(sequence, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]
    with torch.no_grad():
        model_llama.eval()
        output = model_llama(input_ids=input_ids)
    shift_labels = input_ids[..., 1:].contiguous()
    shift_logits = output.logits[..., :-1, :].contiguous()
    log_probs_tensor = F.log_softmax(shift_logits, dim=-1)  
    log_probs_flat = log_probs_tensor.view(-1, log_probs_tensor.size(-1)) 
    indices = shift_labels.view(-1, 1)
    log_probs_flat_indexed = torch.gather(log_probs_flat, 1, indices)
    log_probs = log_probs_flat_indexed.view(shift_labels.size())

    return torch.sum(log_probs).item()

    # log_probs = {}
    # log_probs[tokenizer_llama.decode(input_ids[0].tolist()[0])] = None
    # for idx, (label_id, logit) in enumerate(zip(shift_labels[0].tolist(), shift_logits[0])):
    #     logprob = F.log_softmax(logit, dim=0).tolist()[label_id]
    #     log_probs[tokenizer_llama.decode(label_id)] = float(logprob)
    # return sum(filter(None, log_probs.values()))  

model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model = model.to(device)
model.eval()


text = "Do the medicine people charge too much money for the special medicine that saves lives? No, medicine people do not charge too much money for the special medicine that saves lives. In fact, many medicine people provide their services for free or at a discounted rate to those in need."
# text = "Is the weather going to bad? No, it is going to be just fine."
filled_toks, filled_text, question_mark_index = get_init_text(text, tokenizer, model)

print("Original Text:", text)
print("Infilled Text:", filled_text)



# MH algorithm
import numpy as np

iter = 1000
top_k = 5
sample = True

# delta
# max_patience

accepted = 0

current_tokens = filled_toks.copy()
current_seq = filled_text

max_seq = {}

for i in range(iter):
    print("Iteration:", i)
    print("Current Sequence: ", current_seq)
    curr_seq_energy = get_llama2_logprobs(current_seq)
    print("Current Sequence Energy: ", curr_seq_energy)

    for t in range(question_mark_index):
        old_token = current_tokens[t]
        encode = tokenizer(current_seq, return_tensors='pt').to(device)
        ids = encode["input_ids"]
        # print(ids)
        # print(t, ids.shape)
        # print(ids[0][t+1], tokenizer.decode(ids[0][t+1]))
        old_token_id = ids[0][t+1].item()
        ids[0][t+1] = tokenizer.mask_token_id
        # print(ids)
        # print(ids[0][t+1], tokenizer.decode(ids[0][t+1]))   

        with torch.no_grad():
            logits = model(ids).logits
        # print(ids.shape, logits.shape, tokenizer.mask_token_id)
        probs = logits[0, t+1, :].softmax(dim=0)
        # print(probs.shape)


        # old_token_id = tokenizer.convert_tokens_to_ids(old_token)
        # print(old_token_id, tokenizer.convert_tokens_to_ids(old_token))
        new_token_id = generate_step(logits, t+1, temperature=1.0, sample=sample).item()
        # print(new_token_id)
        # print(f"Old token: {old_token}, New token: {tokenizer.convert_ids_to_tokens(new_token_id)}")
        old_token_prob = probs[old_token_id].item()
        new_token_prob = probs[new_token_id].item()

        ids[0][t+1] = new_token_id

        proposal_tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist()[1:-1])
        proposal_seq = tokenizer.convert_tokens_to_string(proposal_tokens)
        proposal_seq_energy = get_llama2_logprobs(proposal_seq)

        print(f"Curr seq logprobs: {curr_seq_energy}, Proposal seq logprobs: {proposal_seq_energy}, New token prob: {new_token_prob}, Old token prob: {old_token_prob}")

        u = np.random.uniform(0, 1)
        alpha = min(1, (np.exp(proposal_seq_energy - curr_seq_energy) * (old_token_prob/new_token_prob)))
        if u <= alpha:
            current_seq = proposal_seq
            current_tokens = proposal_tokens.copy()
            curr_seq_energy = proposal_seq_energy
            accepted += 1

        if proposal_seq_energy > curr_seq_energy:
            max_seq[proposal_seq] = proposal_seq_energy
        else:
            max_seq[current_seq] = curr_seq_energy

    print("Proposal Sequence: ", proposal_seq)

    if i % 50 == 0:
        print(f"Acceptance rate after {i+1} iterations: {accepted/((i+1)*question_mark_index)*100} %")


print("Final proposed sequence: ", proposal_seq)
print("Final acceptance rate:", accepted/(iter*question_mark_index)*100, "%")

print("Max energy sequence: ", max(max_seq, key=max_seq.get))