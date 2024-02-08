# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import json
import os
from sys import stdout
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, StoppingCriteria

class StopWordCriteria(StoppingCriteria):
    def __init__(self, stops=[]):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids:torch.LongTensor, scores:torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    input_filename: str,
    output_filename: str,
    stop_word: str = None,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_gen_len: int = 256,
    max_batch_size: int = 4,
    resume: bool = True,
    max_examples: int = 100,
):
    print("stop_word: '{}'".format(stop_word))
    print("temperature: {}".format(temperature))
    print("top_p: {}".format(top_p))
    print("max_gen_len: {}".format(max_gen_len))
    print("max_batch_size: {}".format(max_batch_size))
    print("resume: {}".format(resume))
    print("max_examples: {}".format(max_examples))

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(ckpt_dir).to("cuda")

    stopping_criteria = []
    if stop_word != None:
        stop_word_id = tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()
        stopping_criteria.append(StopWordCriteria(stops=[stop_word_id]))

    if input_filename == output_filename:
        raise ValueError("`output_filename` and `input_filename` cannot be the same.")
    if os.path.isfile(input_filename):
        input_files = [input_filename]
    else:
        if os.path.isfile(output_filename):
            raise ValueError("`output_filename` must be a directory if `input_filename` is a directory.")
        contents = os.listdir(input_filename)
        input_files = [input_filename + '/' + f for f in contents]
        input_files = [f for f in input_files if os.path.isfile(f) and f.endswith('.json') or f.endswith('.jsonl')]

    print("input_files: {}".format(input_files))
    for input_file in input_files:
        if os.path.isdir(output_filename):
            output_file = output_filename + '/' + os.path.basename(input_file)
        else:
            output_file = output_filename
        output_resumed = not resume
        print("running experiment")
        print("  input_file: " + input_file)
        print("  output_file: " + output_file)
        stdout.flush()
        with open(input_file) as stream, open(output_file, 'a+' if resume else 'w') as output:
            output.seek(0)
            num_examples = 0
            for line in stream:
                if not output_resumed:
                    resume_position = output.tell()
                    output_line = output.readline()
                    if output_line != None:
                        try:
                            #print("output_line: " + output_line)
                            json.loads(output_line)
                            num_examples += 1
                            if num_examples == max_examples:
                                break
                            continue
                        except ValueError:
                            pass
                    output.seek(resume_position)
                    output_resumed = True
                    print("Resuming from example ID " + str(num_examples))
                    stdout.flush()

                record = json.loads(line)
                prompt = record['prompt']
                #print("\nGenerating for input:")
                #print(prompt)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                if temperature == 0.0:
                    generate_ids = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_gen_len,
                        do_sample=False,
                        num_beams=1,
                        stopping_criteria=stopping_criteria)
                else:
                    generate_ids = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=1,
                        stopping_criteria=stopping_criteria)
                result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                #print("\nOutput:")
                #print(result)
                record['response'] = result
                output.write(json.dumps(record) + '\n')
                output.flush()
                num_examples += 1
                if num_examples == max_examples:
                    break


if __name__ == "__main__":
    fire.Fire(main)
