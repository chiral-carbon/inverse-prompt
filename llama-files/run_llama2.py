# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import json
import os
from sys import stdout

from llama import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    input_filename: str,
    output_filename: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 256,
    max_batch_size: int = 4,
    max_examples: int = 1000,
    resume: bool = True
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

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

    if os.environ["RANK"] == "0":
        print("input_files: {}".format(input_files))
    for input_file in input_files:
        if os.path.isdir(output_filename):
            output_file = output_filename + '/' + os.path.basename(input_file)
        else:
            output_file = output_filename
        output_resumed = not resume
        if os.environ["RANK"] == "0":
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
                #if os.environ["RANK"] == "0":
                #    print("\nGenerating for input:")
                #    print(prompt)
                prompt_tokens = generator.tokenizer.encode(prompt, bos=True, eos=False)
                if len(prompt_tokens) > generator.model.params.max_seq_len:
                    if os.environ["RANK"] == "0":
                        print("WARNING: Example {} in input file {} has length {} which is longer than max_seq_len.".format(num_examples, input_file, len(prompt_tokens), generator.model.params.max_seq_len))
                    prompt_tokens = prompt_tokens[-generator.model.params.max_seq_len:]
                generation_tokens, _ = generator.generate(
                    prompt_tokens=[prompt_tokens],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                    logprobs=False,
                    echo=False,
                )
                result = generator.tokenizer.decode(generation_tokens[0])
                if os.environ["RANK"] == "0":
                    #print("\nOutput:")
                    #print(result)
                    record['response'] = result
                    output.write(json.dumps(record) + '\n')
                    output.flush()
                num_examples += 1
                if num_examples == max_examples:
                    break


if __name__ == "__main__":
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        print("RANK not set; setting it from SLURM_PROCID")
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    print("entering main (rank: {})".format(os.environ["RANK"]))
    fire.Fire(main)
