# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from tasks.utils import load_model_and_processor
from dataset.mm_dataset import MMDataset

import json
import os
import math
from tqdm import tqdm


ANN_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../data/annotations'


def check_video_path_processed(anns):
    if "<placeholder>" in anns[0]['video_file']:
        return False
    else:
        return True

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument("--max_n_frames", type=int, default=8, help="Max number of frames to apply average sampling from the given video.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--top_p", type=float, default=1, help="Top_p sampling")
    parser.add_argument("--temperature", type=float, default=0, help="Set temperature > 0 to enable sampling generation.")

    parser.add_argument("--input_file", type=str, help="Directory to input_file (jsonline)", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save the model results", required=True)
    parser.add_argument("--output_name", type=str, default="predictions", help="Name of the file for storing results")

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)

    parser.add_argument("--max_n_samples_per_benchmark", type=int, default=-1, help="Set as a small number (like 100) to run as debug.")
    parser.add_argument("--resume", type=lambda x: (str(x).lower() == 'true'), default=True, help="Resume from existing inference results file or overwrite them.")

    args = parser.parse_args()

    return args


def run_inference(args):
    """
    Run inference on selected benchmarks.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model, processor = load_model_and_processor(args.model_name_or_path, args.max_n_frames)

    all_chunks = []
    count = 0
    print(f"Start loading dataset...")
    ann_fpath = args.input_file
    cur_anns = [json.loads(line) for line in open(ann_fpath)]
    if args.max_n_samples_per_benchmark > 0:
        cur_anns = cur_anns[:args.max_n_samples_per_benchmark]
    count += len(cur_anns)
    assert check_video_path_processed(cur_anns), f"You must firstly process the fake \'video_file\' in the annotations to real video path according to the \'vid\'!"
    cur_chunk = get_chunk(cur_anns, args.num_chunks, args.chunk_idx)
    all_chunks.extend(cur_chunk)
    print(f"### Load chunk with {len(cur_chunk)} samples from {len(cur_anns)} samples.")
    print(f"###Finish loading chunk with {len(all_chunks)} samples from {count} samples in total.")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.num_chunks > 1:
        output_name = f"{args.output_name}_{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.jsonl")
    if args.resume and os.path.exists(answers_file):
        processed_data = [json.loads(line) for line in open(answers_file)]
        processed_idxs = set([f"{d['dataset']}-{d['idx']}" for d in processed_data])
        all_chunks = [d for d in all_chunks if f"{d['dataset']}-{d['idx']}" not in processed_idxs]
        print(f"### Resume from {len(processed_idxs)} samples. {len(all_chunks)} samples to run.", flush=True)
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    dataset = MMDataset(
        anns=all_chunks,
        processor=processor
    )

    generate_kwargs = {
        "do_sample": True if args.temperature > 0 else False,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "use_cache": True
    }

    if len(dataset) == 0:
        return
    for ann, inputs in tqdm(dataset, total=len(dataset)):
        if inputs is not None:
            if "prompt" in inputs:
                prompt = inputs.pop('prompt')
                # print(f"Prompt: {prompt}", flush=True)
            # print(f"Input: {processor.tokenizer.decode(inputs['input_ids'][0])}", flush=True)
            try:
                inputs = {k:v.to(model.device) for k,v in inputs.items() if v is not None}
                outputs = model.generate(
                    **inputs,
                    **generate_kwargs,
                )
                output_text = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
            except Exception as e:
                print(f"Error: {e}")
                output_text = "<error>"
            print(f"Prediction: {output_text}", flush=True)
            ann["text"]['prediction'] = output_text
        else:
            ann["text"]['prediction'] = "<error>"
        try:
            ans_file.write(json.dumps(ann, ensure_ascii=False) + "\n")
        except:
            ans_file.write(json.dumps(ann) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
