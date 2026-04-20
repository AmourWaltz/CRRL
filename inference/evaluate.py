# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import os
import json
import csv
import numpy as np
from tabulate import tabulate
import argparse
from openai import OpenAI
from search_r1.utils import *
from datetime import datetime
import tiktoken
from transformers import AutoTokenizer

from verl.utils.reward_score import qa_em

parser = argparse.ArgumentParser(description="Hallucination Generation")

parser.add_argument("--model_id", default="qwen25_3b_it", help="model name")
parser.add_argument("--data_dir", default="./exp/{}_{}/{}_results.jsonl", help="data file path")
# parser.add_argument("--output_dir", default="./exp/{}_T{}_{}", help="output file path")
parser.add_argument("--dataset", default="triviaqa", help="dataset name")
parser.add_argument("--method", default="search", type=str, help="prompt type")
args = parser.parse_args()

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError

def evaluate(data_pool, dataset, model):
    # try:
    #     data_pool = read_json(input_path)
    # except json.JSONDecodeError:
    #     data_pool = read_jsonl(input_path)
    passes = 0

    # import pdb; pdb.set_trace()
    total = len(data_pool)
    total_scores = []
    total_search = []

    for data in data_pool:
        # import pdb; pdb.set_trace()
        # select reward score based on data_source
        ground_truth = data["golden_answers"]
        sequences_str = data["output"]["answer"]
        search_times = data["output"].get("search_times", 0)
        # if search_times == 1:
        #     continue

        compute_score_fn = _select_rm_score_fn(dataset)

        score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=0)
        total_scores.append(score)
        total_search.append(search_times)


    print(f"Model: {model}, Dataset: {dataset}, Method: {args.method}, Score: {np.mean(total_scores)}, Search times: {np.mean(total_search)}, Total: {len(total_search)}")

    # pass_at_n = passes / total
    # pass_at_1 = np.mean(total_scores)
    # avg_length = token_count / total

    # Save metrics to CSV
    # score_path = os.path.join(input_dir, "pass.csv")
    
    # now = datetime.now()
    # formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # # Check if file exists
    # # file_exists = os.path.isfile(score_path)
    
    # # Write to CSV
    # with open(score_path, mode='a', newline='') as f:
    #     writer = csv.DictWriter(f, fieldnames=row_data.keys())
    #     if not file_exists:
    #         writer.writeheader()
    #     writer.writerow(row_data)

    # # Convert the row data into a list of lists format for tabulate
    # table_data = [[k, v] for k, v in row_data.items()]

    # # Print table
    # print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))


if __name__ == '__main__':
    # Load the dataset
    input_path = args.data_dir.format(args.model_id, args.method, args.dataset)
    data_pool = read_jsonl(input_path)
    evaluate(data_pool, args.dataset, args.model_id)
