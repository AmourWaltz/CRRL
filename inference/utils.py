import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

import copy

from datetime import datetime
import pytz

import torch


model_dict = {
    "qwen25_7b_it": "Qwen/Qwen2.5-7B-Instruct",
    "qwen25_3b_it": "Qwen/Qwen2.5-3B-Instruct",
    "qwen25_7b_sr1": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3",
    "qwen3_4b_it": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_4b": "Qwen/Qwen3-4B"
}

prompt_dict = {
    "direct": """Answer the given question in /no_think mode. You can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n""",
    "cot": """Answer the given questionin /think mode. You must conduct reasoning inside <think> and </think> to think step by step first. Then you can provide the answer inside <answer> and </answer>. For example, <answer> Beijing </answer>. Question: {question}\n""",
    "search": """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
}


"""
Json utils
"""
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format."""
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def read_json(filename):
    with open(filename, "r", encoding="utf-8") as fr:
        data_pool = json.load(fr)

    return data_pool

def read_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as fr:
        data_pool = [json.loads(line) for line in fr.readlines()]

    return data_pool

def load_json(filename, mode="r"):
    """Load a .jsonl file into a list of dictionaries."""
    with open(filename, "r", encoding="utf-8") as fr:
        data = json.loads(fr.read())
    return data

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')

def write_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as fw:
        fw.writelines([json.dumps(obj=ins, ensure_ascii=False) + '\n' for ins in dataset])

def write_json(filename, dataset):
    with open(filename, "w", encoding="utf-8") as fw:
        json.dump(fp=fw, obj=dataset, indent=4, ensure_ascii=False)

def jsonl2json(file1, file2):
    write_json(file2, read_jsonl(file1))

def json2jsonl(file1, file2):
    dataset = read_json(file1)
    write_jsonl(file2, dataset)

def json_merge(files, out_file):
    data = []
    for file in files:
        data += read_json(file)
    write_json(out_file, data)

def read_jsons(files):
    data = []
    for file in files:
        data += read_json(file)
    return data


# Time utils
def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)  # 1h = 3600s
    minutes, seconds = divmod(remainder, 60)  # 1m = 60s
    return [int(hours), int(minutes), int(seconds)]

def get_current_time():
    tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z%z')

