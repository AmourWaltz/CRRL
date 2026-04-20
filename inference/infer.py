import transformers
import torch
import random
from datasets import load_dataset
import requests
import pandas as pd
import argparse
from tqdm import tqdm

from search_r1.utils import *
from openai_api import get_response_openai

parser = argparse.ArgumentParser(description="Inference on QA datasets using a search agent.")
parser.add_argument("--input_path", type=str, default="./data/test/test_6k.parquet", help="Path to input dataset file")
parser.add_argument("--save_dir", type=str, default="./exp/{}_{}", help="Local directory to save files")
parser.add_argument("--model_id", type=str, default="qwen25_7b_it", help="Model ID from Hugging Face")
parser.add_argument("--dataset", type=str, default="nq", help="Dataset name, e.g., nq, hotpotqa")
parser.add_argument("--method", type=str, default="search", help="Prompt type to use")
parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device ID")

args = parser.parse_args()


# Model ID and device setup
# model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3"
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer and model
model_name = model_dict[args.model_id]
# model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)

curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\nWhen you decide to output the answer, provide the answer in a brief phrase like <answer> Beijing </answer> or <answer> North America </answer>.\n\n'
# curr_search_template = '\n\n{output_text}<information>{search_results}.'


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def get_answer(text):
    import re
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    if "k1" in args.method:
        payload = {
                "queries": [query],
                "topk": 1,
                "return_scores": True
            }
    elif "k5" in args.method:
        payload = {
                "queries": [query],
                "topk": 5,
                "return_scores": True
            }
    else:
        payload = {
                "queries": [query],
                "topk": 3,
                "return_scores": True
            }
    results = requests.post("http://127.0.0.1:8001/retrieve", json=payload)
    results = results.json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])


# Encode the chat-formatted prompt and move it to the correct device
def generate(input_prompt):
    prompt = input_prompt
    cnt = 0
    while True:
        # print(f"\n\n================== [Round {cnt+1}] ==================\n\n")
        # print(prompt)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # import pdb; pdb.set_trace()
        if outputs[0][-1].item() in curr_eos or cnt >= 8 or get_answer(output_text):
            input_prompt_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
            generated_tokens = outputs[0][input_prompt_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(output_text)
            # break
            return output_text, cnt

        # print(output_text)
        # import pdb; pdb.set_trace()
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print("="*20 + f'\nsearching "{tmp_query}"...\n' + "="*20)
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        # print(search_text)


# Encode the chat-formatted prompt and move it to the correct device
def generate_selfref(input_prompt):
    prompt = input_prompt
    search_docs = []
    extract_docs = []
    cnt = 0
    while True:
        # print(f"\n\n================== [Round {cnt+1}] ==================\n\n")
        # print(prompt)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        # import pdb; pdb.set_trace()
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if outputs[0][-1].item() in curr_eos or cnt >= 8 or get_answer(output_text):
            input_prompt_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
            generated_tokens = outputs[0][input_prompt_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(output_text)
            # break
            return output_text, cnt, search_docs, extract_docs
        
        # print("================== Generated Text: ==================")
        # print(output_text)
        # import pdb; pdb.set_trace()
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print("="*20 + f'\nsearching "{tmp_query}"...\n' + "="*20)
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_docs.append(search_results)
        # print("================== Search Texts: ==================")
        # print(search_results)
        refine_prompt = f"Given the search query {tmp_query}\n\nExtract useful information to the query from the following text:\n\n{search_results} \n\nOnly extract and preserve the critical information and discard the irrelevant content."
        
        refine_input_ids = tokenizer.encode(refine_prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(refine_input_ids)
        
        # Generate text with the stopping criteria
        refine_outputs = model.generate(
            refine_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0
        )

        # import pdb; pdb.set_trace()
        generated_refine_tokens = refine_outputs[0][refine_input_ids.shape[1]:]
        refine_output_text = tokenizer.decode(generated_refine_tokens, skip_special_tokens=True)

        extract_docs.append(refine_output_text)
        # print("================== Extracted Texts: ==================")
        # print(search_results)
        search_text = curr_search_template.format(output_text=output_text, search_results=refine_output_text)
        prompt += search_text
        cnt += 1
        # print(search_text)


# Encode the chat-formatted prompt and move it to the correct device
def generate_info_extract(input_prompt):
    prompt = input_prompt
    search_docs = []
    extract_docs = []
    cnt = 0
    while True:
        # print(f"\n\n================== [Round {cnt+1}] ==================\n\n")
        # print(prompt)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        # import pdb; pdb.set_trace()
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if outputs[0][-1].item() in curr_eos or cnt >= 8 or get_answer(output_text):
            input_prompt_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
            generated_tokens = outputs[0][input_prompt_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(output_text)
            # break
            return output_text, cnt, search_docs, extract_docs
        
        # print("================== Generated Text: ==================")
        # print(output_text)
        # import pdb; pdb.set_trace()
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print("="*20 + f'\nsearching "{tmp_query}"...\n' + "="*20)
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_docs.append(search_results)
        # print("================== Search Texts: ==================")
        # print(search_results)
        search_results = get_response_openai(f"Given the search query {tmp_query}\n\nExtract useful information to the query from the following text:\n\n{search_results} \n\nOnly extract and preserve the critical information and discard the irrelevant content.", instruction="You are a helpful assistant.")

        extract_docs.append(search_results)
        # print("================== Extracted Texts: ==================")
        # print(search_results)
        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        # print(search_text)


# Encode the chat-formatted prompt and move it to the correct device
def generate_compress(input_prompt):
    prompt = input_prompt
    search_docs = []
    extract_docs = []
    cnt = 0
    while True:
        # print(f"\n\n================== [Round {cnt+1}] ==================\n\n")
        # print(prompt)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        # import pdb; pdb.set_trace()
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if outputs[0][-1].item() in curr_eos or cnt >= 8 or get_answer(output_text):
            input_prompt_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
            generated_tokens = outputs[0][input_prompt_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(output_text)
            # break
            return output_text, cnt
        
        # print("================== Generated Text: ==================")
        # print(output_text)
        # import pdb; pdb.set_trace()
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print("="*20 + f'\nsearching "{tmp_query}"...\n' + "="*20)
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_docs.append(search_results)
        # print("================== Search Texts: ==================")
        # print(search_results)
        search_results = get_response_openai(f"Summarize the following texts:\n\n{search_results}", instruction="You are a helpful assistant.")

        extract_docs.append(search_results)
        # print("================== Extracted Texts: ==================")
        # print(search_results)
        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        # print(search_text)


# Encode the chat-formatted prompt and move it to the correct device
def think_generate(input_prompt):
    prompt = input_prompt
    think_lst, search_lst = [], []
    cnt = 0
    while True:
        # print(f"\n\n================== [Round {cnt+1}] ==================\n\n")
        # print(prompt)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        # import pdb; pdb.set_trace()
        if outputs[0][-1].item() in curr_eos or cnt >= 8:
            input_prompt_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
            generated_tokens = outputs[0][input_prompt_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(output_text)
            # break
            return output_text, cnt

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # print(output_text)
        # import pdb; pdb.set_trace()
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print("="*20 + f'\nsearching "{tmp_query}"...\n' + "="*20)
            search_results = search(tmp_query)
        else:
            search_results = ''
        
        # Equal to standard generate with search
        # search_lst.append(search_results)
        # think_lst.append(output_text)
        # history_lst = []
        # for think_text, search_text in zip(think_lst, search_lst):
        #     cur_state = curr_search_template.format(output_text=think_text, search_results=search_text)
        #     history_lst.append(cur_state)

        # prompt = input_prompt + "".join(history_lst)

        search_lst.append(search_results)
        think_lst.append(output_text)
        history = curr_search_template.format(output_text=think_lst[-1], search_results=search_lst[-1])
        prompt = input_prompt + history
        cnt += 1


# Encode the chat-formatted prompt and move it to the correct device
def noquery_generate(input_prompt):
    prompt = input_prompt
    think_lst, search_lst = [], []
    cnt = 0
    while True:
        # print(f"\n\n================== [Round {cnt+1}] ==================\n\n")
        # print(prompt)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        # import pdb; pdb.set_trace()
        if outputs[0][-1].item() in curr_eos or cnt >= 8:
            input_prompt_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
            generated_tokens = outputs[0][input_prompt_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(output_text)
            # break
            return output_text, cnt

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # print(output_text)
        # import pdb; pdb.set_trace()
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print("="*20 + f'\nsearching "{tmp_query}"...\n' + "="*20)
            search_results = search(tmp_query)
        else:
            search_results = ''
        
        # Equal to standard generate with search
        # search_lst.append(search_results)
        # think_lst.append(output_text)
        # history_lst = []
        # for think_text, search_text in zip(think_lst, search_lst):
        #     cur_state = curr_search_template.format(output_text=think_text, search_results=search_text)
        #     history_lst.append(cur_state)

        # prompt = input_prompt + "".join(history_lst)

        output_text = output_text.replace("<search>", "").replace("</search>", "").replace(tmp_query, "")
        search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

        search_text = search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1


# Encode the chat-formatted prompt and move it to the correct device
def lastklg_generate(input_prompt):
    prompt = input_prompt
    think_lst, search_lst = [], []
    cnt = 0
    while True:
        # print(f"\n\n================== [Round {cnt+1}] ==================\n\n")
        # print(prompt)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        # import pdb; pdb.set_trace()
        if outputs[0][-1].item() in curr_eos or cnt >= 8:
            input_prompt_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
            generated_tokens = outputs[0][input_prompt_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(output_text)
            # break
            return output_text, cnt

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # print(output_text)
        # import pdb; pdb.set_trace()
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print("="*20 + f'\nsearching "{tmp_query}"...\n' + "="*20)
            search_results = search(tmp_query)
        else:
            search_results = ''
        
        # Equal to standard generate with search
        # search_lst.append(search_results)
        # think_lst.append(output_text)
        # history_lst = []
        # for think_text, search_text in zip(think_lst, search_lst):
        #     cur_state = curr_search_template.format(output_text=think_text, search_results=search_text)
        #     history_lst.append(cur_state)

        # prompt = input_prompt + "".join(history_lst)

        search_lst.append(search_results)
        think_lst.append(output_text)
        pre_history = "".join(think_lst[:-1] if len(think_lst) > 1 else think_lst)
        history = pre_history + curr_search_template.format(output_text=think_lst[-1], search_results=search_lst[-1])
        prompt = input_prompt + history
        cnt += 1


def quick_test():
    # question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"
    question = "Who was born later, Stephen Sowerby or Just Mathias Thiele?"
    question = question.strip()
    if question[-1] != '?':
        question += '?'

    # Prepare the message
    prompt = f"""Answer the given question. \
    You must conduct reasoning inside <think> and </think> first every time you get new information. \
    After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
    You can search as many times as your want. \
    If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    # lastklg_generate(prompt)
    generate_info_extract(prompt)


def batch_generation():
    dataset = pd.read_parquet(args.input_path)
    save_dir = args.save_dir.format(args.model_id, args.method)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{args.dataset}_results.jsonl")

    records_list = dataset.to_dict(orient="records")
    records_list = [d for d in records_list if d.get("data_source") == args.dataset]

    if os.path.isfile(save_path):
        data_processed = read_jsonl(save_path)
        processed_ids = set([d['id'] for d in data_processed])
        data_list = [d for d in records_list if d['id'] not in processed_ids]
    else:
        data_list = records_list

    prompt_use = prompt_dict["search" if "search" in args.method else args.method]
    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    print(f"Total {len(records_list)} and {len(data_list)} to be processed on {args.dataset} dataset.\n\n")

    # import pdb; pdb.set_trace()
    with tqdm(total=len(data_list)) as t:
        for record in data_list:
            instance = {}
            # question = record.get("prompt", "")[0]["content"]
            question = record.get("question", "")
            question = prompt_use.format(question=question.strip() if question.strip()[-1] == '?' else question.strip() + '?')
            # print(question)
            if "lastklg" in args.method:
                output_text, cnt = lastklg_generate(question)
            elif "extract" in args.method:
                output_text, cnt, search_docs, extract_docs = generate_info_extract(question)
            elif "noquery" in args.method:
                output_text, cnt = noquery_generate(question)
            elif "think" in args.method:
                output_text, cnt = think_generate(question)
            elif "compress" in args.method:
                output_text, cnt = generate_compress(question)
            elif "selfref" in args.method:
                output_text, cnt, search_docs, extract_docs = generate_selfref(question)
            else:
                output_text, cnt = generate(question)
            
            instance = {
                "id": record.get("id"),
                "data_source": record.get("data_source"),
                "question": record.get("question"),
                "golden_answers": record.get("golden_answers").tolist()
            }

            instance['output'] = {
                "generation": output_text,
                "answer": get_answer(output_text),
                "search_docs": search_docs if "extract" in args.method or "selfref" in args.method else [],
                "extract_docs": extract_docs if "extract" in args.method or "selfref" in args.method else [],
                "search_times": cnt
            }

            # print(instance)
            dump_jsonl(instance, save_path, append=True)

            t.set_postfix()
            t.update(1)


if __name__ == "__main__":
    # quick_test()
    batch_generation()