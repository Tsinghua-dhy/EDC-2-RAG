import torch
import json
from tqdm import tqdm
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, "/your_path/codes")
from utils import GPT_Instruct_request, GPT4omini_request, ChatGPT_request, llama3_request, GPT4o_request, qwen_batch_request

eval_model = sys.argv[1]  # llama3_request, GPT_Instruct_request, ChatGPT_request, qwen_request
date = sys.argv[2]
dataset = sys.argv[3]  # 496 or 300
benchmark = sys.argv[4]

if eval_model == "GPT_Instruct_request":
    assess_model = GPT_Instruct_request
elif eval_model == "ChatGPT_request":
    assess_model = ChatGPT_request
elif eval_model == "GPT4omini_request":
    assess_model = GPT4omini_request

def _run_nli_GPT3turbo_batch(cases):
    """
    批量处理 NLI 任务，生成提示并调用模型。
    
    Args:
        cases (list): 包含多个案例的列表。
    
    Returns:
        list: 每个案例的生成结果。
    """
    prompts = []
    for case in cases:
        prompt = "Question:\n\n{}Answer:".format(case["question"])
        prompts.append(prompt)
    
    # 批量调用模型
    responses = assess_model(prompts) if eval_model == "qwen_request" else [assess_model(prompt) for prompt in prompts]
    return responses

def process_slice(slice_cases):
    # 批量处理案例
    responses = _run_nli_GPT3turbo_batch(slice_cases)
    
    # 将响应添加到案例中
    for case, response in zip(slice_cases, responses):
        case["response"] = response
    
    return slice_cases

if eval_model == "GPT_Instruct_request":
    eval_method = "eval_3.5instruct"
elif eval_model == "GPT4omini_request":
    eval_method = "eval_4omini"
else:
    eval_method = "eval_3.5turbo"

def run():
    global eval_method, date, dataset, benchmark
    res_file = f"/your_path/{benchmark}/results/{date}_{dataset}_baseline_wo_retrieve_{eval_method}.json"
    if dataset == "full":
        case_file = f"/your_path/{benchmark}/datasets/{benchmark}_results_w_negative_passages_{dataset}.json"
    
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        json_data = []
        num_slices = 1
        slice_length = len(cases) // num_slices
        slices = [cases[i:i+slice_length] for i in range(0, len(cases), slice_length)]
        final_result = []
        
        # 单线程顺序处理切片
        for slice_cases in tqdm(slices, desc="Processing slices"):
            result = process_slice(slice_cases)
            final_result.extend(result)
        
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)

run()
print(f"In Eval Baseline Wo Retrieve: {eval_model} {date} {dataset}")