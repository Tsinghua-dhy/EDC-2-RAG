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
import ast

eval_model = sys.argv[1]  # llama3_request, GPT_Instruct_request, ChatGPT_request
date = sys.argv[2]
dataset = sys.argv[3]  # 496 or 300 or full
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]

if eval_model == "GPT_Instruct_request":
    assess_model = GPT_Instruct_request
elif eval_model == "ChatGPT_request":
    assess_model = ChatGPT_request
elif eval_model == "GPT4omini_request":
    assess_model = GPT4omini_request

def _run_nli_GPT3turbo_batch(cases, topk, dataset):
    """
    批量处理 NLI 任务，生成提示并调用模型。
    
    Args:
        cases (list): 包含多个案例的列表。
        topk (int): 使用的参考文本数量。
        dataset (str): 数据集类型。
    
    Returns:
        list: 每个案例的生成结果。
    """
    prompts = []
    for case in cases:
        if dataset == "redundancy":
            ref_text = "\n".join([f"{i+1}.{case['docs'][i].strip()}" for i in range(topk)])
        else:
            ref_text = "\n".join([f"{i+1}.{case['passages'][i]['text'].strip()}" for i in range(topk)])
        prompt = "Instruction:\nPlease refer to the following text and answer the following question in simple words.\n\nQuestion:\n{}\n\nReference text:\n{}\n\nAnswer:".format(case["question"], ref_text)
        prompts.append(prompt)
    
    # 批量调用模型
    responses = assess_model(prompts) if eval_model == "qwen_request" else [assess_model(prompt) for prompt in prompts]
    return responses

def process_slice(slice_cases):
    global topk, dataset
    topk = int(topk)
    
    # 删除嵌入数据
    for case in slice_cases:
        for passage in case['passages']:
            if 'embedding' in passage:
                del passage['embedding']
    
    # 批量处理案例
    responses = _run_nli_GPT3turbo_batch(slice_cases, topk, dataset)
    
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

def run(topk, noise):
    global eval_method, date, dataset
    res_file = f"/your_path/{benchmark}/results/{date}_{dataset}_rag_{eval_method}_noise{noise}_topk{topk}.json"
    if dataset == "redundancy":
        if topk == 30:
            case_file = f"./datasets/case_0329_rewrite_3.5turbo_webq_noise{noise}_topk{topk}.json"
        else:
            case_file = f"./datasets/case_0327_rewrite_3.5turbo_webq_noise{noise}_topk{topk}.json"
    else:
        case_file = f"/your_path/{benchmark}/datasets/{benchmark}_results_random_{dataset}_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
    
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

for topk in topkk:
    for noise in noises:
        run(topk, noise)
        print(f"In Eval Baseline RAG: {topk}, {noise}")