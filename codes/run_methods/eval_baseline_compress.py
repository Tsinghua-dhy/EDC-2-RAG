import torch
import json
from tqdm import tqdm
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
import concurrent.futures
import sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "../")
from utils import GPT_Instruct_request, ChatGPT_request, llama3_request, GPT4o_request, qwen_request
from copy import deepcopy
import ast

eval_model = sys.argv[1]#llama3_request, GPT_Instruct_request, ChatGPT_request
date = sys.argv[2]
dataset = sys.argv[3]#496 or 300 or full
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
benchmark = sys.argv[6]
if eval_model == "llama3_request":
    assess_model = llama3_request
elif eval_model == "GPT_Instruct_request":
    assess_model = GPT_Instruct_request
elif eval_model == "ChatGPT_request":
    assess_model = ChatGPT_request
elif eval_model == "GPT4o_request":
    assess_model = GPT4o_request
elif eval_model == "qwen_request":
    assess_model = qwen_request


def _run_nli_GPT3turbo(case):
    global topk
    topk = int(topk)
    ref_text = "\n".join([f"{i+1}.{case['summary_docs_baseline'][i]}" for i in range(topk)])
    prompt = "Instruction:\nPlease refer to the following text and answer the following question in simple words.\n\nQuestion:\n{}\n\nReference text:\n{}\n\nAnswer:".format(case["question"], ref_text) 
    res = 0
    while (True):
        try:
            text = assess_model(prompt)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    return text


def process_slice(slice_cases):
    outs = []
    for case_1 in tqdm(slice_cases):
        case = deepcopy(case_1)
        for passage in case['passages']:
            if 'embedding' in passage:
                del passage['embedding']
        text = _run_nli_GPT3turbo(case)
        case["response"] = text
        outs.append(case)
    return outs

if eval_model == "llama3_request":
    eval_method = "eval_llama3"
elif eval_model == "GPT_Instruct_request":
    eval_method = "eval_3.5instruct"
elif eval_model == "GPT4o_request":
    eval_method = "eval_4o"
elif eval_model == "qwen_request":
    eval_method = "eval_qwen"
else:
    eval_method = "eval_3.5turbo"

def run(topk,noise):
    global eval_method, date, dataset, benchmark
    res_file = f"../{benchmark}/results/{date}_{dataset}_compress_{eval_method}_noise{noise}_topk{topk}.json"
    eval_method_1 = eval_method.split("_")[-1]
    case_file = f"../{benchmark}/datasets/case_{date}_{dataset}_summary_baseline_compress_{eval_method_1}_noise{noise}_topk{topk}.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        json_data = []
        num_slices = 10
        slice_length = len(cases) // num_slices
        slices = [cases[i:i+slice_length] for i in range(0, len(cases), slice_length)]
        final_result = []
        # 并行评测八份切片
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_slice, slices)
        # 合并八份切片的结果
        for result in results:
            final_result.extend(result)
        with open(res_file, "w", encoding = "utf-8" ) as json_file:
            json.dump(final_result, json_file,  ensure_ascii=False, indent=4)

for topk in topkk:
    for noise in noises:
        run(topk, noise)
        print(f"In Eval Baseline RAG: {topk}, {noise}")

# 这个脚本的目的是使用不同的模型（GPT-4、GPT-3.5、GPT-3.5-turbo和自定义的T5模型）来评估案例中的前提和断言之间的逻辑关系