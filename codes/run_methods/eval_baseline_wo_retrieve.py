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

eval_model = sys.argv[1]#llama3_request, GPT_Instruct_request, ChatGPT_request
date = sys.argv[2]
dataset = sys.argv[3]#496 or 300
benchmark = sys.argv[4]

if eval_model == "llama3_request":
    assess_model = llama3_request
elif eval_model == "GPT_Instruct_request":
    assess_model = GPT_Instruct_request
elif eval_model == "ChatGPT_request":
    assess_model = ChatGPT_request
elif eval_model == "GPT4o_request":
    assess_model = GPT4o_request
def _run_nli_GPT3turbo(case):
    prompt = "Question:\n\n{}Answer:".format(case["question"]) 
    res = 0
    while (True):
        try:
            text = assess_model(prompt)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    return text



def process_slice(slice_cases):
    for case in tqdm(slice_cases):
        res=0
        text = _run_nli_GPT3turbo(case)
        case["response"] = text
    return slice_cases

if eval_model == "llama3_request":
    eval_method = "eval_llama3"
elif eval_model == "GPT_Instruct_request":
    eval_method = "eval_3.5instruct"
elif eval_model == "GPT4o_request":
    eval_method = "eval_4o"
else:
    eval_method = "eval_3.5turbo"

def run():
    global eval_method, date, dataset, benchmark
    res_file = f"../{benchmark}/results/{date}_{dataset}_baseline_wo_retrieve_{eval_method}.json"
    if dataset == "500":
        case_file = "./datasets/case_webq_nq_ddtags_noise20_topk5_1101.json"
    elif dataset == "113":
        case_file = "./datasets/case_113_webq_ddtags_noise20_topk20_simcse_0.65.json"
    elif dataset == "400":
        case_file = "./datasets/webq_results_random_400_w_negative_passages_noise0_topk5.json"
    elif dataset == "full":
        case_file = f"../../{benchmark}/datasets/{benchmark}_results_w_negative_passages_{dataset}.json"
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

run()
print(f"In Eval Baseline Wo Retrieve: {eval_model} {date} {dataset}")    

# 这个脚本的目的是使用不同的模型（GPT-4、GPT-3.5、GPT-3.5-turbo和自定义的T5模型）来评估案例中的前提和断言之间的逻辑关系