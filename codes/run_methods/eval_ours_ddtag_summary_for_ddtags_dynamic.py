import torch
import json
from tqdm import tqdm
import os
import requests
from requests.auth import HTTPBasicAuth
import concurrent.futures
import sys
import ast
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "../")
from utils import GPT_Instruct_request, ChatGPT_request, llama3_request, GPT4o_request, qwen_request

date = sys.argv[1]
dataset = sys.argv[2]
eval_model = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
length = sys.argv[6]
summary_prompt = sys.argv[7]
clustering_type = sys.argv[8]
benchmark = sys.argv[9]
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


def _run_nli_GPT3turbo(question, ref_text):
    prompt = "Instruction:\nPlease refer to the following text and answer the following question in simple words.\n\nQuestion:\n{}\n\nReference text:\n{}\n\nAnswer:".format(question, ref_text) 
    res = 0
    while (True):
        try:
            text = assess_model(prompt)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    return text

def process_slice(cases):
    global topk, dataset
    topk = int(topk)
    for i, case in enumerate(tqdm(cases)):
        if case["summary_docs"]:
            ref_text = "\n".join([f"{i+1}.{case['summary_docs'][i]}" for i in range(len(case['summary_docs']))])
        else:
            if dataset == "redundancy":
                ref_text = "\n".join([f"{i+1}.{case['docs'][i].strip()}" for i in range(topk)])
            else:
                ref_text = "\n".join([f"{i+1}.{case['passages'][i]['text'].strip()}" for i in range(topk)])
        text= _run_nli_GPT3turbo(case["question"], ref_text)
        case["response"] = text
    return cases

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
    global eval_method, date, dataset, length, summary_prompt
    res_file = f"../{benchmark}/results/{date}_{dataset}_ours_summary_{summary_prompt}_ddtags_{clustering_type}_{length}_{eval_method}_noise{noise}_topk{topk}.json"
    eval_method_1 = eval_method.split("_")[-1]
    case_file = f"../{benchmark}/datasets/case_{date}_summary_{eval_method_1}_{summary_prompt}_{dataset}_results_ddtags_{clustering_type}_{length}_noise{noise}_topk{topk}.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        json_data = []
        num_slices = 40
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
        print(f"Finished running for topk={topk} and noise={noise} and length={length} and summary prompt={summary_prompt} In Eval Ours")
