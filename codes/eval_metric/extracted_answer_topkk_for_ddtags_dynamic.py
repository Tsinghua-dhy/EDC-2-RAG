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
from utils import GPT_Instruct_request, GPT4omini_request, ChatGPT_request, qwen_request
eval_model = GPT_Instruct_request
import ast


date = sys.argv[1]
dataset = sys.argv[2]
eval_method = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
length = sys.argv[6]
summary_prompt = sys.argv[7]
clustering_type = sys.argv[8]
benchmark = sys.argv[9]
def _run_nli_GPT3turbo(case):
    prompt = f"""Task Description:
You need to extract the essential information from a generated answer and reformat it to match the structure of the golden answer. We will provide a question, a golden nnswer, and a generated answer. Carefully compare the generated answer with the golden answer, and extract key information from the generated answer to make it as close as possible to the golden answer in format. This will facilitate subsequent evaluation using Exact Match (EM) and F1 metrics.

Input:

Question: {case["question"]}
Golden Answer: {case["answers"][0]}
Generated Answer: {case["response"]}
Requirements:

Extract information from the generated answer that corresponds to the essential content of the golden answer.
Reorganize the extracted content to align with the structure of the golden answer, including phrasing and order of information where relevant.
If the generated answer contains information not covered in the golden answer, include only information crucial to answering the question. Disregard redundant or irrelevant details.
Output Format:
Provide a reformatted answer, aligned as closely as possible with the golden answer:

Reformatted Answer: """
    res = 0
    while (True):
        try:
            text = eval_model(prompt)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    return text



def process_slice(slice_cases):
    for case in tqdm(slice_cases):
        res=0
        text = _run_nli_GPT3turbo(case)
        case["extracted_answer"] = text
    return slice_cases

def run(topk, noise):
    global eval_method, date, dataset, length, summary_prompt, clustering_type
    case_file = f"../{benchmark}/results/{date}_{dataset}_ours_summary_{summary_prompt}_ddtags_{clustering_type}_{length}_{eval_method}_noise{noise}_topk{topk}.json"
    res_file = f"../{benchmark}/extracted_answer/{date}_{dataset}_ours_summary_{summary_prompt}_ddtags_{clustering_type}_{length}_{eval_method}_noise{noise}_topk{topk}.json"
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
        print(f"In Extracted Answer TopkK:{topk} Noise:{noise} Length:{length} Summary Prompt:{summary_prompt}")

# 这个脚本的目的是使用不同的模型（GPT-4、GPT-3.5、GPT-3.5-turbo和自定义的T5模型）来评估案例中的前提和断言之间的逻辑关系