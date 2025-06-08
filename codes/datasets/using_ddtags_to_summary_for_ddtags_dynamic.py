import torch
import json
from tqdm import tqdm
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
import concurrent.futures
from utils import GPT_Instruct_request, ChatGPT_request, llama3_request, GPT4o_request, qwen_request, GPT4omini_request
import re
import ast
from copy import deepcopy 
#from fastchat.model import load_model


topkk = ast.literal_eval(sys.argv[1])
noises = ast.literal_eval(sys.argv[2])
dataset = sys.argv[3]
eval_model = sys.argv[4]
date = sys.argv[5]
summary_prompt = sys.argv[6]
clustering_type = sys.argv[7]
benchmark = sys.argv[8]

if eval_model == "GPT_Instruct_request":
    assess_model = GPT_Instruct_request
elif eval_model == "ChatGPT_request":
    assess_model = ChatGPT_request
elif eval_model == "GPT4omini_request":
    assess_model = GPT4omini_request

filter_paragraph = ["No content to", "no content to", "I'm sorry", "I am sorry", "I can not provide", "I can't provide", "Could you clarify", "Sorry, I", "Could you clarify", "?"]


def count_docs_by_pattern(text):
    # 找到所有形如 \n数字. 的匹配
    pattern = re.findall(r'\n(\d+)\.', text)
    doc_count_estimate = len(pattern) + 1  # 加上开头的 1.

    if not pattern:
        return 1  # 只有1.，没别的编号，直接返回1

    last_number = int(pattern[-1])  # 最后一个匹配到的编号

    if last_number == doc_count_estimate:
        return doc_count_estimate
    else:
        print(f"⚠️ 编号可能异常：共匹配到 {doc_count_estimate} 条，但最后编号是 {last_number}。将返回 {min(doc_count_estimate, last_number)} 条。")
        return min(doc_count_estimate, last_number)

def replace_numbers_with_newline(text):
    # 使用正则表达式查找所有数字，并用换行符替换
    return re.sub(r'\d+.', '\n', text)

def _run_nli_GPT3(query, docs):
    global eval_model, summary_prompt
    if summary_prompt == "final":
        prompt = f"Instruction:\nPlease refer to the following text and answer the following question, providing supporting evidence.\n\nQuestion:\n{query}\n\nReference text:\n{docs}\n\nAnswer:"
    res = 0
    while (True):
        try:
            text = assess_model(prompt,temperature=0.7)
            return text
        except Exception as e:
            print(f"An error occurred: {e}")

def process_slice(slice_cases):
    global topk, summary_prompt, dataset
    outs = []
    for case_1 in tqdm(slice_cases):
        case = deepcopy(case_1)
        res=0
        topk = int(topk)
        if dataset == "redundancy":
            docs = case["docs"]
        else:
            docs = [case['passages'][i]['text'] for i in range(topk)]
        tagss = [case["tags"][str(i)][0:topk-1-i] for i in range(topk)]
        for passage in case['passages']:
            if 'embedding' in passage:
                del passage['embedding']
        Tags = {i:0 for i in range(topk)}
        docs_final = []
        if topk == 1:
            docs_final = [f"1.{docs[0]}"]
        for i, tags in enumerate(tagss):
            if Tags[i] == 0:
                doc = []
                doc.append(docs[i])
                for j, tag in enumerate(tags):
                    if tag == 1:
                        Tags[i+j+1] = 1
                        doc.append(docs[i+j+1])
                docs_final.append("\n".join([f"{k+1}.{doc[k]}" for k in range(len(doc))]))
        case["docs_final"] = docs_final
        case["summary_docs"] = []
        if docs_final:
            for doc in docs_final:
                query = case["question"]
                res = _run_nli_GPT3(query, doc)
                tag = 0
                for paragraph in filter_paragraph:
                    if paragraph in res:
                        tag = 1
                        break
                if tag == 0:
                    case["summary_docs"].append(res)
        outs.append(case)
    return outs

def run(topk, noise):
    global eval_model, date, dataset, summary_prompt
    if eval_model == "GPT_Instruct_request":
        eval_method = "3.5instruct"
    elif eval_model == "ChatGPT_request":
        eval_method = "3.5turbo"
    elif eval_model == "GPT4omini_request":
        eval_method = "4omini"
    if topk <= 10:
        length = 1
    else:
        length = 3
    res_file = f"/your_path/multihop_qa/{benchmark}/datasets/case_{date}_summary_{eval_method}_{summary_prompt}_{dataset}_results_ddtags_{clustering_type}_{length}_noise{noise}_topk{topk}.json"
    case_file = f"/your_path/multihop_qa/{benchmark}/datasets/case_{dataset}_{benchmark}_ddtags_noise{noise}_topk{topk}_{clustering_type}_{length}_embedding.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        json_data = []
        num_slices = 100
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
        print(f"Finished running for topk={topk} and noise={noise} and summary_prompt={summary_prompt} In Summarize Docs") 