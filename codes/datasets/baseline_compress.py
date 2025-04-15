import torch
import json
from tqdm import tqdm
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
import concurrent.futures
from utils import GPT_Instruct_request, ChatGPT_request, llama3_request, GPT4o_request
import re
import ast

# 选择评测模型
eval_model = sys.argv[1]  # llama3_request, GPT_Instruct_request, ChatGPT_request
date = sys.argv[2]
dataset = sys.argv[3]  # 496 or 300 or full
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

filter_paragraph = ["No content to", "no content to", "I'm sorry", "I am sorry", "I can not provide", "I can't provide", "Could you clarify", "Sorry, I", "Could you clarify", "?"]

def _run_nli_GPT3(num, docs):
    global eval_model
    prompt = f"""
**#Instruction#:** Rewrite each of the following {num} documents separately, making them more concise and clear. Try to ensure that the compressed documents are no more than half the length of the original documents.

**#Documents#:**  
{docs}  

**#Rewritten Documents#:**  
1. <to be rewritten>  
2. <to be rewritten>  
...  
{num}. <to be rewritten>

**#Attention#:** Follow the format of the examples above, ensuring the rewritten documents are concise, clear, and well-structured. Output only the rewritten documents in the specified format without additional explanations.
"""
    while True:
        try:
            text = assess_model(prompt)
            return text.strip()
        except Exception as e:
            print(f"An error occurred: {e}")


def extract_numbered_sections(text):
    """
    解析 GPT 生成的文本，按 `1. 2. 3. ...` 的格式提取内容
    """
    sections = {}
    lines = text.split("\n")
    current_index = None
    
    for line in lines:
        line = line.strip()
        match = re.match(r'^(\d+)\.\s*(.*)', line)
        if match:
            current_index = int(match.group(1))
            sections[current_index] = match.group(2)
        elif current_index is not None and line:
            sections[current_index] += " " + line
    
    return [sections[i].strip() for i in sorted(sections.keys())]


def process_slice(slice_cases):
    global topk, dataset
    for case in tqdm(slice_cases):
        for passage in case['passages']:
            if 'embedding' in passage:
                del passage['embedding']
        topk = int(topk)
        if dataset == "redundancy":
            docs = case["docs"]
        else:
            docs = [case['passages'][i]['text'].strip() for i in range(min(topk, len(case['passages'])))]
        compressed_docs = []
        times = 0
        
        for i in range(0, len(docs), 20):
            doc_chunk = "\n\n".join([f"{j+1}. {doc}" for j, doc in enumerate(docs[i:i+20])])
            k = 0
            extracted_docs = []
            
            while len(extracted_docs) != len(docs[i:i+20]) and k < 3:
                compressed_text = _run_nli_GPT3(len(docs[i:i+20]), doc_chunk)
                extracted_docs = extract_numbered_sections(compressed_text)
                k += 1
                times += 1
            
            if len(extracted_docs) != len(docs[i:i+20]):
                extracted_docs = docs[i:i+20]
            
            compressed_docs.extend(extracted_docs)
        
        case["summary_docs_baseline"] = compressed_docs
    return slice_cases


def run(topk, noise):
    global eval_model, date, dataset, benchmark
    eval_method = {
        "llama3_request": "llama3",
        "GPT_Instruct_request": "3.5instruct",
        "ChatGPT_request": "3.5turbo"
    }.get(eval_model, eval_model)
    
    res_file = f"../../{benchmark}/datasets/case_{date}_{dataset}_summary_baseline_compress_{eval_method}_noise{noise}_topk{topk}.json"
    case_file = f"../../{benchmark}/datasets/{benchmark}_results_random_{dataset}_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        num_slices = 20
        slice_length = len(cases) // num_slices
        slices = [cases[i:i+slice_length] for i in range(0, len(cases), slice_length)]
        final_result = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_slice, slices)
        
        for result in results:
            final_result.extend(result)
        
        with open(res_file, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, ensure_ascii=False, indent=4)

for topk in topkk:
    for noise in noises:
        run(topk, noise)
        print(f"Finished running for topk={topk} and noise={noise} In Summarize Docs")