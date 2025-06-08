import torch
import json
from tqdm import tqdm
import os
import requests
from requests.auth import HTTPBasicAuth
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, "/your_path/codes")
import ast
from utils import GPT_Instruct_request, GPT4omini_request, ChatGPT_request, llama3_request, GPT4o_request, qwen_batch_request

date = sys.argv[1]
dataset = sys.argv[2]
eval_model = sys.argv[3]
topkk = ast.literal_eval(sys.argv[4])
noises = ast.literal_eval(sys.argv[5])
summary_prompt = sys.argv[6]
clustering_type = sys.argv[7]
benchmark = sys.argv[8]

if eval_model == "GPT_Instruct_request":
    assess_model = GPT_Instruct_request
elif eval_model == "ChatGPT_request":
    assess_model = ChatGPT_request
elif eval_model == "GPT4omini_request":
    assess_model = GPT4omini_request


def _run_nli_GPT3turbo_ours_longagent_batch(answers_list, questions):
    """
    批量处理 long-agent 提示，汇总候选回答。
    
    Args:
        answers_list (list): 包含多个案例候选回答的列表。
        questions (list): 包含多个问题的列表。
    
    Returns:
        list: 每个案例的汇总结果。
    """
    prompts = []
    for answers, question in zip(answers_list, questions):
        prompt = f'''Task: Analyze the following set of candidate answers to a question and select the single most consistent/plausible answer based on majority consensus and logical coherence.

Instructions:
1. Carefully compare all candidate answers.
2. Identify the core factual claims or entities in each answer.
3. Group semantically equivalent answers (e.g., "1990", "the year 1990", "nineteen ninety").
4. Select the answer that:
   - Appears most frequently in the candidate set
   - Has strong internal consistency (no self-contradictions)
5. If multiple answers have equal validity, prefer the most specific and concise one.

Format Requirements:

Reasoning: Concise justification for selection
Selected_Answer: ...

Below is an example.

Candidate Answers: 
["Paris", "The capital is Paris", "France", "paris", "It's Paris in France"]

Question: What is the capital of France?

Expected Response:

Reasoning: 4/5 answers directly state 'Paris'. While 'France' is incorrect alone, the most frequent and unambiguous consensus is 'Paris'
Selected_Answer: Paris

Candidate Answers:
{answers}

Question: {question}
'''
        prompts.append(prompt)
    
    # 批量调用模型
    responses = assess_model(prompts) if eval_model == "qwen_request" else [assess_model(prompt) for prompt in prompts]
    return responses

def process_slice(cases):
    global topk, dataset, summary_prompt
    topk = int(topk)
    
    questions = []
    ref_texts = []
    longagent_cases = []
    
    for case in cases:
        if summary_prompt == "final":
            ref_text = [case['summary_docs'][i] for i in range(len(case['summary_docs']))]
            questions.append(case["question"])
            ref_texts.append(ref_text)
            longagent_cases.append(True)
        else:
            print("wrong prompt!")
    
    # 批量处理
    if summary_prompt == "final":
        responses = _run_nli_GPT3turbo_ours_longagent_batch(ref_texts, questions)
    
    # 将响应添加到案例中
    for case, response in zip(cases, responses):
        case["response"] = response
    
    return cases

if eval_model == "GPT_Instruct_request":
    eval_method = "eval_3.5instruct"
elif eval_model == "GPT4omini_request":
    eval_method = "eval_4omini"
else:
    eval_method = "eval_3.5turbo"

def run(topk, noise, length):
    global eval_method, date, dataset, summary_prompt
    res_file = f"/your_path/{benchmark}/results/{date}_{dataset}_ours_summary_{summary_prompt}_ddtags_{clustering_type}_{length}_{eval_method}_noise{noise}_topk{topk}.json"
    eval_method_1 = eval_method.split("_")[-1]
    case_file = f"/your_path/{benchmark}/datasets/case_{date}_summary_{eval_method_1}_{summary_prompt}_{dataset}_results_ddtags_{clustering_type}_{length}_noise{noise}_topk{topk}.json"
    
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
        if topk <= 10:
            length = 1
        else:
            length = 3
        run(topk, noise, length)
        print(f"Finished running for topk={topk} and noise={noise} and summary prompt={summary_prompt} In Eval Ours")