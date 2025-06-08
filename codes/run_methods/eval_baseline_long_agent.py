import torch
import json
from tqdm import tqdm
import os
import requests
from requests.auth import HTTPBasicAuth
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, "/your_path/codes")
from utils import GPT_Instruct_request, GPT4omini_request, ChatGPT_request, llama3_request, GPT4o_request, qwen_batch_request
import ast

eval_model = sys.argv[1]  # llama3_request, GPT_Instruct_request, ChatGPT_request, qwen_request
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

times = {5: 3, 10: 3, 20: 4, 30: 5, 50: 6, 70: 7, 100: 8}

def split_chunks(text_list, n_chunks):
    # 将 text_list 平均分成 n_chunks 块
    k, m = divmod(len(text_list), n_chunks)
    chunks = [text_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]
    return chunks

def _run_chunked_prompt_batch(questions, chunk_texts_list, case_indices):
    """
    批量处理分块提示，生成每个分块的回答，将所有提示拼接后一次性发送到API。
    
    Args:
        questions (list): 包含多个问题的列表。
        chunk_texts_list (list): 包含每个问题对应分块文本的列表。
        case_indices (list): 每个case的索引，用于追踪原始case。
    
    Returns:
        list: 每个分块的生成结果，按case_indices顺序排列。
    """
    prompts = []
    prompt_case_map = []
    
    for case_idx, (question, chunk_texts) in enumerate(zip(questions, chunk_texts_list)):
        for chunk_idx, chunk in enumerate(chunk_texts):
            ref_text = "\n".join([f"{i+1}. {text.strip()}" for i, text in enumerate(chunk)])
            prompt = "Instruction:\nPlease refer to the following text and answer the following question in simple words.\n\nQuestion:\n{}\n\nReference text:\n{}\n\nAnswer:".format(question, ref_text)
            prompts.append(prompt)
            prompt_case_map.append((case_idx, chunk_idx))
    
    # 批量调用模型
    responses = assess_model(prompts, temp=0.7) if eval_model == "qwen_request" else [assess_model(prompt, temp=0.7) for prompt in prompts]
    
    # 按case组织响应
    organized_responses = [[] for _ in range(len(questions))]
    for (case_idx, chunk_idx), response in zip(prompt_case_map, responses):
        organized_responses[case_idx].append((chunk_idx, response))
    
    # 按chunk_idx排序，确保顺序正确
    for case_responses in organized_responses:
        case_responses.sort(key=lambda x: x[0])
        case_responses[:] = [resp for _, resp in case_responses]
    
    return organized_responses

def _run_nli_GPT3turbo_long_agent_batch(answers_list, questions, case_indices):
    """
    批量处理 long-agent 提示，汇总候选回答，将所有提示拼接后一次性发送到API。
    
    Args:
        answers_list (list): 包含多个案例候选回答的列表。
        questions (list): 包含多个问题的列表。
        case_indices (list): 每个case的索引，用于追踪原始case。
    
    Returns:
        list: 每个案例的汇总结果。
    """
    prompts = []
    prompt_case_map = []
    
    for case_idx, (answers, question) in enumerate(zip(answers_list, questions)):
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
        prompt_case_map.append(case_idx)
    
    # 批量调用模型
    responses = assess_model(prompts) if eval_model == "qwen_request" else [assess_model(prompt) for prompt in prompts]
    
    # 按case_idx组织响应
    organized_responses = [None] * len(questions)
    for case_idx, response in zip(prompt_case_map, responses):
        organized_responses[case_idx] = response
    
    return organized_responses

def process_slice(slice_cases):
    global topk, dataset
    topk = int(topk)
    chunk_num = times[topk]
    
    questions = []
    chunk_texts_list = []
    answers_list = []
    case_indices = []
    
    # 1. 准备所有case的chunks和questions
    for idx, case in enumerate(tqdm(slice_cases, desc="Preparing cases")):
        if dataset == "redundancy":
            docs = case.get("docs", [])[:topk]
            doc_chunks = split_chunks(docs, chunk_num)
        else:
            for passage in case['passages']:
                if 'embedding' in passage:
                    del passage['embedding']
            passages = [p["text"] for p in case.get("passages", [])[:topk]]
            doc_chunks = split_chunks(passages, chunk_num)
        
        questions.extend([case["question"]] * len(doc_chunks))
        chunk_texts_list.extend(doc_chunks)
        case_indices.extend([idx] * len(doc_chunks))
    
    # 2. 批量处理所有chunk的回答
    chunk_responses = _run_chunked_prompt_batch(questions, chunk_texts_list, case_indices)
    
    # 3. 组织每个case的回答
    for idx, case in enumerate(slice_cases):
        case_answers = chunk_responses[idx]
        for i, response in enumerate(case_answers):
            case[f"response_{i}"] = response
        answers_list.append(case_answers)
        case_indices.append(idx)
    
    # 4. 批量处理 long-agent 汇总
    long_agent_responses = _run_nli_GPT3turbo_long_agent_batch(answers_list, [case["question"] for case in slice_cases], case_indices)
    
    # 5. 保存结果
    for idx, (case, long_agent_response) in enumerate(zip(slice_cases, long_agent_responses)):
        case["long_agent"] = long_agent_response
        case["response"] = long_agent_response.replace("*", "").split("Selected_Answer:")[-1].strip()
    
    return slice_cases

if eval_model == "GPT_Instruct_request":
    eval_method = "eval_3.5instruct"
elif eval_model == "GPT4omini_request":
    eval_method = "eval_4omini"
elif eval_model == "ChatGPT_request":
    eval_method = "eval_3.5turbo"

def run(topk, noise):
    global eval_method, date, dataset
    res_file = f"/your_path/{benchmark}/results/{date}_{dataset}_long_agent_{eval_method}_noise{noise}_topk{topk}.json"
    if dataset == "redundancy":
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
        print(f"Finished running for topk={topk} and noise={noise} In Eval Long-Agent")