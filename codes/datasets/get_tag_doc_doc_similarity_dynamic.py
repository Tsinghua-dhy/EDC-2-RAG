import torch
import json
from tqdm import tqdm
import os
import sys
import ast 
import requests
from requests.auth import HTTPBasicAuth
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
from copy import deepcopy 

topkk = ast.literal_eval(sys.argv[1])
noises = ast.literal_eval(sys.argv[2])
dataset = sys.argv[3]
benchmark = sys.argv[4]
def calculate_cosine_similarity(vector1, vector2):
    similarity = cosine_similarity(vector1, vector2)[0][0]
    return similarity

def process_slice(slice_cases):
    global topk, length, dataset
    outs = []
    for case_1 in tqdm(slice_cases):
        case = deepcopy(case_1)
        res=0
        if dataset == "redundancy":
            docs = case["docs"]
            features = case["embeddings"]
        else:
            docs = [case['passages'][i]['text'] for i in range(topk)]
            features = [case['passages'][i]['embedding'] for i in range(topk)]
        for passage in case['passages']:
            if 'embedding' in passage:
                del passage["embedding"]
        if "embeddings" in case:
            del case["embeddings"]
        case["tags"] = {i: [0] * (topk-i-1) for i in range(topk)}
        tags = {i:0 for i in range(topk)}
        length_1 = length
        all = 0
        for i in range(topk):
            if tags[i] == 0:
                sims = []
                tags[i] = 1
                all += 1
                for j in range(i+1, topk):
                    doc1 = docs[i]
                    doc2 = docs[j]
                    feature_1 = features[i]
                    feature_2 = features[j]
                    if tags[j] == 1:
                        sims.append(0)
                        continue
                    sim = calculate_cosine_similarity(feature_1, feature_2)
                    sims.append(sim)
                top_indices = sorted(range(len(sims)), key=lambda x: sims[x], reverse=True)[:min(length_1,topk-all)]
                for index in top_indices:
                    tags[i+1+index] = 1
                    all += 1
                    case["tags"][i][index]=1
                if length_1*2 < 20:
                    length_1 = length_1*2
                else:
                    length_1 = 20
        outs.append(case)
    return outs

for topk in topkk:
    for noise in noises:
        if topk <= 10:
            length = 1
        else :
            length = 3
        res_file = f"/your_path/{benchmark}/datasets/case_{dataset}_{benchmark}_ddtags_noise{noise}_topk{topk}_dynamic_{length}_embedding.json"
        if dataset == "redundancy":
            if topk == 30:
                case_file = f"/your_path/webq/datasets/case_0329_rewrite_3.5turbo_webq_noise{noise}_topk{topk}_embedding.json"
            else:
                case_file = f"/your_path/webq/datasets/case_0327_rewrite_3.5turbo_webq_noise{noise}_topk{topk}_embedding.json"
        else:
            case_file = f"/your_path/{benchmark}/datasets/{benchmark}_results_random_{dataset}_w_negative_passages_noise{noise}_topk{topk}_embedding.json"
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

    
