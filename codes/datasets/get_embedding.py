import sys
import requests
from requests.auth import HTTPBasicAuth
import concurrent.futures
from tqdm import tqdm
import torch
import json
from utils import GPT_Instruct_request, ChatGPT_request
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = #your embedding model here, or you can use close-end api like text-adam-002
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path, torch_dtype = torch.float16).to(device)

dataset = sys.argv[1]

def get_embedding(text):
    with torch.no_grad():
        input = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        feature = model(**input, output_hidden_states=True, return_dict=True).pooler_output
    return feature.tolist()

input_file = f"../../{dataset}/datasets/{dataset}_results_w_negative_passages_full.json"
output_file = f"../../{dataset}/datasets/{dataset}_results_w_negative_passages_full_embedding.json"

with open(input_file, "r", encoding="utf-8") as lines:
    cases = json.load(lines)
    for case in tqdm(cases):
        for passage in case["positive_passages"]:
            passage["embedding"] = get_embedding(passage["text"])
        for passage in case["negative_passages"]:
            passage["embedding"] = get_embedding(passage["text"])
    
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(cases, json_file, ensure_ascii=False, indent=4)