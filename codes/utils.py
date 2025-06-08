from openai import OpenAI
import httpx
import json
import concurrent.futures
import functools
import requests
from typing import Literal
from transformers import AutoTokenizer
import time
##目前4的key不可用
import tiktoken

def truncate_prompt(prompt, max_tokens):
  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 适用于 gpt-3.5-turbo 和 gpt-4
  truncated_tokens = encoding.encode(prompt)[:max_tokens]
  return encoding.decode(truncated_tokens)

api_key = "your_key"
client = OpenAI(
    base_url="your_url",
    api_key=api_key,
    http_client=httpx.Client(
        base_url="your_url",
        follow_redirects=True,
    ),
)
api_4_key = "your_key"
client_4 = OpenAI(
    base_url="your_url",
    api_key=api_4_key,
    http_client=httpx.Client(
        base_url="your_url",
        follow_redirects=True,
    ),
)
def ChatGPT_request(prompt,temperature=0): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  while(True):
    try:
      encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 适用于 gpt-3.5-turbo 和 gpt-4
      if len(encoding.encode(prompt)) > 15700:
        prompt = truncate_prompt(prompt, 15700)
      rst = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature=temperature,
        max_tokens = 512,
        messages=[
          {"role": "user", "content": prompt}
        ]
      )
      return rst.choices[0].message.content.strip()
    except Exception as e:
      print(f"Error when calling deepseek: {e}")
      time.sleep(5)
def GPT_Instruct_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  while(True):
    try:
      rst = client.completions.create(
          model="gpt-3.5-turbo-instruct",
          temperature=0.0,
          max_tokens = 4096,
          prompt=prompt
      )
      return rst.choices[0].text.strip()
    except:
      print("ChatGPT ERROR")

def GPT4omini_request(prompt, temperature=0.0):
  while True:
    try:
      rst = client_4.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        temperature=temperature,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
      )
      return rst.choices[0].message.content.strip()
    except Exception as e:
      print("ChatGPT ERROR:", e)
      time.sleep(1)

def run(topk, res_file, case_file, process_slice):
    topk = int(topk)    
    with open(case_file, "r", encoding="utf-8") as lines:
        cases = json.load(lines)
        num_slices = 100
        slice_length = len(cases) // num_slices
        slices = [cases[i:i+slice_length] for i in range(0, len(cases), slice_length)]
        final_result = []
        # 并行评测八份切片
        results = []
        process_slice = functools.partial(process_slice, topk = topk)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_slice, slices)
        # 合并八份切片的结果
        for result in results:
            final_result.extend(result)
        with open(res_file, "w", encoding = "utf-8" ) as json_file:
            json.dump(final_result, json_file,  ensure_ascii=False, indent=4)