import sys
import subprocess
import os
import ast
date = sys.argv[1]
dataset = sys.argv[2] # 400 or 500 or 113
eval_model = sys.argv[3]  # ChatGPT_request, llama3_request, GPT_Instruct_request
topkk = sys.argv[4] # "[20, 50, 70]"
noises = sys.argv[5] # "[20 ,60, 80]"
summary_prompt = sys.argv[6] # final
clustering_type = sys.argv[7] # avg or dynamic or random
benchmark = sys.argv[8]
#python run_all.py 0518 full ChatGPT_request "[5,10,20,30,50,70,100]" "[0]" final dynamic musique 

"""
print("run_baseline_wo_retrieve")
subprocess.run(["python", "./run_baseline_wo_retrieve.py", eval_model, date, dataset, benchmark])
print("end_baseline_wo_retrieve")
"""
print("run_baseline_rag")
subprocess.run(["python", "./run_baseline_rag.py", eval_model, date, dataset,topkk,noises,benchmark])
print("end_baseline_rag")

print("run_baseline_compress")
subprocess.run(["python", "./run_baseline_compress.py", eval_model, date, dataset,topkk,noises,benchmark])
print("end_baseline_compress")

print("run_baseline_long_agent")
subprocess.run(["python", "./run_baseline_long_agent.py", eval_model, date, dataset,topkk,noises,benchmark])
print("end_baseline_long_agent")
print("run_ours") 

subprocess.run(["python", "./run_ours_ddtag_for_ddtags_dynamic.py", date, dataset, eval_model, topkk, noises, summary_prompt, clustering_type,benchmark])
print("end_ours")