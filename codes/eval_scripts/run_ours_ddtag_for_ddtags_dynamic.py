import sys
import subprocess
import os
import ast
date = sys.argv[1]
dataset = sys.argv[2] # 400 or 500 or 113
eval_model = sys.argv[3]  # ChatGPT_request, llama3_request, GPT_Instruct_request
topkk = sys.argv[4] # "[20, 50, 70]"
noises = sys.argv[5] # "[20 ,60, 80]"
summary_prompt = sys.argv[6] # 1110 or 1121
clustering_type = sys.argv[7] # avg or dynamic or random
benchmark = sys.argv[8]
#python run_ours_ddtag_for_ddtags_dynamic.py 0328 redundancy ChatGPT_request "[20]" "[20,40,60,80]" 1110 dynamic
for length in ["3"]:
    # 使用 subprocess 运行其他 Python 文件，并传递环境变量
    # 使用 subprocess 运行其他 Python 文件，并传递环境变量
    print(f"start_to_run_{length}")
    print("start_to_get_ddtags")
    #subprocess.run(["python", "../datasets/get_tag_doc_doc_similarity_dynamic.py", topkk, noises, length, dataset,benchmark])
    print("end_get_ddtags")
    print("start_to_summarize")
    subprocess.run(["python", "../datasets/using_ddtags_to_summary_for_ddtags_dynamic.py", topkk, noises, length, dataset, eval_model, date, summary_prompt, clustering_type,benchmark])
    print("end_summarize")
    print("start_to_eval")
    subprocess.run(["python", "../run_methods/eval_ours_ddtag_summary_for_ddtags_dynamic.py", date, dataset, eval_model, topkk, noises, length, summary_prompt, clustering_type,benchmark])
    print("end_eval")
    print("start_to_extracte_answer")  
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
    subprocess.run(["python", "../eval_metric/extracted_answer_topkk_for_ddtags_dynamic.py", date, dataset, eval_method, topkk, noises, length, summary_prompt, clustering_type,benchmark])
    print("end_extracte_answer")
    print("start_to_caculate_F1_EM")
    subprocess.run(["python", "../eval_metric/caculate_F1_EM_for_ddtags_dynamic.py", date, dataset, eval_method, topkk, noises, length, summary_prompt, clustering_type,benchmark])
    print("end_caculate_F1_EM")
    print(f"end_run_{length}")