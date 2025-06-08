import sys
import subprocess
import threading
import os
import time
import ast

date = sys.argv[1]
dataset = sys.argv[2]  # 400 or 500 or 113
eval_model = sys.argv[3]  # ChatGPT_request, llama3_request, GPT_Instruct_request
topkk = sys.argv[4]  # "[20, 50, 70]"
noises = sys.argv[5]  # "[20 ,60, 80]"
clustering_type = sys.argv[6]  # avg or dynamic or random
benchmark = sys.argv[7]

# ✅ 设定唯一允许输出日志的 summary_prompt
log_summary_prompt = "0519"

#python run_ours_ddtag_for_ddtags_dynamic.py 0518 full qwen_request "[5,10,20,30,50,70,100]" "[0]" dynamic twowiki


def run_task(summary_prompt, topkk, noises, dataset, eval_model, date, clustering_type, benchmark):
    log_enabled = (summary_prompt == log_summary_prompt)

    def log(msg):
        if log_enabled:
            print(msg)

    log(f"start_to_run summary_prompt {summary_prompt}")
    log("start_to_get_ddtags")
    #subprocess.run(["python", "../datasets/get_tag_doc_doc_similarity_dynamic.py", topkk, noises, dataset, benchmark])
    log("end_get_ddtags")

    log("start_to_summarize")
    subprocess.run(["python", "../datasets/using_ddtags_to_summary_for_ddtags_dynamic.py",topkk, noises, dataset, eval_model, date, summary_prompt, clustering_type, benchmark])
    log("end_summarize")

    log("start_to_eval")
    subprocess.run([    "python", "../run_methods/eval_ours_ddtag_summary_for_ddtags_dynamic.py",date, dataset, eval_model, topkk, noises, summary_prompt, clustering_type, benchmark])
    log("end_eval")

    log("start_to_extracte_answer")
    if eval_model == "GPT_Instruct_request":
        eval_method = "eval_3.5instruct"
    elif eval_model == "GPT4omini_request":
        eval_method = "eval_4omini"
    else:
        eval_method = "eval_3.5turbo"
    subprocess.run(["python", "../eval_metric/extracted_answer_topkk_for_ddtags_dynamic.py",date, dataset, eval_method, topkk, noises, summary_prompt, clustering_type, benchmark])
    log("end_extracte_answer")

    log("start_to_caculate_F1_EM")
    subprocess.run(["python", "../eval_metric/caculate_F1_EM_for_ddtags_dynamic.py",date, dataset, eval_method, topkk, noises, summary_prompt, clustering_type, benchmark])
    log("end_caculate_F1_EM")
    log(f"end_run_ for summary_prompt {summary_prompt}")


# 主逻辑
threads = []
summary_prompts = [
    "final"
]

for summary_prompt in summary_prompts:
    thread = threading.Thread(
        target=run_task,
        args=(summary_prompt, topkk, noises, dataset, eval_model, date, clustering_type, benchmark)
    )
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()

print("All tasks completed.")
